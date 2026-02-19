
import argparse
import os
import json
import torch
import torch.distributed as dist
import numpy as np

from diffsynth.models import ModelConfig
from diffsynth import VideoData, save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline


def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def load_hl_codes(hl_path: str, num_frames: int):
    if (hl_path is None) or (hl_path == "") or (not os.path.exists(hl_path)):
        return None
    try:
        data = np.load(hl_path)
    except Exception as e:
        print(f"[WARN] Failed to load HL from {hl_path}: {e}")
        return None
        
    hl = data["hl"] if "hl" in data else data[list(data.keys())[0]]
    
    # Resample to latent timeline
    T_lat = (num_frames - 1) // 4 + 1
    t_src = np.linspace(0, 1, hl.shape[0])
    t_tgt = np.linspace(0, 1, T_lat)
    idx = np.searchsorted(t_src, t_tgt, side="left")
    idx = np.clip(idx, 0, hl.shape[0] - 1)
    hl_resampled = hl[idx]
    
    return torch.from_numpy(hl_resampled).long()


def parse_args():
    p = argparse.ArgumentParser("WanVideo Control Inference (jsonl) with HL")

    # input jsonl
    p.add_argument("--jsonl", type=str, required=True, help="Path to jsonl file, one sample per line")

    # output root dir
    p.add_argument("--out_dir", type=str, required=True, help="New directory to save outputs")

    # model root dir
    p.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/bn/aicoding-lq/luhaotian/ckpt/Wan2.1-Fun-V1.1-14B-Control",
        help="Root directory containing WanVideo model files"
    )

    # prompts
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--default_prompt", type=str, default="", help="Used when prompt in jsonl is empty")

    # generation settings
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)

    # video settings
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--quality", type=int, default=5)
    p.add_argument("--ext", type=str, default=".mp4", help="Output file extension, e.g. .mp4")

    # lora (optional)
    p.add_argument("--lora_path", type=str, default="")
    p.add_argument("--lora_alpha", type=float, default=1.0)
    
    # rank0 print freq
    p.add_argument("--print_freq", type=int, default=1)

    # resume / skip if exists
    p.add_argument("--skip_if_exists", action="store_true")

    return p.parse_args()


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                line_data = json.loads(line)
                yield line_no, line_data
            except Exception as e:
                print(f"[WARN] Bad json at line {line_no}: {e}")
                continue


def main():
    args = parse_args()

    # Build model paths
    model_dir = args.model_dir
    model_paths = [
        os.path.join(model_dir, "diffusion_pytorch_model.safetensors"),
        os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(model_dir, "Wan2.1_VAE.pth"),
        os.path.join(model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]

    # Sanity check
    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing and is_rank0():
        print("Missing model files:\n" + "\n".join(missing))
        # Depending on env, you might want to raise error or let it try downloading if implemented
        # raise FileNotFoundError(...)

    # Create pipeline (USP enabled if available/needed)
    # Note: from_pretrained supports list of ModelConfig
    config_list = [ModelConfig(model_id=p, origin_file_pattern=os.path.basename(p)) for p in model_paths]
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=config_list,
        use_usp=False, # Disable USP for inference simplicity unless needed
    )

    # Optional LoRA
    if args.lora_path and os.path.exists(args.lora_path):
        if is_rank0():
            print(f"Loading LoRA from {args.lora_path} with alpha={args.lora_alpha}")
        pipe.load_lora(pipe.dit, lora_config=args.lora_path, alpha=args.lora_alpha)

    # Directories
    if is_rank0():
        os.makedirs(args.out_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Process jsonl
    for line_no, item in iter_jsonl(args.jsonl):
        ref_path = item.get("reference_image", None)
        ctrl_path = item.get("control_video", None)
        video_id = item.get("video_id", f"line_{line_no}")
        hl_path = item.get("hl_npz", None)

        prompt = item.get("prompt", "")
        if (prompt is None) or (not str(prompt).strip()):
            prompt = args.default_prompt

        # Basic check
        if not ref_path or not ctrl_path:
            if is_rank0():
                print(f"[WARN] line={line_no} missing base paths (ref/ctrl), skip. item={item}")
            continue

        out_path = os.path.join(args.out_dir, f"{video_id}{args.ext}")
        
        if args.skip_if_exists and os.path.exists(out_path):
            if is_rank0() and line_no % args.print_freq == 0:
                print(f"[SKIP] exists: {out_path}")
            continue

        if is_rank0():
            print(f"[RUN] line={line_no} video_id={video_id}")
            print(f"      ref={ref_path}")
            print(f"      ctrl={ctrl_path}")
            print(f"      hl={hl_path}")
            print(f"      out={out_path}")

        # Data Loading
        try:
            # Control Video
            control_video_data = VideoData(ctrl_path, height=args.height, width=args.width)
            actual_frames = len(control_video_data)
            num_frames = min(args.num_frames, actual_frames)
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            control_video = [control_video_data[i] for i in range(num_frames)]

            # Reference Image
            reference_image_data = VideoData(ref_path, height=args.height, width=args.width)
            reference_image = reference_image_data[0]

            # HL Codes
            hl_codes = load_hl_codes(hl_path, num_frames=num_frames)
        except Exception as e:
            if is_rank0():
                print(f"[ERROR] Failed loading data for line {line_no}: {e}")
            continue

        # Pipeline Call
        video = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            control_video=control_video,
            reference_image=reference_image,
            hl_codes=hl_codes,
            seed=args.seed,
            tiled=True,
            num_inference_steps=args.num_inference_steps,
            num_frames=num_frames,
        )

        if is_rank0():
            save_video(video, out_path, fps=args.fps, quality=args.quality)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


if __name__ == "__main__":
    main()
