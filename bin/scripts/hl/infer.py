import argparse
import os
import json
import numpy as np
import torch
import torch.distributed as dist

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def parse_args():
    p = argparse.ArgumentParser("WanVideo Control + HL Inference (jsonl)")

    p.add_argument("--jsonl", type=str, required=True, help="Path to jsonl file, one sample per line")
    p.add_argument("--out_dir", type=str, required=True, help="New directory to save outputs")

    p.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/bn/aicoding-lq/luhaotian/ckpt/Wan2.1-Fun-V1.1-14B-Control",
        help="Root directory containing WanVideo model files"
    )

    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--default_prompt", type=str, default="", help="Used when prompt in jsonl is empty")

    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--quality", type=int, default=5)
    p.add_argument("--ext", type=str, default=".mp4", help="Output file extension, e.g. .mp4")

    p.add_argument("--lora_path", type=str, default="")
    p.add_argument("--lora_alpha", type=float, default=1.0)

    p.add_argument("--hl_key", type=str, default="hl", help="Key name inside hl npz")
    p.add_argument("--hl_scale", type=float, default=0.1, help="Scale factor for HL context")

    p.add_argument("--skip_if_exists", action="store_true")

    return p.parse_args()


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad json at line {line_no}: {e}\nLINE: {line[:2000]}")


def load_hl_codes(hl_path, num_frames, hl_key="hl"):
    if not hl_path or (not os.path.exists(hl_path)):
        return None
    npz = np.load(hl_path)
    if hl_key in npz:
        hl = npz[hl_key]
    elif "hl" in npz:
        hl = npz["hl"]
    else:
        hl = npz[list(npz.keys())[0]]
    hl = np.asarray(hl)
    if hl.ndim == 1:
        hl = hl[:, None]
    t_lat = (num_frames - 1) // 4 + 1
    t_src = np.linspace(0, 1, hl.shape[0])
    t_tgt = np.linspace(0, 1, t_lat)
    idx = np.searchsorted(t_src, t_tgt, side="left")
    idx = np.clip(idx, 0, hl.shape[0] - 1)
    hl_resampled = hl[idx]
    return torch.from_numpy(hl_resampled).long()


def main():
    args = parse_args()

    model_dir = args.model_dir
    model_paths = [
        os.path.join(model_dir, "diffusion_pytorch_model.safetensors"),
        os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(model_dir, "Wan2.1_VAE.pth"),
        os.path.join(model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]

    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing and is_rank0():
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(path=p) for p in model_paths],
        use_usp=True,
    )

    if args.lora_path:
        pipe.load_lora(pipe.dit, lora_config=args.lora_path, alpha=args.lora_alpha)

    if is_rank0():
        os.makedirs(args.out_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    for line_no, item in iter_jsonl(args.jsonl):
        ref_path = item.get("reference_image", None)
        ctrl_path = item.get("control_video", None)
        hl_path = item.get("hl_npz", None)
        video_id = item.get("video_id", f"line_{line_no}")

        prompt = item.get("prompt", "")
        if (prompt is None) or (not str(prompt).strip()):
            prompt = args.default_prompt

        if not ref_path or not ctrl_path:
            if is_rank0():
                print(f"[WARN] line={line_no} missing paths, skip. item={item}")
            continue

        out_path = os.path.join(args.out_dir, video_id + args.ext)
        out_parent = os.path.dirname(out_path)

        if args.skip_if_exists and os.path.exists(out_path):
            if is_rank0():
                print(f"[SKIP] exists: {out_path}")
            continue

        if is_rank0():
            os.makedirs(out_parent, exist_ok=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if is_rank0():
            print(f"[RUN] line={line_no} video_id={video_id}")
            print(f"      ref={ref_path}")
            print(f"      ctrl={ctrl_path}")
            print(f"      hl={hl_path}")
            print(f"      out={out_path}")

        control_video_data = VideoData(ctrl_path, height=args.height, width=args.width)
        actual_frames = len(control_video_data)
        num_frames = min(args.num_frames, actual_frames)
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        control_video = [control_video_data[i] for i in range(num_frames)]

        reference_image_data = VideoData(ref_path, height=args.height, width=args.width)
        reference_image = reference_image_data[0]

        hl_codes = load_hl_codes(hl_path, num_frames, hl_key=args.hl_key)

        video = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            control_video=control_video,
            reference_image=reference_image,
            hl_codes=hl_codes,
            hl_scale=args.hl_scale,
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
