# -*- coding: utf-8 -*-
"""
并发批量提取视频手部关键点（MediaPipe Tasks HandLandmarker, VIDEO mode）
- 输入：一个包含多段视频的目录，或一个 metadata jsonl 的列表
- 输出：每个视频一个 JSON（保留输入子目录结构，文件名无后缀） + 总汇总 processing_summary.json
"""

import os
import json
import math
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def download_hand_model(model_path: str):
    model_path = Path(model_path)
    if model_path.exists():
        return str(model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    print(f"[MODEL] downloading to: {model_path}")
    urllib.request.urlretrieve(url, str(model_path))
    print("[MODEL] download done.")
    return str(model_path)


def iter_videos(input_dir: str, recursive: bool = False):
    """不使用 metadata 时，遍历目录中所有视频"""
    input_dir = Path(input_dir)
    if recursive:
        files = input_dir.rglob("*")
    else:
        files = input_dir.glob("*")
    for p in files:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def read_videos_from_metadata(metadata_jsonl: str, input_dir: str):
    """
    从 metadata jsonl 里读取要处理的视频路径。

    每行形如：
    {"reference_image": ".../dry_hands/24.jpg",
     "control_video": ".../dry_hands/24.mp4",
     "video": ".../dry_hands/24.mp4",
     "prompt": "",
     "video_id": "dry_hands/24"}

    这里假设要处理的是字段 "video" 对应的视频路径。
    若 video 是绝对路径，直接使用；
    若是相对路径，则相对于 input_dir。
    """
    input_dir = Path(input_dir)
    videos = []
    metadata_items = []

    metadata_path = Path(metadata_jsonl)
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            metadata_items.append(obj)
            if "video" not in obj:
                continue
            v = obj["video"]
            v_path = Path(v)
            if not v_path.is_absolute():
                v_path = input_dir / v_path
            videos.append(v_path)

    return videos, metadata_items


def safe_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or math.isnan(fps) or fps <= 1e-6:
        return 30.0
    return float(fps)


def process_single_video(video_path: str, input_dir: str, output_dir: str,
                         model_path: str, num_hands: int = 2):
    video_path = Path(video_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 保留相对路径结构
    try:
        rel_path = video_path.relative_to(input_dir)
    except ValueError:
        # 如果 video_path 不在 input_dir 下，就把它平铺到输出根目录
        rel_path = Path(video_path.name)

    out_dir = output_dir / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{video_path.stem}.json"

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = safe_fps(cap)

    results_per_frame = []
    frame_idx = 0

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                timestamp_ms = int(round(frame_idx * 1000.0 / fps))

                det = landmarker.detect_for_video(mp_image, timestamp_ms)

                frame_item = {
                    "frame_index": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "hands": []
                }

                if det.hand_landmarks:
                    for h in range(len(det.hand_landmarks)):
                        handed = None
                        if det.handedness and len(det.handedness) > h and det.handedness[h]:
                            handed = {
                                "category_name": det.handedness[h][0].category_name,
                                "score": float(det.handedness[h][0].score),
                            }

                        lm_2d = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                                 for lm in det.hand_landmarks[h]]

                        lm_3d = []
                        if det.hand_world_landmarks and len(det.hand_world_landmarks) > h:
                            lm_3d = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                                     for lm in det.hand_world_landmarks[h]]

                        frame_item["hands"].append({
                            "handedness": handed,
                            "landmarks": lm_2d,
                            "world_landmarks": lm_3d,
                        })

                results_per_frame.append(frame_item)
                frame_idx += 1

    finally:
        cap.release()

    payload = {
        "video_path": str(video_path),
        "video_name": video_path.stem,
        "fps": fps,
        "total_frames_processed": frame_idx,
        "results": results_per_frame
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "video_path": str(video_path),
        "output_json": str(out_json),
        "frames": frame_idx,
        "success": True
    }


def batch_process(input_dir: str, output_dir: str, model_path: str,
                  max_workers: int = 4, recursive: bool = False,
                  metadata_jsonl: str | None = None):
    model_path = download_hand_model(model_path)

    # 新增：根据 metadata_jsonl 决定要处理哪些视频
    metadata_items = None
    if metadata_jsonl:
        videos, metadata_items = read_videos_from_metadata(metadata_jsonl, input_dir)
    else:
        videos = list(iter_videos(input_dir, recursive=recursive))

    if not videos:
        raise RuntimeError(f"No videos found to process. "
                           f"input_dir={input_dir}, metadata_jsonl={metadata_jsonl}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {
            ex.submit(process_single_video, str(v), str(input_dir), str(output_dir), model_path): v
            for v in videos
        }

        for fut in as_completed(fut_map):
            v = fut_map[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"[OK] {v.name} -> {Path(r['output_json']).relative_to(output_dir)} ({r['frames']} frames)")
            except Exception as e:
                err = {"video_path": str(v), "success": False, "error": str(e)}
                results.append(err)
                print(f"[FAIL] {v.name} -> {e}")

    summary = {
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_path": str(Path(model_path).resolve()),
        "total_videos": len(videos),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
    }

    # 若使用了 metadata，可以顺便把 metadata 信息也写入 summary
    if metadata_items is not None:
        summary["metadata_jsonl"] = str(Path(metadata_jsonl).resolve())
        summary["metadata_items_count"] = len(metadata_items)

    print(f"\n[DONE] summary -> {summary_path}")


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="包含视频的目录（用于相对路径基准）")
    ap.add_argument("--output_dir", type=str, required=True, help="输出目录（保留子目录结构）")
    ap.add_argument("--model_path", type=str,
                    default="/mnt/bn/aicoding-lq/luhaotian/ckpt/hand_landmarker.task",
                    help="hand_landmarker.task 路径（不存在会自动下载）")
    ap.add_argument("--max_workers", type=int, default=4, help="并发进程数")
    ap.add_argument("--recursive", type=str2bool, default="true", help="递归搜索子目录视频（未提供 metadata 时生效）")
    ap.add_argument("--metadata_jsonl", type=str, default=None,
                    help="可选：metadata jsonl 文件，只处理其中列出的 video 字段对应视频")
    args = ap.parse_args()

    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        max_workers=args.max_workers,
        recursive=args.recursive,
        metadata_jsonl=args.metadata_jsonl,
    )


if __name__ == "__main__":
    main()