#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import fnmatch
from pathlib import Path

import h5py
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

MP_HAND_CONNECTIONS = [
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

def get_egodex_to_mediapipe_mapping(hand_side: str = 'left'):
    prefix = hand_side
    return {
        0:  f'{prefix}Hand',
        1:  f'{prefix}ThumbKnuckle',
        2:  f'{prefix}ThumbIntermediateBase',
        3:  f'{prefix}ThumbIntermediateTip',
        4:  f'{prefix}ThumbTip',
        5:  f'{prefix}IndexFingerKnuckle',
        6:  f'{prefix}IndexFingerIntermediateBase',
        7:  f'{prefix}IndexFingerIntermediateTip',
        8:  f'{prefix}IndexFingerTip',
        9:  f'{prefix}MiddleFingerKnuckle',
        10: f'{prefix}MiddleFingerIntermediateBase',
        11: f'{prefix}MiddleFingerIntermediateTip',
        12: f'{prefix}MiddleFingerTip',
        13: f'{prefix}RingFingerKnuckle',
        14: f'{prefix}RingFingerIntermediateBase',
        15: f'{prefix}RingFingerIntermediateTip',
        16: f'{prefix}RingFingerTip',
        17: f'{prefix}LittleFingerKnuckle',
        18: f'{prefix}LittleFingerIntermediateBase',
        19: f'{prefix}LittleFingerIntermediateTip',
        20: f'{prefix}LittleFingerTip',
    }

def convert_to_camera_frame(tfs: np.ndarray, cam_ext: np.ndarray) -> np.ndarray:
    if tfs.ndim == 2:
        tfs = tfs[None]
    inv_cam = np.linalg.inv(cam_ext)[None]
    return inv_cam @ tfs

def project_point_3d_to_2d(pt3: np.ndarray, cam_int: np.ndarray, W: int, H: int):
    x, y, z = pt3
    if z <= 0:
        return None
    uvw = cam_int @ np.array([x, y, z], dtype=np.float32)
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    u_i = int(round(u))
    v_i = int(round(v))
    if u_i < 0 or u_i >= W or v_i < 0 or v_i >= H:
        return None
    return (u_i, v_i)

def draw_hand_skeleton(img, landmarks_2d, connections, color=(0, 255, 0),
                       line_thickness=2, circle_radius=3):
    for i, j in connections:
        pi = landmarks_2d[i] if i < len(landmarks_2d) else None
        pj = landmarks_2d[j] if j < len(landmarks_2d) else None
        if pi is not None and pj is not None:
            cv2.line(img, pi, pj, color, line_thickness, lineType=cv2.LINE_AA)
    for pt in landmarks_2d:
        if pt is not None:
            cv2.circle(img, pt, circle_radius + 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, pt, circle_radius, color, 1, lineType=cv2.LINE_AA)

def get_video_size_from_original(hdf5_path: Path, video_ext: str = ".mp4"):
    video_path = hdf5_path.with_suffix(video_ext)
    if not video_path.exists():
        raise FileNotFoundError(f"Cannot find original video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def make_output_path(hdf5_path: Path, data_dir: Path, output_dir: Path, keep_tree: bool):
    """
    keep_tree=True:
      data_dir/task/0.hdf5 -> output_dir/task/0.mp4
    keep_tree=False:
      output_dir/task__0.mp4 (旧方式，当前不建议)
    """
    if keep_tree:
        rel = hdf5_path.relative_to(data_dir).with_suffix(".mp4")  # task/0.mp4
        out_mp4 = output_dir / rel
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        return out_mp4
    else:
        task_name = hdf5_path.parent.name
        episode_id = hdf5_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{task_name}__{episode_id}.mp4"

def process_single_episode(hdf5_path: str,
                           data_dir: str,
                           output_dir: str,
                           fps: float = 30.0,
                           video_ext: str = ".mp4",
                           keep_tree: bool = True):
    hdf5_path = Path(hdf5_path)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    out_mp4 = make_output_path(hdf5_path, data_dir, output_dir, keep_tree)

    render_width, render_height = get_video_size_from_original(hdf5_path, video_ext=video_ext)

    with h5py.File(hdf5_path, "r") as f:
        cam_ext_all = f['/transforms/camera'][:]      # (N,4,4)
        N = cam_ext_all.shape[0]
        cam_int = f['/camera/intrinsic'][:]          # (3,3)

        left_map = get_egodex_to_mediapipe_mapping('left')
        right_map = get_egodex_to_mediapipe_mapping('right')

        def exists_joint(name):
            return f'/transforms/{name}' in f

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (render_width, render_height))

        task_name = hdf5_path.parent.name
        episode_id = hdf5_path.stem

        for frame_idx in tqdm(range(N), desc=f"{task_name}/{episode_id}"):
            frame_img = np.zeros((render_height, render_width, 3), dtype=np.uint8)
            cam_ext = cam_ext_all[frame_idx]

            left_landmarks_2d = []
            for mp_idx in range(21):
                jname = left_map[mp_idx]
                if exists_joint(jname):
                    tf_world = f[f'/transforms/{jname}'][frame_idx]
                    tf_cam = convert_to_camera_frame(tf_world, cam_ext)[0]
                    pos3 = tf_cam[:3, 3]
                    left_landmarks_2d.append(
                        project_point_3d_to_2d(pos3, cam_int, render_width, render_height)
                    )
                else:
                    left_landmarks_2d.append(None)

            right_landmarks_2d = []
            for mp_idx in range(21):
                jname = right_map[mp_idx]
                if exists_joint(jname):
                    tf_world = f[f'/transforms/{jname}'][frame_idx]
                    tf_cam = convert_to_camera_frame(tf_world, cam_ext)[0]
                    pos3 = tf_cam[:3, 3]
                    right_landmarks_2d.append(
                        project_point_3d_to_2d(pos3, cam_int, render_width, render_height)
                    )
                else:
                    right_landmarks_2d.append(None)

            draw_hand_skeleton(frame_img, left_landmarks_2d, MP_HAND_CONNECTIONS, color=(0, 255, 0))
            draw_hand_skeleton(frame_img, right_landmarks_2d, MP_HAND_CONNECTIONS, color=(0, 0, 255))

            vw.write(frame_img)

        vw.release()

    return str(out_mp4)

def find_all_hdf5(dataset_root: str):
    h5_files = []
    for root, dirs, files in os.walk(dataset_root):
        for filename in fnmatch.filter(files, "*.hdf5"):
            h5_files.append(os.path.join(root, filename))
    h5_files.sort()
    return h5_files

def _worker_process_one(args_tuple):
    h5_path, data_dir, output_dir, fps, video_ext, overwrite, keep_tree = args_tuple
    h5_path = Path(h5_path)
    out_mp4 = make_output_path(h5_path, Path(data_dir), Path(output_dir), keep_tree)

    if out_mp4.exists() and not overwrite:
        return str(out_mp4)

    return process_single_episode(
        hdf5_path=str(h5_path),
        data_dir=data_dir,
        output_dir=output_dir,
        fps=fps,
        video_ext=video_ext,
        keep_tree=keep_tree
    )

def main():
    parser = argparse.ArgumentParser(
        description="把 EgoDex 的 3D 手部姿态按 MediaPipe 21 点格式渲染为黑底骨架视频"
    )
    parser.add_argument('--data_dir', required=True,
                        help='EgoDex 数据集根目录（包含各个 task 子目录）')
    parser.add_argument('--output_dir', required=True,
                        help='输出目录（默认保持与 data_dir 相同的子目录结构）')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--max_episodes', type=int, default=-1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--video_ext', type=str, default=".mp4")
    parser.add_argument('--keep_tree', action='store_true', default=True,
                        help='保持 data_dir 的嵌套目录结构输出（默认开启）')
    args = parser.parse_args()

    h5_files = find_all_hdf5(args.data_dir)
    if args.max_episodes > 0:
        h5_files = h5_files[:args.max_episodes]

    print(f"Found {len(h5_files)} hdf5 episodes")

    num_workers = 4
    args_list = [
        (h5_path, args.data_dir, args.output_dir, args.fps, args.video_ext, args.overwrite, args.keep_tree)
        for h5_path in h5_files
    ]

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_worker_process_one, args_list),
                      total=len(args_list), desc='All episodes'):
            pass

if __name__ == "__main__":
    main()