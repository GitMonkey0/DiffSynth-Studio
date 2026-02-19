#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fnmatch
import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# MediaPipe Hands 21 bones (parent->child) edges = 20
MP_BONES_20 = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # little
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


def find_all_hdf5(dataset_root: str):
    h5_files = []
    for root, _, files in os.walk(dataset_root):
        for filename in fnmatch.filter(files, "*.hdf5"):
            h5_files.append(os.path.join(root, filename))
    h5_files.sort()
    return h5_files


def make_output_path(hdf5_path: Path, data_dir: Path, output_dir: Path, keep_tree: bool, suffix=".npz"):
    if keep_tree:
        rel = hdf5_path.relative_to(data_dir).with_suffix(suffix)  # task/0.npz
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    else:
        task = hdf5_path.parent.name
        eid = hdf5_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{task}__{eid}{suffix}"


def convert_to_camera_frame(tf_world: np.ndarray, cam_ext_world: np.ndarray) -> np.ndarray:
    # cam_ext_world == T_cam_world
    # want T_joint_cam = inv(T_cam_world) @ T_joint_world
    return np.linalg.inv(cam_ext_world) @ tf_world


def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def build_hand_frame(P21: np.ndarray):
    """
    P21: (21,3) in camera frame
    returns R (3,3) where columns are [X,Y,Z] in camera coordinates,
    so that v_hand = R^T (v_cam)
    """
    wrist = P21[0]
    mid_mcp = P21[9]
    idx_mcp = P21[5]

    Y = normalize(mid_mcp - wrist)
    if Y is None:
        return None

    v = idx_mcp - wrist
    # remove Y component to get something in palm plane perpendicular-ish to Y
    v_proj = v - np.dot(v, Y) * Y
    Z = normalize(v_proj)
    if Z is None:
        return None

    X = normalize(np.cross(Y, Z))
    if X is None:
        return None

    # re-orthogonalize Z to be safe
    Z = normalize(np.cross(X, Y))
    if Z is None:
        return None

    R = np.stack([X, Y, Z], axis=1)  # columns
    return R


def prototypes_26():
    """
    26 direction prototypes on a cube:
    - 8 vertices: (±1,±1,±1)
    - 12 edge centers: permutations of (±1,±1,0)
    - 6 face centers: (±1,0,0),(0,±1,0),(0,0,±1)
    normalized to unit vectors
    returns (26,3)
    """
    dirs = []

    # vertices
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                dirs.append([sx, sy, sz])

    # edge centers: two non-zero coords
    vals = (-1, 1)
    # (±1,±1,0)
    for sx in vals:
        for sy in vals:
            dirs.append([sx, sy, 0])
            dirs.append([sx, 0, sy])
            dirs.append([0, sx, sy])

    # face centers
    dirs += [
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
    ]

    D = np.array(dirs, dtype=np.float32)
    # remove potential duplicates (shouldn't, but safe)
    D = np.unique(D, axis=0)
    # normalize
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
    assert Dn.shape[0] == 26, f"Expected 26 prototypes, got {Dn.shape[0]}"
    return Dn


PROTO_26 = prototypes_26()


def quantize_dir_to_26(v_unit: np.ndarray) -> int:
    # choose max cosine similarity
    sims = PROTO_26 @ v_unit.astype(np.float32)
    return int(np.argmax(sims))


def extract_hl_one_hand(P21_cam: np.ndarray):
    """
    P21_cam: (21,3) camera frame
    returns:
      hl20: (20,) int64, each in [0,25]
      valid: bool
    """
    if np.any(~np.isfinite(P21_cam)):
        return None, False

    R = build_hand_frame(P21_cam)
    if R is None:
        return None, False

    hl = np.zeros((20,), dtype=np.int64)
    for k, (p, c) in enumerate(MP_BONES_20):
        v_cam = P21_cam[c] - P21_cam[p]
        v_cam_u = normalize(v_cam)
        if v_cam_u is None:
            return None, False
        # to hand frame
        v_hand = R.T @ v_cam_u
        v_hand_u = normalize(v_hand)
        if v_hand_u is None:
            return None, False
        hl[k] = quantize_dir_to_26(v_hand_u)
    return hl, True


def process_single_episode(hdf5_path: str,
                           data_dir: str,
                           output_dir: str,
                           keep_tree: bool = True,
                           overwrite: bool = False):
    hdf5_path = Path(hdf5_path)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    out_npz = make_output_path(hdf5_path, data_dir, output_dir, keep_tree, suffix=".npz")
    if out_npz.exists() and not overwrite:
        return str(out_npz)

    left_map = get_egodex_to_mediapipe_mapping('left')
    right_map = get_egodex_to_mediapipe_mapping('right')

    with h5py.File(hdf5_path, "r") as f:
        cam_ext_all = f['/transforms/camera'][:]  # (N,4,4) T_cam_world
        N = cam_ext_all.shape[0]

        def has_joint(name): return f'/transforms/{name}' in f
        # ensure all 21 exist, but not strictly required (we'll mark invalid frames)
        for mp_i in range(21):
            if not has_joint(left_map[mp_i]) or not has_joint(right_map[mp_i]):
                # still continue, will produce invalid frames
                pass

        hl_left = np.full((N, 20), -1, dtype=np.int64)
        hl_right = np.full((N, 20), -1, dtype=np.int64)
        hl_both = np.full((N, 40), -1, dtype=np.int64)
        valid_left = np.zeros((N,), dtype=np.uint8)
        valid_right = np.zeros((N,), dtype=np.uint8)

        # optional confidences (not all users want to threshold)
        conf = f.get('/confidences', None)

        for i in range(N):
            T_cam_world = cam_ext_all[i]

            # gather P21 in camera frame for each hand
            P_left = np.full((21, 3), np.nan, dtype=np.float32)
            P_right = np.full((21, 3), np.nan, dtype=np.float32)

            for mp_i in range(21):
                jn = left_map[mp_i]
                if has_joint(jn):
                    T_joint_world = f[f'/transforms/{jn}'][i]
                    T_joint_cam = convert_to_camera_frame(T_joint_world, T_cam_world)
                    P_left[mp_i] = T_joint_cam[:3, 3]

                jn = right_map[mp_i]
                if has_joint(jn):
                    T_joint_world = f[f'/transforms/{jn}'][i]
                    T_joint_cam = convert_to_camera_frame(T_joint_world, T_cam_world)
                    P_right[mp_i] = T_joint_cam[:3, 3]

            hlL, okL = extract_hl_one_hand(P_left)
            hlR, okR = extract_hl_one_hand(P_right)

            if okL:
                hl_left[i] = hlL
                valid_left[i] = 1
            if okR:
                hl_right[i] = hlR
                valid_right[i] = 1

            if okR and okL:
                hl_both[i, :20] = hlR
                hl_both[i, 20:] = hlL
            elif okR:
                hl_both[i, :20] = hlR
            elif okL:
                hl_both[i, 20:] = hlL

        # save
        meta = {}
        for k in ['task', 'session_name', 'environment', 'object', 'llm_type',
                  'llm_description', 'llm_description2', 'which_llm_description']:
            if k in f.attrs:
                v = f.attrs[k]
                try:
                    meta[k] = v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else str(v)
                except Exception:
                    meta[k] = str(v)

        np.savez_compressed(
            out_npz,
            hl_both=hl_both,           # (N,40) 右20 + 左20，invalid=-1
            hl_right=hl_right,         # (N,20)
            hl_left=hl_left,           # (N,20)
            valid_right=valid_right,   # (N,)
            valid_left=valid_left,     # (N,)
            mp_bones_20=np.array(MP_BONES_20, dtype=np.int32),
            proto_26=PROTO_26.astype(np.float32),
            meta=meta
        )

    return str(out_npz)


def _worker(args):
    return process_single_episode(*args)


def main():
    ap = argparse.ArgumentParser("Extract HL features (26-way) from EgoDex hdf5")
    ap.add_argument('--data_dir', required=True, help='EgoDex root containing task subdirs')
    ap.add_argument('--output_dir', required=True, help='Output root dir for HL features')
    ap.add_argument('--keep_tree', action='store_true', default=True, help='Keep task/episode tree (default=True)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max_episodes', type=int, default=-1)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    h5s = find_all_hdf5(args.data_dir)
    if args.max_episodes > 0:
        h5s = h5s[:args.max_episodes]

    tasks = [(p, args.data_dir, args.output_dir, args.keep_tree, args.overwrite) for p in h5s]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc="Extract HL"):
            pass


if __name__ == "__main__":
    main()