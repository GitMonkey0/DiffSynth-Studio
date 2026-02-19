# -*- coding: utf-8 -*-  
"""  
只依赖手部关键点3D坐标( world_landmarks )的评估指标（GT vs Pred）  
输入：  
- gt_dir: GT关键点json目录（每视频一个 json，可带或不带 _hand_landmarks 后缀）  
- pred_dir: Pred关键点json目录（与gt同名stem匹配）  
输出：  
- out_dir: 每视频指标（保留子目录结构） + 总汇总  
可选：  
- occlusion_json: 外部遮挡/可见性标注（若没有则跳过遮挡相关指标）  
"""  
  
import os  
import json  
import argparse  
from pathlib import Path  
from typing import Dict, Any, List, Optional, Tuple  
  
import numpy as np  
  
  
FINGER_GROUPS = {  
    "thumb":  [1, 2, 3, 4],  
    "index":  [5, 6, 7, 8],  
    "middle": [9, 10, 11, 12],  
    "ring":   [13, 14, 15, 16],  
    "pinky":  [17, 18, 19, 20],  
    "wrist":  [0],  
    "all":    list(range(21)),  
}  
  
BONES = [  
    (0, 1), (1, 2), (2, 3), (3, 4),  
    (0, 5), (5, 6), (6, 7), (7, 8),  
    (0, 9), (9,10), (10,11), (11,12),  
    (0,13), (13,14), (14,15), (15,16),  
    (0,17), (17,18), (18,19), (19,20),  
]  
  
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}  
  
  
def load_json(path: Path) -> Dict[str, Any]:  
    with open(path, "r", encoding="utf-8") as f:  
        return json.load(f)  
  
  
def extract_frame_hands_world(frame_item: Dict[str, Any]) -> List[Dict[str, Any]]:  
    hands = []  
    for h in frame_item.get("hands", []):  
        handed = None  
        score = None  
        if h.get("handedness"):  
            handed = h["handedness"].get("category_name")  
            score = h["handedness"].get("score")  
        w = h.get("world_landmarks") or []  
        if len(w) != 21:  
            xyz = None  
        else:  
            xyz = np.array([[p["x"], p["y"], p["z"]] for p in w], dtype=np.float32)  
        hands.append({"handedness": handed, "score": score, "xyz": xyz})  
    return hands  
  
  
def center_root(xyz: np.ndarray, root_idx: int = 0) -> np.ndarray:  
    return xyz - xyz[root_idx:root_idx+1, :]  
  
  
def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:  
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))  
  
  
def rigid_align_pa(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:  
    mu_p = pred.mean(axis=0, keepdims=True)  
    mu_g = gt.mean(axis=0, keepdims=True)  
    X = pred - mu_p  
    Y = gt - mu_g  
    H = X.T @ Y  
    U, S, Vt = np.linalg.svd(H)  
    R = Vt.T @ U.T  
    if np.linalg.det(R) < 0:  
        Vt[-1, :] *= -1  
        R = Vt.T @ U.T  
    pred_aligned = (X @ R) + mu_g  
    return pred_aligned  
  
  
def linear_regression_slope(y: np.ndarray) -> float:  
    T = len(y)  
    if T < 2:  
        return float("nan")  
    t = np.arange(T, dtype=np.float32)  
    A = np.stack([t, np.ones_like(t)], axis=1)  
    a, b = np.linalg.lstsq(A, y.astype(np.float32), rcond=None)[0]  
    return float(a)  
  
  
def second_diff_jitter(seq: np.ndarray) -> float:  
    if seq.shape[0] < 3:  
        return float("nan")  
    acc = seq[2:] - 2 * seq[1:-1] + seq[:-2]  
    return float(np.mean(np.linalg.norm(acc, axis=-1)))  
  
  
def diff_variance(seq: np.ndarray, order: int = 1) -> float:  
    if seq.shape[0] <= order:  
        return float("nan")  
    d = seq.copy()  
    for _ in range(order):  
        d = np.diff(d, axis=0)  
    var = np.var(d, axis=0)  
    return float(np.mean(var))  
  
  
def pck_3d(pred: np.ndarray, gt: np.ndarray, delta: float) -> float:  
    if pred.ndim == 2:  
        dist = np.linalg.norm(pred - gt, axis=1)  
        return float(np.mean(dist < delta))  
    dist = np.linalg.norm(pred - gt, axis=-1)  
    return float(np.mean(dist < delta))  
  
  
def bone_length_consistency(seq: np.ndarray, bones=BONES) -> Dict[str, float]:  
    if seq.shape[0] < 2:  
        return {"mean_std": float("nan")}  
    stds = []  
    out = {}  
    for (i, k) in bones:  
        L = np.linalg.norm(seq[:, i, :] - seq[:, k, :], axis=1)  
        s = float(np.std(L))  
        out[f"{i}-{k}"] = s  
        stds.append(s)  
    out["mean_std"] = float(np.mean(stds)) if stds else float("nan")  
    return out  
  
  
def fingertip_independence_corr(seq: np.ndarray, use_velocity: bool = True) -> Dict[str, Any]:  
    if seq.shape[0] < 3:  
        return {"names": list(FINGERTIPS.keys()), "corr": None}  
    X = seq  
    if use_velocity:  
        X = np.diff(X, axis=0)  
    names = list(FINGERTIPS.keys())  
    feats = []  
    for n in names:  
        j = FINGERTIPS[n]  
        v = X[:, j, :].reshape(-1)  
        feats.append(v)  
    M = np.stack(feats, axis=0)  
    corr = np.corrcoef(M)  
    return {"names": names, "corr": corr.tolist()}  
  
  
def match_hands(gt_hands: List[Dict[str, Any]], pr_hands: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, np.ndarray]]:  
    gt_valid = [h for h in gt_hands if h["xyz"] is not None]  
    pr_valid = [h for h in pr_hands if h["xyz"] is not None]  
    if not gt_valid or not pr_valid:  
        return []  
    gt_by = {h["handedness"]: h for h in gt_valid if h["handedness"] in ("Left", "Right")}  
    pr_by = {h["handedness"]: h for h in pr_valid if h["handedness"] in ("Left", "Right")}  
    pairs = []  
    for side in ("Left", "Right"):  
        if side in gt_by and side in pr_by:  
            pairs.append((gt_by[side]["xyz"], pr_by[side]["xyz"]))  
    if pairs:  
        return pairs  
    best = []  
    used_pr = set()  
    for g in gt_valid:  
        best_j = None  
        best_e = None  
        for j, p in enumerate(pr_valid):  
            if j in used_pr:  
                continue  
            e = mpjpe(p["xyz"], g["xyz"])  
            if best_e is None or e < best_e:  
                best_e = e  
                best_j = j  
        if best_j is not None:  
            used_pr.add(best_j)  
            best.append((g["xyz"], pr_valid[best_j]["xyz"]))  
        if len(best) == 2:  
            break  
    return best  
  
  
def build_aligned_sequences(gt_data: Dict[str, Any], pr_data: Dict[str, Any]) -> Dict[str, Any]:  
    gt_frames = gt_data.get("results", [])  
    pr_frames = pr_data.get("results", [])  
    T = min(len(gt_frames), len(pr_frames))  
    gt_seq = []  
    pr_seq = []  
    valid = []  
    for t in range(T):  
        gt_hands = extract_frame_hands_world(gt_frames[t])  
        pr_hands = extract_frame_hands_world(pr_frames[t])  
        pairs = match_hands(gt_hands, pr_hands)  
        if not pairs:  
            valid.append(False)  
            gt_seq.append(np.zeros((21, 3), dtype=np.float32))  
            pr_seq.append(np.zeros((21, 3), dtype=np.float32))  
            continue  
        g_stack = np.stack([g for (g, p) in pairs], axis=0)  
        p_stack = np.stack([p for (g, p) in pairs], axis=0)  
        gt_seq.append(np.mean(g_stack, axis=0))  
        pr_seq.append(np.mean(p_stack, axis=0))  
        valid.append(True)  
    return {  
        "valid_mask": np.array(valid, dtype=bool),  
        "gt_seq": np.stack(gt_seq, axis=0),  
        "pr_seq": np.stack(pr_seq, axis=0),  
    }  
  
  
def eval_video_metrics(gt_path: Path, pr_path: Path,
                       pck_delta: float,
                       occlusion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    gt = load_json(gt_path)
    pr = load_json(pr_path)

    seq = build_aligned_sequences(gt, pr)
    mask = seq["valid_mask"]
    gt_seq = seq["gt_seq"][mask]  # (Tv,21,3)
    pr_seq = seq["pr_seq"][mask]
    Tv = gt_seq.shape[0]

    total_frames_compared = int(
        min(len(gt.get("results", [])), len(pr.get("results", [])))
    )

    out: Dict[str, Any] = {
        "video_name": gt.get("video_name", gt_path.stem),
        "gt_path": str(gt_path),
        "pred_path": str(pr_path),
        "total_frames_compared": total_frames_compared,
        "valid_frames": int(Tv),
        "valid_ratio": float(Tv / max(1, total_frames_compared)),
    }

    if Tv == 0:
        out["error"] = "No valid matched frames (no hands detected or failed match)."
        return out
    
    e_mpjpe = np.array([mpjpe(pr_seq[t], gt_seq[t]) for t in range(Tv)], dtype=np.float32)
    e_ra = np.array([mpjpe(center_root(pr_seq[t]), center_root(gt_seq[t])) for t in range(Tv)], dtype=np.float32)
    e_pa = np.array([mpjpe(rigid_align_pa(pr_seq[t], gt_seq[t]), gt_seq[t]) for t in range(Tv)], dtype=np.float32)

    out["MPJPE_mean"] = float(np.mean(e_mpjpe))
    out["RA_MPJPE_mean"] = float(np.mean(e_ra))
    out["PA_MPJPE_mean"] = float(np.mean(e_pa))
    out["MPJPE_drift_slope"] = linear_regression_slope(e_mpjpe)
    out["Jitter_acc2_mean"] = second_diff_jitter(pr_seq)
    out["Velocity_var_mean"] = diff_variance(pr_seq, order=1)
    out["Acceleration_var_mean"] = diff_variance(pr_seq, order=2)
    out["Jerk_var_mean"] = diff_variance(pr_seq, order=3) 
    pr_ra = center_root(pr_seq, root_idx=0)  
    gt_ra = center_root(gt_seq, root_idx=0)  
    finger_wise = {}  
    for name, idxs in FINGER_GROUPS.items():  
        if name == "all":  
            continue  
        idxs = np.array(idxs, dtype=int)  
        d = np.linalg.norm(pr_ra[:, idxs, :] - gt_ra[:, idxs, :], axis=-1)  
        finger_wise[name] = float(np.mean(d))  
    out["FingerWise_RA_MPJPE"] = finger_wise  
    out["PCK3D_delta"] = float(pck_delta)  
    out["PCK3D"] = pck_3d(pr_seq, gt_seq, pck_delta)  
    out["PCK3D_RA"] = pck_3d(pr_ra, gt_ra, pck_delta)  
    out["BLC_pred"] = bone_length_consistency(pr_seq)  
    out["BLC_gt"] = bone_length_consistency(gt_seq)  
    out["FingerIndependenceCorr_pred"] = fingertip_independence_corr(pr_seq, use_velocity=True)  
    if occlusion is not None:  
        stem = gt_path.stem  # 直接使用 stem，不再去掉后缀  
        vid_ann = occlusion.get(stem) or occlusion.get(gt.get("video_name", "")) or None  
        if vid_ann and "visible" in vid_ann:  
            vis_full = np.array(vid_ann["visible"], dtype=np.int32)  
            T0 = out["total_frames_compared"]  
            vis_full = vis_full[:T0]  
            vis_valid = vis_full[seq["valid_mask"]]  
            occ = (vis_valid == 0)  
            vis = (vis_valid == 1)  
            def mean_or_nan(x):  
                return float(np.mean(x)) if x.size else float("nan")  
            out["MPJPE_visible"] = mean_or_nan(e_mpjpe[vis])  
            out["MPJPE_occluded"] = mean_or_nan(e_mpjpe[occ])  
            if np.isfinite(out["MPJPE_visible"]) and out["MPJPE_visible"] > 1e-9:  
                out["ORD"] = float((out["MPJPE_occluded"] - out["MPJPE_visible"]) / out["MPJPE_visible"])  
            else:  
                out["ORD"] = float("nan")  
            flips = np.sum(np.abs(np.diff(vis_valid)))  
            out["Visibility_flicker_count"] = int(flips)  
            out["Visibility_flicker_rate"] = float(flips / max(1, len(vis_valid) - 1))  
            K = int(vid_ann.get("window", 5))  
            vis01 = vis_valid.astype(np.int32)  
            segments = []  
            in_occ = False  
            s = 0  
            for i, v in enumerate(vis01):  
                if v == 0 and not in_occ:  
                    in_occ = True  
                    s = i  
                if v == 1 and in_occ:  
                    in_occ = False  
                    segments.append((s, i - 1))  
            if in_occ:  
                segments.append((s, len(vis01) - 1))  
            diffs = []  
            for (s, e) in segments:  
                pre = np.arange(max(0, s - K), s)  
                post = np.arange(e + 1, min(len(vis01), e + 1 + K))  
                if pre.size == 0 or post.size == 0:  
                    continue  
                pre_pa = np.mean(e_pa[pre])  
                post_pa = np.mean(e_pa[post])  
                diffs.append(float(abs(post_pa - pre_pa)))  
            out["PrePostOcc_PA_MPJPE_diff_mean"] = float(np.mean(diffs)) if diffs else float("nan")  
    return out  
  
  
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True,
                    help="输出汇总结果json路径，例如 /xxx/metrics_summary.json")
    ap.add_argument("--pck_delta", type=float, default=0.02,
                    help="PCK阈值delta（单位同world_landmarks，一般近似米）")
    ap.add_argument("--occlusion_json", type=str, default=None,
                    help="可选遮挡标注json（见脚本内说明格式）")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pr_dir = Path(args.pred_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    occ = load_json(Path(args.occlusion_json)) if args.occlusion_json else None

    # 递归遍历 GT 目录，兼容任意 .json 文件（不再限制 *_hand_landmarks.json）
    gt_files = sorted(gt_dir.rglob("*.json"))
    if not gt_files:
        raise RuntimeError(f"No .json files found in {gt_dir}")

    per_video = []

    for gt_path in gt_files:
        # 相对路径，用于在 out_dir 下保持同样的层级结构
        rel = gt_path.relative_to(gt_dir)
        stem = gt_path.stem  # 不再假设有 `_hand_landmarks` 后缀

        # Pred 搜索策略：
        # 1) 同样的相对路径（保持目录结构）
        # 2) 在 pred 根目录下按 stem.json 匹配（退化情况）
        pr_path = pr_dir / rel
        if not pr_path.exists():
            pr_path = pr_dir / f"{stem}.json"
        if not pr_path.exists():
            per_video.append({
                "video_name": stem,
                "gt_path": str(gt_path),
                "pred_path": None,
                "error": "pred json not found",
            })
            print(f"[MISS] {gt_path} -> pred json not found")
            continue

        m = eval_video_metrics(gt_path, pr_path,
                               pck_delta=args.pck_delta,
                               occlusion=occ)
        per_video.append(m)
        print(f"[OK] {gt_path} vs {pr_path} "
              f"| valid_frames={m.get('valid_frames')} "
              f"MPJPE={m.get('MPJPE_mean')}")

    # 先保留你已有的 valid
    valid = [
        m for m in per_video
        if "MPJPE_mean" in m
        and isinstance(m["MPJPE_mean"], (float, int))
        and np.isfinite(m["MPJPE_mean"])
    ]

    # Jitter：只用非 NaN 的
    jitter_vals = [
        m["Jitter_acc2_mean"]
        for m in valid
        if "Jitter_acc2_mean" in m and np.isfinite(m["Jitter_acc2_mean"])
    ]

    def avg_of(key):
        vals = [
            m[key] for m in valid
            if key in m and isinstance(m[key], (float, int)) and np.isfinite(m[key])
        ]
        return float(np.mean(vals)) if vals else None

    summary = {
        "gt_dir": str(gt_dir.resolve()),
        "pred_dir": str(pr_dir.resolve()),
        "num_videos": len(per_video),
        "num_valid": len(valid),

        # 精度相关
        "avg_MPJPE_mean": avg_of("MPJPE_mean"),
        "avg_RA_MPJPE_mean": avg_of("RA_MPJPE_mean"),
        "avg_PA_MPJPE_mean": avg_of("PA_MPJPE_mean"),
        "avg_PCK3D": avg_of("PCK3D"),
        "avg_PCK3D_RA": avg_of("PCK3D_RA"),   # 如果你在 eval 里有算
        "avg_PCK3D_PA": avg_of("PCK3D_PA"),   # 同上

        # 抖动／稳定性
        "avg_Jitter_acc2_mean": float(np.mean(jitter_vals)) if jitter_vals else None,
        "avg_Velocity_var_mean": avg_of("Velocity_var_mean"),
        "avg_Acceleration_var_mean": avg_of("Acceleration_var_mean"),
        "avg_Jerk_var_mean": avg_of("Jerk_var_mean"),

        # 漂移
        "avg_MPJPE_drift_slope": avg_of("MPJPE_drift_slope"),

        # 遮挡相关（如果有 occlusion）
        "avg_MPJPE_visible": avg_of("MPJPE_visible"),
        "avg_MPJPE_occluded": avg_of("MPJPE_occluded"),
        "avg_ORD": avg_of("ORD"),

        # 一些简单统计
        "avg_valid_frames": avg_of("valid_frames"),
        "avg_valid_ratio": avg_of("valid_ratio"),

        "per_video": per_video,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] -> {out_path}")


if __name__ == "__main__":
    main()