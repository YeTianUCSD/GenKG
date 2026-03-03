#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal "Detection Ceiling Coverage + Detection-style P/R/F1" for CAMERA detections.

Given GT per current frame, within the same scene:
- Take a temporal window of radius frame_radius (2R+1 frames).
- Use the UNION of camera detections within that window.
- A GT is considered "covered" if there exists a same-class det within XY distance <= match_thr.

1) Coverage ceiling recall:
   For each GT, if the temporal window contains at least one same-class det within threshold,
   then this GT is "covered".

2) Detection-style P/R/F1 (over temporal window union):
   For current-frame GT and window-union det, perform class-aware greedy one-to-one matching:
     TP = #matched pairs
     FP = #det_in_window - TP
     FN = #gt_in_center  - TP
   Report micro and per-class P/R/F1.

Important:
- Camera det translations are assumed GLOBAL by default (det_frame=global).
- GT boxes are typically in LIDAR frame in your dataset JSON; we transform GT centers to GLOBAL
  using lidar2ego + ego2global.

Usage example:

python -u /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/check_detection_max_camera.py \
  --camera_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/camera/nuscenes_val_pre.json \
  --gt_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --match_thr 0.5 1 2 3 5 8 10 \
  --frame_radius 0 \
  --max_per_frame 500 \
  --ignore_classes -1 \
  --det_frame global
"""

import argparse
import json
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


# -------------------- Utilities: traverse GT JSON to get samples --------------------

def iter_gt_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively traverse a nested JSON and yield dicts that look like "GT samples":
    - has 'sample_token' (str)
    - has 'scene_token' (str) and 'timestamp' (int-like)
    - has 'gt' dict with 'boxes_3d' list
    - has pose blocks: lidar2ego and ego2global (or ego_pose)
    """
    if isinstance(obj, dict):
        token = obj.get("sample_token", None)
        scene = obj.get("scene_token", None)
        gt = obj.get("gt", None)
        ts = obj.get("timestamp", obj.get("timestamp_us", None))
        lidar2ego = obj.get("lidar2ego", None)
        ego2global = obj.get("ego2global", obj.get("ego_pose", None))

        if (
            isinstance(token, str) and token
            and isinstance(scene, str) and scene
            and isinstance(gt, dict) and isinstance(gt.get("boxes_3d"), list)
            and ts is not None
            and isinstance(lidar2ego, dict) and isinstance(lidar2ego.get("translation"), list) and isinstance(lidar2ego.get("rotation"), list)
            and isinstance(ego2global, dict) and isinstance(ego2global.get("translation"), list) and isinstance(ego2global.get("rotation"), list)
        ):
            yield obj

        for v in obj.values():
            yield from iter_gt_samples(v)

    elif isinstance(obj, list):
        for it in obj:
            yield from iter_gt_samples(it)


def safe_stack_boxes(xs: List[List[float]], exp_dim: int = 9) -> np.ndarray:
    """Safely convert boxes_3d into an [N, exp_dim] numpy array."""
    arr = np.array(xs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, exp_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, exp_dim)
    return arr


# -------------------- Pose / Transform helpers --------------------

def quat_to_rot(q: List[float]) -> np.ndarray:
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def make_T(translation: List[float], quat_wxyz: List[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform matrix from translation and quaternion."""
    R = quat_to_rot(quat_wxyz)
    t = np.array(translation, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform T to points pts of shape [N,3]."""
    if pts.size == 0:
        return pts
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1), dtype=pts.dtype)])  # [N,4]
    out = (T @ homo.T).T[:, :3]
    return out


# -------------------- Camera detection parsing --------------------

# Default nuScenes 10-class name->id map (override via --class_map if needed)
DEFAULT_CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "trailer": 3,
    "construction_vehicle": 4,
    "pedestrian": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic_cone": 8,
    "barrier": 9,
}

def load_class_map(path: Optional[str]) -> Dict[str, int]:
    """Load class name->id mapping from a JSON file, or use default."""
    if not path:
        return dict(DEFAULT_CLASS_MAP)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"--class_map must be a JSON dict, got: {type(obj)}")
    return {str(k): int(v) for k, v in obj.items()}


def load_camera_dets(camera_json_path: str) -> List[Dict[str, Any]]:
    """Load camera detection JSON (top-level list)."""
    with open(camera_json_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, list):
        raise TypeError(f"Expected camera JSON top-level to be a list, got: {type(root)}")
    return root


# -------------------- Main logic --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_json", required=True, help="Camera detections JSON (top-level list).")
    parser.add_argument("--gt_json", required=True, help="GT+pose JSON (sorted_by_scene_*).")
    parser.add_argument(
        "--match_thr",
        type=float,
        nargs="+",
        default=[2.0],
        help="XY distance thresholds (meters), can pass multiple values.",
    )
    parser.add_argument(
        "--frame_radius",
        type=int,
        default=5,
        help="Temporal window radius R: use frames in [idx-R, ..., idx+R].",
    )
    parser.add_argument(
        "--max_per_frame",
        type=int,
        default=None,
        help="Top-K detections per frame by score; None or <=0 means no truncation.",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=None,
        help="Optional score threshold for camera detections (keep det if score>=score_thr).",
    )
    parser.add_argument(
        "--ignore_classes",
        type=str,
        default="-1",
        help="Comma-separated class ids to ignore (default: -1).",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        default=None,
        help="Optional JSON file: {\"car\":0, ...} for detection_name -> class_id mapping.",
    )
    parser.add_argument(
        "--det_frame",
        choices=["global", "lidar", "ego"],
        default="global",
        help="Coordinate frame of camera det translation. Default: global.",
    )
    args = parser.parse_args()

    max_per_frame = args.max_per_frame
    if max_per_frame is not None and max_per_frame <= 0:
        max_per_frame = None

    ignore = set(int(x.strip()) for x in args.ignore_classes.split(",") if x.strip())
    class_map = load_class_map(args.class_map)
    match_thrs = sorted(set(args.match_thr))
    R = args.frame_radius

    print(f"[Info] Loading GT JSON from: {args.gt_json}")
    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_root = json.load(f)

    gt_samples = list(iter_gt_samples(gt_root))
    print(f"[Info] Found {len(gt_samples)} GT samples.")

    # Build token -> GT frame info (including GT global XY and scene/timestamp)
    token_to_gt_frame: Dict[str, Dict[str, Any]] = {}

    for s in gt_samples:
        token = s["sample_token"]
        scene_token = s["scene_token"]
        ts = int(s.get("timestamp", s.get("timestamp_us", 0)))

        gt = s.get("gt", {}) or {}
        gt_boxes = safe_stack_boxes(gt.get("boxes_3d", []) or [], exp_dim=9)
        gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64)

        # Align GT lengths
        if gt_boxes.shape[0] > 0:
            n_gt = min(gt_boxes.shape[0], gt_labels.shape[0])
            gt_boxes = gt_boxes[:n_gt]
            gt_labels = gt_labels[:n_gt]
        else:
            gt_boxes = np.zeros((0, 9), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)

        # Poses: lidar -> ego -> global
        lidar_info = s.get("lidar2ego")
        ego_global_info = s.get("ego2global", s.get("ego_pose"))

        Tl2e = make_T(lidar_info["translation"], lidar_info["rotation"])
        Te2g = make_T(ego_global_info["translation"], ego_global_info["rotation"])
        T_lidar2global = Te2g @ Tl2e

        gt_xyz_global = transform_points(T_lidar2global, gt_boxes[:, :3]) if gt_boxes.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
        gt_xy_global = gt_xyz_global[:, :2] if gt_xyz_global.size > 0 else np.zeros((0, 2), dtype=np.float32)

        # Ignore classes for GT
        if ignore and gt_labels.size > 0:
            mask_gt = np.array([int(c) not in ignore for c in gt_labels], dtype=bool)
            gt_xy_global = gt_xy_global[mask_gt]
            gt_labels = gt_labels[mask_gt]

        token_to_gt_frame[token] = {
            "scene_token": scene_token,
            "timestamp": ts,
            "gt_xy": gt_xy_global,
            "gt_labels": gt_labels,
            "T_lidar2global": T_lidar2global,
            "T_ego2global": Te2g,
        }

    # Load camera detections, aggregate by sample_token
    print(f"[Info] Loading camera detections from: {args.camera_json}")
    cam_recs = load_camera_dets(args.camera_json)
    print(f"[Info] Camera records: {len(cam_recs)}")

    token_to_det_xy: Dict[str, np.ndarray] = {}
    token_to_det_labels: Dict[str, np.ndarray] = {}

    # Temporary storage for aggregation
    tmp_xy: Dict[str, List[List[float]]] = {}
    tmp_lab: Dict[str, List[int]] = {}
    tmp_score: Dict[str, List[float]] = {}

    for rec in cam_recs:
        if not isinstance(rec, dict):
            continue
        token = rec.get("sample_token", None)
        if not isinstance(token, str) or not token:
            continue
        dets = rec.get("detections", [])
        if not isinstance(dets, list):
            continue

        for d in dets:
            if not isinstance(d, dict):
                continue
            trans = d.get("translation", None)
            if not (isinstance(trans, list) and len(trans) >= 3):
                continue

            score = d.get("detection_score", None)
            if args.score_thr is not None:
                if not isinstance(score, (int, float)):
                    continue
                if float(score) < float(args.score_thr):
                    continue
            if score is None:
                score = 0.0

            name = str(d.get("detection_name", "unknown"))
            cls_id = int(class_map.get(name, -1))
            if ignore and cls_id in ignore:
                continue

            tmp_xy.setdefault(token, []).append([float(trans[0]), float(trans[1]), float(trans[2])])
            tmp_lab.setdefault(token, []).append(cls_id)
            tmp_score.setdefault(token, []).append(float(score))

    # Finalize per-token det arrays with optional frame transform and Top-K truncation
    for token, xyz_list in tmp_xy.items():
        if token not in token_to_gt_frame:
            # No pose / no GT info for this token (likely not in this split)
            continue

        xyz = np.array(xyz_list, dtype=np.float64)  # [N,3]
        labels = np.array(tmp_lab[token], dtype=np.int64)
        scores = np.array(tmp_score[token], dtype=np.float32)

        # Optional Top-K truncation by score
        if max_per_frame is not None and xyz.shape[0] > max_per_frame:
            idxs = np.argsort(-scores)[:max_per_frame]
            xyz = xyz[idxs]
            labels = labels[idxs]
            scores = scores[idxs]

        # Convert det coords to GLOBAL if needed
        if args.det_frame == "global":
            xyz_global = xyz
        elif args.det_frame == "lidar":
            T = token_to_gt_frame[token]["T_lidar2global"]
            xyz_global = transform_points(T, xyz.astype(np.float32)).astype(np.float64)
        else:  # ego
            T = token_to_gt_frame[token]["T_ego2global"]
            xyz_global = transform_points(T, xyz.astype(np.float32)).astype(np.float64)

        det_xy_global = xyz_global[:, :2] if xyz_global.size > 0 else np.zeros((0, 2), dtype=np.float32)

        token_to_det_xy[token] = det_xy_global.astype(np.float32)
        token_to_det_labels[token] = labels

    # Build scenes: each GT frame gets its det (may be empty)
    scenes: Dict[str, List[Dict[str, Any]]] = {}
    for token, info in token_to_gt_frame.items():
        scene_token = info["scene_token"]
        frame = {
            "timestamp": info["timestamp"],
            "gt_xy": info["gt_xy"],
            "gt_labels": info["gt_labels"],
            "det_xy": token_to_det_xy.get(token, np.zeros((0, 2), dtype=np.float32)),
            "det_labels": token_to_det_labels.get(token, np.zeros((0,), dtype=np.int64)),
        }
        scenes.setdefault(scene_token, []).append(frame)

    for sc in scenes:
        scenes[sc].sort(key=lambda x: x["timestamp"])

    print(f"[Info] Scenes loaded: {len(scenes)}")

    # -------------------- Temporal ceiling coverage + PRF statistics --------------------

    total_gt = 0
    cls_total: Dict[int, int] = {}

    covered_gt: Dict[float, int] = {thr: 0 for thr in match_thrs}
    cls_covered: Dict[float, Dict[int, int]] = {thr: {} for thr in match_thrs}

    prf_stats: Dict[float, Dict[str, Any]] = {}
    for thr in match_thrs:
        prf_stats[thr] = {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "cls_tp": {},
            "cls_fp": {},
            "cls_fn": {},
        }

    all_d_min: List[float] = []

    for sc, frames in scenes.items():
        n_frames = len(frames)
        if n_frames == 0:
            continue

        for idx in range(n_frames):
            center = frames[idx]
            gt_xy = center["gt_xy"]
            gt_labels = center["gt_labels"]

            if gt_xy.shape[0] == 0:
                continue

            N_gt = gt_xy.shape[0]
            total_gt += N_gt
            for cls_id in gt_labels.tolist():
                cls_total[int(cls_id)] = cls_total.get(int(cls_id), 0) + 1

            left = max(0, idx - R)
            right = min(n_frames - 1, idx + R)

            det_xy_list = []
            det_labels_list = []
            for j in range(left, right + 1):
                det_xy_j = frames[j]["det_xy"]
                det_lab_j = frames[j]["det_labels"]
                if det_xy_j.shape[0] == 0:
                    continue
                det_xy_list.append(det_xy_j)
                det_labels_list.append(det_lab_j)

            if not det_xy_list:
                for thr in match_thrs:
                    stat = prf_stats[thr]
                    stat["fn"] += float(N_gt)
                    for c in gt_labels.tolist():
                        stat["cls_fn"][int(c)] = stat["cls_fn"].get(int(c), 0) + 1
                continue

            det_xy_win = np.concatenate(det_xy_list, axis=0)
            det_labels_win = np.concatenate(det_labels_list, axis=0)
            N_det = det_labels_win.shape[0]

            diff = gt_xy[:, None, :] - det_xy_win[None, :, :]
            dist = np.linalg.norm(diff, axis=2)  # [N_gt, M]

            # Coverage ceiling
            for i in range(N_gt):
                c = int(gt_labels[i])
                same_cls = np.where(det_labels_win == c)[0]
                if same_cls.size == 0:
                    continue
                d_min = float(dist[i, same_cls].min())
                all_d_min.append(d_min)
                for thr in match_thrs:
                    if d_min <= thr:
                        covered_gt[thr] += 1
                        cov_dict = cls_covered[thr]
                        cov_dict[c] = cov_dict.get(c, 0) + 1

            # Detection-style PRF (greedy one-to-one)
            for thr in match_thrs:
                stat = prf_stats[thr]

                if N_det == 0 and N_gt > 0:
                    stat["fn"] += float(N_gt)
                    for c in gt_labels.tolist():
                        stat["cls_fn"][int(c)] = stat["cls_fn"].get(int(c), 0) + 1
                    continue
                if N_det > 0 and N_gt == 0:
                    stat["fp"] += float(N_det)
                    for c in det_labels_win.tolist():
                        stat["cls_fp"][int(c)] = stat["cls_fp"].get(int(c), 0) + 1
                    continue

                pairs = []
                for gi in range(N_gt):
                    c = int(gt_labels[gi])
                    same_cls = np.where(det_labels_win == c)[0]
                    if same_cls.size == 0:
                        continue
                    dists_g = dist[gi, same_cls]
                    for k, dj in enumerate(same_cls):
                        d_ij = float(dists_g[k])
                        if d_ij <= thr:
                            pairs.append((d_ij, gi, dj))

                pairs.sort(key=lambda x: x[0])

                used_gt = set()
                used_det = set()
                matches = []

                for d_ij, gi, dj in pairs:
                    if gi in used_gt or dj in used_det:
                        continue
                    used_gt.add(gi)
                    used_det.add(dj)
                    matches.append((gi, dj))

                tp = float(len(matches))
                fp = float(N_det - len(matches))
                fn = float(N_gt - len(matches))

                stat["tp"] += tp
                stat["fp"] += fp
                stat["fn"] += fn

                for gi, dj in matches:
                    c = int(gt_labels[gi])
                    stat["cls_tp"][c] = stat["cls_tp"].get(c, 0) + 1

                for dj in range(N_det):
                    if dj in used_det:
                        continue
                    c = int(det_labels_win[dj])
                    stat["cls_fp"][c] = stat["cls_fp"].get(c, 0) + 1

                for gi in range(N_gt):
                    if gi in used_gt:
                        continue
                    c = int(gt_labels[gi])
                    stat["cls_fn"][c] = stat["cls_fn"].get(c, 0) + 1

    # -------------------- Reporting --------------------

    if total_gt == 0:
        print("[Warn] No GT boxes found after filtering. Nothing to compute.")
        return

    print("========== Temporal Detection Ceiling (Camera det union over window) ==========")
    print(f"Total GT boxes (after ignore):        {total_gt}")
    print(f"Ignore classes:                       {sorted(ignore) if ignore else 'None'}")
    print(f"Match thresholds (xy distance, m):    {match_thrs}")
    print(f"Frame radius:                         {R} (window = {2*R+1} frames)")
    print(f"Max det per frame (Top-K by score):   {max_per_frame if max_per_frame is not None else 'no limit'}")
    print(f"Score threshold (optional):           {args.score_thr if args.score_thr is not None else 'None'}")
    print(f"Det translation frame:                {args.det_frame} (GT is transformed to global)")

    for thr in match_thrs:
        cov = covered_gt[thr] / float(total_gt)
        print(f"\n--- Threshold = {thr:.3f} m ---")
        print(f"Covered GT boxes (temporal):          {covered_gt[thr]}")
        print(f"Overall coverage (temporal):          {cov:.4f}  ({cov*100:.2f}%)")
        print("Per-class coverage (class_id: covered / total = rate):")
        for c in sorted(cls_total.keys()):
            tot_c = cls_total[c]
            cov_c = cls_covered[thr].get(c, 0)
            rate_c = cov_c / float(tot_c) if tot_c > 0 else 0.0
            print(f"  {c:3d}: {cov_c:6d} / {tot_c:6d} = {rate_c:.4f}")

    if len(all_d_min) > 0:
        d_arr = np.array(all_d_min, dtype=np.float32)
        print("\n========== Distance Error Stats over temporal window ==========")
        print(f"GT count with same-class det in window: {d_arr.shape[0]}")
        print(f"Mean distance:                          {d_arr.mean():.3f} m")
        print(f"Median distance:                        {np.median(d_arr):.3f} m")
        for p in [50, 75, 90, 95, 99]:
            print(f"{p:2d}th percentile:                      {np.percentile(d_arr, p):.3f} m")

    print("\n\n========== Temporal-window Detection Metrics (P/R/F1 over GT frames) ==========")
    for thr in match_thrs:
        stat = prf_stats[thr]
        tp = stat["tp"]
        fp = stat["fp"]
        fn = stat["fn"]

        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        Rr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = 2 * P * Rr / (P + Rr) if (P + Rr) > 0 else 0.0

        print("\n" + "=" * 70)
        print(f"Distance threshold = {thr:.3f} meters")
        print("-" * 70)
        print("class |       TP       FP       FN ||   Prec    Rec     F1")
        print("-" * 70)

        all_classes = sorted(cls_total.keys())
        for c in all_classes:
            tp_c = float(stat["cls_tp"].get(c, 0))
            fp_c = float(stat["cls_fp"].get(c, 0))
            fn_c = float(stat["cls_fn"].get(c, 0))

            prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0

            print(f"{c:4d} | {int(tp_c):7d} {int(fp_c):7d} {int(fn_c):7d} ||"
                  f" {prec_c:6.3f} {rec_c:6.3f} {f1_c:6.3f}")

        print("-" * 70)
        print(f" MICRO| {int(tp):7d} {int(fp):7d} {int(fn):7d} ||"
              f" {P:6.3f} {Rr:6.3f} {F1:6.3f}")

        macro_P, macro_R, macro_F1 = [], [], []
        for c in all_classes:
            if cls_total.get(c, 0) == 0:
                continue
            tp_c = float(stat["cls_tp"].get(c, 0))
            fp_c = float(stat["cls_fp"].get(c, 0))
            fn_c = float(stat["cls_fn"].get(c, 0))
            p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            macro_P.append(p_c); macro_R.append(r_c); macro_F1.append(f1_c)

        if len(macro_P) > 0:
            mP = float(np.mean(macro_P))
            mR = float(np.mean(macro_R))
            mF1 = float(np.mean(macro_F1))
        else:
            mP = mR = mF1 = 0.0

        print(f" MACRO|    -       -       -   || {mP:6.3f} {mR:6.3f} {mF1:6.3f}")


if __name__ == "__main__":
    main()