#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal "Detection Ceiling Coverage + Detection-style P/R/F1" for LiDAR detections.

Given center-frame GT within a scene:
- Take a temporal window of radius R (2R+1 frames).
- Use the UNION of LiDAR detections within that window.
- A GT is "covered" if there exists a same-class detection within XY distance <= threshold.

Outputs:
1) Upper-bound coverage (ceiling recall): covered_gt / total_gt, plus per-class coverage.
2) Detection-style P/R/F1: greedy one-to-one matching under threshold, per-class + MICRO/MACRO.

Coordinate frames:
- GT boxes_3d centers are assumed in LiDAR(local) frame.
- LiDAR detection translation can be in {global, lidar, ego} via --det_frame.
  Default: global (det global -> lidar using GT poses).

IMPORTANT:
- This version fixes the rotation/transpose bug by using explicit column-vector formulas.

Usage:
python -u /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/check_detection_max_lidar.py \
  --lidar_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/lidar/nuscenes_val_focalformer.json \
  --gt_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --match_thr 0.5 1 2 3 5 8 10 \
  --frame_radius 0 \
  --max_per_frame 500 \
  --det_frame global \
  --ignore_classes -1 \
  --debug_k 3
"""

import argparse
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# -------------------- GT traversal --------------------

def iter_gt_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively traverse nested GT JSON and yield sample dicts that contain:
    - sample_token (str)
    - scene_token (str)
    - timestamp (int-like)
    - gt: {boxes_3d, labels_3d}
    - lidar2ego: {translation, rotation}
    - ego2global or ego_pose: {translation, rotation}
    """
    if isinstance(obj, dict):
        token = obj.get("sample_token", None)
        scene = obj.get("scene_token", None)
        gt = obj.get("gt", None)
        ts = obj.get("timestamp", obj.get("timestamp_us", None))

        lidar2ego = obj.get("lidar2ego", None)
        ego2global = obj.get("ego2global", obj.get("ego_pose", None))

        ok = (
            isinstance(token, str) and token
            and isinstance(scene, str) and scene
            and isinstance(gt, dict) and isinstance(gt.get("boxes_3d"), list)
            and ts is not None
            and isinstance(lidar2ego, dict) and isinstance(lidar2ego.get("translation"), list) and isinstance(lidar2ego.get("rotation"), list)
            and isinstance(ego2global, dict) and isinstance(ego2global.get("translation"), list) and isinstance(ego2global.get("rotation"), list)
            and len(lidar2ego["translation"]) >= 3 and len(lidar2ego["rotation"]) >= 4
            and len(ego2global["translation"]) >= 3 and len(ego2global["rotation"]) >= 4
        )
        if ok:
            yield obj

        for v in obj.values():
            yield from iter_gt_samples(v)

    elif isinstance(obj, list):
        for it in obj:
            yield from iter_gt_samples(it)


def safe_stack_boxes(xs: List[List[float]], exp_dim: int = 9) -> np.ndarray:
    """Convert boxes_3d into [N,exp_dim] float array safely."""
    arr = np.array(xs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, exp_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, exp_dim)
    return arr


# -------------------- Quaternion / rotation --------------------

def quat_to_rot_wxyz(q: List[float]) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q[:4]]
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


# -------------------- Frame conversions (explicit column-vector formulas) --------------------

def global_to_lidar(pts_global: np.ndarray,
                    t_e2g: np.ndarray, R_e2g: np.ndarray,
                    t_l2e: np.ndarray, R_l2e: np.ndarray) -> np.ndarray:
    """
    Column-vector convention:
      p_global = R_e2g * p_ego + t_e2g
      p_ego    = R_e2g^T * (p_global - t_e2g)
      p_ego    = R_l2e * p_lidar + t_l2e
      p_lidar  = R_l2e^T * (p_ego - t_l2e)
    """
    if pts_global.size == 0:
        return pts_global
    x = (pts_global - t_e2g[None, :])  # [N,3]
    p_ego = (R_e2g.T @ x.T).T
    y = (p_ego - t_l2e[None, :])
    p_lidar = (R_l2e.T @ y.T).T
    return p_lidar


def ego_to_lidar(pts_ego: np.ndarray,
                 t_l2e: np.ndarray, R_l2e: np.ndarray) -> np.ndarray:
    """p_lidar = R_l2e^T * (p_ego - t_l2e)."""
    if pts_ego.size == 0:
        return pts_ego
    y = (pts_ego - t_l2e[None, :])
    p_lidar = (R_l2e.T @ y.T).T
    return p_lidar


# -------------------- Detection parsing --------------------

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
    """Load class name -> id map, or use default."""
    if not path:
        return dict(DEFAULT_CLASS_MAP)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"--class_map must be a JSON dict, got: {type(obj)}")
    return {str(k): int(v) for k, v in obj.items()}


def load_det_json(path: str) -> List[Dict[str, Any]]:
    """Load detection JSON top-level list."""
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, list):
        raise TypeError(f"Expected detection JSON top-level to be a list, got: {type(root)}")
    return root


# -------------------- Greedy matching count for multiple thresholds --------------------

def greedy_tp_counts(dist: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """
    Compute greedy one-to-one TP counts for each threshold (ascending).
    dist: [G,D] distances (same class)
    returns: [T] tp counts
    """
    T = len(thresholds)
    out = np.zeros((T,), dtype=np.int64)
    G, D = dist.shape
    if G == 0 or D == 0:
        return out

    max_thr = thresholds[-1]
    gi, dj = np.where(dist <= max_thr)
    if gi.size == 0:
        return out

    dvals = dist[gi, dj]
    order = np.argsort(dvals)
    gi = gi[order]; dj = dj[order]; dvals = dvals[order]

    used_g = np.zeros((G,), dtype=bool)
    used_d = np.zeros((D,), dtype=bool)

    k = 0
    matched = 0
    for t_i, thr in enumerate(thresholds):
        while k < dvals.size and dvals[k] <= thr:
            g = int(gi[k]); d = int(dj[k])
            if (not used_g[g]) and (not used_d[d]):
                used_g[g] = True
                used_d[d] = True
                matched += 1
            k += 1
        out[t_i] = matched
    return out


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar_json", required=True, help="LiDAR detection JSON (top-level list).")
    ap.add_argument("--gt_json", required=True, help="GT JSON (sorted_by_scene_*).")
    ap.add_argument("--match_thr", type=float, nargs="+", default=[2.0],
                    help="XY distance thresholds in meters (multiple allowed).")
    ap.add_argument("--frame_radius", type=int, default=0,
                    help="Temporal window radius R (use frames in [idx-R, ..., idx+R]).")
    ap.add_argument("--max_per_frame", type=int, default=None,
                    help="Top-K detections per frame by score; None means no truncation.")
    ap.add_argument("--score_thr", type=float, default=None,
                    help="Optional score threshold (keep det if score>=score_thr).")
    ap.add_argument("--ignore_classes", type=str, default="-1",
                    help="Comma-separated class ids to ignore (default: -1).")
    ap.add_argument("--class_map", type=str, default=None,
                    help="Optional JSON class map {\"car\":0,...}.")
    ap.add_argument("--det_frame", choices=["global", "lidar", "ego"], default="global",
                    help="Coordinate frame of det.translation. Default: global.")
    ap.add_argument("--debug_k", type=int, default=0,
                    help="Print debug for first K tokens (det center after conversion vs GT center).")
    args = ap.parse_args()

    thresholds = sorted(set(float(x) for x in args.match_thr))
    R = int(args.frame_radius)
    max_k = args.max_per_frame if (args.max_per_frame is not None and args.max_per_frame > 0) else None

    ignore = set(int(x.strip()) for x in args.ignore_classes.split(",") if x.strip())
    class_map = load_class_map(args.class_map)

    print(f"[Info] Loading GT JSON: {args.gt_json}")
    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_root = json.load(f)
    gt_samples = list(iter_gt_samples(gt_root))
    print(f"[Info] GT samples found: {len(gt_samples)}")

    # token -> frame info (GT in lidar)
    token_to_frame: Dict[str, Dict[str, Any]] = {}
    for s in gt_samples:
        token = s["sample_token"]
        scene = s["scene_token"]
        ts = int(s.get("timestamp", s.get("timestamp_us", 0)))

        gt = s.get("gt", {}) or {}
        gt_boxes = safe_stack_boxes(gt.get("boxes_3d", []) or [], exp_dim=9)
        gt_labels = np.array(gt.get("labels_3d", []) or gt.get("labels", []) or [], dtype=np.int64)

        if gt_boxes.shape[0] > 0:
            n = min(gt_boxes.shape[0], gt_labels.shape[0])
            gt_boxes = gt_boxes[:n]
            gt_labels = gt_labels[:n]
        else:
            gt_boxes = np.zeros((0, 9), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)

        if ignore and gt_labels.size > 0:
            m = np.array([int(c) not in ignore for c in gt_labels], dtype=bool)
            gt_boxes = gt_boxes[m]
            gt_labels = gt_labels[m]

        lidar2ego = s["lidar2ego"]
        ego2global = s.get("ego2global", s.get("ego_pose"))

        t_l2e = np.array(lidar2ego["translation"][:3], dtype=np.float64)
        R_l2e = quat_to_rot_wxyz(lidar2ego["rotation"])

        t_e2g = np.array(ego2global["translation"][:3], dtype=np.float64)
        R_e2g = quat_to_rot_wxyz(ego2global["rotation"])

        token_to_frame[token] = {
            "scene_token": scene,
            "timestamp": ts,
            "gt_xy": gt_boxes[:, :2].astype(np.float32),
            "gt_labels": gt_labels.astype(np.int64),
            "t_l2e": t_l2e, "R_l2e": R_l2e,
            "t_e2g": t_e2g, "R_e2g": R_e2g,
        }

    print(f"[Info] Loading LiDAR detections: {args.lidar_json}")
    det_recs = load_det_json(args.lidar_json)
    print(f"[Info] Detection records: {len(det_recs)}")

    # aggregate dets per token
    tmp_xyz: Dict[str, List[List[float]]] = {}
    tmp_lab: Dict[str, List[int]] = {}
    tmp_scr: Dict[str, List[float]] = {}

    for rec in det_recs:
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
            score = float(score) if isinstance(score, (int, float)) else 0.0

            name = str(d.get("detection_name", "unknown"))
            cls = int(class_map.get(name, -1))
            if cls in ignore:
                continue

            tmp_xyz.setdefault(token, []).append([float(trans[0]), float(trans[1]), float(trans[2])])
            tmp_lab.setdefault(token, []).append(cls)
            tmp_scr.setdefault(token, []).append(score)

    # token -> det in lidar frame (xy + labels)
    token_to_det_xy: Dict[str, np.ndarray] = {}
    token_to_det_labels: Dict[str, np.ndarray] = {}

    debug_printed = 0

    for token, xyz_list in tmp_xyz.items():
        if token not in token_to_frame:
            continue

        frame = token_to_frame[token]
        xyz = np.array(xyz_list, dtype=np.float64)  # [N,3]
        lab = np.array(tmp_lab[token], dtype=np.int64)
        scr = np.array(tmp_scr[token], dtype=np.float32)

        if max_k is not None and xyz.shape[0] > max_k:
            idx = np.argsort(-scr)[:max_k]
            xyz = xyz[idx]; lab = lab[idx]; scr = scr[idx]

        # convert det.translation -> lidar
        if args.det_frame == "lidar":
            xyz_l = xyz
        elif args.det_frame == "ego":
            xyz_l = ego_to_lidar(xyz, frame["t_l2e"], frame["R_l2e"])
        else:  # global
            xyz_l = global_to_lidar(xyz, frame["t_e2g"], frame["R_e2g"], frame["t_l2e"], frame["R_l2e"])

        det_xy = xyz_l[:, :2].astype(np.float32) if xyz_l.size > 0 else np.zeros((0, 2), dtype=np.float32)
        token_to_det_xy[token] = det_xy
        token_to_det_labels[token] = lab

        if args.debug_k > 0 and debug_printed < args.debug_k:
            gt_xy = frame["gt_xy"]
            gt0 = gt_xy[0].tolist() if gt_xy.shape[0] > 0 else None
            det0 = det_xy[0].tolist() if det_xy.shape[0] > 0 else None
            det_raw0 = xyz_list[0][:2] if len(xyz_list) > 0 else None
            print(f"[DEBUG] token={token} det_frame={args.det_frame} det_raw_xy={det_raw0} det_lidar_xy={det0} gt0_xy={gt0}")
            debug_printed += 1

    # build scenes from GT frames (det may be empty)
    scenes: Dict[str, List[Dict[str, Any]]] = {}
    for token, info in token_to_frame.items():
        sc = info["scene_token"]
        scenes.setdefault(sc, []).append({
            "token": token,
            "timestamp": info["timestamp"],
            "gt_xy": info["gt_xy"],
            "gt_labels": info["gt_labels"],
            "det_xy": token_to_det_xy.get(token, np.zeros((0, 2), dtype=np.float32)),
            "det_labels": token_to_det_labels.get(token, np.zeros((0,), dtype=np.int64)),
        })
    for sc in scenes:
        scenes[sc].sort(key=lambda x: x["timestamp"])
    print(f"[Info] Scenes loaded: {len(scenes)}")

    # stats
    total_gt = 0
    cls_total: Dict[int, int] = {}
    covered_gt: Dict[float, int] = {thr: 0 for thr in thresholds}
    cls_covered: Dict[float, Dict[int, int]] = {thr: {} for thr in thresholds}

    prf: Dict[float, Dict[str, Any]] = {}
    for thr in thresholds:
        prf[thr] = {"tp": 0, "fp": 0, "fn": 0, "cls_tp": {}, "cls_fp": {}, "cls_fn": {}}

    all_dmin = []

    for sc, frames in scenes.items():
        n = len(frames)
        for i in range(n):
            center = frames[i]
            gt_xy = center["gt_xy"]
            gt_lab = center["gt_labels"]
            if gt_xy.shape[0] == 0:
                continue

            G = int(gt_xy.shape[0])
            total_gt += G
            for c in gt_lab.tolist():
                cls_total[int(c)] = cls_total.get(int(c), 0) + 1

            l = max(0, i - R); r = min(n - 1, i + R)

            det_xy_list = []
            det_lab_list = []
            for j in range(l, r + 1):
                dxy = frames[j]["det_xy"]
                dlb = frames[j]["det_labels"]
                if dxy.shape[0] == 0:
                    continue
                det_xy_list.append(dxy)
                det_lab_list.append(dlb)

            if not det_xy_list:
                for thr in thresholds:
                    prf[thr]["fn"] += G
                    for c in gt_lab.tolist():
                        prf[thr]["cls_fn"][int(c)] = prf[thr]["cls_fn"].get(int(c), 0) + 1
                continue

            det_xy = np.concatenate(det_xy_list, axis=0)
            det_lab = np.concatenate(det_lab_list, axis=0)
            D = int(det_xy.shape[0])

            classes = sorted(set(gt_lab.tolist()) | set(det_lab.tolist()))
            for c in classes:
                if c in ignore:
                    continue
                gi = np.where(gt_lab == c)[0]
                dj = np.where(det_lab == c)[0]
                g = int(gi.size); d = int(dj.size)

                if g > 0 and d > 0:
                    gxy = gt_xy[gi]
                    dxy = det_xy[dj]
                    dist = np.linalg.norm(gxy[:, None, :] - dxy[None, :, :], axis=2)  # [g,d]

                    dmin = dist.min(axis=1)
                    all_dmin.extend(dmin.tolist())
                    for thr in thresholds:
                        cov = int(np.sum(dmin <= thr))
                        covered_gt[thr] += cov
                        cls_covered[thr][c] = cls_covered[thr].get(c, 0) + cov

                    tp_arr = greedy_tp_counts(dist, thresholds)
                else:
                    tp_arr = np.zeros((len(thresholds),), dtype=np.int64)

                for t_i, thr in enumerate(thresholds):
                    tp = int(tp_arr[t_i])
                    fp = d - tp
                    fn = g - tp
                    prf[thr]["tp"] += tp
                    prf[thr]["fp"] += fp
                    prf[thr]["fn"] += fn
                    prf[thr]["cls_tp"][c] = prf[thr]["cls_tp"].get(c, 0) + tp
                    prf[thr]["cls_fp"][c] = prf[thr]["cls_fp"].get(c, 0) + fp
                    prf[thr]["cls_fn"][c] = prf[thr]["cls_fn"].get(c, 0) + fn

    # report
    print("========== Temporal Detection Ceiling (LiDAR) ==========")
    print(f"Total GT boxes (after ignore):        {total_gt}")
    print(f"Ignore classes:                       {sorted(ignore) if ignore else 'None'}")
    print(f"Match thresholds (xy distance, m):    {thresholds}")
    print(f"Frame radius:                         {R} (window = {2*R+1} frames)")
    print(f"Max det per frame (Top-K by score):   {max_k if max_k is not None else 'no limit'}")
    print(f"Score threshold (optional):           {args.score_thr if args.score_thr is not None else 'None'}")
    print(f"Det translation frame:                {args.det_frame} -> lidar (GT is lidar)")

    for thr in thresholds:
        cov = safe_div(float(covered_gt[thr]), float(total_gt))
        print(f"\n--- Threshold = {thr:.3f} m ---")
        print(f"Covered GT boxes (temporal):          {covered_gt[thr]}")
        print(f"Overall coverage (temporal):          {cov:.4f}  ({cov*100:.2f}%)")
        print("Per-class coverage (class_id: covered / total = rate):")
        for c in sorted(cls_total.keys()):
            tot = cls_total[c]
            covc = cls_covered[thr].get(c, 0)
            rate = safe_div(float(covc), float(tot))
            print(f"  {c:3d}: {covc:6d} / {tot:6d} = {rate:.4f}")

    if len(all_dmin) > 0:
        d = np.array(all_dmin, dtype=np.float32)
        print("\n========== Distance Stats (min same-class distance in window) ==========")
        print(f"Count:    {d.shape[0]}")
        print(f"Mean:     {float(d.mean()):.3f} m")
        print(f"Median:   {float(np.median(d)):.3f} m")
        for p in [50, 75, 90, 95, 99]:
            print(f"P{p:02d}:     {float(np.percentile(d, p)):.3f} m")

    print("\n\n========== Temporal-window Detection Metrics (P/R/F1) ==========")
    all_classes = sorted(cls_total.keys())
    for thr in thresholds:
        st = prf[thr]
        tp = float(st["tp"]); fp = float(st["fp"]); fn = float(st["fn"])
        P = safe_div(tp, tp + fp)
        Rr = safe_div(tp, tp + fn)
        F1 = safe_div(2 * P * Rr, P + Rr) if (P + Rr) > 0 else 0.0

        print("\n" + "=" * 70)
        print(f"Distance threshold = {thr:.3f} meters")
        print("-" * 70)
        print("class |       TP       FP       FN ||   Prec    Rec     F1")
        print("-" * 70)

        macro_P, macro_R, macro_F1 = [], [], []
        for c in all_classes:
            tp_c = float(st["cls_tp"].get(c, 0))
            fp_c = float(st["cls_fp"].get(c, 0))
            fn_c = float(st["cls_fn"].get(c, 0))
            p_c = safe_div(tp_c, tp_c + fp_c)
            r_c = safe_div(tp_c, tp_c + fn_c)
            f1_c = safe_div(2 * p_c * r_c, p_c + r_c) if (p_c + r_c) > 0 else 0.0
            print(f"{c:4d} | {int(tp_c):7d} {int(fp_c):7d} {int(fn_c):7d} || {p_c:6.3f} {r_c:6.3f} {f1_c:6.3f}")
            macro_P.append(p_c); macro_R.append(r_c); macro_F1.append(f1_c)

        print("-" * 70)
        print(f" MICRO| {int(tp):7d} {int(fp):7d} {int(fn):7d} || {P:6.3f} {Rr:6.3f} {F1:6.3f}")
        mP = float(np.mean(macro_P)) if macro_P else 0.0
        mR = float(np.mean(macro_R)) if macro_R else 0.0
        mF1 = float(np.mean(macro_F1)) if macro_F1 else 0.0
        print(f" MACRO|    -       -       -   || {mP:6.3f} {mR:6.3f} {mF1:6.3f}")


if __name__ == "__main__":
    main()