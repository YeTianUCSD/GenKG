#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Optional


'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/cal_lidar_det_acc.py \
  --lidar /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/output/camera__score0p2.json\
  --gt /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --thresholds 2 \
  --det-frame global 
'''



# ----------------------------- IO helpers -----------------------------

def open_maybe_gzip(path: str, mode: str):
    # Open normal or gz file based on suffix
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def detect_format_from_content(path: str) -> str:
    # Detect JSON vs JSONL roughly by trying to json.load the file head
    with open_maybe_gzip(path, "rt") as f:
        head = f.read(2048)
    for ch in head:
        if not ch.isspace():
            if ch in ("[", "{"):
                try:
                    with open_maybe_gzip(path, "rt") as f2:
                        json.load(f2)
                    return "json"
                except Exception:
                    return "jsonl"
            else:
                return "jsonl"
    return "json"

def load_root(path: str, fmt: str):
    # Load JSON/JSONL (.gz supported)
    if fmt == "jsonl":
        objs = []
        with open_maybe_gzip(path, "rt") as fin:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    objs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] skip line {line_no}: {e}", file=sys.stderr)
        return objs
    else:
        with open_maybe_gzip(path, "rt") as f:
            return json.load(f)

# -------------------------- Matching utilities (same as your code) ------------------------

def to_int_list(xs: List[Any]) -> List[int]:
    # Convert a list of labels to ints (strings will be hashed consistently within a run)
    out = []
    for v in xs or []:
        try:
            out.append(int(v))
        except Exception:
            out.append(int(abs(hash(str(v))) % (2**31)))
    return out

def extract_centers(boxes: List[List[float]], use_3d: bool) -> List[Tuple[float, float, float]]:
    # Extract center (x,y[,z]) from boxes; if use_3d=False, z is ignored (set to 0)
    centers = []
    for b in boxes or []:
        if not isinstance(b, (list, tuple)) or len(b) < 3:
            continue
        x, y, z = float(b[0]), float(b[1]), float(b[2])
        centers.append((x, y, z if use_3d else 0.0))
    return centers

def pairwise_dists(a: List[Tuple[float, float, float]],
                   b: List[Tuple[float, float, float]]) -> List[Tuple[float, int, int]]:
    # Compute all pairwise Euclidean distances
    pairs = []
    for i, (ax, ay, az) in enumerate(a):
        for j, (bx, by, bz) in enumerate(b):
            dx, dy, dz = ax - bx, ay - by, az - bz
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            pairs.append((d, i, j))
    return pairs

def greedy_match_within_threshold(a_centers, b_centers, threshold: float) -> List[Tuple[int, int, float]]:
    # Greedy one-to-one matching by smallest distance under a threshold
    pairs = pairwise_dists(a_centers, b_centers)
    pairs.sort(key=lambda x: x[0])
    used_a, used_b, matches = set(), set(), []
    for d, i, j in pairs:
        if d > threshold:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        matches.append((i, j, d))
    return matches

# --------------------------- Metric helpers (same as your code) ---------------------------

def safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0

def metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * tp, 2 * tp + fp + fn)
    macc      = safe_div(tp, tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "matching_accuracy": macc}

# --------------------------- Sample traversal (same as your code) -------------------------

def is_sample_dict(d: Dict[str, Any]) -> bool:
    # A sample must have both det and gt dicts each with boxes_3d.
    if not isinstance(d, dict):
        return False
    det = d.get("det")
    gt  = d.get("gt")
    if not (isinstance(det, dict) and isinstance(gt, dict)):
        return False
    return isinstance(det.get("boxes_3d"), list) and isinstance(gt.get("boxes_3d"), list)

def iter_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    # Recursively yield dicts that look like detection samples anywhere in the tree.
    if isinstance(obj, dict):
        if is_sample_dict(obj):
            yield obj
        for v in obj.values():
            yield from iter_samples(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_samples(it)

# --------------------------- Quaternion / Transform helpers ------------------------------

def quat_to_rot_wxyz(q: List[float]) -> List[List[float]]:
    # Convert quaternion [w,x,y,z] to 3x3 rotation matrix
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    r00 = 1 - 2*(yy + zz)
    r01 = 2*(xy - wz)
    r02 = 2*(xz + wy)

    r10 = 2*(xy + wz)
    r11 = 1 - 2*(xx + zz)
    r12 = 2*(yz - wx)

    r20 = 2*(xz - wy)
    r21 = 2*(yz + wx)
    r22 = 1 - 2*(xx + yy)

    return [[r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]]

def matT_vec(R: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # Compute R^T * v for 3x3
    x, y, z = v
    return (R[0][0]*x + R[1][0]*y + R[2][0]*z,
            R[0][1]*x + R[1][1]*y + R[2][1]*z,
            R[0][2]*x + R[1][2]*y + R[2][2]*z)

def sub3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def global_to_lidar_center(
    p_global: Tuple[float, float, float],
    ego2global_t: Tuple[float, float, float],
    ego2global_R: List[List[float]],
    lidar2ego_t: Tuple[float, float, float],
    lidar2ego_R: List[List[float]],
) -> Tuple[float, float, float]:
    """
    Convert point from global to lidar coordinates:
      p_global = R_e2g * (R_l2e * p_lidar + t_l2e) + t_e2g
    So:
      p_ego   = R_e2g^T * (p_global - t_e2g)
      p_lidar = R_l2e^T * (p_ego   - t_l2e)
    """
    p_ego = matT_vec(ego2global_R, sub3(p_global, ego2global_t))
    p_lidar = matT_vec(lidar2ego_R, sub3(p_ego, lidar2ego_t))
    return p_lidar

def get_pose_block(sample: Dict[str, Any], key_candidates: List[str]) -> Optional[Dict[str, Any]]:
    # Fetch the first existing pose block among candidates (each should have translation + rotation)
    for k in key_candidates:
        blk = sample.get(k, None)
        if isinstance(blk, dict) and isinstance(blk.get("translation"), list) and isinstance(blk.get("rotation"), list):
            if len(blk["translation"]) >= 3 and len(blk["rotation"]) >= 4:
                return blk
    return None

# ------------------------- Build token -> sample (GT + poses) ----------------------------

def build_gt_sample_map(gt_root: Any) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping: sample_token -> full sample dict (contains gt + poses).
    We recursively scan and keep dicts that have 'sample_token' and a dict 'gt' with 'boxes_3d'.
    """
    token_map: Dict[str, Dict[str, Any]] = {}

    def rec(obj: Any):
        if isinstance(obj, dict):
            token = obj.get("sample_token", None)
            gt = obj.get("gt", None)
            if isinstance(token, str) and isinstance(gt, dict) and isinstance(gt.get("boxes_3d"), list):
                token_map[token] = obj
            for v in obj.values():
                rec(v)
        elif isinstance(obj, list):
            for it in obj:
                rec(it)

    rec(gt_root)
    return token_map

def infer_gt_label_mode(gt_sample_map: Dict[str, Dict[str, Any]]) -> str:
    # Infer whether GT labels are ints or strings
    for s in gt_sample_map.values():
        gt = s.get("gt", {})
        labels = gt.get("labels_3d", None)
        if not isinstance(labels, list) or len(labels) == 0:
            labels = gt.get("labels", None)
        if isinstance(labels, list) and len(labels) > 0:
            if any(isinstance(x, str) for x in labels):
                return "str"
            if any(isinstance(x, (int, float)) for x in labels):
                return "int"
    return "int"

# Default nuScenes 10-class mapping (override-able)
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
    # Load detection_name -> class_id mapping
    if not path:
        return dict(DEFAULT_CLASS_MAP)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"--class-map must be a JSON dict, but got: {type(obj)}")
    return {str(k): int(v) for k, v in obj.items()}

# ------------------------- Build flat samples for evaluation -----------------------------

def build_flat_samples_lidar_with_gt(
    lidar_root: Any,
    gt_sample_map: Dict[str, Dict[str, Any]],
    gt_label_mode: str,
    class_map: Dict[str, int],
    score_thr: Optional[float],
    det_frame: str,
    debug_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build flat list of {sample_token, det:{boxes_3d,labels_3d}, gt:{...}}.
    det_frame:
      - "global": det.translation is in global, convert to lidar using GT poses (recommended for your case)
      - "lidar":  det.translation already in lidar, no conversion
      - "ego":    det.translation in ego, convert to lidar using lidar2ego only (rare)
    """
    if not isinstance(lidar_root, list):
        raise TypeError(f"Expected LiDAR JSON top-level to be a list, but got: {type(lidar_root)}")

    # Aggregate detections per token (sum if duplicated tokens appear)
    token_to_det_boxes: Dict[str, List[List[float]]] = defaultdict(list)
    token_to_det_labels: Dict[str, List[Any]] = defaultdict(list)

    total_raw_det = 0
    total_kept_det = 0
    missing_pose = 0

    debug_printed = 0

    for rec in lidar_root:
        if not isinstance(rec, dict):
            continue
        token = rec.get("sample_token", None)
        if not isinstance(token, str) or not token:
            continue
        dets = rec.get("detections", [])
        if not isinstance(dets, list):
            continue

        gt_sample = gt_sample_map.get(token, None)
        # We still allow accumulating dets even if GT missing; alignment later decides
        ego2global_blk = None
        lidar2ego_blk = None
        if gt_sample is not None:
            ego2global_blk = get_pose_block(gt_sample, ["ego2global", "ego_pose"])
            lidar2ego_blk  = get_pose_block(gt_sample, ["lidar2ego"])

        # Precompute transforms if needed
        if det_frame == "global":
            if ego2global_blk is None or lidar2ego_blk is None:
                missing_pose += 1
                # Without pose, we cannot convert global->lidar reliably; skip these dets
                continue
            ego2global_t = (float(ego2global_blk["translation"][0]),
                            float(ego2global_blk["translation"][1]),
                            float(ego2global_blk["translation"][2]))
            ego2global_R = quat_to_rot_wxyz(ego2global_blk["rotation"])

            lidar2ego_t = (float(lidar2ego_blk["translation"][0]),
                           float(lidar2ego_blk["translation"][1]),
                           float(lidar2ego_blk["translation"][2]))
            lidar2ego_R = quat_to_rot_wxyz(lidar2ego_blk["rotation"])
        elif det_frame == "ego":
            if lidar2ego_blk is None:
                missing_pose += 1
                continue
            lidar2ego_t = (float(lidar2ego_blk["translation"][0]),
                           float(lidar2ego_blk["translation"][1]),
                           float(lidar2ego_blk["translation"][2]))
            lidar2ego_R = quat_to_rot_wxyz(lidar2ego_blk["rotation"])
        else:
            ego2global_t = ego2global_R = lidar2ego_t = lidar2ego_R = None

        for d in dets:
            if not isinstance(d, dict):
                continue
            total_raw_det += 1

            s = d.get("detection_score", None)
            if score_thr is not None and isinstance(s, (int, float)) and float(s) < float(score_thr):
                continue

            trans = d.get("translation", None)
            if not (isinstance(trans, list) and len(trans) >= 3):
                continue

            p = (float(trans[0]), float(trans[1]), float(trans[2]))

            # Convert center into lidar frame if necessary
            if det_frame == "global":
                p = global_to_lidar_center(p, ego2global_t, ego2global_R, lidar2ego_t, lidar2ego_R)
            elif det_frame == "ego":
                # p_ego -> p_lidar = R_l2e^T * (p_ego - t_l2e)
                p = matT_vec(lidar2ego_R, sub3(p, lidar2ego_t))
            else:
                # det_frame == "lidar": keep as is
                pass

            token_to_det_boxes[token].append([p[0], p[1], p[2]])

            name = d.get("detection_name", "unknown")
            if gt_label_mode == "str":
                token_to_det_labels[token].append(str(name))
            else:
                token_to_det_labels[token].append(int(class_map.get(str(name), -1)))

            total_kept_det += 1

        # Optional debug: print one token's coordinate magnitudes
        if debug_k > 0 and gt_sample is not None and debug_printed < debug_k:
            gt_boxes = gt_sample.get("gt", {}).get("boxes_3d", []) or []
            det_first = token_to_det_boxes[token][0] if token_to_det_boxes[token] else None
            gt_first = gt_boxes[0][:3] if (isinstance(gt_boxes, list) and len(gt_boxes) > 0 and isinstance(gt_boxes[0], list) and len(gt_boxes[0]) >= 3) else None
            print(f"[DEBUG] token={token} det_frame={det_frame} det_first(lidar)={det_first} gt_first={gt_first}")
            debug_printed += 1

    # Align tokens with GT and build flat samples
    det_tokens = set(token_to_det_boxes.keys())
    gt_tokens = set(gt_sample_map.keys())
    matched = sorted(det_tokens & gt_tokens)

    flat_samples: List[Dict[str, Any]] = []
    for token in matched:
        gt_sample = gt_sample_map[token]
        gt = gt_sample.get("gt", {}) or {}
        gt_boxes = gt.get("boxes_3d", []) or []
        gt_labels = gt.get("labels_3d", []) or gt.get("labels", []) or []

        sample = {
            "sample_token": token,
            "det": {
                "boxes_3d": token_to_det_boxes[token],
                "labels_3d": token_to_det_labels[token],
            },
            "gt": {
                "boxes_3d": gt_boxes,
                "labels_3d": gt_labels,
            }
        }
        flat_samples.append(sample)

    stats = {
        "det_unique_tokens": len(det_tokens),
        "gt_unique_tokens": len(gt_tokens),
        "matched_tokens": len(matched),
        "missing_in_gt": len(det_tokens - gt_tokens),
        "missing_in_det": len(gt_tokens - det_tokens),
        "total_raw_det": total_raw_det,
        "total_kept_det": total_kept_det,
        "missing_pose": missing_pose,
    }
    return flat_samples, stats

# ------------------------------ Main eval (same printing as your code) -----------------------------

def evaluate_dataset(root_obj: Any, thresholds: List[float], use_3d_center: bool):
    stats: Dict[float, Dict[int, Dict[str, int]]] = {
        thr: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for thr in thresholds
    }
    classes_seen = set()

    n_samples = 0
    total_det_boxes = 0
    total_gt_boxes  = 0

    for sample in iter_samples(root_obj):
        n_samples += 1

        det = sample.get("det", {}) or {}
        gt  = sample.get("gt", {})  or {}

        det_boxes  = det.get("boxes_3d", []) or []
        det_labels = to_int_list(det.get("labels_3d", []) or det.get("labels", []) or [])
        gt_boxes   = gt.get("boxes_3d", []) or []
        gt_labels  = to_int_list(gt.get("labels_3d", []) or gt.get("labels", []) or [])

        det_centers = extract_centers(det_boxes, use_3d_center)
        gt_centers  = extract_centers(gt_boxes,  use_3d_center)

        total_det_boxes += len(det_centers)
        total_gt_boxes  += len(gt_centers)

        by_cls_det: Dict[int, List[int]] = defaultdict(list)
        for i, c in enumerate(det_labels):
            if i < len(det_centers):
                by_cls_det[c].append(i)

        by_cls_gt: Dict[int, List[int]] = defaultdict(list)
        for j, c in enumerate(gt_labels):
            if j < len(gt_centers):
                by_cls_gt[c].append(j)

        classes = set(by_cls_det.keys()) | set(by_cls_gt.keys())
        classes_seen.update(classes)

        for cls in classes:
            det_idx = by_cls_det.get(cls, [])
            gt_idx  = by_cls_gt.get(cls, [])
            det_c = [det_centers[i] for i in det_idx]
            gt_c  = [gt_centers[j]  for j in gt_idx]

            for thr in thresholds:
                matches = greedy_match_within_threshold(det_c, gt_c, thr)
                tp = len(matches)
                fp = max(0, len(det_c) - tp)
                fn = max(0, len(gt_c)  - tp)
                agg = stats[thr][cls]
                agg["tp"] += tp; agg["fp"] += fp; agg["fn"] += fn

    return stats, classes_seen, n_samples, total_det_boxes, total_gt_boxes

def print_report(stats: Dict[float, Dict[int, Dict[str, int]]], classes_seen: set,
                 n_samples: int, total_det_boxes: int, total_gt_boxes: int):
    classes_sorted = sorted(classes_seen)

    for thr in sorted(stats.keys()):
        print("=" * 80)
        print(f"Distance threshold = {thr} meters")
        print("-" * 80)
        print(f"{'class':>8} | {'TP':>6} {'FP':>6} {'FN':>6} || {'Prec':>7} {'Rec':>7} {'F1':>7} {'MatchAcc':>9}")
        print("-" * 80)

        tot_tp = tot_fp = tot_fn = 0
        macro_vals = []

        for cls in classes_sorted:
            c = stats[thr].get(cls)
            if not c:
                continue
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            tot_tp += tp; tot_fp += fp; tot_fn += fn
            m = metrics_from_counts(tp, fp, fn)
            macro_vals.append(m)
            print(f"{cls:>8} | {tp:>6} {fp:>6} {fn:>6} || {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['matching_accuracy']:>9.4f}")

        micro = metrics_from_counts(tot_tp, tot_fp, tot_fn)
        print("-" * 80)
        print(f"{'MICRO':>8} | {tot_tp:>6} {tot_fp:>6} {tot_fn:>6} || {micro['precision']:>7.4f} {micro['recall']:>7.4f} {micro['f1']:>7.4f} {micro['matching_accuracy']:>9.4f}")
        if macro_vals:
            macro = {k: sum(x[k] for x in macro_vals)/len(macro_vals) for k in macro_vals[0].keys()}
            print(f"{'MACRO':>8} | {'-':>6} {'-':>6} {'-':>6} || {macro['precision']:>7.4f} {macro['recall']:>7.4f} {macro['f1']:>7.4f} {macro['matching_accuracy']:>9.4f}")
        print()

    print("=" * 80)
    print(f"Sanity summary: samples={n_samples}, total_det_boxes={total_det_boxes}, total_gt_boxes={total_gt_boxes}")
    if n_samples == 0:
        print("[WARN] No samples found. Check JSON structure and keys 'det'/'gt'.")
    elif total_det_boxes == 0 and total_gt_boxes == 0:
        print("[WARN] Found samples but both det and gt were always empty.")
    elif total_gt_boxes == 0:
        print("[WARN] Found det boxes but ZERO GT boxes. Are you evaluating the right split/file?")
    elif total_det_boxes == 0:
        print("[WARN] Found GT boxes but ZERO detections (maybe filtered out by score?).")

def parse_thresholds(thr_str: str) -> List[float]:
    out = []
    for tok in thr_str.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out

def main():
    ap = argparse.ArgumentParser(description="Evaluate LiDAR detections by aligning GT via sample_token and fixing coordinate frame.")
    ap.add_argument("--lidar", required=True, help="Path to LiDAR detections JSON/JSONL (optionally .gz).")
    ap.add_argument("--lidar-format", choices=["auto", "json", "jsonl"], default="auto", help="Force LiDAR input format.")
    ap.add_argument("--gt", required=True, help="Path to nested GT JSON (sorted_by_scene_*.json).")
    ap.add_argument("--thresholds", "-t", default="2.0", help="Comma-separated distance thresholds in meters.")
    ap.add_argument("--use-3d-center", action="store_true", help="Use 3D center distance (x,y,z). Default: 2D (x,y).")
    ap.add_argument("--score-thr", type=float, default=None, help="Optional additional score threshold applied to LiDAR detections.")
    ap.add_argument("--class-map", default=None, help="Optional JSON file mapping detection_name -> class_id.")
    ap.add_argument("--det-frame", choices=["global", "lidar", "ego"], default="global",
                    help="Coordinate frame of LiDAR detection translation. Default: global (will convert to lidar using GT poses).")
    ap.add_argument("--debug-k", type=int, default=0, help="Print debug for first K matched tokens (centers).")
    args = ap.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    if not thresholds:
        print("[ERROR] No valid thresholds provided.", file=sys.stderr)
        sys.exit(2)

    lidar_fmt = args.lidar_format if args.lidar_format != "auto" else detect_format_from_content(args.lidar)
    lidar_root = load_root(args.lidar, lidar_fmt)

    gt_root = load_root(args.gt, "json")
    gt_sample_map = build_gt_sample_map(gt_root)
    if not gt_sample_map:
        print("[ERROR] Could not find any GT samples with keys {'sample_token','gt','gt.boxes_3d'} in GT JSON.", file=sys.stderr)
        sys.exit(2)

    gt_label_mode = infer_gt_label_mode(gt_sample_map)
    class_map = load_class_map(args.class_map)

    flat_samples, align_stats = build_flat_samples_lidar_with_gt(
        lidar_root=lidar_root,
        gt_sample_map=gt_sample_map,
        gt_label_mode=gt_label_mode,
        class_map=class_map,
        score_thr=args.score_thr,
        det_frame=args.det_frame,
        debug_k=args.debug_k,
    )

    # Alignment summary (English output)
    print("=" * 80)
    print("Alignment summary")
    print("-" * 80)
    print(f"LiDAR unique sample_tokens: {align_stats['det_unique_tokens']}")
    print(f"GT unique sample_tokens:    {align_stats['gt_unique_tokens']}")
    print(f"Matched tokens:             {align_stats['matched_tokens']}")
    print(f"Missing in GT:              {align_stats['missing_in_gt']}")
    print(f"Missing in LiDAR:           {align_stats['missing_in_det']}")
    print(f"LiDAR detections raw/kept:  {align_stats['total_kept_det']} / {align_stats['total_raw_det']}"
          + (f" (score_thr={args.score_thr})" if args.score_thr is not None else ""))
    print(f"Det frame:                  {args.det_frame} -> lidar")
    print(f"GT label mode inferred:     {gt_label_mode}")
    print(f"Tokens skipped (missing pose for conversion): {align_stats['missing_pose']}")
    print("=" * 80)

    stats, classes_seen, n_samples, n_det, n_gt = evaluate_dataset(flat_samples, thresholds, args.use_3d_center)
    print_report(stats, classes_seen, n_samples, n_det, n_gt)

if __name__ == "__main__":
    main()