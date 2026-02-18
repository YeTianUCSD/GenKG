#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate PR/F1 for 3D detection with class-aware center-distance matching,
recursively scanning nested JSON (e.g., root.scenes[*].samples[*]).

Key features:
- Recursively traverse the JSON tree; any dict that has BOTH "det" and "gt"
  (each with "boxes_3d") is treated as one sample.
- Per-class greedy one-to-one matching under center-distance thresholds
  (2D by default, 3D optional).
- Supports JSON array, JSON object, and JSONL/NDJSON; .gz supported.
- Prints per-threshold per-class (MICRO/MACRO) metrics and a sanity summary.

Usage:
  python /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/cal_det_acc.py \
      --input /home/code/3Ddetection/IS-Fusion/GenKG/code/model/out_ourKG/pred_full_outV5_search_res_V1_r3_mt1_wd0p001.json\
      --thresholds 0.5,1,2,3,4,5,6,7,8,9,10 

    python /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/cal_det_acc.py \
      --input /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/sweep_full/best_det_refined.json\
      --thresholds 0.5,1,2,3,4,5,6,7,8,9,10


  # Use 3D center (x,y,z)
  python eval_prf1_center_dist_nested.py -i data.json -t 2.0 --use-3d-center
"""

import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

# ----------------------------- IO helpers -----------------------------

def open_maybe_gzip(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def detect_format_from_content(path: str) -> str:
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

# -------------------------- Matching utilities ------------------------

def to_int_list(xs: List[Any]) -> List[int]:
    out = []
    for v in xs or []:
        try:
            out.append(int(v))
        except Exception:
            out.append(int(abs(hash(str(v))) % (2**31)))
    return out

def extract_centers(boxes: List[List[float]], use_3d: bool) -> List[Tuple[float, float, float]]:
    centers = []
    for b in boxes or []:
        if not isinstance(b, (list, tuple)) or len(b) < 3:
            continue
        x, y, z = float(b[0]), float(b[1]), float(b[2])
        centers.append((x, y, z if use_3d else 0.0))
    return centers

def pairwise_dists(a: List[Tuple[float, float, float]],
                   b: List[Tuple[float, float, float]]) -> List[Tuple[float, int, int]]:
    pairs = []
    for i, (ax, ay, az) in enumerate(a):
        for j, (bx, by, bz) in enumerate(b):
            dx, dy, dz = ax - bx, ay - by, az - bz
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            pairs.append((d, i, j))
    return pairs

def greedy_match_within_threshold(a_centers, b_centers, threshold: float) -> List[Tuple[int, int, float]]:
    pairs = pairwise_dists(a_centers, b_centers)
    pairs.sort(key=lambda x: x[0])  # ascending distance
    used_a, used_b, matches = set(), set(), []
    for d, i, j in pairs:
        if d > threshold:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        matches.append((i, j, d))
    return matches

# --------------------------- Metric helpers ---------------------------

def safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0

def metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * tp, 2 * tp + fp + fn)
    macc      = safe_div(tp, tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "matching_accuracy": macc}

# --------------------------- Sample traversal -------------------------

def is_sample_dict(d: Dict[str, Any]) -> bool:
    """A 'sample' must have both det and gt dicts each with boxes_3d."""
    if not isinstance(d, dict):
        return False
    det = d.get("det")
    gt  = d.get("gt")
    if not (isinstance(det, dict) and isinstance(gt, dict)):
        return False
    return isinstance(det.get("boxes_3d"), list) and isinstance(gt.get("boxes_3d"), list)

def iter_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively yield dicts that look like detection samples anywhere in the tree.
    """
    if isinstance(obj, dict):
        if is_sample_dict(obj):
            yield obj
        # Recurse into all values
        for v in obj.values():
            yield from iter_samples(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_samples(it)
    # other types -> ignore

# ------------------------------ Main eval -----------------------------

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

        # For sanity summary
        total_det_boxes += len(det_centers)
        total_gt_boxes  += len(gt_centers)

        # Build indices by class (only indices that have a center)
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

def load_root(path: str, fmt: str):
    if fmt == "jsonl":
        # For JSONL, build a pseudo-root list of objects
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

def parse_thresholds(thr_str: str) -> List[float]:
    out = []
    for tok in thr_str.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out

def main():
    ap = argparse.ArgumentParser(description="Evaluate PR/F1 with recursive JSON traversal and center-distance matching.")
    ap.add_argument("--input", "-i", required=True, help="Path to JSON / JSONL (optionally .gz).")
    ap.add_argument("--format", choices=["auto", "json", "jsonl"], default="auto", help="Force input format if needed.")
    ap.add_argument("--thresholds", "-t", default="2.0", help="Comma-separated distance thresholds in meters.")
    ap.add_argument("--use-3d-center", action="store_true", help="Use 3D center distance (x,y,z). Default: 2D (x,y).")
    args = ap.parse_args()

    fmt = args.format if args.format != "auto" else detect_format_from_content(args.input)
    thresholds = parse_thresholds(args.thresholds)
    if not thresholds:
        print("[ERROR] No valid thresholds provided.", file=sys.stderr); sys.exit(2)

    root = load_root(args.input, fmt)
    stats, classes_seen, n_samples, n_det, n_gt = evaluate_dataset(root, thresholds, args.use_3d_center)
    print_report(stats, classes_seen, n_samples, n_det, n_gt)

if __name__ == "__main__":
    main()
