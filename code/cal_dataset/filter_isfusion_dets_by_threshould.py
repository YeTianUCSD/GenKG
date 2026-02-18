#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively filter IS-Fusion detections by score threshold in nested JSON.

Goal
----
For every sample anywhere in the JSON tree, if it has:
  det: { boxes_3d: [...], scores_3d: [...], labels_3d: [...] }
keep only entries with scores_3d > threshold (STRICTLY greater).
All other fields (including "gt") remain unchanged.

Works with:
- Nested JSON objects/arrays (e.g., scenes -> frames -> samples -> {...})
- Top-level JSON array or object
- JSON Lines (NDJSON)
- Optional .gz compression

Usage:
  python /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/filter_isfusion_dets_by_threshould.py \
    --input /home/code/3Ddetection/IS-Fusion/GenKG/data/sorted_by_scene_ISFUSIONandGTattr.json \
    --output /home/code/3Ddetection/IS-Fusion/GenKG/data/filter_json/sorted_by_scene_ISFUSIONandGTattr_filter0p1.json \
    --threshold 0.1

    python /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/filter_isfusion_dets.py \
    --input /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/sorted_by_scene_ISFUSIONres_GT_attributions_train.json \
    --output /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/filter_json/sorted_by_scene_ISFUSIONandGT_train_filter0p3.json \
    --threshold 0.3

  # JSONL
  python filter_isfusion_dets_nested.py -i data.jsonl -o data.filt.jsonl -t 0.5
"""

import argparse
import gzip
import json
import os
import sys
from typing import Any, Dict, List, Tuple, Union

JsonObj = Dict[str, Any]

# ----------------------------- I/O helpers -----------------------------

def open_maybe_gzip(path: str, mode: str):
    """Open regular or .gz files transparently (text mode 'rt'/'wt')."""
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def detect_format_from_content(path: str) -> str:
    """
    Detect file format by peeking the beginning: 'json' (single doc) or 'jsonl'.
    """
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

# ----------------------------- Core logic -----------------------------

def safe_min_len(*lists: List[Any]) -> int:
    return min((len(lst) for lst in lists), default=0)

def looks_like_det(d: Any) -> bool:
    """Check if an object looks like a 'det' dict with the three aligned lists."""
    if not isinstance(d, dict):
        return False
    return (
        isinstance(d.get("boxes_3d"), list) and
        isinstance(d.get("scores_3d"), list) and
        isinstance(d.get("labels_3d"), list)
    )

def filter_one_det_in_place(det: Dict[str, Any], threshold: float) -> Tuple[int, int]:
    """
    Filter a single 'det' dict in place. Keep entries with score > threshold.
    Returns (n_before, n_after) for statistics.
    """
    boxes = det.get("boxes_3d", [])
    scores = det.get("scores_3d", [])
    labels = det.get("labels_3d", [])

    if not (isinstance(boxes, list) and isinstance(scores, list) and isinstance(labels, list)):
        return 0, 0

    n_before = min(len(boxes), len(scores), len(labels))

    if not (len(boxes) == len(scores) == len(labels)):
        # Align to the shortest; keep consistent triplets
        print(
            f"[WARN] Length mismatch in 'det': boxes={len(boxes)} "
            f"scores={len(scores)} labels={len(labels)}; using first {n_before}.",
            file=sys.stderr
        )
        boxes = boxes[:n_before]
        scores = scores[:n_before]
        labels = labels[:n_before]

    # STRICT greater-than as required
    mask = [(s is not None) and (float(s) >= threshold) for s in scores]

    filtered_boxes  = [b for b, m in zip(boxes,  mask) if m]
    filtered_scores = [s for s, m in zip(scores, mask) if m]
    filtered_labels = [l for l, m in zip(labels, mask) if m]

    # Write back only the three lists; keep all other keys in det untouched
    det["boxes_3d"]  = filtered_boxes
    det["scores_3d"] = filtered_scores
    det["labels_3d"] = filtered_labels

    n_after = len(filtered_scores)
    return n_before, n_after

def recursive_filter_in_place(obj: Any, threshold: float, stats: Dict[str, int]):
    """
    Recursively traverse the JSON tree.
    Whenever a dict contains a 'det' that looks valid, filter it in place.
    'stats' collects simple counts for a sanity summary.
    """
    if isinstance(obj, dict):
        if "det" in obj and looks_like_det(obj["det"]):
            before, after = filter_one_det_in_place(obj["det"], threshold)
            if before > 0:
                stats["samples_touched"] += 1
                stats["total_det_before"] += before
                stats["total_det_after"]  += after
        # Recurse into all child values (including gt or other nested nodes)
        for v in obj.values():
            recursive_filter_in_place(v, threshold, stats)

    elif isinstance(obj, list):
        for it in obj:
            recursive_filter_in_place(it, threshold, stats)
    # other types -> do nothing

# ----------------------------- JSONL/JSON runners -----------------------------

def process_json_document(data: Any, threshold: float) -> Tuple[Any, Dict[str, int]]:
    stats = {"samples_touched": 0, "total_det_before": 0, "total_det_after": 0}
    recursive_filter_in_place(data, threshold, stats)
    return data, stats

def process_jsonl_stream(in_path: str, out_path: str, threshold: float) -> Dict[str, int]:
    agg = {"samples_touched": 0, "total_det_before": 0, "total_det_after": 0}
    with open_maybe_gzip(in_path, "rt") as fin, open_maybe_gzip(out_path, "wt") as fout:
        for line_no, line in enumerate(fin, start=1):
            s = line.strip()
            if not s:
                fout.write(line)
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping line {line_no}: {e}", file=sys.stderr)
                continue
            obj, stats = process_json_document(obj, threshold)
            for k in agg:
                agg[k] += stats[k]
            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write("\n")
    return agg

def default_output_path(in_path: str, fmt: str) -> str:
    base = in_path
    gz = ""
    if base.endswith(".gz"):
        base = base[:-3]
        gz = ".gz"
    root, ext = os.path.splitext(base)
    if fmt == "jsonl" and ext.lower() not in (".jsonl", ".ndjson"):
        ext = ".jsonl"
    elif fmt == "json" and ext.lower() not in (".json", ".geojson"):
        ext = ".json"
    return f"{root}.filtered{ext}{gz}"

# ---------------------------------- CLI ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Recursively filter IS-Fusion detections by score threshold.")
    ap.add_argument("--input", "-i", required=True, help="Path to input JSON or JSONL/NDJSON (optionally .gz).")
    ap.add_argument("--output", "-o", help="Path to output. If omitted, '<input>.filtered.*' is used.")
    ap.add_argument("--threshold", "-t", type=float, default=0.5, help="Keep detections with score > threshold (strict).")
    ap.add_argument("--indent", type=int, default=None, help="Pretty-print JSON with indent (JSON mode only).")
    ap.add_argument("--format", choices=["auto", "json", "jsonl"], default="auto",
                    help="Force input format. 'auto' tries to detect.")
    args = ap.parse_args()

    fmt = args.format
    if fmt == "auto":
        fmt = detect_format_from_content(args.input)
    if fmt not in ("json", "jsonl"):
        print(f"[ERROR] Unknown format: {fmt}", file=sys.stderr)
        sys.exit(2)

    out_path = args.output or default_output_path(args.input, fmt)

    if fmt == "jsonl":
        stats = process_jsonl_stream(args.input, out_path, args.threshold)
    else:
        with open_maybe_gzip(args.input, "rt") as f:
            data = json.load(f)
        data, stats = process_json_document(data, args.threshold)
        with open_maybe_gzip(out_path, "wt") as f:
            json.dump(data, f, ensure_ascii=False, indent=args.indent)
            if args.indent is None:
                f.write("\n")

    removed = stats["total_det_before"] - stats["total_det_after"]
    print(f"[OK] Wrote filtered data to: {out_path}")
    print(f"[SUMMARY] samples_touched={stats['samples_touched']}  "
          f"det_before={stats['total_det_before']}  det_after={stats['total_det_after']}  "
          f"removed={removed}")

if __name__ == "__main__":
    main()
