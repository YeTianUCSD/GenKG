#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/filterbynumber/filter_isfusion_by_thr.py \
  --in_json /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/filterbynumber/isfusion_gt_leq10.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/filterbynumber/isfusion_det_thr0p1.json \
  --thr 0.1 \
  --print_stats
'''
import json
import argparse
from copy import deepcopy

def filter_det_by_score(sample: dict, thr: float):
    """
    Filter sample['det'] by score threshold, keep GT unchanged.
    Returns (kept_count, total_count).
    """
    det = sample.get("det", None)
    if not isinstance(det, dict):
        return 0, 0

    boxes = det.get("boxes_3d", [])
    scores = det.get("scores_3d", [])
    labels = det.get("labels_3d", [])

    # If any field missing or not list, treat as empty
    boxes = boxes if isinstance(boxes, list) else []
    scores = scores if isinstance(scores, list) else []
    labels = labels if isinstance(labels, list) else []

    total = len(scores)

    # If scores are missing, we can't threshold reliably -> keep original det as-is.
    # You can change this behavior to "drop all" if you prefer.
    if total == 0:
        return 0, 0

    # Align lengths conservatively to avoid index errors
    n = min(len(boxes), len(scores), len(labels)) if len(labels) > 0 else min(len(boxes), len(scores))
    if n == 0:
        # Nothing aligned to filter
        det["boxes_3d"] = []
        det["scores_3d"] = []
        if "labels_3d" in det:
            det["labels_3d"] = []
        return 0, total

    keep_idx = [i for i in range(n) if float(scores[i]) >= thr]

    det["boxes_3d"] = [boxes[i] for i in keep_idx]
    det["scores_3d"] = [scores[i] for i in keep_idx]
    if len(labels) > 0:
        det["labels_3d"] = [labels[i] for i in keep_idx]
    else:
        # If labels missing/empty originally, keep it empty (or you can remove the key)
        det["labels_3d"] = det.get("labels_3d", [])

    return len(keep_idx), total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", type=str, required=True, help="input json")
    parser.add_argument("--out_json", type=str, required=True, help="output json after thresholding det")
    parser.add_argument("--thr", type=float, default=0.5, help="score threshold, keep det with score >= thr")
    parser.add_argument("--print_stats", action="store_true", help="print some summary stats")
    args = parser.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = deepcopy(data)

    kept_total = 0
    det_total = 0
    sample_count = 0
    samples_with_det = 0

    for scene in new_data.get("scenes", []):
        samples = scene.get("samples", [])
        for s in samples:
            sample_count += 1
            kept, total = filter_det_by_score(s, args.thr)
            if total > 0:
                samples_with_det += 1
                kept_total += kept
                det_total += total

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {args.out_json}")
    if args.print_stats:
        avg_before = (det_total / samples_with_det) if samples_with_det > 0 else 0.0
        avg_after = (kept_total / samples_with_det) if samples_with_det > 0 else 0.0
        keep_ratio = (kept_total / det_total) if det_total > 0 else 0.0
        print(f"Samples total: {sample_count}")
        print(f"Samples with det.scores_3d: {samples_with_det}")
        print(f"Det count before: {det_total}, after: {kept_total}")
        print(f"Avg det/sample before: {avg_before:.4f}, after: {avg_after:.4f}")
        print(f"Keep ratio: {keep_ratio:.4f}")

if __name__ == "__main__":
    main()