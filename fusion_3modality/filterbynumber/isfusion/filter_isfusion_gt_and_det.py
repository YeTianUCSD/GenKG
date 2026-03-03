#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/filterbynumber/isfusion/filter_isfusion_gt_and_det.py \
  --in_json  /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/filterbynumber/isfusion/isfusion_gt_leq30_det_thr0p2.json \
  --max_gt 30 \
  --thr 0.2 \
  --print_stats
"""

import json
import argparse
from copy import deepcopy


def gt_count(sample: dict) -> int:
    gt = sample.get("gt", {})
    boxes = gt.get("boxes_3d", [])
    if not isinstance(boxes, list) or boxes is None:
        return 0
    return len(boxes)


def filter_det_by_score_inplace(sample: dict, thr: float):
    """
    In-place filter sample['det'] by score threshold; GT unchanged.
    Returns (kept_count, total_count, used_n).
      - total_count: len(scores_3d) (before filtering)
      - used_n: actually aligned length used for filtering (min of fields)
    """
    det = sample.get("det", None)
    if not isinstance(det, dict):
        return 0, 0, 0

    boxes = det.get("boxes_3d", [])
    scores = det.get("scores_3d", [])
    labels = det.get("labels_3d", [])

    boxes = boxes if isinstance(boxes, list) else []
    scores = scores if isinstance(scores, list) else []
    labels = labels if isinstance(labels, list) else []

    total = len(scores)
    if total == 0:
        # 没有 scores_3d -> 不做阈值过滤（保持 det 原样）
        return 0, 0, 0

    # 对齐长度，避免不一致导致越界
    if len(labels) > 0:
        n = min(len(boxes), len(scores), len(labels))
    else:
        n = min(len(boxes), len(scores))

    if n == 0:
        det["boxes_3d"] = []
        det["scores_3d"] = []
        if "labels_3d" in det:
            det["labels_3d"] = []
        return 0, total, 0

    keep_idx = [i for i in range(n) if float(scores[i]) >= thr]

    det["boxes_3d"] = [boxes[i] for i in keep_idx]
    det["scores_3d"] = [scores[i] for i in keep_idx]
    if len(labels) > 0:
        det["labels_3d"] = [labels[i] for i in keep_idx]
    else:
        # 若原本 labels_3d 缺失/为空，保持为空（或你也可以选择 del det["labels_3d"]）
        det["labels_3d"] = det.get("labels_3d", [])

    return len(keep_idx), total, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", type=str, required=True, help="input json")
    parser.add_argument("--out_json", type=str, required=True, help="output json")
    parser.add_argument("--max_gt", type=int, default=10, help="keep samples whose GT count <= max_gt")
    parser.add_argument("--thr", type=float, default=0.5, help="keep det with score >= thr")
    parser.add_argument("--print_stats", action="store_true", help="print summary stats for output json")
    args = parser.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {
        "version": data.get("version", ""),
        "scene_count": 0,
        "scenes": []
    }

    # stats (for kept samples only)
    total_samples_kept = 0
    total_gt_kept = 0

    det_total_before = 0     # sum of total scores_3d (before filtering) on kept samples
    det_total_after = 0      # sum of kept det after filtering
    samples_with_det_scores = 0
    mismatch_samples = 0     # samples where boxes/scores/labels length mismatch

    for scene in data.get("scenes", []):
        kept_samples = []

        for s in scene.get("samples", []):
            c = gt_count(s)
            if c > args.max_gt:
                continue

            # keep this sample
            s2 = deepcopy(s)

            # filter det in-place by thr
            kept_det, total_det, used_n = filter_det_by_score_inplace(s2, args.thr)
            if total_det > 0:
                samples_with_det_scores += 1
                det_total_before += total_det
                det_total_after += kept_det

                # 粗略检查长度不一致（只要 used_n < total_det 或 used_n < len(boxes) 等都可能发生）
                det = s2.get("det", {})
                boxes_len = len(det.get("boxes_3d", [])) if isinstance(det.get("boxes_3d", []), list) else 0
                scores_len = len(det.get("scores_3d", [])) if isinstance(det.get("scores_3d", []), list) else 0
                labels_len = len(det.get("labels_3d", [])) if isinstance(det.get("labels_3d", []), list) else 0
                # 过滤后长度应该一致（labels 可能为空/缺失则不强制）
                if labels_len not in (0, boxes_len) or scores_len != boxes_len:
                    mismatch_samples += 1

            kept_samples.append(s2)
            total_samples_kept += 1
            total_gt_kept += c

        if kept_samples:
            new_scene = deepcopy(scene)
            new_scene["samples"] = kept_samples
            new_scene["num_samples"] = len(kept_samples)
            new_data["scenes"].append(new_scene)

    new_data["scene_count"] = len(new_data["scenes"])

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {args.out_json}")

    if args.print_stats:
        avg_gt = (total_gt_kept / total_samples_kept) if total_samples_kept > 0 else 0.0
        avg_det_before = (det_total_before / samples_with_det_scores) if samples_with_det_scores > 0 else 0.0
        avg_det_after = (det_total_after / samples_with_det_scores) if samples_with_det_scores > 0 else 0.0
        keep_ratio = (det_total_after / det_total_before) if det_total_before > 0 else 0.0

        print(f"New JSON scenes: {new_data['scene_count']}")
        print(f"New JSON samples: {total_samples_kept}")
        print(f"Avg GT objects per sample_token (kept): {avg_gt:.4f}")
        print(f"Samples with det.scores_3d: {samples_with_det_scores}")
        print(f"Det count before: {det_total_before}, after: {det_total_after}")
        print(f"Avg det/sample before: {avg_det_before:.4f}, after: {avg_det_after:.4f}")
        print(f"Keep ratio: {keep_ratio:.4f}")
        print(f"Potential det length mismatch samples (post-filter): {mismatch_samples}")


if __name__ == "__main__":
    main()