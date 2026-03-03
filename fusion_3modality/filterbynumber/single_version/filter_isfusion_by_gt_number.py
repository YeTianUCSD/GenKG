#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/filterbynumber/filter_isfusion_by_gt_number.py \
  --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/filterbynumber/isfusion_gt_leq10.json \
  --max_gt 10
'''

import json
import argparse
from copy import deepcopy

def gt_count(sample: dict) -> int:
    gt = sample.get("gt", {})
    boxes = gt.get("boxes_3d", [])
    if boxes is None:
        return 0
    return len(boxes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", type=str, required=True, help="input isfusion json")
    parser.add_argument("--out_json", type=str, required=True, help="output filtered json")
    parser.add_argument("--max_gt", type=int, default=10, help="keep samples whose GT count <= max_gt")
    args = parser.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {
        "version": data.get("version", ""),
        "scene_count": 0,
        "scenes": []
    }

    total_samples_new = 0
    total_gt_new = 0

    for scene in data.get("scenes", []):
        samples = scene.get("samples", [])
        kept = []
        for s in samples:
            c = gt_count(s)
            if c <= args.max_gt:
                kept.append(s)
                total_samples_new += 1
                total_gt_new += c

        if kept:
            new_scene = deepcopy(scene)
            new_scene["samples"] = kept
            new_scene["num_samples"] = len(kept)
            new_data["scenes"].append(new_scene)

    new_data["scene_count"] = len(new_data["scenes"])

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    avg_gt = (total_gt_new / total_samples_new) if total_samples_new > 0 else 0.0
    print(f"[OK] saved: {args.out_json}")
    print(f"New JSON scenes: {new_data['scene_count']}")
    print(f"New JSON samples: {total_samples_new}")
    print(f"Avg GT objects per sample_token (in new JSON): {avg_gt:.4f}")

if __name__ == "__main__":
    main()