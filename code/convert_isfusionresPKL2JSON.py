#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert IS-Fusion results PKL to JSON, and append sample_data_token & timestamp.

Usage:
  python /home/code/3Ddetection/IS-Fusion/GenKG/code/convert_isfusionresPKL2JSON.py \
      --infos /home/dataset/nuscene/nuscenes_infos_val.pkl \
      --results /home/code/3Ddetection/IS-Fusion/GenKG/data/res_from_ISFUSION.pkl \
      --dataroot /home/dataset/nuscene \
      --version v1.0-trainval \
      --output /home/code/3Ddetection/IS-Fusion/GenKG/data/ISFUSIONres_with_tokens.json

Notes:
  - results.pkl is the list produced by mmdet3d/IS-Fusion testing.
  - Each entry will include:
      sample_token, sample_data_token (LIDAR_TOP), timestamp,
      boxes_3d (list of [x,y,z,w,l,h,yaw,vx,vy] if available),
      scores_3d, labels_3d.
"""

import os
import sys
import json
import argparse
import pickle
from typing import Any, Dict, List

# Ensure these are importable for unpickling & token lookup
try:
    import torch  # noqa: F401
except Exception:
    pass

try:
    # mmdet3d structures are referenced inside the pickled results
    import mmdet3d  # noqa: F401
except Exception as e:
    print("[WARN] mmdet3d not importable. If unpickling fails, please install mmdet3d.", file=sys.stderr)

try:
    from nuscenes.nuscenes import NuScenes
except Exception as e:
    print("[ERR ] nuscenes-devkit not importable. Please `pip install nuscenes-devkit`.", file=sys.stderr)
    raise


def to_numpy_array(obj):
    """Convert torch / mmdet3d structures to numpy array safely."""
    import numpy as np
    # LiDARInstance3DBoxes-like
    if hasattr(obj, "tensor"):
        t = obj.tensor
        try:
            return t.detach().cpu().numpy()
        except Exception:
            return np.asarray(t)
    # torch.Tensor
    if hasattr(obj, "detach") and callable(obj.detach):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def extract_detection_dict(one_result: Dict[str, Any]) -> Dict[str, Any]:
    """Pick pts_bbox/img_bbox branch; return boxes/scores/labels as python lists."""
    if not isinstance(one_result, dict):
        return {"boxes_3d": [], "scores_3d": [], "labels_3d": []}

    # Prefer LiDAR branch
    branch = None
    for key in ("pts_bbox", "pts_bbox_NMS", "img_bbox", "bbox"):
        if key in one_result:
            branch = one_result[key]
            break

    if branch is None or not isinstance(branch, dict):
        return {"boxes_3d": [], "scores_3d": [], "labels_3d": []}

    boxes = branch.get("boxes_3d", None)
    scores = branch.get("scores_3d", None)
    labels = branch.get("labels_3d", None)

    boxes_np = to_numpy_array(boxes) if boxes is not None else None
    scores_np = to_numpy_array(scores) if scores is not None else None
    labels_np = to_numpy_array(labels) if labels is not None else None

    boxes_out = boxes_np.tolist() if boxes_np is not None else []
    scores_out = scores_np.tolist() if scores_np is not None else []
    labels_out = labels_np.tolist() if labels_np is not None else []

    return {
        "boxes_3d": boxes_out,
        "scores_3d": scores_out,
        "labels_3d": labels_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert IS-Fusion PKL results to JSON with tokens & timestamps.")
    parser.add_argument("--infos", required=True, help="Path to nuscenes_infos_*.pkl (from mmdet3d)")
    parser.add_argument("--results", required=True, help="Path to results pkl from IS-Fusion/mmdet3d")
    parser.add_argument("--dataroot", required=True, help="NuScenes dataroot (folder containing v1.0-*)")
    parser.add_argument("--version", default="v1.0-trainval", help="NuScenes version (default: v1.0-trainval)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional: only process first N samples")
    args = parser.parse_args()

    # Load infos (mapping index -> sample_token & timestamp)
    with open(args.infos, "rb") as f:
        infos_obj = pickle.load(f)
    if isinstance(infos_obj, dict) and "infos" in infos_obj:
        infos = infos_obj["infos"]
    elif isinstance(infos_obj, list):
        infos = infos_obj
    else:
        print("[ERR ] Unrecognized infos format. Expect dict['infos'] or list.", file=sys.stderr)
        sys.exit(1)

    # Load detection results list
    with open(args.results, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, list):
        print("[ERR ] results pkl must be a list.", file=sys.stderr)
        sys.exit(1)

    if len(infos) != len(results):
        print(f"[WARN] Length mismatch: infos={len(infos)} vs results={len(results)}. Will use min length.", file=sys.stderr)

    n = min(len(infos), len(results))
    if args.limit is not None:
        n = min(n, args.limit)

    print(f"[INFO] Using first {n} items.")

    # Init NuScenes for sample_data_token lookup
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    output: List[Dict[str, Any]] = []
    for idx in range(n):
        inf = infos[idx]
        res = results[idx]

        # sample & timestamp from infos
        sample_token = inf.get("token")
        timestamp = inf.get("timestamp", None)

        # LIDAR_TOP sample_data_token via devkit
        lidar_sd_token = None
        if sample_token is not None:
            sample = nusc.get("sample", sample_token)
            lidar_sd_token = sample["data"].get("LIDAR_TOP", None)

        det = extract_detection_dict(res)

        item = {
            "index": idx,
            "sample_token": sample_token,
            "sample_data_token": lidar_sd_token,  # LIDAR_TOP keyframe SampleData
            "timestamp": timestamp,
            "boxes_3d": det["boxes_3d"],
            "scores_3d": det["scores_3d"],
            "labels_3d": det["labels_3d"],
        }
        output.append(item)

        if (idx + 1) % 500 == 0 or idx + 1 == n:
            print(f"[INFO] Processed {idx + 1}/{n}")

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK  ] Wrote {len(output)} entries to: {args.output}")


if __name__ == "__main__":
    main()
