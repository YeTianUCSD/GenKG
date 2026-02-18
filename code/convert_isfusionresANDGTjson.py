#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert IS-Fusion results PKL to a JSON that also contains GT and poses.
Adds per-sample:
  - sample_token / sample_data_token(LIDAR_TOP) / timestamp
  - ego_pose (translation, rotation)  # from NuScenes ego_pose table
  - lidar2ego / ego2global            # from infos

Output item:
{
  "index": int,
  "sample_token": str,
  "sample_data_token": str,
  "timestamp": int,
  "ego_pose": {"translation":[x,y,z], "rotation":[w,x,y,z]},
  "lidar2ego": {"translation":[...], "rotation":[...]},
  "ego2global": {"translation":[...], "rotation":[...]},
  "det": {
    "boxes_3d": [[x,y,z,w,l,h,yaw,vx,vy], ...],
    "scores_3d": [...],
    "labels_3d": [...]
  },
  "gt": {
    "boxes_3d": [[x,y,z,w,l,h,yaw,vx,vy], ...],
    "names": [str, ...],
    "labels_3d": [int, ...],
    "velocity": [[vx,vy], ...]
  }
}

Usage:
python /home/code/3Ddetection/IS-Fusion/GenKG/code/convert_isfusionresANDGTjson.py \
  --infos   /home/dataset/nuscene/nuscenes_infos_val.pkl \
  --results /home/code/3Ddetection/IS-Fusion/GenKG/data/res_from_ISFUSION.pkl \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --output  /home/code/3Ddetection/IS-Fusion/GenKG/data/ISFUSIONres_and_GT.json
"""

import os, sys, json, argparse, pickle
from typing import Any, Dict, List

# allow unpickling tensors/boxes
try:
    import torch  # noqa
except Exception:
    pass
try:
    import mmdet3d  # noqa
except Exception:
    print("[WARN] mmdet3d not importable; unpickling may fail.", file=sys.stderr)

from nuscenes.nuscenes import NuScenes

DEFAULT_CLASS_NAMES = [
    'car','truck','construction_vehicle','bus','trailer','barrier',
    'motorcycle','bicycle','pedestrian','traffic_cone'
]

def to_numpy_array(obj):
    import numpy as np
    if obj is None:
        return None
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
    # numpy array / python list
    try:
        return np.asarray(obj)
    except Exception:
        return None

def to_list(obj):
    """Robustly convert numpy/torch/list to Python list (including numpy string arrays)."""
    if obj is None:
        return []
    try:
        return obj.tolist()
    except Exception:
        if isinstance(obj, (list, tuple)):
            return list(obj)
    return []

def extract_det_branch(one_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(one_result, dict):
        return {"boxes_3d": [], "scores_3d": [], "labels_3d": []}
    branch = None
    for k in ("pts_bbox", "pts_bbox_NMS", "img_bbox", "bbox"):
        if k in one_result:
            branch = one_result[k]; break
    if branch is None or not isinstance(branch, dict):
        return {"boxes_3d": [], "scores_3d": [], "labels_3d": []}
    boxes = to_numpy_array(branch.get("boxes_3d"))
    scores = to_numpy_array(branch.get("scores_3d"))
    labels = to_numpy_array(branch.get("labels_3d"))
    return {
        "boxes_3d": boxes.tolist() if boxes is not None else [],
        "scores_3d": scores.tolist() if scores is not None else [],
        "labels_3d": labels.tolist() if labels is not None else [],
    }

def build_gt_pack(inf: Dict[str, Any], cls2id: Dict[str,int]) -> Dict[str, Any]:
    """
    Normalize GT to JSON-friendly dict:
      - boxes_3d: [x,y,z,w,l,h,yaw,vx,vy]
      - names: list[str]
      - labels_3d: list[int] (mapped by class_names)
      - velocity: [[vx,vy], ...]
    """
    import numpy as np

    # gt_names 兼容 numpy 字符串数组
    names = [str(x) for x in to_list(inf.get("gt_names", []))]

    # gt_boxes 可能是 ndarray(N, >=7) 或 LiDARInstance3DBoxes
    gb = to_numpy_array(inf.get("gt_boxes"))
    if gb is None:
        # 有些版本字段名不同，兜底试一下
        gb = to_numpy_array(inf.get("gt_bboxes_3d") or inf.get("boxes_3d"))

    # gt_velocity: ndarray(N,2)，有时包含 NaN
    vel = to_numpy_array(inf.get("gt_velocity"))
    if vel is not None:
        # 统一形状 & NaN -> 0
        vel = np.nan_to_num(vel, nan=0.0)
        if vel.ndim == 1:
            vel = vel.reshape(1, -1)
        # 只取前两维 vx, vy
        if vel.shape[1] >= 2:
            vel = vel[:, :2]
        else:
            vel = np.pad(vel, ((0,0),(0, 2-vel.shape[1])), mode="constant")
    else:
        vel = None

    boxes_3d = []
    if gb is not None:
        if gb.ndim == 1:
            gb = gb.reshape(1, -1)
        # 先保证有 7 个基本参数
        if gb.shape[1] < 7:
            pad = np.zeros((gb.shape[0], 7 - gb.shape[1]), dtype=float)
            gb7 = np.concatenate([gb, pad], axis=1)
        else:
            gb7 = gb[:, :7]

        # 对齐速度长度（若缺失则补零）
        if vel is None or len(vel) != gb7.shape[0]:
            vv = np.zeros((gb7.shape[0], 2), dtype=float)
        else:
            vv = vel
        boxes_3d = np.concatenate([gb7, vv], axis=1).tolist()

    labels = [cls2id.get(n, -1) for n in names]
    velocity_out = vel.tolist() if vel is not None else [[0.0, 0.0] for _ in range(len(boxes_3d))]

    return {
        "boxes_3d": boxes_3d,
        "names": names,
        "labels_3d": labels,
        "velocity": velocity_out,
    }

def parse_class_names(s: str) -> List[str]:
    if not s:
        return DEFAULT_CLASS_NAMES
    arr = [x.strip() for x in s.split(",") if x.strip()]
    return arr or DEFAULT_CLASS_NAMES

def main():
    ap = argparse.ArgumentParser(description="Merge IS-Fusion PKL results with GT + poses into JSON.")
    ap.add_argument("--infos", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--output", required=True)
    ap.add_argument("--classes", default="")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    class_names = parse_class_names(args.classes)
    cls2id = {n: i for i, n in enumerate(class_names)}

    # load infos
    with open(args.infos, "rb") as f:
        infos_obj = pickle.load(f)
    infos = infos_obj["infos"] if isinstance(infos_obj, dict) and "infos" in infos_obj else infos_obj
    if not isinstance(infos, list):
        print("[ERR] Bad infos format.", file=sys.stderr); sys.exit(1)

    # load det results
    with open(args.results, "rb") as f:
        results = pickle.load(f)
    if not isinstance(results, list):
        print("[ERR] results must be a list.", file=sys.stderr); sys.exit(1)

    n = min(len(infos), len(results))
    if args.limit is not None:
        n = min(n, args.limit)
    print(f"[INFO] Using first {n} items.")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    out = []
    for idx in range(n):
        inf = infos[idx]
        res = results[idx]

        sample_token = inf.get("token")
        timestamp    = inf.get("timestamp")

        # LIDAR_TOP sample_data_token + ego_pose
        lidar_sd_token = None
        ego_pose = {"translation": [], "rotation": []}
        if sample_token:
            sample = nusc.get("sample", sample_token)
            lidar_sd_token = sample["data"].get("LIDAR_TOP")
            if lidar_sd_token:
                sd = nusc.get("sample_data", lidar_sd_token)
                ep = nusc.get("ego_pose", sd["ego_pose_token"])
                ego_pose = {"translation": ep["translation"], "rotation": ep["rotation"]}

        det = extract_det_branch(res)
        gt  = build_gt_pack(inf, cls2id)

        item = {
            "index": idx,
            "sample_token": sample_token,
            "sample_data_token": lidar_sd_token,
            "timestamp": timestamp,
            "ego_pose": ego_pose,
            "lidar2ego": {
                "translation": inf.get("lidar2ego_translation", []),
                "rotation":    inf.get("lidar2ego_rotation", [])
            },
            "ego2global": {
                "translation": inf.get("ego2global_translation", []),
                "rotation":    inf.get("ego2global_rotation", [])
            },
            "det": det,
            "gt": gt
        }
        out.append(item)

        if (idx+1) % 500 == 0 or (idx+1) == n:
            print(f"[INFO] Processed {idx+1}/{n}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {len(out)} entries to: {args.output}")

if __name__ == "__main__":
    main()
