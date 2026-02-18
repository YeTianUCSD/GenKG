#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge IS-Fusion PKL results with GT & poses (plus GT attribute names via devkit).
Outputs per-sample JSON with:
  - sample_token / sample_data_token(LIDAR_TOP) / timestamp
  - ego_pose (translation, rotation)
  - lidar2ego / ego2global (from infos)
  - det: boxes_3d/scores_3d/labels_3d
  - gt : boxes_3d/names/labels_3d/velocity/attr_names  (NO num_*_pts / valid_flag)

%--infos   /home/dataset/nuscene/nuscenes_infos_val.pkl \

Usage:
python /home/code/3Ddetection/IS-Fusion/GenKG/code/convert_isfusionres_gt_attribution2json.py \
  --infos   /home/code/3Ddetection/IS-Fusion/GenKG/data/nuscenes_infos_val_sorted.pkl \
  --results /home/code/3Ddetection/IS-Fusion/GenKG/data/res_from_ISFUSION.pkl \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --output  /home/code/3Ddetection/IS-Fusion/GenKG/data/ISFUSIONres_GT_attributions.json



python /home/code/3Ddetection/IS-Fusion/GenKG/code/convert_isfusionres_gt_attribution2json.py \
  --infos   /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/nuscenes_infos_train_sorted.pkl \
  --results /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/isfusion_nusece_train_res.pkl \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --output  /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/ISFUSIONres_GT_attributions_train.json
"""

import os, sys, json, argparse, pickle
from typing import Any, Dict, List

try:
    import torch  # noqa
except Exception:
    pass
try:
    import mmdet3d  # noqa
except Exception:
    print("[WARN] mmdet3d not importable; unpickling may fail.", file=sys.stderr)

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

DEFAULT_CLASS_NAMES = [
    'car','truck','construction_vehicle','bus','trailer','barrier',
    'motorcycle','bicycle','pedestrian','traffic_cone'
]

def to_numpy_array(obj):
    import numpy as np
    if obj is None:
        return None
    if hasattr(obj, "tensor"):
        t = obj.tensor
        try:
            return t.detach().cpu().numpy()
        except Exception:
            return np.asarray(t)
    if hasattr(obj, "detach") and callable(obj.detach):
        return obj.detach().cpu().numpy()
    try:
        return __import__("numpy").asarray(obj)
    except Exception:
        return None

def to_list(obj):
    if obj is None: return []
    try: return obj.tolist()
    except Exception:
        if isinstance(obj, (list, tuple)): return list(obj)
    return []

def extract_det_branch(one_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(one_result, dict):
        return {"boxes_3d": [], "scores_3d": [], "labels_3d": []}
    branch = None
    for k in ("pts_bbox","pts_bbox_NMS","img_bbox","bbox"):
        if k in one_result: branch = one_result[k]; break
    if branch is None or not isinstance(branch, dict):
        return {"boxes_3d": [], "scores_3d": [], "labels_3d": []}
    boxes  = to_numpy_array(branch.get("boxes_3d"))
    scores = to_numpy_array(branch.get("scores_3d"))
    labels = to_numpy_array(branch.get("labels_3d"))
    return {
        "boxes_3d": boxes.tolist()  if boxes  is not None else [],
        "scores_3d": scores.tolist() if scores is not None else [],
        "labels_3d": labels.tolist() if labels is not None else [],
    }

def build_gt_pack(inf: Dict[str, Any], cls2id: Dict[str,int]) -> Dict[str, Any]:
    import numpy as np
    names = [str(x) for x in to_list(inf.get("gt_names", []))]
    gb = to_numpy_array(inf.get("gt_boxes"))
    if gb is None:
        gb = to_numpy_array(inf.get("gt_bboxes_3d") or inf.get("boxes_3d"))
    vel = to_numpy_array(inf.get("gt_velocity"))
    if vel is not None:
        vel = np.nan_to_num(vel, nan=0.0)
        if vel.ndim == 1: vel = vel.reshape(1, -1)
        if vel.shape[1] >= 2: vel = vel[:, :2]
        else: vel = np.pad(vel, ((0,0),(0, 2-vel.shape[1])), mode="constant")
    boxes_3d = []
    if gb is not None:
        if gb.ndim == 1: gb = gb.reshape(1, -1)
        if gb.shape[1] < 7:
            pad = np.zeros((gb.shape[0], 7-gb.shape[1]), dtype=float)
            gb7 = np.concatenate([gb, pad], axis=1)
        else:
            gb7 = gb[:, :7]
        vv = vel if (vel is not None and len(vel)==gb7.shape[0]) else np.zeros((gb7.shape[0],2), dtype=float)
        boxes_3d = np.concatenate([gb7, vv], axis=1).tolist()
    labels = [cls2id.get(n, -1) for n in names]
    velocity_out = vel.tolist() if vel is not None else [[0.0,0.0] for _ in range(len(boxes_3d))]
    return {
        "boxes_3d": boxes_3d,
        "names": names,
        "labels_3d": labels,
        "velocity": velocity_out,
    }


def align_attr_names_with_gt(nusc: NuScenes, lidar_sd_token: str, gt_boxes_xyz: List[List[float]]) -> List[str]:
    """
    从 devkit 取出该 LIDAR_TOP 关键帧的 Box（已在 LiDAR 坐标系），
    读取每个 Box 的属性名，并与 gt_boxes 做最近邻(以中心点)对齐。
    返回与 gt_boxes 同长度的属性名列表（没有属性或匹配失败则为空字符串）。
    """
    import numpy as np
    if not lidar_sd_token or not gt_boxes_xyz:
        return []

    # 取回此帧的 boxes（在 LiDAR 坐标系）
    _, boxes, _ = nusc.get_sample_data(lidar_sd_token, box_vis_level=BoxVisibility.NONE)

    # 如果没有任何标注 box，则全空
    if not boxes:
        return [""] * len(gt_boxes_xyz)

    # 提取标注 box 的中心与属性
    ann_centers = np.array([b.center for b in boxes], dtype=float)  # (M,3)

    ann_attrs = []
    for b in boxes:
        attr = ""
        # devkit Box 可能有 attribute_names 或 attribute_name（不同版本）
        if hasattr(b, "attribute_names"):
            names = getattr(b, "attribute_names") or []
            if isinstance(names, list) and names:
                attr = names[0]
        elif hasattr(b, "attribute_name"):
            attr = getattr(b, "attribute_name") or ""
        else:
            # 兜底：通过 sample_annotation 表解析
            try:
                sa = nusc.get("sample_annotation", b.token)
                toks = sa.get("attribute_tokens", [])
                if toks:
                    attr = nusc.get("attribute", toks[0])["name"]
            except Exception:
                pass
        ann_attrs.append(attr)

    # GT 中心点：直接取前三列 (x, y, z)，兼容 N×7 或 N×9
    gt_arr = np.array(gt_boxes_xyz, dtype=float)
    if gt_arr.size == 0:
        return []
    gt_centers = gt_arr[:, :3]  # (N,3)

    # 最近邻匹配
    d = np.linalg.norm(gt_centers[:, None, :] - ann_centers[None, :, :], axis=2)  # (N, M)
    nn_idx = d.argmin(axis=1)
    nn_dist = d[np.arange(len(nn_idx)), nn_idx]

    # 阈值（米）：0.5 足够稳，也可根据需要调 0.3~0.75
    THRESH = 0.5
    out = [ann_attrs[j] if nn_dist[i] <= THRESH else "" for i, j in enumerate(nn_idx)]
    return out



def parse_class_names(s: str) -> List[str]:
    if not s: return DEFAULT_CLASS_NAMES
    arr = [x.strip() for x in s.split(",") if x.strip()]
    return arr or DEFAULT_CLASS_NAMES

def main():
    ap = argparse.ArgumentParser(description="Merge IS-Fusion PKL results with GT + poses + attributes into JSON.")
    ap.add_argument("--infos", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--output", required=True)
    ap.add_argument("--classes", default="")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    class_names = parse_class_names(args.classes)
    cls2id = {n:i for i,n in enumerate(class_names)}

    with open(args.infos, "rb") as f:
        infos_obj = pickle.load(f)
    infos = infos_obj["infos"] if isinstance(infos_obj, dict) and "infos" in infos_obj else infos_obj
    if not isinstance(infos, list):
        print("[ERR] Bad infos format.", file=sys.stderr); sys.exit(1)

    with open(args.results, "rb") as f:
        results = pickle.load(f)
    if not isinstance(results, list):
        print("[ERR] results must be a list.", file=sys.stderr); sys.exit(1)

    n = min(len(infos), len(results))
    if args.limit is not None: n = min(n, args.limit)
    print(f"[INFO] Using first {n} items.")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    out = []
    for idx in range(n):
        inf = infos[idx]; res = results[idx]
        sample_token = inf.get("token"); timestamp = inf.get("timestamp")

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

        # 取属性并对齐到 gt.boxes_3d 的顺序
        gt_attr_names = align_attr_names_with_gt(nusc, lidar_sd_token, gt["boxes_3d"])
        gt["attr_names"] = gt_attr_names

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
