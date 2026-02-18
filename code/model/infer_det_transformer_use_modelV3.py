#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只做推理 / 过滤 + keep_thr & max_per_frame 搜索版本（检测 P/R/F1 度量）：
- 加载训练好的 best.pt
- 使用给定的 JSON（含 det + gt + 位姿）
- 对每一帧的 det 进行“keep + 改类别”预测
- 在 (max_per_frame, keep_thr) 网格上搜索：
    - 每个组合都在 val 上计算「检测」P/R/F1：
        * TP: 预测框与某个 GT 匹配到的数量
        * FP: 预测框中没有匹配到任何 GT 的数量
        * FN: GT 中没有被任何预测框匹配到的数量
      （匹配使用类别感知 + 中心点距离 <= match_thr 的贪心一对一匹配）
    - 只要出现新的全局 best F1：
        * 用这组参数重新跑一次推理
        * 覆盖 out_dir/pred_flat.json 和 out_dir/pred_full.json
        * 写 out_dir/best_meta.json 记录这组参数和指标

示例用法（单组合，相当于传统推理）：

python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/model/infer_det_transformer_use_modelV3.py \
  --ckpt /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV5/search_res_V1/r3_mt1_wd0p001/best.pt\
  --input_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir /home/code/3Ddetection/IS-Fusion/GenKG/code/model/out_ourKG \
  --frame_radius 3 \
  --match_thr 3 \
  --max_per_frame 200 \
  --keep_thr 0.6

--keep_thr 0.5


%%% testtime
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/model/infer_det_transformer_use_modelV3.py \
  --ckpt /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV5/search_res_V1/r3_mt1_wd0p001/best.pt\
  --input_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir /home/code/3Ddetection/IS-Fusion/GenKG/code/model/testtime \
  --frame_radius 3 \
  --match_thr 3 \
  --max_per_frame 200 \
  --keep_thr 0.6








LOGFILE=/home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV4/infer_res_V3/infer_search.log
PIDFILE=/home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV4/infer_res_V3/infer_search.pid

setsid nohup python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/model/infer_det_transformer_use_modelV3.py \
  --ckpt /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV4/search_res/best_overall.pt \
  --input_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV4/infer_res_V3 \
  --frame_radius 3 \
  --match_thr 1 \
  --keep_thr_list 0.5,0.7,0.3 \
  --max_per_frame_list 200,150,100,50 \
  > "$LOGFILE" 2>&1 & printf '%s\n' "$!" > "$PIDFILE"

"""

import argparse
import json
import math
import os
import random
import copy
from typing import Any, Dict, Iterable, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from datetime import datetime
# ============================= Utils =============================
def log_step(msg: str, t0: float, newline: bool = True):
    """
    打印带时间戳的日志：
    - msg: 原本要 print 的内容
    - t0:  整个脚本开始的时刻（用来算已经过了多久）
    """
    now = time.time()
    elapsed = now - t0  # 从脚本开始到现在的秒数
    wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if newline:
        print(msg)
    # 这一行就是你要的“时间信息”
    print(f"[TIME] {wall} | +{elapsed:.2f}s since start")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj

def quat_to_rot(q: List[float]) -> np.ndarray:
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [    2*(x*z - y*w), 2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R

def make_T(translation: List[float], quat_wxyz: List[float]) -> np.ndarray:
    R = quat_to_rot(quat_wxyz)
    t = np.array(translation, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64); T[:3, :3] = R; T[:3, 3:] = t
    return T

def T_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0: return pts
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N,1), dtype=pts.dtype)])
    return (T @ homo.T).T[:, :3]

def rotate_yaw(T: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    if yaw.size == 0: return yaw
    R = T[:3, :3]
    v = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    v2 = (R @ v.T).T
    return np.arctan2(v2[:,1], v2[:,0])

def euclid2(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """二维欧式距离矩阵"""
    if a_xy.size == 0 or b_xy.size == 0:
        return np.zeros((a_xy.shape[0], b_xy.shape[0]), dtype=a_xy.dtype)
    a = a_xy[:, None, :]
    b = b_xy[None, :, :]
    return np.linalg.norm(a - b, axis=2)

def class_aware_greedy_match(det_xyz: np.ndarray, det_labels: np.ndarray,
                             gt_xyz: np.ndarray, gt_labels: np.ndarray,
                             thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    与训练脚本一致的“类别感知贪心匹配”：
    - 只在同一类别内匹配
    - 距离 <= thr 的 (det, gt) 才有可能匹配
    - 按距离从小到大贪心匹配，保证一对一
    返回：
      keep: [N_det] 0/1，是否匹配到某个 gt
      cls_tgt: [N_det] 若匹配到，则为该 gt 的类别；否则 -100
    """
    N = det_xyz.shape[0]
    keep = np.zeros((N,), dtype=np.int64)
    cls_tgt = np.full((N,), -100, dtype=np.int64)
    if gt_xyz.size == 0 or det_xyz.size == 0:
        return keep, cls_tgt

    classes = set(det_labels.tolist()) | set(gt_labels.tolist())
    for c in classes:
        di = np.where(det_labels == c)[0]
        gi = np.where(gt_labels == c)[0]
        if di.size == 0 or gi.size == 0:
            continue
        D = euclid2(det_xyz[di, :2], gt_xyz[gi, :2])
        pairs = []
        for ii in range(D.shape[0]):
            for jj in range(D.shape[1]):
                d = D[ii, jj]
                if d <= thr:
                    pairs.append((d, di[ii], gi[jj]))
        pairs.sort(key=lambda x: x[0])
        used_det, used_gt = set(), set()
        for d, i_det, j_gt in pairs:
            if i_det in used_det or j_gt in used_gt:
                continue
            used_det.add(i_det); used_gt.add(j_gt)
            keep[i_det] = 1
            cls_tgt[i_det] = int(gt_labels[j_gt])
    return keep, cls_tgt

# ============================= JSON parsing =============================

def iter_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    """递归遍历 JSON，找到所有带 det/gt 的样本"""
    if isinstance(obj, dict):
        det, gt = obj.get("det"), obj.get("gt")
        if isinstance(det, dict) and isinstance(gt, dict) and isinstance(det.get("boxes_3d"), list):
            yield obj
        for v in obj.values():
            yield from iter_samples(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_samples(it)

def load_root_and_samples(json_path: str):
    with open(json_path, "r") as f:
        root = json.load(f)
    samples = list(iter_samples(root))
    samples = [s for s in samples if "scene_token" in s and ("timestamp" in s or "timestamp_us" in s)]
    for s in samples:
        s["_ts"] = int(s.get("timestamp", s.get("timestamp_us", 0)))
    return root, samples

def group_by_scene(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_scene.setdefault(s["scene_token"], []).append(s)
    for k in by_scene:
        by_scene[k].sort(key=lambda x: x["_ts"])
    return by_scene

# ============================= Dataset =============================

class ClipDataset(torch.utils.data.Dataset):
    """每一帧都当中心帧；边界缺帧不产生 token。"""
    def __init__(self, scenes: Dict[str, List[Dict[str, Any]]],
                 frame_radius: int = 2,
                 max_per_frame: Optional[int] = None,
                 match_thr: float = 2.0,
                 ignore_classes: Optional[Set[int]] = None):
        self.scenes = scenes
        self.frame_radius = frame_radius
        self.max_per_frame = max_per_frame
        self.match_thr = match_thr
        self.ignore_classes = set(ignore_classes or [])
        self.items: List[Tuple[str, int]] = []
        for sc, frames in scenes.items():
            for i in range(len(frames)):
                self.items.append((sc, i))

    def __len__(self): return len(self.items)

    @staticmethod
    def _safe_stack(xs: List[List[float]], exp_dim: int) -> np.ndarray:
        arr = np.array(xs, dtype=np.float32)
        if arr.ndim == 1: arr = arr.reshape(-1, exp_dim)
        return arr

    def _get_T_k2ref(self, ref: Dict[str, Any], k: Dict[str, Any]) -> np.ndarray:
        Tl2e_ref = make_T(ref["lidar2ego"]["translation"], ref["lidar2ego"]["rotation"])
        Te2g_ref = make_T(ref["ego2global"]["translation"], ref["ego2global"]["rotation"])
        Tl2e_k   = make_T(k["lidar2ego"]["translation"],   k["lidar2ego"]["rotation"])
        Te2g_k   = make_T(k["ego2global"]["translation"],  k["ego2global"]["rotation"])
        return np.linalg.multi_dot([T_inv(Tl2e_ref), T_inv(Te2g_ref), Te2g_k, Tl2e_k])

    def __getitem__(self, idx: int):
        scene_token, ci = self.items[idx]
        frames = self.scenes[scene_token]
        ref = frames[ci]

        feats_list, pos_list, dt_list = [], [], []
        label_list, is_center_list, box_raw_list = [], [], []

        for rel in range(-self.frame_radius, self.frame_radius + 1):
            ki = ci + rel
            if ki < 0 or ki >= len(frames):
                continue
            k = frames[ki]
            T_k2ref = self._get_T_k2ref(ref, k)

            det = k.get("det", {}) or {}
            boxes = det.get("boxes_3d", []) or []
            scores = det.get("scores_3d", []) or []
            labels = det.get("labels_3d", []) or []

            boxes = self._safe_stack(boxes, 9)
            scores = np.array(scores, dtype=np.float32).reshape(-1)
            labels = np.array(labels, dtype=np.int64).reshape(-1)

            if self.max_per_frame is not None and boxes.shape[0] > self.max_per_frame:
                idxs = np.argsort(-scores)[:self.max_per_frame]
                boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]

            xyz = boxes[:, :3]; yaw = boxes[:, 6]
            xyz_ref = transform_points(T_k2ref, xyz)
            yaw_ref = rotate_yaw(T_k2ref, yaw)

            dxdy_dz = boxes[:, 3:6]; vxy = boxes[:, 7:9]
            cos_sin = np.stack([np.cos(yaw_ref), np.sin(yaw_ref)], axis=1)
            feat = np.concatenate([xyz_ref, dxdy_dz, cos_sin, vxy, scores[:, None]], axis=1)

            feats_list.append(feat.astype(np.float32))
            pos_list.append(xyz_ref.astype(np.float32))
            dt_list.extend([rel] * feat.shape[0])
            label_list.append(labels.astype(np.int64))
            is_center_list.append(np.full((feat.shape[0],), int(rel == 0), dtype=np.int64))
            if rel == 0:
                box_raw_list.append(boxes.astype(np.float32))

        feats = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 11), np.float32)
        pos   = np.concatenate(pos_list,   axis=0) if pos_list   else np.zeros((0, 3),  np.float32)
        labels= np.concatenate(label_list, axis=0) if label_list else np.zeros((0,),   np.int64)
        is_center = np.concatenate(is_center_list, axis=0) if is_center_list else np.zeros((0,), np.int64)
        dt = np.array(dt_list, dtype=np.int64)

        center_boxes = (box_raw_list[0] if box_raw_list else np.zeros((0,9), np.float32))
        center_xyz = center_boxes[:, :3]
        det_center_mask = (is_center == 1)
        center_labels = labels[det_center_mask] if det_center_mask.size > 0 else np.zeros((0,), np.int64)

        # ==== GT 框（用于后面检测指标评估） ====
        gt = ref.get("gt", {}) or {}
        gt_boxes = self._safe_stack(gt.get("boxes_3d", []) or [], 9)
        gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64)

        # ==== 下面这段 keep_target 目前不再用于评估，只保留兼容性（你也可以删掉） ====
        if len(self.ignore_classes) > 0:
            keep_idx_c = np.array(
                [i for i in range(center_labels.size) if center_labels[i] not in self.ignore_classes],
                dtype=np.int64
            )
            keep_idx_g = np.array(
                [i for i in range(gt_labels.size) if gt_labels[i] not in self.ignore_classes],
                dtype=np.int64
            )
        else:
            keep_idx_c = np.arange(center_labels.size, dtype=np.int64)
            keep_idx_g = np.arange(gt_labels.size, dtype=np.int64)

        keep_target = np.zeros(center_labels.shape, dtype=np.int64)
        class_target = np.full(center_labels.shape, -100, dtype=np.int64)

        if (
            keep_idx_c.size > 0 and keep_idx_g.size > 0
            and center_xyz.shape[0] > 0 and gt_boxes.shape[0] > 0
        ):
            keep_mask = np.zeros(center_labels.shape, dtype=np.int64)
            cls_mask  = np.full(center_labels.shape, -100, dtype=np.int64)
            k_det_xyz = center_xyz[keep_idx_c]
            k_gt_xyz  = gt_boxes[keep_idx_g, :3]
            k_det_lab = center_labels[keep_idx_c]
            k_gt_lab  = gt_labels[keep_idx_g]
            k_keep, k_cls = class_aware_greedy_match(
                k_det_xyz, k_det_lab, k_gt_xyz, k_gt_lab, thr=self.match_thr
            )
            keep_mask[keep_idx_c] = k_keep
            cls_mask[keep_idx_c]  = k_cls
            keep_target = keep_mask
            class_target = cls_mask

        return {
            "feats": torch.from_numpy(feats),
            "pos": torch.from_numpy(pos),
            "dt": torch.from_numpy(dt),
            "labels": torch.from_numpy(labels),
            "is_center": torch.from_numpy(is_center),
            "center_boxes": torch.from_numpy(center_boxes),
            "center_keep_tgt": torch.from_numpy(keep_target),
            "center_class_tgt": torch.from_numpy(class_target),
            "gt_boxes": torch.from_numpy(gt_boxes),
            "gt_labels": torch.from_numpy(gt_labels),
            "meta": {
                "scene_token": scene_token,
                "center_index": ci,
                "sample_token": ref.get("sample_token", ""),
                "sample_data_token": ref.get("sample_data_token", ""),
                "timestamp": int(ref.get("timestamp", ref.get("timestamp_us", 0))),
            }
        }

# ============================= Model =============================

class RelBias(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, pos: torch.Tensor, dt: torch.Tensor):
        S = pos.size(0)
        if S == 0:
            return torch.zeros((0, 0), device=pos.device, dtype=pos.dtype)
        dpos = pos[:, None, :] - pos[None, :, :]
        dtime = dt[:, None] - dt[None, :]
        f = torch.cat([dpos, dtime.unsqueeze(-1).to(pos.dtype)], dim=-1)
        return self.mlp(f).squeeze(-1)

class TransformerEncoderRel(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 num_classes=10,
                 time_emb_num: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(11, d_model)
        self.label_emb  = nn.Embedding(512, d_model)

        # 这里不再写死为 5，而是用 time_emb_num
        self.time_emb   = nn.Embedding(time_emb_num, d_model)
        # 让时间索引以 0..time_emb_num-1 为区间，中间那个对应 dt=0
        self.time_offset = (time_emb_num - 1) // 2

        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=False,
                activation="gelu"
            )
            for _ in range(num_layers)
        ])
        self.rel_biases = nn.ModuleList([RelBias(hidden=32) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.keep_head  = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, feats, labels, dt, pos, is_center):
        S = feats.size(0)
        if S == 0:
            device = feats.device
            C = self.class_head.out_features
            return (
                torch.empty(0, device=device),
                torch.empty(0, C, device=device),
                torch.empty(0, self.keep_head.in_features, device=device),
            )

        # 根据 time_emb_num 动态映射 dt -> [0, num_embeddings-1]
        time_idx = (dt + self.time_offset).clamp(
            0, self.time_emb.num_embeddings - 1
        )

        x = self.input_proj(feats) \
          + self.label_emb(labels.clamp(min=0, max=self.label_emb.num_embeddings-1)) \
          + self.time_emb(time_idx)

        x = x.unsqueeze(1)  # [S,1,D]
        for layer, rb in zip(self.enc_layers, self.rel_biases):
            bias = rb(pos, dt)  # [S,S]
            x = layer(x, src_mask=bias)
        x = self.norm(x).squeeze(1)  # [S,D]
        keep_logits = self.keep_head(x).squeeze(-1)
        class_logits= self.class_head(x)
        return keep_logits, class_logits, x


# ============================= Full-structure writeback =============================

def apply_det_replacements(root_obj: Any, repl_map: Dict[str, Dict[str, Any]]):
    """在整个 JSON 树里递归，把 det 替换成 repl_map 里的新结果"""
    if isinstance(root_obj, dict):
        maybe_det = root_obj.get("det", None)
        if isinstance(maybe_det, dict):
            keys = [
                root_obj.get("sample_token"),
                root_obj.get("sample_data_token"),
                str(root_obj.get("timestamp_us") if "timestamp_us" in root_obj else root_obj.get("timestamp")),
            ]
            chosen = None
            for k in keys:
                if k and k in repl_map:
                    chosen = k; break
            if chosen:
                new_det = repl_map[chosen]
                det = root_obj.get("det") or {}
                det["boxes_3d"] = new_det["boxes_3d"]
                det["labels_3d"] = new_det["labels_3d"]
                det["scores_3d"] = new_det["scores_3d"]
                root_obj["det"] = det
        for v in root_obj.values():
            apply_det_replacements(v, repl_map)
    elif isinstance(root_obj, list):
        for it in root_obj:
            apply_det_replacements(it, repl_map)

# ============================= Inference & Eval =============================

@torch.no_grad()
def run_inference(model, loader, device, keep_thr: float):
    """只负责根据 keep_thr 生成预测 JSON 所需的结果（不算指标）"""
    model.eval()
    flat_outputs = []
    det_replacements: Dict[str, Dict[str, Any]] = {}

    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"])
        is_center = b["is_center"]
        center_idx = (is_center == 1).nonzero(as_tuple=False).squeeze(-1)

        if center_idx.numel() == 0:
            boxes_kept = np.zeros((0,9), np.float32); labels_kept=[]; scores_kept=[]
        else:
            keep_scores = torch.sigmoid(keep_logits[center_idx])
            pred_keep = keep_scores > keep_thr
            pred_labels = class_logits[center_idx].argmax(dim=-1)

            boxes = b["center_boxes"].cpu().numpy()   # [Nc,9]
            labels = pred_labels.cpu().numpy()        # [Nc]
            scores = keep_scores.cpu().numpy()        # [Nc]
            mask = pred_keep.cpu().numpy().astype(bool)

            boxes_kept = boxes[mask]
            labels_kept= labels[mask].tolist()
            scores_kept= scores[mask].tolist()

        meta = b["meta"]
        scene_token = meta["scene_token"]
        sample_token = meta.get("sample_token", "")
        sample_data_token = meta.get("sample_data_token", "")
        timestamp = int(meta.get("timestamp", 0))

        flat_outputs.append({
            "scene_token": scene_token,
            "sample_token": sample_token,
            "sample_data_token": sample_data_token,
            "timestamp": timestamp,
            "det": {
                "boxes_3d": boxes_kept.tolist(),
                "labels_3d": labels_kept,
                "scores_3d": scores_kept
            }
        })

        payload = {
            "boxes_3d": boxes_kept.tolist(),
            "labels_3d": labels_kept,
            "scores_3d": scores_kept
        }
        if sample_token:
            det_replacements[sample_token] = payload
        elif sample_data_token:
            det_replacements[sample_data_token] = payload
        else:
            det_replacements[str(timestamp)] = payload

    return flat_outputs, det_replacements

@torch.no_grad()
def evaluate_det_prf(model, loader, device,
                     keep_thr: float,
                     match_thr: float,
                     ignore_classes: Optional[Set[int]] = None):
    """
    真正的“检测式” P/R/F1：
      - 先用 keep_thr 过滤预测框，再和 GT 框做类别感知 + 距离匹配；
      - TP: 被某个预测框匹配到的 GT 数；
      - FP: 预测框中没匹配到任何 GT 的数；
      - FN: GT 中没被任何预测框匹配到的数。
    """
    model.eval()
    ignore = set(ignore_classes or [])
    tot = {"tp": 0.0, "fp": 0.0, "fn": 0.0}

    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(
            b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"]
        )
        center_idx = (b["is_center"] == 1).nonzero(as_tuple=False).squeeze(-1)

        gt_boxes = b["gt_boxes"].cpu().numpy()
        gt_labels = b["gt_labels"].cpu().numpy()

        # 没有预测，也没有 GT：跳过
        if center_idx.numel() == 0 and gt_boxes.shape[0] == 0:
            continue

        # === 预测框（先从 center_boxes 里拿，再用 keep_thr 过滤） ===
        if center_idx.numel() == 0:
            # 没有任何 det token，相当于没有预测框
            det_boxes_kept = np.zeros((0, 9), np.float32)
            det_labels_kept = np.zeros((0,), np.int64)
        else:
            keep_scores = torch.sigmoid(keep_logits[center_idx])
            pred_keep = keep_scores > keep_thr
            pred_labels = class_logits[center_idx].argmax(dim=-1)

            boxes_all = b["center_boxes"].cpu().numpy()      # [Nc,9]
            labels_all = pred_labels.cpu().numpy()           # [Nc]
            mask = pred_keep.cpu().numpy().astype(bool)

            det_boxes_kept = boxes_all[mask]
            det_labels_kept = labels_all[mask]

        # === 按 ignore_classes 过滤标签（可选） ===
        if len(ignore) > 0:
            if det_labels_kept.size > 0:
                idx_d = np.array(
                    [i for i in range(det_labels_kept.size) if det_labels_kept[i] not in ignore],
                    dtype=np.int64
                )
                det_boxes_kept = det_boxes_kept[idx_d]
                det_labels_kept = det_labels_kept[idx_d]
            if gt_labels.size > 0:
                idx_g = np.array(
                    [i for i in range(gt_labels.size) if gt_labels[i] not in ignore],
                    dtype=np.int64
                )
                gt_boxes = gt_boxes[idx_g]
                gt_labels = gt_labels[idx_g]

        Nd = det_labels_kept.size
        Ng = gt_labels.size

        if Nd == 0 and Ng == 0:
            continue
        elif Nd == 0 and Ng > 0:
            tp = 0.0
            fp = 0.0
            fn = float(Ng)
        elif Nd > 0 and Ng == 0:
            tp = 0.0
            fp = float(Nd)
            fn = 0.0
        else:
            det_xyz = det_boxes_kept[:, :3]
            gt_xyz = gt_boxes[:, :3]
            keep_vec, _ = class_aware_greedy_match(
                det_xyz, det_labels_kept, gt_xyz, gt_labels, thr=match_thr
            )
            tp = float(keep_vec.sum())
            fp = float(Nd - tp)
            fn = float(Ng - tp)

        tot["tp"] += tp
        tot["fp"] += fp
        tot["fn"] += fn

    P = tot["tp"] / (tot["tp"] + tot["fp"]) if (tot["tp"] + tot["fp"]) > 0 else 0.0
    R = tot["tp"] / (tot["tp"] + tot["fn"]) if (tot["tp"] + tot["fn"]) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {"P": P, "R": R, "F1": F1, **tot}

def make_loader(scenes, frame_radius=2, max_per_frame=None,
                match_thr=2.0, ignore_classes=None, batch_size=1):
    ds = ClipDataset(scenes, frame_radius=frame_radius,
                     max_per_frame=max_per_frame,
                     match_thr=match_thr,
                     ignore_classes=ignore_classes)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        shuffle=False, num_workers=0,
        collate_fn=lambda x: x[0]
    )
    return ds, loader

# ============================= Main =============================

def main():
    # ---- 整体开始时间 ----
    total_start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练好的 best.pt 路径")
    ap.add_argument("--input_json", required=True, help="含 det+gt 的 JSON（val 集）")
    ap.add_argument("--out_dir", required=True, help="输出目录")

    ap.add_argument("--frame_radius", type=int, default=2)
    ap.add_argument("--max_per_frame", type=int, default=None)
    ap.add_argument("--match_thr", type=float, default=2.0,
                    help="预测与 GT 的匹配阈值（中心点距离），应与训练时一致或稍微调参")
    ap.add_argument("--ignore_classes", type=str, default="-1")

    ap.add_argument("--keep_thr", type=float, default=0.5,
                    help="默认的 keep 阈值（当不提供 keep_thr_list 时使用）")
    ap.add_argument("--keep_thr_list", type=str, default=None,
                    help="可选，多个 keep_thr，用逗号分隔，如: 0.3,0.5,0.7")
    ap.add_argument("--max_per_frame_list", type=str, default=None,
                    help="可选，多个 max_per_frame，用逗号分隔，如: 150,200,250")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 0) 解析 keep_thr / max_per_frame 的搜索列表
    if args.keep_thr_list:
        keep_thr_list = [float(x.strip()) for x in args.keep_thr_list.split(",") if x.strip()]
    else:
        keep_thr_list = [float(args.keep_thr)]

    if args.max_per_frame_list:
        max_per_frame_list = [int(x.strip()) for x in args.max_per_frame_list.split(",") if x.strip()]
    else:
        max_per_frame_list = [args.max_per_frame]

    log_step(f"[INFO] keep_thr_list      = {keep_thr_list}", total_start)
    log_step(f"[INFO] max_per_frame_list = {max_per_frame_list}", total_start)

    # 1) Load JSON & scenes
    log_step("[1/4] Loading JSON & samples...", total_start)
    root, samples = load_root_and_samples(args.input_json)
    scenes_all = group_by_scene(samples)
    log_step(f"[INFO] Scenes: {len(scenes_all)} | Samples: {len(samples)}", total_start)

    # 2) Load model
    log_step("[2/4] Loading model from ckpt...", total_start)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    num_classes = cfg.get("num_classes", 10)
    d_model     = cfg.get("d_model", 128)
    nhead       = cfg.get("nhead", 4)
    num_layers  = cfg.get("num_layers", 2)

    # 1) 优先从 cfg 里读（如果当时保存过）
    time_emb_num = cfg.get("time_emb_num", None)
    # 2) 如果 cfg 里没有，就直接从权重形状推出来
    if time_emb_num is None:
        time_emb_num = ckpt["model"]["time_emb.weight"].shape[0]

    model = TransformerEncoderRel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=4 * d_model,
        num_classes=num_classes,
        time_emb_num=time_emb_num,
    ).to(device)

    model.load_state_dict(ckpt["model"])

    print(f"[INFO] Model loaded: d_model={d_model}, nhead={nhead}, "
          f"layers={num_layers}, num_classes={num_classes}")

    # 3) 搜索 (max_per_frame, keep_thr) 组合
    log_step("[3/4] Searching over (max_per_frame, keep_thr) ...", total_start)
    ignore = set(int(x.strip()) for x in args.ignore_classes.split(",") if x.strip())

    best_F1 = -1.0
    best_info = None

    for mpf in max_per_frame_list:
        log_step(f"\n[INFO] Building dataset for max_per_frame={mpf} ...", total_start)
        ds, loader = make_loader(
            scenes_all,
            frame_radius=args.frame_radius,
            max_per_frame=mpf,
            match_thr=args.match_thr,
            ignore_classes=ignore,
            batch_size=1
        )
        print(f"[INFO] Clips (all frames as center): {len(ds)}")

        for kth in keep_thr_list:
            print(f"\n[EVAL] max_per_frame={mpf}, keep_thr={kth}")
            metrics = evaluate_det_prf(
                model, loader, device,
                keep_thr=kth,
                match_thr=args.match_thr,
                ignore_classes=ignore
            )
            print(f"      P={metrics['P']:.3f} R={metrics['R']:.3f} F1={metrics['F1']:.3f} "
                  f"(tp={int(metrics['tp'])}, fp={int(metrics['fp'])}, fn={int(metrics['fn'])})")

            if metrics["F1"] > best_F1:
                print("      -> New BEST found, regenerating JSON outputs ...")
                best_F1 = metrics["F1"]
                best_info = {
                    "frame_radius": args.frame_radius,
                    "match_thr": args.match_thr,
                    "ignore_classes": args.ignore_classes,
                    "keep_thr": kth,
                    "max_per_frame": mpf,
                    **metrics,
                    "ckpt": os.path.abspath(args.ckpt),
                    "input_json": os.path.abspath(args.input_json)
                }

                # 4) 用这一组 (mpf, kth) 生成 JSON
                flat_outputs, det_repl = run_inference(model, loader, device, keep_thr=kth)

                flat_path = os.path.join(args.out_dir, "pred_flat.json")
                with open(flat_path, "w") as f:
                    json.dump(flat_outputs, f, ensure_ascii=False, indent=2)
                print(f"      [OK] Wrote flat predictions to: {flat_path}")

                root_copy = copy.deepcopy(root)
                apply_det_replacements(root_copy, det_repl)
                full_path = os.path.join(args.out_dir, "pred_full.json")
                with open(full_path, "w") as f:
                    json.dump(root_copy, f, ensure_ascii=False, indent=2)
                print(f"      [OK] Wrote full-structure predictions to: {full_path}")

                # 记录最优组合 meta
                meta_path = os.path.join(args.out_dir, "best_meta.json")
                with open(meta_path, "w") as f:
                    json.dump(best_info, f, ensure_ascii=False, indent=2)
                print(f"      [OK] Wrote best meta to: {meta_path}")

    log_step("\n[4/4] Search finished.", total_start)
    if best_info is not None:
        print("[SUMMARY] Best config:")
        print(f"  max_per_frame={best_info['max_per_frame']}, keep_thr={best_info['keep_thr']}")
        print(f"  P={best_info['P']:.3f} R={best_info['R']:.3f} F1={best_info['F1']:.3f}")
        print(f"  Outputs: pred_flat.json, pred_full.json, best_meta.json in {args.out_dir}")
    else:
        print("[WARN] No valid clips / metrics computed; nothing was written.")

if __name__ == "__main__":
    main()
