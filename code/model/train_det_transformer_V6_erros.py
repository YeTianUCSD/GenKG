#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP version with mid-epoch loss logging, explicit train/val split by JSON
（跨帧候选框版本：在前后 frame_radius 帧的所有 det 上学习“补框”）:

- 训练集：train_json（仅用于训练）
- 验证/测试集：val_json（每个 epoch 结束后评估）
- 模型输入：
    * 当前场景下，前后 frame_radius 帧的所有 det，配准到中心帧坐标系
- 监督：
    * 以“当前帧的 GT”为基准，在整个时间窗口内的所有 det 中，
      找到与 GT 匹配（距离 <= match_thr、同类别）的候选框，标记 keep=1
    * 其它候选框 keep=0
- 输出：
    * 对每个时间窗口（中心帧），输出一组“补框后”的 det：
      在窗口内所有候选框中，模型判为 keep 的框，统一写成当前帧的 det。

示例命令（多卡）：

torchrun --nproc_per_node=2 /home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_det_transformer_V6.py \
  --train_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONres_GT_attributions_train.json \
  --val_json   /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir    /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV6/ \
  --epochs 20 \
  --frame_radius 1 \
  --match_thr 2.0 \
  --keep_thr 0.5 \
  --ignore_classes -1 \
  --max_per_frame 80 \
  --max_tokens 1024 \
  --merge_radius 2 \
  --log_interval 50 \
  --lr 2e-4 \
  --weight_decay 1e-3 \
  --grad_clip 5.0 \
  --ema_momentum 0.9

"""

import argparse
import torch.cuda.amp as amp

import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# ============================= DDP utils =============================

def ddp_init():
    """Initialize torch.distributed if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def all_reduce_dict(d: Dict[str, float], op=dist.ReduceOp.SUM):
    if not (dist.is_available() and dist.is_initialized()):
        return d
    keys = sorted(d.keys())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor([d[k] for k in keys], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=op)
    return {k: float(v) for k, v in zip(keys, tensor.tolist())}

def reduce_mean_scalar(x: float, device) -> float:
    """平均一个标量到所有进程（用于打印更稳定的 step 指标）"""
    t = torch.tensor([x], dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())

# ============================= Utils =============================

def set_seed(seed: int = 42, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

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
    if pts.size == 0:
        return pts
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N,1), dtype=pts.dtype)])
    return (T @ homo.T).T[:, :3]

def rotate_yaw(T: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    if yaw.size == 0:
        return yaw
    R = T[:3, :3]
    v = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    v2 = (R @ v.T).T
    return np.arctan2(v2[:,1], v2[:,0])

def euclid2(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    if a_xy.size == 0 or b_xy.size == 0:
        return np.zeros((a_xy.shape[0], b_xy.shape[0]), dtype=a_xy.dtype)
    a = a_xy[:, None, :]; b = b_xy[None, :, :]
    return np.linalg.norm(a - b, axis=2)

# ============================= JSON parsing =============================

def iter_samples(obj: Any) -> Iterable[Dict[str, Any]]:
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

# ============================= Matching =============================

def class_aware_greedy_match(det_xyz: np.ndarray, det_labels: np.ndarray,
                             gt_xyz: np.ndarray, gt_labels: np.ndarray,
                             thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    det_xyz: [N_det,3] (这里是整个时间窗口内、已配准到中心帧的所有候选框)
    gt_xyz : [N_gt,3]  (当前帧的 GT)
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

# ============================= Dataset (temporal candidates) =============================

class ClipDataset(torch.utils.data.Dataset):
    """每一帧都当中心帧；前后 frame_radius 帧的 det 统一配准到中心帧坐标系，作为候选框。"""
    def __init__(self, scenes: Dict[str, List[Dict[str, Any]]],
                 frame_radius: int = 2,
                 max_per_frame: Optional[int] = None,
                 match_thr: float = 2.0,
                 ignore_classes: Optional[Set[int]] = None,
                 max_tokens: Optional[int] = None,
                 merge_radius: Optional[float] = None):
        self.scenes = scenes
        self.frame_radius = frame_radius
        self.max_per_frame = max_per_frame
        self.match_thr = match_thr
        self.ignore_classes = set(ignore_classes or [])

        # ✨ 新增两个配置
        self.max_tokens = max_tokens
        self.merge_radius = merge_radius

        self.items: List[Tuple[str, int]] = []
        for sc, frames in scenes.items():
            for i in range(len(frames)):
                self.items.append((sc, i))


    def __len__(self):
        return len(self.items)

    @staticmethod
    def _safe_stack(xs: List[List[float]], exp_dim: int) -> np.ndarray:
        arr = np.array(xs, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, exp_dim), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, exp_dim)
        return arr

    def _merge_close_tokens(self,
                            feats: np.ndarray,
                            pos: np.ndarray,
                            dt: np.ndarray,
                            labels: np.ndarray,
                            is_center: np.ndarray,
                            boxes_ref_all: np.ndarray,
                            radius: float):
        """
        在中心帧坐标系下，对“同类且距离 <= radius”的 token 做简单聚类合并：
        - 聚类种子：score 最大的 token
        - 其它成员：与该 token label 相同且距离 <= radius 的未访问 token
        - 合并方式：
            * 连续特征 (feats, pos, boxes_ref)：用 score 做加权平均
            * score：取成员中的最大值，写回 feats 的最后一维
            * dt：取 score 最大那个成员的 dt
            * label：保持该簇的 label 不变
            * is_center：该簇中任意一个来自中心帧，则 is_center=1
        """
        N = feats.shape[0]
        if N == 0:
            return feats, pos, dt, labels, is_center, boxes_ref_all

        scores = feats[:, -1]  # 最后一维是 score
        order = np.argsort(-scores)  # 从高分到低分
        visited = np.zeros(N, dtype=bool)

        new_feats = []
        new_pos = []
        new_dt = []
        new_labels = []
        new_is_center = []
        new_boxes = []

        for idx in order:
            if visited[idx]:
                continue
            # 找到与当前 token 同类且在 radius 内的所有未访问 token
            same_cls = (labels == labels[idx])
            d = np.linalg.norm(pos - pos[idx], axis=1)  # [N]
            mask = (same_cls & (d <= radius) & (~visited))
            members = np.where(mask)[0]
            if members.size == 0:
                # 理论上不会出现，但防御一下
                members = np.array([idx], dtype=np.int64)
            visited[members] = True

            m_scores = scores[members]
            w = m_scores / (m_scores.sum() + 1e-6)

            # 连续量用 score 加权平均
            merged_pos = (pos[members] * w[:, None]).sum(axis=0)
            merged_feat = (feats[members] * w[:, None]).sum(axis=0)
            merged_box = (boxes_ref_all[members] * w[:, None]).sum(axis=0)

            # score 和 dt 特殊处理
            max_score_idx = members[np.argmax(m_scores)]
            merged_feat[-1] = scores[max_score_idx]  # 用该簇里最高 score
            merged_dt = dt[max_score_idx]
            merged_label = labels[idx]
            merged_is_center = int((is_center[members] == 1).any())

            new_feats.append(merged_feat.astype(np.float32))
            new_pos.append(merged_pos.astype(np.float32))
            new_dt.append(int(merged_dt))
            new_labels.append(int(merged_label))
            new_is_center.append(int(merged_is_center))
            new_boxes.append(merged_box.astype(np.float32))

        new_feats = np.stack(new_feats, axis=0)
        new_pos = np.stack(new_pos, axis=0)
        new_dt = np.array(new_dt, dtype=np.int64)
        new_labels = np.array(new_labels, dtype=np.int64)
        new_is_center = np.array(new_is_center, dtype=np.int64)
        new_boxes = np.stack(new_boxes, axis=0)

        return new_feats, new_pos, new_dt, new_labels, new_is_center, new_boxes



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
        label_list, is_center_list = [], []
        boxes_ref_list = []   # 所有 token（跨帧）在中心帧坐标系下的 boxes_3d

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

            boxes = self._safe_stack(boxes, 9)  # [N,9]
            scores = np.array(scores, dtype=np.float32).reshape(-1)
            labels = np.array(labels, dtype=np.int64).reshape(-1)

            if boxes.shape[0] == 0:
                continue

            if self.max_per_frame is not None and boxes.shape[0] > self.max_per_frame:
                idxs = np.argsort(-scores)[:self.max_per_frame]
                boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]

            xyz = boxes[:, :3]
            yaw = boxes[:, 6]
            dxdy_dz = boxes[:, 3:6]
            vxy = boxes[:, 7:9]

            # 变换到中心帧坐标系
            xyz_ref = transform_points(T_k2ref, xyz)
            yaw_ref = rotate_yaw(T_k2ref, yaw)

            cos_sin = np.stack([np.cos(yaw_ref), np.sin(yaw_ref)], axis=1)
            feat = np.concatenate([xyz_ref, dxdy_dz, cos_sin, vxy, scores[:, None]], axis=1)

            # 在中心帧坐标系下的 boxes_ref: [x,y,z,dx,dy,dz,yaw,vx,vy]
            boxes_ref = np.concatenate(
                [xyz_ref, dxdy_dz, yaw_ref[:, None], vxy],
                axis=1
            )

            feats_list.append(feat.astype(np.float32))
            pos_list.append(xyz_ref.astype(np.float32))
            dt_list.extend([rel] * feat.shape[0])
            label_list.append(labels.astype(np.int64))
            is_center_list.append(np.full((feat.shape[0],), int(rel == 0), dtype=np.int64))
            boxes_ref_list.append(boxes_ref.astype(np.float32))

        feats = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 11), np.float32)
        pos   = np.concatenate(pos_list,   axis=0) if pos_list   else np.zeros((0, 3),  np.float32)
        labels= np.concatenate(label_list, axis=0) if label_list else np.zeros((0,),   np.int64)
        is_center = np.concatenate(is_center_list, axis=0) if is_center_list else np.zeros((0,), np.int64)
        dt = np.array(dt_list, dtype=np.int64)
        boxes_ref_all = np.concatenate(boxes_ref_list, axis=0) if boxes_ref_list else np.zeros((0, 9), np.float32)

        # ✨ 可选：在中心帧坐标系下合并相邻重复框，减少 token 数量
        if self.merge_radius is not None and self.merge_radius > 0.0 and feats.shape[0] > 0:
            feats, pos, dt, labels, is_center, boxes_ref_all = self._merge_close_tokens(
                feats, pos, dt, labels, is_center, boxes_ref_all, self.merge_radius
            )

        # 当前帧 GT（在中心帧本地坐标系下）
        gt = ref.get("gt", {}) or {}
        gt_boxes = self._safe_stack(gt.get("boxes_3d", []) or [], 9)
        gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64)


        # 过滤 ignore_classes
        if len(self.ignore_classes) > 0:
            # det token 层面
            keep_idx_det = np.array(
                [i for i in range(labels.size) if labels[i] not in self.ignore_classes],
                dtype=np.int64
            )
            # GT 层面
            keep_idx_g = np.array(
                [i for i in range(gt_labels.size) if gt_labels[i] not in self.ignore_classes],
                dtype=np.int64
            )
        else:
            keep_idx_det = np.arange(labels.size, dtype=np.int64)
            keep_idx_g = np.arange(gt_labels.size, dtype=np.int64)

        keep_target = np.zeros(labels.shape, dtype=np.int64)
        class_target = np.full(labels.shape, -100, dtype=np.int64)

        if keep_idx_det.size > 0 and keep_idx_g.size > 0 and boxes_ref_all.shape[0] == labels.size:
            k_det_xyz = boxes_ref_all[keep_idx_det, :3]  # 所有候选框的 xyz（中心帧坐标）
            k_gt_xyz  = gt_boxes[keep_idx_g, :3]
            k_det_lab = labels[keep_idx_det]
            k_gt_lab  = gt_labels[keep_idx_g]

            k_keep, k_cls = class_aware_greedy_match(
                k_det_xyz, k_det_lab, k_gt_xyz, k_gt_lab, thr=self.match_thr
            )
            keep_mask = np.zeros(labels.shape, dtype=np.int64)
            cls_mask  = np.full(labels.shape, -100, dtype=np.int64)
            keep_mask[keep_idx_det] = k_keep
            cls_mask[keep_idx_det]  = k_cls
            keep_target = keep_mask
            class_target = cls_mask

        num_gt_valid = int(keep_idx_g.size)
        
            # ✨ 可选：对单个 clip 做 token 总数的硬限制
        if self.max_tokens is not None and feats.shape[0] > self.max_tokens:
            scores = feats[:, -1]  # 最后一维是 score
            center_idx = np.where(is_center == 1)[0]
            non_center_idx = np.where(is_center == 0)[0]

            keep_indices = []

            if center_idx.size >= self.max_tokens:
                # 中心帧太多，只能在中心帧里按 score 截断
                order = center_idx[np.argsort(-scores[center_idx])[:self.max_tokens]]
                keep = order
            else:
                # 先全保中心帧
                keep_indices.extend(center_idx.tolist())
                remain = self.max_tokens - len(keep_indices)
                if remain > 0 and non_center_idx.size > 0:
                    order = non_center_idx[np.argsort(-scores[non_center_idx])[:remain]]
                    keep_indices.extend(order.tolist())
                keep = np.array(keep_indices, dtype=np.int64)

            feats = feats[keep]
            pos = pos[keep]
            dt = dt[keep]
            labels = labels[keep]
            is_center = is_center[keep]
            boxes_ref_all = boxes_ref_all[keep]
            keep_target = keep_target[keep]
            class_target = class_target[keep]


        return {
            "feats": torch.from_numpy(feats),
            "pos": torch.from_numpy(pos),
            "dt": torch.from_numpy(dt),
            "labels": torch.from_numpy(labels),
            "is_center": torch.from_numpy(is_center),
            # 注意：现在 center_boxes 实际上是“所有跨帧候选框在中心帧坐标系下的 boxes_ref_all”
            "center_boxes": torch.from_numpy(boxes_ref_all),
            "center_keep_tgt": torch.from_numpy(keep_target),
            "center_class_tgt": torch.from_numpy(class_target),
            "num_gt_valid": torch.tensor(num_gt_valid, dtype=torch.int64),
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
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
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
                 max_dt: int = 2):
        """
        max_dt: 时间偏移的最大绝对值（即 frame_radius），
                用于构造 time embedding 词表大小：2*max_dt + 1
        """
        super().__init__()
        self.input_proj = nn.Linear(11, d_model)
        self.label_emb  = nn.Embedding(512, d_model)

        # 时间 embedding 跟 frame_radius 绑定
        self.max_dt = max_dt
        self.time_emb = nn.Embedding(2 * max_dt + 1, d_model)  # [-max_dt, max_dt] -> [0..2*max_dt]

        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=False,
                activation="gelu",
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
            D = self.keep_head.in_features
            return (
                torch.empty(0, device=device),
                torch.empty(0, C, device=device),
                torch.empty(0, D, device=device),
            )

        # 标签 embedding（防止越界）
        label_idx = labels.clamp(min=0, max=self.label_emb.num_embeddings - 1)

        # dt: [-max_dt, max_dt] -> [0, 2*max_dt]
        dt_long = dt.to(torch.long)
        dt_clamped = dt_long.clamp(-self.max_dt, self.max_dt)
        time_idx = dt_clamped + self.max_dt  # [0..2*max_dt]

        x = (
            self.input_proj(feats)
            + self.label_emb(label_idx)
            + self.time_emb(time_idx)
        )
        x = x.unsqueeze(1)  # [S,1,D]

        for layer, rb in zip(self.enc_layers, self.rel_biases):
            bias = rb(pos, dt)  # [S,S]
            x = layer(x, src_mask=bias)

        x = self.norm(x).squeeze(1)  # [S,D]
        keep_logits  = self.keep_head(x).squeeze(-1)
        class_logits = self.class_head(x)
        return keep_logits, class_logits, x

# ============================= Train / Eval =============================

def compute_loss(keep_logits, class_logits, keep_target, class_target):
    """
    现在的 keep_target / class_target 长度都是所有 token（跨帧候选框）的长度。
    """
    if keep_logits.numel() == 0:
        z = torch.tensor(0.0, device=keep_logits.device)
        return z, {"loss_keep": 0.0, "loss_cls": 0.0}

    keep_target = keep_target.to(keep_logits.device)
    class_target = class_target.to(class_logits.device)

    # keep: 所有候选框都参与监督
    bce = F.binary_cross_entropy_with_logits(keep_logits, keep_target.float())

    # class: 只在 keep_target == 1 的 token 上监督类别
    pos_mask = (keep_target == 1)
    if pos_mask.any():
        ce = F.cross_entropy(
            class_logits[pos_mask],
            class_target[pos_mask]
        )
    else:
        ce = keep_logits.sum() * 0.0

    loss = bce + ce
    return loss, {"loss_keep": float(bce.detach().item()), "loss_cls": float(ce.detach().item())}

@torch.no_grad()
def evaluate(model, loader, device, keep_thr=0.5):
    """
    新评估逻辑（GT 视角 + 跨帧候选）：

      - 对每一帧（一个 clip）：
          * 候选：窗口内所有 det token（已配准到当前帧坐标系）
          * 预测：sigmoid(keep_logits) > keep_thr 的 token
          * GT-匹配标记：center_keep_tgt == 1 的 token（来自 class_aware_greedy_match，1-1 匹配）
          * TP：预测为 keep 且 keep_target==1 的 token 数
          * FP：预测为 keep 但 keep_target==0 的 token 数
          * FN：这一帧的 GT 数(num_gt_valid) - TP
      - 汇总全数据集，计算：
          P = TP / (TP + FP)
          R = TP / (TP + FN)
    """
    model.eval()
    tot = {"tp": 0.0, "fp": 0.0, "fn": 0.0}

    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(
            b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"]
        )

        num_gt = int(b["num_gt_valid"].item())
        if keep_logits.numel() == 0:
            if num_gt > 0:
                tot["fn"] += float(num_gt)
            continue

        keep_scores = torch.sigmoid(keep_logits)
        pred_keep = (keep_scores > keep_thr).long()
        num_pred = int(pred_keep.sum().item())

        if num_gt == 0:
            if num_pred > 0:
                tot["fp"] += float(num_pred)
            continue

        tgt_keep = b["center_keep_tgt"].long().to(device)  # [N_tokens]
        tp = int(((pred_keep == 1) & (tgt_keep == 1)).sum().item())
        fp = num_pred - tp
        fn = num_gt - tp

        tot["tp"] += float(tp)
        tot["fp"] += float(fp)
        tot["fn"] += float(fn)

    # DDP 上做 all_reduce
    tot = all_reduce_dict(tot, op=dist.ReduceOp.SUM) if dist.is_initialized() else tot

    P = tot["tp"] / (tot["tp"] + tot["fp"]) if (tot["tp"] + tot["fp"]) > 0 else 0.0
    R = tot["tp"] / (tot["tp"] + tot["fn"]) if (tot["tp"] + tot["fn"]) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {"P": P, "R": R, "F1": F1, **tot}

# ============================= Full-structure writeback =============================

def apply_det_replacements(root_obj: Any, repl_map: Dict[str, Dict[str, Any]]):
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
                    chosen = k
                    break
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

@torch.no_grad()
def predict_shard(model, loader, device, keep_thr, shard_flat_path, shard_repl_path):
    """
    使用训练好的模型在 loader（通常是 val_loader）上跑一遍推理：
    - 对每个 clip（中心帧），取窗口内所有候选框 center_boxes（已配准到中心帧）
    - keep_score > keep_thr 的框作为当前帧的最终 det 输出
    """
    model.eval()
    flat_outputs = []
    det_replacements: Dict[str, Dict[str, Any]] = {}
    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(
            b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"]
        )

        if keep_logits.numel() == 0:
            boxes_kept = np.zeros((0,9), np.float32); labels_kept=[]; scores_kept=[]
        else:
            keep_scores = torch.sigmoid(keep_logits)
            pred_keep = keep_scores > keep_thr
            pred_labels = class_logits.argmax(dim=-1)

            boxes = b["center_boxes"].cpu().numpy()  # [N_tokens, 9] in center frame coords
            labels_all = pred_labels.cpu().numpy()
            scores_all = keep_scores.cpu().numpy()
            mask = pred_keep.cpu().numpy().astype(bool)

            boxes_kept = boxes[mask]
            labels_kept = labels_all[mask].tolist()
            scores_kept = scores_all[mask].tolist()

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

    with open(shard_flat_path, "w") as f:
        json.dump(flat_outputs, f, ensure_ascii=False, indent=2)
    with open(shard_repl_path, "w") as f:
        json.dump(det_replacements, f, ensure_ascii=False, indent=2)

# ============================= Loader helper =============================

def make_loader(scenes, batch_size=1, shuffle=True, sampler=None, **ds_kwargs):
    ds = ClipDataset(scenes, **ds_kwargs)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False if sampler is not None else shuffle,
        sampler=sampler, num_workers=0, collate_fn=lambda x: x[0]
    )
    return ds, loader

# ============================= Main =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True, help="训练集 JSON（带 det+gt）")
    ap.add_argument("--val_json",   required=True, help="验证/测试集 JSON（带 det+gt）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--match_thr", type=float, default=2.0,
                    help="det 与 gt 最近邻匹配的距离阈值（米），用于监督（跨帧匹配）。")
    ap.add_argument("--keep_thr", type=float, default=0.5,
                    help="推理时 keep 概率阈值（sigmoid 之后 > keep_thr 判为保留）")
    ap.add_argument("--ignore_classes", type=str, default="-1",
                    help="用逗号分隔的要忽略的类别 id，例如 '10,11'，默认只忽略 -1（无效类）")
    ap.add_argument("--max_per_frame", type=int, default=None,
                    help="每帧最多保留多少个 det（按 score 排序截断）；None 表示全保留")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_interval", type=int, default=50,
                    help="每隔多少 step 打印一次（各卡平均）损失")
     # ✨ 新增：单个 clip 的总 token 上限
    ap.add_argument("--max_tokens", type=int, default=0,
                    help="单个 clip 中 token 的最大个数（0 表示不限制）；"
                         "超过时优先保留中心帧 + 高分框")

    # ✨ 新增：跨帧合并重复框
    ap.add_argument("--merge_radius", type=float, default=0.0,
                    help=">0 时：在中心帧坐标系下，距离 <= merge_radius 且 label 相同的候选框会被合并为一个 token")

    # ==== 超参 ====
    ap.add_argument("--frame_radius", type=int, default=2,
                    help="时间窗口半径（前后多少帧）；例如 3 表示 [-3,-2,-1,0,1,2,3]")
    ap.add_argument("--grad_clip", type=float, default=5.0,
                    help="梯度裁剪的 max_norm 阈值，防止梯度爆炸")
    ap.add_argument("--ema_momentum", type=float, default=0.9,
                    help="训练日志里 EMA 平滑系数，越大曲线越平滑")
    ap.add_argument("--weight_decay", type=float, default=1e-2,
                    help="AdamW 的 weight decay（L2 正则强度）")

    ap.add_argument("--keep_shards", action="store_true",
                    help="保留每个rank的中间分片文件；默认合并后自动删除")
    args = ap.parse_args()

    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    merge_radius = args.merge_radius if args.merge_radius > 0.0 else None


    ddp, rank, world_size, local_rank = ddp_init()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed, rank)

    if is_main_process():
        print(f"[DDP] distributed={ddp} world_size={world_size} device={device}")

    # 1) Load train & val
    if is_main_process():
        print("[1/5] Loading train & val samples...")
    train_root, train_samples = load_root_and_samples(args.train_json)
    val_root,   val_samples   = load_root_and_samples(args.val_json)

    train_scenes = group_by_scene(train_samples)
    val_scenes   = group_by_scene(val_samples)

    if is_main_process():
        print(f"Scenes: train={len(train_scenes)}  val={len(val_scenes)}")
        print(f"Samples: train={len(train_samples)} val={len(val_samples)}")

    ignore = set(int(x.strip()) for x in args.ignore_classes.split(",") if x.strip())

    # 2) Loaders with DistributedSampler
    if is_main_process():
        print("[2/5] Building loaders (DDP, temporal candidates)...")

    train_dataset_for_sampler = ClipDataset(
        train_scenes,
        frame_radius=args.frame_radius,
        max_per_frame=args.max_per_frame,
        match_thr=args.match_thr,
        ignore_classes=ignore,
        max_tokens=max_tokens,
        merge_radius=merge_radius,
    )
    val_dataset_for_sampler = ClipDataset(
        val_scenes,
        frame_radius=args.frame_radius,
        max_per_frame=args.max_per_frame,
        match_thr=args.match_thr,
        ignore_classes=ignore,
        max_tokens=max_tokens,
        merge_radius=merge_radius,
    )

    # ✨ 少了这两行，要加回去
    train_sampler = DistributedSampler(
        train_dataset_for_sampler, shuffle=True
    ) if ddp else None
    val_sampler = DistributedSampler(
        val_dataset_for_sampler, shuffle=False
    ) if ddp else None    

    train_ds, train_loader = make_loader(
        train_scenes,
        sampler=train_sampler,
        frame_radius=args.frame_radius,
        max_per_frame=args.max_per_frame,
        match_thr=args.match_thr,
        ignore_classes=ignore,
        max_tokens=max_tokens,
        merge_radius=merge_radius,
    )
    val_ds,  val_loader  = make_loader(
        val_scenes,
        sampler=val_sampler,
        frame_radius=args.frame_radius,
        max_per_frame=args.max_per_frame,
        match_thr=args.match_thr,
        ignore_classes=ignore,
        max_tokens=max_tokens,
        merge_radius=merge_radius,
    )


    if is_main_process():
        print(f"Train clips (all frames as center): {len(train_ds)} | "
              f"Val clips (all frames as center): {len(val_ds)}")

    # 3) Model
    if is_main_process():
        print("[3/5] Building model...")
    model = TransformerEncoderRel(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=4 * args.d_model,
        num_classes=args.num_classes,
        max_dt=args.frame_radius,   # 跟 frame_radius 绑定
    ).to(device)

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = amp.GradScaler()
    best_f1, best_path = -1.0, os.path.join(args.out_dir, "best.pt")

    # 4) Train
    if is_main_process():
        print("[4/5] Training...")

    for epoch in range(1, args.epochs + 1):
        if ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()

        ema_loss = ema_keep = ema_cls = None

        for it, batch in enumerate(train_loader, 1):
            b = to_device(batch, device)

            # ====== AMP 前向 ======
            with amp.autocast():
                keep_logits, class_logits, _ = model(
                    b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"]
                )
                # 注意：这里对应你 V6 版本的 compute_loss(keep_logits, class_logits, keep_tgt, cls_tgt)
                loss, loss_dict = compute_loss(
                    keep_logits, class_logits,
                    b["center_keep_tgt"], b["center_class_tgt"]
                )

            opt.zero_grad()

            # ====== AMP 反向 ======
            scaler.scale(loss).backward()

            # 先 unscale，再做梯度裁剪
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(opt)
            scaler.update()

            # ====== 各卡平均 + EMA 平滑 ======
            step_loss = reduce_mean_scalar(float(loss.item()), device)
            step_keep = reduce_mean_scalar(float(loss_dict["loss_keep"]), device)
            step_cls  = reduce_mean_scalar(float(loss_dict["loss_cls"]), device)

            if ema_loss is None:
                ema_loss, ema_keep, ema_cls = step_loss, step_keep, step_cls
            else:
                m = args.ema_momentum
                ema_loss = m * ema_loss + (1 - m) * step_loss
                ema_keep = m * ema_keep + (1 - m) * step_keep
                ema_cls  = m * ema_cls  + (1 - m) * step_cls

            if is_main_process() and (it % args.log_interval == 0):
                print(f"[Epoch {epoch}] step {it}/{len(train_loader)} | "
                      f"loss={ema_loss:.4f} keep={ema_keep:.4f} cls={ema_cls:.4f}")

        # ====== 每个 epoch 结束做一次验证 ======
        metrics = evaluate(
            model.module if isinstance(model, DDP) else model,
            val_loader, device, keep_thr=args.keep_thr
        )

        new_best = False
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            new_best = True

        if is_main_process():
            print(f"[E{epoch}] Val: P={metrics['P']:.3f} R={metrics['R']:.3f} F1={metrics['F1']:.3f} "
                  f"(tp={int(metrics['tp'])}, fp={int(metrics['fp'])}, fn={int(metrics['fn'])})")
            if new_best:
                torch.save(
                    {
                        "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                        "cfg": vars(args)
                    },
                    best_path
                )
                print(f"  -> new best saved to {best_path}")

    barrier()
    if is_main_process():
        print("[5/5] Done.")
    

if __name__ == "__main__":
    main()
