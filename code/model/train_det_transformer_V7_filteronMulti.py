#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP version with mid-epoch loss logging, explicit train/val split by JSON:
- 训练集：train_json（仅用于训练）
- 验证/测试集：val_json（每个 epoch 结束后评估）
- 每次出现新的 best 模型时，用 best 在 val 集上跑推理，写出：
    - pred_flat.json
    - pred_full.json
- 边界 pending：每一帧都当中心帧；缺帧不造 token
- 结构保留写回：仅替换 det.{boxes_3d, labels_3d, scores_3d}
- 多 GPU 训练/评估/推理（torchrun 启动）
- 训练中间过程打印（各卡平均 + EMA 平滑）

示例命令：

torchrun \
  --nproc_per_node=2 \
  --master_port=29600 \
  /home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_det_transformer_V7.py \
  --train_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONres_GT_attributions_train.json \
  --val_json   /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir    /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV7/ \
  --epochs 8 \
  --frame_radius 9 \
  --match_thr 2.0 \
  --keep_thr 0.5 \
  --ignore_classes -1 \
  --max_per_frame 150 \
  --max_tokens 200 \
  --assoc_thr 1.5 \
  --log_interval 20 \
  --lr 2e-4 \
  --weight_decay 1e-3 \
  --grad_clip 5.0 \
  --ema_momentum 0.9 \
  --num_workers 4

PIDFILE=/home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_v7_2.pid
LOGFILE=/home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_v7_2.log
CUDA_VISIBLE_DEVICES=0 \
python /home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_det_transformer_V7.py \
  --train_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONres_GT_attributions_train.json \
  --val_json   /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir    /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV7/ \
  --epochs 100 \
  --frame_radius 5 \
  --match_thr 2.0 \
  --keep_thr 0.5 \
  --ignore_classes -1 \
  --max_per_frame 200 \
  --max_tokens 1000 \
  --assoc_thr 1.5 \
  --log_interval 20 \
  --lr 2e-4 \
  --weight_decay 1e-3 \
  --grad_clip 5.0 \
  --ema_momentum 0.9 \
  --num_workers 0\
  >> "$LOGFILE" 2>&1 & printf '%s\n' "$!" > "$PIDFILE"


pkill -TERM -f train_det_transformer_V7.py 
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
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T

def T_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1), dtype=pts.dtype)])
    return (T @ homo.T).T[:, :3]

def rotate_yaw(T: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    if yaw.size == 0:
        return yaw
    R = T[:3, :3]
    v = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    v2 = (R @ v.T).T
    return np.arctan2(v2[:, 1], v2[:, 0])

def euclid2(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    if a_xy.size == 0 or b_xy.size == 0:
        return np.zeros((a_xy.shape[0], b_xy.shape[0]), dtype=a_xy.dtype)
    a = a_xy[:, None, :]
    b = b_xy[None, :, :]
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
            used_det.add(i_det)
            used_gt.add(j_gt)
            keep[i_det] = 1
            cls_tgt[i_det] = int(gt_labels[j_gt])
    return keep, cls_tgt

# ============================= Dataset (edge-pending, proposal-level) =============================

class ClipDataset(torch.utils.data.Dataset):
    """每一帧都当中心帧；边界缺帧不产生 token。"""
    def __init__(self, scenes: Dict[str, List[Dict[str, Any]]],
                 frame_radius: int = 2,
                 max_per_frame: Optional[int] = None,
                 match_thr: float = 2.0,
                 ignore_classes: Optional[Set[int]] = None,
                 max_tokens: Optional[int] = None,
                 assoc_thr: float = 2.0):
        """
        max_per_frame: 每帧内的 det top-K（按 score）
        max_tokens   : 整个时间窗口内，送进 Transformer 的最大 token 数（全局 top-M）
        assoc_thr    : 预留给时序关联的阈值（现在没用上，先占位）
        """
        self.scenes = scenes
        self.frame_radius = frame_radius
        self.max_per_frame = max_per_frame
        self.match_thr = match_thr
        self.ignore_classes = set(ignore_classes or [])
        self.max_tokens = max_tokens
        self.assoc_thr = assoc_thr

        self.items: List[Tuple[str, int]] = []
        for sc, frames in scenes.items():
            for i in range(len(frames)):
                self.items.append((sc, i))


    def __len__(self):
        return len(self.items)

    @staticmethod
    def _safe_stack(xs: List[List[float]], exp_dim: int) -> np.ndarray:
        arr = np.array(xs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, exp_dim)
        return arr

    def _get_T_k2ref(self, ref: Dict[str, Any], k: Dict[str, Any]) -> np.ndarray:
        Tl2e_ref = make_T(ref["lidar2ego"]["translation"], ref["lidar2ego"]["rotation"])
        Te2g_ref = make_T(ref["ego2global"]["translation"], ref["ego2global"]["rotation"])
        Tl2e_k   = make_T(k["lidar2ego"]["translation"],   k["lidar2ego"]["rotation"])
        Te2g_k   = make_T(k["ego2global"]["translation"],  k["ego2global"]["rotation"])
        # p_ref = T_inv(l2e_ref)*T_inv(e2g_ref)*e2g_k*l2e_k*p_k
        return np.linalg.multi_dot([T_inv(Tl2e_ref), T_inv(Te2g_ref), Te2g_k, Tl2e_k])

    def __getitem__(self, idx: int):
        scene_token, ci = self.items[idx]
        frames = self.scenes[scene_token]
        ref = frames[ci]

        # ========= 1) 收集窗口内所有帧的 det，统一变到 ref 坐标 =========
        all_xyz_ref_list: List[np.ndarray] = []
        all_boxes_ref_list: List[np.ndarray] = []
        all_scores_list: List[np.ndarray] = []
        all_labels_list: List[np.ndarray] = []
        all_rel_list: List[np.ndarray] = []

        for rel in range(-self.frame_radius, self.frame_radius + 1):
            ki = ci + rel
            if ki < 0 or ki >= len(frames):
                continue
            k = frames[ki]

            det = k.get("det", {}) or {}
            boxes = det.get("boxes_3d", []) or []
            scores = det.get("scores_3d", []) or []
            labels = det.get("labels_3d", []) or []

            boxes = self._safe_stack(boxes, 9)
            scores = np.array(scores, dtype=np.float32).reshape(-1)
            labels = np.array(labels, dtype=np.int64).reshape(-1)

            if boxes.shape[0] == 0:
                continue

            # 长度对齐
            n_det = min(boxes.shape[0], scores.shape[0], labels.shape[0])
            boxes = boxes[:n_det]
            scores = scores[:n_det]
            labels = labels[:n_det]

            # 每帧 top-K（max_per_frame）
            if self.max_per_frame is not None and n_det > self.max_per_frame:
                idxs_top = np.argsort(-scores)[:self.max_per_frame]
                boxes = boxes[idxs_top]
                scores = scores[idxs_top]
                labels = labels[idxs_top]

            # 变换到 ref 坐标
            T_k2ref = self._get_T_k2ref(ref, k)
            xyz = boxes[:, :3]
            yaw = boxes[:, 6]
            xyz_ref = transform_points(T_k2ref, xyz)
            yaw_ref = rotate_yaw(T_k2ref, yaw)

            box_ref = boxes.copy()
            box_ref[:, :3] = xyz_ref.astype(np.float32)
            box_ref[:, 6] = yaw_ref.astype(np.float32)

            all_xyz_ref_list.append(xyz_ref.astype(np.float32))
            all_boxes_ref_list.append(box_ref.astype(np.float32))
            all_scores_list.append(scores.astype(np.float32))
            all_labels_list.append(labels.astype(np.int64))
            all_rel_list.append(np.full((box_ref.shape[0],), rel, dtype=np.int64))

        if all_boxes_ref_list:
            xyz_ref_all = np.concatenate(all_xyz_ref_list, axis=0)    # [N_all,3]
            boxes_ref_all = np.concatenate(all_boxes_ref_list, axis=0)  # [N_all,9]
            scores_all = np.concatenate(all_scores_list, axis=0)      # [N_all]
            labels_all = np.concatenate(all_labels_list, axis=0)      # [N_all]
            rel_all = np.concatenate(all_rel_list, axis=0)            # [N_all]
        else:
            xyz_ref_all = np.zeros((0, 3), dtype=np.float32)
            boxes_ref_all = np.zeros((0, 9), dtype=np.float32)
            scores_all = np.zeros((0,), dtype=np.float32)
            labels_all = np.zeros((0,), dtype=np.int64)
            rel_all = np.zeros((0,), dtype=np.int64)

        # ========= 1.5) 全窗口 raw det 预截断（防止 tracklet 构造 O(N^2) 太慢） =========
        if xyz_ref_all.shape[0] > 0 and self.max_tokens is not None and self.max_tokens > 0:
            # 这里用一个放大系数，意味着：最多让 2 * max_tokens 个 det 参与 tracklet 构造
            #max_raw = self.max_tokens * 3
            max_raw = self.max_tokens

            N_all = xyz_ref_all.shape[0]
            if N_all > max_raw:
                # 用 scores_all 做全局排序，从整个时间窗口里取分数最高的 max_raw 个 proposal
                order = np.argsort(-scores_all)[:max_raw]

                xyz_ref_all   = xyz_ref_all[order]
                boxes_ref_all = boxes_ref_all[order]
                scores_all    = scores_all[order]
                labels_all    = labels_all[order]
                rel_all       = rel_all[order]
        # ========= 1.5) 预截断结束 =========
    



        # ========= 2) 在窗口内做时序关联，构造 tracklet =========
        tracklets: List[Dict[str, Any]] = []
        if boxes_ref_all.shape[0] > 0:
            unique_rels = np.unique(rel_all)
            unique_rels.sort()
            for rel_val in unique_rels:
                det_indices = np.where(rel_all == rel_val)[0]
                if det_indices.size == 0:
                    continue
                used_track_in_frame: Set[int] = set()
                # 遍历这一帧的每个 det，尝试接到已有 tracklet，否则新建
                for di in det_indices:
                    c = int(labels_all[di])
                    xy = xyz_ref_all[di, :2]
                    best_tid = -1
                    best_d = None
                    for tid, t in enumerate(tracklets):
                        if tid in used_track_in_frame:
                            continue
                        if t["label"] != c:
                            continue

                        last_idx = t["members"][-1]
                        # 新增：只跟“上一帧”的 tracklet 关联，避免跨很多帧全局搜索
                        if rel_all[last_idx] != rel_val - 1:
                            continue

                        xy_last = xyz_ref_all[last_idx, :2]
                        d = float(np.linalg.norm(xy - xy_last))
                        if d <= self.assoc_thr and (best_d is None or d < best_d):
                            best_d = d
                            best_tid = tid
                    if best_tid >= 0:
                        t = tracklets[best_tid]
                        t["members"].append(int(di))
                        s_di = float(scores_all[di])
                        if s_di > t["max_score"]:
                            t["max_score"] = s_di
                        used_track_in_frame.add(best_tid)
                    else:
                        tracklets.append({
                            "label": c,
                            "members": [int(di)],
                            "max_score": float(scores_all[di]),
                        })


        # ========= 3) 每个 tracklet 生成一个 proposal，并做全局 top-M =========
        if len(tracklets) > 0:
            prop_boxes = []
            prop_labels = []
            prop_scores = []
            for t in tracklets:
                members = np.array(t["members"], dtype=np.int64)  # indices into *_all
                if members.size == 0:
                    continue
                rels_abs = np.abs(rel_all[members])
                best_rel = rels_abs.min()
                cand = members[rels_abs == best_rel]
                if cand.size > 1:
                    cand_scores = scores_all[cand]
                    best_member = int(cand[np.argmax(cand_scores)])
                else:
                    best_member = int(cand[0])
                prop_boxes.append(boxes_ref_all[best_member])
                prop_labels.append(int(t["label"]))
                # 融合分数：取整个 tracklet 的 max_score
                prop_scores.append(float(t["max_score"]))

            if len(prop_boxes) > 0:
                prop_boxes = np.stack(prop_boxes, axis=0).astype(np.float32)   # [P,9]
                prop_labels = np.array(prop_labels, dtype=np.int64)            # [P]
                prop_scores = np.array(prop_scores, dtype=np.float32)          # [P]
                # 全局 top-M
                if self.max_tokens is not None and self.max_tokens > 0 and prop_boxes.shape[0] > self.max_tokens:
                    order = np.argsort(-prop_scores)[:self.max_tokens]
                    prop_boxes = prop_boxes[order]
                    prop_labels = prop_labels[order]
                    prop_scores = prop_scores[order]
            else:
                prop_boxes = np.zeros((0, 9), dtype=np.float32)
                prop_labels = np.zeros((0,), dtype=np.int64)
                prop_scores = np.zeros((0,), dtype=np.float32)
        else:
            prop_boxes = np.zeros((0, 9), dtype=np.float32)
            prop_labels = np.zeros((0,), dtype=np.int64)
            prop_scores = np.zeros((0,), dtype=np.float32)

        # ========= 4) 从 proposal 构建 feats / pos / dt / is_center =========
        if prop_boxes.shape[0] > 0:
            xyz_ref = prop_boxes[:, :3]
            dxdy_dz = prop_boxes[:, 3:6]
            yaw_ref = prop_boxes[:, 6]
            vxy_ref = prop_boxes[:, 7:9]
            cos_sin = np.stack([np.cos(yaw_ref), np.sin(yaw_ref)], axis=1)
            feats = np.concatenate(
                [xyz_ref, dxdy_dz, cos_sin, vxy_ref, prop_scores[:, None]],
                axis=1
            ).astype(np.float32)
            pos = xyz_ref.astype(np.float32)
            # 所有 token 都是“当前帧的 proposal”，dt=0, is_center=1
            dt = np.zeros((prop_boxes.shape[0],), dtype=np.int64)
            is_center = np.ones((prop_boxes.shape[0],), dtype=np.int64)
            labels = prop_labels
            center_boxes = prop_boxes.copy()
        else:
            feats = np.zeros((0, 11), dtype=np.float32)
            pos = np.zeros((0, 3), dtype=np.float32)
            dt = np.zeros((0,), dtype=np.int64)
            is_center = np.zeros((0,), dtype=np.int64)
            labels = np.zeros((0,), dtype=np.int64)
            center_boxes = np.zeros((0, 9), dtype=np.float32)

        # ========= 5) proposal 与当前帧 GT 匹配，生成监督 =========
        gt = ref.get("gt", {}) or {}
        gt_boxes = self._safe_stack(gt.get("boxes_3d", []) or [], 9)
        gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64)

        if len(self.ignore_classes) > 0:
            keep_idx_c = np.array(
                [i for i in range(labels.size) if int(labels[i]) not in self.ignore_classes],
                dtype=np.int64
            )
            keep_idx_g = np.array(
                [i for i in range(gt_labels.size) if int(gt_labels[i]) not in self.ignore_classes],
                dtype=np.int64
            )
        else:
            keep_idx_c = np.arange(labels.size, dtype=np.int64)
            keep_idx_g = np.arange(gt_labels.size, dtype=np.int64)

        keep_target = np.zeros(labels.shape, dtype=np.int64)
        class_target = np.full(labels.shape, -100, dtype=np.int64)

        if keep_idx_c.size > 0 and keep_idx_g.size > 0 and center_boxes.shape[0] > 0:
            keep_mask = np.zeros(labels.shape, dtype=np.int64)
            cls_mask = np.full(labels.shape, -100, dtype=np.int64)
            k_det_xyz = center_boxes[keep_idx_c, :3]
            k_gt_xyz = gt_boxes[keep_idx_g, :3]
            k_det_lab = labels[keep_idx_c]
            k_gt_lab = gt_labels[keep_idx_g]
            k_keep, k_cls = class_aware_greedy_match(
                k_det_xyz, k_det_lab, k_gt_xyz, k_gt_lab, thr=self.match_thr
            )
            keep_mask[keep_idx_c] = k_keep
            cls_mask[keep_idx_c] = k_cls
            keep_target = keep_mask
            class_target = cls_mask

        # 这一帧里参与评估的 GT 数量（已经按 ignore_classes 过滤）
        num_gt_valid = int(keep_idx_g.size)
    
        # Debug：如果 token 数异常大，打印一下
        if feats.shape[0] > 200 and ci % 2000 == 0:
            print(f"[DEBUG][scene={scene_token} idx={idx}] tokens={feats.shape[0]}")



        return {
            "feats": torch.from_numpy(feats),
            "pos": torch.from_numpy(pos),
            "dt": torch.from_numpy(dt),
            "labels": torch.from_numpy(labels),
            "is_center": torch.from_numpy(is_center),
            "center_boxes": torch.from_numpy(center_boxes),
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

        # === 时间 embedding 跟 frame_radius 绑定 ===
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

def compute_loss(keep_logits, class_logits, is_center, center_keep_tgt, center_class_tgt):
    center_idx = (is_center == 1).nonzero(as_tuple=False).squeeze(-1)
    if center_idx.numel() == 0:
        z = torch.tensor(0.0, device=center_keep_tgt.device)
        return z, {"loss_keep": 0.0, "loss_cls": 0.0}
    keep_logits_c = keep_logits[center_idx]
    class_logits_c= class_logits[center_idx]
    if keep_logits_c.numel() == 0:
        z = torch.tensor(0.0, device=center_keep_tgt.device)
        return z, {"loss_keep": 0.0, "loss_cls": 0.0}
    bce = F.binary_cross_entropy_with_logits(keep_logits_c, center_keep_tgt.float())
    pos_mask = (center_keep_tgt == 1)
    ce = F.cross_entropy(
        class_logits_c[pos_mask], center_class_tgt[pos_mask]
    ) if pos_mask.any() else keep_logits_c.sum()*0.0
    loss = bce + ce
    return loss, {"loss_keep": float(bce.detach().item()), "loss_cls": float(ce.detach().item())}

@torch.no_grad()
def evaluate(model, loader, device, keep_thr=0.5):
    """
    评估逻辑（GT 视角）：
      - 对每一帧：
          * 预测：sigmoid(keep_logits) > keep_thr 的中心 det/proposal
          * TP：这些预测里 center_keep_tgt == 1 的个数
          * FP：预测个数 - TP
          * FN：这一帧的 GT 数量(num_gt_valid) - TP
      - 汇总全数据集，计算
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

        center_idx = (b["is_center"] == 1).nonzero(as_tuple=False).squeeze(-1)
        # 这一帧 GT 的数量（已按 ignore_classes 过滤）
        num_gt = int(b["num_gt_valid"].item())

        if center_idx.numel() == 0:
            # 没有任何预测，所有 GT 都算 FN
            if num_gt > 0:
                tot["fn"] += float(num_gt)
            continue

        keep_scores = torch.sigmoid(keep_logits[center_idx])
        pred_keep = (keep_scores > keep_thr).long()
        num_pred = int(pred_keep.sum().item())

        # 没有 GT 但有预测 => 全部 FP
        if num_gt == 0:
            if num_pred > 0:
                tot["fp"] += float(num_pred)
            continue

        # center_keep_tgt: 1 表示这个 proposal 与某个 GT 一一匹配成功
        tgt_keep = b["center_keep_tgt"].long()  # [N_center]
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
    model.eval()
    flat_outputs = []
    det_replacements: Dict[str, Dict[str, Any]] = {}
    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(
            b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"]
        )
        center_idx = (b["is_center"] == 1).nonzero(as_tuple=False).squeeze(-1)

        if center_idx.numel() == 0:
            boxes_kept = np.zeros((0, 9), np.float32)
            labels_kept = []
            scores_kept = []
        else:
            keep_scores = torch.sigmoid(keep_logits[center_idx])
            pred_keep = keep_scores > keep_thr
            pred_labels = class_logits[center_idx].argmax(dim=-1)
            boxes = b["center_boxes"].cpu().numpy()
            labels = pred_labels.cpu().numpy()
            scores = keep_scores.cpu().numpy()
            mask = pred_keep.cpu().numpy().astype(bool)
            boxes_kept = boxes[mask]
            labels_kept = labels[mask].tolist()
            scores_kept = scores[mask].tolist()

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
# ============================= Loader helper =============================

def collate_single(batch):
    # 你的 batch_size 一直是 1，所以这里就直接返回第 0 个样本
    return batch[0]

def make_loader(dataset,
                batch_size=1,
                shuffle=True,
                sampler=None,
                num_workers=0):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_single,
    )
    return dataset, loader



# ============================= Main =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True, help="训练集 JSON（带 det+gt）")
    ap.add_argument("--val_json",   required=True, help="验证/测试集 JSON（带 det+gt）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--match_thr", type=float, default=2.0,
                    help="det 与 gt 最近邻匹配的距离阈值（米）")
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
    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader 的 num_workers（每个 rank 的 worker 数）")


    # ==== 新增的几个超参 ====
    ap.add_argument("--max_tokens",
                    type=int,
                    default=250,
                    help="每个中心帧最终送进 Transformer 的最大 proposal 数（全局 top-M）")
    ap.add_argument("--assoc_thr",
                    type=float,
                    default=2.0,
                    help="窗口内 det 时序关联（tracklet）的 xy 距离阈值")

    ap.add_argument("--frame_radius", type=int, default=2,
                    help="时间窗口半径（前后多少帧）；缺帧不造 token，例如 2 表示 [-2,-1,0,1,2]")
    ap.add_argument("--grad_clip", type=float, default=5.0,
                    help="梯度裁剪的 max_norm 阈值，防止梯度爆炸")
    ap.add_argument("--ema_momentum", type=float, default=0.9,
                    help="训练日志里 EMA 平滑系数，越大曲线越平滑")
    ap.add_argument("--weight_decay", type=float, default=1e-2,
                    help="AdamW 的 weight decay（L2 正则强度）")

    ap.add_argument("--keep_shards", action="store_true",
                    help="保留每个rank的中间分片文件；默认合并后自动删除")
    args = ap.parse_args()

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

    # 2) Datasets + Loaders with DistributedSampler
    if is_main_process():
        print("[2/5] Building loaders (DDP, edge pending)...")

    # --- 只建一份 Dataset ---
    train_ds = ClipDataset(
        train_scenes,
        frame_radius=args.frame_radius,
        max_per_frame=args.max_per_frame,
        match_thr=args.match_thr,
        ignore_classes=ignore,
        max_tokens=args.max_tokens,
        assoc_thr=args.assoc_thr,
    )
    val_ds = ClipDataset(
        val_scenes,
        frame_radius=args.frame_radius,
        max_per_frame=args.max_per_frame,
        match_thr=args.match_thr,
        ignore_classes=ignore,
        max_tokens=args.max_tokens,
        assoc_thr=args.assoc_thr,
    )

    # --- Sampler 挂在同一份 Dataset 上 ---
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if ddp else None

    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if ddp else None

    # --- DataLoader 使用同一份 Dataset + 同一个 Sampler ---
    train_ds, train_loader = make_loader(
        train_ds,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_ds, val_loader = make_loader(
        val_ds,
        sampler=val_sampler,
        num_workers=args.num_workers,
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
        max_dt=args.frame_radius,   # <- 跟 frame_radius 绑定
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

    best_f1, best_path = -1.0, os.path.join(args.out_dir, "best.pt")

    # 4) Train
    if is_main_process():
        print("[4/5] Training...")
    for epoch in range(1, args.epochs+1):
        if ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()

        ema_loss = ema_keep = ema_cls = None
        for it, batch in enumerate(train_loader, 1):
            b = to_device(batch, device)
            keep_logits, class_logits, _ = model(
                b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"]
            )
            loss, loss_dict = compute_loss(
                keep_logits, class_logits, b["is_center"],
                b["center_keep_tgt"], b["center_class_tgt"]
            )

            opt.zero_grad()
            loss.backward()
            # === 使用超参 grad_clip ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

            # 各卡平均 + EMA 平滑
            step_loss = reduce_mean_scalar(float(loss.item()), device)
            step_keep = reduce_mean_scalar(float(loss_dict["loss_keep"]), device)
            step_cls  = reduce_mean_scalar(float(loss_dict["loss_cls"]), device)
            if ema_loss is None:
                ema_loss, ema_keep, ema_cls = step_loss, step_keep, step_cls
            else:
                m = args.ema_momentum  # 使用超参 ema_momentum
                ema_loss = m * ema_loss + (1 - m) * step_loss
                ema_keep = m * ema_keep + (1 - m) * step_keep
                ema_cls  = m * ema_cls  + (1 - m) * step_cls

            if is_main_process() and (it % args.log_interval == 0):
                print(f"[Epoch {epoch}] step {it}/{len(train_loader)} | "
                      f"loss={ema_loss:.4f} keep={ema_keep:.4f} cls={ema_cls:.4f}")

        # Eval on val set
        metrics = evaluate(
            model.module if isinstance(model, DDP) else model,
            val_loader, device, keep_thr=args.keep_thr
        )

        # 所有 rank 上 best_f1 一致，这样 new_best 标志一致
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

        # 每次产生 new_best，就在 val 集上跑一遍推理，写 pred_flat.json / pred_full.json（已删除）

    barrier()
    if is_main_process():
        print("[5/5] Done.")

if __name__ == "__main__":
    main()
