#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP version: Spatio-Temporal Proposal Refinement (Transformer, relative pos/time bias)
- 边界 pending：每一帧都当中心帧；缺帧不造 token
- 结构保留写回：仅替换 det.{boxes_3d, labels_3d, scores_3d}
- 多 GPU 训练/评估/推理（torchrun 启动）

# 多卡（例如 4 卡）
torchrun --nproc_per_node=2 /home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_det_transformer_V1.py \
  --input_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr.json \
  --out_dir    /home/code/3Ddetection/IS-Fusion/GenKG/code/model/out \
  --epochs 8 \
  --match_thr 2.0 \
  --keep_thr 0.5 \
  --ignore_classes -1 \
  
  
  
  %--max_per_frame 200

# 单卡（等价于原先）
python /home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_det_transformer_V1.py \
  --input_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr.json \
  --out_dir    /home/code/3Ddetection/IS-Fusion/GenKG/code/model/out/ \
  --epochs 8 --match_thr 2.0 --keep_thr 0.5 --ignore_classes -1

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

# ----------------------------- DDP utils -----------------------------

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
        # single process
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
    tensor = torch.tensor([d[k] for k in keys], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    dist.all_reduce(tensor, op=op)
    return {k: float(v) for k, v in zip(keys, tensor.tolist())}

# ----------------------------- Utils -----------------------------

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
    if a_xy.size == 0 or b_xy.size == 0:
        return np.zeros((a_xy.shape[0], b_xy.shape[0]), dtype=a_xy.dtype)
    a = a_xy[:, None, :]; b = b_xy[None, :, :]
    return np.linalg.norm(a - b, axis=2)

# ----------------------------- JSON parsing -----------------------------

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

# ----------------------------- Matching ---------------------------------

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
        if di.size == 0 or gi.size == 0: continue
        D = euclid2(det_xyz[di, :2], gt_xyz[gi, :2])
        pairs = []
        for ii in range(D.shape[0]):
            for jj in range(D.shape[1]):
                d = D[ii, jj]
                if d <= thr: pairs.append((d, di[ii], gi[jj]))
        pairs.sort(key=lambda x: x[0])
        used_det, used_gt = set(), set()
        for d, i_det, j_gt in pairs:
            if i_det in used_det or j_gt in used_gt: continue
            used_det.add(i_det); used_gt.add(j_gt)
            keep[i_det] = 1
            cls_tgt[i_det] = int(gt_labels[j_gt])
    return keep, cls_tgt

# ----------------------------- Dataset (edge-pending) -------------------

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

        gt = ref.get("gt", {}) or {}
        gt_boxes = self._safe_stack(gt.get("boxes_3d", []) or [], 9)
        gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64)

        if len(self.ignore_classes) > 0:
            keep_idx_c = np.array([i for i in range(center_labels.size) if center_labels[i] not in self.ignore_classes], dtype=np.int64)
            keep_idx_g = np.array([i for i in range(gt_labels.size) if gt_labels[i] not in self.ignore_classes], dtype=np.int64)
        else:
            keep_idx_c = np.arange(center_labels.size, dtype=np.int64)
            keep_idx_g = np.arange(gt_labels.size, dtype=np.int64)

        keep_target = np.zeros(center_labels.shape, dtype=np.int64)
        class_target = np.full(center_labels.shape, -100, dtype=np.int64)
        if keep_idx_c.size > 0 and keep_idx_g.size > 0:
            keep_mask = np.zeros(center_labels.shape, dtype=np.int64)
            cls_mask  = np.full(center_labels.shape, -100, dtype=np.int64)
            k_det_xyz = center_xyz[keep_idx_c]; k_det_lab = center_labels[keep_idx_c]
            k_gt_xyz  = gt_boxes[keep_idx_g, :3]; k_gt_lab  = gt_labels[keep_idx_g]
            k_keep, k_cls = class_aware_greedy_match(k_det_xyz, k_det_lab, k_gt_xyz, k_gt_lab, thr=self.match_thr)
            keep_mask[keep_idx_c] = k_keep; cls_mask[keep_idx_c]  = k_cls
            keep_target = keep_mask; class_target = cls_mask

        return {
            "feats": torch.from_numpy(feats),
            "pos": torch.from_numpy(pos),
            "dt": torch.from_numpy(dt),
            "labels": torch.from_numpy(labels),
            "is_center": torch.from_numpy(is_center),
            "center_boxes": torch.from_numpy(center_boxes),
            "center_keep_tgt": torch.from_numpy(keep_target),
            "center_class_tgt": torch.from_numpy(class_target),
            "meta": {
                "scene_token": scene_token,
                "center_index": ci,
                "sample_token": ref.get("sample_token", ""),
                "sample_data_token": ref.get("sample_data_token", ""),
                "timestamp": int(ref.get("timestamp", ref.get("timestamp_us", 0))),
            }
        }

# ----------------------------- Model ------------------------------------

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
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(11, d_model)
        self.label_emb  = nn.Embedding(512, d_model)
        self.time_emb   = nn.Embedding(5, d_model)
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                       batch_first=False, activation="gelu")
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
            C = self.class_head.out_features; D = self.keep_head.in_features
            return (torch.empty(0, device=device),
                    torch.empty(0, C, device=device),
                    torch.empty(0, D, device=device))
        x = self.input_proj(feats) \
          + self.label_emb(labels.clamp(min=0, max=self.label_emb.num_embeddings-1)) \
          + self.time_emb((dt + 2).clamp(0, 4))
        x = x.unsqueeze(1)  # [S,1,D]
        for layer, rb in zip(self.enc_layers, self.rel_biases):
            bias = rb(pos, dt)  # [S,S]
            x = layer(x, src_mask=bias)
        x = self.norm(x).squeeze(1)  # [S,D]
        keep_logits = self.keep_head(x).squeeze(-1)
        class_logits= self.class_head(x)
        return keep_logits, class_logits, x

# ----------------------------- Training / Eval --------------------------

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
    ce = F.cross_entropy(class_logits_c[pos_mask], center_class_tgt[pos_mask]) if pos_mask.any() else keep_logits_c.sum()*0.0
    loss = bce + ce
    return loss, {"loss_keep": float(bce.detach().item()), "loss_cls": float(ce.detach().item())}

@torch.no_grad()
def evaluate(model, loader, device, keep_thr=0.5):
    model.eval()
    tot = {"tp":0.0, "fp":0.0, "fn":0.0}
    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"])
        center_idx = (b["is_center"] == 1).nonzero(as_tuple=False).squeeze(-1)
        if center_idx.numel() == 0: continue
        keep = (torch.sigmoid(keep_logits[center_idx]) > keep_thr).long()
        tgt_keep = b["center_keep_tgt"].long()
        tot["tp"] += float(((keep == 1) & (tgt_keep == 1)).sum().item())
        tot["fp"] += float(((keep == 1) & (tgt_keep == 0)).sum().item())
        tot["fn"] += float(((keep == 0) & (tgt_keep == 1)).sum().item())
    # 聚合所有进程
    tot = all_reduce_dict(tot, op=dist.ReduceOp.SUM) if dist.is_initialized() else tot
    P = tot["tp"] / (tot["tp"] + tot["fp"]) if (tot["tp"] + tot["fp"])>0 else 0.0
    R = tot["tp"] / (tot["tp"] + tot["fn"]) if (tot["tp"] + tot["fn"])>0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    return {"P":P, "R":R, "F1":F1, **tot}

# ----------------------------- Full-structure writeback -----------------

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

@torch.no_grad()
def predict_shard(model, loader, device, keep_thr, shard_flat_path, shard_repl_path):
    """各 rank 写自己的 shard 文件（flat 与 repl 映射）。"""
    model.eval()
    flat_outputs = []
    det_replacements: Dict[str, Dict[str, Any]] = {}
    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"])
        center_idx = (b["is_center"] == 1).nonzero(as_tuple=False).squeeze(-1)

        if center_idx.numel() == 0:
            boxes_kept = np.zeros((0,9), np.float32); labels_kept=[]; scores_kept=[]
        else:
            keep_scores = torch.sigmoid(keep_logits[center_idx])
            pred_keep = keep_scores > keep_thr
            pred_labels = class_logits[center_idx].argmax(dim=-1)
            boxes = b["center_boxes"].cpu().numpy()
            labels = pred_labels.cpu().numpy()
            scores = keep_scores.cpu().numpy()
            mask = pred_keep.cpu().numpy()
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

# ----------------------------- Split & Loader ---------------------------

def split_train_val(scenes: Dict[str, List[Dict[str, Any]]], ratio=0.8, seed=42):
    sc_ids = list(scenes.keys())
    rng = random.Random(seed); rng.shuffle(sc_ids)
    n_train = max(1, int(len(sc_ids)*ratio))
    train_ids = set(sc_ids[:n_train]); val_ids = set(sc_ids[n_train:])
    train_s = {k:v for k,v in scenes.items() if k in train_ids}
    val_s   = {k:v for k,v in scenes.items() if k in val_ids}
    return train_s, val_s

def make_loader(scenes, batch_size=1, shuffle=True, sampler=None, **ds_kwargs):
    ds = ClipDataset(scenes, **ds_kwargs)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False if sampler is not None else shuffle,
        sampler=sampler, num_workers=0, collate_fn=lambda x: x[0]
    )
    return ds, loader

# ----------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--match_thr", type=float, default=2.0)
    ap.add_argument("--keep_thr", type=float, default=0.5)
    ap.add_argument("--ignore_classes", type=str, default="-1")
    ap.add_argument("--max_per_frame", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    ddp, rank, world_size, local_rank = ddp_init()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed, rank)

    if is_main_process():
        print(f"[DDP] distributed={ddp} world_size={world_size} device={device}")

    # 1) Load
    if is_main_process(): print("[1/6] Loading samples...")
    root, samples = load_root_and_samples(args.input_json)
    scenes_all = group_by_scene(samples)
    train_scenes, test_scenes = split_train_val(scenes_all, ratio=0.8, seed=args.seed)
    if is_main_process():
        print(f"Scenes: train={len(train_scenes)}  test={len(test_scenes)}")

    ignore = set(int(x.strip()) for x in args.ignore_classes.split(",") if x.strip())

    # 2) Loaders with DistributedSampler
    if is_main_process(): print("[2/6] Building loaders (DDP, edge pending)...")
    train_sampler = DistributedSampler(ClipDataset(train_scenes, frame_radius=2, max_per_frame=args.max_per_frame,
                                                   match_thr=args.match_thr, ignore_classes=ignore),
                                       shuffle=True) if ddp else None
    test_sampler  = DistributedSampler(ClipDataset(test_scenes,  frame_radius=2, max_per_frame=args.max_per_frame,
                                                   match_thr=args.match_thr, ignore_classes=ignore),
                                       shuffle=False) if ddp else None
    # 为了不重复构建两次 Dataset，我们用 make_loader 再创建（注意 sampler 传入）
    train_ds, train_loader = make_loader(train_scenes, sampler=train_sampler,
                                         frame_radius=2, max_per_frame=args.max_per_frame,
                                         match_thr=args.match_thr, ignore_classes=ignore)
    test_ds,  test_loader  = make_loader(test_scenes,  sampler=test_sampler,
                                         frame_radius=2, max_per_frame=args.max_per_frame,
                                         match_thr=args.match_thr, ignore_classes=ignore)
    if is_main_process():
        print(f"Train clips (all frames): {len(train_ds)} | Test clips (all frames): {len(test_ds)}")

    # 3) Model
    if is_main_process(): print("[3/6] Building model...")
    model = TransformerEncoderRel(d_model=args.d_model, nhead=args.nhead,
                                  num_layers=args.num_layers, dim_feedforward=4*args.d_model,
                                  num_classes=args.num_classes).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    output_device=local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    best_f1, best_path = -1.0, os.path.join(args.out_dir, "best.pt")

    # 4) Train
    if is_main_process(): print("[4/6] Training...")
    for epoch in range(1, args.epochs+1):
        if ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        running = {"loss":0.0, "keep":0.0, "cls":0.0}
        iters = 0
        for batch in train_loader:
            iters += 1
            b = to_device(batch, device)
            keep_logits, class_logits, _ = model(b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"])
            loss, loss_dict = compute_loss(keep_logits, class_logits, b["is_center"],
                                           b["center_keep_tgt"], b["center_class_tgt"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running["loss"] += float(loss.item())
            running["keep"] += float(loss_dict["loss_keep"])
            running["cls"]  += float(loss_dict["loss_cls"])

        # 仅 rank0 打日志
        if is_main_process() and iters > 0:
            print(f"Epoch {epoch} | loss={running['loss']/iters:.4f} keep={running['keep']/iters:.4f} cls={running['cls']/iters:.4f}")

        # Eval
        metrics = evaluate(model.module if isinstance(model, DDP) else model,
                           test_loader, device, keep_thr=args.keep_thr)
        if is_main_process():
            print(f"[E{epoch}] P={metrics['P']:.3f} R={metrics['R']:.3f} F1={metrics['F1']:.3f} "
                  f"(tp={int(metrics['tp'])}, fp={int(metrics['fp'])}, fn={int(metrics['fn'])})")
            if metrics["F1"] > best_f1:
                best_f1 = metrics["F1"]
                torch.save({"model": (model.module if isinstance(model, DDP) else model).state_dict(),
                            "cfg":vars(args)}, best_path)
                print(f"  -> new best saved to {best_path}")

    barrier()

    # 5) Inference (each rank -> shard)
    if is_main_process(): print("[5/6] Inference (DDP shards) ...")
    # 加载 best
    if is_main_process() and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
    barrier()  # 确保权重已加载
    # 若不是 rank0，也从磁盘加载
    if (not is_main_process()) and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])

    # 重新构造 test_loader 的 sampler，以确保推理完整覆盖
    if ddp and isinstance(test_loader.sampler, DistributedSampler):
        test_loader.sampler.set_epoch(999)

    shard_flat = os.path.join(args.out_dir, f"pred_test.rank{dist.get_rank() if ddp else 0}.json")
    shard_repl = os.path.join(args.out_dir, f"repl.rank{dist.get_rank() if ddp else 0}.json")
    predict_shard(model.module if isinstance(model, DDP) else model,
                  test_loader, device, args.keep_thr, shard_flat, shard_repl)
    barrier()

    # rank0 合并
    if is_main_process():
        pred_path_flat = os.path.join(args.out_dir, "pred_test.json")
        pred_path_full = os.path.join(args.out_dir, "pred_test_full.json")

        # 合并 flat
        flat_all = []
        for r in range(world_size):
            path = os.path.join(args.out_dir, f"pred_test.rank{r}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    flat_all.extend(json.load(f))
        with open(pred_path_flat, "w") as f:
            json.dump(flat_all, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote flat predictions to: {pred_path_flat}")

        # 合并 repl 并写结构保留版
        repl_all: Dict[str, Dict[str, Any]] = {}
        for r in range(world_size):
            path = os.path.join(args.out_dir, f"repl.rank{r}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    repl_all.update(json.load(f))
        root_copy = copy.deepcopy(root)
        apply_det_replacements(root_copy, repl_all)
        with open(pred_path_full, "w") as f:
            json.dump(root_copy, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote full-structure predictions to: {pred_path_full}")

    barrier()
    if is_main_process():
        print("[6/6] Done.")

if __name__ == "__main__":
    main()
