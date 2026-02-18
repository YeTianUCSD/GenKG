#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatio-Temporal Proposal Refinement (Transformer, relative pos/time bias)

What it does
------------
- Reads a single nested JSON file that contains samples with "det" and "gt".
- Groups samples by scene_token, sorted by timestamp.
- For each center frame t, builds a 5-frame clip (t-2..t+2) *within the same scene*.
- Uses ALL detections (unfiltered) from all 5 frames as tokens.
- Transformer Encoder with learnable relative (dx,dy,dz,dt) attention bias.
- Trains only on CENTER frame tokens: keep (binary) + class (multiclass) heads.
- At inference, keeps tokens with sigmoid(keep) > threshold and outputs
  the final boxes + labels for the center frame, preserving other JSON fields.

Assumptions
-----------
- det.boxes_3d is [x,y,z,dx,dy,dz,yaw,vx,vy] (IS-Fusion outputs)
- gt has boxes_3d and labels_3d (int), optionally names/attrs/velocity ignored here.
- JSON may be nested (e.g., scenes -> ... -> {det,gt,...}). We'll recursively find all samples.
- Each frame typically has ~200 dets; 5 frames -> up to ~1000 tokens per clip.

Usage
-----
python /home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_det_transformer_draft.py \
  --input_json /home/code/3Ddetection/IS-Fusion/GenKG/data/sorted_by_scene_ISFUSIONandGTattr.json \
  --out_dir /home/code/3Ddetection/IS-Fusion/GenKG/code/model \
  --epochs 8 \
  --match_thr 2.0 \
  --keep_thr 0.5 \
  --ignore_classes -1

Outputs
-------
- {out_dir}/best.pt           : best model checkpoint on val split
- {out_dir}/pred_test.json    : test predictions (JSON with det filtered by keep_thr)
- Training logs printed to stdout
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Utilities -----------------------------

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
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    # Normalize
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R

def make_T(translation: List[float], quat_wxyz: List[float]) -> np.ndarray:
    """Create 4x4 SE(3) from translation and quaternion [w,x,y,z]."""
    R = quat_to_rot(quat_wxyz)
    t = np.array(translation, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T

def T_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of 4x4 SE(3)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N,1), dtype=pts.dtype)])
    out = (T @ homo.T).T[:, :3]
    return out

def rotate_yaw(T: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate yaw (around z) by applying T's rotation to unit heading vectors."""
    R = T[:3, :3]
    v = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)  # Nx3
    v2 = (R @ v.T).T  # Nx3
    return np.arctan2(v2[:,1], v2[:,0])

def euclid2(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean in XY (N x 2 vs M x 2). Returns N x M."""
    a = a_xy[:, None, :]   # N x 1 x 2
    b = b_xy[None, :, :]   # 1 x M x 2
    d = np.linalg.norm(a - b, axis=2)
    return d

# ----------------------------- JSON parsing -----------------------------

def iter_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    """Recursively yield dicts that look like samples (have det & gt)."""
    if isinstance(obj, dict):
        det, gt = obj.get("det"), obj.get("gt")
        if isinstance(det, dict) and isinstance(gt, dict) and isinstance(det.get("boxes_3d"), list):
            yield obj
        for v in obj.values():
            yield from iter_samples(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_samples(it)

def load_all_samples(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        root = json.load(f)
    samples = list(iter_samples(root))
    # Keep only samples that have scene_token and timestamp
    samples = [s for s in samples if "scene_token" in s and ("timestamp" in s or "timestamp_us" in s)]
    # Normalize timestamp key
    for s in samples:
        if "timestamp" in s:
            s["_ts"] = int(s["timestamp"])
        else:
            s["_ts"] = int(s["timestamp_us"])
    return samples

def group_by_scene(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_scene.setdefault(s["scene_token"], []).append(s)
    # Sort each scene by timestamp
    for k in by_scene:
        by_scene[k].sort(key=lambda x: x["_ts"])
    return by_scene

# ----------------------------- Matching (labels+center distance) --------

def class_aware_greedy_match(det_xyz: np.ndarray, det_labels: np.ndarray,
                             gt_xyz: np.ndarray, gt_labels: np.ndarray,
                             thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform class-aware one-to-one greedy matching by XY center distance.
    Returns:
      keep_target: (N_det,) 0/1
      class_target: (N_det,) int label for positives, -100 for negatives
    """
    N = det_xyz.shape[0]
    keep = np.zeros((N,), dtype=np.int64)
    cls_tgt = np.full((N,), -100, dtype=np.int64)  # -100 -> ignore in CE

    if gt_xyz.size == 0 or det_xyz.size == 0:
        return keep, cls_tgt

    # Build per-class indices
    classes = set(det_labels.tolist()) | set(gt_labels.tolist())
    for c in classes:
        di = np.where(det_labels == c)[0]
        gi = np.where(gt_labels == c)[0]
        if di.size == 0 or gi.size == 0:
            continue
        D = euclid2(det_xyz[di, :2], gt_xyz[gi, :2])  # |di| x |gi|
        # Take pairs within threshold
        pairs: List[Tuple[float, int, int]] = []
        for ii in range(D.shape[0]):
            for jj in range(D.shape[1]):
                d = D[ii, jj]
                if d <= thr:
                    pairs.append((d, di[ii], gi[jj]))
        # Greedy from small distance
        pairs.sort(key=lambda x: x[0])
        used_det, used_gt = set(), set()
        for d, i_det, j_gt in pairs:
            if i_det in used_det or j_gt in used_gt:
                continue
            used_det.add(i_det); used_gt.add(j_gt)
            keep[i_det] = 1
            cls_tgt[i_det] = int(gt_labels[np.where(gi == j_gt)[0][0]])  # same class 'c'
    return keep, cls_tgt

# ----------------------------- Dataset (clips) ---------------------------

class ClipDataset(torch.utils.data.Dataset):
    """
    Each item is a 5-frame clip (t-2..t+2) within the same scene.
    Tokens are all dets from all 5 frames, transformed into the center frame coords.
    Loss is computed only for center frame tokens.
    """

    def __init__(self,
                 scenes: Dict[str, List[Dict[str, Any]]],
                 frame_radius: int = 2,
                 max_per_frame: Optional[int] = None,
                 match_thr: float = 2.0,
                 ignore_classes: Optional[Set[int]] = None,
                 use_cuda: bool = False):
        self.scenes = scenes
        self.frame_radius = frame_radius
        self.max_per_frame = max_per_frame
        self.match_thr = match_thr
        self.ignore_classes = set(ignore_classes or [])
        self.items: List[Tuple[str, int]] = []  # (scene_token, center_index)
        for sc, frames in scenes.items():
            # Use only frames with full window
            for i in range(frame_radius, len(frames) - frame_radius):
                self.items.append((sc, i))
        self.use_cuda = use_cuda

    def __len__(self):
        return len(self.items)

    def _get_T_k2ref(self, ref: Dict[str, Any], k: Dict[str, Any]) -> np.ndarray:
        """Compute T_k->ref: lidar_k -> lidar_ref."""
        Tl2e_ref = make_T(ref["lidar2ego"]["translation"], ref["lidar2ego"]["rotation"])
        Te2g_ref = make_T(ref["ego2global"]["translation"], ref["ego2global"]["rotation"])
        Tl2e_k   = make_T(k["lidar2ego"]["translation"],   k["lidar2ego"]["rotation"])
        Te2g_k   = make_T(k["ego2global"]["translation"],  k["ego2global"]["rotation"])
        T = np.linalg.multi_dot([
            T_inv(Tl2e_ref),
            T_inv(Te2g_ref),
            Te2g_k,
            Tl2e_k
        ])
        return T

    @staticmethod
    def _safe_stack(xs: List[List[float]], exp_dim: int) -> np.ndarray:
        arr = np.array(xs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, exp_dim)
        return arr

    def __getitem__(self, idx: int):
        scene_token, ci = self.items[idx]
        frames = self.scenes[scene_token]
        ref = frames[ci]

        # Collect tokens across 5 frames -> features + meta
        feats_list: List[np.ndarray] = []
        pos_list: List[np.ndarray] = []     # absolute positions (x,y,z) in ref coords
        dt_list: List[int] = []             # time offset [-2..2]
        label_list: List[np.ndarray] = []   # int labels
        is_center_list: List[np.ndarray] = []
        box_raw_list: List[np.ndarray] = [] # for output at inference

        for rel in range(-self.frame_radius, self.frame_radius + 1):
            k = frames[ci + rel]
            T_k2ref = self._get_T_k2ref(ref, k)

            det = k.get("det", {}) or {}
            boxes = det.get("boxes_3d", []) or []
            scores = det.get("scores_3d", []) or []
            labels = det.get("labels_3d", []) or []

            boxes = self._safe_stack(boxes, 9)
            scores = np.array(scores, dtype=np.float32).reshape(-1)
            labels = np.array(labels, dtype=np.int64).reshape(-1)

            # Optional truncation: take top-K by score
            if self.max_per_frame is not None and boxes.shape[0] > self.max_per_frame:
                idxs = np.argsort(-scores)[:self.max_per_frame]
                boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]

            # Transform xyz + yaw to ref
            xyz = boxes[:, :3]
            yaw = boxes[:, 6]
            xyz_ref = transform_points(T_k2ref, xyz)
            yaw_ref = rotate_yaw(T_k2ref, yaw)

            # Build per-token features
            dxdy_dz = boxes[:, 3:6]
            vxy = boxes[:, 7:9]
            cos_sin = np.stack([np.cos(yaw_ref), np.sin(yaw_ref)], axis=1)
            feat = np.concatenate([xyz_ref, dxdy_dz, cos_sin, vxy, scores[:, None]], axis=1)  # 3+3+2+2+1=11

            feats_list.append(feat.astype(np.float32))
            pos_list.append(xyz_ref.astype(np.float32))
            dt_list.extend([rel] * feat.shape[0])
            label_list.append(labels.astype(np.int64))
            is_center_list.append(np.full((feat.shape[0],), int(rel == 0), dtype=np.int64))

            # For inference output (only center tokens matter)
            if rel == 0:
                box_raw_list.append(boxes.astype(np.float32))  # keep original boxes for t

        feats = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 11), np.float32)
        pos   = np.concatenate(pos_list,   axis=0) if pos_list   else np.zeros((0, 3),  np.float32)
        labels= np.concatenate(label_list, axis=0) if label_list else np.zeros((0,),   np.int64)
        is_center = np.concatenate(is_center_list, axis=0) if is_center_list else np.zeros((0,), np.int64)
        dt = np.array(dt_list, dtype=np.int64)

        # Supervision (only for center tokens)
        det_center_mask = (is_center == 1)
        center_boxes = (box_raw_list[0] if box_raw_list else np.zeros((0,9), np.float32))
        center_xyz = center_boxes[:, :3]
        center_labels = labels[det_center_mask]

        gt = ref.get("gt", {}) or {}
        gt_boxes = self._safe_stack(gt.get("boxes_3d", []) or [], 9)
        gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64)

        # remove ignored classes
        if len(self.ignore_classes) > 0:
            keep_idx_c = np.array([i for i in range(center_labels.size) if center_labels[i] not in self.ignore_classes], dtype=np.int64)
            keep_idx_g = np.array([i for i in range(gt_labels.size) if gt_labels[i] not in self.ignore_classes], dtype=np.int64)
        else:
            keep_idx_c = np.arange(center_labels.size, dtype=np.int64)
            keep_idx_g = np.arange(gt_labels.size, dtype=np.int64)

        # Create targets: keep (0/1) and class target for positives
        keep_target = np.zeros(center_labels.shape, dtype=np.int64)
        class_target = np.full(center_labels.shape, -100, dtype=np.int64)

        if keep_idx_c.size > 0 and keep_idx_g.size > 0:
            keep_mask = np.zeros(center_labels.shape, dtype=np.int64)
            cls_mask  = np.full(center_labels.shape, -100, dtype=np.int64)

            k_det_xyz = center_xyz[keep_idx_c]
            k_det_lab = center_labels[keep_idx_c]
            k_gt_xyz  = gt_boxes[keep_idx_g, :3]
            k_gt_lab  = gt_labels[keep_idx_g]

            k_keep, k_cls = class_aware_greedy_match(
                k_det_xyz, k_det_lab, k_gt_xyz, k_gt_lab, thr=self.match_thr
            )
            # scatter back
            keep_mask[keep_idx_c] = k_keep
            # For positives, use their original class (or gt class). Here we use gt class.
            cls_mask[keep_idx_c]  = k_cls

            keep_target = keep_mask
            class_target = cls_mask

        # Package tensors
        sample = {
            "feats": torch.from_numpy(feats),    # [T, 11]
            "pos": torch.from_numpy(pos),        # [T, 3]
            "dt": torch.from_numpy(dt),          # [T]
            "labels": torch.from_numpy(labels),  # [T]
            "is_center": torch.from_numpy(is_center),     # [T]
            "center_boxes": torch.from_numpy(center_boxes),  # [N0, 9] for output
            "center_keep_tgt": torch.from_numpy(keep_target),   # [N0]
            "center_class_tgt": torch.from_numpy(class_target),  # [N0] (-100 ignored)
            "meta": {
                "scene_token": scene_token,
                "center_index": ci,
                "sample_token": ref.get("sample_token", ""),
                "timestamp": int(ref.get("timestamp", ref.get("timestamp_us", 0)))
            }
        }
        return sample

# ----------------------------- Model ------------------------------------

class RelBias(nn.Module):
    """Learnable pairwise bias from (dx,dy,dz,dt) -> scalar."""
    def __init__(self, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, pos: torch.Tensor, dt: torch.Tensor):
        """
        pos: [S, 3] absolute positions in ref
        dt : [S]     time offsets in {-2,-1,0,1,2}
        returns: [S, S] bias matrix
        """
        S = pos.size(0)
        # pairwise relative
        dpos = pos[:, None, :] - pos[None, :, :]   # [S, S, 3]
        dtime = dt[:, None] - dt[None, :]          # [S, S]
        f = torch.cat([dpos, dtime.unsqueeze(-1).to(pos.dtype)], dim=-1)  # [S,S,4]
        b = self.mlp(f).squeeze(-1)  # [S, S]
        return b  # add to attention logits

class TransformerEncoderRel(nn.Module):
    """Transformer Encoder with additive relative bias per layer."""
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(11, d_model)     # project numeric features
        self.label_emb  = nn.Embedding(512, d_model) # label id embedding (assuming <=512)
        self.time_emb   = nn.Embedding(5, d_model)   # t in {-2..2} -> {0..4}
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                       batch_first=False, activation="gelu")
            for _ in range(num_layers)
        ])
        self.rel_biases = nn.ModuleList([RelBias(hidden=32) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        # Heads (applied to all tokens; we only use center tokens for loss/inference)
        self.keep_head  = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, feats, labels, dt, pos, is_center):
        """
        feats: [S, 11]
        labels: [S]  (int)
        dt: [S] in {-2..2}
        pos: [S, 3]
        is_center: [S] (0/1)
        Returns:
            keep_logits_all [S], class_logits_all [S, C], token_feats [S, D]
        """
        S = feats.size(0)
        # absolute enc
        x = self.input_proj(feats) \
            + self.label_emb(labels.clamp(min=0, max=self.label_emb.num_embeddings-1)) \
            + self.time_emb((dt + 2).clamp(0, 4))
        x = x.unsqueeze(1)  # [S, 1, D] because torch Transformer expects [S, N, D] (N=batch=1)

        for layer, rb in zip(self.enc_layers, self.rel_biases):
            # Build per-sample attention bias [S,S]
            bias = rb(pos, dt)  # [S, S]
            # torch Transformer uses attn_mask added to logits where negative large masks out.
            # We just add bias directly by using "attn_mask" argument.
            x = layer(x, src_mask=bias)

        x = self.norm(x)  # [S, 1, D]
        x = x.squeeze(1)  # [S, D]

        keep_logits = self.keep_head(x).squeeze(-1)  # [S]
        class_logits= self.class_head(x)             # [S, C]
        return keep_logits, class_logits, x

# ----------------------------- Training / Eval --------------------------

def compute_loss(keep_logits, class_logits, is_center, center_keep_tgt, center_class_tgt):
    """
    Only tokens with is_center==1 participate in loss.
    keep: BCEWithLogits over all center tokens
    class: CE over positive center tokens (target!=-100)
    """
    center_idx = (is_center == 1).nonzero(as_tuple=False).squeeze(-1)
    keep_logits_c = keep_logits[center_idx]
    class_logits_c= class_logits[center_idx]
    # Targets are aligned with center tokens in dataset order
    bce = F.binary_cross_entropy_with_logits(
        keep_logits_c, center_keep_tgt.float()
    )
    # Class loss only on positives
    pos_mask = (center_keep_tgt == 1)
    if pos_mask.any():
        ce = F.cross_entropy(class_logits_c[pos_mask], center_class_tgt[pos_mask])
    else:
        ce = torch.tensor(0.0, device=keep_logits.device)
    loss = bce + ce
    return loss, {"loss_keep": bce.detach().item(), "loss_cls": ce.detach().item()}

@torch.no_grad()
def evaluate(model, loader, device, num_classes, keep_thr=0.5):
    model.eval()
    tot = {"tp":0, "fp":0, "fn":0}
    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(
            b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"]
        )
        # select center tokens
        center_idx = (b["is_center"] == 1).nonzero(as_tuple=False).squeeze(-1)
        keep = (torch.sigmoid(keep_logits[center_idx]) > keep_thr).long()
        pred_cls = class_logits[center_idx].argmax(dim=-1).long()

        # targets
        tgt_keep = b["center_keep_tgt"].long()
        # Count TP/FP/FN by keep only (class ignored here)
        tp = int(((keep == 1) & (tgt_keep == 1)).sum().item())
        fp = int(((keep == 1) & (tgt_keep == 0)).sum().item())
        fn = int(((keep == 0) & (tgt_keep == 1)).sum().item())
        tot["tp"] += tp; tot["fp"] += fp; tot["fn"] += fn
    P = tot["tp"] / (tot["tp"] + tot["fp"]) if (tot["tp"] + tot["fp"])>0 else 0.0
    R = tot["tp"] / (tot["tp"] + tot["fn"]) if (tot["tp"] + tot["fn"])>0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    return {"P":P, "R":R, "F1":F1, **tot}

@torch.no_grad()
def predict_and_dump(model, loader, device, out_path, keep_thr=0.5, num_classes=10):
    """
    Write a JSON with the same structure but 'det' filtered for center frames only.
    For each center frame item, we keep boxes whose keep_score>thr and write predicted labels.
    """
    model.eval()
    outputs = []
    for batch in loader:
        b = to_device(batch, device)
        keep_logits, class_logits, _ = model(
            b["feats"], b["labels"], b["dt"], b["pos"], b["is_center"]
        )
        center_idx = (b["is_center"] == 1).nonzero(as_tuple=False).squeeze(-1)

        keep_scores = torch.sigmoid(keep_logits[center_idx])
        pred_keep = keep_scores > keep_thr
        pred_labels = class_logits[center_idx].argmax(dim=-1)

        boxes = b["center_boxes"].cpu().numpy()
        labels = pred_labels.cpu().numpy()
        scores = keep_scores.cpu().numpy()

        boxes_kept = boxes[pred_keep.cpu().numpy()]
        labels_kept= labels[pred_keep.cpu().numpy()].tolist()
        scores_kept= scores[pred_keep.cpu().numpy()].tolist()

        meta = b["meta"]
        outputs.append({
            "scene_token": meta["scene_token"],
            "sample_token": meta["sample_token"],
            "timestamp": int(meta["timestamp"]),
            "det": {
                "boxes_3d": boxes_kept.tolist(),
                "labels_3d": labels_kept,
                "scores_3d": scores_kept
            }
        })
    with open(out_path, "w") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote predictions to: {out_path}")

# ----------------------------- Split & Loader ---------------------------

def split_train_val(scenes: Dict[str, List[Dict[str, Any]]], ratio=0.8, seed=42):
    """Random 8:2 split by scenes to avoid leakage across clips."""
    sc_ids = list(scenes.keys())
    rng = random.Random(seed)
    rng.shuffle(sc_ids)
    n_train = max(1, int(len(sc_ids)*ratio))
    train_ids = set(sc_ids[:n_train])
    val_ids   = set(sc_ids[n_train:])
    train_s = {k:v for k,v in scenes.items() if k in train_ids}
    val_s   = {k:v for k,v in scenes.items() if k in val_ids}
    return train_s, val_s

def make_loader(scenes, batch_size=1, shuffle=True, **ds_kwargs):
    ds = ClipDataset(scenes, **ds_kwargs)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=0, collate_fn=lambda x: x[0])
    return ds, loader

# ----------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--match_thr", type=float, default=2.0, help="center distance (m) for GT matching")
    ap.add_argument("--keep_thr", type=float, default=0.5, help="threshold at inference")
    ap.add_argument("--ignore_classes", type=str, default="-1", help="comma-separated list, e.g. '-1'")
    ap.add_argument("--max_per_frame", type=int, default=None, help="optional cap per frame (e.g., 200)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/6] Loading samples...")
    samples = load_all_samples(args.input_json)
    scenes_all = group_by_scene(samples)
    train_scenes, test_scenes = split_train_val(scenes_all, ratio=0.8, seed=args.seed)
    print(f"Scenes: train={len(train_scenes)}  test={len(test_scenes)}")

    ignore = set(int(x.strip()) for x in args.ignore_classes.split(",") if x.strip())

    print("[2/6] Building loaders...")
    train_ds, train_loader = make_loader(train_scenes, shuffle=True,
                                         frame_radius=2, max_per_frame=args.max_per_frame,
                                         match_thr=args.match_thr, ignore_classes=ignore)
    test_ds,  test_loader  = make_loader(test_scenes,  shuffle=False,
                                         frame_radius=2, max_per_frame=args.max_per_frame,
                                         match_thr=args.match_thr, ignore_classes=ignore)
    print(f"Train clips: {len(train_ds)} | Test clips: {len(test_ds)}")

    print("[3/6] Building model...")
    model = TransformerEncoderRel(d_model=args.d_model, nhead=args.nhead,
                                  num_layers=args.num_layers, dim_feedforward=4*args.d_model,
                                  num_classes=args.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_f1, best_path = -1.0, os.path.join(args.out_dir, "best.pt")

    print("[4/6] Training...")
    for epoch in range(1, args.epochs+1):
        model.train()
        running = {"loss":0.0, "keep":0.0, "cls":0.0}
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            running["loss"] += loss.item()
            running["keep"] += loss_dict["loss_keep"]
            running["cls"]  += loss_dict["loss_cls"]
            if it % 50 == 0:
                print(f"Epoch {epoch} iter {it}/{len(train_loader)} | "
                      f"loss={running['loss']/it:.4f} keep={running['keep']/it:.4f} cls={running['cls']/it:.4f}")

        # simple eval (keep-based PR)
        metrics = evaluate(model, test_loader, device, args.num_classes, keep_thr=args.keep_thr)
        print(f"[E{epoch}] P={metrics['P']:.3f} R={metrics['R']:.3f} F1={metrics['F1']:.3f} "
              f"(tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']})")
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            torch.save({"model":model.state_dict(), "cfg":vars(args)}, best_path)
            print(f"  -> new best saved to {best_path}")

    print("[5/6] Inference on test split with best checkpoint...")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    pred_path = os.path.join(args.out_dir, "pred_test.json")
    predict_and_dump(model, test_loader, device, pred_path, keep_thr=args.keep_thr, num_classes=args.num_classes)

    print("[6/6] Done.")

if __name__ == "__main__":
    main()
