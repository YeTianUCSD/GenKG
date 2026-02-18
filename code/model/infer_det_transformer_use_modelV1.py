#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只做推理 / 过滤的版本：
- 加载训练好的 best.pt
- 使用给定的 JSON（含 det + gt + 位姿）
- 对每一帧的 det 进行“keep + 改类别”预测
- 把新的 det 写回原始 JSON 结构，输出到指定目录

示例用法：

python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/model/infer_det_transformer_use_model.py \
  --ckpt /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV4/search_res/best_overall.pt \
  --input_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV4 \
  --keep_thr 0.5 \
  --max_per_frame 200
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

# ============================= Utils =============================

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

        return {
            "feats": torch.from_numpy(feats),
            "pos": torch.from_numpy(pos),
            "dt": torch.from_numpy(dt),
            "labels": torch.from_numpy(labels),
            "is_center": torch.from_numpy(is_center),
            "center_boxes": torch.from_numpy(center_boxes),
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
            C = self.class_head.out_features
            return (
                torch.empty(0, device=device),
                torch.empty(0, C, device=device),
                torch.empty(0, self.keep_head.in_features, device=device),
            )
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

# ============================= Inference =============================

@torch.no_grad()
def run_inference(model, loader, device, keep_thr: float):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练好的 best.pt 路径")
    ap.add_argument("--input_json", required=True, help="含 det+gt 的 JSON")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--frame_radius", type=int, default=2)
    ap.add_argument("--max_per_frame", type=int, default=None)
    ap.add_argument("--match_thr", type=float, default=2.0,
                    help="仅用于 Dataset 构造，与训练保持一致即可（不用于推理标签）")
    ap.add_argument("--ignore_classes", type=str, default="-1")
    ap.add_argument("--keep_thr", type=float, default=0.5,
                    help="keep 过滤阈值（越高越严格）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Load JSON & scenes
    print("[1/4] Loading JSON & samples...")
    root, samples = load_root_and_samples(args.input_json)
    scenes_all = group_by_scene(samples)
    print(f"[INFO] Scenes: {len(scenes_all)} | Samples: {len(samples)}")

    # 2) Build Dataset & Loader
    print("[2/4] Building dataset & dataloader...")
    ignore = set(int(x.strip()) for x in args.ignore_classes.split(",") if x.strip())
    ds, loader = make_loader(
        scenes_all,
        frame_radius=args.frame_radius,
        max_per_frame=args.max_per_frame,
        match_thr=args.match_thr,
        ignore_classes=ignore,
        batch_size=1
    )
    print(f"[INFO] Clips (all frames as center): {len(ds)}")

    # 3) Load model
    print("[3/4] Loading model from ckpt...")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    num_classes = cfg.get("num_classes", 10)
    d_model     = cfg.get("d_model", 128)
    nhead       = cfg.get("nhead", 4)
    num_layers  = cfg.get("num_layers", 2)

    model = TransformerEncoderRel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=4*d_model,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] Model loaded: d_model={d_model}, nhead={nhead}, "
          f"layers={num_layers}, num_classes={num_classes}")

    # 4) Inference
    print("[4/4] Running inference and writing outputs...")
    flat_outputs, det_repl = run_inference(model, loader, device, keep_thr=args.keep_thr)

    # 4.1 写 flat 结果
    flat_path = os.path.join(args.out_dir, "pred_flat.json")
    with open(flat_path, "w") as f:
        json.dump(flat_outputs, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote flat predictions to: {flat_path}")

    # 4.2 写 structure-preserving JSON
    root_copy = copy.deepcopy(root)
    apply_det_replacements(root_copy, det_repl)
    full_path = os.path.join(args.out_dir, "pred_full.json")
    with open(full_path, "w") as f:
        json.dump(root_copy, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote full-structure predictions to: {full_path}")

    print("[DONE] Inference completed.")

if __name__ == "__main__":
    main()
