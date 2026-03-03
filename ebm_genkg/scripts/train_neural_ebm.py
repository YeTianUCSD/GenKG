#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Neural EBM (Transformer unary + MLP pair) and export checkpoint for model/ebm.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


PAIR_FEATURE_NAMES: List[str] = [
    "same_label",
    "close",
    "overlap",
    "abs_dt_diff",
    "both_warp",
    "either_warp",
    "score_min",
    "speed_diff",
    "same_dt",
]


def _load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml.") from e
        obj = yaml.safe_load(txt)
    else:
        obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("config root must be dict")
    return obj


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    y = (y_true.reshape(-1) > 0).astype(np.int64)
    p = (probs.reshape(-1) >= float(thr)).astype(np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    P = _safe_div(tp, tp + fp)
    R = _safe_div(tp, tp + fn)
    F1 = _safe_div(2.0 * P * R, P + R) if (P + R) > 0 else 0.0
    beta2 = 4.0
    F2 = _safe_div((1.0 + beta2) * P * R, beta2 * P + R) if (beta2 * P + R) > 0 else 0.0
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    return {"threshold": float(thr), "P": P, "R": R, "F1": F1, "F2": F2, "acc": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _split_indices_by_scene(scene_idx: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    scene_idx = np.asarray(scene_idx, dtype=np.int64).reshape(-1)
    uniq = np.unique(scene_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_val = int(round(len(uniq) * float(val_ratio)))
    n_val = max(1, min(len(uniq) - 1, n_val)) if len(uniq) > 1 else 0
    val_set = set(perm[:n_val].tolist())
    is_val = np.array([int(s) in val_set for s in scene_idx.tolist()], dtype=bool)
    va = np.nonzero(is_val)[0]
    tr = np.nonzero(~is_val)[0]
    if tr.size == 0:
        tr = np.arange(scene_idx.shape[0], dtype=np.int64)
        va = np.zeros((0,), dtype=np.int64)
    return tr.astype(np.int64), va.astype(np.int64)


def _build_frame_map(scene_idx: np.ndarray, frame_idx: np.ndarray, row_idx: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    out: Dict[Tuple[int, int], List[int]] = {}
    for i in row_idx.tolist():
        k = (int(scene_idx[i]), int(frame_idx[i]))
        out.setdefault(k, []).append(int(i))
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def _predict_probs_by_rows(
    model: "NeuralEBMTransformer",
    X: np.ndarray,
    frame_map: Dict[Tuple[int, int], np.ndarray],
    rows_ref: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    probs = np.zeros((X.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for fk, rows in frame_map.items():
            _ = fk
            if rows.size == 0:
                continue
            xf = torch.from_numpy(X[rows]).to(device=device, dtype=torch.float32).unsqueeze(0)
            lg, _, _ = model.forward_unary(xf)
            pr = torch.sigmoid(lg.squeeze(0)).detach().cpu().numpy().astype(np.float32, copy=False)
            probs[rows] = pr
    return probs[rows_ref].astype(np.float32, copy=False)


def _pair_overlap_min(xi: float, yi: float, dxi: float, dyi: float, xj: float, yj: float, dxj: float, dyj: float) -> float:
    if dxi <= 1e-6 or dyi <= 1e-6 or dxj <= 1e-6 or dyj <= 1e-6:
        return 0.0
    li, ri = xi - 0.5 * dxi, xi + 0.5 * dxi
    bi, ti = yi - 0.5 * dyi, yi + 0.5 * dyi
    lj, rj = xj - 0.5 * dxj, xj + 0.5 * dxj
    bj, tj = yj - 0.5 * dyj, yj + 0.5 * dyj
    iw = max(0.0, min(ri, rj) - max(li, lj))
    ih = max(0.0, min(ti, tj) - max(bi, bj))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ai = dxi * dyi
    aj = dxj * dyj
    return float(inter / max(min(ai, aj), 1e-6))


def _build_pair_graph(
    X_raw_f: np.ndarray,
    y_unique_f: np.ndarray,
    idx: Dict[str, int],
    pair_radius: float,
    pair_max_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(X_raw_f.shape[0])
    if n <= 1:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    r = float(max(pair_radius, 1e-6))
    cell = r
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(n):
        li = int(X_raw_f[i, idx["label"]])
        x, y = float(X_raw_f[i, idx["x"]]), float(X_raw_f[i, idx["y"]])
        ix = int(np.floor(x / cell))
        iy = int(np.floor(y / cell))
        buckets.setdefault((li, ix, iy), []).append(int(i))

    feats: List[np.ndarray] = []
    pairs: List[Tuple[int, int]] = []
    labels: List[float] = []
    for i in range(n):
        li = int(X_raw_f[i, idx["label"]])
        x1, y1 = float(X_raw_f[i, idx["x"]]), float(X_raw_f[i, idx["y"]])
        dx1 = abs(float(X_raw_f[i, idx["dx"]]))
        dy1 = abs(float(X_raw_f[i, idx["dy"]]))
        dt1 = int(round(float(X_raw_f[i, idx["from_dt"]])))
        wi = 1.0 if float(X_raw_f[i, idx["is_warp"]]) > 0.5 else 0.0
        sc1 = float(X_raw_f[i, idx["score"]])
        sp1 = float(X_raw_f[i, idx["speed"]])
        ix = int(np.floor(x1 / cell))
        iy = int(np.floor(y1 / cell))
        neigh: List[Tuple[float, int]] = []
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                for j in buckets.get((li, ix + dx_cell, iy + dy_cell), []):
                    if int(j) <= i:
                        continue
                    x2, y2 = float(X_raw_f[j, idx["x"]]), float(X_raw_f[j, idx["y"]])
                    d = float(np.hypot(x1 - x2, y1 - y2))
                    if d <= r:
                        neigh.append((d, int(j)))
        neigh.sort(key=lambda x: x[0])
        if pair_max_neighbors > 0:
            neigh = neigh[: int(pair_max_neighbors)]
        for d, j in neigh:
            close = float(np.exp(-d / r))
            dx2 = abs(float(X_raw_f[j, idx["dx"]]))
            dy2 = abs(float(X_raw_f[j, idx["dy"]]))
            dt2 = int(round(float(X_raw_f[j, idx["from_dt"]])))
            wj = 1.0 if float(X_raw_f[j, idx["is_warp"]]) > 0.5 else 0.0
            sc2 = float(X_raw_f[j, idx["score"]])
            x2, y2 = float(X_raw_f[j, idx["x"]]), float(X_raw_f[j, idx["y"]])
            sp2 = float(X_raw_f[j, idx["speed"]])
            ov = _pair_overlap_min(x1, y1, dx1, dy1, x2, y2, dx2, dy2)
            feats.append(np.asarray([
                1.0,
                close,
                ov,
                float(min(abs(dt1 - dt2), 8)) / 8.0,
                wi * wj,
                1.0 if (wi + wj) > 0.0 else 0.0,
                min(sc1, sc2),
                float(min(abs(sp1 - sp2), 10.0)) / 10.0,
                1.0 if dt1 == dt2 else 0.0,
            ], dtype=np.float32))
            pairs.append((i, j))
            labels.append(1.0 if (y_unique_f[i] > 0.5 and y_unique_f[j] > 0.5) else 0.0)
    if len(feats) == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return np.stack(feats, axis=0), np.asarray(pairs, dtype=np.int64), np.asarray(labels, dtype=np.float32)


class NeuralEBMTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
        num_classes: int,
        num_attrs: int,
        pair_dim: int = 9,
        pair_hidden: int = 64,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.unary_head = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, num_classes) if num_classes > 0 else None
        self.attr_head = nn.Linear(d_model, num_attrs) if num_attrs > 0 else None
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, pair_hidden),
            nn.ReLU(),
            nn.Linear(pair_hidden, 1),
        )

    def forward_unary(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = self.encoder(self.in_proj(x))
        keep = self.unary_head(h).squeeze(-1)
        cls = self.class_head(h) if self.class_head is not None else None
        attr = self.attr_head(h) if self.attr_head is not None else None
        return keep, cls, attr

    def forward_pair(self, pair_feat: torch.Tensor) -> torch.Tensor:
        return self.pair_mlp(pair_feat).squeeze(-1)


def _build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Neural EBM (Transformer unary + MLP pair).")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--in_npz", type=str, default=cfg.get("in_npz"), required=("in_npz" not in cfg))
    p.add_argument("--out_ckpt", type=str, default=cfg.get("out_ckpt"), required=("out_ckpt" not in cfg))
    p.add_argument("--out_summary", type=str, default=cfg.get("out_summary"))
    p.add_argument("--val_ratio", type=float, default=float(cfg.get("val_ratio", 0.2)))
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)))
    p.add_argument("--split_by", type=str, default=str(cfg.get("split_by", "scene")), choices=["scene", "random"])
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=bool(cfg.get("normalize", True)))
    p.add_argument("--max_train_rows", type=int, default=cfg.get("max_train_rows"))
    p.add_argument("--device", type=str, default=str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    p.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 6)))
    p.add_argument("--frame_batch_size", type=int, default=int(cfg.get("frame_batch_size", 24)))
    p.add_argument("--max_cands_per_frame", type=int, default=int(cfg.get("max_cands_per_frame", 196)))
    p.add_argument("--lr", type=float, default=float(cfg.get("lr", 2e-4)))
    p.add_argument("--weight_decay", type=float, default=float(cfg.get("weight_decay", 1e-5)))
    p.add_argument("--d_model", type=int, default=int(cfg.get("d_model", 128)))
    p.add_argument("--nhead", type=int, default=int(cfg.get("nhead", 4)))
    p.add_argument("--num_layers", type=int, default=int(cfg.get("num_layers", 2)))
    p.add_argument("--dim_ff", type=int, default=int(cfg.get("dim_ff", 256)))
    p.add_argument("--dropout", type=float, default=float(cfg.get("dropout", 0.1)))
    p.add_argument("--pair_hidden", type=int, default=int(cfg.get("pair_hidden", 64)))
    p.add_argument("--pair_radius", type=float, default=float(cfg.get("pair_radius", 2.5)))
    p.add_argument("--pair_max_neighbors", type=int, default=int(cfg.get("pair_max_neighbors", 12)))
    p.add_argument("--lambda_pair", type=float, default=float(cfg.get("lambda_pair", 0.25)))
    p.add_argument("--lambda_cls", type=float, default=float(cfg.get("lambda_cls", 0.25)))
    p.add_argument("--lambda_attr", type=float, default=float(cfg.get("lambda_attr", 0.10)))
    p.add_argument("--class_pos_only", action=argparse.BooleanOptionalAction, default=bool(cfg.get("class_pos_only", True)))
    p.add_argument("--attr_pos_only", action=argparse.BooleanOptionalAction, default=bool(cfg.get("attr_pos_only", True)))
    p.add_argument("--threshold_metric", type=str, default=str(cfg.get("threshold_metric", "f2")), choices=["f1", "f2"])
    p.add_argument("--threshold_grid", type=str, default=str(cfg.get("threshold_grid", "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")))
    p.add_argument("--target_recall", type=float, default=cfg.get("target_recall"))
    p.add_argument("--target_select_metric", type=str, default=str(cfg.get("target_select_metric", "precision")), choices=["precision", "f1"])
    return p


def main() -> None:
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default=None)
    a0, _ = p0.parse_known_args()
    cfg = _load_cfg(a0.config)
    args = _build_parser(cfg).parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    in_npz = os.path.abspath(args.in_npz)
    out_ckpt = os.path.abspath(args.out_ckpt)
    out_summary = os.path.abspath(args.out_summary) if args.out_summary else (os.path.splitext(out_ckpt)[0] + ".summary.json")
    out_state = os.path.splitext(out_ckpt)[0] + ".pt"
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)

    print("[1/6] Loading dataset...")
    data = np.load(in_npz, allow_pickle=True)
    X_raw = np.asarray(data["X"], dtype=np.float32)
    y_hard = np.asarray(data["y_keep"], dtype=np.float32).reshape(-1)
    y_soft = np.asarray(data["y_keep_soft"], dtype=np.float32).reshape(-1) if "y_keep_soft" in data.files else y_hard.copy()
    y = y_soft
    y_unique = np.asarray(data["y_unique"], dtype=np.float32).reshape(-1) if "y_unique" in data.files else (
        np.asarray(data["cand_is_gt_best"], dtype=np.float32).reshape(-1) if "cand_is_gt_best" in data.files else np.zeros_like(y_hard)
    )
    y_cls = np.asarray(data["y_cls"], dtype=np.int64).reshape(-1) if "y_cls" in data.files else np.full((y_hard.shape[0],), -100, dtype=np.int64)
    y_attr = np.asarray(data["y_attr"], dtype=np.int64).reshape(-1) if "y_attr" in data.files else np.full((y_hard.shape[0],), -1, dtype=np.int64)
    scene_idx = np.asarray(data["scene_idx"], dtype=np.int64).reshape(-1)
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64).reshape(-1)
    feat_names = [str(x) for x in data["feature_names"].tolist()]
    N, D = X_raw.shape
    print(f"  rows={N} dim={D} pos_rate_hard={float((y_hard > 0.5).mean()):.6f}")

    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    required = ["label", "x", "y", "dx", "dy", "from_dt", "is_warp", "score", "speed"]
    miss = [k for k in required if k not in name_to_idx]
    if miss:
        raise ValueError(f"missing required feature(s): {miss}")

    print("[2/6] Train/val split...")
    if args.split_by == "scene":
        tr_idx, va_idx = _split_indices_by_scene(scene_idx, val_ratio=float(args.val_ratio), seed=int(args.seed))
    else:
        rng = np.random.default_rng(int(args.seed))
        order = np.arange(N, dtype=np.int64)
        rng.shuffle(order)
        n_val = int(round(N * float(args.val_ratio)))
        n_val = max(1, min(N - 1, n_val)) if N > 1 else 0
        va_idx = order[:n_val]
        tr_idx = order[n_val:]
    if args.max_train_rows is not None and tr_idx.size > int(args.max_train_rows):
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(tr_idx)
        tr_idx = tr_idx[: int(args.max_train_rows)]
    print(f"  train_rows={tr_idx.size} val_rows={va_idx.size}")

    X = X_raw.astype(np.float32, copy=True)
    if bool(args.normalize):
        mu = X[tr_idx].mean(axis=0)
        std = X[tr_idx].std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        X = (X - mu) / std
    else:
        mu = np.zeros((D,), dtype=np.float32)
        std = np.ones((D,), dtype=np.float32)

    print("[3/6] Build frame index...")
    tr_frame_map = _build_frame_map(scene_idx, frame_idx, tr_idx)
    va_frame_map = _build_frame_map(scene_idx, frame_idx, va_idx) if va_idx.size > 0 else {}
    tr_keys = list(tr_frame_map.keys())
    print(f"  train_frames={len(tr_keys)} val_frames={len(va_frame_map)}")

    cls_ids = sorted(np.unique(y_cls[y_cls >= 0]).astype(np.int64).tolist())
    attr_ids = sorted(np.unique(y_attr[y_attr >= 0]).astype(np.int64).tolist())
    cls_to_idx = {int(v): i for i, v in enumerate(cls_ids)}
    attr_to_idx = {int(v): i for i, v in enumerate(attr_ids)}

    device = torch.device(str(args.device))
    model = NeuralEBMTransformer(
        in_dim=int(D),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        dim_ff=int(args.dim_ff),
        dropout=float(args.dropout),
        num_classes=int(len(cls_ids)),
        num_attrs=int(len(attr_ids)),
        pair_dim=9,
        pair_hidden=int(args.pair_hidden),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    pos = int((y_hard[tr_idx] > 0.5).sum())
    neg = int((y_hard[tr_idx] <= 0.5).sum())
    pos_w = float(neg / max(1, pos))

    print("[4/6] Training...")
    rng = np.random.default_rng(int(args.seed))
    epochs = int(max(1, args.epochs))
    fbs = int(max(1, args.frame_batch_size))
    max_cands = int(max(8, args.max_cands_per_frame))
    lambda_pair = float(args.lambda_pair)
    lambda_cls = float(args.lambda_cls)
    lambda_attr = float(args.lambda_attr)

    for ep in range(1, epochs + 1):
        random.shuffle(tr_keys)
        ep_loss = 0.0
        ep_keep = 0.0
        ep_pair = 0.0
        ep_cls = 0.0
        ep_attr = 0.0
        steps = 0

        for s in range(0, len(tr_keys), fbs):
            keys_batch = tr_keys[s:s + fbs]
            loss_acc = torch.tensor(0.0, dtype=torch.float32, device=device)
            n_frames = 0

            for fk in keys_batch:
                rows = tr_frame_map[fk]
                if rows.size == 0:
                    continue
                if rows.size > max_cands:
                    score_col = int(name_to_idx["score"])
                    ord_idx = np.argsort(-X_raw[rows, score_col])[:max_cands]
                    rows = rows[ord_idx]
                xf = torch.from_numpy(X[rows]).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,Nf,D]
                yf = torch.from_numpy(y[rows]).to(device=device, dtype=torch.float32)
                yhf = torch.from_numpy(y_hard[rows]).to(device=device, dtype=torch.float32)
                yuf = torch.from_numpy(y_unique[rows]).to(device=device, dtype=torch.float32)
                ycl = y_cls[rows]
                yat = y_attr[rows]

                keep_logits, cls_logits, attr_logits = model.forward_unary(xf)
                keep_logits = keep_logits.squeeze(0)
                w = torch.ones_like(yf)
                w[yhf > 0.5] = float(pos_w)
                keep_loss = F.binary_cross_entropy_with_logits(keep_logits, yf, weight=w)

                pair_feat_np, _, pair_lab_np = _build_pair_graph(
                    X_raw[rows],
                    y_unique_f=yuf.detach().cpu().numpy(),
                    idx=name_to_idx,
                    pair_radius=float(args.pair_radius),
                    pair_max_neighbors=int(args.pair_max_neighbors),
                )
                if pair_feat_np.shape[0] > 0:
                    pf = torch.from_numpy(pair_feat_np).to(device=device, dtype=torch.float32)
                    pl = torch.from_numpy(pair_lab_np).to(device=device, dtype=torch.float32)
                    pair_logits = model.forward_pair(pf)
                    pair_loss = F.binary_cross_entropy_with_logits(pair_logits, pl)
                else:
                    pair_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

                cls_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if cls_logits is not None and len(cls_ids) > 1:
                    cls_mask = (ycl >= 0)
                    if bool(args.class_pos_only):
                        cls_mask = cls_mask & (yhf.detach().cpu().numpy() > 0.5)
                    if np.any(cls_mask):
                        tgt = np.asarray([cls_to_idx.get(int(v), -1) for v in ycl[cls_mask].tolist()], dtype=np.int64)
                        valid = tgt >= 0
                        if np.any(valid):
                            lg = cls_logits.squeeze(0)[torch.from_numpy(np.nonzero(cls_mask)[0][valid]).to(device=device, dtype=torch.long)]
                            tt = torch.from_numpy(tgt[valid]).to(device=device, dtype=torch.long)
                            cls_loss = F.cross_entropy(lg, tt)

                attr_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if attr_logits is not None and len(attr_ids) > 1:
                    attr_mask = (yat >= 0)
                    if bool(args.attr_pos_only):
                        attr_mask = attr_mask & (yhf.detach().cpu().numpy() > 0.5)
                    if np.any(attr_mask):
                        tgt = np.asarray([attr_to_idx.get(int(v), -1) for v in yat[attr_mask].tolist()], dtype=np.int64)
                        valid = tgt >= 0
                        if np.any(valid):
                            lg = attr_logits.squeeze(0)[torch.from_numpy(np.nonzero(attr_mask)[0][valid]).to(device=device, dtype=torch.long)]
                            tt = torch.from_numpy(tgt[valid]).to(device=device, dtype=torch.long)
                            attr_loss = F.cross_entropy(lg, tt)

                loss_f = keep_loss + lambda_pair * pair_loss + lambda_cls * cls_loss + lambda_attr * attr_loss
                loss_acc = loss_acc + loss_f
                ep_keep += float(keep_loss.item())
                ep_pair += float(pair_loss.item())
                ep_cls += float(cls_loss.item())
                ep_attr += float(attr_loss.item())
                n_frames += 1

            if n_frames <= 0:
                continue
            loss = loss_acc / float(n_frames)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item())
            steps += 1

        with torch.no_grad():
            tr_probs = _predict_probs_by_rows(model, X, tr_frame_map, tr_idx, device)
            tr_m = _metrics_from_probs(y_hard[tr_idx], tr_probs, thr=0.5)

            va_msg = ""
            if va_idx.size > 0:
                va_probs = _predict_probs_by_rows(model, X, va_frame_map, va_idx, device)
                va_m = _metrics_from_probs(y_hard[va_idx], va_probs, thr=0.5)
                va_msg = f" val_f1@0.5={va_m['F1']:.4f} val_f2@0.5={va_m['F2']:.4f}"

            print(
                f"  epoch={ep:03d} loss={ep_loss/max(1,steps):.6f} "
                f"keep={ep_keep/max(1,steps):.6f} pair={ep_pair/max(1,steps):.6f} "
                f"cls={ep_cls/max(1,steps):.6f} attr={ep_attr/max(1,steps):.6f} "
                f"train_f1@0.5={tr_m['F1']:.4f}{va_msg}"
            )

    print("[5/6] Select threshold...")
    thr_grid = _parse_float_list(args.threshold_grid)
    if len(thr_grid) == 0:
        thr_grid = [0.5]
    metric_key = "F2" if str(args.threshold_metric).lower() == "f2" else "F1"
    target_recall = float(args.target_recall) if (args.target_recall is not None) else None
    target_select_metric = str(args.target_select_metric).lower()

    def _pick_best(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if len(metrics_list) == 0:
            return _metrics_from_probs(y_hard[tr_idx], np.zeros((tr_idx.shape[0],), dtype=np.float32), thr=0.5)
        if target_recall is None:
            return max(metrics_list, key=lambda m: float(m.get(metric_key, 0.0)))
        feas = [m for m in metrics_list if float(m.get("R", 0.0)) >= target_recall]
        if len(feas) > 0:
            if target_select_metric == "f1":
                return max(feas, key=lambda m: (float(m.get("F1", 0.0)), float(m.get("P", 0.0)), float(m.get("R", 0.0))))
            return max(feas, key=lambda m: (float(m.get("P", 0.0)), float(m.get("F1", 0.0)), float(m.get("R", 0.0))))
        return max(metrics_list, key=lambda m: (float(m.get("R", 0.0)), float(m.get("P", 0.0)), float(m.get("F1", 0.0))))

    with torch.no_grad():
        if va_idx.size > 0:
            va_probs = _predict_probs_by_rows(model, X, va_frame_map, va_idx, device)
            all_ms = [_metrics_from_probs(y_hard[va_idx], va_probs, thr=float(t)) for t in thr_grid]
            best = _pick_best(all_ms)
        else:
            tr_probs = _predict_probs_by_rows(model, X, tr_frame_map, tr_idx, device)
            all_ms = [_metrics_from_probs(y_hard[tr_idx], tr_probs, thr=float(t)) for t in thr_grid]
            best = _pick_best(all_ms)
    best_thr = float(best["threshold"])

    print("[6/6] Saving checkpoint...")
    torch.save(model.state_dict(), out_state)
    ckpt = {
        "model_type": "neural_ebm_transformer",
        "version": 1,
        "input_dim": int(D),
        "feature_names": feat_names,
        "normalize": bool(args.normalize),
        "mu": [float(x) for x in mu.tolist()],
        "std": [float(x) for x in std.tolist()],
        "best_threshold": float(best_thr),
        "threshold_metric": str(metric_key).lower(),
        "class_head": {
            "enabled": len(cls_ids) > 1,
            "class_labels": [int(v) for v in cls_ids],
        },
        "attr_head": {
            "enabled": len(attr_ids) > 1,
            "attr_ids": [int(v) for v in attr_ids],
            "attr_names": [f"attr_{int(v)}" for v in attr_ids],
        },
        "pair": {
            "enabled": True,
            "feature_names": PAIR_FEATURE_NAMES,
            "scale": 1.0,
            "radius": float(args.pair_radius),
        },
        "neural_ebm": {
            "d_model": int(args.d_model),
            "nhead": int(args.nhead),
            "num_layers": int(args.num_layers),
            "dim_ff": int(args.dim_ff),
            "dropout": float(args.dropout),
            "pair_hidden": int(args.pair_hidden),
            "torch_state_path": os.path.abspath(out_state),
        },
        "train": {
            "epochs": int(epochs),
            "frame_batch_size": int(fbs),
            "max_cands_per_frame": int(max_cands),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lambda_pair": float(lambda_pair),
            "lambda_cls": float(lambda_cls),
            "lambda_attr": float(lambda_attr),
            "class_pos_only": bool(args.class_pos_only),
            "attr_pos_only": bool(args.attr_pos_only),
            "split_by": str(args.split_by),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "device": str(device),
        },
    }
    with open(out_ckpt, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)

    summary = {
        "in_npz": in_npz,
        "out_ckpt": out_ckpt,
        "out_state": out_state,
        "rows_total": int(N),
        "rows_train": int(tr_idx.size),
        "rows_val": int(va_idx.size),
        "dim": int(D),
        "best_threshold": float(best_thr),
        "best_metrics": best,
        "threshold_metric": str(metric_key).lower(),
        "args": vars(args),
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  ckpt: {out_ckpt}")
    print(f"  state: {out_state}")
    print(f"  summary: {out_summary}")
    print(f"  best@thr={best_thr:.3f} P={best['P']:.4f} R={best['R']:.4f} F1={best['F1']:.4f} F2={best['F2']:.4f}")


if __name__ == "__main__":
    main()
