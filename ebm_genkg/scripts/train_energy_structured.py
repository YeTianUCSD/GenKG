#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a truly learned structured EBM:
  - Unary energy: MLP on candidate features
  - Pair energy: learned linear head on pair features
  - Structured frame-level loss: margin + E(y_pos) - E(y_neg)

Checkpoint model_type:
  structured_energy_mlp
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
    idx: Dict[str, int],
    pair_radius: float,
    close_min: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(X_raw_f.shape[0])
    if n <= 1:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)
    feats: List[np.ndarray] = []
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        li = int(X_raw_f[i, idx["label"]])
        x1, y1 = float(X_raw_f[i, idx["x"]]), float(X_raw_f[i, idx["y"]])
        dx1 = abs(float(X_raw_f[i, idx["dx"]]))
        dy1 = abs(float(X_raw_f[i, idx["dy"]]))
        dt1 = int(round(float(X_raw_f[i, idx["from_dt"]])))
        wi = 1.0 if float(X_raw_f[i, idx["is_warp"]]) > 0.5 else 0.0
        sc1 = float(X_raw_f[i, idx["score"]])
        sp1 = float(X_raw_f[i, idx["speed"]])
        for j in range(i + 1, n):
            lj = int(X_raw_f[j, idx["label"]])
            if li != lj:
                continue
            x2, y2 = float(X_raw_f[j, idx["x"]]), float(X_raw_f[j, idx["y"]])
            d = float(np.hypot(x1 - x2, y1 - y2))
            close = float(np.exp(-d / max(pair_radius, 1e-6)))
            if close < close_min:
                continue
            dx2 = abs(float(X_raw_f[j, idx["dx"]]))
            dy2 = abs(float(X_raw_f[j, idx["dy"]]))
            dt2 = int(round(float(X_raw_f[j, idx["from_dt"]])))
            wj = 1.0 if float(X_raw_f[j, idx["is_warp"]]) > 0.5 else 0.0
            sc2 = float(X_raw_f[j, idx["score"]])
            sp2 = float(X_raw_f[j, idx["speed"]])
            ov = _pair_overlap_min(x1, y1, dx1, dy1, x2, y2, dx2, dy2)
            feat = np.asarray(
                [
                    1.0,
                    close,
                    ov,
                    float(min(abs(dt1 - dt2), 8)) / 8.0,
                    wi * wj,
                    1.0 if (wi + wj) > 0.0 else 0.0,
                    min(sc1, sc2),
                    float(min(abs(sp1 - sp2), 10.0)) / 10.0,
                    1.0 if dt1 == dt2 else 0.0,
                ],
                dtype=np.float32,
            )
            feats.append(feat)
            pairs.append((i, j))
    if len(feats) == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)
    return np.stack(feats, axis=0), np.asarray(pairs, dtype=np.int64)


def _greedy_negative_from_energy(unary_e: np.ndarray, pair_e: np.ndarray, pair_ij: np.ndarray) -> np.ndarray:
    n = int(unary_e.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    emat = np.zeros((n, n), dtype=np.float64)
    for k in range(pair_ij.shape[0]):
        i, j = int(pair_ij[k, 0]), int(pair_ij[k, 1])
        v = float(pair_e[k])
        emat[i, j] = v
        emat[j, i] = v
    sel = np.zeros((n,), dtype=np.float32)
    order = np.argsort(unary_e)  # lower unary energy first
    selected_idx: List[int] = []
    for i in order.tolist():
        delta = float(unary_e[i])
        if selected_idx:
            delta += float(np.sum(emat[i, np.asarray(selected_idx, dtype=np.int64)]))
        if delta < 0.0:
            sel[i] = 1.0
            selected_idx.append(int(i))
    return sel


class StructuredEBM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, pair_dim: int = 9):
        super().__init__()
        self.unary_fc1 = nn.Linear(in_dim, hidden_dim)
        self.unary_fc2 = nn.Linear(hidden_dim, 1)
        self.pair_fc = nn.Linear(pair_dim, 1)

    def unary_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.unary_fc1(x))
        return self.unary_fc2(h).squeeze(-1)

    def pair_logits(self, f: torch.Tensor) -> torch.Tensor:
        return self.pair_fc(f).squeeze(-1)


def _build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train structured end-to-end EBM.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--in_npz", type=str, default=cfg.get("in_npz"), required=("in_npz" not in cfg))
    p.add_argument("--out_ckpt", type=str, default=cfg.get("out_ckpt"), required=("out_ckpt" not in cfg))
    p.add_argument("--out_summary", type=str, default=cfg.get("out_summary"))

    p.add_argument("--val_ratio", type=float, default=float(cfg.get("val_ratio", 0.2)))
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)))
    p.add_argument("--split_by", type=str, default=str(cfg.get("split_by", "scene")), choices=["scene", "random"])
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=bool(cfg.get("normalize", True)))
    p.add_argument("--max_train_rows", type=int, default=cfg.get("max_train_rows"))
    p.add_argument("--device", type=str, default=str(cfg.get("device", "cpu")))

    p.add_argument("--hidden_dim", type=int, default=int(cfg.get("hidden_dim", 64)))
    p.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 8)))
    p.add_argument("--batch_size", type=int, default=int(cfg.get("batch_size", 32768)))
    p.add_argument("--lr", type=float, default=float(cfg.get("lr", 1e-3)))
    p.add_argument("--weight_decay", type=float, default=float(cfg.get("weight_decay", 1e-6)))

    p.add_argument("--pos_weight_mode", type=str, default=str(cfg.get("pos_weight_mode", "auto")), choices=["auto", "none", "value"])
    p.add_argument("--pos_weight", type=float, default=float(cfg.get("pos_weight", 1.0)))
    p.add_argument("--threshold_metric", type=str, default=str(cfg.get("threshold_metric", "f2")), choices=["f1", "f2"])
    p.add_argument("--threshold_grid", type=str, default=str(cfg.get("threshold_grid", "0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")))

    p.add_argument("--struct_weight", type=float, default=float(cfg.get("struct_weight", 1.0)))
    p.add_argument("--struct_margin", type=float, default=float(cfg.get("struct_margin", 1.0)))
    p.add_argument("--struct_frames_per_step", type=int, default=int(cfg.get("struct_frames_per_step", 4)))
    p.add_argument("--struct_max_cands_per_frame", type=int, default=int(cfg.get("struct_max_cands_per_frame", 64)))
    p.add_argument("--pair_radius", type=float, default=float(cfg.get("pair_radius", 2.5)))
    p.add_argument("--pair_energy_scale", type=float, default=float(cfg.get("pair_energy_scale", 1.0)))
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
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)

    print("[1/6] Loading dataset...")
    data = np.load(in_npz, allow_pickle=True)
    X_raw = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y_keep"], dtype=np.float32).reshape(-1)
    scene_idx = np.asarray(data["scene_idx"], dtype=np.int64).reshape(-1)
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64).reshape(-1)
    feat_names = [str(x) for x in data["feature_names"].tolist()]
    N, D = X_raw.shape
    print(f"  rows={N} dim={D} pos_rate={float((y > 0.5).mean()):.6f}")

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

    print("[3/6] Build frame map...")
    tr_frame_map = _build_frame_map(scene_idx, frame_idx, tr_idx)
    tr_frame_keys = list(tr_frame_map.keys())
    print(f"  train_frames={len(tr_frame_keys)}")

    device = torch.device(str(args.device))
    model = StructuredEBM(in_dim=int(D), hidden_dim=int(args.hidden_dim), pair_dim=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    pos = int((y[tr_idx] > 0.5).sum())
    neg = int((y[tr_idx] <= 0.5).sum())
    if args.pos_weight_mode == "auto":
        pos_w = float(neg / max(1, pos))
    elif args.pos_weight_mode == "value":
        pos_w = float(args.pos_weight)
    else:
        pos_w = 1.0
    pos_weight_t = torch.tensor(float(pos_w), dtype=torch.float32, device=device)

    X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(y).to(device=device, dtype=torch.float32)

    print("[4/6] Training...")
    bs = int(max(1024, args.batch_size))
    epochs = int(max(1, args.epochs))
    struct_weight = float(args.struct_weight)
    struct_margin = float(args.struct_margin)
    struct_frames_per_step = int(max(0, args.struct_frames_per_step))
    struct_max_cands = int(max(4, args.struct_max_cands_per_frame))
    pair_radius = float(args.pair_radius)
    pair_scale = float(args.pair_energy_scale)

    rng = np.random.default_rng(int(args.seed))
    tr_rows = tr_idx.copy()
    steps_per_epoch = int(math.ceil(max(1, tr_rows.size) / float(bs)))

    for ep in range(1, epochs + 1):
        rng.shuffle(tr_rows)
        loss_sum = 0.0
        bce_sum = 0.0
        struct_sum = 0.0

        for st in range(steps_per_epoch):
            s = st * bs
            e = min(s + bs, tr_rows.size)
            rb = tr_rows[s:e]
            xb = X_t[rb]
            yb = y_t[rb]

            logits_b = model.unary_logits(xb)
            bce_loss = F.binary_cross_entropy_with_logits(logits_b, yb, pos_weight=pos_weight_t)

            struct_losses: List[torch.Tensor] = []
            for _ in range(struct_frames_per_step):
                if not tr_frame_keys:
                    break
                fk = tr_frame_keys[int(rng.integers(0, len(tr_frame_keys)))]
                rows = tr_frame_map[fk]
                if rows.size <= 1:
                    continue
                if rows.size > struct_max_cands:
                    rows = rng.choice(rows, size=struct_max_cands, replace=False)
                rows = np.asarray(rows, dtype=np.int64)

                xf = X_t[rows]
                yf = y_t[rows]
                logits_f = model.unary_logits(xf)
                unary_e = -logits_f

                X_raw_f = X_raw[rows]
                pair_feat_np, pair_ij_np = _build_pair_graph(X_raw_f, name_to_idx, pair_radius=pair_radius, close_min=0.10)
                if pair_feat_np.shape[0] > 0:
                    pf = torch.from_numpy(pair_feat_np).to(device=device, dtype=torch.float32)
                    p_logits = model.pair_logits(pf)
                    p_prob = torch.sigmoid(p_logits)
                    pair_e = pair_scale * (0.5 - p_prob)
                    pair_e_np = pair_e.detach().cpu().numpy()
                else:
                    pair_e = torch.zeros((0,), dtype=torch.float32, device=device)
                    pair_e_np = np.zeros((0,), dtype=np.float32)

                y_neg_np = _greedy_negative_from_energy(
                    unary_e.detach().cpu().numpy(),
                    pair_e_np,
                    pair_ij_np,
                )
                y_neg = torch.from_numpy(y_neg_np).to(device=device, dtype=torch.float32)

                e_pos = torch.sum(yf * unary_e)
                e_neg = torch.sum(y_neg * unary_e)
                if pair_e.shape[0] > 0:
                    ii = torch.from_numpy(pair_ij_np[:, 0]).to(device=device, dtype=torch.long)
                    jj = torch.from_numpy(pair_ij_np[:, 1]).to(device=device, dtype=torch.long)
                    e_pos = e_pos + torch.sum(yf[ii] * yf[jj] * pair_e)
                    e_neg = e_neg + torch.sum(y_neg[ii] * y_neg[jj] * pair_e)

                frame_loss = F.relu(torch.tensor(struct_margin, dtype=torch.float32, device=device) + e_pos - e_neg)
                struct_losses.append(frame_loss)

            if struct_losses:
                struct_loss = torch.stack(struct_losses).mean()
            else:
                struct_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

            loss = bce_loss + struct_weight * struct_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            bce_sum += float(bce_loss.item())
            struct_sum += float(struct_loss.item())

        with torch.no_grad():
            tr_logits = model.unary_logits(X_t[tr_idx]).detach().cpu().numpy()
            tr_probs = 1.0 / (1.0 + np.exp(-tr_logits))
            tr_m = _metrics_from_probs(y[tr_idx], tr_probs, thr=0.5)
            if va_idx.size > 0:
                va_logits = model.unary_logits(X_t[va_idx]).detach().cpu().numpy()
                va_probs = 1.0 / (1.0 + np.exp(-va_logits))
                va_m = _metrics_from_probs(y[va_idx], va_probs, thr=0.5)
                print(
                    f"  epoch={ep:03d} loss={loss_sum/max(1,steps_per_epoch):.6f} "
                    f"bce={bce_sum/max(1,steps_per_epoch):.6f} struct={struct_sum/max(1,steps_per_epoch):.6f} "
                    f"train_f1@0.5={tr_m['F1']:.4f} val_f1@0.5={va_m['F1']:.4f} val_f2@0.5={va_m['F2']:.4f}"
                )
            else:
                print(
                    f"  epoch={ep:03d} loss={loss_sum/max(1,steps_per_epoch):.6f} "
                    f"bce={bce_sum/max(1,steps_per_epoch):.6f} struct={struct_sum/max(1,steps_per_epoch):.6f} "
                    f"train_f1@0.5={tr_m['F1']:.4f}"
                )

    print("[5/6] Select threshold...")
    thr_grid = _parse_float_list(args.threshold_grid)
    if len(thr_grid) == 0:
        thr_grid = [0.5]
    metric_key = "F2" if str(args.threshold_metric).lower() == "f2" else "F1"
    with torch.no_grad():
        if va_idx.size > 0:
            va_probs = torch.sigmoid(model.unary_logits(X_t[va_idx])).detach().cpu().numpy()
            best = None
            for t in thr_grid:
                m = _metrics_from_probs(y[va_idx], va_probs, thr=float(t))
                if best is None or float(m.get(metric_key, 0.0)) > float(best.get(metric_key, 0.0)):
                    best = m
            assert best is not None
        else:
            tr_probs = torch.sigmoid(model.unary_logits(X_t[tr_idx])).detach().cpu().numpy()
            best = _metrics_from_probs(y[tr_idx], tr_probs, thr=0.5)
    best_thr = float(best["threshold"])

    print("[6/6] Saving checkpoint...")
    w1 = model.unary_fc1.weight.detach().cpu().numpy().T  # [D,H]
    b1 = model.unary_fc1.bias.detach().cpu().numpy()
    w2 = model.unary_fc2.weight.detach().cpu().numpy().reshape(-1)  # [H]
    b2 = float(model.unary_fc2.bias.detach().cpu().numpy().reshape(-1)[0])
    pw = model.pair_fc.weight.detach().cpu().numpy().reshape(-1)
    pb = float(model.pair_fc.bias.detach().cpu().numpy().reshape(-1)[0])

    ckpt = {
        "model_type": "structured_energy_mlp",
        "version": 1,
        "input_dim": int(D),
        "feature_names": feat_names,
        "normalize": bool(args.normalize),
        "mu": [float(x) for x in mu.tolist()],
        "std": [float(x) for x in std.tolist()],
        "unary_mlp": {
            "hidden_dim": int(args.hidden_dim),
            "activation": "relu",
            "w1": [[float(v) for v in row] for row in w1.tolist()],
            "b1": [float(v) for v in b1.tolist()],
            "w2": [float(v) for v in w2.tolist()],
            "b2": float(b2),
        },
        "pair": {
            "enabled": True,
            "feature_names": PAIR_FEATURE_NAMES,
            "weights": [float(v) for v in pw.tolist()],
            "bias": float(pb),
            "scale": float(pair_scale),
            "radius": float(pair_radius),
        },
        "best_threshold": float(best_thr),
        "threshold_metric": str(metric_key).lower(),
        "train": {
            "epochs": int(epochs),
            "batch_size": int(bs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "pos_weight_mode": str(args.pos_weight_mode),
            "pos_weight": float(pos_w),
            "struct_weight": float(struct_weight),
            "struct_margin": float(struct_margin),
            "struct_frames_per_step": int(struct_frames_per_step),
            "struct_max_cands_per_frame": int(struct_max_cands),
            "pair_radius": float(pair_radius),
            "pair_energy_scale": float(pair_scale),
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
        "rows_total": int(N),
        "rows_train": int(tr_idx.size),
        "rows_val": int(va_idx.size),
        "dim": int(D),
        "best_threshold": float(best_thr),
        "best_metrics": best,
        "threshold_metric": str(metric_key).lower(),
        "pair_feature_dim": int(len(PAIR_FEATURE_NAMES)),
        "args": vars(args),
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  ckpt: {out_ckpt}")
    print(f"  summary: {out_summary}")
    print(
        f"  best@thr={best_thr:.3f} "
        f"P={best['P']:.4f} R={best['R']:.4f} F1={best['F1']:.4f} F2={best['F2']:.4f}"
    )


if __name__ == "__main__":
    main()

