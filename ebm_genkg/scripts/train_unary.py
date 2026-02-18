#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a unary keep model (logistic regression) for EBM candidate selection.

Input:
- NPZ built by scripts/build_trainset.py
  Required keys: X [N,D], y_keep [N], scene_idx [N], feature_names [D]

Output:
- ckpt json with normalization stats + linear weights
- summary json (optional)

Example:
  python scripts/train_unary.py \
    --in_npz /path/to/trainset.npz \
    --out_ckpt /path/to/unary_ckpt.json \
    --out_summary /path/to/unary_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        txt = f.read()

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml. Install it or use JSON config.") from e
        obj = yaml.safe_load(txt)
    else:
        obj = json.loads(txt)

    if not isinstance(obj, dict):
        raise ValueError(f"Config root must be dict, got: {type(obj)}")
    return obj


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
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
    F1 = _safe_div(2 * P * R, P + R) if (P + R) > 0 else 0.0
    acc = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "threshold": float(thr),
        "P": float(P),
        "R": float(R),
        "F1": float(F1),
        "acc": float(acc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _bce_loss_and_grad(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    sample_weight: np.ndarray,
    l2: float,
) -> Tuple[float, np.ndarray, float]:
    logits = X @ w + b
    probs = _sigmoid(logits)
    probs = np.clip(probs, 1e-8, 1.0 - 1e-8)

    # weighted BCE
    loss_vec = -(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
    loss = float(np.mean(sample_weight * loss_vec) + 0.5 * l2 * np.sum(w * w))

    diff = (probs - y) * sample_weight
    grad_w = (X.T @ diff) / max(1, X.shape[0]) + l2 * w
    grad_b = float(np.mean(diff))
    return loss, grad_w, grad_b


def _split_indices_by_scene(
    scene_idx: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scene_idx = np.asarray(scene_idx, dtype=np.int64).reshape(-1)
    uniq = np.unique(scene_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)

    n_val_scene = int(round(len(uniq) * float(val_ratio)))
    n_val_scene = max(1, min(len(uniq) - 1, n_val_scene)) if len(uniq) > 1 else 0

    val_scenes = set(perm[:n_val_scene].tolist())
    val_mask = np.array([int(s) in val_scenes for s in scene_idx.tolist()], dtype=bool)
    tr_mask = ~val_mask

    tr_idx = np.nonzero(tr_mask)[0]
    va_idx = np.nonzero(val_mask)[0]

    if tr_idx.size == 0:
        tr_idx = np.arange(scene_idx.shape[0], dtype=np.int64)
        va_idx = np.zeros((0,), dtype=np.int64)

    return tr_idx.astype(np.int64), va_idx.astype(np.int64)


def _build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train unary logistic model for EBM keep scores.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")

    p.add_argument("--in_npz", type=str, default=cfg.get("in_npz"), required=("in_npz" not in cfg))
    p.add_argument("--out_ckpt", type=str, default=cfg.get("out_ckpt"), required=("out_ckpt" not in cfg))
    p.add_argument("--out_summary", type=str, default=cfg.get("out_summary"))

    p.add_argument("--val_ratio", type=float, default=float(cfg.get("val_ratio", 0.2)))
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)))
    p.add_argument("--split_by", type=str, default=str(cfg.get("split_by", "scene")), choices=["scene", "random"])

    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=bool(cfg.get("normalize", True)))
    else:
        g = p.add_mutually_exclusive_group()
        g.add_argument("--normalize", dest="normalize", action="store_true")
        g.add_argument("--no-normalize", dest="normalize", action="store_false")
        p.set_defaults(normalize=bool(cfg.get("normalize", True)))
    p.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 10)))
    p.add_argument("--batch_size", type=int, default=int(cfg.get("batch_size", 32768)))
    p.add_argument("--lr", type=float, default=float(cfg.get("lr", 0.05)))
    p.add_argument("--l2", type=float, default=float(cfg.get("l2", 1e-6)))

    p.add_argument("--pos_weight_mode", type=str, default=str(cfg.get("pos_weight_mode", "auto")),
                   choices=["auto", "none", "value"])
    p.add_argument("--pos_weight", type=float, default=float(cfg.get("pos_weight", 1.0)))

    p.add_argument("--threshold_grid", type=str, default=str(cfg.get("threshold_grid", "0.1,0.2,0.3,0.4,0.5,0.6")))
    p.add_argument("--max_train_rows", type=int, default=cfg.get("max_train_rows"),
                   help="Optional subsample cap for training rows.")

    return p


def main() -> None:
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default=None)
    a0, _ = p0.parse_known_args()

    cfg = _load_cfg(a0.config)
    parser = _build_parser(cfg)
    args = parser.parse_args()

    in_npz = os.path.abspath(args.in_npz)
    out_ckpt = os.path.abspath(args.out_ckpt)
    out_summary = os.path.abspath(args.out_summary) if args.out_summary else (os.path.splitext(out_ckpt)[0] + ".summary.json")
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)

    print("[1/5] Loading dataset...")
    data = np.load(in_npz, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y_keep"], dtype=np.float64).reshape(-1)
    scene_idx = np.asarray(data["scene_idx"], dtype=np.int64).reshape(-1)
    feat_names = [str(x) for x in data["feature_names"].tolist()]

    assert X.ndim == 2, f"X shape must be [N,D], got {X.shape}"
    assert y.shape[0] == X.shape[0], f"y length mismatch: {y.shape[0]} vs {X.shape[0]}"

    N, D = X.shape
    print(f"  rows={N} dim={D} pos_rate={float((y > 0).mean()):.6f}")

    # split
    print("[2/5] Train/val split...")
    rng = np.random.default_rng(int(args.seed))
    if args.split_by == "scene":
        tr_idx, va_idx = _split_indices_by_scene(scene_idx, val_ratio=float(args.val_ratio), seed=int(args.seed))
    else:
        idx = np.arange(N, dtype=np.int64)
        rng.shuffle(idx)
        n_val = int(round(N * float(args.val_ratio)))
        n_val = max(1, min(N - 1, n_val)) if N > 1 else 0
        va_idx = idx[:n_val]
        tr_idx = idx[n_val:]

    if args.max_train_rows is not None and tr_idx.size > int(args.max_train_rows):
        rng.shuffle(tr_idx)
        tr_idx = tr_idx[: int(args.max_train_rows)]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    print(f"  train_rows={X_tr.shape[0]} val_rows={X_va.shape[0]}")

    # normalize
    if args.normalize:
        mu = X_tr.mean(axis=0)
        std = X_tr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        X_tr = (X_tr - mu) / std
        if X_va.shape[0] > 0:
            X_va = (X_va - mu) / std
    else:
        mu = np.zeros((D,), dtype=np.float64)
        std = np.ones((D,), dtype=np.float64)

    # sample weighting for imbalance
    print("[3/5] Preparing optimizer...")
    pos = int((y_tr > 0.5).sum())
    neg = int((y_tr <= 0.5).sum())
    if args.pos_weight_mode == "auto":
        pos_w = float(neg / max(1, pos))
    elif args.pos_weight_mode == "value":
        pos_w = float(args.pos_weight)
    else:
        pos_w = 1.0

    sw_tr = np.ones_like(y_tr, dtype=np.float64)
    sw_tr[y_tr > 0.5] = pos_w

    w = np.zeros((D,), dtype=np.float64)
    b = 0.0

    epochs = int(args.epochs)
    bs = max(1, int(args.batch_size))
    lr = float(args.lr)
    l2 = float(args.l2)

    print(f"  pos={pos} neg={neg} pos_weight={pos_w:.4f} epochs={epochs} batch_size={bs} lr={lr} l2={l2}")

    print("[4/5] Training...")
    for ep in range(1, epochs + 1):
        perm = rng.permutation(X_tr.shape[0])
        X_ep = X_tr[perm]
        y_ep = y_tr[perm]
        sw_ep = sw_tr[perm]

        ep_loss = 0.0
        ep_batches = 0
        for s in range(0, X_ep.shape[0], bs):
            e = min(s + bs, X_ep.shape[0])
            Xb = X_ep[s:e]
            yb = y_ep[s:e]
            sb = sw_ep[s:e]

            loss, gw, gb = _bce_loss_and_grad(Xb, yb, w, b, sb, l2=l2)
            w -= lr * gw
            b -= lr * gb

            ep_loss += loss
            ep_batches += 1

        tr_probs = _sigmoid(X_tr @ w + b)
        tr_m = _metrics_from_probs(y_tr, tr_probs, thr=0.5)

        if X_va.shape[0] > 0:
            va_probs = _sigmoid(X_va @ w + b)
            va_m = _metrics_from_probs(y_va, va_probs, thr=0.5)
            print(
                f"  epoch={ep:03d} loss={ep_loss / max(1, ep_batches):.6f} "
                f"train_f1@0.5={tr_m['F1']:.4f} val_f1@0.5={va_m['F1']:.4f}"
            )
        else:
            print(
                f"  epoch={ep:03d} loss={ep_loss / max(1, ep_batches):.6f} "
                f"train_f1@0.5={tr_m['F1']:.4f}"
            )

    # threshold selection on val
    thr_grid = _parse_float_list(args.threshold_grid)
    if len(thr_grid) == 0:
        thr_grid = [0.5]

    if X_va.shape[0] > 0:
        va_probs = _sigmoid(X_va @ w + b)
        best = None
        for t in thr_grid:
            m = _metrics_from_probs(y_va, va_probs, thr=float(t))
            if best is None or m["F1"] > best["F1"]:
                best = m
        assert best is not None
        best_thr = float(best["threshold"])
        best_metrics = best
    else:
        best_thr = 0.5
        best_metrics = _metrics_from_probs(y_tr, _sigmoid(X_tr @ w + b), thr=best_thr)

    print("[5/5] Saving checkpoint...")
    ckpt = {
        "model_type": "logistic_unary",
        "version": 1,
        "input_dim": int(D),
        "feature_names": feat_names,
        "normalize": bool(args.normalize),
        "mu": [float(x) for x in mu.tolist()],
        "std": [float(x) for x in std.tolist()],
        "weights": [float(x) for x in w.tolist()],
        "bias": float(b),
        "best_threshold": float(best_thr),
        "train": {
            "epochs": epochs,
            "batch_size": bs,
            "lr": lr,
            "l2": l2,
            "pos_weight_mode": str(args.pos_weight_mode),
            "pos_weight": float(pos_w),
            "seed": int(args.seed),
            "split_by": str(args.split_by),
            "val_ratio": float(args.val_ratio),
        },
    }

    with open(out_ckpt, "w") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)

    summary = {
        "in_npz": in_npz,
        "out_ckpt": out_ckpt,
        "rows_total": int(N),
        "rows_train": int(X_tr.shape[0]),
        "rows_val": int(X_va.shape[0]),
        "dim": int(D),
        "pos_rate_total": float((y > 0.5).mean()),
        "best_threshold": float(best_thr),
        "best_metrics": best_metrics,
        "threshold_grid": [float(x) for x in thr_grid],
        "args": vars(args),
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  ckpt: {out_ckpt}")
    print(f"  summary: {out_summary}")
    print(
        f"  best@thr={best_thr:.3f} "
        f"P={best_metrics['P']:.4f} R={best_metrics['R']:.4f} F1={best_metrics['F1']:.4f}"
    )


if __name__ == "__main__":
    main()
