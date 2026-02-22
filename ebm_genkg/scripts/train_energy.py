#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a learned energy unary model (logistic) for EBM candidate selection.

This is similar to train_unary.py but:
1) checkpoint model_type is "logistic_energy"
2) default threshold selection metric is F2 (recall-favored)
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
    with open(path, "r", encoding="utf-8") as f:
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


def _parse_str_list(s: str) -> List[str]:
    out: List[str] = []
    for x in (s or "").split(","):
        x = str(x).strip()
        if x:
            out.append(x)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.float32:
        dt = np.float32
    else:
        dt = np.float64
    x = x.astype(dt, copy=False)
    out = np.empty_like(x, dtype=dt)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _softmax_logits(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.float32:
        dt = np.float32
    else:
        dt = np.float64
    x = x.astype(dt, copy=False)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    m = np.max(x, axis=1, keepdims=True)
    ex = np.exp(x - m)
    den = np.sum(ex, axis=1, keepdims=True)
    return ex / np.clip(den, 1e-12, None)


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
    beta2 = 4.0
    F2 = _safe_div((1.0 + beta2) * P * R, beta2 * P + R) if (beta2 * P + R) > 0 else 0.0
    acc = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "threshold": float(thr),
        "P": float(P),
        "R": float(R),
        "F1": float(F1),
        "F2": float(F2),
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

    loss_vec = -(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
    loss = float(np.mean(sample_weight * loss_vec) + 0.5 * l2 * np.sum(w * w))

    diff = (probs - y) * sample_weight
    grad_w = (X.T @ diff) / max(1, X.shape[0]) + l2 * w
    grad_b = float(np.mean(diff))
    return loss, grad_w, grad_b


def _multitask_loss_and_grad(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    sample_weight: np.ndarray,
    l2: float,
    gain_target: Optional[np.ndarray],
    gain_weight: Optional[np.ndarray],
    dup_weight: Optional[np.ndarray],
    unique_target: Optional[np.ndarray],
    unique_weight: Optional[np.ndarray],
    lambda_gain: float,
    lambda_dup: float,
    lambda_unique: float,
) -> Tuple[float, float, float, float, np.ndarray, float]:
    logits = X @ w + b
    probs = _sigmoid(logits)
    probs = np.clip(probs, 1e-8, 1.0 - 1e-8)

    loss_bce_vec = -(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
    loss_bce = float(np.mean(sample_weight * loss_bce_vec))

    diff = (probs - y) * sample_weight

    loss_gain = 0.0
    if (gain_target is not None) and (gain_weight is not None) and (lambda_gain > 0.0):
        gt = np.asarray(gain_target, dtype=np.float64).reshape(-1)
        gw = np.asarray(gain_weight, dtype=np.float64).reshape(-1)
        err = probs - gt
        loss_gain = float(np.mean(gw * (err ** 2)))
        diff += float(lambda_gain) * (2.0 * gw * err * probs * (1.0 - probs))

    loss_dup = 0.0
    if (dup_weight is not None) and (lambda_dup > 0.0):
        dw = np.asarray(dup_weight, dtype=np.float64).reshape(-1)
        loss_dup = float(np.mean(dw * (-np.log(1.0 - probs))))
        # d/dlogit [-log(1-p)] = p
        diff += float(lambda_dup) * (dw * probs)

    loss_unique = 0.0
    if (unique_target is not None) and (unique_weight is not None) and (lambda_unique > 0.0):
        ut = np.asarray(unique_target, dtype=np.float64).reshape(-1)
        uw = np.asarray(unique_weight, dtype=np.float64).reshape(-1)
        loss_u_vec = -(ut * np.log(probs) + (1.0 - ut) * np.log(1.0 - probs))
        loss_unique = float(np.mean(uw * loss_u_vec))
        diff += float(lambda_unique) * (uw * (probs - ut))

    loss = float(
        loss_bce
        + float(lambda_gain) * loss_gain
        + float(lambda_dup) * loss_dup
        + float(lambda_unique) * loss_unique
        + 0.5 * l2 * np.sum(w * w)
    )
    grad_w = (X.T @ diff) / max(1, X.shape[0]) + l2 * w
    grad_b = float(np.mean(diff))
    return loss, loss_bce, loss_gain, loss_unique, grad_w, grad_b


def _pairwise_rank_loss_and_grad(
    X: np.ndarray,
    w: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    sign: np.ndarray,
    pair_weight: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Logistic ranking loss on pairwise logit difference:
      L = mean(weight * log(1 + exp(-s * (li - lj))))
    where s in {-1, +1}.
    """
    if i_idx.size == 0:
        return 0.0, np.zeros_like(w, dtype=np.float64)

    Xi = X[i_idx]
    Xj = X[j_idx]
    delta = (Xi @ w) - (Xj @ w)  # bias cancels
    z = -sign * delta
    # stable log(1+exp(z))
    loss_vec = np.logaddexp(0.0, z)
    loss = float(np.mean(pair_weight * loss_vec))

    sig = 1.0 / (1.0 + np.exp(-z))  # sigmoid(z)
    grad_delta = -sign * pair_weight * sig
    grad_w = ((Xi - Xj).T @ grad_delta) / max(1, i_idx.size)
    return loss, grad_w.astype(np.float64, copy=False)


def _fit_relation_block(
    X: np.ndarray,
    y: np.ndarray,
    base_logits: np.ndarray,
    feature_names: List[str],
    sample_weight: np.ndarray,
    relation_feature_names: List[str],
    *,
    epochs: int = 6,
    lr: float = 0.05,
    l2: float = 1e-6,
) -> Optional[Dict[str, Any]]:
    """
    Fit an explicit relation-energy block on selected temporal/neighborhood features:
      rel_logit = wr^T z + br
      total_logit = base_logits + rel_logit
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    names = [str(n).strip() for n in relation_feature_names if str(n).strip() != ""]
    names = [n for n in names if n in name_to_idx]
    if len(names) == 0:
        return None

    idx = [int(name_to_idx[n]) for n in names]
    Z = X[:, idx].astype(np.float64, copy=False)

    wr = np.zeros((Z.shape[1],), dtype=np.float64)
    br = 0.0
    # Keep soft-target semantics for post blocks; do not hard-binarize here.
    yb = np.clip(np.asarray(y, dtype=np.float64).reshape(-1), 0.0, 1.0)

    for _ in range(int(max(1, epochs))):
        logits = base_logits + (Z @ wr + br)
        probs = _sigmoid(logits)
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)

        diff = (probs - yb) * sample_weight
        grad_w = (Z.T @ diff) / max(1, Z.shape[0]) + float(l2) * wr
        grad_b = float(np.mean(diff))

        wr -= float(lr) * grad_w
        br -= float(lr) * grad_b

    return {
        "enabled": True,
        "feature_names": names,
        "weights": [float(v) for v in wr.tolist()],
        "bias": float(br),
    }


def _build_context_neighbors(
    scene_idx: np.ndarray,
    frame_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius_xy: float,
    max_neighbors: int,
) -> List[np.ndarray]:
    n = int(x.shape[0])
    out: List[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in range(n)]
    if n == 0:
        return out
    r = float(max(radius_xy, 1e-6))
    r2 = r * r
    max_n = int(max(1, max_neighbors))
    keys = np.stack([scene_idx.reshape(-1), frame_idx.reshape(-1)], axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    groups: Dict[int, List[int]] = {}
    for i, g in enumerate(inv.tolist()):
        groups.setdefault(int(g), []).append(int(i))
    for idx_list in groups.values():
        idx = np.asarray(idx_list, dtype=np.int64)
        if idx.size <= 1:
            continue
        xx = x[idx].astype(np.float64, copy=False)
        yy = y[idx].astype(np.float64, copy=False)
        cell = r
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for li in range(idx.size):
            gx = int(np.floor(float(xx[li]) / cell))
            gy = int(np.floor(float(yy[li]) / cell))
            buckets.setdefault((gx, gy), []).append(int(li))
        for li in range(idx.size):
            gx = int(np.floor(float(xx[li]) / cell))
            gy = int(np.floor(float(yy[li]) / cell))
            cand_local: List[int] = []
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    cand_local.extend(buckets.get((gx + dx_cell, gy + dy_cell), []))
            if len(cand_local) == 0:
                continue
            arr_local = np.asarray(cand_local, dtype=np.int64)
            arr_local = arr_local[arr_local != li]
            if arr_local.size == 0:
                continue
            dxv = xx[arr_local] - float(xx[li])
            dyv = yy[arr_local] - float(yy[li])
            d2 = dxv * dxv + dyv * dyv
            keep = d2 <= r2
            if not np.any(keep):
                continue
            arr_local = arr_local[keep]
            d2 = d2[keep]
            if arr_local.size > max_n:
                ord_local = np.argpartition(d2, max_n - 1)[:max_n]
                arr_local = arr_local[ord_local]
            out[int(idx[li])] = idx[arr_local]
    return out


def _context_feature_matrix(
    probs: np.ndarray,
    neighbors: List[np.ndarray],
    labels: np.ndarray,
    from_dt: np.ndarray,
    is_raw: np.ndarray,
    is_warp: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    n = int(probs.shape[0])
    feat_names = [
        "context_self_prob",
        "context_nbr_prob_mean",
        "context_nbr_prob_max",
        "context_nbr_same_label_prob_mean",
        "context_nbr_warp_prob_mean",
        "context_nbr_raw_prob_mean",
        "context_nbr_abs_dt_mean",
        "context_nbr_score_mean",
    ]
    F = np.zeros((n, len(feat_names)), dtype=np.float64)
    for i in range(n):
        p_i = float(probs[i])
        F[i, 0] = p_i
        nei = neighbors[i]
        if nei.size == 0:
            continue
        p_n = probs[nei].astype(np.float64, copy=False)
        F[i, 1] = float(np.mean(p_n))
        F[i, 2] = float(np.max(p_n))
        same = (labels[nei] == labels[i])
        if np.any(same):
            F[i, 3] = float(np.mean(p_n[same]))
        else:
            F[i, 3] = 0.0
        F[i, 4] = float(np.mean(p_n * is_warp[nei]))
        F[i, 5] = float(np.mean(p_n * is_raw[nei]))
        F[i, 6] = float(np.mean(np.abs(from_dt[nei] - from_dt[i])))
        F[i, 7] = float(np.mean(scores[nei]))
    return F, feat_names


def _fit_context_block(
    *,
    y: np.ndarray,
    sample_weight: np.ndarray,
    base_logits: np.ndarray,
    scene_idx: np.ndarray,
    frame_idx: np.ndarray,
    cand_label: np.ndarray,
    from_dt: np.ndarray,
    is_raw: np.ndarray,
    is_warp: np.ndarray,
    score: np.ndarray,
    x: np.ndarray,
    y_xy: np.ndarray,
    radius_xy: float,
    max_neighbors: int,
    rounds: int,
    epochs: int,
    lr: float,
    l2: float,
) -> Optional[Dict[str, Any]]:
    n = int(y.shape[0])
    if n == 0:
        return None
    neighbors = _build_context_neighbors(
        scene_idx=scene_idx,
        frame_idx=frame_idx,
        x=x,
        y=y_xy,
        radius_xy=radius_xy,
        max_neighbors=max_neighbors,
    )
    if all(nei.size == 0 for nei in neighbors):
        return None

    logits_cur = base_logits.astype(np.float64, copy=True)
    rounds_out: List[Dict[str, Any]] = []
    feat_names_ref: Optional[List[str]] = None
    # Keep soft-target semantics for post blocks; do not hard-binarize here.
    yb = np.clip(np.asarray(y, dtype=np.float64).reshape(-1), 0.0, 1.0)
    sw = sample_weight.astype(np.float64, copy=False)
    n_inv = 1.0 / max(1, n)
    for _ in range(int(max(1, rounds))):
        probs_cur = _sigmoid(logits_cur.astype(np.float64, copy=False))
        F, feat_names = _context_feature_matrix(
            probs=probs_cur,
            neighbors=neighbors,
            labels=cand_label,
            from_dt=from_dt,
            is_raw=is_raw,
            is_warp=is_warp,
            scores=score,
        )
        if feat_names_ref is None:
            feat_names_ref = feat_names
        w = np.zeros((F.shape[1],), dtype=np.float64)
        b = 0.0
        for _ep in range(int(max(1, epochs))):
            z = logits_cur + (F @ w + b)
            p = np.clip(_sigmoid(z), 1e-8, 1.0 - 1e-8)
            diff = (p - yb) * sw
            grad_w = (F.T @ diff) * n_inv + float(l2) * w
            grad_b = float(np.mean(diff))
            w -= float(lr) * grad_w
            b -= float(lr) * grad_b
        delta = F @ w + b
        logits_cur = logits_cur + delta
        rounds_out.append({"weights": [float(v) for v in w.tolist()], "bias": float(b)})

    if feat_names_ref is None:
        return None
    return {
        "enabled": True,
        "radius_xy": float(radius_xy),
        "max_neighbors": int(max_neighbors),
        "feature_names": feat_names_ref,
        "rounds": rounds_out,
    }


def _build_context_coords(
    X_raw: np.ndarray,
    feature_names: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    required = ["x", "y", "label", "from_dt", "is_raw", "is_warp", "score"]
    for n in required:
        if n not in name_to_idx:
            return None
    return {
        "x": X_raw[:, int(name_to_idx["x"])].astype(np.float64, copy=False),
        "y": X_raw[:, int(name_to_idx["y"])].astype(np.float64, copy=False),
        "label": np.rint(X_raw[:, int(name_to_idx["label"])]).astype(np.int64, copy=False),
        "from_dt": np.rint(X_raw[:, int(name_to_idx["from_dt"])]).astype(np.int64, copy=False),
        "is_raw": np.clip(X_raw[:, int(name_to_idx["is_raw"])].astype(np.float64, copy=False), 0.0, 1.0),
        "is_warp": np.clip(X_raw[:, int(name_to_idx["is_warp"])].astype(np.float64, copy=False), 0.0, 1.0),
        "score": np.asarray(X_raw[:, int(name_to_idx["score"])], dtype=np.float64),
    }


def _apply_context_block_logits(
    logits: np.ndarray,
    context_block: Optional[Dict[str, Any]],
    scene_idx: np.ndarray,
    frame_idx: np.ndarray,
    coords: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    if context_block is None or (not bool(context_block.get("enabled", False))):
        return logits
    if coords is None:
        return logits
    rounds = context_block.get("rounds", [])
    if not isinstance(rounds, list) or len(rounds) == 0:
        return logits
    feat_names_ckpt = [str(x) for x in context_block.get("feature_names", [])]
    radius_xy = float(context_block.get("radius_xy", 1.5))
    max_neighbors = int(context_block.get("max_neighbors", 24))
    neighbors = _build_context_neighbors(
        scene_idx=scene_idx,
        frame_idx=frame_idx,
        x=coords["x"],
        y=coords["y"],
        radius_xy=radius_xy,
        max_neighbors=max_neighbors,
    )
    out = logits.astype(np.float64, copy=True)
    for rd in rounds:
        w = np.asarray(rd.get("weights", []), dtype=np.float64).reshape(-1)
        b = float(rd.get("bias", 0.0))
        F, feat_names = _context_feature_matrix(
            probs=_sigmoid(out),
            neighbors=neighbors,
            labels=coords["label"],
            from_dt=coords["from_dt"],
            is_raw=coords["is_raw"],
            is_warp=coords["is_warp"],
            scores=coords["score"],
        )
        if len(feat_names_ckpt) > 0 and feat_names_ckpt != feat_names:
            return logits
        if w.shape[0] != F.shape[1]:
            return logits
        out = out + (F @ w + b)
    return out


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
    p = argparse.ArgumentParser(description="Train learned energy logistic model for EBM keep scores.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")

    p.add_argument("--in_npz", type=str, default=cfg.get("in_npz"), required=("in_npz" not in cfg))
    p.add_argument("--out_ckpt", type=str, default=cfg.get("out_ckpt"), required=("out_ckpt" not in cfg))
    p.add_argument("--out_summary", type=str, default=cfg.get("out_summary"))
    p.add_argument(
        "--dtype",
        type=str,
        default=str(cfg.get("dtype", "float32")),
        choices=["float32", "float64"],
        help="Training numeric dtype. float32 uses less memory.",
    )

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
    p.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 12)))
    p.add_argument("--batch_size", type=int, default=int(cfg.get("batch_size", 32768)))
    p.add_argument("--lr", type=float, default=float(cfg.get("lr", 0.05)))
    p.add_argument("--l2", type=float, default=float(cfg.get("l2", 1e-6)))

    p.add_argument("--pos_weight_mode", type=str, default=str(cfg.get("pos_weight_mode", "auto")),
                   choices=["auto", "none", "value"])
    p.add_argument("--pos_weight", type=float, default=float(cfg.get("pos_weight", 1.0)))
    p.add_argument(
        "--weight_label_source",
        type=str,
        default=str(cfg.get("weight_label_source", "hard")),
        choices=["hard", "train"],
        help="Label source for pos/neg balancing sample weight.",
    )
    p.add_argument(
        "--aux_mask_source",
        type=str,
        default=str(cfg.get("aux_mask_source", "hard")),
        choices=["hard", "train"],
        help="Label source for gain/dup/class/attr positive/negative masks and pair conflict direction.",
    )
    p.add_argument(
        "--aux_mask_thr",
        type=float,
        default=float(cfg.get("aux_mask_thr", 0.5)),
        help="Threshold used when aux_mask_source=train.",
    )
    p.add_argument(
        "--eval_label_source",
        type=str,
        default=str(cfg.get("eval_label_source", "hard")),
        choices=["hard", "train"],
        help="Label source for threshold search and reported keep metrics.",
    )
    p.add_argument(
        "--eval_label_thr",
        type=float,
        default=float(cfg.get("eval_label_thr", 0.5)),
        help="Binarization threshold used when eval_label_source=train.",
    )
    p.add_argument(
        "--keep_target_type",
        type=str,
        default=str(cfg.get("keep_target_type", "soft")),
        choices=["hard", "soft"],
        help="Use hard y_keep or soft y_keep_soft as keep training target.",
    )
    p.add_argument(
        "--soft_target_mix",
        type=float,
        default=float(cfg.get("soft_target_mix", 1.0)),
        help="Mix ratio for keep target when keep_target_type=soft: y=(1-a)*hard + a*soft.",
    )
    p.add_argument(
        "--unique_target_field",
        type=str,
        default=str(cfg.get("unique_target_field", "y_unique")),
        choices=["y_unique", "cand_is_gt_best"],
        help="NPZ field used as unique-primary target.",
    )
    p.add_argument("--lambda_unique", type=float, default=float(cfg.get("lambda_unique", 0.05)))
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--unique_pos_only",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("unique_pos_only", True)),
            help="If true, unique auxiliary loss only emphasizes positive unique anchors.",
        )
    else:
        gu = p.add_mutually_exclusive_group()
        gu.add_argument("--unique_pos_only", dest="unique_pos_only", action="store_true")
        gu.add_argument("--no-unique_pos_only", dest="unique_pos_only", action="store_false")
        p.set_defaults(unique_pos_only=bool(cfg.get("unique_pos_only", True)))
    p.add_argument("--unique_neg_weight", type=float, default=float(cfg.get("unique_neg_weight", 0.05)))

    p.add_argument("--threshold_grid", type=str, default=str(cfg.get("threshold_grid", "0.1,0.2,0.3,0.4,0.5,0.6")))
    p.add_argument("--threshold_metric", type=str, default=str(cfg.get("threshold_metric", "f2")),
                   choices=["f1", "f2"], help="Metric used to select best threshold.")
    p.add_argument(
        "--relation_feature_names",
        type=str,
        default=str(
            cfg.get(
                "relation_feature_names",
                "temporal_stability,local_density,support_count,support_unique_dt,support_score_mean,support_abs_dt_mean,support_raw_ratio,support_warp_ratio,temporal_local_interaction,support_density_interaction",
            )
        ),
        help="Comma-separated feature names used by relation block.",
    )
    p.add_argument("--max_train_rows", type=int, default=cfg.get("max_train_rows"),
                   help="Optional subsample cap for training rows.")
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--enable_context_block",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("enable_context_block", True)),
            help="Train/apply neighborhood context block on top of unary logits.",
        )
    else:
        gcb = p.add_mutually_exclusive_group()
        gcb.add_argument("--enable_context_block", dest="enable_context_block", action="store_true")
        gcb.add_argument("--no-enable_context_block", dest="enable_context_block", action="store_false")
        p.set_defaults(enable_context_block=bool(cfg.get("enable_context_block", True)))
    p.add_argument("--context_radius_xy", type=float, default=float(cfg.get("context_radius_xy", 1.5)))
    p.add_argument("--context_max_neighbors", type=int, default=int(cfg.get("context_max_neighbors", 24)))
    p.add_argument("--context_rounds", type=int, default=int(cfg.get("context_rounds", 2)))
    p.add_argument("--context_epochs", type=int, default=int(cfg.get("context_epochs", 6)))
    p.add_argument("--context_lr", type=float, default=float(cfg.get("context_lr", 0.05)))
    p.add_argument("--context_l2", type=float, default=float(cfg.get("context_l2", 1e-6)))

    # stage-2 multitask targets (optional, backward-compatible)
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--enable_multitask",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("enable_multitask", True)),
            help="Enable multitask auxiliary losses when target fields exist in NPZ.",
        )
        p.add_argument(
            "--gain_pos_only",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("gain_pos_only", True)),
            help="Apply gain regression mainly on positive keep targets.",
        )
        p.add_argument(
            "--dup_neg_only",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("dup_neg_only", True)),
            help="Apply duplicate suppression mainly on negative keep targets.",
        )
    else:
        g1 = p.add_mutually_exclusive_group()
        g1.add_argument("--enable_multitask", dest="enable_multitask", action="store_true")
        g1.add_argument("--no-enable_multitask", dest="enable_multitask", action="store_false")
        p.set_defaults(enable_multitask=bool(cfg.get("enable_multitask", True)))
        g2 = p.add_mutually_exclusive_group()
        g2.add_argument("--gain_pos_only", dest="gain_pos_only", action="store_true")
        g2.add_argument("--no-gain_pos_only", dest="gain_pos_only", action="store_false")
        p.set_defaults(gain_pos_only=bool(cfg.get("gain_pos_only", True)))
        g3 = p.add_mutually_exclusive_group()
        g3.add_argument("--dup_neg_only", dest="dup_neg_only", action="store_true")
        g3.add_argument("--no-dup_neg_only", dest="dup_neg_only", action="store_false")
        p.set_defaults(dup_neg_only=bool(cfg.get("dup_neg_only", True)))

    p.add_argument(
        "--gain_target_field",
        type=str,
        default=str(cfg.get("gain_target_field", "cand_cover_gain_greedy")),
        choices=["cand_cover_gain_greedy", "cand_cover_gain"],
        help="NPZ field used as gain regression target.",
    )
    p.add_argument("--gain_target_clip", type=float, default=float(cfg.get("gain_target_clip", 1.0)))
    p.add_argument("--lambda_gain", type=float, default=float(cfg.get("lambda_gain", 0.25)))
    p.add_argument("--lambda_dup", type=float, default=float(cfg.get("lambda_dup", 0.15)))
    p.add_argument("--dup_weight_power", type=float, default=float(cfg.get("dup_weight_power", 1.0)))
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--enable_pairwise_dup",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("enable_pairwise_dup", True)),
            help="Enable pairwise ranking duplicate suppression loss from pair graph.",
        )
    else:
        g4 = p.add_mutually_exclusive_group()
        g4.add_argument("--enable_pairwise_dup", dest="enable_pairwise_dup", action="store_true")
        g4.add_argument("--no-enable_pairwise_dup", dest="enable_pairwise_dup", action="store_false")
        p.set_defaults(enable_pairwise_dup=bool(cfg.get("enable_pairwise_dup", True)))
    p.add_argument("--lambda_pair_dup", type=float, default=float(cfg.get("lambda_pair_dup", 0.20)))
    p.add_argument("--pair_rank_margin_eps", type=float, default=float(cfg.get("pair_rank_margin_eps", 0.05)))
    p.add_argument(
        "--pair_conflict_dup_min",
        type=float,
        default=float(cfg.get("pair_conflict_dup_min", 0.20)),
        help="Only pair edges with max dup risk above this value are used for pairwise duplicate ranking.",
    )
    p.add_argument("--pair_max_samples_per_epoch", type=int, default=int(cfg.get("pair_max_samples_per_epoch", 200000)))
    p.add_argument("--pair_batch_size", type=int, default=int(cfg.get("pair_batch_size", 4096)))
    p.add_argument(
        "--pair_warp_relax",
        type=float,
        default=float(cfg.get("pair_warp_relax", 0.5)),
        help="Extra multiplier for pairwise duplicate edge weight when both endpoints are warp.",
    )

    p.add_argument(
        "--target_recall",
        type=float,
        default=cfg.get("target_recall"),
        help="If set, threshold selection first filters candidates with R >= target_recall.",
    )
    p.add_argument(
        "--target_select_metric",
        type=str,
        default=str(cfg.get("target_select_metric", "precision")),
        choices=["precision", "f1"],
        help="Secondary objective after recall constraint: maximize precision or F1.",
    )
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--enable_class_head",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("enable_class_head", True)),
            help="Train learned class correction head from y_cls.",
        )
    else:
        g5 = p.add_mutually_exclusive_group()
        g5.add_argument("--enable_class_head", dest="enable_class_head", action="store_true")
        g5.add_argument("--no-enable_class_head", dest="enable_class_head", action="store_false")
        p.set_defaults(enable_class_head=bool(cfg.get("enable_class_head", True)))
    p.add_argument("--lambda_cls", type=float, default=float(cfg.get("lambda_cls", 0.25)))
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--class_pos_only",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("class_pos_only", True)),
            help="Train class head only on positive keep targets.",
        )
        p.add_argument(
            "--enable_attr_head",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("enable_attr_head", True)),
            help="Train learned attr head from y_attr.",
        )
    else:
        g6 = p.add_mutually_exclusive_group()
        g6.add_argument("--class_pos_only", dest="class_pos_only", action="store_true")
        g6.add_argument("--no-class_pos_only", dest="class_pos_only", action="store_false")
        p.set_defaults(class_pos_only=bool(cfg.get("class_pos_only", True)))
        g7 = p.add_mutually_exclusive_group()
        g7.add_argument("--enable_attr_head", dest="enable_attr_head", action="store_true")
        g7.add_argument("--no-enable_attr_head", dest="enable_attr_head", action="store_false")
        p.set_defaults(enable_attr_head=bool(cfg.get("enable_attr_head", True)))
    p.add_argument("--lambda_attr", type=float, default=float(cfg.get("lambda_attr", 0.10)))
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--attr_pos_only",
            action=argparse.BooleanOptionalAction,
            default=bool(cfg.get("attr_pos_only", True)),
            help="Train attr head only on positive keep targets.",
        )
    else:
        g8 = p.add_mutually_exclusive_group()
        g8.add_argument("--attr_pos_only", dest="attr_pos_only", action="store_true")
        g8.add_argument("--no-attr_pos_only", dest="attr_pos_only", action="store_false")
        p.set_defaults(attr_pos_only=bool(cfg.get("attr_pos_only", True)))

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
    data = np.load(in_npz, allow_pickle=True, mmap_mode="r")
    np_dtype = np.float32 if str(args.dtype).lower() == "float32" else np.float64
    X_raw = data["X"]
    X_raw_full = np.asarray(X_raw, dtype=np.float64)
    X = np.asarray(X_raw, dtype=np_dtype) if X_raw.dtype != np_dtype else np.asarray(X_raw)
    y_hard_raw = data["y_keep"]
    y_hard = np.asarray(y_hard_raw, dtype=np_dtype).reshape(-1) if y_hard_raw.dtype != np_dtype else np.asarray(y_hard_raw).reshape(-1)
    keep_target_type = str(args.keep_target_type).lower()
    if keep_target_type == "soft" and ("y_keep_soft" in data.files):
        y_soft_raw = data["y_keep_soft"]
        y_soft = np.asarray(y_soft_raw, dtype=np_dtype).reshape(-1) if y_soft_raw.dtype != np_dtype else np.asarray(y_soft_raw).reshape(-1)
        mix = float(np.clip(args.soft_target_mix, 0.0, 1.0))
        y = ((1.0 - mix) * y_hard + mix * y_soft).astype(np_dtype, copy=False)
    else:
        y = y_hard.copy()
    scene_idx = np.asarray(data["scene_idx"], dtype=np.int64).reshape(-1)
    frame_idx_all = np.asarray(data["frame_idx"], dtype=np.int64).reshape(-1) if "frame_idx" in data.files else np.zeros_like(scene_idx)
    feat_names = [str(x) for x in data["feature_names"].tolist()]
    y_cls_all: Optional[np.ndarray] = None
    if "y_cls" in data.files:
        y_cls_all = np.asarray(data["y_cls"], dtype=np.int64).reshape(-1)
    y_attr_all: Optional[np.ndarray] = None
    if "y_attr" in data.files:
        y_attr_all = np.asarray(data["y_attr"], dtype=np.int64).reshape(-1)

    gain_target_all: Optional[np.ndarray] = None
    gain_field = str(args.gain_target_field)
    if gain_field in data.files:
        g_raw = data[gain_field]
        gain_target_all = np.asarray(g_raw, dtype=np_dtype).reshape(-1) if g_raw.dtype != np_dtype else np.asarray(g_raw).reshape(-1)
    elif "cand_cover_gain" in data.files:
        g_raw = data["cand_cover_gain"]
        gain_target_all = np.asarray(g_raw, dtype=np_dtype).reshape(-1) if g_raw.dtype != np_dtype else np.asarray(g_raw).reshape(-1)
        gain_field = "cand_cover_gain"

    dup_target_all: Optional[np.ndarray] = None
    if "cand_dup_risk" in data.files:
        d_raw = data["cand_dup_risk"]
        dup_target_all = np.asarray(d_raw, dtype=np_dtype).reshape(-1) if d_raw.dtype != np_dtype else np.asarray(d_raw).reshape(-1)

    pair_i_all: Optional[np.ndarray] = None
    pair_j_all: Optional[np.ndarray] = None
    pair_label_all: Optional[np.ndarray] = None
    if ("pair_i" in data.files) and ("pair_j" in data.files) and ("pair_label" in data.files):
        pair_i_all = np.asarray(data["pair_i"], dtype=np.int64).reshape(-1)
        pair_j_all = np.asarray(data["pair_j"], dtype=np.int64).reshape(-1)
        pair_label_all = np.asarray(data["pair_label"], dtype=np.int64).reshape(-1)
    unique_target_all: Optional[np.ndarray] = None
    unique_field = str(args.unique_target_field)
    if unique_field in data.files:
        u_raw = data[unique_field]
        unique_target_all = np.asarray(u_raw, dtype=np_dtype).reshape(-1) if u_raw.dtype != np_dtype else np.asarray(u_raw).reshape(-1)
    elif "cand_is_gt_best" in data.files:
        u_raw = data["cand_is_gt_best"]
        unique_target_all = np.asarray(u_raw, dtype=np_dtype).reshape(-1) if u_raw.dtype != np_dtype else np.asarray(u_raw).reshape(-1)
        unique_field = "cand_is_gt_best"

    assert X.ndim == 2, f"X shape must be [N,D], got {X.shape}"
    assert y.shape[0] == X.shape[0], f"y length mismatch: {y.shape[0]} vs {X.shape[0]}"
    if y_cls_all is not None and y_cls_all.shape[0] != y.shape[0]:
        y_cls_all = None
    if y_attr_all is not None and y_attr_all.shape[0] != y.shape[0]:
        y_attr_all = None
    if unique_target_all is not None and unique_target_all.shape[0] != y.shape[0]:
        unique_target_all = None

    attr_id_to_name: Optional[List[str]] = None
    meta_candidates = [
        os.path.splitext(in_npz)[0] + ".meta.json",
        os.path.join(os.path.dirname(in_npz), "trainset.meta.json"),
    ]
    for mp in meta_candidates:
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, "r", encoding="utf-8") as f:
                mobj = json.load(f)
            av = mobj.get("attr_vocab", {})
            if isinstance(av, dict):
                id2name: Dict[int, str] = {}
                for name, idx in av.items():
                    try:
                        ii = int(idx)
                    except Exception:
                        continue
                    if ii >= 0:
                        id2name[ii] = str(name)
                if len(id2name) > 0:
                    max_id = max(id2name.keys())
                    attr_id_to_name = [id2name.get(i, "") for i in range(max_id + 1)]
                    break
        except Exception:
            pass

    N, D = X.shape
    print(
        f"  rows={N} dim={D} pos_rate_hard={float((y_hard > 0.5).mean()):.6f} "
        f"keep_target={keep_target_type}"
    )

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
    y_hard_tr = y_hard[tr_idx]
    y_hard_va = y_hard[va_idx] if va_idx.size > 0 else np.zeros((0,), dtype=np_dtype)
    X_raw_tr = X_raw_full[tr_idx]
    X_raw_va = X_raw_full[va_idx] if va_idx.size > 0 else np.zeros((0, X_raw_full.shape[1]), dtype=np.float64)
    scene_tr = scene_idx[tr_idx]
    scene_va = scene_idx[va_idx] if va_idx.size > 0 else np.zeros((0,), dtype=np.int64)
    frame_tr = frame_idx_all[tr_idx]
    frame_va = frame_idx_all[va_idx] if va_idx.size > 0 else np.zeros((0,), dtype=np.int64)
    y_cls_tr = (y_cls_all[tr_idx] if y_cls_all is not None else None)
    y_cls_va = (y_cls_all[va_idx] if y_cls_all is not None and va_idx.size > 0 else None)
    y_attr_tr = (y_attr_all[tr_idx] if y_attr_all is not None else None)
    y_attr_va = (y_attr_all[va_idx] if y_attr_all is not None and va_idx.size > 0 else None)
    gain_tr = (gain_target_all[tr_idx] if gain_target_all is not None else None)
    gain_va = (gain_target_all[va_idx] if gain_target_all is not None and va_idx.size > 0 else None)
    dup_tr = (dup_target_all[tr_idx] if dup_target_all is not None else None)
    dup_va = (dup_target_all[va_idx] if dup_target_all is not None and va_idx.size > 0 else None)
    unique_tr = (unique_target_all[tr_idx] if unique_target_all is not None else None)
    unique_va = (unique_target_all[va_idx] if unique_target_all is not None and va_idx.size > 0 else None)
    print(f"  train_rows={X_tr.shape[0]} val_rows={X_va.shape[0]}")
    aux_thr = float(args.aux_mask_thr)
    eval_thr = float(args.eval_label_thr)
    y_mask_tr = y_hard_tr if str(args.aux_mask_source).lower() == "hard" else y_tr
    y_mask_va = y_hard_va if str(args.aux_mask_source).lower() == "hard" else y_va
    y_eval_tr = y_hard_tr if str(args.eval_label_source).lower() == "hard" else y_tr
    y_eval_va = y_hard_va if str(args.eval_label_source).lower() == "hard" else y_va
    y_weight_tr = y_hard_tr if str(args.weight_label_source).lower() == "hard" else y_tr
    y_mask_tr_bin = (y_mask_tr > aux_thr)
    y_mask_va_bin = (y_mask_va > aux_thr) if y_mask_va.shape[0] > 0 else np.zeros((0,), dtype=bool)
    y_eval_tr_bin = (y_eval_tr > eval_thr).astype(np_dtype, copy=False)
    y_eval_va_bin = (y_eval_va > eval_thr).astype(np_dtype, copy=False) if y_eval_va.shape[0] > 0 else np.zeros((0,), dtype=np_dtype)

    # prepare pairwise ranking dataset on train split
    pair_i_tr = np.zeros((0,), dtype=np.int64)
    pair_j_tr = np.zeros((0,), dtype=np.int64)
    pair_sign_tr = np.zeros((0,), dtype=np_dtype)
    pair_w_tr = np.zeros((0,), dtype=np_dtype)

    if args.normalize:
        mu = X_tr.mean(axis=0)
        std = X_tr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        X_tr = (X_tr - mu) / std
        if X_va.shape[0] > 0:
            X_va = (X_va - mu) / std
    else:
        mu = np.zeros((D,), dtype=np_dtype)
        std = np.ones((D,), dtype=np_dtype)

    print("[3/5] Preparing optimizer...")
    pos = int((y_weight_tr > 0.5).sum())
    neg = int((y_weight_tr <= 0.5).sum())
    if args.pos_weight_mode == "auto":
        pos_w = float(neg / max(1, pos))
    elif args.pos_weight_mode == "value":
        pos_w = float(args.pos_weight)
    else:
        pos_w = 1.0

    sw_tr = np.ones_like(y_tr, dtype=np_dtype)
    sw_tr[y_weight_tr > 0.5] = pos_w

    # optional multitask targets
    multitask_active = bool(args.enable_multitask) and (gain_tr is not None or dup_tr is not None or unique_tr is not None)
    lambda_gain = float(args.lambda_gain) if (multitask_active and gain_tr is not None) else 0.0
    lambda_dup = float(args.lambda_dup) if (multitask_active and dup_tr is not None) else 0.0
    lambda_unique = float(args.lambda_unique) if (multitask_active and unique_tr is not None) else 0.0

    if gain_tr is not None:
        gc = float(max(1e-8, args.gain_target_clip))
        gain_tr = np.clip(gain_tr, 0.0, gc) / gc
        if gain_va is not None:
            gain_va = np.clip(gain_va, 0.0, gc) / gc
        gain_w_tr = np.ones_like(gain_tr, dtype=np_dtype)
        if bool(args.gain_pos_only):
            gain_w_tr = np.where(y_mask_tr_bin, 1.0, 0.15).astype(np_dtype)
    else:
        gain_w_tr = None

    if dup_tr is not None:
        pwr = float(max(0.1, args.dup_weight_power))
        dup_tr = np.clip(dup_tr, 0.0, 1.0)
        if dup_va is not None:
            dup_va = np.clip(dup_va, 0.0, 1.0)
        dup_w_tr = np.power(dup_tr, pwr).astype(np_dtype)
        if bool(args.dup_neg_only):
            dup_w_tr = np.where(~y_mask_tr_bin, dup_w_tr, 0.05 * dup_w_tr)
    else:
        dup_w_tr = None

    if unique_tr is not None:
        unique_tr = np.clip(unique_tr, 0.0, 1.0)
        unique_w_tr = np.ones_like(unique_tr, dtype=np_dtype)
        if bool(args.unique_pos_only):
            unique_w_tr = np.where(unique_tr > 0.5, 1.0, 0.0).astype(np_dtype)
        else:
            unique_w_tr = np.where(unique_tr > 0.5, 1.0, float(max(0.0, min(1.0, args.unique_neg_weight)))).astype(np_dtype)
    else:
        unique_w_tr = None

    if bool(args.enable_pairwise_dup) and pair_i_all is not None and pair_j_all is not None and pair_label_all is not None:
        g2l = np.full((N,), -1, dtype=np.int64)
        g2l[tr_idx] = np.arange(tr_idx.shape[0], dtype=np.int64)
        li = g2l[pair_i_all]
        lj = g2l[pair_j_all]
        valid = (li >= 0) & (lj >= 0)
        if np.any(valid):
            li = li[valid]
            lj = lj[valid]
            pl = pair_label_all[valid]

            # Only use explicit support-negative edges where keep labels disagree.
            # This avoids treating all non-positive pair labels as hard conflicts.
            yi = y_mask_tr_bin[li]
            conflict = (pl == 0) & (yi != y_mask_tr_bin[lj])
            li = li[conflict]
            lj = lj[conflict]
            yi = yi[conflict]

            if li.size > 0:
                sign = np.where(yi, 1.0, -1.0).astype(np_dtype)
                pw = np.ones_like(sign, dtype=np_dtype)
                if dup_tr is not None:
                    dwi = np.asarray(dup_tr, dtype=np_dtype)[li]
                    dwj = np.asarray(dup_tr, dtype=np_dtype)[lj]
                    dmax = np.clip(np.maximum(dwi, dwj), 0.0, 1.0)
                    keep_dup = dmax >= float(max(0.0, min(1.0, args.pair_conflict_dup_min)))
                    li = li[keep_dup]
                    lj = lj[keep_dup]
                    sign = sign[keep_dup]
                    dmax = dmax[keep_dup]
                    pw = np.clip(dmax, 0.1, 1.0)
                pair_warp_relax = float(np.clip(args.pair_warp_relax, 0.0, 1.0))
                if pair_warp_relax < 0.999:
                    name_to_idx = {n: i for i, n in enumerate(feat_names)}
                    if "is_warp" in name_to_idx:
                        is_warp_col = int(name_to_idx["is_warp"])
                        iw = np.clip(X_raw_tr[:, is_warp_col].astype(np.float64, copy=False), 0.0, 1.0)
                        both_warp = (iw[li] > 0.5) & (iw[lj] > 0.5)
                        if np.any(both_warp):
                            pw = pw.astype(np.float64, copy=False)
                            pw[both_warp] *= pair_warp_relax
                            pw = pw.astype(np_dtype, copy=False)

                if li.size > 0:
                    pair_i_tr = li.astype(np.int64, copy=False)
                    pair_j_tr = lj.astype(np.int64, copy=False)
                    pair_sign_tr = sign
                    pair_w_tr = pw

    w = np.zeros((D,), dtype=np_dtype)
    b = 0.0
    class_enabled = bool(args.enable_class_head) and (y_cls_tr is not None) and (int(np.sum(y_cls_tr >= 0)) > 0)
    attr_enabled = bool(args.enable_attr_head) and (y_attr_tr is not None) and (int(np.sum(y_attr_tr >= 0)) > 0)
    class_labels: List[int] = []
    class_label_to_idx: Dict[int, int] = {}
    class_W: Optional[np.ndarray] = None
    class_b: Optional[np.ndarray] = None
    if class_enabled and y_cls_tr is not None:
        class_labels = sorted(np.unique(y_cls_tr[y_cls_tr >= 0]).astype(np.int64).tolist())
        class_label_to_idx = {int(v): i for i, v in enumerate(class_labels)}
        if len(class_labels) > 1:
            class_W = np.zeros((D, len(class_labels)), dtype=np_dtype)
            class_b = np.zeros((len(class_labels),), dtype=np_dtype)
        else:
            class_enabled = False

    attr_ids: List[int] = []
    attr_id_to_idx: Dict[int, int] = {}
    attr_W: Optional[np.ndarray] = None
    attr_b: Optional[np.ndarray] = None
    if attr_enabled and y_attr_tr is not None:
        attr_ids = sorted(np.unique(y_attr_tr[y_attr_tr >= 0]).astype(np.int64).tolist())
        attr_id_to_idx = {int(v): i for i, v in enumerate(attr_ids)}
        if len(attr_ids) > 1:
            attr_W = np.zeros((D, len(attr_ids)), dtype=np_dtype)
            attr_b = np.zeros((len(attr_ids),), dtype=np_dtype)
        else:
            attr_enabled = False

    epochs = int(args.epochs)
    bs = max(1, int(args.batch_size))
    lr = float(args.lr)
    l2 = float(args.l2)

    print(
        f"  pos={pos} neg={neg} pos_weight={pos_w:.4f} epochs={epochs} batch_size={bs} lr={lr} l2={l2} "
        f"multitask={multitask_active} gain_field={gain_field if gain_target_all is not None else 'none'} "
        f"unique_field={unique_field if unique_target_all is not None else 'none'} "
        f"pair_rank_edges={int(pair_i_tr.size)} class_head={class_enabled} attr_head={attr_enabled}"
    )

    print("[4/5] Training...")
    for ep in range(1, epochs + 1):
        perm = rng.permutation(X_tr.shape[0])
        X_ep = X_tr[perm]
        y_ep = y_tr[perm]
        ymask_ep = y_mask_tr_bin[perm]
        sw_ep = sw_tr[perm]
        gain_ep = (gain_tr[perm] if gain_tr is not None else None)
        gain_w_ep = (gain_w_tr[perm] if gain_w_tr is not None else None)
        dup_w_ep = (dup_w_tr[perm] if dup_w_tr is not None else None)
        unique_ep = (unique_tr[perm] if unique_tr is not None else None)
        unique_w_ep = (unique_w_tr[perm] if unique_w_tr is not None else None)
        cls_ep = (y_cls_tr[perm] if y_cls_tr is not None else None)
        attr_ep = (y_attr_tr[perm] if y_attr_tr is not None else None)

        ep_unary_loss = 0.0
        ep_bce = 0.0
        ep_gain = 0.0
        ep_unique = 0.0
        ep_dup = 0.0
        ep_cls = 0.0
        ep_attr = 0.0
        ep_batches = 0
        ep_pair = 0.0
        ep_pair_weighted = 0.0
        ep_pair_batches = 0
        ep_unary_sum = 0.0
        ep_pair_weighted_sum = 0.0
        lambda_pair = float(args.lambda_pair_dup) if pair_i_tr.size > 0 else 0.0

        # sample pair subset for this epoch
        if pair_i_tr.size > 0:
            n_pair_max = int(max(0, args.pair_max_samples_per_epoch))
            if n_pair_max > 0 and pair_i_tr.size > n_pair_max:
                pick = rng.choice(pair_i_tr.size, size=n_pair_max, replace=False)
                pi_ep = pair_i_tr[pick]
                pj_ep = pair_j_tr[pick]
                ps_ep = pair_sign_tr[pick]
                pw_ep = pair_w_tr[pick]
            else:
                pi_ep = pair_i_tr
                pj_ep = pair_j_tr
                ps_ep = pair_sign_tr
                pw_ep = pair_w_tr
            pair_order = rng.permutation(pi_ep.size)
            pi_ep = pi_ep[pair_order]
            pj_ep = pj_ep[pair_order]
            ps_ep = ps_ep[pair_order]
            pw_ep = pw_ep[pair_order]
            pair_bs = int(max(128, args.pair_batch_size))
        else:
            pi_ep = pj_ep = np.zeros((0,), dtype=np.int64)
            ps_ep = pw_ep = np.zeros((0,), dtype=np_dtype)
            pair_bs = 0

        for s in range(0, X_ep.shape[0], bs):
            e = min(s + bs, X_ep.shape[0])
            Xb = X_ep[s:e]
            yb = y_ep[s:e]
            ymb = ymask_ep[s:e]
            sb = sw_ep[s:e]
            gb = (gain_ep[s:e] if gain_ep is not None else None)
            gwb = (gain_w_ep[s:e] if gain_w_ep is not None else None)
            dwb = (dup_w_ep[s:e] if dup_w_ep is not None else None)
            ub = (unique_ep[s:e] if unique_ep is not None else None)
            uwb = (unique_w_ep[s:e] if unique_w_ep is not None else None)
            clsb = (cls_ep[s:e] if cls_ep is not None else None)
            attrb = (attr_ep[s:e] if attr_ep is not None else None)

            loss, loss_bce, loss_gain, loss_unique, grad_w, grad_b = _multitask_loss_and_grad(
                Xb,
                yb,
                w,
                b,
                sb,
                l2=l2,
                gain_target=gb,
                gain_weight=gwb,
                dup_weight=dwb,
                unique_target=ub,
                unique_weight=uwb,
                lambda_gain=lambda_gain,
                lambda_dup=lambda_dup,
                lambda_unique=lambda_unique,
            )

            loss_cls = 0.0
            if class_enabled and class_W is not None and class_b is not None and clsb is not None:
                cls_mask = (clsb >= 0)
                if bool(args.class_pos_only):
                    cls_mask = cls_mask & ymb
                if np.any(cls_mask):
                    Xc = Xb[cls_mask]
                    ycl = clsb[cls_mask]
                    y_idx = np.asarray([class_label_to_idx.get(int(v), -1) for v in ycl.tolist()], dtype=np.int64)
                    valid = y_idx >= 0
                    if np.any(valid):
                        Xc = Xc[valid]
                        y_idx = y_idx[valid]
                        cls_logits = Xc @ class_W + class_b
                        cls_prob = _softmax_logits(cls_logits)
                        n_cls = int(y_idx.shape[0])
                        pick = cls_prob[np.arange(n_cls), y_idx]
                        loss_cls = float(-np.mean(np.log(np.clip(pick, 1e-8, 1.0))))
                        g_cls = cls_prob
                        g_cls[np.arange(n_cls), y_idx] -= 1.0
                        g_cls /= max(1, n_cls)
                        grad_wc = Xc.T @ g_cls + float(l2) * class_W
                        grad_bc = np.sum(g_cls, axis=0)
                        class_W -= float(lr) * float(args.lambda_cls) * grad_wc.astype(np_dtype, copy=False)
                        class_b -= float(lr) * float(args.lambda_cls) * grad_bc.astype(np_dtype, copy=False)

            loss_attr = 0.0
            if attr_enabled and attr_W is not None and attr_b is not None and attrb is not None:
                attr_mask = (attrb >= 0)
                if bool(args.attr_pos_only):
                    attr_mask = attr_mask & ymb
                if np.any(attr_mask):
                    Xa = Xb[attr_mask]
                    yat = attrb[attr_mask]
                    y_idx = np.asarray([attr_id_to_idx.get(int(v), -1) for v in yat.tolist()], dtype=np.int64)
                    valid = y_idx >= 0
                    if np.any(valid):
                        Xa = Xa[valid]
                        y_idx = y_idx[valid]
                        attr_logits = Xa @ attr_W + attr_b
                        attr_prob = _softmax_logits(attr_logits)
                        n_attr = int(y_idx.shape[0])
                        pick = attr_prob[np.arange(n_attr), y_idx]
                        loss_attr = float(-np.mean(np.log(np.clip(pick, 1e-8, 1.0))))
                        g_attr = attr_prob
                        g_attr[np.arange(n_attr), y_idx] -= 1.0
                        g_attr /= max(1, n_attr)
                        grad_wa = Xa.T @ g_attr + float(l2) * attr_W
                        grad_ba = np.sum(g_attr, axis=0)
                        attr_W -= float(lr) * float(args.lambda_attr) * grad_wa.astype(np_dtype, copy=False)
                        attr_b -= float(lr) * float(args.lambda_attr) * grad_ba.astype(np_dtype, copy=False)

            w -= lr * grad_w
            b -= lr * grad_b

            ep_unary_loss += loss
            ep_unary_sum += float(loss) + float(args.lambda_cls) * float(loss_cls) + float(args.lambda_attr) * float(loss_attr)
            ep_bce += loss_bce
            ep_gain += loss_gain
            ep_unique += loss_unique
            ep_cls += float(loss_cls)
            ep_attr += float(loss_attr)
            if (dwb is not None) and (lambda_dup > 0.0):
                # derive dup loss for logging from current batch probs
                logits_dbg = Xb @ w + b
                probs_dbg = np.clip(_sigmoid(logits_dbg), 1e-8, 1.0 - 1e-8)
                ep_dup += float(np.mean(dwb * (-np.log(1.0 - probs_dbg))))
            ep_batches += 1

        if pair_bs > 0 and lambda_pair > 0.0 and pi_ep.size > 0:
            for ps in range(0, pi_ep.size, pair_bs):
                pe = min(ps + pair_bs, pi_ep.size)
                pidx_i = pi_ep[ps:pe]
                pidx_j = pj_ep[ps:pe]
                psgn = ps_ep[ps:pe]
                pww = pw_ep[ps:pe]
                ploss, pgrad_w = _pairwise_rank_loss_and_grad(
                    X_tr,
                    w,
                    pidx_i,
                    pidx_j,
                    psgn,
                    pww,
                )
                w -= lr * float(lambda_pair) * pgrad_w
                ep_pair += float(ploss)
                ep_pair_weighted += float(lambda_pair) * float(ploss)
                ep_pair_weighted_sum += float(lambda_pair) * float(ploss)
                ep_pair_batches += 1

        tr_probs = _sigmoid(X_tr @ w + b)
        tr_m = _metrics_from_probs(y_eval_tr_bin, tr_probs, thr=0.5)
        tr_cls_acc = 0.0
        va_cls_acc = 0.0
        tr_attr_acc = 0.0
        va_attr_acc = 0.0
        if class_enabled and class_W is not None and class_b is not None and y_cls_tr is not None:
            cls_mask_tr = (y_cls_tr >= 0)
            if bool(args.class_pos_only):
                cls_mask_tr = cls_mask_tr & y_mask_tr_bin
            if np.any(cls_mask_tr):
                lg = X_tr[cls_mask_tr] @ class_W + class_b
                pred_idx = np.argmax(lg, axis=1)
                tgt_idx = np.asarray([class_label_to_idx.get(int(v), -1) for v in y_cls_tr[cls_mask_tr].tolist()], dtype=np.int64)
                valid = tgt_idx >= 0
                if np.any(valid):
                    tr_cls_acc = float(np.mean(pred_idx[valid] == tgt_idx[valid]))
            if X_va.shape[0] > 0 and y_cls_va is not None:
                cls_mask_va = (y_cls_va >= 0)
                if bool(args.class_pos_only):
                    cls_mask_va = cls_mask_va & y_mask_va_bin
                if np.any(cls_mask_va):
                    lg = X_va[cls_mask_va] @ class_W + class_b
                    pred_idx = np.argmax(lg, axis=1)
                    tgt_idx = np.asarray([class_label_to_idx.get(int(v), -1) for v in y_cls_va[cls_mask_va].tolist()], dtype=np.int64)
                    valid = tgt_idx >= 0
                    if np.any(valid):
                        va_cls_acc = float(np.mean(pred_idx[valid] == tgt_idx[valid]))
        if attr_enabled and attr_W is not None and attr_b is not None and y_attr_tr is not None:
            attr_mask_tr = (y_attr_tr >= 0)
            if bool(args.attr_pos_only):
                attr_mask_tr = attr_mask_tr & y_mask_tr_bin
            if np.any(attr_mask_tr):
                lg = X_tr[attr_mask_tr] @ attr_W + attr_b
                pred_idx = np.argmax(lg, axis=1)
                tgt_idx = np.asarray([attr_id_to_idx.get(int(v), -1) for v in y_attr_tr[attr_mask_tr].tolist()], dtype=np.int64)
                valid = tgt_idx >= 0
                if np.any(valid):
                    tr_attr_acc = float(np.mean(pred_idx[valid] == tgt_idx[valid]))
            if X_va.shape[0] > 0 and y_attr_va is not None:
                attr_mask_va = (y_attr_va >= 0)
                if bool(args.attr_pos_only):
                    attr_mask_va = attr_mask_va & y_mask_va_bin
                if np.any(attr_mask_va):
                    lg = X_va[attr_mask_va] @ attr_W + attr_b
                    pred_idx = np.argmax(lg, axis=1)
                    tgt_idx = np.asarray([attr_id_to_idx.get(int(v), -1) for v in y_attr_va[attr_mask_va].tolist()], dtype=np.int64)
                    valid = tgt_idx >= 0
                    if np.any(valid):
                        va_attr_acc = float(np.mean(pred_idx[valid] == tgt_idx[valid]))
        pair_log = ep_pair / max(1, ep_pair_batches)
        loss_unary_log = ep_unary_loss / max(1, ep_batches)
        loss_pair_log = ep_pair_weighted / max(1, ep_pair_batches)
        loss_total_log = (ep_unary_sum + ep_pair_weighted_sum) / max(1, ep_batches + ep_pair_batches)
        if X_va.shape[0] > 0:
            va_probs = _sigmoid(X_va @ w + b)
            va_m = _metrics_from_probs(y_eval_va_bin, va_probs, thr=0.5)
            print(
                f"  epoch={ep:03d} loss={loss_total_log:.6f} "
                f"unary={loss_unary_log:.6f} "
                f"bce={ep_bce / max(1, ep_batches):.6f} "
                f"gain={ep_gain / max(1, ep_batches):.6f} "
                f"uniq={ep_unique / max(1, ep_batches):.6f} "
                f"dup={ep_dup / max(1, ep_batches):.6f} "
                f"cls={ep_cls / max(1, ep_batches):.6f} "
                f"attr={ep_attr / max(1, ep_batches):.6f} "
                f"pair={pair_log:.6f} pair_w={loss_pair_log:.6f} "
                f"train_f1@0.5={tr_m['F1']:.4f} val_f1@0.5={va_m['F1']:.4f} val_f2@0.5={va_m['F2']:.4f} "
                f"train_cls_acc={tr_cls_acc:.4f} val_cls_acc={va_cls_acc:.4f} "
                f"train_attr_acc={tr_attr_acc:.4f} val_attr_acc={va_attr_acc:.4f}"
            )
        else:
            print(
                f"  epoch={ep:03d} loss={loss_total_log:.6f} "
                f"unary={loss_unary_log:.6f} "
                f"bce={ep_bce / max(1, ep_batches):.6f} "
                f"gain={ep_gain / max(1, ep_batches):.6f} "
                f"uniq={ep_unique / max(1, ep_batches):.6f} "
                f"dup={ep_dup / max(1, ep_batches):.6f} "
                f"cls={ep_cls / max(1, ep_batches):.6f} "
                f"attr={ep_attr / max(1, ep_batches):.6f} "
                f"pair={pair_log:.6f} pair_w={loss_pair_log:.6f} "
                f"train_f1@0.5={tr_m['F1']:.4f} "
                f"train_cls_acc={tr_cls_acc:.4f} train_attr_acc={tr_attr_acc:.4f}"
            )

    thr_grid = _parse_float_list(args.threshold_grid)
    if len(thr_grid) == 0:
        thr_grid = [0.5]

    metric_key = "F2" if str(args.threshold_metric).lower() == "f2" else "F1"
    rel_block: Optional[Dict[str, Any]] = None
    rel_feat_names = _parse_str_list(str(args.relation_feature_names))
    if X_tr.shape[0] > 0:
        base_logits_tr = (X_tr @ w + b).astype(np.float64, copy=False)
        rel_block = _fit_relation_block(
            X_tr,
            y_tr,
            base_logits=base_logits_tr,
            feature_names=feat_names,
            sample_weight=sw_tr,
            relation_feature_names=rel_feat_names,
            epochs=6,
            lr=min(0.1, float(args.lr)),
            l2=float(args.l2),
        )

    def _apply_rel(logits: np.ndarray, X_use: np.ndarray) -> np.ndarray:
        if not rel_block or (not bool(rel_block.get("enabled", False))):
            return logits
        names = [str(x) for x in rel_block.get("feature_names", [])]
        ws = np.asarray(rel_block.get("weights", []), dtype=np.float64).reshape(-1)
        if len(names) != ws.shape[0]:
            return logits
        idx = []
        name_to_idx = {n: i for i, n in enumerate(feat_names)}
        for n in names:
            if n not in name_to_idx:
                return logits
            idx.append(int(name_to_idx[n]))
        Z = X_use[:, idx].astype(np.float64, copy=False)
        br = float(rel_block.get("bias", 0.0))
        return logits + (Z @ ws + br)

    context_block: Optional[Dict[str, Any]] = None
    coords_tr = _build_context_coords(X_raw_tr, feat_names) if X_tr.shape[0] > 0 else None
    coords_va = _build_context_coords(X_raw_va, feat_names) if X_va.shape[0] > 0 else None
    if bool(args.enable_context_block) and X_tr.shape[0] > 0 and coords_tr is not None:
        logits_tr_for_ctx = _apply_rel((X_tr @ w + b).astype(np.float64, copy=False), X_tr)
        context_block = _fit_context_block(
            y=y_tr,
            sample_weight=sw_tr,
            base_logits=logits_tr_for_ctx,
            scene_idx=scene_tr,
            frame_idx=frame_tr,
            cand_label=coords_tr["label"],
            from_dt=coords_tr["from_dt"],
            is_raw=coords_tr["is_raw"],
            is_warp=coords_tr["is_warp"],
            score=coords_tr["score"],
            x=coords_tr["x"],
            y_xy=coords_tr["y"],
            radius_xy=float(args.context_radius_xy),
            max_neighbors=int(args.context_max_neighbors),
            rounds=int(args.context_rounds),
            epochs=int(args.context_epochs),
            lr=float(args.context_lr),
            l2=float(args.context_l2),
        )

    def _apply_post(
        logits: np.ndarray,
        X_use: np.ndarray,
        scene_use: np.ndarray,
        frame_use: np.ndarray,
        coords_use: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        out = _apply_rel(logits, X_use)
        return _apply_context_block_logits(
            out,
            context_block=context_block,
            scene_idx=scene_use,
            frame_idx=frame_use,
            coords=coords_use,
        )

    target_recall = float(args.target_recall) if (args.target_recall is not None) else None
    target_select_metric = str(args.target_select_metric).lower()

    def _pick_best_by_target(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if len(metrics_list) == 0:
            tr_logits_fb = _apply_post(
                (X_tr @ w + b).astype(np.float64, copy=False),
                X_tr,
                scene_tr,
                frame_tr,
                coords_tr,
            )
            return _metrics_from_probs(y_eval_tr_bin, _sigmoid(tr_logits_fb), thr=0.5)
        if target_recall is None:
            return max(metrics_list, key=lambda m: float(m.get(metric_key, 0.0)))
        feas = [m for m in metrics_list if float(m.get("R", 0.0)) >= target_recall]
        if len(feas) > 0:
            if target_select_metric == "f1":
                return max(feas, key=lambda m: (float(m.get("F1", 0.0)), float(m.get("P", 0.0)), float(m.get("R", 0.0))))
            return max(feas, key=lambda m: (float(m.get("P", 0.0)), float(m.get("F1", 0.0)), float(m.get("R", 0.0))))
        # fallback: no threshold satisfies target recall
        return max(metrics_list, key=lambda m: (float(m.get("R", 0.0)), float(m.get("P", 0.0)), float(m.get("F1", 0.0))))

    if X_va.shape[0] > 0:
        va_logits = _apply_post(
            (X_va @ w + b).astype(np.float64, copy=False),
            X_va,
            scene_va,
            frame_va,
            coords_va,
        )
        va_probs = _sigmoid(va_logits)
        all_ms = [_metrics_from_probs(y_eval_va_bin, va_probs, thr=float(t)) for t in thr_grid]
        best = _pick_best_by_target(all_ms)
        best_thr = float(best["threshold"])
        best_metrics = best
    else:
        best_thr = 0.5
        tr_logits = _apply_post(
            (X_tr @ w + b).astype(np.float64, copy=False),
            X_tr,
            scene_tr,
            frame_tr,
            coords_tr,
        )
        tr_probs = _sigmoid(tr_logits)
        all_ms = [_metrics_from_probs(y_eval_tr_bin, tr_probs, thr=float(t)) for t in thr_grid]
        best_metrics = _pick_best_by_target(all_ms)
        best_thr = float(best_metrics["threshold"])

    print("[5/5] Saving checkpoint...")
    ckpt = {
        "model_type": "logistic_energy",
        "version": 1,
        "input_dim": int(D),
        "feature_names": feat_names,
        "normalize": bool(args.normalize),
        "mu": [float(x) for x in mu.tolist()],
        "std": [float(x) for x in std.tolist()],
        "weights": [float(x) for x in w.tolist()],
        "bias": float(b),
        "best_threshold": float(best_thr),
        "threshold_metric": str(metric_key).lower(),
        "relation": rel_block if rel_block is not None else {"enabled": False},
        "context_block": context_block if context_block is not None else {"enabled": False},
        "train": {
            "epochs": epochs,
            "batch_size": bs,
            "lr": lr,
            "l2": l2,
            "keep_target_type": str(keep_target_type),
            "soft_target_mix": float(args.soft_target_mix),
            "enable_multitask": bool(multitask_active),
            "gain_target_field": str(gain_field) if gain_target_all is not None else None,
            "unique_target_field": str(unique_field) if unique_target_all is not None else None,
            "gain_target_clip": float(args.gain_target_clip),
            "gain_pos_only": bool(args.gain_pos_only),
            "dup_neg_only": bool(args.dup_neg_only),
            "lambda_gain": float(lambda_gain),
            "lambda_dup": float(lambda_dup),
            "lambda_unique": float(lambda_unique),
            "unique_pos_only": bool(args.unique_pos_only),
            "unique_neg_weight": float(args.unique_neg_weight),
            "dup_weight_power": float(args.dup_weight_power),
            "enable_pairwise_dup": bool(pair_i_tr.size > 0),
            "lambda_pair_dup": float(args.lambda_pair_dup),
            "pair_rank_margin_eps": float(args.pair_rank_margin_eps),
            "pair_conflict_dup_min": float(args.pair_conflict_dup_min),
            "pair_warp_relax": float(args.pair_warp_relax),
            "pair_max_samples_per_epoch": int(args.pair_max_samples_per_epoch),
            "pair_batch_size": int(args.pair_batch_size),
            "pair_rank_edges_train": int(pair_i_tr.size),
            "pos_weight_mode": str(args.pos_weight_mode),
            "pos_weight": float(pos_w),
            "weight_label_source": str(args.weight_label_source),
            "aux_mask_source": str(args.aux_mask_source),
            "aux_mask_thr": float(args.aux_mask_thr),
            "eval_label_source": str(args.eval_label_source),
            "eval_label_thr": float(args.eval_label_thr),
            "enable_context_block": bool(args.enable_context_block),
            "context_radius_xy": float(args.context_radius_xy),
            "context_max_neighbors": int(args.context_max_neighbors),
            "context_rounds": int(args.context_rounds),
            "context_epochs": int(args.context_epochs),
            "context_lr": float(args.context_lr),
            "context_l2": float(args.context_l2),
            "seed": int(args.seed),
            "split_by": str(args.split_by),
            "val_ratio": float(args.val_ratio),
            "dtype": str(args.dtype).lower(),
        },
    }
    if class_enabled and class_W is not None and class_b is not None:
        ckpt["class_head"] = {
            "enabled": True,
            "class_labels": [int(v) for v in class_labels],
            "weights": class_W.astype(np.float64).tolist(),
            "bias": class_b.astype(np.float64).tolist(),
            "lambda_cls": float(args.lambda_cls),
            "class_pos_only": bool(args.class_pos_only),
        }
    else:
        ckpt["class_head"] = {"enabled": False}
    if attr_enabled and attr_W is not None and attr_b is not None:
        attr_names: List[str] = []
        for aid in attr_ids:
            if attr_id_to_name is not None and int(aid) < len(attr_id_to_name):
                attr_names.append(str(attr_id_to_name[int(aid)]))
            else:
                attr_names.append(f"attr_{int(aid)}")
        ckpt["attr_head"] = {
            "enabled": True,
            "attr_ids": [int(v) for v in attr_ids],
            "attr_names": attr_names,
            "weights": attr_W.astype(np.float64).tolist(),
            "bias": attr_b.astype(np.float64).tolist(),
            "lambda_attr": float(args.lambda_attr),
            "attr_pos_only": bool(args.attr_pos_only),
        }
    else:
        ckpt["attr_head"] = {"enabled": False}

    with open(out_ckpt, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)

    summary = {
        "in_npz": in_npz,
        "out_ckpt": out_ckpt,
        "rows_total": int(N),
        "rows_train": int(X_tr.shape[0]),
        "rows_val": int(X_va.shape[0]),
        "dim": int(D),
        "pos_rate_total": float((y_hard > 0.5).mean()),
        "keep_target_type": str(keep_target_type),
        "soft_target_mix": float(args.soft_target_mix),
        "best_threshold": float(best_thr),
        "best_metrics": best_metrics,
        "threshold_metric": str(metric_key).lower(),
        "relation_enabled": bool(rel_block is not None and rel_block.get("enabled", False)),
        "relation": rel_block if rel_block is not None else {"enabled": False},
        "context_enabled": bool(context_block is not None and context_block.get("enabled", False)),
        "context_block": context_block if context_block is not None else {"enabled": False},
        "multitask": {
            "enabled": bool(multitask_active),
            "gain_target_field": str(gain_field) if gain_target_all is not None else None,
            "unique_target_field": str(unique_field) if unique_target_all is not None else None,
            "gain_target_clip": float(args.gain_target_clip),
            "gain_pos_only": bool(args.gain_pos_only),
            "dup_neg_only": bool(args.dup_neg_only),
            "lambda_gain": float(lambda_gain),
            "lambda_dup": float(lambda_dup),
            "lambda_unique": float(lambda_unique),
            "unique_pos_only": bool(args.unique_pos_only),
            "unique_neg_weight": float(args.unique_neg_weight),
            "dup_weight_power": float(args.dup_weight_power),
            "enable_pairwise_dup": bool(pair_i_tr.size > 0),
            "lambda_pair_dup": float(args.lambda_pair_dup),
            "pair_rank_margin_eps": float(args.pair_rank_margin_eps),
            "pair_conflict_dup_min": float(args.pair_conflict_dup_min),
            "pair_rank_edges_train": int(pair_i_tr.size),
            "target_recall": target_recall,
            "target_select_metric": target_select_metric,
            "weight_label_source": str(args.weight_label_source),
            "aux_mask_source": str(args.aux_mask_source),
            "aux_mask_thr": float(args.aux_mask_thr),
            "eval_label_source": str(args.eval_label_source),
            "eval_label_thr": float(args.eval_label_thr),
            "pair_warp_relax": float(args.pair_warp_relax),
        },
        "heads": {
            "class_enabled": bool(class_enabled),
            "num_classes": int(len(class_labels)) if class_enabled else 0,
            "lambda_cls": float(args.lambda_cls),
            "class_pos_only": bool(args.class_pos_only),
            "attr_enabled": bool(attr_enabled),
            "num_attrs": int(len(attr_ids)) if attr_enabled else 0,
            "lambda_attr": float(args.lambda_attr),
            "attr_pos_only": bool(args.attr_pos_only),
        },
        "threshold_grid": [float(x) for x in thr_grid],
        "dtype": str(args.dtype).lower(),
        "args": vars(args),
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  ckpt: {out_ckpt}")
    print(f"  summary: {out_summary}")
    print(
        f"  best@thr={best_thr:.3f} "
        f"P={best_metrics['P']:.4f} R={best_metrics['R']:.4f} "
        f"F1={best_metrics['F1']:.4f} F2={best_metrics['F2']:.4f}"
    )


if __name__ == "__main__":
    main()
