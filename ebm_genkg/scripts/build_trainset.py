#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build training dataset for learned EBM unary terms.

Output:
- NPZ fields:
  X            float32 [N, D]
  y_keep       int64   [N]   (0/1)
  y_cls        int64   [N]   (matched class, -100 for unmatched)
  y_attr       int64   [N]   (attr id, -1 unknown/unmatched)
  cand_label   int64   [N]   (candidate original label)
  cand_score   float32 [N]
  from_dt      int32   [N]
  scene_idx    int32   [N]
  frame_idx    int32   [N]
  feature_names object [D]

- Meta JSON:
  dataset stats + attr vocab + args snapshot.

Example:
  python scripts/build_trainset.py \
    --in_json /path/to/train.json \
    --out_npz /path/to/trainset.npz \
    --use_warp --warp_radius 2 --warp_topk 120 --warp_decay 0.9 \
    --target_candidates all \
    --match_thr 2.0 --ignore_classes -1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from array import array
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from json_io import load_root_and_samples, group_by_scene
from data import (
    Candidate,
    CandidateConfig,
    build_candidates_for_scene,
    build_candidate_gt_links_xy,
    match_all_candidates_to_gt,
    match_raw_det_to_gt,
)


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    flag = "--" + name.replace("_", "-")
    alt_flag = "--" + name
    flags = [flag] if flag == alt_flag else [flag, alt_flag]
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(*flags, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return
    group = parser.add_mutually_exclusive_group()
    group.add_argument(*flags, dest=name, action="store_true", help=help_text)
    group.add_argument(
        "--no-" + name.replace("_", "-"),
        "--no-" + name,
        dest=name,
        action="store_false",
        help=f"Disable {help_text}",
    )
    parser.set_defaults(**{name: default})


def _parse_ignore(s: str) -> Set[int]:
    out: Set[int] = set()
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.add(int(x))
    return out


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


def _cand_speed(c: Candidate) -> float:
    b = np.asarray(c.box, dtype=np.float32)
    if b.shape[0] < 9:
        return 0.0
    vx, vy = float(b[7]), float(b[8])
    if not (np.isfinite(vx) and np.isfinite(vy)):
        return 0.0
    return float(np.hypot(vx, vy))


def _source_flags(c: Candidate) -> Tuple[float, float, float]:
    src = str(getattr(c, "source", ""))
    dt = int(getattr(c, "from_dt", 0))
    is_raw = 1.0 if (src == "raw" and dt == 0) else 0.0
    is_warp = 1.0 if (src.startswith("warp(") or dt != 0) else 0.0
    is_other = 1.0 - min(1.0, is_raw + is_warp)
    return is_raw, is_warp, is_other


def _support_stats(
    cands: List[Candidate],
    cell: float,
) -> Tuple[
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int], int],
    Dict[Tuple[int, int, int], float],
    Dict[Tuple[int, int, int], float],
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int, int], int],
]:
    """
    Return:
      - count_map: keyed by (label, gx, gy)
      - unique_dt_count_map: keyed by (label, gx, gy)
      - local_density_map: keyed by (gx, gy), counts all labels in the same cell
      - score_sum_map: keyed by (label, gx, gy)
      - abs_dt_sum_map: keyed by (label, gx, gy)
      - raw_count_map: keyed by (label, gx, gy)
      - warp_count_map: keyed by (label, gx, gy)
    """
    cs = float(max(cell, 1e-6))
    count_map: Dict[Tuple[int, int, int], int] = {}
    dt_map: Dict[Tuple[int, int, int], Set[int]] = {}
    local_density_map: Dict[Tuple[int, int], int] = {}
    score_sum_map: Dict[Tuple[int, int, int], float] = {}
    abs_dt_sum_map: Dict[Tuple[int, int, int], float] = {}
    raw_count_map: Dict[Tuple[int, int, int], int] = {}
    warp_count_map: Dict[Tuple[int, int, int], int] = {}

    for c in cands:
        b = np.asarray(c.box, dtype=np.float32)
        x, y = float(b[0]), float(b[1])
        gx = int(np.round(x / cs))
        gy = int(np.round(y / cs))
        k = (int(c.label), gx, gy)
        count_map[k] = count_map.get(k, 0) + 1
        dt_map.setdefault(k, set()).add(int(c.from_dt))
        score_v = float(getattr(c, "score", 0.0))
        score_v = score_v if np.isfinite(score_v) else 0.0
        score_sum_map[k] = float(score_sum_map.get(k, 0.0) + score_v)
        abs_dt_sum_map[k] = float(abs_dt_sum_map.get(k, 0.0) + float(abs(int(getattr(c, "from_dt", 0)))))
        src = str(getattr(c, "source", ""))
        dt = int(getattr(c, "from_dt", 0))
        if src == "raw" and dt == 0:
            raw_count_map[k] = int(raw_count_map.get(k, 0) + 1)
        if src.startswith("warp(") or dt != 0:
            warp_count_map[k] = int(warp_count_map.get(k, 0) + 1)
        k2 = (gx, gy)
        local_density_map[k2] = local_density_map.get(k2, 0) + 1

    unique_dt_count = {k: len(v) for k, v in dt_map.items()}
    return (
        count_map,
        unique_dt_count,
        local_density_map,
        score_sum_map,
        abs_dt_sum_map,
        raw_count_map,
        warp_count_map,
    )


def _feature_row(
    c: Candidate,
    count_map: Dict[Tuple[int, int, int], int],
    unique_dt_count_map: Dict[Tuple[int, int, int], int],
    local_density_map: Dict[Tuple[int, int], int],
    score_sum_map: Dict[Tuple[int, int, int], float],
    abs_dt_sum_map: Dict[Tuple[int, int, int], float],
    raw_count_map: Dict[Tuple[int, int, int], int],
    warp_count_map: Dict[Tuple[int, int, int], int],
    cell: float,
) -> List[float]:
    b = np.asarray(c.box, dtype=np.float32)
    if b.shape[0] < 9:
        b = np.pad(b, (0, max(0, 9 - b.shape[0])), mode="constant")

    score = float(c.score) if np.isfinite(float(c.score)) else 0.0
    label = float(int(c.label))
    from_dt = float(int(c.from_dt))
    abs_dt = float(abs(int(c.from_dt)))
    speed = _cand_speed(c)

    is_raw, is_warp, is_other = _source_flags(c)

    x, y, z, dx, dy, dz, yaw, vx, vy = [float(v) if np.isfinite(float(v)) else 0.0 for v in b[:9]]

    cs = float(max(cell, 1e-6))
    gx = int(np.round(x / cs))
    gy = int(np.round(y / cs))
    k = (int(c.label), gx, gy)
    support_count = float(count_map.get(k, 1))
    support_unique_dt = float(unique_dt_count_map.get(k, 1))
    temporal_stability = float(support_unique_dt / max(1.0, support_count))
    local_density = float(local_density_map.get((gx, gy), 1))
    score_sum = float(score_sum_map.get(k, score))
    abs_dt_sum = float(abs_dt_sum_map.get(k, abs_dt))
    raw_count = float(raw_count_map.get(k, 0))
    warp_count = float(warp_count_map.get(k, 0))
    support_score_mean = float(score_sum / max(1.0, support_count))
    support_abs_dt_mean = float(abs_dt_sum / max(1.0, support_count))
    support_raw_ratio = float(raw_count / max(1.0, support_count))
    support_warp_ratio = float(warp_count / max(1.0, support_count))
    temporal_local_interaction = float(temporal_stability * np.log1p(max(0.0, local_density - 1.0)))
    support_density_interaction = float(support_count / max(1.0, local_density))

    return [
        score,
        label,
        is_raw,
        is_warp,
        is_other,
        from_dt,
        abs_dt,
        speed,
        x,
        y,
        z,
        dx,
        dy,
        dz,
        yaw,
        vx,
        vy,
        support_count,
        support_unique_dt,
        temporal_stability,
        local_density,
        support_score_mean,
        support_abs_dt_mean,
        support_raw_ratio,
        support_warp_ratio,
        temporal_local_interaction,
        support_density_interaction,
    ]


def _feature_names() -> List[str]:
    return [
        "score",
        "label",
        "is_raw",
        "is_warp",
        "is_other",
        "from_dt",
        "abs_dt",
        "speed",
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
        "yaw",
        "vx",
        "vy",
        "support_count",
        "support_unique_dt",
        "temporal_stability",
        "local_density",
        "support_score_mean",
        "support_abs_dt_mean",
        "support_raw_ratio",
        "support_warp_ratio",
        "temporal_local_interaction",
        "support_density_interaction",
    ]


def _pair_overlap_min_xy(ci: Candidate, cj: Candidate) -> float:
    bi = np.asarray(ci.box, dtype=np.float32)
    bj = np.asarray(cj.box, dtype=np.float32)
    if bi.shape[0] < 5 or bj.shape[0] < 5:
        return 0.0
    xi, yi, dxi, dyi = float(bi[0]), float(bi[1]), abs(float(bi[3])), abs(float(bi[4]))
    xj, yj, dxj, dyj = float(bj[0]), float(bj[1]), abs(float(bj[3])), abs(float(bj[4]))
    if dxi <= 1e-6 or dyi <= 1e-6 or dxj <= 1e-6 or dyj <= 1e-6:
        return 0.0
    li, ri = xi - 0.5 * dxi, xi + 0.5 * dxi
    biy, tiy = yi - 0.5 * dyi, yi + 0.5 * dyi
    lj, rj = xj - 0.5 * dxj, xj + 0.5 * dxj
    bjy, tjy = yj - 0.5 * dyj, yj + 0.5 * dyj
    iw = max(0.0, min(ri, rj) - max(li, lj))
    ih = max(0.0, min(tiy, tjy) - max(biy, bjy))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    return float(inter / max(min(dxi * dyi, dxj * dyj), 1e-6))


def _build_local_neighbors(
    cands: List[Candidate],
    radius: float,
) -> List[List[Tuple[float, int]]]:
    n = len(cands)
    if n == 0:
        return []
    r = float(max(radius, 1e-6))
    cell = r
    buckets: Dict[Tuple[int, int, int], List[int]] = {}

    for i, c in enumerate(cands):
        x, y = float(c.box[0]), float(c.box[1])
        ix = int(np.floor(x / cell))
        iy = int(np.floor(y / cell))
        k = (int(c.label), ix, iy)
        buckets.setdefault(k, []).append(int(i))

    out: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
    for i, ci in enumerate(cands):
        xi, yi = float(ci.box[0]), float(ci.box[1])
        li = int(ci.label)
        ix = int(np.floor(xi / cell))
        iy = int(np.floor(yi / cell))
        neigh: List[Tuple[float, int]] = []
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                ids = buckets.get((li, ix + dx_cell, iy + dy_cell), [])
                for j in ids:
                    if j == i:
                        continue
                    xj, yj = float(cands[j].box[0]), float(cands[j].box[1])
                    d = float(np.hypot(xj - xi, yj - yi))
                    if d <= r:
                        neigh.append((d, int(j)))
        out[i] = neigh
    return out


def _compute_dup_risk(cands: List[Candidate], radius: float, neighbors: Optional[List[List[Tuple[float, int]]]] = None) -> np.ndarray:
    n = len(cands)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    r = float(max(radius, 1e-6))
    neigh = neighbors if neighbors is not None else _build_local_neighbors(cands, radius=r)
    out = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        best = 0.0
        for d, j in neigh[i]:
            prox = 1.0 - min(float(d) / r, 1.0)
            best = max(best, prox * float(max(0.0, min(1.0, float(cands[j].score)))))
        out[i] = np.float32(best)
    return out


def _compute_cover_gain(cands: List[Candidate], links) -> np.ndarray:
    n = len(cands)
    gain = np.zeros((n,), dtype=np.float32)
    if n == 0:
        return gain

    cand_to_gt = np.asarray(getattr(links, "cand_to_gt", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)
    gt_candidate_counts = np.asarray(getattr(links, "gt_candidate_counts", np.zeros((0,), dtype=np.int64)), dtype=np.float32)
    if cand_to_gt.shape[0] != n:
        m = min(n, cand_to_gt.shape[0])
        tmp = np.full((n,), -1, dtype=np.int64)
        tmp[:m] = cand_to_gt[:m]
        cand_to_gt = tmp

    gain = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        g = int(cand_to_gt[i])
        if g < 0 or g >= gt_candidate_counts.shape[0]:
            continue
        cnt = float(max(1.0, gt_candidate_counts[g]))
        gain[i] = np.float32(1.0 / cnt)

    for best_c in np.asarray(getattr(links, "gt_best_cand", np.zeros((0,), dtype=np.int64)), dtype=np.int64).tolist():
        if 0 <= int(best_c) < n:
            gain[int(best_c)] += np.float32(0.5)
    return gain


def _compute_cover_gain_greedy(cands: List[Candidate], links) -> np.ndarray:
    """
    Approximate marginal coverage using greedy order (score desc, dist asc):
      gain[i] = 1 if candidate i is the first one that covers its linked gt, else 0.
    """
    n = len(cands)
    out = np.zeros((n,), dtype=np.float32)
    if n == 0:
        return out

    cand_to_gt = np.asarray(getattr(links, "cand_to_gt", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)
    cand_to_gt_dist = np.asarray(getattr(links, "cand_to_gt_dist", np.full((n,), np.inf, dtype=np.float32)), dtype=np.float32)
    scores = np.asarray([float(c.score) for c in cands], dtype=np.float32)
    order = np.lexsort((cand_to_gt_dist, -scores))
    covered: Set[int] = set()
    for i in order.tolist():
        g = int(cand_to_gt[i]) if i < cand_to_gt.shape[0] else -1
        if g < 0:
            continue
        if g in covered:
            continue
        covered.add(g)
        out[i] = np.float32(1.0)
    return out


def _compute_keep_soft_targets(
    links,
    *,
    n_cands: int,
    match_thr: float,
    tau: float,
    best_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Soft keep target:
      - 0 for unmatched candidates
      - exp(-(d/tau)^2 / 2) for matched candidates (d: cand->nearest-feasible-gt XY distance)
      - force best-linked candidates to at least 1.0 to anchor unique primary matches
    """
    out = np.zeros((int(n_cands),), dtype=np.float32)
    if int(n_cands) <= 0:
        return out
    cand_to_gt = np.asarray(getattr(links, "cand_to_gt", np.full((n_cands,), -1, dtype=np.int64)), dtype=np.int64)
    cand_to_gt_dist = np.asarray(getattr(links, "cand_to_gt_dist", np.full((n_cands,), np.inf, dtype=np.float32)), dtype=np.float32)
    if cand_to_gt.shape[0] != int(n_cands) or cand_to_gt_dist.shape[0] != int(n_cands):
        return out
    thr = float(max(1e-6, match_thr))
    tau_eff = float(max(1e-6, tau))
    for i in range(int(n_cands)):
        if int(cand_to_gt[i]) < 0:
            continue
        d = float(cand_to_gt_dist[i])
        if not np.isfinite(d) or d > thr:
            continue
        out[i] = np.float32(np.exp(-0.5 * (d / tau_eff) * (d / tau_eff)))
    if best_mask is not None:
        bm = np.asarray(best_mask, dtype=np.int64).reshape(-1)
        if bm.shape[0] == int(n_cands):
            out = np.maximum(out, (bm > 0).astype(np.float32))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _build_pair_edges(
    cands: List[Candidate],
    row_base: int,
    pair_radius: float,
    pair_max_neighbors: int,
    cand_is_gt_best_local: np.ndarray,
    neighbors: Optional[List[List[Tuple[float, int]]]] = None,
) -> Tuple[List[int], List[int], List[List[float]], List[int]]:
    n = len(cands)
    if n <= 1:
        return [], [], [], []

    pr = float(max(pair_radius, 1e-6))
    pii: List[int] = []
    pjj: List[int] = []
    pff: List[List[float]] = []
    psl: List[int] = []

    neigh_all = neighbors if neighbors is not None else _build_local_neighbors(cands, radius=pr)

    for i, ci in enumerate(cands):
        li = int(ci.label)
        dti = int(ci.from_dt)
        si = _cand_speed(ci)
        neigh: List[Tuple[float, int]] = []
        for d, j in neigh_all[i]:
            if int(j) <= i:
                continue
            cj = cands[int(j)]
            if li != int(cj.label):
                continue
            neigh.append((float(d), int(j)))
        if len(neigh) == 0:
            continue
        neigh.sort(key=lambda x: x[0])
        if pair_max_neighbors > 0:
            neigh = neigh[: int(pair_max_neighbors)]

        for d, j in neigh:
            cj = cands[j]
            dtj = int(cj.from_dt)
            sj = _cand_speed(cj)
            close = float(np.exp(-d / pr))
            overlap = _pair_overlap_min_xy(ci, cj)
            feat = [
                1.0,  # same_label
                close,
                overlap,
                float(min(abs(dti - dtj), 8)) / 8.0,
                1.0 if (dti != 0 and dtj != 0) else 0.0,
                1.0 if (dti != 0 or dtj != 0) else 0.0,
                float(min(float(ci.score), float(cj.score))),
                float(min(abs(si - sj), 10.0)) / 10.0,
                1.0 if dti == dtj else 0.0,
            ]
            pii.append(int(row_base + i))
            pjj.append(int(row_base + j))
            pff.append(feat)
            # pair positive if both endpoints are gt-best candidates
            yi = int(cand_is_gt_best_local[i]) if i < cand_is_gt_best_local.shape[0] else 0
            yj = int(cand_is_gt_best_local[j]) if j < cand_is_gt_best_local.shape[0] else 0
            psl.append(1 if (yi > 0 and yj > 0) else 0)
    return pii, pjj, pff, psl


def _build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build NPZ training set from scene JSON for EBM training.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")

    p.add_argument("--in_json", type=str, default=cfg.get("in_json"), required=("in_json" not in cfg))
    p.add_argument("--out_npz", type=str, default=cfg.get("out_npz"), required=("out_npz" not in cfg))
    p.add_argument("--out_meta", type=str, default=cfg.get("out_meta"), help="Optional meta json path.")
    p.add_argument(
        "--max_rows",
        type=int,
        default=cfg.get("max_rows"),
        help="Optional hard cap on exported candidate rows to bound memory usage.",
    )
    p.add_argument(
        "--max_pair_edges",
        type=int,
        default=cfg.get("max_pair_edges"),
        help="Optional hard cap on exported pair edges to bound memory usage.",
    )
    p.add_argument(
        "--log_every_frames",
        type=int,
        default=int(cfg.get("log_every_frames", 200)),
        help="Progress heartbeat during long extraction stage.",
    )

    _add_bool_arg(p, "require_gt", bool(cfg.get("require_gt", True)), "Require GT while loading samples")
    p.add_argument("--limit_scenes", type=int, default=cfg.get("limit_scenes"))

    p.add_argument("--target_candidates", type=str, default=str(cfg.get("target_candidates", "all")),
                   choices=["all", "raw"], help="Which candidates to export/match as training targets.")

    p.add_argument("--match_thr", type=float, default=float(cfg.get("match_thr", 2.0)))
    p.add_argument("--ignore_classes", type=str, default=str(cfg.get("ignore_classes", "-1")))

    _add_bool_arg(p, "use_warp", bool(cfg.get("use_warp", False)), "Use warped candidates from neighbor frames")
    p.add_argument("--warp_radius", type=int, default=int(cfg.get("warp_radius", 2)))
    p.add_argument("--warp_topk", type=int, default=int(cfg.get("warp_topk", 200)))
    p.add_argument("--warp_decay", type=float, default=float(cfg.get("warp_decay", 0.9)))
    p.add_argument("--raw_score_min", type=float, default=float(cfg.get("raw_score_min", 0.0)))
    p.add_argument("--max_per_frame", type=int, default=cfg.get("max_per_frame"))

    p.add_argument("--support_cell", type=float, default=float(cfg.get("support_cell", 1.0)),
                   help="Cell size for support_count/support_unique_dt features.")
    p.add_argument("--dup_radius", type=float, default=float(cfg.get("dup_radius", 1.5)),
                   help="Radius for duplicate-risk feature label.")
    p.add_argument("--pair_radius", type=float, default=float(cfg.get("pair_radius", 2.5)),
                   help="Frame-local pair graph radius for exported pair edges.")
    p.add_argument("--pair_max_neighbors", type=int, default=int(cfg.get("pair_max_neighbors", 12)),
                   help="Max neighbors per node for exported pair edges.")
    _add_bool_arg(
        p,
        "export_pair_graph",
        bool(cfg.get("export_pair_graph", True)),
        "Export sparse pair graph arrays for structured training",
    )
    p.add_argument(
        "--keep_soft_tau",
        type=float,
        default=float(cfg.get("keep_soft_tau", 0.8)),
        help="Distance decay sigma (meters) for soft keep targets y_keep_soft.",
    )

    return p


def main() -> None:
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default=None)
    a0, _ = p0.parse_known_args()

    cfg = _load_cfg(a0.config)
    parser = _build_parser(cfg)
    args = parser.parse_args()

    ignore = _parse_ignore(args.ignore_classes)

    out_npz = os.path.abspath(args.out_npz)
    out_meta = os.path.abspath(args.out_meta) if args.out_meta else (os.path.splitext(out_npz)[0] + ".meta.json")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)

    print("[1/4] Loading & grouping samples...")
    lr = load_root_and_samples(args.in_json, require_gt=bool(args.require_gt), keep_root=False)
    scenes = group_by_scene(lr.samples)
    if args.limit_scenes is not None:
        keys = sorted(list(scenes.keys()))[: int(args.limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
        print(f"  limit_scenes={args.limit_scenes} -> using scenes={len(scenes)}")
    print(f"  samples_loaded={len(lr.samples)} scenes_used={len(scenes)}")

    print("[2/4] Building candidates + extracting features (stream by scene)...")
    cand_cfg = CandidateConfig(
        raw_score_min=float(args.raw_score_min),
        max_per_frame=args.max_per_frame,
        use_warp=bool(args.use_warp),
        warp_radius=int(args.warp_radius),
        warp_topk=int(args.warp_topk),
        warp_score_decay=float(args.warp_decay),
        ignore_classes=ignore,
        match_thr_xy=float(args.match_thr),
    )
    print("[3/4] Extracting features & targets...")
    feat_dim = len(_feature_names())
    pair_feat_dim = 9
    X_flat = array("f")
    y_keep_buf = array("q")
    y_keep_soft_buf = array("f")
    y_cls_buf = array("q")
    y_attr_buf = array("q")
    cand_label_buf = array("q")
    cand_score_buf = array("f")
    from_dt_buf = array("i")
    scene_idx_buf = array("i")
    frame_idx_buf = array("i")
    cand_is_gt_best_buf = array("q")
    y_unique_buf = array("q")
    cand_cover_gain_buf = array("f")
    cand_cover_gain_greedy_buf = array("f")
    cand_dup_risk_buf = array("f")
    gt_best_global_idx_buf = array("q")
    gt_best_scene_idx_buf = array("i")
    gt_best_frame_idx_buf = array("i")
    gt_best_gt_idx_buf = array("i")
    gt_best_gt_label_buf = array("q")

    pair_i_buf = array("q")
    pair_j_buf = array("q")
    pair_feat_flat = array("f")
    pair_label_buf = array("q")

    attr_counter: Counter = Counter()
    total_frames = 0
    total_candidates = 0
    total_pos = 0
    total_gt_best = 0
    total_gt = 0
    row_base = 0
    truncated_rows = False
    truncated_pairs = False
    stop_early = False
    scenes_processed = 0
    attr_vocab: Dict[str, int] = {"": -1}
    next_attr_id = 0

    scene_keys = sorted(list(scenes.keys()))
    for si, sc in enumerate(scene_keys):
        if stop_early:
            break
        frames = build_candidates_for_scene(scenes[sc], cand_cfg)
        scenes_processed += 1
        for fr in frames:
            if stop_early:
                break
            total_frames += 1
            total_gt += int(fr.gt_boxes.shape[0])
            if int(args.log_every_frames) > 0 and (total_frames % int(args.log_every_frames) == 0):
                print(
                    "  progress:"
                    f" frames={total_frames}"
                    f" rows={total_candidates}"
                    f" pair_edges={len(pair_i_buf)}"
                    f" scenes={scenes_processed}/{len(scene_keys)}"
                )

            if args.target_candidates == "all":
                cands = list(fr.candidates)
                mt = match_all_candidates_to_gt(fr, cand_cfg)
            else:
                cands = [c for c in fr.candidates if c.source == "raw" and c.from_dt == 0]
                mt = match_raw_det_to_gt(fr, cand_cfg)

            if len(cands) == 0:
                continue

            if len(cands) != len(mt.keep):
                n = min(len(cands), len(mt.keep))
                cands = cands[:n]
                keep_arr = mt.keep[:n]
                cls_arr = mt.cls_tgt[:n]
                attr_list = mt.attr_tgt[:n]
            else:
                keep_arr = mt.keep
                cls_arr = mt.cls_tgt
                attr_list = mt.attr_tgt

            (
                count_map,
                unique_dt_count_map,
                local_density_map,
                score_sum_map,
                abs_dt_sum_map,
                raw_count_map,
                warp_count_map,
            ) = _support_stats(cands, cell=float(args.support_cell))
            det_boxes = np.stack([c.box for c in cands], axis=0).astype(np.float32, copy=False)
            det_xyz = det_boxes[:, :3].astype(np.float32, copy=False)
            det_labels = np.asarray([int(c.label) for c in cands], dtype=np.int64)
            det_scores = np.asarray([float(c.score) for c in cands], dtype=np.float32)
            links = build_candidate_gt_links_xy(
                det_xyz=det_xyz,
                det_labels=det_labels,
                gt_xyz=fr.gt_boxes[:, :3].astype(np.float32, copy=False),
                gt_labels=fr.gt_labels.astype(np.int64, copy=False),
                thr_xy=float(args.match_thr),
                det_scores=det_scores,
            )
            cand_is_gt_best_local = np.zeros((len(cands),), dtype=np.int64)
            for j in links.gt_best_cand.tolist():
                if int(j) >= 0 and int(j) < len(cands):
                    cand_is_gt_best_local[int(j)] = 1
            cover_gain_local = _compute_cover_gain(
                cands=cands,
                links=links,
            )
            cover_gain_greedy_local = _compute_cover_gain_greedy(cands=cands, links=links)
            keep_soft_local = _compute_keep_soft_targets(
                links=links,
                n_cands=len(cands),
                match_thr=float(args.match_thr),
                tau=float(args.keep_soft_tau),
                best_mask=cand_is_gt_best_local,
            )
            dup_neighbors = _build_local_neighbors(cands, radius=float(args.dup_radius))
            dup_risk_local = _compute_dup_risk(
                cands,
                radius=float(args.dup_radius),
                neighbors=dup_neighbors,
            )

            n_added_frame = 0
            for i, c in enumerate(cands):
                if args.max_rows is not None and total_candidates >= int(args.max_rows):
                    truncated_rows = True
                    stop_early = True
                    break
                feat = _feature_row(
                    c,
                    count_map,
                    unique_dt_count_map,
                    local_density_map,
                    score_sum_map,
                    abs_dt_sum_map,
                    raw_count_map,
                    warp_count_map,
                    cell=float(args.support_cell),
                )
                X_flat.extend([float(x) for x in feat])

                k = int(keep_arr[i])
                cl = int(cls_arr[i])
                at = str(attr_list[i]) if attr_list[i] is not None else ""

                y_keep_buf.append(k)
                y_keep_soft_buf.append(float(keep_soft_local[i]))
                y_cls_buf.append(cl)
                if at not in attr_vocab:
                    attr_vocab[at] = int(next_attr_id)
                    next_attr_id += 1
                y_attr_buf.append(int(attr_vocab.get(at, -1)))
                cand_label_buf.append(int(c.label))
                cand_score_buf.append(float(c.score))
                from_dt_buf.append(int(c.from_dt))
                scene_idx_buf.append(int(si))
                frame_idx_buf.append(int(fr.index_in_scene))
                cand_is_gt_best_buf.append(int(cand_is_gt_best_local[i]))
                y_unique_buf.append(int(cand_is_gt_best_local[i]))
                cand_cover_gain_buf.append(float(cover_gain_local[i]))
                cand_cover_gain_greedy_buf.append(float(cover_gain_greedy_local[i]))
                cand_dup_risk_buf.append(float(dup_risk_local[i]))

                total_candidates += 1
                total_pos += int(k)
                n_added_frame += 1
                if at != "":
                    attr_counter[at] += 1

            for gt_idx, cand_idx in enumerate(links.gt_best_cand.tolist()):
                cj = int(cand_idx)
                if cj < 0 or cj >= int(n_added_frame):
                    continue
                gt_best_global_idx_buf.append(int(row_base + cj))
                gt_best_scene_idx_buf.append(int(si))
                gt_best_frame_idx_buf.append(int(fr.index_in_scene))
                gt_best_gt_idx_buf.append(int(gt_idx))
                gt_best_gt_label_buf.append(int(fr.gt_labels[gt_idx]))
                total_gt_best += 1

            if bool(args.export_pair_graph):
                cands_used = cands[: int(n_added_frame)]
                cand_is_gt_best_used = cand_is_gt_best_local[: int(n_added_frame)]
                pair_neighbors = _build_local_neighbors(cands_used, radius=float(args.pair_radius))
                pii, pjj, pff, psl = _build_pair_edges(
                    cands=cands_used,
                    row_base=int(row_base),
                    pair_radius=float(args.pair_radius),
                    pair_max_neighbors=int(args.pair_max_neighbors),
                    cand_is_gt_best_local=cand_is_gt_best_used,
                    neighbors=pair_neighbors,
                )
                if args.max_pair_edges is not None:
                    rem = int(args.max_pair_edges) - len(pair_i_buf)
                    if rem <= 0:
                        truncated_pairs = True
                        pii, pjj, pff, psl = [], [], [], []
                    elif len(pii) > rem:
                        truncated_pairs = True
                        pii = pii[:rem]
                        pjj = pjj[:rem]
                        pff = pff[:rem]
                        psl = psl[:rem]
                for vv in pii:
                    pair_i_buf.append(int(vv))
                for vv in pjj:
                    pair_j_buf.append(int(vv))
                for ff in pff:
                    pair_feat_flat.extend([float(x) for x in ff])
                for vv in psl:
                    pair_label_buf.append(int(vv))

            row_base += int(n_added_frame)
        if stop_early:
            break

    X_arr = np.frombuffer(X_flat, dtype=np.float32)
    if X_arr.size == 0:
        X_arr = X_arr.reshape(0, feat_dim)
    else:
        X_arr = X_arr.reshape(-1, feat_dim)
    y_keep_arr = np.frombuffer(y_keep_buf, dtype=np.int64)
    y_cls_arr = np.frombuffer(y_cls_buf, dtype=np.int64)
    y_attr = np.frombuffer(y_attr_buf, dtype=np.int64)
    pair_feat_arr = np.frombuffer(pair_feat_flat, dtype=np.float32)
    if pair_feat_arr.size == 0:
        pair_feat_arr = pair_feat_arr.reshape(0, pair_feat_dim)
    else:
        pair_feat_arr = pair_feat_arr.reshape(-1, pair_feat_dim)
    if truncated_rows:
        print(f"  early-stop: reached max_rows={int(args.max_rows)}")
    if truncated_pairs:
        print(f"  pair-edge cap reached: max_pair_edges={int(args.max_pair_edges)}")

    print("[4/4] Saving dataset...")
    np.savez_compressed(
        out_npz,
        X=X_arr,
        y_keep=y_keep_arr,
        y_keep_soft=np.frombuffer(y_keep_soft_buf, dtype=np.float32),
        y_unique=np.frombuffer(y_unique_buf, dtype=np.int64),
        y_cls=y_cls_arr,
        y_attr=y_attr,
        cand_label=np.frombuffer(cand_label_buf, dtype=np.int64),
        cand_score=np.frombuffer(cand_score_buf, dtype=np.float32),
        from_dt=np.frombuffer(from_dt_buf, dtype=np.int32),
        scene_idx=np.frombuffer(scene_idx_buf, dtype=np.int32),
        frame_idx=np.frombuffer(frame_idx_buf, dtype=np.int32),
        cand_is_gt_best=np.frombuffer(cand_is_gt_best_buf, dtype=np.int64),
        cand_cover_gain=np.frombuffer(cand_cover_gain_buf, dtype=np.float32),
        cand_cover_gain_greedy=np.frombuffer(cand_cover_gain_greedy_buf, dtype=np.float32),
        cand_dup_risk=np.frombuffer(cand_dup_risk_buf, dtype=np.float32),
        gt_best_global_idx=np.frombuffer(gt_best_global_idx_buf, dtype=np.int64),
        gt_best_scene_idx=np.frombuffer(gt_best_scene_idx_buf, dtype=np.int32),
        gt_best_frame_idx=np.frombuffer(gt_best_frame_idx_buf, dtype=np.int32),
        gt_best_gt_idx=np.frombuffer(gt_best_gt_idx_buf, dtype=np.int32),
        gt_best_gt_label=np.frombuffer(gt_best_gt_label_buf, dtype=np.int64),
        pair_i=np.frombuffer(pair_i_buf, dtype=np.int64),
        pair_j=np.frombuffer(pair_j_buf, dtype=np.int64),
        pair_feat=pair_feat_arr,
        pair_label=np.frombuffer(pair_label_buf, dtype=np.int64),
        pair_feature_names=np.asarray(
            [
                "same_label",
                "close",
                "overlap",
                "abs_dt_diff",
                "both_warp",
                "either_warp",
                "score_min",
                "speed_diff",
                "same_dt",
            ],
            dtype=object,
        ),
        feature_names=np.array(_feature_names(), dtype=object),
    )

    meta = {
        "in_json": str(args.in_json),
        "out_npz": out_npz,
        "target_candidates": str(args.target_candidates),
        "ignore_classes": sorted(list(ignore)),
        "limit_scenes": args.limit_scenes,
        "num_samples_loaded": int(len(lr.samples)),
        "num_scenes_used": int(scenes_processed),
        "num_frames_used": int(total_frames),
        "num_rows": int(X_arr.shape[0]),
        "num_features": int(X_arr.shape[1] if X_arr.ndim == 2 else 0),
        "num_positive": int(total_pos),
        "pos_rate": float(total_pos / max(1, total_candidates)),
        "soft_pos_mean": float(np.mean(np.frombuffer(y_keep_soft_buf, dtype=np.float32))) if len(y_keep_soft_buf) > 0 else 0.0,
        "num_gt_best_links": int(total_gt_best),
        "gt_best_coverage": float(total_gt_best / max(1, int(total_gt))),
        "num_pair_edges": int(len(pair_i_buf)),
        "truncated_rows": bool(truncated_rows),
        "truncated_pairs": bool(truncated_pairs),
        "max_rows": int(args.max_rows) if args.max_rows is not None else None,
        "max_pair_edges": int(args.max_pair_edges) if args.max_pair_edges is not None else None,
        "cover_gain_mean": float(np.mean(np.frombuffer(cand_cover_gain_buf, dtype=np.float32))) if len(cand_cover_gain_buf) > 0 else 0.0,
        "cover_gain_greedy_mean": float(np.mean(np.frombuffer(cand_cover_gain_greedy_buf, dtype=np.float32))) if len(cand_cover_gain_greedy_buf) > 0 else 0.0,
        "keep_soft_tau": float(args.keep_soft_tau),
        "feature_names": _feature_names(),
        "pair_feature_names": [
            "same_label",
            "close",
            "overlap",
            "abs_dt_diff",
            "both_warp",
            "either_warp",
            "score_min",
            "speed_diff",
            "same_dt",
        ],
        "attr_vocab": attr_vocab,
        "args": vars(args),
    }

    with open(out_meta, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  npz: {out_npz}")
    print(f"  meta: {out_meta}")
    print(f"  rows={meta['num_rows']} pos_rate={meta['pos_rate']:.4f} features={meta['num_features']}")


if __name__ == "__main__":
    main()
