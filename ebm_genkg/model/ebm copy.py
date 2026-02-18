# ebm_genkg/model/ebm.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EBM-lite inference core (no training) for detection refinement.

This module defines:
- EBMConfig: hyperparameters for energy terms & solver behavior
- ebm_infer_frame(frame, cfg): run EBM-lite selection on one FrameData, return payload
- ebm_infer_frames_by_scene(frames_by_scene, cfg): convenience wrapper to produce repl_map

Energy formulation (subset selection, z_i ∈ {0,1}):
  E(S) = Σ_{i∈S} E_unary(i) + Σ_{(i,j)∈S} E_pair(i,j)

We choose S by greedy approximate minimization:
- unary keep term uses *negative log-odds* so good candidates can reduce energy:
    E_keep_on(i) = - w_keep * logit(p_keep(i))
  (=> if p_keep > 0.5, logit>0, E_keep_on negative, selection becomes favorable)

- pair term is "duplicate penalty" (NMS-like):
    if too close => very large energy (hard constraint),
    else optional soft repulsion.

- attribute term:
    For each selected detection we also infer attr ∈ {moving, standing, parked}.
    We compute p(attr | speed) and add:
        E_attr(i) = w_attr * (-log max_attr_prob)
    plus output attr = argmax.

- dt-support:
    We compute support per spatial cell across dt candidates; support increases keep logit
    (interpreted as higher posterior odds, i.e., lower energy).

Notes:
- This is an MVP “EBM-lite” core meant to replace/augment infer.py heuristics.
- It assumes you already built candidates (raw + optional warp) in data.py.

Import expectations:
- Your repo root (ebm_genkg/) is on sys.path (scripts already do this).
- data.py defines Candidate and FrameData.

If you put this under ebm_genkg/model/, make sure you also have:
  ebm_genkg/model/__init__.py
so `from model.ebm import ...` works.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from data import Candidate, FrameData  # repo-root import (scripts add ebm_genkg to sys.path)


# -----------------------------------------------------------------------------
# Small math helpers
# -----------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    # stable sigmoid
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


def _softmax_logits(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / np.clip(s, 1e-12, None)


def _finite(x: np.ndarray) -> bool:
    return bool(np.isfinite(x).all())


def _cand_xy(c: Candidate) -> Tuple[float, float]:
    b = np.asarray(c.box, dtype=np.float32)
    return float(b[0]), float(b[1])


def _cand_xyz(c: Candidate) -> Tuple[float, float, float]:
    b = np.asarray(c.box, dtype=np.float32)
    return float(b[0]), float(b[1]), float(b[2])


def _cand_speed(c: Candidate) -> float:
    b = np.asarray(c.box, dtype=np.float32)
    if b.shape[0] >= 9:
        vx, vy = float(b[7]), float(b[8])
        if not (np.isfinite(vx) and np.isfinite(vy)):
            return 0.0
        return float(np.hypot(vx, vy))
    return 0.0


def _dist2_xy(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> float:
    dx = float(a_xy[0] - b_xy[0])
    dy = float(a_xy[1] - b_xy[1])
    return dx * dx + dy * dy


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------

@dataclass
class AttrConfig:
    """
    Attribute inference from speed (sqrt(vx^2 + vy^2)).
    We use a simple 3-component Gaussian-like softmax over speed:
      parked  ~ mu=0
      standing~ mu=(parked_thr+moving_thr)/2
      moving  ~ mu=moving_thr + moving_mu_offset
    """
    parked_thr: float = 0.10
    moving_thr: float = 0.50
    moving_mu_offset: float = 1.00
    sigma: float = 0.40  # speed sigma for softmax
    names: Tuple[str, str, str] = ("moving", "standing", "parked")

    # optional: label ids that should be treated as static (force parked)
    static_label_ids: Optional[Set[int]] = None

    # cell-based aggregation: use cell-average speed to stabilize attr
    use_cell_speed: bool = True


@dataclass
class EBMConfig:
    # candidate eligibility
    use_sources: str = "all"  # "raw" or "all"
    raw_only_score_min: float = 0.0

    # keep probability calibration (score is assumed in [0,1])
    # p_keep = sigmoid( logit(score)/temp + log_odds_bias(source) + support_gain )
    temp_raw: float = 1.0
    temp_warp: float = 1.2
    temp_other: float = 1.1

    # source prior in odds domain (odds *= bias_odds)
    bias_odds_raw: float = 1.0
    bias_odds_warp: float = 0.7
    bias_odds_other: float = 0.85

    # dt support (adds to logit): logit += support_logit_gain * (support_dt - 1)
    support_cell_xy: float = 1.0   # meters (grid hashing)
    support_cell_z: float = 2.0    # meters
    support_logit_gain: float = 0.35
    support_use_unique_dt: bool = True  # if True, support = #unique dt in cell; else #cands in cell

    # energy weights
    w_keep: float = 1.0
    w_attr: float = 0.15
    w_pair: float = 1.0

    # selection thresholds (probability thresholds on p_keep)
    keep_thr: float = 0.50
    topk: int = 400

    # pairwise (NMS-like)
    nms_thr_xy: float = 1.0
    nms_cross_class: bool = False
    hard_nms: bool = True                 # if True, treat conflicts as +INF energy
    soft_pair_scale: float = 5.0          # for soft penalty: E_pair += w_pair * soft_pair_scale * (1 - d/thr)

    # optional per-class cap
    max_per_class: Optional[int] = None

    # 2-stage fill (补洞) using dt-support
    two_stage: bool = True
    seed_sources: str = "raw"             # "raw" or "all"
    seed_keep_thr: float = 0.50           # high-precision seeds
    fill_keep_thr: float = 0.35           # lower threshold for fill
    min_dt_support: int = 2               # require >=2 dt support in cell
    dt_cell_size: float = 1.0             # alias to support_cell_xy (kept for compatibility)
    min_dist_to_seed: float = 0.8
    max_fill: int = 300

    # label vote refinement (helps strict-label recall)
    enable_label_vote: bool = True
    label_vote_radius: float = 1.5
    label_vote_use_sources: str = "all"   # "all" or "raw"
    label_vote_min_mass: float = 0.0      # optional: if total vote mass too small, keep original label

    # attr config
    attr_cfg: AttrConfig = field(default_factory=AttrConfig)


    # debug
    include_sources: bool = True
    include_energy: bool = False          # if True, add per-item energy summary in payload
    inf_energy: float = 1e9


# -----------------------------------------------------------------------------
# Support / cell hashing
# -----------------------------------------------------------------------------

def _cell_key_xyz(x: float, y: float, z: float, cell_xy: float, cell_z: float) -> Tuple[int, int, int]:
    gx = int(np.round(x / max(cell_xy, 1e-6)))
    gy = int(np.round(y / max(cell_xy, 1e-6)))
    gz = int(np.round(z / max(cell_z, 1e-6)))
    return gx, gy, gz


def build_cell_stats(
    cands: List[Candidate],
    cfg: EBMConfig,
) -> Tuple[Dict[Tuple[int, int, int], int], Dict[Tuple[int, int, int], Set[int]]]:
    """
    Returns:
      - cell_count: cell -> #cands
      - cell_dtset: cell -> set(dt)
    """
    cell_count: Dict[Tuple[int, int, int], int] = {}
    cell_dtset: Dict[Tuple[int, int, int], Set[int]] = {}

    cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
    cell_z = float(cfg.support_cell_z)

    for c in cands:
        x, y, z = _cand_xyz(c)
        ck = _cell_key_xyz(x, y, z, cell_xy, cell_z)
        cell_count[ck] = cell_count.get(ck, 0) + 1
        if ck not in cell_dtset:
            cell_dtset[ck] = set()
        cell_dtset[ck].add(int(getattr(c, "from_dt", 0)))

    return cell_count, cell_dtset


def cell_support_value(
    c: Candidate,
    cfg: EBMConfig,
    cell_count: Dict[Tuple[int, int, int], int],
    cell_dtset: Dict[Tuple[int, int, int], Set[int]],
) -> int:
    cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
    cell_z = float(cfg.support_cell_z)
    x, y, z = _cand_xyz(c)
    ck = _cell_key_xyz(x, y, z, cell_xy, cell_z)
    if cfg.support_use_unique_dt:
        return int(len(cell_dtset.get(ck, set([int(getattr(c, "from_dt", 0))]))))
    return int(cell_count.get(ck, 1))


# -----------------------------------------------------------------------------
# Keep probability / unary energy
# -----------------------------------------------------------------------------

def _source_group(c: Candidate) -> str:
    src = str(getattr(c, "source", ""))
    dt = int(getattr(c, "from_dt", 0))
    if src == "raw" and dt == 0:
        return "raw"
    if src.startswith("warp(") or dt != 0:
        return "warp"
    return "other"


def keep_logit_for_candidate(
    c: Candidate,
    cfg: EBMConfig,
    support_val: int,
) -> float:
    s = float(getattr(c, "score", 0.0))
    s = 0.0 if not np.isfinite(s) else s
    s = float(np.clip(s, 1e-6, 1.0 - 1e-6))

    base_logit = float(_safe_logit(np.array([s], dtype=np.float32))[0])
    grp = _source_group(c)

    if grp == "raw":
        temp = float(cfg.temp_raw)
        bias_odds = float(cfg.bias_odds_raw)
    elif grp == "warp":
        temp = float(cfg.temp_warp)
        bias_odds = float(cfg.bias_odds_warp)
    else:
        temp = float(cfg.temp_other)
        bias_odds = float(cfg.bias_odds_other)

    temp = max(temp, 1e-6)
    bias_odds = max(bias_odds, 1e-6)
    logit = base_logit / temp + float(np.log(bias_odds))

    # support gain in logit
    if support_val > 1 and cfg.support_logit_gain != 0.0:
        logit += float(cfg.support_logit_gain) * float(max(0, support_val - 1))

    return float(logit)


def keep_prob_for_candidate(
    c: Candidate,
    cfg: EBMConfig,
    support_val: int,
) -> float:
    lg = keep_logit_for_candidate(c, cfg, support_val)
    return float(_sigmoid(np.array([lg], dtype=np.float32))[0])


def unary_energy_keep_on(
    keep_logit: float,
    cfg: EBMConfig,
) -> float:
    # E_keep_on = -w_keep * logit(p_keep)  (negative if p_keep>0.5)
    return float(-cfg.w_keep * keep_logit)


# -----------------------------------------------------------------------------
# Attribute probability / energy
# -----------------------------------------------------------------------------

def attr_probs_from_speed(speed: float, cfg_attr: AttrConfig) -> np.ndarray:
    """
    Return probs over [moving, standing, parked] (in that order).
    """
    v = float(speed if np.isfinite(speed) else 0.0)

    if cfg_attr.sigma <= 1e-6:
        # degenerate: hard thresholds
        if v >= cfg_attr.moving_thr:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if v <= cfg_attr.parked_thr:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    mu_moving = float(cfg_attr.moving_thr + cfg_attr.moving_mu_offset)
    mu_parked = 0.0
    mu_stand = float(0.5 * (cfg_attr.parked_thr + cfg_attr.moving_thr))

    mus = np.array([mu_moving, mu_stand, mu_parked], dtype=np.float32)  # [moving, standing, parked]
    vv = np.array([v, v, v], dtype=np.float32)
    sigma2 = float(cfg_attr.sigma * cfg_attr.sigma)

    # logits ~ - (v-mu)^2 / (2*sigma^2)
    logits = - (vv - mus) ** 2 / (2.0 * max(sigma2, 1e-6))
    probs = _softmax_logits(logits, axis=-1)
    return probs.astype(np.float32)


def attr_energy_from_probs(probs: np.ndarray, cfg: EBMConfig) -> float:
    # E_attr = w_attr * (-log max_p)
    pmax = float(np.max(probs)) if probs.size > 0 else 0.0
    pmax = float(np.clip(pmax, 1e-6, 1.0))
    return float(cfg.w_attr * (-np.log(pmax)))


def infer_attr_for_candidate(
    c: Candidate,
    cfg: EBMConfig,
    cell_speed_map: Optional[Dict[Tuple[int, int, int], float]] = None,
) -> Tuple[str, np.ndarray]:
    """
    Return (attr_name, probs[moving,standing,parked]).
    If static_label_ids contains the label, force "parked".
    """
    ac = cfg.attr_cfg
    label = int(getattr(c, "label", -1))

    if ac.static_label_ids is not None and label in ac.static_label_ids:
        probs = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return ac.names[2], probs

    # choose speed source: cell-avg speed for stability, else candidate speed
    if ac.use_cell_speed and cell_speed_map is not None:
        cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
        cell_z = float(cfg.support_cell_z)
        x, y, z = _cand_xyz(c)
        ck = _cell_key_xyz(x, y, z, cell_xy, cell_z)
        v = float(cell_speed_map.get(ck, _cand_speed(c)))
    else:
        v = _cand_speed(c)

    probs = attr_probs_from_speed(v, ac)
    idx = int(np.argmax(probs))
    return ac.names[idx], probs


def build_cell_speed_map(
    cands: List[Candidate],
    cfg: EBMConfig,
    keep_logits: np.ndarray,
) -> Dict[Tuple[int, int, int], float]:
    """
    Compute a stable speed estimate per spatial cell:
      speed(cell) = weighted avg of candidate speeds using keep_prob as weights.
    """
    cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
    cell_z = float(cfg.support_cell_z)

    probs = _sigmoid(keep_logits.astype(np.float32))
    cell_wsum: Dict[Tuple[int, int, int], float] = {}
    cell_vsum: Dict[Tuple[int, int, int], float] = {}

    for i, c in enumerate(cands):
        x, y, z = _cand_xyz(c)
        ck = _cell_key_xyz(x, y, z, cell_xy, cell_z)
        w = float(probs[i])
        v = float(_cand_speed(c))
        if not np.isfinite(v):
            v = 0.0
        cell_wsum[ck] = cell_wsum.get(ck, 0.0) + w
        cell_vsum[ck] = cell_vsum.get(ck, 0.0) + w * v

    out: Dict[Tuple[int, int, int], float] = {}
    for ck, wsum in cell_wsum.items():
        if wsum <= 1e-9:
            out[ck] = 0.0
        else:
            out[ck] = float(cell_vsum.get(ck, 0.0) / wsum)
    return out


# -----------------------------------------------------------------------------
# Pair energy (NMS-like)
# -----------------------------------------------------------------------------

def pair_energy(
    ci: Candidate,
    cj: Candidate,
    cfg: EBMConfig,
) -> float:
    """
    NMS-like duplicate penalty. We mainly operate in XY distance.
    """
    if (not cfg.nms_cross_class) and (int(getattr(ci, "label", -1)) != int(getattr(cj, "label", -1))):
        return 0.0

    thr = float(cfg.nms_thr_xy)
    if thr <= 0:
        return 0.0

    thr2 = thr * thr
    d2 = _dist2_xy(_cand_xy(ci), _cand_xy(cj))
    if d2 > thr2:
        return 0.0

    if cfg.hard_nms:
        return float(cfg.inf_energy)

    d = float(np.sqrt(max(d2, 0.0)))
    # soft penalty: (1 - d/thr) in [0,1]
    pen = 1.0 - min(d / max(thr, 1e-6), 1.0)
    return float(cfg.w_pair * cfg.soft_pair_scale * pen)


# -----------------------------------------------------------------------------
# Label vote refinement
# -----------------------------------------------------------------------------

def refine_label_by_vote(
    target: Candidate,
    cands: List[Candidate],
    keep_probs: np.ndarray,
    cfg: EBMConfig,
) -> int:
    if not cfg.enable_label_vote:
        return int(getattr(target, "label", -1))

    rad = float(cfg.label_vote_radius)
    if rad <= 0:
        return int(getattr(target, "label", -1))
    rad2 = rad * rad

    # choose voters
    if cfg.label_vote_use_sources == "raw":
        voters_idx = [i for i, c in enumerate(cands) if (_source_group(c) == "raw")]
    else:
        voters_idx = list(range(len(cands)))

    tx, ty = _cand_xy(target)
    votes: Dict[int, float] = {}
    mass = 0.0
    for i in voters_idx:
        c = cands[i]
        cx, cy = _cand_xy(c)
        d2 = (cx - tx) ** 2 + (cy - ty) ** 2
        if d2 > rad2:
            continue
        w = float(keep_probs[i])
        lab = int(getattr(c, "label", -1))
        votes[lab] = votes.get(lab, 0.0) + w
        mass += w

    if mass < float(cfg.label_vote_min_mass) or len(votes) == 0:
        return int(getattr(target, "label", -1))

    # argmax vote
    best_lab = max(votes.items(), key=lambda kv: kv[1])[0]
    return int(best_lab)


# -----------------------------------------------------------------------------
# Candidate filtering
# -----------------------------------------------------------------------------

def prefilter_candidates(frame: FrameData, cfg: EBMConfig) -> List[Candidate]:
    return prefilter_candidates_list(getattr(frame, "candidates", []) or [], cfg)

def prefilter_candidates_list(cands: List[Candidate], cfg: EBMConfig) -> List[Candidate]:
    cands = list(cands) if cands is not None else []
    if cfg.use_sources == "raw":
        cands = [c for c in cands if (_source_group(c) == "raw")]
        if cfg.raw_only_score_min > 0.0:
            thr = float(cfg.raw_only_score_min)
            cands = [c for c in cands if float(getattr(c, "score", 0.0)) >= thr]
        return cands
    if cfg.use_sources == "all":
        return cands
    raise ValueError(f"Unknown use_sources={cfg.use_sources}. Use 'raw' or 'all'.")

# -----------------------------------------------------------------------------
# Solver (greedy EBM-lite)
# -----------------------------------------------------------------------------

def _greedy_select(
    pool_idx: List[int],
    cands: List[Candidate],
    keep_logits: np.ndarray,
    keep_probs: np.ndarray,
    attr_energy_arr: np.ndarray,
    cfg: EBMConfig,
    *,
    prob_thr: float,
    already_selected: Optional[List[int]] = None,
    max_add: Optional[int] = None,
    min_dist_to_selected: float = 0.0,
) -> List[int]:

    selected: List[int] = list(already_selected) if already_selected is not None else []
    max_add = int(max_add) if max_add is not None else None

    # 用 nms_thr 作为 grid cell size（同 NMS）
    cell = float(max(cfg.nms_thr_xy, 1e-6))
    min_d2 = float(min_dist_to_selected) ** 2

    # grid: key -> list of selected indices
    # key: (label or -1 if cross_class, ix, iy)
    grid: Dict[Tuple[int, int, int], List[int]] = {}

    def grid_key_for(c: Candidate) -> Tuple[int, int, int]:
        x, y = _cand_xy(c)
        ix, iy = _grid_cell_xy(x, y, cell)
        lab = -1 if cfg.nms_cross_class else int(getattr(c, "label", -1))
        return (lab, ix, iy)

    def insert_idx(i: int):
        k = grid_key_for(cands[i])
        grid.setdefault(k, []).append(i)

    # init grid with already_selected
    for j in selected:
        insert_idx(j)

    # sort by descending keep_logit
    order = sorted(pool_idx, key=lambda i: float(keep_logits[i]), reverse=True)
    kept_count_by_class: Dict[int, int] = {}

    for i in order:
        # stage add limit
        if max_add is not None:
            added = len(selected) - (len(already_selected) if already_selected else 0)
            if added >= max_add:
                break

        p = float(keep_probs[i])
        if p < float(prob_thr):
            continue

        ci = cands[i]
        lab_i = int(getattr(ci, "label", -1))

        # per-class cap
        if cfg.max_per_class is not None:
            if kept_count_by_class.get(lab_i, 0) >= int(cfg.max_per_class):
                continue

        # 只检查邻居格子里的 selected
        x, y = _cand_xy(ci)
        ix, iy = _grid_cell_xy(x, y, cell)
        lab_key = -1 if cfg.nms_cross_class else lab_i

        neighbor_sel: List[int] = []
        for nix, niy in _neighbor_cells(ix, iy):
            neighbor_sel.extend(grid.get((lab_key, nix, niy), []))

        # hard min_dist_to_selected（用于 stage2 vs seeds）
        if min_d2 > 0.0 and neighbor_sel:
            too_close = False
            for j in neighbor_sel:
                xj, yj = _cand_xy(cands[j])
                d2 = (x - xj) ** 2 + (y - yj) ** 2
                if d2 <= min_d2:
                    too_close = True
                    break
            if too_close:
                continue

        # delta energy = unary + pair (只对邻居算 pair)
        unary = unary_energy_keep_on(float(keep_logits[i]), cfg) + float(attr_energy_arr[i])

        pair_sum = 0.0
        reject = False
        for j in neighbor_sel:
            eij = pair_energy(ci, cands[j], cfg)
            if eij >= cfg.inf_energy * 0.5:
                reject = True
                break
            pair_sum += float(eij)
        if reject:
            continue

        if float(unary + pair_sum) < 0.0:
            selected.append(i)
            insert_idx(i)
            if cfg.max_per_class is not None:
                kept_count_by_class[lab_i] = kept_count_by_class.get(lab_i, 0) + 1

        # 全局 topk
        if cfg.topk is not None and len(selected) >= int(cfg.topk):
            break

    return selected


def _grid_cell_xy(x: float, y: float, cell: float) -> Tuple[int, int]:
    cs = float(max(cell, 1e-6))
    return int(np.floor(x / cs)), int(np.floor(y / cs))

def _neighbor_cells(ix: int, iy: int) -> List[Tuple[int, int]]:
    return [(ix+dx, iy+dy) for dx in (-1,0,1) for dy in (-1,0,1)]

def solve_candidates(
    cands: List[Candidate],
    cfg: EBMConfig,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Solve one frame given candidate list directly.
    Returns indices w.r.t. this (prefiltered) candidate list.
    """
    prefilter_topm_seed: int = 1500
    prefilter_topm_fill: int = 3000

    cands = prefilter_candidates_list(cands, cfg)
    N = len(cands)
    if N == 0:
        return [], {"num_cands": 0, "selected": 0}

    # ====== 下面基本就是你 solve_frame 里从 cell_stats 开始的原逻辑 ======
    cell_count, cell_dtset = build_cell_stats(cands, cfg)

    keep_logits = np.zeros((N,), dtype=np.float32)
    keep_probs = np.zeros((N,), dtype=np.float32)
    support_vals = np.zeros((N,), dtype=np.int64)

    for i, c in enumerate(cands):
        sv = cell_support_value(c, cfg, cell_count, cell_dtset)
        support_vals[i] = int(sv)
        lg = keep_logit_for_candidate(c, cfg, int(sv))
        keep_logits[i] = float(lg)
        keep_probs[i] = float(_sigmoid(np.array([lg], dtype=np.float32))[0])

    cell_speed_map = build_cell_speed_map(cands, cfg, keep_logits) if cfg.attr_cfg.use_cell_speed else None

    attr_energy_arr = np.zeros((N,), dtype=np.float32)
    for i, c in enumerate(cands):
        _, probs = infer_attr_for_candidate(c, cfg, cell_speed_map=cell_speed_map)
        attr_energy_arr[i] = float(attr_energy_from_probs(probs, cfg))

    selected_idx: List[int] = []

    if cfg.two_stage:
        if cfg.seed_sources == "raw":
            pool_seed = [i for i, c in enumerate(cands) if _source_group(c) == "raw"]
        else:
            pool_seed = list(range(N))

        selected_idx = _greedy_select(
            pool_idx=pool_seed,
            cands=cands,
            keep_logits=keep_logits,
            keep_probs=keep_probs,
            attr_energy_arr=attr_energy_arr,
            cfg=cfg,
            prob_thr=float(cfg.seed_keep_thr),
            already_selected=[],
            max_add=None,
            min_dist_to_selected=0.0,
        )

        sel_set = set(selected_idx)
        pool_fill = []
        for i, c in enumerate(cands):
            if i in sel_set:
                continue
            if float(keep_probs[i]) < float(cfg.fill_keep_thr):
                continue
            if int(support_vals[i]) < int(cfg.min_dt_support):
                continue
            pool_fill.append(i)

        selected_idx = _greedy_select(
            pool_idx=pool_fill,
            cands=cands,
            keep_logits=keep_logits,
            keep_probs=keep_probs,
            attr_energy_arr=attr_energy_arr,
            cfg=cfg,
            prob_thr=float(cfg.fill_keep_thr),
            already_selected=selected_idx,
            max_add=int(cfg.max_fill),
            min_dist_to_selected=float(cfg.min_dist_to_seed),
        )
    else:
        selected_idx = _greedy_select(
            pool_idx=list(range(N)),
            cands=cands,
            keep_logits=keep_logits,
            keep_probs=keep_probs,
            attr_energy_arr=attr_energy_arr,
            cfg=cfg,
            prob_thr=float(cfg.keep_thr),
            already_selected=[],
            max_add=None,
            min_dist_to_selected=0.0,
        )

    if cfg.topk is not None and len(selected_idx) > int(cfg.topk):
        selected_idx = sorted(selected_idx, key=lambda i: float(keep_logits[i]), reverse=True)[: int(cfg.topk)]

    dbg: Dict[str, Any] = {"num_cands": int(N), "selected": int(len(selected_idx))}
    return selected_idx, dbg

def solve_frame(frame: FrameData, cfg: EBMConfig) -> Tuple[List[int], Dict[str, Any]]:
    return solve_candidates(getattr(frame, "candidates", []) or [], cfg)


# -----------------------------------------------------------------------------
# Public API: inference payloads
# -----------------------------------------------------------------------------

def ebm_infer_candidates(cands: List[Candidate], cfg: EBMConfig) -> Dict[str, Any]:
    cands = prefilter_candidates_list(cands, cfg)
    if len(cands) == 0:
        out: Dict[str, Any] = {"boxes_3d": [], "labels_3d": [], "scores_3d": [], "attrs": []}
        if cfg.include_sources:
            out["sources"] = []
        if cfg.include_energy:
            out["energy"] = {"num_cands": 0, "selected": 0}
        return out

    sel_idx, dbg = solve_candidates(cands, cfg)
    if len(sel_idx) == 0:
        out = {"boxes_3d": [], "labels_3d": [], "scores_3d": [], "attrs": []}
        if cfg.include_sources:
            out["sources"] = []
        if cfg.include_energy:
            out["energy"] = dbg
        return out

    # 重新算 keep_probs / cell_speed_map / attr（保持你现有逻辑）
    cell_count, cell_dtset = build_cell_stats(cands, cfg)
    support_vals = np.array([cell_support_value(c, cfg, cell_count, cell_dtset) for c in cands], dtype=np.int64)
    keep_logits = np.array([keep_logit_for_candidate(c, cfg, int(support_vals[i])) for i, c in enumerate(cands)], dtype=np.float32)
    keep_probs = _sigmoid(keep_logits)
    cell_speed_map = build_cell_speed_map(cands, cfg, keep_logits) if cfg.attr_cfg.use_cell_speed else None

    boxes_out: List[List[float]] = []
    labels_out: List[int] = []
    scores_out: List[float] = []
    attrs_out: List[str] = []
    sources_out: List[str] = []

    for i in sel_idx:
        c = cands[i]
        b = np.asarray(c.box, dtype=np.float32)
        if b.ndim != 1 or b.shape[0] < 9 or (not _finite(b[:3])):
            continue

        lab = refine_label_by_vote(c, cands, keep_probs, cfg)
        attr_name, _ = infer_attr_for_candidate(c, cfg, cell_speed_map=cell_speed_map)

        boxes_out.append([float(x) for x in b.tolist()])
        labels_out.append(int(lab))
        scores_out.append(float(keep_probs[i]))
        attrs_out.append(str(attr_name))

        if cfg.include_sources:
            src = str(getattr(c, "source", ""))
            dt = int(getattr(c, "from_dt", 0))
            sources_out.append(f"{src}" if dt == 0 else f"{src}|dt={dt}")

    payload: Dict[str, Any] = {"boxes_3d": boxes_out, "labels_3d": labels_out, "scores_3d": scores_out, "attrs": attrs_out}
    if cfg.include_sources:
        payload["sources"] = sources_out
    if cfg.include_energy:
        payload["energy"] = dbg
    return payload


def ebm_infer_frame(frame: FrameData, cfg: EBMConfig) -> Dict[str, Any]:
    return ebm_infer_candidates(getattr(frame, "candidates", []) or [], cfg)


def ebm_infer_scene(frames: List[FrameData], cfg: EBMConfig, key_fn) -> Dict[str, Dict[str, Any]]:
    """
    Build repl_map for one scene.
    key_fn(frame.sample) -> key string (typically sample_key_candidates -> first available).
    """
    repl: Dict[str, Dict[str, Any]] = {}
    for fr in frames:
        payload = ebm_infer_frame(fr, cfg)
        key = key_fn(fr.sample)
        repl[str(key)] = payload
    return repl


def ebm_infer_frames_by_scene(
    frames_by_scene: Dict[str, List[FrameData]],
    cfg: EBMConfig,
    key_fn,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience wrapper: infer all scenes => global repl_map.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for _, frames in frames_by_scene.items():
        out.update(ebm_infer_scene(frames, cfg, key_fn=key_fn))
    return out

# -----------------------------------------------------------------------------
# Exported class wrapper (what infer.py expects)
# -----------------------------------------------------------------------------

def _choose_key_from_candidates(sample: Dict[str, Any], key_fn) -> str:
    """
    key_fn can be:
      - sample_key_candidates(sample) -> List[str]
      - or a function returning a single key
    We pick the first non-empty key.
    """
    try:
        ks = key_fn(sample)
    except Exception:
        ks = None

    if isinstance(ks, (list, tuple)):
        for k in ks:
            if k is None:
                continue
            s = str(k)
            if s != "":
                return s
        return ""
    if ks is None:
        return ""
    return str(ks)

class EBMRefiner:
    def __init__(self, cfg: Optional[EBMConfig] = None):
        self.cfg = cfg if cfg is not None else EBMConfig()

    def infer_frame(self, frame: FrameData, candidates=None, infer_cfg=None, **kwargs) -> Dict[str, Any]:
        base = candidates if candidates is not None else (getattr(frame, "candidates", []) or [])
        return ebm_infer_candidates(base, self.cfg)

    # alias (just in case infer.py uses other names)
    def refine_frame(self, frame: FrameData, candidates=None, infer_cfg=None, **kwargs) -> Dict[str, Any]:
        return self.infer_frame(frame=frame, candidates=candidates, infer_cfg=infer_cfg, **kwargs)

    def infer_frames_by_scene(self, frames_by_scene: Dict[str, List[FrameData]], key_fn) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for _, frames in frames_by_scene.items():
            for fr in frames:
                payload = self.infer_frame(fr)
                key = _choose_key_from_candidates(fr.sample, key_fn)
                if key == "":
                    key = f"{fr.scene_token}:{fr.timestamp}"
                out[key] = payload
        return out



    def refine_frames_by_scene(self, frames_by_scene: Dict[str, List[FrameData]], key_fn) -> Dict[str, Dict[str, Any]]:
        return self.infer_frames_by_scene(frames_by_scene, key_fn)


# Optional: make exports explicit
__all__ = [
    "AttrConfig",
    "EBMConfig",
    "EBMRefiner",
    "ebm_infer_frame",
    "ebm_infer_frames_by_scene",
]
