# ebm_genkg/model/ebm.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EBM-lite inference core (no training) for detection refinement.

Exports:
- AttrConfig
- EBMConfig
- EBMRefiner   (what infer.py lazy-import expects)
- ebm_infer_candidates(cands, cfg) -> payload
- ebm_infer_frame(frame, cfg) -> payload
- ebm_infer_frames_by_scene(frames_by_scene, cfg, key_fn) -> repl_map

Design goals:
1) Keep your existing two-stage (seed + fill) behavior.
2) Add attr energy term (moving/standing/parked) because it matters downstream.
3) Avoid any dependency on infer.py (no circular import).
4) Do NOT construct FrameData manually (your FrameData has required fields).
5) Be fast enough: greedy selection uses a grid to avoid O(K^2) pair checks.

Notes:
- Candidate and FrameData are imported from data.py (repo-root import).
- Candidates expected fields:
    c.box: np.ndarray-like length>=9  [x,y,z,dx,dy,dz,yaw,vx,vy]
    c.score: float in [0,1] (or will be clipped)
    c.label: int
    c.source: str ("raw" or "warp(...)")
    c.from_dt: int (0 for raw current frame, +/- for warps)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from data import Candidate, FrameData  # scripts add ebm_genkg/ to sys.path


# -----------------------------------------------------------------------------
# Small math helpers
# -----------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
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


def _finite_vec(x: np.ndarray) -> bool:
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
    We use a 3-component Gaussian-like softmax over speed:
      parked  ~ mu=0
      standing~ mu=(parked_thr+moving_thr)/2
      moving  ~ mu=moving_thr + moving_mu_offset
    """
    parked_thr: float = 0.10
    moving_thr: float = 0.50
    moving_mu_offset: float = 1.00
    sigma: float = 0.40
    names: Tuple[str, str, str] = ("moving", "standing", "parked")

    static_label_ids: Optional[Set[int]] = None
    use_cell_speed: bool = True  # use cell-avg speed to stabilize attr


@dataclass
class EBMConfig:
    # candidate eligibility
    use_sources: str = "all"  # "raw" or "all"
    raw_only_score_min: float = 0.0

    # keep probability calibration (score assumed in [0,1])
    # keep_logit = logit(score)/temp + log(bias_odds) + support_gain
    temp_raw: float = 1.0
    temp_warp: float = 1.2
    temp_other: float = 1.1

    bias_odds_raw: float = 1.0
    bias_odds_warp: float = 0.7
    bias_odds_other: float = 0.85

    # dt-support (adds to logit)
    support_cell_xy: float = 1.0   # meters
    support_cell_z: float = 2.0    # meters
    support_logit_gain: float = 0.35
    support_use_unique_dt: bool = True

    # energy weights
    w_keep: float = 1.0
    w_attr: float = 0.15
    w_pair: float = 1.0

    # selection thresholds (probability thresholds on keep_prob)
    keep_thr: float = 0.50
    topk: int = 400

    # pairwise (NMS-like)
    nms_thr_xy: float = 1.0
    nms_cross_class: bool = False
    hard_nms: bool = True
    soft_pair_scale: float = 5.0

    # optional per-class cap
    max_per_class: Optional[int] = None

    # 2-stage fill
    two_stage: bool = True
    seed_sources: str = "raw"      # "raw" or "all"
    seed_keep_thr: float = 0.50
    fill_keep_thr: float = 0.35
    min_dt_support: int = 2
    dt_cell_size: float = 1.0      # alias to support_cell_xy
    min_dist_to_seed: float = 0.8
    max_fill: int = 300

    # IMPORTANT speed-up: keep only top-M by keep_logit in each stage pool
    prefilter_topm_seed: int = 1500
    prefilter_topm_fill: int = 3000

    # learned unary model (optional)
    unary_use_learned: bool = True
    unary_ckpt_path: Optional[str] = None

    # label vote refinement
    enable_label_vote: bool = False
    label_vote_radius: float = 1.5
    label_vote_use_sources: str = "all"  # "all" or "raw"
    label_vote_min_mass: float = 0.0

    # attr config
    attr_cfg: AttrConfig = field(default_factory=AttrConfig)

    # debug
    include_sources: bool = True
    include_energy: bool = False
    inf_energy: float = 1e9


# -----------------------------------------------------------------------------
# Source grouping
# -----------------------------------------------------------------------------

def _source_group(c: Candidate) -> str:
    src = str(getattr(c, "source", ""))
    dt = int(getattr(c, "from_dt", 0))
    if src == "raw" and dt == 0:
        return "raw"
    if src.startswith("warp(") or dt != 0:
        return "warp"
    return "other"


_UNARY_MODEL_CACHE: Dict[str, Optional["_UnaryLinearModel"]] = {}


class _UnaryLinearModel:
    """
    Lightweight runtime for train_unary.py checkpoints.
    """

    def __init__(self, ckpt: Dict[str, Any]):
        self.model_type = str(ckpt.get("model_type", ""))
        self.feature_names = [str(x) for x in ckpt.get("feature_names", [])]
        self.normalize = bool(ckpt.get("normalize", False))

        self.mu = np.asarray(ckpt.get("mu", []), dtype=np.float64).reshape(-1)
        self.std = np.asarray(ckpt.get("std", []), dtype=np.float64).reshape(-1)
        self.w = np.asarray(ckpt.get("weights", []), dtype=np.float64).reshape(-1)
        self.b = float(ckpt.get("bias", 0.0))

        if self.model_type != "logistic_unary":
            raise ValueError(f"Unsupported unary model_type: {self.model_type}")
        if len(self.feature_names) != self.w.shape[0]:
            raise ValueError("feature_names and weights length mismatch in unary checkpoint")
        if self.normalize:
            if self.mu.shape[0] != self.w.shape[0] or self.std.shape[0] != self.w.shape[0]:
                raise ValueError("mu/std length mismatch in unary checkpoint")

    def _feature_dict(
        self,
        c: Candidate,
        support_count: int,
        support_unique_dt: int,
    ) -> Dict[str, float]:
        b = np.asarray(c.box, dtype=np.float32)
        if b.shape[0] < 9:
            b = np.pad(b, (0, max(0, 9 - b.shape[0])), mode="constant")

        score = float(getattr(c, "score", 0.0))
        score = score if np.isfinite(score) else 0.0
        label = float(int(getattr(c, "label", -1)))
        dt = int(getattr(c, "from_dt", 0))
        abs_dt = float(abs(dt))
        speed = _cand_speed(c)

        is_raw = 1.0 if _source_group(c) == "raw" else 0.0
        is_warp = 1.0 if _source_group(c) == "warp" else 0.0
        is_other = 1.0 - min(1.0, is_raw + is_warp)

        x = float(b[0]) if np.isfinite(float(b[0])) else 0.0
        y = float(b[1]) if np.isfinite(float(b[1])) else 0.0
        z = float(b[2]) if np.isfinite(float(b[2])) else 0.0
        dx = float(b[3]) if np.isfinite(float(b[3])) else 0.0
        dy = float(b[4]) if np.isfinite(float(b[4])) else 0.0
        dz = float(b[5]) if np.isfinite(float(b[5])) else 0.0
        yaw = float(b[6]) if np.isfinite(float(b[6])) else 0.0
        vx = float(b[7]) if np.isfinite(float(b[7])) else 0.0
        vy = float(b[8]) if np.isfinite(float(b[8])) else 0.0

        return {
            "score": score,
            "label": label,
            "is_raw": is_raw,
            "is_warp": is_warp,
            "is_other": is_other,
            "from_dt": float(dt),
            "abs_dt": abs_dt,
            "speed": float(speed),
            "x": x,
            "y": y,
            "z": z,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "yaw": yaw,
            "vx": vx,
            "vy": vy,
            "support_count": float(support_count),
            "support_unique_dt": float(support_unique_dt),
        }

    def predict_logits(
        self,
        cands: List[Candidate],
        support_count_map: Dict[Tuple[int, int, int], int],
        support_udt_map: Dict[Tuple[int, int, int], int],
        cell_xy: float,
    ) -> np.ndarray:
        out = np.zeros((len(cands),), dtype=np.float32)
        cs = float(max(cell_xy, 1e-6))
        for i, c in enumerate(cands):
            x, y = _cand_xy(c)
            key = (int(getattr(c, "label", -1)), int(np.round(x / cs)), int(np.round(y / cs)))
            fd = self._feature_dict(
                c,
                support_count=int(support_count_map.get(key, 1)),
                support_unique_dt=int(support_udt_map.get(key, 1)),
            )
            xv = np.asarray([float(fd.get(n, 0.0)) for n in self.feature_names], dtype=np.float64)
            if self.normalize:
                xv = (xv - self.mu) / np.where(np.abs(self.std) < 1e-8, 1.0, self.std)
            lg = float(np.dot(self.w, xv) + self.b)
            out[i] = np.float32(lg)
        return out


def _load_unary_model(path: Optional[str]) -> Optional[_UnaryLinearModel]:
    if path is None or str(path).strip() == "":
        return None
    p = str(path)
    if p in _UNARY_MODEL_CACHE:
        return _UNARY_MODEL_CACHE[p]
    try:
        with open(p, "r") as f:
            ckpt = json.load(f)
        model = _UnaryLinearModel(ckpt)
        _UNARY_MODEL_CACHE[p] = model
        print(f"[EBM] loaded unary checkpoint: {p}")
        return model
    except Exception as e:
        print(f"[EBM] warning: failed to load unary checkpoint '{p}': {repr(e)}")
        _UNARY_MODEL_CACHE[p] = None
        return None


def _build_unary_support_xy_maps(
    cands: List[Candidate],
    cell_xy: float,
) -> Tuple[Dict[Tuple[int, int, int], int], Dict[Tuple[int, int, int], int]]:
    cs = float(max(cell_xy, 1e-6))
    count_map: Dict[Tuple[int, int, int], int] = {}
    dtset_map: Dict[Tuple[int, int, int], Set[int]] = {}
    for c in cands:
        x, y = _cand_xy(c)
        key = (int(getattr(c, "label", -1)), int(np.round(x / cs)), int(np.round(y / cs)))
        count_map[key] = count_map.get(key, 0) + 1
        dtset_map.setdefault(key, set()).add(int(getattr(c, "from_dt", 0)))
    udt_count_map = {k: len(v) for k, v in dtset_map.items()}
    return count_map, udt_count_map


# -----------------------------------------------------------------------------
# Support / cell hashing (label-aware to avoid cross-label support pollution)
# -----------------------------------------------------------------------------

def _cell_key_lxyz(
    label: int,
    x: float,
    y: float,
    z: float,
    cell_xy: float,
    cell_z: float,
) -> Tuple[int, int, int, int]:
    gx = int(np.round(x / max(cell_xy, 1e-6)))
    gy = int(np.round(y / max(cell_xy, 1e-6)))
    gz = int(np.round(z / max(cell_z, 1e-6)))
    return int(label), gx, gy, gz


def build_cell_stats(
    cands: List[Candidate],
    cfg: EBMConfig,
) -> Tuple[Dict[Tuple[int, int, int, int], int], Dict[Tuple[int, int, int, int], Set[int]]]:
    """
    Returns:
      - cell_count: (label,gx,gy,gz) -> #cands
      - cell_dtset: (label,gx,gy,gz) -> set(dt)
    """
    cell_count: Dict[Tuple[int, int, int, int], int] = {}
    cell_dtset: Dict[Tuple[int, int, int, int], Set[int]] = {}

    cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
    cell_z = float(cfg.support_cell_z)

    for c in cands:
        x, y, z = _cand_xyz(c)
        lab = int(getattr(c, "label", -1))
        ck = _cell_key_lxyz(lab, x, y, z, cell_xy, cell_z)
        cell_count[ck] = cell_count.get(ck, 0) + 1
        cell_dtset.setdefault(ck, set()).add(int(getattr(c, "from_dt", 0)))

    return cell_count, cell_dtset


def cell_support_value(
    c: Candidate,
    cfg: EBMConfig,
    cell_count: Dict[Tuple[int, int, int, int], int],
    cell_dtset: Dict[Tuple[int, int, int, int], Set[int]],
) -> int:
    cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
    cell_z = float(cfg.support_cell_z)

    x, y, z = _cand_xyz(c)
    lab = int(getattr(c, "label", -1))
    ck = _cell_key_lxyz(lab, x, y, z, cell_xy, cell_z)

    if cfg.support_use_unique_dt:
        return int(len(cell_dtset.get(ck, set([int(getattr(c, "from_dt", 0))]))))
    return int(cell_count.get(ck, 1))


# -----------------------------------------------------------------------------
# Keep probability / unary energy
# -----------------------------------------------------------------------------

def keep_logit_for_candidate(c: Candidate, cfg: EBMConfig, support_val: int) -> float:
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

    if support_val > 1 and cfg.support_logit_gain != 0.0:
        logit += float(cfg.support_logit_gain) * float(max(0, support_val - 1))

    return float(logit)


def unary_energy_keep_on(keep_logit: float, cfg: EBMConfig) -> float:
    # E_keep_on = - w_keep * keep_logit   (negative when keep_logit>0)
    return float(-cfg.w_keep * float(keep_logit))


# -----------------------------------------------------------------------------
# Attribute probability / energy
# -----------------------------------------------------------------------------

def attr_probs_from_speed(speed: float, cfg_attr: AttrConfig) -> np.ndarray:
    """
    Return probs over [moving, standing, parked] (in that order).
    """
    v = float(speed if np.isfinite(speed) else 0.0)

    if cfg_attr.sigma <= 1e-6:
        if v >= cfg_attr.moving_thr:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if v <= cfg_attr.parked_thr:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    mu_moving = float(cfg_attr.moving_thr + cfg_attr.moving_mu_offset)
    mu_parked = 0.0
    mu_stand = float(0.5 * (cfg_attr.parked_thr + cfg_attr.moving_thr))

    mus = np.array([mu_moving, mu_stand, mu_parked], dtype=np.float32)
    vv = np.array([v, v, v], dtype=np.float32)
    sigma2 = float(cfg_attr.sigma * cfg_attr.sigma)

    logits = - (vv - mus) ** 2 / (2.0 * max(sigma2, 1e-6))
    probs = _softmax_logits(logits, axis=-1)
    return probs.astype(np.float32)


def attr_energy_from_probs(probs: np.ndarray, cfg: EBMConfig) -> float:
    # E_attr = w_attr * (-log max_p)
    pmax = float(np.max(probs)) if probs.size > 0 else 0.0
    pmax = float(np.clip(pmax, 1e-6, 1.0))
    return float(cfg.w_attr * (-np.log(pmax)))


def build_cell_speed_map(
    cands: List[Candidate],
    cfg: EBMConfig,
    keep_logits: np.ndarray,
) -> Dict[Tuple[int, int, int, int], float]:
    """
    speed(cell) = weighted avg of candidate speeds using keep_prob as weights.
    cell is label-aware to avoid cross-label mixing.
    """
    cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
    cell_z = float(cfg.support_cell_z)

    probs = _sigmoid(keep_logits.astype(np.float32))
    cell_wsum: Dict[Tuple[int, int, int, int], float] = {}
    cell_vsum: Dict[Tuple[int, int, int, int], float] = {}

    for i, c in enumerate(cands):
        x, y, z = _cand_xyz(c)
        lab = int(getattr(c, "label", -1))
        ck = _cell_key_lxyz(lab, x, y, z, cell_xy, cell_z)
        w = float(probs[i])
        v = float(_cand_speed(c))
        if not np.isfinite(v):
            v = 0.0
        cell_wsum[ck] = cell_wsum.get(ck, 0.0) + w
        cell_vsum[ck] = cell_vsum.get(ck, 0.0) + w * v

    out: Dict[Tuple[int, int, int, int], float] = {}
    for ck, wsum in cell_wsum.items():
        out[ck] = 0.0 if wsum <= 1e-9 else float(cell_vsum.get(ck, 0.0) / wsum)
    return out


def infer_attr_for_candidate(
    c: Candidate,
    cfg: EBMConfig,
    cell_speed_map: Optional[Dict[Tuple[int, int, int, int], float]] = None,
) -> Tuple[str, np.ndarray]:
    """
    Return (attr_name, probs[moving,standing,parked]).
    """
    ac = cfg.attr_cfg
    lab = int(getattr(c, "label", -1))

    if ac.static_label_ids is not None and lab in ac.static_label_ids:
        probs = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return ac.names[2], probs

    if ac.use_cell_speed and cell_speed_map is not None:
        cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
        cell_z = float(cfg.support_cell_z)
        x, y, z = _cand_xyz(c)
        ck = _cell_key_lxyz(lab, x, y, z, cell_xy, cell_z)
        v = float(cell_speed_map.get(ck, _cand_speed(c)))
    else:
        v = _cand_speed(c)

    probs = attr_probs_from_speed(v, ac)
    idx = int(np.argmax(probs))
    return ac.names[idx], probs


# -----------------------------------------------------------------------------
# Pair energy (NMS-like)
# -----------------------------------------------------------------------------

def pair_energy(ci: Candidate, cj: Candidate, cfg: EBMConfig) -> float:
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

    if mass < float(cfg.label_vote_min_mass) or not votes:
        return int(getattr(target, "label", -1))

    return int(max(votes.items(), key=lambda kv: kv[1])[0])


# -----------------------------------------------------------------------------
# Candidate filtering
# -----------------------------------------------------------------------------

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


def prefilter_candidates(frame: FrameData, cfg: EBMConfig) -> List[Candidate]:
    return prefilter_candidates_list(getattr(frame, "candidates", []) or [], cfg)


# -----------------------------------------------------------------------------
# Greedy solver (grid accelerated)
# -----------------------------------------------------------------------------

def _grid_cell_xy(x: float, y: float, cell: float) -> Tuple[int, int]:
    cs = float(max(cell, 1e-6))
    return int(np.floor(x / cs)), int(np.floor(y / cs))


def _neighbor_cells(ix: int, iy: int) -> List[Tuple[int, int]]:
    return [(ix + dx, iy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]


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
    """
    Greedy approximate minimization:
      accept i if deltaE = unary(i) + sum_{j in neighbor-selected} pair(i,j) < 0

    Uses a grid keyed by (label or -1, ix, iy) to only check neighbors.
    """
    selected: List[int] = list(already_selected) if already_selected is not None else []
    base_sel_len = len(selected)
    max_add = int(max_add) if max_add is not None else None

    # use NMS threshold as grid cell size
    cell = float(max(cfg.nms_thr_xy, 1e-6))
    min_d2 = float(min_dist_to_selected) ** 2

    # grid: (label_key, ix, iy) -> selected indices
    grid: Dict[Tuple[int, int, int], List[int]] = {}

    def key_for_idx(i: int) -> Tuple[int, int, int]:
        x, y = _cand_xy(cands[i])
        ix, iy = _grid_cell_xy(x, y, cell)
        lab = -1 if cfg.nms_cross_class else int(getattr(cands[i], "label", -1))
        return (lab, ix, iy)

    def insert_idx(i: int) -> None:
        k = key_for_idx(i)
        grid.setdefault(k, []).append(i)

    for j in selected:
        insert_idx(j)

    # per-class cap for "newly added" items in THIS call
    kept_count_by_class: Dict[int, int] = {}

    # sort by descending keep_logit
    order = sorted(pool_idx, key=lambda i: float(keep_logits[i]), reverse=True)

    for i in order:
        # stage add limit
        if max_add is not None:
            added = len(selected) - base_sel_len
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

        x, y = _cand_xy(ci)
        ix, iy = _grid_cell_xy(x, y, cell)
        lab_key = -1 if cfg.nms_cross_class else lab_i

        neighbor_sel: List[int] = []
        for nix, niy in _neighbor_cells(ix, iy):
            neighbor_sel.extend(grid.get((lab_key, nix, niy), []))

        # hard min_dist_to_selected (e.g., stage2 vs seeds)
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

    return selected


def solve_candidates(
    cands: List[Candidate],
    cfg: EBMConfig,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Solve one frame given candidate list directly.
    Returns indices w.r.t. the (prefiltered) candidate list used inside this function.
    """
    cands = prefilter_candidates_list(cands, cfg)
    N = len(cands)
    if N == 0:
        return [], {"num_cands": 0, "selected": 0}

    # dt-support stats
    cell_count, cell_dtset = build_cell_stats(cands, cfg)

    # keep logits/probs
    keep_logits = np.zeros((N,), dtype=np.float32)
    keep_probs = np.zeros((N,), dtype=np.float32)
    support_vals = np.zeros((N,), dtype=np.int64)

    for i, c in enumerate(cands):
        sv = cell_support_value(c, cfg, cell_count, cell_dtset)
        support_vals[i] = int(sv)

    unary_model: Optional[_UnaryLinearModel] = None
    if bool(cfg.unary_use_learned):
        unary_model = _load_unary_model(cfg.unary_ckpt_path)

    if unary_model is not None:
        cell_xy = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
        sup_count_map, sup_udt_map = _build_unary_support_xy_maps(cands, cell_xy=cell_xy)
        keep_logits = unary_model.predict_logits(
            cands,
            support_count_map=sup_count_map,
            support_udt_map=sup_udt_map,
            cell_xy=cell_xy,
        )
        keep_probs = _sigmoid(keep_logits.astype(np.float32))
    else:
        for i, c in enumerate(cands):
            lg = keep_logit_for_candidate(c, cfg, int(support_vals[i]))
            keep_logits[i] = float(lg)
            keep_probs[i] = float(_sigmoid(np.array([lg], dtype=np.float32))[0])

    # Optional attr-energy for selection; can be disabled by setting w_attr<=0.
    use_attr_energy = float(cfg.w_attr) > 0.0
    cell_speed_map = (
        build_cell_speed_map(cands, cfg, keep_logits)
        if (use_attr_energy and cfg.attr_cfg.use_cell_speed)
        else None
    )

    # attr energies (selection-time term)
    attr_energy_arr = np.zeros((N,), dtype=np.float32)
    if use_attr_energy:
        for i, c in enumerate(cands):
            _, probs = infer_attr_for_candidate(c, cfg, cell_speed_map=cell_speed_map)
            attr_energy_arr[i] = float(attr_energy_from_probs(probs, cfg))

    selected_idx: List[int] = []

    if cfg.two_stage:
        # -------- Stage-1 seed pool --------
        if cfg.seed_sources == "raw":
            pool_seed = [i for i, c in enumerate(cands) if _source_group(c) == "raw"]
        elif cfg.seed_sources == "all":
            pool_seed = list(range(N))
        else:
            raise ValueError(f"seed_sources must be 'raw' or 'all', got: {cfg.seed_sources}")

        M = int(getattr(cfg, "prefilter_topm_seed", 0) or 0)
        if M > 0 and len(pool_seed) > M:
            pool_seed = sorted(pool_seed, key=lambda i: float(keep_logits[i]), reverse=True)[:M]

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

        # -------- Stage-2 fill pool --------
        sel_set = set(selected_idx)
        pool_fill: List[int] = []
        for i in range(N):
            if i in sel_set:
                continue
            # Stage-2 is "fill from temporal warp support"; keep this pool warp-only.
            if _source_group(cands[i]) != "warp":
                continue
            if float(keep_probs[i]) < float(cfg.fill_keep_thr):
                continue
            if int(support_vals[i]) < int(cfg.min_dt_support):
                continue
            pool_fill.append(i)

        M = int(getattr(cfg, "prefilter_topm_fill", 0) or 0)
        if M > 0 and len(pool_fill) > M:
            pool_fill = sorted(pool_fill, key=lambda i: float(keep_logits[i]), reverse=True)[:M]

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

    # final topk guard (global)
    if cfg.topk is not None and len(selected_idx) > int(cfg.topk):
        selected_idx = sorted(selected_idx, key=lambda i: float(keep_logits[i]), reverse=True)[: int(cfg.topk)]

    dbg: Dict[str, Any] = {
        "num_cands": int(N),
        "selected": int(len(selected_idx)),
        "use_learned_unary": bool(unary_model is not None),
        "unary_ckpt_path": str(cfg.unary_ckpt_path) if cfg.unary_ckpt_path else None,
    }
    # Internal cache for ebm_infer_candidates to avoid recomputing keep/attr stats.
    dbg["_cache"] = {
        "keep_logits": keep_logits,
        "keep_probs": keep_probs,
        "cell_speed_map": cell_speed_map,
    }
    if cfg.include_energy:
        # optional energy summary on selected set (still cheap since selected is small)
        tot_unary = 0.0
        tot_pair = 0.0
        for ii in selected_idx:
            tot_unary += float(unary_energy_keep_on(float(keep_logits[ii]), cfg) + float(attr_energy_arr[ii]))
        for a in range(len(selected_idx)):
            for b in range(a + 1, len(selected_idx)):
                tot_pair += float(pair_energy(cands[selected_idx[a]], cands[selected_idx[b]], cfg))
        dbg.update(
            {
                "E_unary": float(tot_unary),
                "E_pair": float(tot_pair),
                "E_total": float(tot_unary + tot_pair),
                "mean_keep_prob": float(np.mean(keep_probs[selected_idx]) if selected_idx else 0.0),
            }
        )
    return selected_idx, dbg


def solve_frame(frame: FrameData, cfg: EBMConfig) -> Tuple[List[int], Dict[str, Any]]:
    return solve_candidates(getattr(frame, "candidates", []) or [], cfg)


# -----------------------------------------------------------------------------
# Public API: inference payloads
# -----------------------------------------------------------------------------

def ebm_infer_candidates(cands: List[Candidate], cfg: EBMConfig) -> Dict[str, Any]:
    """
    Produce payload for one frame from a candidate list.
    """
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

    # Reuse selection-time cache when available (saves one full O(N) recomputation pass).
    cache = dbg.get("_cache", None)
    if isinstance(cache, dict):
        keep_logits = cache.get("keep_logits", None)
        keep_probs = cache.get("keep_probs", None)
        cell_speed_map = cache.get("cell_speed_map", None)
    else:
        keep_logits = None
        keep_probs = None
        cell_speed_map = None

    if keep_logits is None or keep_probs is None:
        cell_count, cell_dtset = build_cell_stats(cands, cfg)
        support_vals = np.array([cell_support_value(c, cfg, cell_count, cell_dtset) for c in cands], dtype=np.int64)
        keep_logits = np.array(
            [keep_logit_for_candidate(c, cfg, int(support_vals[i])) for i, c in enumerate(cands)],
            dtype=np.float32,
        )
        keep_probs = _sigmoid(keep_logits)
        if cfg.attr_cfg.use_cell_speed:
            cell_speed_map = build_cell_speed_map(cands, cfg, keep_logits)

    boxes_out: List[List[float]] = []
    labels_out: List[int] = []
    scores_out: List[float] = []
    attrs_out: List[str] = []
    sources_out: List[str] = []
    track_ids_out: List[int] = []

    for i in sel_idx:
        c = cands[i]
        b = np.asarray(c.box, dtype=np.float32)
        if b.ndim != 1 or b.shape[0] < 9 or (not _finite_vec(b[:3])):
            continue

        lab = refine_label_by_vote(c, cands, keep_probs, cfg)
        attr_name, _ = infer_attr_for_candidate(c, cfg, cell_speed_map=cell_speed_map)

        boxes_out.append([float(x) for x in b.tolist()])
        labels_out.append(int(lab))
        scores_out.append(float(keep_probs[i]))  # keep-prob as score
        attrs_out.append(str(attr_name))

        if cfg.include_sources:
            src = str(getattr(c, "source", ""))
            dt = int(getattr(c, "from_dt", 0))
            sources_out.append(f"{src}" if dt == 0 else f"{src}|dt={dt}")

        if hasattr(c, "track_id"):
            try:
                track_ids_out.append(int(getattr(c, "track_id")))
            except Exception:
                track_ids_out.append(-1)

    payload: Dict[str, Any] = {
        "boxes_3d": boxes_out,
        "labels_3d": labels_out,
        "scores_3d": scores_out,
        "attrs": attrs_out,
    }
    if cfg.include_sources:
        payload["sources"] = sources_out
    if len(track_ids_out) == len(boxes_out) and len(track_ids_out) > 0:
        payload["track_ids"] = track_ids_out
    if cfg.include_energy:
        dbg.pop("_cache", None)
        payload["energy"] = dbg
    return payload


def ebm_infer_frame(frame: FrameData, cfg: EBMConfig) -> Dict[str, Any]:
    return ebm_infer_candidates(getattr(frame, "candidates", []) or [], cfg)


def ebm_infer_scene(frames: List[FrameData], cfg: EBMConfig, key_fn) -> Dict[str, Dict[str, Any]]:
    repl: Dict[str, Dict[str, Any]] = {}
    for fr in frames:
        payload = ebm_infer_frame(fr, cfg)
        key = _choose_key_from_candidates(fr.sample, key_fn)
        if key == "":
            key = f"{getattr(fr, 'scene_token', 'scene')}:{getattr(fr, 'timestamp', 't')}"
        repl[key] = payload
    return repl


def ebm_infer_frames_by_scene(
    frames_by_scene: Dict[str, List[FrameData]],
    cfg: EBMConfig,
    key_fn,
) -> Dict[str, Dict[str, Any]]:
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
    """
    Wrapper class for infer.py lazy import:
      from model.ebm import EBMRefiner, EBMConfig
    """

    def __init__(self, cfg: Optional[EBMConfig] = None):
        self.cfg = cfg if cfg is not None else EBMConfig()

    def _merge_from_infer_cfg(self, infer_cfg) -> EBMConfig:
        """
        Allow infer.py to pass its InferConfig; we map overlapping field names.
        This avoids depending on infer.py types.
        """
        if infer_cfg is None:
            return self.cfg

        kw: Dict[str, Any] = {}
        # common shared knobs (only override if infer_cfg actually has them)
        for name in [
            "use_sources",
            "raw_only_score_min",
            "keep_thr",
            "topk",
            "nms_thr_xy",
            "nms_cross_class",
            "two_stage",
            "seed_sources",
            "seed_keep_thr",
            "fill_keep_thr",
            "min_dt_support",
            "dt_cell_size",
            "min_dist_to_seed",
            "max_fill",
            "include_sources",
        ]:
            if hasattr(infer_cfg, name):
                kw[name] = getattr(infer_cfg, name)

        # if you later add CLI flags for these, they will auto-merge too
        for name in ["prefilter_topm_seed", "prefilter_topm_fill"]:
            if hasattr(infer_cfg, name):
                kw[name] = getattr(infer_cfg, name)

        return replace(self.cfg, **kw) if kw else self.cfg

    def infer_frame(self, frame: FrameData, candidates=None, infer_cfg=None, **kwargs) -> Dict[str, Any]:
        base = candidates if candidates is not None else (getattr(frame, "candidates", []) or [])
        cfg_run = self._merge_from_infer_cfg(infer_cfg)
        return ebm_infer_candidates(base, cfg_run)

    def refine_frame(self, frame: FrameData, candidates=None, infer_cfg=None, **kwargs) -> Dict[str, Any]:
        return self.infer_frame(frame=frame, candidates=candidates, infer_cfg=infer_cfg, **kwargs)

    def infer_frames_by_scene(self, frames_by_scene: Dict[str, List[FrameData]], key_fn, infer_cfg=None) -> Dict[str, Dict[str, Any]]:
        cfg_run = self._merge_from_infer_cfg(infer_cfg)
        out: Dict[str, Dict[str, Any]] = {}
        for _, frames in frames_by_scene.items():
            for fr in frames:
                payload = ebm_infer_frame(fr, cfg_run)
                key = _choose_key_from_candidates(fr.sample, key_fn)
                if key == "":
                    key = f"{getattr(fr, 'scene_token', 'scene')}:{getattr(fr, 'timestamp', 't')}"
                out[key] = payload
        return out

    def refine_frames_by_scene(self, frames_by_scene: Dict[str, List[FrameData]], key_fn, infer_cfg=None) -> Dict[str, Dict[str, Any]]:
        return self.infer_frames_by_scene(frames_by_scene, key_fn=key_fn, infer_cfg=infer_cfg)


__all__ = [
    "AttrConfig",
    "EBMConfig",
    "EBMRefiner",
    "prefilter_candidates_list",
    "solve_candidates",
    "ebm_infer_candidates",
    "ebm_infer_frame",
    "ebm_infer_frames_by_scene",
]
