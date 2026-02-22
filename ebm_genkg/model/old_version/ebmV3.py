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


def _bev_overlap_ratio_min(ci: Candidate, cj: Candidate) -> float:
    """
    Approximate BEV overlap ratio: inter_area / min(area_i, area_j).
    Uses axis-aligned boxes from (x,y,dx,dy) for speed.
    """
    bi = np.asarray(ci.box, dtype=np.float32)
    bj = np.asarray(cj.box, dtype=np.float32)
    if bi.shape[0] < 5 or bj.shape[0] < 5:
        return 0.0

    xi, yi = float(bi[0]), float(bi[1])
    xj, yj = float(bj[0]), float(bj[1])
    dxi, dyi = abs(float(bi[3])), abs(float(bi[4]))
    dxj, dyj = abs(float(bj[3])), abs(float(bj[4]))
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

    ai = dxi * dyi
    aj = dxj * dyj
    return float(inter / max(min(ai, aj), 1e-6))


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
    # model mode
    energy_mode: str = "four_term"  # "four_term" | "legacy"
    energy_select_margin: float = 0.0
    energy_prob_gate: float = 0.0
    energy_local_refine: bool = True
    energy_local_refine_rounds: int = 2

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
    hard_nms: bool = False
    soft_pair_scale: float = 5.0
    # overlap soft penalty in BEV
    enable_overlap_soft: bool = True
    overlap_min_ratio: float = 0.10
    overlap_soft_scale: float = 1.00
    # temporal pair bonus (negative energy)
    enable_temporal_pair: bool = True
    temporal_pair_radius: float = 1.5
    temporal_pair_bonus: float = 0.30
    temporal_pair_warp_only: bool = True

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
    # Soft relation-energy for dt support shortfall in fill stage.
    # When enabled, candidates below min_dt_support are not hard-filtered;
    # they get an added positive energy penalty.
    soft_dt_support: bool = True
    dt_shortfall_penalty: float = 0.35
    # context-density energy
    enable_context_density: bool = True
    context_cell_xy: float = 1.0
    context_min_density: int = 2
    context_shortfall_penalty: float = 0.15
    context_warp_only: bool = True

    # IMPORTANT speed-up: keep only top-M by keep_logit in each stage pool
    prefilter_topm_seed: int = 1500
    prefilter_topm_fill: int = 3000
    # Adaptive pool budget for unified-context soft-budget solve:
    #   M_eff = min(M_cap, clip(base + sqrt_scale * sqrt(N), min, max))
    adaptive_prefilter: bool = True
    adaptive_prefilter_base: int = 128
    adaptive_prefilter_sqrt_scale: float = 48.0
    adaptive_prefilter_min: int = 256
    adaptive_prefilter_max: int = 4096

    # dual-head staged solve (StageA keep raw core -> StageB warp add -> StageC conflict resolve)
    dual_head_solver: bool = True
    # If enabled, dual-head uses one unified candidate pool (raw+warp) instead of
    # hard StageA raw-only then StageB warp-only gating.
    dual_head_unified_context: bool = True
    keep_head_raw_bias: float = 0.20
    keep_head_nonraw_bias: float = -0.80
    add_head_warp_bias: float = 0.25
    add_head_nonwarp_bias: float = -6.0
    add_head_support_gain: float = 0.20
    add_head_potential_gain: float = 0.35
    add_head_class_gain: float = 0.25
    stage_a_keep_thr: float = 0.50
    stage_b_add_thr: float = 0.35
    stage_b_min_potential_dist: float = 1.2
    stage_b_min_potential: float = 0.25
    stage_b_min_class_conf: float = 0.0
    stage_b_enforce_class_conf: bool = False
    stage_c_attr_scale: float = 0.20
    stage_c_rel_scale: float = 0.20

    # learned unary model (optional)
    unary_use_learned: bool = True
    unary_ckpt_path: Optional[str] = None
    enable_learned_pair: bool = True
    use_learned_class: bool = True
    learned_class_min_prob: float = 0.55
    use_learned_attr: bool = True
    learned_attr_min_prob: float = 0.50

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
_CONTEXT_FEATURE_NAMES: List[str] = [
    "context_self_prob",
    "context_nbr_prob_mean",
    "context_nbr_prob_max",
    "context_nbr_same_label_prob_mean",
    "context_nbr_warp_prob_mean",
    "context_nbr_raw_prob_mean",
    "context_nbr_abs_dt_mean",
    "context_nbr_score_mean",
]


class _UnaryLinearModel:
    """
    Lightweight runtime for train_unary.py checkpoints.
    """

    def __init__(self, ckpt: Dict[str, Any]):
        self.model_type = str(ckpt.get("model_type", ""))
        self.feature_names = [str(x) for x in ckpt.get("feature_names", [])]
        self.normalize = bool(ckpt.get("normalize", False))
        self.is_structured = self.model_type == "structured_energy_mlp"

        self.mu = np.asarray(ckpt.get("mu", []), dtype=np.float64).reshape(-1)
        self.std = np.asarray(ckpt.get("std", []), dtype=np.float64).reshape(-1)
        self.w = np.asarray(ckpt.get("weights", []), dtype=np.float64).reshape(-1)
        self.b = float(ckpt.get("bias", 0.0))
        rel = ckpt.get("relation", {}) if isinstance(ckpt.get("relation", {}), dict) else {}
        self.rel_enabled = bool(rel.get("enabled", False))
        self.rel_feature_names = [str(x) for x in rel.get("feature_names", [])]
        self.rel_w = np.asarray(rel.get("weights", []), dtype=np.float64).reshape(-1)
        self.rel_b = float(rel.get("bias", 0.0))
        cls_head = ckpt.get("class_head", {}) if isinstance(ckpt.get("class_head", {}), dict) else {}
        self.class_enabled = bool(cls_head.get("enabled", False))
        self.class_labels = [int(x) for x in cls_head.get("class_labels", [])] if isinstance(cls_head.get("class_labels", []), list) else []
        self.class_w = np.asarray(cls_head.get("weights", []), dtype=np.float64)
        self.class_b = np.asarray(cls_head.get("bias", []), dtype=np.float64).reshape(-1)
        attr_head = ckpt.get("attr_head", {}) if isinstance(ckpt.get("attr_head", {}), dict) else {}
        self.attr_enabled = bool(attr_head.get("enabled", False))
        self.attr_ids = [int(x) for x in attr_head.get("attr_ids", [])] if isinstance(attr_head.get("attr_ids", []), list) else []
        self.attr_names = [str(x) for x in attr_head.get("attr_names", [])] if isinstance(attr_head.get("attr_names", []), list) else []
        self.attr_w = np.asarray(attr_head.get("weights", []), dtype=np.float64)
        self.attr_b = np.asarray(attr_head.get("bias", []), dtype=np.float64).reshape(-1)
        self._feat_idx = {n: i for i, n in enumerate(self.feature_names)}

        # Structured energy unary MLP
        mlp = ckpt.get("unary_mlp", {}) if isinstance(ckpt.get("unary_mlp", {}), dict) else {}
        self.mlp_w1 = np.asarray(mlp.get("w1", []), dtype=np.float64)
        self.mlp_b1 = np.asarray(mlp.get("b1", []), dtype=np.float64).reshape(-1)
        self.mlp_w2 = np.asarray(mlp.get("w2", []), dtype=np.float64).reshape(-1)
        self.mlp_b2 = float(mlp.get("b2", 0.0))

        # Structured learned pair energy
        pair = ckpt.get("pair", {}) if isinstance(ckpt.get("pair", {}), dict) else {}
        self.pair_enabled = bool(pair.get("enabled", False))
        self.pair_feature_names = [str(x) for x in pair.get("feature_names", [])]
        self.pair_w = np.asarray(pair.get("weights", []), dtype=np.float64).reshape(-1)
        self.pair_b = float(pair.get("bias", 0.0))
        self.pair_scale = float(pair.get("scale", 1.0))
        self.pair_radius = float(pair.get("radius", 2.5))
        context = ckpt.get("context_block", {}) if isinstance(ckpt.get("context_block", {}), dict) else {}
        self.context_enabled = bool(context.get("enabled", False))
        self.context_radius_xy = float(context.get("radius_xy", 1.5))
        self.context_max_neighbors = int(context.get("max_neighbors", 24))
        self.context_feature_names = [str(x) for x in context.get("feature_names", [])]
        self.context_rounds = context.get("rounds", []) if isinstance(context.get("rounds", []), list) else []

        if self.model_type not in ("logistic_unary", "logistic_energy", "structured_energy_mlp"):
            raise ValueError(f"Unsupported unary model_type: {self.model_type}")
        if self.is_structured:
            if self.mlp_w1.ndim != 2:
                raise ValueError("structured unary_mlp.w1 must be 2D")
            if self.mlp_w1.shape[0] != len(self.feature_names):
                raise ValueError("structured unary_mlp.w1 input dim mismatch feature_names")
            if self.mlp_b1.shape[0] != self.mlp_w1.shape[1]:
                raise ValueError("structured unary_mlp.b1 hidden dim mismatch")
            if self.mlp_w2.shape[0] != self.mlp_w1.shape[1]:
                raise ValueError("structured unary_mlp.w2 hidden dim mismatch")
        else:
            if len(self.feature_names) != self.w.shape[0]:
                raise ValueError("feature_names and weights length mismatch in unary checkpoint")
        if self.normalize:
            dim = len(self.feature_names)
            if self.mu.shape[0] != dim or self.std.shape[0] != dim:
                raise ValueError("mu/std length mismatch in unary checkpoint")
        if self.rel_enabled and len(self.rel_feature_names) != self.rel_w.shape[0]:
            raise ValueError("relation feature_names and weights length mismatch in unary checkpoint")
        if self.pair_enabled and len(self.pair_feature_names) != self.pair_w.shape[0]:
            raise ValueError("pair feature_names and weights length mismatch in checkpoint")
        if self.context_enabled:
            if len(self.context_rounds) == 0:
                self.context_enabled = False
            if len(self.context_feature_names) == 0:
                all_match = True
                for rd in self.context_rounds:
                    if not isinstance(rd, dict):
                        all_match = False
                        break
                    wctx = np.asarray(rd.get("weights", []), dtype=np.float64).reshape(-1)
                    if wctx.shape[0] != len(_CONTEXT_FEATURE_NAMES):
                        all_match = False
                        break
                if all_match:
                    self.context_feature_names = list(_CONTEXT_FEATURE_NAMES)
            elif self.context_feature_names != _CONTEXT_FEATURE_NAMES:
                self.context_enabled = False
            for rd in self.context_rounds:
                if not isinstance(rd, dict):
                    self.context_enabled = False
                    break
                wctx = np.asarray(rd.get("weights", []), dtype=np.float64).reshape(-1)
                if len(self.context_feature_names) == 0 or wctx.shape[0] != len(self.context_feature_names):
                    self.context_enabled = False
                    break
        if self.class_enabled:
            if self.class_w.ndim != 2:
                raise ValueError("class_head.weights must be 2D")
            if self.class_w.shape[0] != len(self.feature_names):
                raise ValueError("class_head input dim mismatch feature_names")
            if self.class_b.shape[0] != self.class_w.shape[1]:
                raise ValueError("class_head bias dim mismatch")
            if len(self.class_labels) == 0:
                self.class_labels = [int(i) for i in range(self.class_w.shape[1])]
            if len(self.class_labels) != self.class_w.shape[1]:
                raise ValueError("class_head class_labels dim mismatch")
        if self.attr_enabled:
            if self.attr_w.ndim != 2:
                raise ValueError("attr_head.weights must be 2D")
            if self.attr_w.shape[0] != len(self.feature_names):
                raise ValueError("attr_head input dim mismatch feature_names")
            if self.attr_b.shape[0] != self.attr_w.shape[1]:
                raise ValueError("attr_head bias dim mismatch")
            if len(self.attr_names) != self.attr_w.shape[1]:
                if len(self.attr_names) == 0 and len(self.attr_ids) == self.attr_w.shape[1]:
                    self.attr_names = [f"attr_{int(i)}" for i in self.attr_ids]
                else:
                    raise ValueError("attr_head attr_names dim mismatch")

    def _feature_dict(
        self,
        c: Candidate,
        support_count: int,
        support_unique_dt: int,
        temporal_stability: float,
        local_density: int,
        support_score_mean: float = 0.0,
        support_abs_dt_mean: float = 0.0,
        support_raw_ratio: float = 0.0,
        support_warp_ratio: float = 0.0,
        temporal_local_interaction: float = 0.0,
        support_density_interaction: float = 0.0,
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
            "temporal_stability": float(temporal_stability),
            "local_density": float(local_density),
            "support_score_mean": float(support_score_mean),
            "support_abs_dt_mean": float(support_abs_dt_mean),
            "support_raw_ratio": float(support_raw_ratio),
            "support_warp_ratio": float(support_warp_ratio),
            "temporal_local_interaction": float(temporal_local_interaction),
            "support_density_interaction": float(support_density_interaction),
        }

    def predict_all(
        self,
        cands: List[Candidate],
        support_count_map: Dict[Tuple[int, int, int], int],
        support_udt_map: Dict[Tuple[int, int, int], int],
        local_density_map: Dict[Tuple[int, int], int],
        cell_xy: float,
    ) -> Dict[str, np.ndarray]:
        keep_out = np.zeros((len(cands),), dtype=np.float32)
        class_out: Optional[np.ndarray] = None
        attr_out: Optional[np.ndarray] = None
        if self.class_enabled:
            class_out = np.zeros((len(cands), int(self.class_b.shape[0])), dtype=np.float32)
        if self.attr_enabled:
            attr_out = np.zeros((len(cands), int(self.attr_b.shape[0])), dtype=np.float32)
        cs = float(max(cell_xy, 1e-6))
        score_sum_map: Dict[Tuple[int, int, int], float] = {}
        abs_dt_sum_map: Dict[Tuple[int, int, int], float] = {}
        raw_count_map: Dict[Tuple[int, int, int], int] = {}
        warp_count_map: Dict[Tuple[int, int, int], int] = {}
        for c in cands:
            x, y = _cand_xy(c)
            gx = int(np.round(x / cs))
            gy = int(np.round(y / cs))
            key = (int(getattr(c, "label", -1)), gx, gy)
            score_v = float(getattr(c, "score", 0.0))
            score_v = score_v if np.isfinite(score_v) else 0.0
            score_sum_map[key] = float(score_sum_map.get(key, 0.0) + score_v)
            abs_dt_sum_map[key] = float(abs_dt_sum_map.get(key, 0.0) + float(abs(int(getattr(c, "from_dt", 0)))))
            grp = _source_group(c)
            if grp == "raw":
                raw_count_map[key] = int(raw_count_map.get(key, 0) + 1)
            elif grp == "warp":
                warp_count_map[key] = int(warp_count_map.get(key, 0) + 1)
        for i, c in enumerate(cands):
            x, y = _cand_xy(c)
            gx = int(np.round(x / cs))
            gy = int(np.round(y / cs))
            key = (int(getattr(c, "label", -1)), gx, gy)
            support_count = int(support_count_map.get(key, 1))
            support_udt = int(support_udt_map.get(key, 1))
            temporal_stability = float(support_udt / max(1, support_count))
            support_score_mean = float(score_sum_map.get(key, float(getattr(c, "score", 0.0))) / max(1, support_count))
            support_abs_dt_mean = float(abs_dt_sum_map.get(key, float(abs(int(getattr(c, "from_dt", 0))))) / max(1, support_count))
            support_raw_ratio = float(raw_count_map.get(key, 0) / max(1, support_count))
            support_warp_ratio = float(warp_count_map.get(key, 0) / max(1, support_count))
            ld = int(local_density_map.get((gx, gy), 1))
            temporal_local_interaction = float(temporal_stability * np.log1p(max(0.0, float(ld) - 1.0)))
            support_density_interaction = float(float(support_count) / max(1.0, float(ld)))
            fd = self._feature_dict(
                c,
                support_count=support_count,
                support_unique_dt=support_udt,
                temporal_stability=temporal_stability,
                local_density=ld,
                support_score_mean=support_score_mean,
                support_abs_dt_mean=support_abs_dt_mean,
                support_raw_ratio=support_raw_ratio,
                support_warp_ratio=support_warp_ratio,
                temporal_local_interaction=temporal_local_interaction,
                support_density_interaction=support_density_interaction,
            )
            xv = np.asarray([float(fd.get(n, 0.0)) for n in self.feature_names], dtype=np.float64)
            if self.normalize:
                xv = (xv - self.mu) / np.where(np.abs(self.std) < 1e-8, 1.0, self.std)
            if self.is_structured:
                h = np.maximum(0.0, xv @ self.mlp_w1 + self.mlp_b1)
                lg = float(h @ self.mlp_w2 + self.mlp_b2)
            else:
                lg = float(np.dot(self.w, xv) + self.b)
            if self.rel_enabled and len(self.rel_feature_names) > 0:
                rel_vals: List[float] = []
                for n in self.rel_feature_names:
                    v = float(fd.get(n, 0.0))
                    if self.normalize and (n in self._feat_idx):
                        j = int(self._feat_idx[n])
                        den = self.std[j] if abs(self.std[j]) >= 1e-8 else 1.0
                        v = float((v - self.mu[j]) / den)
                    rel_vals.append(v)
                rel_x = np.asarray(rel_vals, dtype=np.float64)
                lg += float(np.dot(self.rel_w, rel_x) + self.rel_b)
            keep_out[i] = np.float32(lg)
            if class_out is not None:
                class_out[i, :] = (xv @ self.class_w + self.class_b).astype(np.float32, copy=False)
            if attr_out is not None:
                attr_out[i, :] = (xv @ self.attr_w + self.attr_b).astype(np.float32, copy=False)
        if self.context_enabled and len(cands) > 0:
            keep_out = self._apply_context_logits(cands, keep_out.astype(np.float64, copy=False)).astype(np.float32, copy=False)
        return {"keep_logits": keep_out, "class_logits": class_out, "attr_logits": attr_out}

    def _build_context_neighbors(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        radius_xy: float,
        max_neighbors: int,
    ) -> List[np.ndarray]:
        n = int(xs.shape[0])
        rad = float(max(radius_xy, 1e-6))
        rad2 = rad * rad
        cap = int(max(1, max_neighbors))
        cell = rad
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for i in range(n):
            gx = int(np.floor(float(xs[i]) / cell))
            gy = int(np.floor(float(ys[i]) / cell))
            buckets.setdefault((gx, gy), []).append(i)
        neigh: List[np.ndarray] = []
        for i in range(n):
            gx = int(np.floor(float(xs[i]) / cell))
            gy = int(np.floor(float(ys[i]) / cell))
            cand_idx: List[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand_idx.extend(buckets.get((gx + dx, gy + dy), []))
            if len(cand_idx) == 0:
                neigh.append(np.zeros((0,), dtype=np.int64))
                continue
            arr = np.asarray(cand_idx, dtype=np.int64)
            arr = arr[arr != i]
            if arr.size == 0:
                neigh.append(np.zeros((0,), dtype=np.int64))
                continue
            dxv = xs[arr] - float(xs[i])
            dyv = ys[arr] - float(ys[i])
            d2 = dxv * dxv + dyv * dyv
            keep = d2 <= rad2
            if not np.any(keep):
                neigh.append(np.zeros((0,), dtype=np.int64))
                continue
            arr = arr[keep]
            d2 = d2[keep]
            if arr.size > cap:
                ord_idx = np.argpartition(d2, cap - 1)[:cap]
                arr = arr[ord_idx]
            neigh.append(arr.astype(np.int64, copy=False))
        return neigh

    def _context_feature_matrix(
        self,
        probs: np.ndarray,
        neighbors: List[np.ndarray],
        labels: np.ndarray,
        from_dt: np.ndarray,
        is_raw: np.ndarray,
        is_warp: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        n = int(probs.shape[0])
        F = np.zeros((n, len(_CONTEXT_FEATURE_NAMES)), dtype=np.float64)
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
            F[i, 3] = float(np.mean(p_n[same])) if np.any(same) else 0.0
            F[i, 4] = float(np.mean(p_n * is_warp[nei]))
            F[i, 5] = float(np.mean(p_n * is_raw[nei]))
            F[i, 6] = float(np.mean(np.abs(from_dt[nei] - from_dt[i])))
            F[i, 7] = float(np.mean(scores[nei]))
        return F

    def _apply_context_logits(self, cands: List[Candidate], logits: np.ndarray) -> np.ndarray:
        if not self.context_enabled or len(self.context_rounds) == 0 or len(cands) == 0:
            return logits
        xs = np.zeros((len(cands),), dtype=np.float64)
        ys = np.zeros((len(cands),), dtype=np.float64)
        labels = np.zeros((len(cands),), dtype=np.int64)
        from_dt = np.zeros((len(cands),), dtype=np.int64)
        is_raw = np.zeros((len(cands),), dtype=np.float64)
        is_warp = np.zeros((len(cands),), dtype=np.float64)
        scores = np.zeros((len(cands),), dtype=np.float64)
        for i, c in enumerate(cands):
            x, y = _cand_xy(c)
            xs[i] = float(x)
            ys[i] = float(y)
            labels[i] = int(getattr(c, "label", -1))
            from_dt[i] = int(getattr(c, "from_dt", 0))
            grp = _source_group(c)
            is_raw[i] = 1.0 if grp == "raw" else 0.0
            is_warp[i] = 1.0 if grp == "warp" else 0.0
            sv = float(getattr(c, "score", 0.0))
            scores[i] = sv if np.isfinite(sv) else 0.0
        neighbors = self._build_context_neighbors(
            xs,
            ys,
            radius_xy=float(self.context_radius_xy),
            max_neighbors=int(self.context_max_neighbors),
        )
        out = logits.astype(np.float64, copy=True)
        for rd in self.context_rounds:
            wctx = np.asarray(rd.get("weights", []), dtype=np.float64).reshape(-1)
            bctx = float(rd.get("bias", 0.0))
            F = self._context_feature_matrix(
                probs=_sigmoid(out),
                neighbors=neighbors,
                labels=labels,
                from_dt=from_dt,
                is_raw=is_raw,
                is_warp=is_warp,
                scores=scores,
            )
            if wctx.shape[0] != F.shape[1]:
                return logits
            out = out + (F @ wctx + bctx)
        return out

    def predict_logits(
        self,
        cands: List[Candidate],
        support_count_map: Dict[Tuple[int, int, int], int],
        support_udt_map: Dict[Tuple[int, int, int], int],
        local_density_map: Dict[Tuple[int, int], int],
        cell_xy: float,
    ) -> np.ndarray:
        out = self.predict_all(
            cands,
            support_count_map=support_count_map,
            support_udt_map=support_udt_map,
            local_density_map=local_density_map,
            cell_xy=cell_xy,
        )
        return np.asarray(out.get("keep_logits", np.zeros((len(cands),), dtype=np.float32)), dtype=np.float32)

    def _pair_feature_dict(self, ci: Candidate, cj: Candidate) -> Dict[str, float]:
        li = int(getattr(ci, "label", -1))
        lj = int(getattr(cj, "label", -1))
        d2 = _dist2_xy(_cand_xy(ci), _cand_xy(cj))
        d = float(np.sqrt(max(d2, 0.0)))
        rad = float(max(self.pair_radius, 1e-6))
        close = float(np.exp(-d / rad))
        ov = float(_bev_overlap_ratio_min(ci, cj))
        dti = int(getattr(ci, "from_dt", 0))
        dtj = int(getattr(cj, "from_dt", 0))
        abi = abs(dti - dtj)
        wi = 1.0 if _source_group(ci) == "warp" else 0.0
        wj = 1.0 if _source_group(cj) == "warp" else 0.0
        return {
            "same_label": 1.0 if li == lj else 0.0,
            "close": close,
            "overlap": ov,
            "abs_dt_diff": float(min(abi, 8)) / 8.0,
            "both_warp": wi * wj,
            "either_warp": 1.0 if (wi + wj) > 0.0 else 0.0,
            "score_min": float(min(float(getattr(ci, "score", 0.0)), float(getattr(cj, "score", 0.0)))),
            "speed_diff": float(min(abs(_cand_speed(ci) - _cand_speed(cj)), 10.0)) / 10.0,
            "same_dt": 1.0 if dti == dtj else 0.0,
        }

    def learned_pair_energy(self, ci: Candidate, cj: Candidate) -> float:
        if (not self.pair_enabled) or len(self.pair_feature_names) == 0:
            return 0.0
        fd = self._pair_feature_dict(ci, cj)
        x = np.asarray([float(fd.get(n, 0.0)) for n in self.pair_feature_names], dtype=np.float64)
        logit = float(np.dot(self.pair_w, x) + self.pair_b)
        p = float(_sigmoid(np.asarray([logit], dtype=np.float64))[0])
        # lower energy for supportive pairs (p -> 1), higher for conflicting pairs (p -> 0)
        return float(self.pair_scale * (0.5 - p))


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
) -> Tuple[
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int], int],
]:
    cs = float(max(cell_xy, 1e-6))
    count_map: Dict[Tuple[int, int, int], int] = {}
    dtset_map: Dict[Tuple[int, int, int], Set[int]] = {}
    local_density_map: Dict[Tuple[int, int], int] = {}
    for c in cands:
        x, y = _cand_xy(c)
        gx = int(np.round(x / cs))
        gy = int(np.round(y / cs))
        key = (int(getattr(c, "label", -1)), gx, gy)
        count_map[key] = count_map.get(key, 0) + 1
        dtset_map.setdefault(key, set()).add(int(getattr(c, "from_dt", 0)))
        k2 = (gx, gy)
        local_density_map[k2] = local_density_map.get(k2, 0) + 1
    udt_count_map = {k: len(v) for k, v in dtset_map.items()}
    return count_map, udt_count_map, local_density_map


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

def pair_energy(
    ci: Candidate,
    cj: Candidate,
    cfg: EBMConfig,
    unary_model: Optional[_UnaryLinearModel] = None,
) -> float:
    li = int(getattr(ci, "label", -1))
    lj = int(getattr(cj, "label", -1))
    same_cls = li == lj
    if (not cfg.nms_cross_class) and (not same_cls):
        return 0.0

    e = 0.0
    d2 = _dist2_xy(_cand_xy(ci), _cand_xy(cj))

    # (A) NMS-like distance penalty
    thr = float(cfg.nms_thr_xy)
    if thr > 0:
        thr2 = thr * thr
        if d2 <= thr2:
            if cfg.hard_nms:
                return float(cfg.inf_energy)
            d = float(np.sqrt(max(d2, 0.0)))
            pen = 1.0 - min(d / max(thr, 1e-6), 1.0)
            e += float(cfg.w_pair * cfg.soft_pair_scale * pen)

    # (B) Soft overlap penalty
    if bool(getattr(cfg, "enable_overlap_soft", True)) and same_cls:
        ov = _bev_overlap_ratio_min(ci, cj)
        if ov >= float(getattr(cfg, "overlap_min_ratio", 0.10)):
            e += float(getattr(cfg, "overlap_soft_scale", 1.0)) * float(ov)

    # (C) Temporal pair consistency bonus (negative energy)
    if bool(getattr(cfg, "enable_temporal_pair", True)) and same_cls:
        dti = int(getattr(ci, "from_dt", 0))
        dtj = int(getattr(cj, "from_dt", 0))
        if dti != dtj:
            if (not bool(getattr(cfg, "temporal_pair_warp_only", True))) or (dti != 0 or dtj != 0):
                rad = float(getattr(cfg, "temporal_pair_radius", 1.5))
                if rad > 0:
                    rad2 = rad * rad
                    if d2 <= rad2:
                        w = 1.0 - min(float(np.sqrt(max(d2, 0.0))) / max(rad, 1e-6), 1.0)
                        e -= float(getattr(cfg, "temporal_pair_bonus", 0.30)) * float(w)

    if bool(getattr(cfg, "enable_learned_pair", True)) and unary_model is not None:
        e += float(unary_model.learned_pair_energy(ci, cj))

    return float(e)


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
    rel_energy_arr: np.ndarray,
    cfg: EBMConfig,
    unary_model: Optional[_UnaryLinearModel] = None,
    *,
    prob_thr: float,
    already_selected: Optional[List[int]] = None,
    max_add: Optional[int] = None,
    min_dist_to_selected: float = 0.0,
    select_margin: float = 0.0,
    preserve_pool_order: bool = False,
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
    if cfg.max_per_class is not None:
        for j in selected:
            lab_j = int(getattr(cands[j], "label", -1))
            kept_count_by_class[lab_j] = kept_count_by_class.get(lab_j, 0) + 1

    # default: sort by descending keep_logit; optional preserve caller order.
    order = list(pool_idx) if preserve_pool_order else sorted(pool_idx, key=lambda i: float(keep_logits[i]), reverse=True)

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

        unary = (
            unary_energy_keep_on(float(keep_logits[i]), cfg)
            + float(attr_energy_arr[i])
            + float(rel_energy_arr[i])
        )

        pair_sum = 0.0
        reject = False
        for j in neighbor_sel:
            eij = pair_energy(ci, cands[j], cfg, unary_model=unary_model)
            if eij >= cfg.inf_energy * 0.5:
                reject = True
                break
            pair_sum += float(eij)
        if reject:
            continue

        if float(unary + pair_sum) < float(select_margin):
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
    class_logits_arr: Optional[np.ndarray] = None
    attr_logits_arr: Optional[np.ndarray] = None
    class_labels_arr: Optional[np.ndarray] = None
    attr_names_list: Optional[List[str]] = None
    support_vals = np.zeros((N,), dtype=np.int64)

    for i, c in enumerate(cands):
        sv = cell_support_value(c, cfg, cell_count, cell_dtset)
        support_vals[i] = int(sv)

    # Shared XY support/density maps for learned terms.
    cell_xy_feat = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
    sup_count_map, sup_udt_map, local_density_map = _build_unary_support_xy_maps(cands, cell_xy=cell_xy_feat)

    unary_model: Optional[_UnaryLinearModel] = None
    if bool(cfg.unary_use_learned):
        unary_model = _load_unary_model(cfg.unary_ckpt_path)

    if unary_model is not None:
        pred_all = unary_model.predict_all(
            cands,
            support_count_map=sup_count_map,
            support_udt_map=sup_udt_map,
            local_density_map=local_density_map,
            cell_xy=cell_xy_feat,
        )
        keep_logits = np.asarray(pred_all.get("keep_logits", keep_logits), dtype=np.float32)
        class_logits_arr = pred_all.get("class_logits", None)
        attr_logits_arr = pred_all.get("attr_logits", None)
        if unary_model.class_enabled and class_logits_arr is not None:
            class_labels_arr = np.asarray(unary_model.class_labels, dtype=np.int64).reshape(-1)
        if unary_model.attr_enabled and attr_logits_arr is not None:
            attr_names_list = [str(x) for x in unary_model.attr_names]
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

    # Relation-energy terms:
    #   (1) soften dt-support filtering in stage-2
    #   (2) context-density shortfall penalty
    rel_energy_arr = np.zeros((N,), dtype=np.float32)
    if bool(getattr(cfg, "soft_dt_support", True)) and float(getattr(cfg, "dt_shortfall_penalty", 0.0)) > 0.0:
        min_sup = int(max(0, cfg.min_dt_support))
        pen = float(cfg.dt_shortfall_penalty)
        for i in range(N):
            shortfall = max(0, min_sup - int(support_vals[i]))
            if shortfall > 0:
                rel_energy_arr[i] = np.float32(pen * float(shortfall))

    if bool(getattr(cfg, "enable_context_density", True)) and float(getattr(cfg, "context_shortfall_penalty", 0.0)) > 0.0:
        cxy = float(max(getattr(cfg, "context_cell_xy", 1.0), 1e-6))
        _, _, ctx_density_map = _build_unary_support_xy_maps(cands, cell_xy=cxy)
        dmin = int(max(0, getattr(cfg, "context_min_density", 2)))
        dpen = float(getattr(cfg, "context_shortfall_penalty", 0.15))
        warp_only = bool(getattr(cfg, "context_warp_only", True))
        for i, c in enumerate(cands):
            if warp_only and (_source_group(c) != "warp"):
                continue
            x, y = _cand_xy(c)
            gx = int(np.round(x / cxy))
            gy = int(np.round(y / cxy))
            dens = int(ctx_density_map.get((gx, gy), 1))
            shortfall = max(0, dmin - dens)
            if shortfall > 0:
                rel_energy_arr[i] += np.float32(dpen * float(shortfall))

    # Class-confidence potential for add-stage expansion.
    # Defaults to zeros so behavior is unchanged when no learned class head is present.
    class_conf_arr = np.zeros((N,), dtype=np.float32)
    class_gate_arr = np.zeros((N,), dtype=np.float32)
    if (
        isinstance(class_logits_arr, np.ndarray)
        and class_logits_arr.ndim == 2
        and class_logits_arr.shape[0] == N
        and class_logits_arr.shape[1] > 0
    ):
        cls_prob = _softmax_logits(class_logits_arr, axis=-1).astype(np.float32, copy=False)
        class_conf_arr = np.max(cls_prob, axis=1).astype(np.float32, copy=False)
        thr_cls = float(np.clip(getattr(cfg, "stage_b_min_class_conf", 0.0), 0.0, 0.999))
        class_gate_arr = np.clip((class_conf_arr - thr_cls) / max(1e-6, 1.0 - thr_cls), 0.0, 1.0).astype(np.float32)

    selected_idx: List[int] = []
    dbg_dual: Dict[str, Any] = {}
    mode = str(getattr(cfg, "energy_mode", "legacy")).strip().lower()
    if mode not in ("four_term", "legacy"):
        raise ValueError(f"Unsupported energy_mode: {mode}")

    def _effective_pool_budget(n_total: int, m_cap: int) -> int:
        cap = int(m_cap)
        if cap <= 0:
            return 0
        if not bool(getattr(cfg, "adaptive_prefilter", True)):
            return cap
        base = int(max(0, getattr(cfg, "adaptive_prefilter_base", 128)))
        sqrt_scale = float(max(0.0, getattr(cfg, "adaptive_prefilter_sqrt_scale", 48.0)))
        m_min = int(max(1, getattr(cfg, "adaptive_prefilter_min", 256)))
        m_max = int(max(m_min, getattr(cfg, "adaptive_prefilter_max", 4096)))
        m_dyn = int(round(base + sqrt_scale * float(np.sqrt(max(0, int(n_total))))))
        m_dyn = max(m_min, min(m_dyn, m_max))
        return int(max(1, min(cap, m_dyn)))

    if mode == "four_term":
        # New staged solve for "keep + add" dual decision.
        # Stage A: keep high-confidence raw core.
        # Stage B: add warp candidates with coverage potential.
        # Stage C: global conflict resolve on union (raw core locked).
        use_dual = bool(getattr(cfg, "dual_head_solver", True))
        unary_term_full = (
            np.asarray([unary_energy_keep_on(float(keep_logits[i]), cfg) for i in range(N)], dtype=np.float32)
            + attr_energy_arr
            + rel_energy_arr
        )
        if not use_dual:
            margin = float(getattr(cfg, "energy_select_margin", 0.0))
            gate_cfg = float(getattr(cfg, "energy_prob_gate", 0.0))
            gate = float(cfg.keep_thr) if gate_cfg <= 0.0 else gate_cfg
            ordered = [i for i in np.argsort(unary_term_full).tolist() if float(keep_probs[i]) >= gate]
            selected_idx = _greedy_select(
                pool_idx=ordered,
                cands=cands,
                keep_logits=keep_logits,
                keep_probs=keep_probs,
                attr_energy_arr=attr_energy_arr,
                rel_energy_arr=rel_energy_arr,
                cfg=cfg,
                unary_model=unary_model,
                prob_thr=gate,
                already_selected=[],
                max_add=None,
                min_dist_to_selected=0.0,
                select_margin=margin,
                preserve_pool_order=True,
            )
            selected_before_refine = int(len(selected_idx))
        else:
            group = [_source_group(c) for c in cands]
            is_raw = np.asarray([1.0 if g == "raw" else 0.0 for g in group], dtype=np.float32)
            is_warp = np.asarray([1.0 if g == "warp" else 0.0 for g in group], dtype=np.float32)

            keep_head_logits = (
                keep_logits
                + float(getattr(cfg, "keep_head_raw_bias", 0.20)) * is_raw
                + float(getattr(cfg, "keep_head_nonraw_bias", -0.80)) * (1.0 - is_raw)
            ).astype(np.float32)
            keep_head_probs = _sigmoid(keep_head_logits)

            stage_b_min_dist = float(max(0.0, getattr(cfg, "stage_b_min_potential_dist", 1.2)))
            stage_b_min_p = float(max(0.0, getattr(cfg, "stage_b_min_potential", 0.25)))
            potential = np.zeros((N,), dtype=np.float32)
            raw_idx = [i for i in range(N) if group[i] == "raw"]
            if len(raw_idx) > 0:
                raw_xy = np.asarray([_cand_xy(cands[i]) for i in raw_idx], dtype=np.float32)
                for i in range(N):
                    if group[i] != "warp":
                        continue
                    xi = np.asarray(_cand_xy(cands[i]), dtype=np.float32)
                    d = raw_xy - xi[None, :]
                    dmin = float(np.min(np.linalg.norm(d, axis=1)))
                    potential[i] = np.float32(min(1.0, dmin / max(stage_b_min_dist, 1e-6)))
            else:
                potential[is_warp > 0.5] = 1.0

            add_head_logits = (
                keep_logits
                + float(getattr(cfg, "add_head_warp_bias", 0.25)) * is_warp
                + float(getattr(cfg, "add_head_nonwarp_bias", -6.0)) * (1.0 - is_warp)
                + float(getattr(cfg, "add_head_support_gain", 0.20)) * np.maximum(0.0, support_vals.astype(np.float32) - 1.0)
                + float(getattr(cfg, "add_head_potential_gain", 0.35)) * potential
                + float(getattr(cfg, "add_head_class_gain", 0.25)) * class_gate_arr
            ).astype(np.float32)
            add_head_probs = _sigmoid(add_head_logits)

            if bool(getattr(cfg, "dual_head_unified_context", True)):
                # Unified-context solve:
                # keep/add heads are fused into one candidate score over the full pool
                # to avoid irreversible raw-only hard filtering.
                # Soft-budget variant: do not hard-gate by probability before solve;
                # rank the full pool by fused logits, then cap by top-M budget.
                fused_logits = (
                    keep_head_logits
                    + float(getattr(cfg, "add_head_warp_bias", 0.25)) * is_warp
                    + float(getattr(cfg, "add_head_support_gain", 0.20)) * np.maximum(0.0, support_vals.astype(np.float32) - 1.0)
                    + float(getattr(cfg, "add_head_potential_gain", 0.35)) * potential
                    + float(getattr(cfg, "add_head_class_gain", 0.25)) * class_gate_arr
                ).astype(np.float32)
                fused_probs = _sigmoid(fused_logits)
                gate_cfg = float(getattr(cfg, "energy_prob_gate", 0.0))
                gate = float(cfg.keep_thr) if gate_cfg <= 0.0 else gate_cfg
                pool_all = list(range(N))
                M_cap = int(max(getattr(cfg, "prefilter_topm_seed", 0) or 0, getattr(cfg, "prefilter_topm_fill", 0) or 0))
                M_eff = _effective_pool_budget(n_total=N, m_cap=M_cap)
                if M_eff > 0 and len(pool_all) > M_eff:
                    pool_all = sorted(pool_all, key=lambda i: float(fused_logits[i]), reverse=True)[:M_eff]
                selected_idx = _greedy_select(
                    pool_idx=pool_all,
                    cands=cands,
                    keep_logits=fused_logits,
                    keep_probs=fused_probs,
                    attr_energy_arr=float(getattr(cfg, "stage_c_attr_scale", 0.20)) * attr_energy_arr,
                    rel_energy_arr=float(getattr(cfg, "stage_c_rel_scale", 0.20)) * rel_energy_arr,
                    cfg=cfg,
                    unary_model=unary_model,
                    prob_thr=0.0,
                    already_selected=[],
                    max_add=None,
                    min_dist_to_selected=0.0,
                    select_margin=float(getattr(cfg, "energy_select_margin", 0.0)),
                    preserve_pool_order=False,
                )
                selected_before_refine = int(len(selected_idx))
                dbg_dual = {
                    "solver_style": "unified_context",
                    "pool_size": int(len(pool_all)),
                    "stage_a_raw_core": 0,
                    "stage_b_add_total": int(len(selected_idx)),
                    "stage_c_final": int(len(selected_idx)),
                    "stage_b_pool": int(len(pool_all)),
                    "stage_b_mean_potential": float(np.mean(potential[pool_all])) if len(pool_all) > 0 else 0.0,
                    "stage_b_mean_class_conf": float(np.mean(class_conf_arr[pool_all])) if len(pool_all) > 0 else 0.0,
                    "soft_budget_no_gate": True,
                    "gate_ref_only": float(gate),
                    "pool_budget_cap": int(M_cap),
                    "pool_budget_eff": int(M_eff),
                }
            else:
                # Legacy staged dual-head solve.
                # Stage A: raw core
                pool_seed = [i for i in range(N) if group[i] == "raw"]
                M = int(getattr(cfg, "prefilter_topm_seed", 0) or 0)
                if M > 0 and len(pool_seed) > M:
                    pool_seed = sorted(pool_seed, key=lambda i: float(keep_head_logits[i]), reverse=True)[:M]
                selected_a = _greedy_select(
                    pool_idx=pool_seed,
                    cands=cands,
                    keep_logits=keep_head_logits,
                    keep_probs=keep_head_probs,
                    attr_energy_arr=np.zeros_like(attr_energy_arr),
                    rel_energy_arr=np.zeros_like(rel_energy_arr),
                    cfg=cfg,
                    unary_model=unary_model,
                    prob_thr=float(getattr(cfg, "stage_a_keep_thr", cfg.keep_thr)),
                    already_selected=[],
                    max_add=None,
                    min_dist_to_selected=0.0,
                    select_margin=float(getattr(cfg, "energy_select_margin", 0.0)),
                )
                # Fail-safe: avoid empty raw core that can cause over-filtering collapse.
                if len(selected_a) == 0 and len(pool_seed) > 0:
                    fallback_pool = sorted(pool_seed, key=lambda i: float(keep_head_logits[i]), reverse=True)
                    fallback_k = max(1, min(8, len(fallback_pool)))
                    selected_a = _greedy_select(
                        pool_idx=fallback_pool[:fallback_k],
                        cands=cands,
                        keep_logits=keep_head_logits,
                        keep_probs=keep_head_probs,
                        attr_energy_arr=np.zeros_like(attr_energy_arr),
                        rel_energy_arr=np.zeros_like(rel_energy_arr),
                        cfg=cfg,
                        unary_model=unary_model,
                        prob_thr=max(0.0, float(getattr(cfg, "stage_a_keep_thr", cfg.keep_thr)) * 0.7),
                        already_selected=[],
                        max_add=None,
                        min_dist_to_selected=0.0,
                        select_margin=float(getattr(cfg, "energy_select_margin", 0.0)),
                    )

                # Stage B: warp add with potential
                seed_set = set(int(i) for i in selected_a)
                pool_fill: List[int] = []
                for i in range(N):
                    if i in seed_set or group[i] != "warp":
                        continue
                    if float(add_head_probs[i]) < float(getattr(cfg, "stage_b_add_thr", cfg.fill_keep_thr)):
                        continue
                    if float(potential[i]) < stage_b_min_p:
                        continue
                    if bool(getattr(cfg, "stage_b_enforce_class_conf", False)) and float(class_gate_arr[i]) <= 0.0:
                        continue
                    pool_fill.append(i)
                M = int(getattr(cfg, "prefilter_topm_fill", 0) or 0)
                if M > 0 and len(pool_fill) > M:
                    pool_fill = sorted(pool_fill, key=lambda i: float(add_head_logits[i]), reverse=True)[:M]

                selected_ab = _greedy_select(
                    pool_idx=pool_fill,
                    cands=cands,
                    keep_logits=add_head_logits,
                    keep_probs=add_head_probs,
                    attr_energy_arr=np.zeros_like(attr_energy_arr),
                    rel_energy_arr=rel_energy_arr,
                    cfg=cfg,
                    unary_model=unary_model,
                    prob_thr=float(getattr(cfg, "stage_b_add_thr", cfg.fill_keep_thr)),
                    already_selected=selected_a,
                    max_add=int(cfg.max_fill),
                    min_dist_to_selected=float(cfg.min_dist_to_seed),
                    select_margin=float(getattr(cfg, "energy_select_margin", 0.0)),
                )

                # Stage C: global conflict resolve with weak priors, keep raw core locked.
                union_set = set(int(i) for i in selected_ab)
                stage_c_pool = [i for i in sorted(union_set, key=lambda t: float(keep_logits[t]), reverse=True) if i not in seed_set]
                stage_c_logits = keep_logits.copy()
                stage_c_probs = _sigmoid(stage_c_logits)
                selected_idx = _greedy_select(
                    pool_idx=stage_c_pool,
                    cands=cands,
                    keep_logits=stage_c_logits,
                    keep_probs=stage_c_probs,
                    attr_energy_arr=float(getattr(cfg, "stage_c_attr_scale", 0.20)) * attr_energy_arr,
                    rel_energy_arr=float(getattr(cfg, "stage_c_rel_scale", 0.20)) * rel_energy_arr,
                    cfg=cfg,
                    unary_model=unary_model,
                    prob_thr=0.0,
                    already_selected=selected_a,
                    max_add=None,
                    min_dist_to_selected=0.0,
                    select_margin=float(getattr(cfg, "energy_select_margin", 0.0)),
                )
                selected_before_refine = int(len(selected_idx))

                dbg_dual = {
                    "solver_style": "staged_legacy",
                    "stage_a_raw_core": int(len(selected_a)),
                    "stage_b_add_total": int(max(0, len(selected_ab) - len(selected_a))),
                    "stage_c_final": int(len(selected_idx)),
                    "stage_b_pool": int(len(pool_fill)),
                    "stage_b_mean_potential": float(np.mean(potential[pool_fill])) if len(pool_fill) > 0 else 0.0,
                    "stage_b_mean_class_conf": float(np.mean(class_conf_arr[pool_fill])) if len(pool_fill) > 0 else 0.0,
                }
    else:
        selected_before_refine = int(len(selected_idx))
    if mode != "four_term" and cfg.two_stage:
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
            rel_energy_arr=rel_energy_arr,
            cfg=cfg,
            unary_model=unary_model,
            prob_thr=float(cfg.seed_keep_thr),
            already_selected=[],
            max_add=None,
            min_dist_to_selected=0.0,
            select_margin=0.0,
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
            if (not bool(getattr(cfg, "soft_dt_support", True))) and int(support_vals[i]) < int(cfg.min_dt_support):
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
            rel_energy_arr=rel_energy_arr,
            cfg=cfg,
            unary_model=unary_model,
            prob_thr=float(cfg.fill_keep_thr),
            already_selected=selected_idx,
            max_add=int(cfg.max_fill),
            min_dist_to_selected=float(cfg.min_dist_to_seed),
            select_margin=0.0,
        )

    elif mode != "four_term":
        selected_idx = _greedy_select(
            pool_idx=list(range(N)),
            cands=cands,
            keep_logits=keep_logits,
            keep_probs=keep_probs,
            attr_energy_arr=attr_energy_arr,
            rel_energy_arr=rel_energy_arr,
            cfg=cfg,
            unary_model=unary_model,
            prob_thr=float(cfg.keep_thr),
            already_selected=[],
            max_add=None,
            min_dist_to_selected=0.0,
            select_margin=0.0,
        )

    # final topk guard (global)
    if cfg.topk is not None and len(selected_idx) > int(cfg.topk):
        selected_idx = sorted(selected_idx, key=lambda i: float(keep_logits[i]), reverse=True)[: int(cfg.topk)]

    dbg: Dict[str, Any] = {
        "energy_mode": str(mode),
        "num_cands": int(N),
        "selected": int(len(selected_idx)),
        "use_learned_unary": bool(unary_model is not None),
        "unary_ckpt_path": str(cfg.unary_ckpt_path) if cfg.unary_ckpt_path else None,
        "soft_dt_support": bool(getattr(cfg, "soft_dt_support", True)),
        "dt_shortfall_penalty": float(getattr(cfg, "dt_shortfall_penalty", 0.0)),
        "mean_rel_energy": float(np.mean(rel_energy_arr)) if rel_energy_arr.size > 0 else 0.0,
        "mean_class_conf": float(np.mean(class_conf_arr)) if class_conf_arr.size > 0 else 0.0,
    }
    if mode == "four_term":
        gate_cfg = float(getattr(cfg, "energy_prob_gate", 0.0))
        dbg["energy_prob_gate_cfg"] = gate_cfg
        dbg["energy_prob_gate_eff"] = float(cfg.keep_thr) if gate_cfg <= 0.0 else gate_cfg
        dbg["energy_select_margin"] = float(getattr(cfg, "energy_select_margin", 0.0))
        dbg["energy_local_refine"] = False
        dbg["energy_local_refine_rounds"] = 0
        dbg["selected_before_local_refine"] = int(selected_before_refine)
        dbg["dual_head_solver"] = bool(getattr(cfg, "dual_head_solver", True))
        dbg.update(dbg_dual)
    # Internal cache for ebm_infer_candidates to avoid recomputing keep/attr stats.
    dbg["_cache"] = {
        "keep_logits": keep_logits,
        "keep_probs": keep_probs,
        "cell_speed_map": cell_speed_map,
        "class_logits": class_logits_arr,
        "attr_logits": attr_logits_arr,
        "class_labels": class_labels_arr,
        "attr_names": attr_names_list,
    }
    if cfg.include_energy:
        # optional energy summary on selected set (still cheap since selected is small)
        tot_unary = 0.0
        tot_pair = 0.0
        for ii in selected_idx:
            tot_unary += float(
                unary_energy_keep_on(float(keep_logits[ii]), cfg)
                + float(attr_energy_arr[ii])
                + float(rel_energy_arr[ii])
            )
        for a in range(len(selected_idx)):
            for b in range(a + 1, len(selected_idx)):
                tot_pair += float(pair_energy(cands[selected_idx[a]], cands[selected_idx[b]], cfg, unary_model=unary_model))
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
        class_logits_arr = cache.get("class_logits", None)
        attr_logits_arr = cache.get("attr_logits", None)
        class_labels_arr = cache.get("class_labels", None)
        attr_names_list = cache.get("attr_names", None)
    else:
        keep_logits = None
        keep_probs = None
        cell_speed_map = None
        class_logits_arr = None
        attr_logits_arr = None
        class_labels_arr = None
        attr_names_list = None

    if keep_logits is None or keep_probs is None:
        cell_count, cell_dtset = build_cell_stats(cands, cfg)
        support_vals = np.array([cell_support_value(c, cfg, cell_count, cell_dtset) for c in cands], dtype=np.int64)
        cell_xy_feat = float(cfg.support_cell_xy if cfg.support_cell_xy > 0 else cfg.dt_cell_size)
        sup_count_map, sup_udt_map, local_density_map = _build_unary_support_xy_maps(cands, cell_xy=cell_xy_feat)
        unary_model: Optional[_UnaryLinearModel] = None
        if bool(cfg.unary_use_learned):
            unary_model = _load_unary_model(cfg.unary_ckpt_path)
        if unary_model is not None:
            pred_all = unary_model.predict_all(
                cands,
                support_count_map=sup_count_map,
                support_udt_map=sup_udt_map,
                local_density_map=local_density_map,
                cell_xy=cell_xy_feat,
            )
            keep_logits = np.asarray(pred_all.get("keep_logits", np.zeros((len(cands),), dtype=np.float32)), dtype=np.float32)
            class_logits_arr = pred_all.get("class_logits", None)
            attr_logits_arr = pred_all.get("attr_logits", None)
            if unary_model.class_enabled and class_logits_arr is not None:
                class_labels_arr = np.asarray(unary_model.class_labels, dtype=np.int64).reshape(-1)
            if unary_model.attr_enabled and attr_logits_arr is not None:
                attr_names_list = [str(x) for x in unary_model.attr_names]
        else:
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
        if (
            bool(getattr(cfg, "use_learned_class", True))
            and isinstance(class_logits_arr, np.ndarray)
            and class_logits_arr.ndim == 2
            and i < class_logits_arr.shape[0]
            and class_logits_arr.shape[1] > 0
        ):
            cls_prob = _softmax_logits(class_logits_arr[i, :], axis=-1).reshape(-1)
            cls_idx = int(np.argmax(cls_prob))
            cls_conf = float(cls_prob[cls_idx])
            if cls_conf >= float(getattr(cfg, "learned_class_min_prob", 0.55)):
                if isinstance(class_labels_arr, np.ndarray) and cls_idx < class_labels_arr.shape[0]:
                    lab = int(class_labels_arr[cls_idx])
                else:
                    lab = int(cls_idx)
        if (
            bool(getattr(cfg, "use_learned_attr", True))
            and isinstance(attr_logits_arr, np.ndarray)
            and attr_logits_arr.ndim == 2
            and i < attr_logits_arr.shape[0]
            and attr_logits_arr.shape[1] > 0
        ):
            at_prob = _softmax_logits(attr_logits_arr[i, :], axis=-1).reshape(-1)
            at_idx = int(np.argmax(at_prob))
            at_conf = float(at_prob[at_idx])
            if at_conf >= float(getattr(cfg, "learned_attr_min_prob", 0.50)):
                if isinstance(attr_names_list, list) and at_idx < len(attr_names_list):
                    attr_name = str(attr_names_list[at_idx])

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
            "dual_head_solver",
            "stage_a_keep_thr",
            "stage_b_add_thr",
            "stage_b_min_potential_dist",
            "stage_b_min_potential",
            "dual_head_unified_context",
            "stage_c_attr_scale",
            "stage_c_rel_scale",
        ]:
            if hasattr(infer_cfg, name):
                kw[name] = getattr(infer_cfg, name)

        # if you later add CLI flags for these, they will auto-merge too
        for name in [
            "prefilter_topm_seed",
            "prefilter_topm_fill",
            "adaptive_prefilter",
            "adaptive_prefilter_base",
            "adaptive_prefilter_sqrt_scale",
            "adaptive_prefilter_min",
            "adaptive_prefilter_max",
        ]:
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
