# infer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference (MVP) for detection refinement.

Goal (MVP):
- Take FrameData.candidates (from data.py) and select a refined set of detections
  with better precision/recall trade-off than raw det.
- Output per-frame payload:
    {
      "boxes_3d": [[...9...], ...],
      "labels_3d": [...],
      "scores_3d": [...],
      "attrs":    [...],          # optional but recommended (moving/standing/parked)
      "sources":  [...],          # optional debugging (raw / warp(dt=+1))
    }

This is NOT a full EBM yet. It's an "EBM-like" heuristic scorer + diversity constraint:
- unary: effective_score = score * source_bias * (1 + support_boost)
- pairwise: distance-NMS to avoid duplicates (approx. repulsive energy)
- optional: allow warped candidates to fill gaps (recall)

NEW: two-stage "fill holes + dt-support" mode:
- Stage-1: keep high-precision seeds (raw-only by default)
- Stage-2: add warp candidates ONLY if they have dt-support (>=K distinct dt in same spatial cell),
           and they are not too close to a seed (avoid duplicates).
- Final: merge seeds+fills, run NMS + topk.

You can later replace scoring/NMS with true energy minimization, but keep the same I/O.

Repo structure note:
- You renamed io.py -> json_io.py (good; avoids shadowing stdlib io)
- This module assumes FrameData/Candidate are from data.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from json_io import sample_key_candidates
from data import FrameData, Candidate


# -----------------------------
# Config
# -----------------------------

@dataclass
class AttrConfig:
    """
    Attribute prediction from velocity (vx, vy) in the box.
    If velocity is unreliable/missing, this will default to "parked" or "standing".
    """
    # speed thresholds (m/s)
    moving_thr: float = 0.5
    parked_thr: float = 0.1

    # if a class is known-static, force parked/standing
    static_label_ids: Optional[Set[int]] = None

    # output strings (keep simple; metrics.py can map by substrings)
    moving_name: str = "moving"
    standing_name: str = "standing"
    parked_name: str = "parked"


@dataclass
class InferConfig:
    # candidate usage
    use_sources: str = "all"   # "raw" or "all" (allow warp candidates)
    raw_only_score_min: float = 0.0  # optional prefilter on raw candidates

    # score shaping
    bias_raw: float = 1.00
    bias_warp: float = 0.90     # penalize warped candidates a bit
    bias_other: float = 0.95

    # temporal support boost (helps recall when multiple neighbors agree)
    enable_support_boost: bool = True
    support_cluster_xy: float = 1.0      # meters; cluster by rounding x,y to this size
    support_weight: float = 0.15         # score multiplier factor per extra support

    # selection constraints
    keep_thr: float = 0.5                # threshold on effective score
    topk: int = 200                      # max outputs per frame (after NMS)
    nms_thr_xy: float = 1.5              # meters; suppress boxes too close (same class by default)
    nms_cross_class: bool = False        # if True, NMS across all labels

    # attribute prediction
    predict_attr: bool = True
    attr_cfg: AttrConfig = field(default_factory=AttrConfig)

    # optional: cap per-class outputs (avoid one class dominating)
    max_per_class: Optional[int] = None

    # optional: if True, keep debug "sources" list in payload
    include_sources: bool = True

    # ===== two-stage fill (optional) =====
    two_stage: bool = False

    # Stage-1 seeds (high precision)
    seed_keep_thr: float = 0.5           # usually == keep_thr
    seed_sources: str = "raw"            # "raw" or "all"

    # Stage-2 fill (recall)
    fill_keep_thr: float = 0.35          # lower threshold for fill candidates
    min_dt_support: int = 2              # require >= K distinct dt support in same cell (label-aware)
    dt_cell_size: float = 1.0            # meters; spatial hashing cell size for dt-support
    min_dist_to_seed: float = 0.8        # meters; too close to seeds => likely duplicate, skip
    max_fill: int = 300                  # cap fill count per frame before final merge


# -----------------------------
# Utilities
# -----------------------------

def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(axis=-1) if x.ndim > 1 else np.isfinite(x)


def _cand_center_xy(c: Candidate) -> Tuple[float, float]:
    b = c.box
    return float(b[0]), float(b[1])


def _cand_speed(c: Candidate) -> float:
    b = c.box
    if b.shape[0] >= 9:
        vx, vy = float(b[7]), float(b[8])
        if not (np.isfinite(vx) and np.isfinite(vy)):
            return 0.0
        return float(np.hypot(vx, vy))
    return 0.0


def predict_attr_from_vel(c: Candidate, cfg: AttrConfig) -> str:
    label = int(c.label)
    if cfg.static_label_ids and label in cfg.static_label_ids:
        return cfg.parked_name

    v = _cand_speed(c)
    if v >= float(cfg.moving_thr):
        return cfg.moving_name
    if v <= float(cfg.parked_thr):
        return cfg.parked_name
    return cfg.standing_name


def _source_bias(c: Candidate, cfg: InferConfig) -> float:
    if c.source == "raw" and c.from_dt == 0:
        return float(cfg.bias_raw)
    if c.source.startswith("warp(") or c.from_dt != 0:
        return float(cfg.bias_warp)
    return float(cfg.bias_other)


def _support_key(c: Candidate, cluster_xy: float) -> Tuple[int, int, int]:
    x, y = _cand_center_xy(c)
    gx = int(np.round(x / cluster_xy))
    gy = int(np.round(y / cluster_xy))
    return (int(c.label), gx, gy)


def compute_support_counts(cands: List[Candidate], cfg: InferConfig) -> Dict[Tuple[int, int, int], int]:
    """
    Count how many candidates fall into the same (label, grid_x, grid_y) cluster.
    Cheap proxy for temporal agreement (esp. with warp candidates).
    """
    if (not cfg.enable_support_boost) or cfg.support_cluster_xy <= 0:
        return {}
    cluster = float(cfg.support_cluster_xy)

    counts: Dict[Tuple[int, int, int], int] = {}
    for c in cands:
        k = _support_key(c, cluster)
        counts[k] = counts.get(k, 0) + 1
    return counts


def effective_scores(cands: List[Candidate], cfg: InferConfig) -> np.ndarray:
    """
    score_eff = base_score * source_bias * (1 + support_weight * (support-1))
    """
    if len(cands) == 0:
        return np.zeros((0,), dtype=np.float32)

    base = np.array([float(c.score) for c in cands], dtype=np.float32)
    base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
    bias = np.array([_source_bias(c, cfg) for c in cands], dtype=np.float32)

    if cfg.enable_support_boost and cfg.support_cluster_xy > 0:
        counts = compute_support_counts(cands, cfg)
        sup = np.array(
            [counts.get(_support_key(c, float(cfg.support_cluster_xy)), 1) for c in cands],
            dtype=np.float32
        )
        boost = 1.0 + float(cfg.support_weight) * np.maximum(0.0, sup - 1.0)
    else:
        boost = 1.0

    return base * bias * boost


def _prefilter_candidates(frame: FrameData, cfg: InferConfig) -> List[Candidate]:
    """
    Eligible candidates for single-stage inference:
      - cfg.use_sources: "raw" => only raw dt=0
                        "all" => all candidates
      - optional: raw_only_score_min
    """
    cands = frame.candidates
    if cfg.use_sources == "raw":
        cands = [c for c in cands if (c.source == "raw" and c.from_dt == 0)]
        if cfg.raw_only_score_min > 0.0:
            thr = float(cfg.raw_only_score_min)
            cands = [c for c in cands if float(c.score) >= thr]
        return cands
    if cfg.use_sources == "all":
        return cands
    raise ValueError(f"Unknown use_sources={cfg.use_sources}. Use 'raw' or 'all'.")


def _nms_greedy_xy(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    thr_xy: float,
    cross_class: bool = False,
    max_per_class: Optional[int] = None,
) -> List[int]:
    """
    Greedy distance-NMS in XY plane.
    - cross_class=False: NMS within each class separately.
    - cross_class=True: NMS across all classes.
    Returns kept indices (in descending score order).
    """
    N = int(boxes.shape[0])
    if N == 0:
        return []

    thr2 = float(thr_xy) ** 2
    xy = boxes[:, :2].astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    labels = labels.astype(np.int64, copy=False)

    order = np.argsort(-scores)  # descending
    keep: List[int] = []

    if cross_class:
        suppressed = np.zeros((N,), dtype=bool)
        for i in order:
            if suppressed[i]:
                continue
            keep.append(int(i))
            dx = xy[:, 0] - xy[i, 0]
            dy = xy[:, 1] - xy[i, 1]
            d2 = dx * dx + dy * dy
            suppressed |= (d2 <= thr2)
        return keep

    max_pc = int(max_per_class) if max_per_class is not None else None
    suppressed_by_class: Dict[int, np.ndarray] = {}
    kept_count_by_class: Dict[int, int] = {}

    for i in order:
        c = int(labels[i])
        if max_pc is not None and kept_count_by_class.get(c, 0) >= max_pc:
            continue

        if c not in suppressed_by_class:
            suppressed_by_class[c] = np.zeros((N,), dtype=bool)
            kept_count_by_class[c] = 0

        if suppressed_by_class[c][i]:
            continue

        keep.append(int(i))
        kept_count_by_class[c] += 1

        dx = xy[:, 0] - xy[i, 0]
        dy = xy[:, 1] - xy[i, 1]
        d2 = dx * dx + dy * dy
        suppressed_by_class[c] |= (d2 <= thr2)

    return keep


def _empty_payload(cfg: InferConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"boxes_3d": [], "labels_3d": [], "scores_3d": []}
    if cfg.predict_attr:
        payload["attrs"] = []
    if cfg.include_sources:
        payload["sources"] = []
    return payload


def _dt_support_key(c: Candidate, cell_size: float) -> Tuple[int, int, int]:
    """(label, ix, iy) spatial hash key for dt-support."""
    x, y = _cand_center_xy(c)
    cs = float(cell_size)
    ix = int(np.floor(x / cs))
    iy = int(np.floor(y / cs))
    return (int(c.label), ix, iy)


def _build_dt_support_sets(cands: List[Candidate], cell_size: float) -> Dict[Tuple[int, int, int], Set[int]]:
    """
    For each (label, cell), collect the set of distinct dt values present.
    dt-support = |set|.
    """
    sup: Dict[Tuple[int, int, int], Set[int]] = {}
    cs = float(cell_size)
    for c in cands:
        k = _dt_support_key(c, cs)
        if k not in sup:
            sup[k] = set()
        sup[k].add(int(c.from_dt))
    return sup


def _min_dist_xy_to_set(xy: np.ndarray, boxes: np.ndarray) -> float:
    """Return min L2 distance in XY from xy to any box center in boxes."""
    if boxes.shape[0] == 0:
        return float("inf")
    d = boxes[:, :2] - xy[None, :]
    return float(np.min(np.linalg.norm(d, axis=1)))


# -----------------------------
# Core inference
# -----------------------------

def infer_frame_single_stage(frame: FrameData, cfg: InferConfig) -> Dict[str, Any]:
    """
    Original single-stage inference (your current logic), kept as-is.
    """
    cands = _prefilter_candidates(frame, cfg)
    if len(cands) == 0:
        return _empty_payload(cfg)

    eff = effective_scores(cands, cfg)
    keep0 = eff >= float(cfg.keep_thr)
    if not np.any(keep0):
        return _empty_payload(cfg)

    idx0 = np.nonzero(keep0)[0].astype(np.int64)
    cands0 = [cands[i] for i in idx0.tolist()]
    eff0 = eff[idx0]

    boxes = np.stack([c.box for c in cands0], axis=0).astype(np.float32)
    labels = np.array([c.label for c in cands0], dtype=np.int64)
    scores = eff0.astype(np.float32)

    ok = _finite(boxes[:, :3])
    if not np.all(ok):
        boxes = boxes[ok]
        labels = labels[ok]
        scores = scores[ok]
        cands0 = [c for c, m in zip(cands0, ok.tolist()) if m]

    kept_idx = _nms_greedy_xy(
        boxes=boxes,
        labels=labels,
        scores=scores,
        thr_xy=float(cfg.nms_thr_xy),
        cross_class=bool(cfg.nms_cross_class),
        max_per_class=cfg.max_per_class,
    )

    if len(kept_idx) == 0:
        return _empty_payload(cfg)

    kept_idx = sorted(kept_idx, key=lambda i: float(scores[i]), reverse=True)

    if cfg.topk is not None and len(kept_idx) > int(cfg.topk):
        kept_idx = kept_idx[: int(cfg.topk)]

    boxes_kept = boxes[kept_idx]
    labels_kept = labels[kept_idx]
    scores_kept = scores[kept_idx]

    payload: Dict[str, Any] = {
        "boxes_3d": boxes_kept.tolist(),
        "labels_3d": labels_kept.astype(int).tolist(),
        "scores_3d": scores_kept.astype(float).tolist(),
    }

    if cfg.predict_attr:
        payload["attrs"] = [predict_attr_from_vel(cands0[i], cfg.attr_cfg) for i in kept_idx]

    if cfg.include_sources:
        payload["sources"] = [cands0[i].source for i in kept_idx]

    return payload


def infer_frame_two_stage(frame: FrameData, cfg: InferConfig) -> Dict[str, Any]:
    """
    Two-stage:
      Stage-1: high-precision seeds (raw-only by default)
      Stage-2: fill from warp candidates with dt-support >= K and lower threshold
      Final: merge, NMS, topk
    """
    cands_all = list(frame.candidates)
    if len(cands_all) == 0:
        return _empty_payload(cfg)

    # -------- Stage-1 seed candidate pool --------
    if cfg.seed_sources == "raw":
        seed_pool = [c for c in cands_all if (c.source == "raw" and c.from_dt == 0)]
    elif cfg.seed_sources == "all":
        seed_pool = cands_all
    else:
        raise ValueError(f"seed_sources must be 'raw' or 'all', got: {cfg.seed_sources}")

    # -------- Stage-2 fill pool (only if cfg.use_sources == 'all') --------
    if cfg.use_sources != "all":
        fill_pool: List[Candidate] = []
    else:
        fill_pool = [c for c in cands_all if (c.from_dt != 0 or c.source.startswith("warp("))]

    # ===== Stage-1: select seeds =====
    if len(seed_pool) == 0:
        seed_boxes = np.zeros((0, 9), dtype=np.float32)
        seed_labels = np.zeros((0,), dtype=np.int64)
        seed_scores = np.zeros((0,), dtype=np.float32)
        seed_cands_kept: List[Candidate] = []
    else:
        eff_seed = effective_scores(seed_pool, cfg)
        thr_seed = float(cfg.seed_keep_thr) if cfg.seed_keep_thr is not None else float(cfg.keep_thr)
        keep_seed = eff_seed >= thr_seed
        if not np.any(keep_seed):
            seed_boxes = np.zeros((0, 9), dtype=np.float32)
            seed_labels = np.zeros((0,), dtype=np.int64)
            seed_scores = np.zeros((0,), dtype=np.float32)
            seed_cands_kept = []
        else:
            idx = np.nonzero(keep_seed)[0].astype(np.int64)
            seed_cands = [seed_pool[i] for i in idx.tolist()]
            seed_scores = eff_seed[idx].astype(np.float32)

            seed_boxes = np.stack([c.box for c in seed_cands], axis=0).astype(np.float32)
            seed_labels = np.array([c.label for c in seed_cands], dtype=np.int64)

            ok = _finite(seed_boxes[:, :3])
            if not np.all(ok):
                seed_boxes = seed_boxes[ok]
                seed_labels = seed_labels[ok]
                seed_scores = seed_scores[ok]
                seed_cands = [c for c, m in zip(seed_cands, ok.tolist()) if m]

            kept_idx = _nms_greedy_xy(
                boxes=seed_boxes,
                labels=seed_labels,
                scores=seed_scores,
                thr_xy=float(cfg.nms_thr_xy),
                cross_class=bool(cfg.nms_cross_class),
                max_per_class=None,  # don't cap seeds per-class here; cap at final merge if needed
            )
            if len(kept_idx) == 0:
                seed_boxes = np.zeros((0, 9), dtype=np.float32)
                seed_labels = np.zeros((0,), dtype=np.int64)
                seed_scores = np.zeros((0,), dtype=np.float32)
                seed_cands_kept = []
            else:
                kept_idx = sorted(kept_idx, key=lambda i: float(seed_scores[i]), reverse=True)
                # (optional) seed topk — keep generous, final topk will handle
                seed_boxes = seed_boxes[kept_idx]
                seed_labels = seed_labels[kept_idx]
                seed_scores = seed_scores[kept_idx]
                seed_cands_kept = [seed_cands[i] for i in kept_idx]

    # ===== Stage-2: select fills (dt-support + not-close-to-seed) =====
    if len(fill_pool) == 0:
        fill_boxes = np.zeros((0, 9), dtype=np.float32)
        fill_labels = np.zeros((0,), dtype=np.int64)
        fill_scores = np.zeros((0,), dtype=np.float32)
        fill_cands_kept: List[Candidate] = []
    else:
        # dt-support sets are built from the full fill_pool (before thresholding)
        sup_sets = _build_dt_support_sets(fill_pool, cell_size=float(cfg.dt_cell_size))

        eff_fill = effective_scores(fill_pool, cfg)
        thr_fill = float(cfg.fill_keep_thr)

        selected_fill: List[Candidate] = []
        selected_scores: List[float] = []
        selected_boxes_xy: List[np.ndarray] = []

        # iterate fill candidates by descending effective score
        order = np.argsort(-eff_fill)
        for ii in order.tolist():
            c = fill_pool[int(ii)]
            sc = float(eff_fill[int(ii)])
            if sc < thr_fill:
                break

            # dt-support requirement
            key = _dt_support_key(c, float(cfg.dt_cell_size))
            dt_sup = len(sup_sets.get(key, set()))
            if dt_sup < int(cfg.min_dt_support):
                continue

            # too close to any seed => skip (avoid duplicates)
            if seed_boxes.shape[0] > 0 and float(cfg.min_dist_to_seed) > 0:
                xy = np.array(_cand_center_xy(c), dtype=np.float32)
                if _min_dist_xy_to_set(xy, seed_boxes) < float(cfg.min_dist_to_seed):
                    continue

            # (optional) also avoid being too close to already selected fills
            if len(selected_boxes_xy) > 0 and float(cfg.nms_thr_xy) > 0:
                xy = np.array(_cand_center_xy(c), dtype=np.float32)
                d = np.stack(selected_boxes_xy, axis=0) - xy[None, :]
                if float(np.min(np.linalg.norm(d, axis=1))) < float(cfg.nms_thr_xy) * 0.75:
                    continue

            selected_fill.append(c)
            selected_scores.append(sc)
            selected_boxes_xy.append(np.array(_cand_center_xy(c), dtype=np.float32))

            if len(selected_fill) >= int(cfg.max_fill):
                break

        if len(selected_fill) == 0:
            fill_boxes = np.zeros((0, 9), dtype=np.float32)
            fill_labels = np.zeros((0,), dtype=np.int64)
            fill_scores = np.zeros((0,), dtype=np.float32)
            fill_cands_kept = []
        else:
            fill_boxes = np.stack([c.box for c in selected_fill], axis=0).astype(np.float32)
            fill_labels = np.array([c.label for c in selected_fill], dtype=np.int64)
            fill_scores = np.array(selected_scores, dtype=np.float32)
            fill_cands_kept = selected_fill

            ok = _finite(fill_boxes[:, :3])
            if not np.all(ok):
                fill_boxes = fill_boxes[ok]
                fill_labels = fill_labels[ok]
                fill_scores = fill_scores[ok]
                fill_cands_kept = [c for c, m in zip(fill_cands_kept, ok.tolist()) if m]

            # NMS within fill set
            kept_idx = _nms_greedy_xy(
                boxes=fill_boxes,
                labels=fill_labels,
                scores=fill_scores,
                thr_xy=float(cfg.nms_thr_xy),
                cross_class=bool(cfg.nms_cross_class),
                max_per_class=None,
            )
            kept_idx = sorted(kept_idx, key=lambda i: float(fill_scores[i]), reverse=True)
            fill_boxes = fill_boxes[kept_idx]
            fill_labels = fill_labels[kept_idx]
            fill_scores = fill_scores[kept_idx]
            fill_cands_kept = [fill_cands_kept[i] for i in kept_idx]

    # ===== Merge + final NMS + topk =====
    all_cands: List[Candidate] = []
    all_scores_list: List[float] = []
    if seed_boxes.shape[0] > 0:
        all_cands.extend(seed_cands_kept)
        all_scores_list.extend(seed_scores.astype(float).tolist())
    if fill_boxes.shape[0] > 0:
        all_cands.extend(fill_cands_kept)
        all_scores_list.extend(fill_scores.astype(float).tolist())

    if len(all_cands) == 0:
        return _empty_payload(cfg)

    boxes = np.stack([c.box for c in all_cands], axis=0).astype(np.float32)
    labels = np.array([c.label for c in all_cands], dtype=np.int64)
    scores = np.array(all_scores_list, dtype=np.float32)

    ok = _finite(boxes[:, :3])
    if not np.all(ok):
        boxes = boxes[ok]
        labels = labels[ok]
        scores = scores[ok]
        all_cands = [c for c, m in zip(all_cands, ok.tolist()) if m]

    kept_idx = _nms_greedy_xy(
        boxes=boxes,
        labels=labels,
        scores=scores,
        thr_xy=float(cfg.nms_thr_xy),
        cross_class=bool(cfg.nms_cross_class),
        max_per_class=cfg.max_per_class,
    )
    if len(kept_idx) == 0:
        return _empty_payload(cfg)

    kept_idx = sorted(kept_idx, key=lambda i: float(scores[i]), reverse=True)
    if cfg.topk is not None and len(kept_idx) > int(cfg.topk):
        kept_idx = kept_idx[: int(cfg.topk)]

    boxes_kept = boxes[kept_idx]
    labels_kept = labels[kept_idx]
    scores_kept = scores[kept_idx]
    cands_kept = [all_cands[i] for i in kept_idx]

    payload: Dict[str, Any] = {
        "boxes_3d": boxes_kept.tolist(),
        "labels_3d": labels_kept.astype(int).tolist(),
        "scores_3d": scores_kept.astype(float).tolist(),
    }

    if cfg.predict_attr:
        payload["attrs"] = [predict_attr_from_vel(c, cfg.attr_cfg) for c in cands_kept]

    if cfg.include_sources:
        # keep explicit dt in source for debugging if you want
        srcs = []
        for c in cands_kept:
            if c.from_dt != 0:
                srcs.append(f"{c.source}(dt={c.from_dt:+d})" if "dt=" not in c.source else c.source)
            else:
                srcs.append(c.source)
        payload["sources"] = srcs

    return payload


def infer_frame(frame: FrameData, cfg: InferConfig) -> Dict[str, Any]:
    """
    Infer refined detections for ONE frame.
    """
    if cfg.two_stage:
        return infer_frame_two_stage(frame, cfg)
    return infer_frame_single_stage(frame, cfg)


def infer_scene(frames: List[FrameData], cfg: InferConfig) -> Dict[str, Dict[str, Any]]:
    """
    Infer a whole scene and return a replacement map for json writeback.

    repl_map: key -> payload
    Where key is chosen from sample_key_candidates(sample) in json_io.py
    (sample_token > sample_data_token > timestamp str).
    """
    repl: Dict[str, Dict[str, Any]] = {}

    for fr in frames:
        payload = infer_frame(fr, cfg)
        keys = sample_key_candidates(fr.sample)
        chosen = None
        for k in keys:
            if k and str(k) != "":
                chosen = str(k)
                break
        if chosen is None:
            chosen = f"{fr.scene_token}:{fr.timestamp}"
        repl[chosen] = payload

    return repl


def infer_frames_by_scene(frames_by_scene: Dict[str, List[FrameData]], cfg: InferConfig) -> Dict[str, Dict[str, Any]]:
    """
    Infer all scenes.
    Returns a global repl_map suitable for json_io.write_refined_json(...).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for _, frames in frames_by_scene.items():
        out.update(infer_scene(frames, cfg))
    return out


# -----------------------------
# Small self-check (optional)
# -----------------------------

def _debug_summary(payload: Dict[str, Any]) -> str:
    n = len(payload.get("boxes_3d", []))
    if n == 0:
        return "n=0"
    s = payload.get("scores_3d", [])
    return f"n={n} score[min/mean/max]={min(s):.3f}/{(sum(s)/len(s)):.3f}/{max(s):.3f}"


if __name__ == "__main__":
    print("infer.py is a library module. Use scripts/test_phase1_infer.py to execute end-to-end.")
