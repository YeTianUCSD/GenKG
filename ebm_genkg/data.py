# data.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data utilities for EBM-based refinement.

This module does THREE things (kept in one file to avoid repo bloat):
1) Parse a nuScenes-style "scene JSON" (already loaded into python dicts) into
   per-frame numpy arrays for det/gt.
2) Build per-frame *candidate lists* (raw + optional warp candidates).
3) Provide GT-matching helpers that generate supervision / evaluation targets:
   keep_target, class_target, attr_target (for matched dets).

It is intentionally numpy-only (no torch), so it can be used by:
- training scripts (to generate targets),
- inference scripts (to build candidates),
- evaluation scripts (to compute oracle coverage etc.).

Assumptions about box format:
  det["boxes_3d"] is a list of boxes, each is:
    [x,y,z, dx,dy,dz, yaw, vx,vy]  (len=9)
  If vx,vy missing (len<9), we pad zeros.

GT:
  gt["boxes_3d"] same format (len=9 or len>=7)
  gt["labels_3d"] list[int]
  gt["attr_names"] list[str] (optional) OR any attribute field you have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# local imports (your repo root)
from geom import T_k2ref, T_l2g_from_sample, transform_points, transform_boxes_center


# -----------------------------
# Config
# -----------------------------

@dataclass
class CandidateConfig:
    # raw det filtering
    raw_score_min: float = 0.0
    max_per_frame: Optional[int] = None  # keep top-K by raw score; None = keep all

    # candidate augmentation: warp from neighbor frames into the target frame
    use_warp: bool = False
    warp_radius: int = 2              # use neighbors within +/- radius
    warp_topk: int = 200              # from neighbor frame, only warp top-K by score
    warp_score_decay: float = 1.0     # optionally downweight warped scores (e.g. 0.9)

    # class filtering (ignored in GT matching and optionally in candidates)
    ignore_classes: Optional[Set[int]] = None

    # GT matching threshold (XY distance in *target frame lidar coords*)
    match_thr_xy: float = 2.0


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Candidate:
    # box in the *current frame lidar coordinate*
    box: np.ndarray          # (9,)
    score: float
    label: int
    source: str              # "raw" or "warp(dt=+1)" etc.
    from_dt: int             # 0 for raw, else neighbor offset

    # cached (optional) global center (x,y,z) for scene-level linking
    center_global: Optional[np.ndarray] = None  # (3,)


@dataclass
class FrameData:
    sample: Dict[str, Any]          # original JSON node (mutable reference)
    scene_token: str
    index_in_scene: int             # 0..T-1
    timestamp: int

    # pose
    T_l2g: np.ndarray               # (4,4)

    # candidates (det side)
    candidates: List[Candidate]

    # GT (for train/eval)
    gt_boxes: np.ndarray            # (G,9)
    gt_labels: np.ndarray           # (G,)
    gt_attrs: List[str]             # len==G (empty strings if missing)


@dataclass
class MatchTargets:
    """
    Targets for *center-frame* dets/candidates (usually you match only raw dets,
    but you can also match all candidates if you want oracle/analysis).
    """
    keep: np.ndarray        # (N,) 0/1
    cls_tgt: np.ndarray     # (N,) matched GT label, -100 for unmatched
    attr_tgt: List[str]     # len==N, "" for unmatched
    num_gt_valid: int       # GT count after ignore_classes filtering


@dataclass
class CandidateGtLinks:
    """
    Dense candidate/gt link structure for set-style supervision.

    - cand_to_gt: one best gt index per candidate (-1 for no feasible gt).
    - cand_to_gt_dist: distance of cand_to_gt in XY space (inf for unmatched).
    - gt_best_cand: one best candidate index per gt (-1 for uncovered gt).
    - cand_cover_counts: how many gt each candidate can cover within class+distance gate.
    - gt_candidate_counts: how many candidates can cover each gt within class+distance gate.
    """
    cand_to_gt: np.ndarray
    cand_to_gt_dist: np.ndarray
    gt_best_cand: np.ndarray
    cand_cover_counts: np.ndarray
    gt_candidate_counts: np.ndarray


# -----------------------------
# Parsing helpers
# -----------------------------

def _safe_np_boxes(x: Any, exp_dim: int = 9) -> np.ndarray:
    """
    Convert boxes list into float32 numpy array (N,exp_dim).
    If box length < exp_dim, pad with zeros.
    """
    if x is None:
        return np.zeros((0, exp_dim), dtype=np.float32)

    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, exp_dim), dtype=np.float32)

    if arr.ndim == 1:
        # single box or flat list
        if arr.shape[0] == exp_dim:
            arr = arr.reshape(1, exp_dim)
        else:
            # could be a single box with shorter length
            arr = arr.reshape(1, -1)

    # pad/truncate
    if arr.shape[1] < exp_dim:
        pad = np.zeros((arr.shape[0], exp_dim - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > exp_dim:
        arr = arr[:, :exp_dim]

    return arr.astype(np.float32, copy=False)


def _safe_np_1d(x: Any, dtype, n: int) -> np.ndarray:
    """
    Convert list/scalar to 1D array length n if possible.
    If scalar and n>0, broadcast. If missing, zeros.
    """
    if x is None:
        return np.zeros((n,), dtype=dtype)
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0:
        return np.full((n,), arr.item(), dtype=dtype)
    arr = arr.reshape(-1)
    if arr.size == 0 and n > 0:
        return np.zeros((n,), dtype=dtype)
    return arr.astype(dtype, copy=False)


def parse_det(sample: Dict[str, Any],
              cfg: CandidateConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse det.{boxes_3d,scores_3d,labels_3d} into numpy arrays.
    Applies score_min + topK (max_per_frame).
    Returns: boxes(N,9), scores(N,), labels(N,)
    """
    det = sample.get("det", {}) or {}
    boxes = _safe_np_boxes(det.get("boxes_3d", []) or [], exp_dim=9)
    N = boxes.shape[0]
    scores = _safe_np_1d(det.get("scores_3d", []) or [], np.float32, N)
    labels = _safe_np_1d(det.get("labels_3d", []) or [], np.int64, N)

    # score filter
    if N > 0 and cfg.raw_score_min > 0.0:
        m = scores >= float(cfg.raw_score_min)
        boxes, scores, labels = boxes[m], scores[m], labels[m]

    # ignore class filter (optional for candidates)
    if cfg.ignore_classes:
        ig = set(cfg.ignore_classes)
        if boxes.shape[0] > 0:
            m = np.array([int(l) not in ig for l in labels.tolist()], dtype=bool)
            boxes, scores, labels = boxes[m], scores[m], labels[m]

    # topK by score
    if cfg.max_per_frame is not None and boxes.shape[0] > int(cfg.max_per_frame):
        k = int(cfg.max_per_frame)
        idx = np.argsort(-scores)[:k]
        boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

    return boxes, scores, labels


def parse_gt(sample: Dict[str, Any],
             ignore_classes: Optional[Set[int]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parse gt.{boxes_3d,labels_3d,attr_names} into numpy arrays / list.
    Applies ignore_classes if provided.
    """
    gt = sample.get("gt", {}) or {}
    boxes = _safe_np_boxes(gt.get("boxes_3d", []) or [], exp_dim=9)
    labels = np.asarray(gt.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)
    # align length if needed
    if labels.size != boxes.shape[0]:
        # best-effort: truncate to min
        n = min(labels.size, boxes.shape[0])
        boxes = boxes[:n]
        labels = labels[:n]

    attrs = gt.get("attr_names", None)
    if attrs is None:
        attrs_list = [""] * int(labels.size)
    else:
        attrs_list = [str(a) if a is not None else "" for a in list(attrs)]
        if len(attrs_list) != int(labels.size):
            n = min(len(attrs_list), int(labels.size))
            attrs_list = attrs_list[:n]
            boxes = boxes[:n]
            labels = labels[:n]

    # ignore class filter (GT side)
    if ignore_classes:
        ig = set(ignore_classes)
        if labels.size > 0:
            keep = np.array([int(l) not in ig for l in labels.tolist()], dtype=bool)
            boxes = boxes[keep]
            labels = labels[keep]
            attrs_list = [a for a, m in zip(attrs_list, keep.tolist()) if m]

    return boxes, labels, attrs_list


# -----------------------------
# Matching (class-aware greedy XY)
# -----------------------------

def _euclid2(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    if a_xy.size == 0 or b_xy.size == 0:
        return np.zeros((a_xy.shape[0], b_xy.shape[0]), dtype=np.float32)
    a = a_xy[:, None, :]  # (N,1,2)
    b = b_xy[None, :, :]  # (1,M,2)
    return np.linalg.norm(a - b, axis=2).astype(np.float32)


def class_aware_greedy_match_xy(
    det_xyz: np.ndarray, det_labels: np.ndarray,
    gt_xyz: np.ndarray, gt_labels: np.ndarray,
    gt_attrs: List[str],
    thr_xy: float,
) -> MatchTargets:
    """
    Greedy one-to-one matching within each class using XY distance threshold.
    For each det, decides keep(0/1), matched class target, matched attr target.
    """
    N = int(det_xyz.shape[0])
    keep = np.zeros((N,), dtype=np.int64)
    cls_tgt = np.full((N,), -100, dtype=np.int64)
    attr_tgt = [""] * N

    if N == 0:
        return MatchTargets(keep=keep, cls_tgt=cls_tgt, attr_tgt=attr_tgt, num_gt_valid=int(gt_xyz.shape[0]))
    if gt_xyz.size == 0:
        return MatchTargets(keep=keep, cls_tgt=cls_tgt, attr_tgt=attr_tgt, num_gt_valid=0)

    det_labels = det_labels.reshape(-1).astype(np.int64, copy=False)
    gt_labels = gt_labels.reshape(-1).astype(np.int64, copy=False)

    classes = set(det_labels.tolist()) | set(gt_labels.tolist())
    used_det: Set[int] = set()
    used_gt: Set[int] = set()

    for c in classes:
        di = np.where(det_labels == c)[0]
        gi = np.where(gt_labels == c)[0]
        if di.size == 0 or gi.size == 0:
            continue

        D = _euclid2(det_xyz[di, :2], gt_xyz[gi, :2])  # (|di|,|gi|)
        pairs: List[Tuple[float, int, int]] = []
        for ii in range(D.shape[0]):
            for jj in range(D.shape[1]):
                d = float(D[ii, jj])
                if d <= float(thr_xy):
                    pairs.append((d, int(di[ii]), int(gi[jj])))
        pairs.sort(key=lambda x: x[0])

        for d, i_det, j_gt in pairs:
            if i_det in used_det or j_gt in used_gt:
                continue
            used_det.add(i_det)
            used_gt.add(j_gt)
            keep[i_det] = 1
            cls_tgt[i_det] = int(gt_labels[j_gt])
            attr_tgt[i_det] = gt_attrs[j_gt] if (0 <= j_gt < len(gt_attrs)) else ""

    return MatchTargets(
        keep=keep,
        cls_tgt=cls_tgt,
        attr_tgt=attr_tgt,
        num_gt_valid=int(gt_xyz.shape[0]),
    )


def build_candidate_gt_links_xy(
    det_xyz: np.ndarray,
    det_labels: np.ndarray,
    gt_xyz: np.ndarray,
    gt_labels: np.ndarray,
    thr_xy: float,
    det_scores: Optional[np.ndarray] = None,
) -> CandidateGtLinks:
    """
    Build candidate<->gt links under class-aware XY threshold.

    Unlike class_aware_greedy_match_xy, this keeps dense feasibility information
    and independent best links from both directions:
      - candidate -> nearest feasible gt
      - gt -> best candidate (nearest, tie-break by higher score)
    """
    det_xyz = np.asarray(det_xyz, dtype=np.float32).reshape(-1, 3)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32).reshape(-1, 3)
    det_labels = np.asarray(det_labels, dtype=np.int64).reshape(-1)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)

    n_det = int(det_xyz.shape[0])
    n_gt = int(gt_xyz.shape[0])

    cand_to_gt = np.full((n_det,), -1, dtype=np.int64)
    cand_to_gt_dist = np.full((n_det,), np.inf, dtype=np.float32)
    gt_best_cand = np.full((n_gt,), -1, dtype=np.int64)
    cand_cover_counts = np.zeros((n_det,), dtype=np.int64)
    gt_candidate_counts = np.zeros((n_gt,), dtype=np.int64)

    if n_det == 0 or n_gt == 0:
        return CandidateGtLinks(
            cand_to_gt=cand_to_gt,
            cand_to_gt_dist=cand_to_gt_dist,
            gt_best_cand=gt_best_cand,
            cand_cover_counts=cand_cover_counts,
            gt_candidate_counts=gt_candidate_counts,
        )

    if det_scores is None:
        det_scores_f = np.zeros((n_det,), dtype=np.float32)
    else:
        det_scores_f = np.asarray(det_scores, dtype=np.float32).reshape(-1)
        if det_scores_f.shape[0] != n_det:
            m = min(det_scores_f.shape[0], n_det)
            tmp = np.zeros((n_det,), dtype=np.float32)
            tmp[:m] = det_scores_f[:m]
            det_scores_f = tmp

    thr = float(thr_xy)
    for cls in (set(det_labels.tolist()) | set(gt_labels.tolist())):
        di = np.where(det_labels == int(cls))[0]
        gi = np.where(gt_labels == int(cls))[0]
        if di.size == 0 or gi.size == 0:
            continue

        dmat = _euclid2(det_xyz[di, :2], gt_xyz[gi, :2])
        feasible = dmat <= thr
        if not np.any(feasible):
            continue

        # candidate -> nearest feasible gt
        for ii in range(dmat.shape[0]):
            row = dmat[ii]
            m = feasible[ii]
            if not np.any(m):
                continue
            local_j = int(np.argmin(np.where(m, row, np.inf)))
            d = float(row[local_j])
            det_idx = int(di[ii])
            gt_idx = int(gi[local_j])
            cand_to_gt[det_idx] = gt_idx
            cand_to_gt_dist[det_idx] = np.float32(d)
            cand_cover_counts[det_idx] = int(np.count_nonzero(m))

        # gt -> best candidate (nearest, tie-break by score desc)
        for jj in range(dmat.shape[1]):
            col = dmat[:, jj]
            m = feasible[:, jj]
            if not np.any(m):
                continue
            gt_candidate_counts[int(gi[jj])] = int(np.count_nonzero(m))
            cand_local = np.nonzero(m)[0].astype(np.int64)
            cand_dist = col[cand_local]
            cand_score = det_scores_f[di[cand_local]]
            order = np.lexsort((-cand_score, cand_dist))
            best_local = int(cand_local[order[0]])
            gt_best_cand[int(gi[jj])] = int(di[best_local])

    return CandidateGtLinks(
        cand_to_gt=cand_to_gt,
        cand_to_gt_dist=cand_to_gt_dist,
        gt_best_cand=gt_best_cand,
        cand_cover_counts=cand_cover_counts,
        gt_candidate_counts=gt_candidate_counts,
    )


# -----------------------------
# Candidate building
# -----------------------------

def _boxes_to_candidates(
    sample: Dict[str, Any],
    T_l2g: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    source: str,
    from_dt: int,
) -> List[Candidate]:
    """
    Convert arrays into Candidate list and cache global center.
    """
    out: List[Candidate] = []
    if boxes.size == 0:
        return out

    centers_l = boxes[:, :3].astype(np.float64, copy=False)
    centers_g = transform_points(T_l2g, centers_l)  # (N,3), float64

    for i in range(boxes.shape[0]):
        out.append(Candidate(
            box=boxes[i].astype(np.float32, copy=False),
            score=float(scores[i]),
            label=int(labels[i]),
            source=source,
            from_dt=int(from_dt),
            center_global=centers_g[i].astype(np.float64, copy=False),
        ))
    return out


def build_candidates_for_scene(
    scene_frames: List[Dict[str, Any]],
    cfg: CandidateConfig,
) -> List[FrameData]:
    """
    Build FrameData list for a scene (frames already sorted by timestamp).

    Candidates:
      - raw det parsed from each frame
      - optional: warp candidates from neighboring frames into current frame

    Notes:
      - For warp: we transform neighbor boxes into the current frame lidar coordinate.
      - We cache each candidate's *global center* for later scene-level linking.
    """
    ignore = set(cfg.ignore_classes or [])

    # precompute per-frame raw det + pose + gt
    raw_boxes: List[np.ndarray] = []
    raw_scores: List[np.ndarray] = []
    raw_labels: List[np.ndarray] = []
    T_l2g_list: List[np.ndarray] = []

    gt_boxes_list: List[np.ndarray] = []
    gt_labels_list: List[np.ndarray] = []
    gt_attrs_list: List[List[str]] = []

    for s in scene_frames:
        T_l2g = T_l2g_from_sample(s)
        T_l2g_list.append(T_l2g)

        b, sc, lab = parse_det(s, cfg)
        raw_boxes.append(b)
        raw_scores.append(sc)
        raw_labels.append(lab)

        gb, gl, ga = parse_gt(s, ignore_classes=ignore)
        gt_boxes_list.append(gb)
        gt_labels_list.append(gl)
        gt_attrs_list.append(ga)

    frames_out: List[FrameData] = []

    T = len(scene_frames)
    for i in range(T):
        sample = scene_frames[i]
        scene_tok = str(sample.get("scene_token", ""))
        ts = int(sample.get("_ts", sample.get("timestamp", sample.get("timestamp_us", 0))))

        candidates: List[Candidate] = []
        # raw
        candidates.extend(_boxes_to_candidates(
            sample=sample,
            T_l2g=T_l2g_list[i],
            boxes=raw_boxes[i],
            scores=raw_scores[i],
            labels=raw_labels[i],
            source="raw",
            from_dt=0,
        ))

        # warp
        if cfg.use_warp and cfg.warp_radius > 0:
            r = int(cfg.warp_radius)
            for dt in range(-r, r + 1):
                if dt == 0:
                    continue
                j = i + dt
                if j < 0 or j >= T:
                    continue

                # take neighbor topK before warping (warp_topk)
                nb = raw_boxes[j]
                ns = raw_scores[j]
                nl = raw_labels[j]
                if nb.shape[0] == 0:
                    continue

                if cfg.warp_topk is not None and nb.shape[0] > int(cfg.warp_topk):
                    kk = int(cfg.warp_topk)
                    idx = np.argsort(-ns)[:kk]
                    nb2, ns2, nl2 = nb[idx], ns[idx], nl[idx]
                else:
                    nb2, ns2, nl2 = nb, ns, nl

                # transform neighbor boxes into current frame lidar coordinate
                T_j2i = T_k2ref(scene_frames[i], scene_frames[j])  # lidar_j -> lidar_i
                nb_w = transform_boxes_center(T_j2i, nb2, rotate_yaw_flag=True, rotate_vel_flag=True).astype(np.float32)

                # optional decay
                ns_w = ns2.astype(np.float32, copy=False) * float(cfg.warp_score_decay)

                candidates.extend(_boxes_to_candidates(
                    sample=sample,
                    T_l2g=T_l2g_list[i],
                    boxes=nb_w,
                    scores=ns_w,
                    labels=nl2,
                    source=f"warp(dt={dt:+d})",
                    from_dt=dt,
                ))

        frames_out.append(FrameData(
            sample=sample,
            scene_token=scene_tok,
            index_in_scene=i,
            timestamp=ts,
            T_l2g=T_l2g_list[i],
            candidates=candidates,
            gt_boxes=gt_boxes_list[i],
            gt_labels=gt_labels_list[i],
            gt_attrs=gt_attrs_list[i],
        ))

    return frames_out


# -----------------------------
# Targets for training/eval
# -----------------------------

def match_raw_det_to_gt(
    frame: FrameData,
    cfg: CandidateConfig,
) -> MatchTargets:
    """
    Generate targets by matching *raw dets only* (source == "raw") to GT.

    Why raw-only by default:
      - It matches your earlier transformer setup and avoids "training on self-generated candidates".
      - You can still compute oracle coverage by matching ALL candidates if needed.
    """
    # collect raw dets from candidates
    raw = [c for c in frame.candidates if c.from_dt == 0 and c.source == "raw"]
    if len(raw) == 0:
        # no dets => all GT are FN (num_gt_valid)
        return MatchTargets(
            keep=np.zeros((0,), dtype=np.int64),
            cls_tgt=np.zeros((0,), dtype=np.int64),
            attr_tgt=[],
            num_gt_valid=int(frame.gt_boxes.shape[0]),
        )

    det_boxes = np.stack([c.box for c in raw], axis=0).astype(np.float32, copy=False)
    det_xyz = det_boxes[:, :3].astype(np.float32, copy=False)
    det_labels = np.array([c.label for c in raw], dtype=np.int64)

    gt_xyz = frame.gt_boxes[:, :3].astype(np.float32, copy=False)
    gt_labels = frame.gt_labels.astype(np.int64, copy=False)
    gt_attrs = frame.gt_attrs

    return class_aware_greedy_match_xy(
        det_xyz=det_xyz,
        det_labels=det_labels,
        gt_xyz=gt_xyz,
        gt_labels=gt_labels,
        gt_attrs=gt_attrs,
        thr_xy=float(cfg.match_thr_xy),
    )


def match_all_candidates_to_gt(
    frame: FrameData,
    cfg: CandidateConfig,
) -> MatchTargets:
    """
    Analysis helper: match ALL candidates to GT (useful to measure "oracle recall upper bound").
    """
    if len(frame.candidates) == 0:
        return MatchTargets(
            keep=np.zeros((0,), dtype=np.int64),
            cls_tgt=np.zeros((0,), dtype=np.int64),
            attr_tgt=[],
            num_gt_valid=int(frame.gt_boxes.shape[0]),
        )
    det_boxes = np.stack([c.box for c in frame.candidates], axis=0).astype(np.float32, copy=False)
    det_xyz = det_boxes[:, :3].astype(np.float32, copy=False)
    det_labels = np.array([c.label for c in frame.candidates], dtype=np.int64)

    gt_xyz = frame.gt_boxes[:, :3].astype(np.float32, copy=False)
    gt_labels = frame.gt_labels.astype(np.int64, copy=False)
    gt_attrs = frame.gt_attrs

    return class_aware_greedy_match_xy(
        det_xyz=det_xyz,
        det_labels=det_labels,
        gt_xyz=gt_xyz,
        gt_labels=gt_labels,
        gt_attrs=gt_attrs,
        thr_xy=float(cfg.match_thr_xy),
    )


# -----------------------------
# Convenience: build a whole dataset (all scenes)
# -----------------------------

def build_frames_by_scene(
    scenes: Dict[str, List[Dict[str, Any]]],
    cfg: CandidateConfig,
) -> Dict[str, List[FrameData]]:
    """
    Given `scenes` as {scene_token: [sample_dict,...sorted...]},
    return {scene_token: [FrameData,...]} with candidates built.

    This is the common entry for train/eval/infer.
    """
    out: Dict[str, List[FrameData]] = {}
    for sc, frames in scenes.items():
        out[sc] = build_candidates_for_scene(frames, cfg)
    return out
