# metrics.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics / evaluation utilities for detection refinement.

This file focuses on:
- Object-level Precision / Recall / F1 using class-aware greedy matching (XY distance threshold).
- Attribute evaluation on true positives (TPs), optionally with a mapping from attr_name -> attr_id.

Two evaluation entry points:
1) eval_json_root(root, det_field="det", match_thr_xy=2.0, ignore_classes={-1})
   - walks the JSON, finds samples, evaluates per-frame and aggregates.
2) eval_frames_by_scene(frames_by_scene, det_source="raw"|"all"|"dt0", match_thr_xy=2.0, ...)
   - evaluates from FrameData structures (built by data.py)

Notes:
- We DO NOT evaluate box size/yaw/IoU. Only center XY distance + class-aware one-to-one match,
  consistent with your current pipeline.
- This file is numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

# local modules
from json_io import iter_samples  # you renamed io.py -> json_io.py
from data import _safe_np_boxes  # internal helper for box parsing


# -----------------------------
# Attribute helpers
# -----------------------------

def default_attr_vocab() -> Dict[str, int]:
    """
    A minimal attr vocabulary aligned with your mention: moving / standing / parked.
    You can extend this as needed.
    """
    return {
        "": -1,
        "unknown": -1,
        "moving": 0,
        "standing": 1,
        "parked": 2,
        # nuScenes-like strings (optional mapping):
        "vehicle.moving": 0,
        "vehicle.parked": 2,
        "vehicle.stopped": 1,
        "pedestrian.moving": 0,
        "pedestrian.standing": 1,
        "pedestrian.sitting_lying_down": 1,
        "cycle.with_rider": 0,
        "cycle.without_rider": 2,
    }


def normalize_attr_name(a: str) -> str:
    if a is None:
        return ""
    s = str(a).strip()
    if s == "":
        return ""
    return s


def map_attr_to_id(attr: str, vocab: Dict[str, int]) -> int:
    """
    Map attribute string to an integer id. Unknown -> -1.
    Tries exact match first, then heuristic match on substrings for robustness.
    """
    a = normalize_attr_name(attr)
    if a in vocab:
        return int(vocab[a])

    # heuristic substring mapping
    low = a.lower()
    if "moving" in low or "with_rider" in low:
        return int(vocab.get("moving", 0))
    if "parked" in low or "without_rider" in low:
        return int(vocab.get("parked", 2))
    if "standing" in low or "stopped" in low or "sitting" in low:
        return int(vocab.get("standing", 1))
    return -1


# -----------------------------
# Matching utilities
# -----------------------------

def euclid2(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    if a_xy.size == 0 or b_xy.size == 0:
        return np.zeros((a_xy.shape[0], b_xy.shape[0]), dtype=np.float32)
    a = a_xy[:, None, :]  # (N,1,2)
    b = b_xy[None, :, :]  # (1,M,2)
    return np.linalg.norm(a - b, axis=2).astype(np.float32)


@dataclass
class MatchResult:
    # per-det
    det_keep: np.ndarray        # (N,) 0/1; 1 means matched to some GT
    det_to_gt: np.ndarray       # (N,) gt index if matched else -1

    # scalars
    tp: int
    fp: int
    fn: int
    num_gt: int
    num_det: int


def class_aware_greedy_match(
    det_xyz: np.ndarray,
    det_labels: np.ndarray,
    gt_xyz: np.ndarray,
    gt_labels: np.ndarray,
    thr_xy: float,
) -> MatchResult:
    """
    Class-aware one-to-one greedy matching using XY distance threshold.

    Returns:
      - det_keep: 1 if det matched to a GT of same class within thr.
      - det_to_gt: GT index for each det, else -1.
      - tp/fp/fn counts in GT-view sense:
          TP = matched det count
          FP = det count - TP
          FN = GT count - TP
    """
    det_xyz = np.asarray(det_xyz, dtype=np.float32).reshape(-1, 3)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32).reshape(-1, 3)
    det_labels = np.asarray(det_labels, dtype=np.int64).reshape(-1)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)

    N = int(det_xyz.shape[0])
    M = int(gt_xyz.shape[0])
    det_keep = np.zeros((N,), dtype=np.int64)
    det_to_gt = np.full((N,), -1, dtype=np.int64)

    if N == 0 and M == 0:
        return MatchResult(det_keep, det_to_gt, tp=0, fp=0, fn=0, num_gt=0, num_det=0)
    if N == 0:
        return MatchResult(det_keep, det_to_gt, tp=0, fp=0, fn=M, num_gt=M, num_det=0)
    if M == 0:
        return MatchResult(det_keep, det_to_gt, tp=0, fp=N, fn=0, num_gt=0, num_det=N)

    used_det: Set[int] = set()
    used_gt: Set[int] = set()

    classes = set(det_labels.tolist()) | set(gt_labels.tolist())
    thr = float(thr_xy)

    for c in classes:
        di = np.where(det_labels == c)[0]
        gi = np.where(gt_labels == c)[0]
        if di.size == 0 or gi.size == 0:
            continue

        D = euclid2(det_xyz[di, :2], gt_xyz[gi, :2])  # (|di|,|gi|)
        pairs: List[Tuple[float, int, int]] = []
        for ii in range(D.shape[0]):
            for jj in range(D.shape[1]):
                d = float(D[ii, jj])
                if d <= thr:
                    pairs.append((d, int(di[ii]), int(gi[jj])))
        pairs.sort(key=lambda x: x[0])

        for d, i_det, j_gt in pairs:
            if i_det in used_det or j_gt in used_gt:
                continue
            used_det.add(i_det)
            used_gt.add(j_gt)
            det_keep[i_det] = 1
            det_to_gt[i_det] = j_gt

    tp = int(det_keep.sum())
    fp = int(N - tp)
    fn = int(M - tp)
    return MatchResult(det_keep, det_to_gt, tp=tp, fp=fp, fn=fn, num_gt=M, num_det=N)


# -----------------------------
# Parsing det/gt from JSON samples
# -----------------------------

def parse_det_from_sample(sample: Dict[str, Any], det_field: str = "det") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    det = sample.get(det_field, {}) or {}
    boxes = _safe_np_boxes(det.get("boxes_3d", []) or [], exp_dim=9)  # (N,9)
    N = boxes.shape[0]
    scores = det.get("scores_3d", []) or []
    labels = det.get("labels_3d", []) or []
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    # align lengths robustly
    if scores.size != N:
        if scores.size == 1 and N > 1:
            scores = np.full((N,), float(scores.item()), dtype=np.float32)
        else:
            n = min(scores.size, N)
            boxes = boxes[:n]
            scores = scores[:n]
            labels = labels[:n] if labels.size >= n else np.resize(labels, (n,))
            N = n
    if labels.size != N:
        if labels.size == 1 and N > 1:
            labels = np.full((N,), int(labels.item()), dtype=np.int64)
        else:
            n = min(labels.size, N)
            boxes = boxes[:n]
            scores = scores[:n]
            labels = labels[:n]
            N = n

    return boxes, scores, labels


def parse_gt_from_sample(sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    gt = sample.get("gt", {}) or {}
    boxes = _safe_np_boxes(gt.get("boxes_3d", []) or [], exp_dim=9)
    labels = np.asarray(gt.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)

    n = min(int(boxes.shape[0]), int(labels.size))
    boxes = boxes[:n]
    labels = labels[:n]

    attrs = gt.get("attr_names", None)
    if attrs is None:
        attr_list = [""] * n
    else:
        attr_list = [normalize_attr_name(a) for a in list(attrs)]
        if len(attr_list) != n:
            attr_list = attr_list[:n]
    return boxes, labels, attr_list


def apply_ignore_classes(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray],
    attrs: Optional[List[str]],
    ignore: Set[int],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    if labels.size == 0 or not ignore:
        return boxes, labels, scores, attrs
    keep = np.array([int(l) not in ignore for l in labels.tolist()], dtype=bool)
    boxes2 = boxes[keep]
    labels2 = labels[keep]
    scores2 = (scores[keep] if scores is not None else None)
    attrs2 = ([a for a, m in zip(attrs, keep.tolist()) if m] if attrs is not None else None)
    return boxes2, labels2, scores2, attrs2


# -----------------------------
# Aggregate metrics
# -----------------------------

@dataclass
class DetMetrics:
    # main
    P: float
    R: float
    F1: float

    # counts
    tp: int
    fp: int
    fn: int
    num_frames: int
    num_det: int
    num_gt: int

    # attr (TP only)
    attr_acc: float
    attr_macro_f1: float
    attr_num_tp_with_attr: int

    # optional per-class can be added later


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _macro_f1_from_confmat(cm: np.ndarray) -> float:
    """
    Compute macro-F1 from confusion matrix cm (C,C), where cm[i,j] is count of gt=i, pred=j.
    Ignores class id -1 (unknown) by assuming cm already excludes it.
    """
    C = cm.shape[0]
    f1s = []
    for i in range(C):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if len(f1s) > 0 else 0.0


def evaluate_samples(
    samples: List[Dict[str, Any]],
    *,
    det_field: str = "det",
    match_thr_xy: float = 2.0,
    ignore_classes: Optional[Set[int]] = None,
    attr_vocab: Optional[Dict[str, int]] = None,
    pred_attr_field: Optional[str] = None,
) -> DetMetrics:
    """
    Evaluate a list of sample dicts.

    Args:
      det_field: which det field to evaluate ("det" or "det_refined")
      pred_attr_field:
        - if None: we do NOT evaluate predicted attrs (attr metrics will be 0)
        - else: we expect sample[det_field][pred_attr_field] is a list[str] aligned with det boxes,
                e.g., pred_attr_field="attrs"

    Attribute evaluation:
      - Only on TPs (where det matched GT).
      - If either GT attr or pred attr is unknown(-1), we skip that TP for attr metrics.
    """
    ignore = set(ignore_classes or [])
    vocab = attr_vocab or default_attr_vocab()

    tot_tp = tot_fp = tot_fn = 0
    tot_det = tot_gt = 0
    num_frames = 0

    # attr accumulators (TP only)
    attr_correct = 0
    attr_total = 0
    # confusion matrix for known attr ids 0..K-1
    known_ids = sorted({v for v in vocab.values() if v >= 0})
    # map known ids to 0..C-1 index
    id_to_idx = {aid: i for i, aid in enumerate(known_ids)}
    C = len(known_ids)
    cm = np.zeros((C, C), dtype=np.int64)

    for s in samples:
        num_frames += 1

        det_boxes, det_scores, det_labels = parse_det_from_sample(s, det_field=det_field)
        gt_boxes, gt_labels, gt_attrs = parse_gt_from_sample(s)

        # ignore classes on both sides
        det_boxes, det_labels, det_scores, _ = apply_ignore_classes(
            det_boxes, det_labels, det_scores, None, ignore
        )
        gt_boxes, gt_labels, _, gt_attrs2 = apply_ignore_classes(
            gt_boxes, gt_labels, None, gt_attrs, ignore
        )
        gt_attrs = gt_attrs2 if gt_attrs2 is not None else [""] * int(gt_labels.size)

        det_xyz = det_boxes[:, :3]
        gt_xyz = gt_boxes[:, :3]

        mr = class_aware_greedy_match(
            det_xyz=det_xyz,
            det_labels=det_labels,
            gt_xyz=gt_xyz,
            gt_labels=gt_labels,
            thr_xy=match_thr_xy,
        )

        tot_tp += mr.tp
        tot_fp += mr.fp
        tot_fn += mr.fn
        tot_det += mr.num_det
        tot_gt += mr.num_gt

        # attribute evaluation (TP only)
        if pred_attr_field is not None and mr.tp > 0:
            det_block = s.get(det_field, {}) or {}
            pred_attrs = det_block.get(pred_attr_field, None)
            if pred_attrs is None:
                continue
            pred_attrs = [normalize_attr_name(a) for a in list(pred_attrs)]
            # align length
            n_det = int(det_labels.size)
            if len(pred_attrs) != n_det:
                pred_attrs = pred_attrs[:n_det]

            for i_det in range(n_det):
                j_gt = int(mr.det_to_gt[i_det])
                if j_gt < 0:
                    continue  # not TP

                gt_a = gt_attrs[j_gt] if 0 <= j_gt < len(gt_attrs) else ""
                pr_a = pred_attrs[i_det] if 0 <= i_det < len(pred_attrs) else ""

                gt_id = map_attr_to_id(gt_a, vocab)
                pr_id = map_attr_to_id(pr_a, vocab)

                if gt_id < 0 or pr_id < 0:
                    # skip unknowns for attr metrics
                    continue

                attr_total += 1
                if gt_id == pr_id:
                    attr_correct += 1

                if gt_id in id_to_idx and pr_id in id_to_idx:
                    cm[id_to_idx[gt_id], id_to_idx[pr_id]] += 1

    P = _safe_div(tot_tp, tot_tp + tot_fp)
    R = _safe_div(tot_tp, tot_tp + tot_fn)
    F1 = _safe_div(2 * P * R, P + R) if (P + R) > 0 else 0.0

    attr_acc = _safe_div(attr_correct, attr_total)
    attr_macro_f1 = _macro_f1_from_confmat(cm) if attr_total > 0 else 0.0

    return DetMetrics(
        P=float(P),
        R=float(R),
        F1=float(F1),
        tp=int(tot_tp),
        fp=int(tot_fp),
        fn=int(tot_fn),
        num_frames=int(num_frames),
        num_det=int(tot_det),
        num_gt=int(tot_gt),
        attr_acc=float(attr_acc),
        attr_macro_f1=float(attr_macro_f1),
        attr_num_tp_with_attr=int(attr_total),
    )


def eval_json_root(
    root: Any,
    *,
    det_field: str = "det",
    match_thr_xy: float = 2.0,
    ignore_classes: Optional[Set[int]] = None,
    require_gt: bool = True,
    pred_attr_field: Optional[str] = None,
    attr_vocab: Optional[Dict[str, int]] = None,
) -> DetMetrics:
    """
    Evaluate from a JSON root object (already json.load-ed).
    """
    samples = list(iter_samples(root, require_det=True, require_gt=require_gt))
    # filter scenes/timestamps if needed (not required for metrics)
    ignore = set(ignore_classes or [])
    return evaluate_samples(
        samples,
        det_field=det_field,
        match_thr_xy=match_thr_xy,
        ignore_classes=ignore,
        pred_attr_field=pred_attr_field,
        attr_vocab=attr_vocab,
    )


def eval_json_path(
    json_path: str,
    *,
    det_field: str = "det",
    match_thr_xy: float = 2.0,
    ignore_classes: Optional[Set[int]] = None,
    require_gt: bool = True,
    pred_attr_field: Optional[str] = None,
    attr_vocab: Optional[Dict[str, int]] = None,
) -> DetMetrics:
    """
    Convenience wrapper: load json then eval.
    """
    import json
    with open(json_path, "r") as f:
        root = json.load(f)
    return eval_json_root(
        root,
        det_field=det_field,
        match_thr_xy=match_thr_xy,
        ignore_classes=ignore_classes,
        require_gt=require_gt,
        pred_attr_field=pred_attr_field,
        attr_vocab=attr_vocab,
    )


# -----------------------------
# FrameData evaluation (optional)
# -----------------------------

def eval_frames_by_scene(
    frames_by_scene: Dict[str, List[Any]],
    *,
    match_thr_xy: float = 2.0,
    ignore_classes: Optional[Set[int]] = None,
    det_source: str = "raw",
    pred_attr_from_candidate: bool = False,
    attr_vocab: Optional[Dict[str, int]] = None,
) -> DetMetrics:
    """
    Evaluate using FrameData (from data.py).

    det_source options:
      - "raw": only candidates with source=="raw" and from_dt==0
      - "all": all candidates in frame
      - "dt0": any candidate with from_dt==0 (if you have multiple sources)

    pred_attr_from_candidate:
      - if True: expects each Candidate has an attribute string stored in a custom field
                (not in current Candidate dataclass). This is left False by default.

    This is mainly for analysis / oracle coverage measurements.
    """
    ignore = set(ignore_classes or [])
    vocab = attr_vocab or default_attr_vocab()

    tot_tp = tot_fp = tot_fn = 0
    tot_det = tot_gt = 0
    num_frames = 0

    attr_correct = 0
    attr_total = 0
    known_ids = sorted({v for v in vocab.values() if v >= 0})
    id_to_idx = {aid: i for i, aid in enumerate(known_ids)}
    C = len(known_ids)
    cm = np.zeros((C, C), dtype=np.int64)

    for sc, frames in frames_by_scene.items():
        for fr in frames:
            num_frames += 1

            # det selection
            cands = fr.candidates
            if det_source == "raw":
                cands = [c for c in cands if c.source == "raw" and c.from_dt == 0]
            elif det_source == "dt0":
                cands = [c for c in cands if c.from_dt == 0]
            elif det_source == "all":
                pass
            else:
                raise ValueError(f"Unknown det_source: {det_source}")

            if len(cands) == 0:
                # all GT are FN
                num_gt = int(fr.gt_boxes.shape[0])
                tot_fn += num_gt
                tot_gt += num_gt
                continue

            det_boxes = np.stack([c.box for c in cands], axis=0).astype(np.float32)
            det_labels = np.array([c.label for c in cands], dtype=np.int64)

            # ignore det classes
            if ignore:
                keep = np.array([int(l) not in ignore for l in det_labels.tolist()], dtype=bool)
                det_boxes = det_boxes[keep]
                det_labels = det_labels[keep]
                cands = [c for c, m in zip(cands, keep.tolist()) if m]

            gt_boxes = fr.gt_boxes.astype(np.float32, copy=False)
            gt_labels = fr.gt_labels.astype(np.int64, copy=False)
            gt_attrs = list(fr.gt_attrs)

            # ignore gt classes (already applied in data.parse_gt usually, but keep safe)
            if ignore and gt_labels.size > 0:
                keepg = np.array([int(l) not in ignore for l in gt_labels.tolist()], dtype=bool)
                gt_boxes = gt_boxes[keepg]
                gt_labels = gt_labels[keepg]
                gt_attrs = [a for a, m in zip(gt_attrs, keepg.tolist()) if m]

            mr = class_aware_greedy_match(
                det_xyz=det_boxes[:, :3],
                det_labels=det_labels,
                gt_xyz=gt_boxes[:, :3],
                gt_labels=gt_labels,
                thr_xy=match_thr_xy,
            )

            tot_tp += mr.tp
            tot_fp += mr.fp
            tot_fn += mr.fn
            tot_det += mr.num_det
            tot_gt += mr.num_gt

            # attribute evaluation (optional)
            if pred_attr_from_candidate and mr.tp > 0:
                # If later you add Candidate.attr, you can plug it here.
                # For now, skip.
                pass

    P = _safe_div(tot_tp, tot_tp + tot_fp)
    R = _safe_div(tot_tp, tot_tp + tot_fn)
    F1 = _safe_div(2 * P * R, P + R) if (P + R) > 0 else 0.0

    attr_acc = _safe_div(attr_correct, attr_total)
    attr_macro_f1 = _macro_f1_from_confmat(cm) if attr_total > 0 else 0.0

    return DetMetrics(
        P=float(P),
        R=float(R),
        F1=float(F1),
        tp=int(tot_tp),
        fp=int(tot_fp),
        fn=int(tot_fn),
        num_frames=int(num_frames),
        num_det=int(tot_det),
        num_gt=int(tot_gt),
        attr_acc=float(attr_acc),
        attr_macro_f1=float(attr_macro_f1),
        attr_num_tp_with_attr=int(attr_total),
    )


# -----------------------------
# Pretty print helper
# -----------------------------

def format_metrics(m: DetMetrics) -> str:
    return (
        f"P={m.P:.4f} R={m.R:.4f} F1={m.F1:.4f} | "
        f"tp={m.tp} fp={m.fp} fn={m.fn} | "
        f"frames={m.num_frames} det={m.num_det} gt={m.num_gt} | "
        f"attr_acc(TP-only)={m.attr_acc:.4f} attr_macroF1={m.attr_macro_f1:.4f} "
        f"(tp_with_attr={m.attr_num_tp_with_attr})"
    )


def _bucket_index(v: float, bins: Sequence[float]) -> int:
    x = float(v)
    for i, b in enumerate(bins):
        if x < float(b):
            return int(i)
    return int(len(bins))


def _bucket_name(i: int, bins: Sequence[float]) -> str:
    if i < len(bins):
        if i == 0:
            return f"[0,{float(bins[i]):.2f})"
        return f"[{float(bins[i-1]):.2f},{float(bins[i]):.2f})"
    if len(bins) == 0:
        return "[0,inf)"
    return f"[{float(bins[-1]):.2f},inf)"


def _safe_metric_dict(tp: int, fp: int, fn: int) -> Dict[str, float]:
    P = _safe_div(tp, tp + fp)
    R = _safe_div(tp, tp + fn)
    F1 = _safe_div(2 * P * R, P + R) if (P + R) > 0 else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "P": float(P), "R": float(R), "F1": float(F1)}


def evaluate_samples_detailed(
    samples: List[Dict[str, Any]],
    *,
    det_field: str = "det",
    match_thr_xy: float = 2.0,
    ignore_classes: Optional[Set[int]] = None,
    attr_vocab: Optional[Dict[str, int]] = None,
    pred_attr_field: Optional[str] = None,
    distance_bins: Optional[Sequence[float]] = None,
    speed_bins: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Extended evaluation with per-class and distance/speed bucket stats.
    """
    dist_bins = list(distance_bins) if distance_bins is not None else [20.0, 40.0, 60.0]
    speed_bins = list(speed_bins) if speed_bins is not None else [0.2, 2.0, 6.0]
    dist_bins = sorted([float(x) for x in dist_bins])
    speed_bins = sorted([float(x) for x in speed_bins])

    m = evaluate_samples(
        samples,
        det_field=det_field,
        match_thr_xy=match_thr_xy,
        ignore_classes=ignore_classes,
        attr_vocab=attr_vocab,
        pred_attr_field=pred_attr_field,
    )

    ignore = set(ignore_classes or [])
    cls_cnt: Dict[int, Dict[str, int]] = {}
    dist_cnt: List[Dict[str, int]] = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(len(dist_bins) + 1)]
    speed_cnt: List[Dict[str, int]] = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(len(speed_bins) + 1)]

    for s in samples:
        det_boxes, det_scores, det_labels = parse_det_from_sample(s, det_field=det_field)
        gt_boxes, gt_labels, _ = parse_gt_from_sample(s)
        det_boxes, det_labels, det_scores, _ = apply_ignore_classes(det_boxes, det_labels, det_scores, None, ignore)
        gt_boxes, gt_labels, _, _ = apply_ignore_classes(gt_boxes, gt_labels, None, None, ignore)

        mr = class_aware_greedy_match(
            det_xyz=det_boxes[:, :3],
            det_labels=det_labels,
            gt_xyz=gt_boxes[:, :3],
            gt_labels=gt_labels,
            thr_xy=match_thr_xy,
        )

        matched_gt = set([int(j) for j in mr.det_to_gt.tolist() if int(j) >= 0])
        for i_det, j_gt in enumerate(mr.det_to_gt.tolist()):
            lab_det = int(det_labels[i_det]) if i_det < det_labels.shape[0] else -1
            if lab_det not in cls_cnt:
                cls_cnt[lab_det] = {"tp": 0, "fp": 0, "fn": 0}
            if int(j_gt) >= 0:
                lab_gt = int(gt_labels[int(j_gt)]) if int(j_gt) < gt_labels.shape[0] else lab_det
                if lab_gt not in cls_cnt:
                    cls_cnt[lab_gt] = {"tp": 0, "fp": 0, "fn": 0}
                cls_cnt[lab_gt]["tp"] += 1

                rg = float(np.hypot(float(gt_boxes[int(j_gt), 0]), float(gt_boxes[int(j_gt), 1])))
                bg = _bucket_index(rg, dist_bins)
                dist_cnt[bg]["tp"] += 1

                vg = 0.0
                if gt_boxes.shape[1] >= 9:
                    vg = float(np.hypot(float(gt_boxes[int(j_gt), 7]), float(gt_boxes[int(j_gt), 8])))
                sg = _bucket_index(vg, speed_bins)
                speed_cnt[sg]["tp"] += 1
            else:
                cls_cnt[lab_det]["fp"] += 1
                rd = float(np.hypot(float(det_boxes[i_det, 0]), float(det_boxes[i_det, 1])))
                bd = _bucket_index(rd, dist_bins)
                dist_cnt[bd]["fp"] += 1
                vd = 0.0
                if det_boxes.shape[1] >= 9:
                    vd = float(np.hypot(float(det_boxes[i_det, 7]), float(det_boxes[i_det, 8])))
                sd = _bucket_index(vd, speed_bins)
                speed_cnt[sd]["fp"] += 1

        for j_gt in range(gt_labels.shape[0]):
            if int(j_gt) in matched_gt:
                continue
            lab_gt = int(gt_labels[j_gt])
            if lab_gt not in cls_cnt:
                cls_cnt[lab_gt] = {"tp": 0, "fp": 0, "fn": 0}
            cls_cnt[lab_gt]["fn"] += 1

            rg = float(np.hypot(float(gt_boxes[j_gt, 0]), float(gt_boxes[j_gt, 1])))
            bg = _bucket_index(rg, dist_bins)
            dist_cnt[bg]["fn"] += 1

            vg = 0.0
            if gt_boxes.shape[1] >= 9:
                vg = float(np.hypot(float(gt_boxes[j_gt, 7]), float(gt_boxes[j_gt, 8])))
            sg = _bucket_index(vg, speed_bins)
            speed_cnt[sg]["fn"] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for c in sorted(cls_cnt.keys()):
        cc = cls_cnt[c]
        per_class[str(int(c))] = _safe_metric_dict(cc["tp"], cc["fp"], cc["fn"])

    by_distance: Dict[str, Dict[str, float]] = {}
    for i, cc in enumerate(dist_cnt):
        by_distance[_bucket_name(i, dist_bins)] = _safe_metric_dict(cc["tp"], cc["fp"], cc["fn"])

    by_speed: Dict[str, Dict[str, float]] = {}
    for i, cc in enumerate(speed_cnt):
        by_speed[_bucket_name(i, speed_bins)] = _safe_metric_dict(cc["tp"], cc["fp"], cc["fn"])

    return {
        "overall": {
            "P": float(m.P),
            "R": float(m.R),
            "F1": float(m.F1),
            "tp": int(m.tp),
            "fp": int(m.fp),
            "fn": int(m.fn),
            "num_frames": int(m.num_frames),
            "num_det": int(m.num_det),
            "num_gt": int(m.num_gt),
            "attr_acc": float(m.attr_acc),
            "attr_macro_f1": float(m.attr_macro_f1),
            "attr_num_tp_with_attr": int(m.attr_num_tp_with_attr),
        },
        "per_class": per_class,
        "by_distance": by_distance,
        "by_speed": by_speed,
        "distance_bins": [float(x) for x in dist_bins],
        "speed_bins": [float(x) for x in speed_bins],
    }
