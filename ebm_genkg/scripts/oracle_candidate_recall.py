#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute oracle recall upper bound for a candidate set (raw / raw+warp) vs GT.

Oracle recall asks:
For each GT box in a frame, does there exist at least one candidate box
within XY distance threshold `match_thr` ?

We report two variants:
  (A) strict-label oracle: candidate label must equal GT label
  (B) any-label oracle: ignore candidate label (upper bound if relabeling were perfect)

Usage examples:

# raw only
python scripts/oracle_candidate_recall.py \
  --json /path/to/val.json \
  --use_sources raw \
  --match_thr 2.0

# raw + warp candidates (same config as your infer script)
python scripts/oracle_candidate_recall.py \
  --json /path/to/val.json \
  --use_warp \
  --warp_radius 2 \
  --warp_topk 200 \
  --warp_decay 0.9 \
  --use_sources all \
  --match_thr 2.0

Optional:
  --ignore_classes -1,10,11
  --topk_classes 15
  --limit_scenes 5   (quick debug)
"""

import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np

# Make repo root importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from json_io import load_root_and_samples, group_by_scene
from data import CandidateConfig, build_frames_by_scene


def parse_ignore(s: str) -> Set[int]:
    out = set()
    if s is None:
        return out
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.add(int(x))
    return out


def is_warp_candidate(c) -> bool:
    # Compatible with Candidate(source, from_dt)
    try:
        if getattr(c, "from_dt", 0) != 0:
            return True
    except Exception:
        pass
    try:
        src = str(getattr(c, "source", ""))
        if src.startswith("warp("):
            return True
    except Exception:
        pass
    return False


def select_candidates(frame, use_sources: str):
    """
    use_sources:
      - "raw": only raw dt=0
      - "all": all candidates
    """
    cands = frame.candidates
    if use_sources == "all":
        return cands
    if use_sources == "raw":
        out = []
        for c in cands:
            src = getattr(c, "source", "")
            dt = getattr(c, "from_dt", 0)
            if (src == "raw") and (dt == 0):
                out.append(c)
        return out
    raise ValueError(f"Unknown use_sources={use_sources}")


def safe_np_boxes(xs, exp_dim=9) -> np.ndarray:
    arr = np.array(xs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, exp_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, exp_dim)
    return arr


def oracle_cover_counts(
    gt_xy: np.ndarray,
    gt_labels: np.ndarray,
    cand_xy: np.ndarray,
    cand_labels: np.ndarray,
    thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      covered_any:    [G] bool (any label)
      covered_strict: [G] bool (must match label)
    """
    G = gt_xy.shape[0]
    C = cand_xy.shape[0]
    if G == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    if C == 0:
        return np.zeros((G,), dtype=bool), np.zeros((G,), dtype=bool)

    # distances: [G, C]
    dx = gt_xy[:, None, 0] - cand_xy[None, :, 0]
    dy = gt_xy[:, None, 1] - cand_xy[None, :, 1]
    d2 = dx * dx + dy * dy
    m_dist = d2 <= (thr * thr)

    covered_any = m_dist.any(axis=1)

    # strict label
    m_label = (gt_labels[:, None] == cand_labels[None, :])
    covered_strict = (m_dist & m_label).any(axis=1)
    return covered_any, covered_strict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Input JSON with det+gt")

    # candidate building (same knobs as test_phase1_infer)
    ap.add_argument("--use_warp", action="store_true")
    ap.add_argument("--warp_radius", type=int, default=2)
    ap.add_argument("--warp_topk", type=int, default=200)
    ap.add_argument("--warp_decay", type=float, default=0.9)

    ap.add_argument("--use_sources", type=str, default="all", choices=["raw", "all"])
    ap.add_argument("--match_thr", type=float, default=2.0)
    ap.add_argument("--ignore_classes", type=str, default="-1")

    # candidate caps / filtering (passed into CandidateConfig if supported by your data.py)
    ap.add_argument("--max_per_frame", type=int, default=None)
    ap.add_argument("--raw_score_min", type=float, default=0.0)

    # reporting
    ap.add_argument("--topk_classes", type=int, default=15)
    ap.add_argument("--limit_scenes", type=int, default=None)
    args = ap.parse_args()

    ignore = parse_ignore(args.ignore_classes)

    print(f"[1/3] Loading JSON: {args.json}")
    res = load_root_and_samples(args.json, require_gt=True)
    samples = res.samples
    scenes = group_by_scene(samples)
    print(f"  samples={len(samples)} scenes={len(scenes)}")

    if args.limit_scenes is not None:
        # deterministic subset by sorted scene_token
        keys = sorted(list(scenes.keys()))[: int(args.limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
        print(f"  limit_scenes={args.limit_scenes} -> using scenes={len(scenes)}")

    print("[2/3] Building candidates...")
    cand_cfg = CandidateConfig(
        use_warp=bool(args.use_warp),
        warp_radius=int(args.warp_radius),
        warp_topk=int(args.warp_topk),
        warp_score_decay=float(args.warp_decay),
        match_thr_xy=float(args.match_thr),
        ignore_classes=ignore,
        raw_score_min=float(args.raw_score_min),
        max_per_frame=args.max_per_frame,
    )
    frames_by_scene = build_frames_by_scene(scenes, cand_cfg)

    # candidate stats
    tot_cand = tot_raw = tot_warp = 0
    tot_frames = 0
    for sc, frames in frames_by_scene.items():
        for fr in frames:
            tot_frames += 1
            for c in fr.candidates:
                tot_cand += 1
                if (getattr(c, "source", "") == "raw") and (getattr(c, "from_dt", 0) == 0):
                    tot_raw += 1
                elif is_warp_candidate(c):
                    tot_warp += 1
    print(f"  [cand stats] frames={tot_frames} total={tot_cand} raw={tot_raw} warp={tot_warp}")

    print("[3/3] Oracle recall computation...")

    # totals
    gt_total = 0
    covered_any_total = 0
    covered_strict_total = 0

    # per-class
    cls_gt = defaultdict(int)
    cls_any = defaultdict(int)
    cls_strict = defaultdict(int)

    # also count how many frames have 0 candidates / 0 gt
    frames_no_gt = 0
    frames_no_cand = 0

    for sc, frames in frames_by_scene.items():
        for fr in frames:
            sample = fr.sample
            gt = (sample.get("gt") or {})
            gt_boxes = safe_np_boxes(gt.get("boxes_3d", []) or [], 9)
            gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)

            if gt_boxes.shape[0] == 0:
                frames_no_gt += 1
                continue

            # ignore classes on GT
            if len(ignore) > 0:
                keep = np.array([int(l) not in ignore for l in gt_labels.tolist()], dtype=bool)
                gt_boxes = gt_boxes[keep]
                gt_labels = gt_labels[keep]

            if gt_boxes.shape[0] == 0:
                frames_no_gt += 1
                continue

            # select candidates
            cands = select_candidates(fr, args.use_sources)
            if len(cands) == 0:
                frames_no_cand += 1

            cand_boxes = np.stack([c.box for c in cands], axis=0).astype(np.float32) if len(cands) > 0 else np.zeros((0, 9), np.float32)
            cand_labels = np.array([int(c.label) for c in cands], dtype=np.int64).reshape(-1) if len(cands) > 0 else np.zeros((0,), np.int64)

            # optional ignore on candidates too (mostly for consistency)
            if len(ignore) > 0 and cand_boxes.shape[0] > 0:
                keepc = np.array([int(l) not in ignore for l in cand_labels.tolist()], dtype=bool)
                cand_boxes = cand_boxes[keepc]
                cand_labels = cand_labels[keepc]

            gt_xy = gt_boxes[:, :2].astype(np.float32)
            cand_xy = cand_boxes[:, :2].astype(np.float32)

            covered_any, covered_strict = oracle_cover_counts(
                gt_xy=gt_xy,
                gt_labels=gt_labels,
                cand_xy=cand_xy,
                cand_labels=cand_labels,
                thr=float(args.match_thr),
            )

            # aggregate
            G = int(gt_boxes.shape[0])
            gt_total += G
            covered_any_total += int(covered_any.sum())
            covered_strict_total += int(covered_strict.sum())

            # per-class
            for i in range(G):
                c = int(gt_labels[i])
                cls_gt[c] += 1
                if bool(covered_any[i]):
                    cls_any[c] += 1
                if bool(covered_strict[i]):
                    cls_strict[c] += 1

    # report overall
    rec_any = covered_any_total / max(1, gt_total)
    rec_strict = covered_strict_total / max(1, gt_total)

    print("\n=== Oracle Recall Upper Bound ===")
    print(f"GT total = {gt_total}")
    print(f"Oracle(any-label)   = {rec_any:.4f}  ({covered_any_total}/{gt_total})")
    print(f"Oracle(strict-label)= {rec_strict:.4f}  ({covered_strict_total}/{gt_total})")
    print(f"Frames with no GT   = {frames_no_gt}")
    print(f"Frames with no cand = {frames_no_cand}  (only among frames that had GT after ignore)")

    # top-k classes by GT count
    topk = int(args.topk_classes)
    items = sorted(cls_gt.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    if len(items) > 0:
        print(f"\n=== Per-class Oracle Recall (Top {topk} by GT count) ===")
        print("label_id | gt_count | oracle_any | oracle_strict")
        for c, n in items:
            ra = cls_any[c] / max(1, n)
            rs = cls_strict[c] / max(1, n)
            print(f"{c:7d} | {n:8d} | {ra:9.4f} | {rs:12.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
