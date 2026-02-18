#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small sweep for refine inference hyperparams, reporting P/R/F1 without rewriting JSON every run.

It reuses the same pipeline modules as scripts/test_phase1_infer.py:
  - json_io.py: load_root_and_samples, group_by_scene, sample_key_candidates, write_refined_json (optional)
  - data.py: CandidateConfig, build_frames_by_scene
  - infer.py: InferConfig, infer_frames_by_scene

We evaluate in-memory by matching each frame's predictions (from repl_map) to that frame's GT.

Example:
  python scripts/sweep_infer_small.py \
  --in_json  /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir  /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/sweep_dbg \
  --use_warp --warp_radius 2 --warp_topk 200 --warp_decay 0.9 \
  --use_sources all \
  --match_thr 2.0 \
  --ignore_classes -1 \
  --limit_scenes 10


python scripts/sweep_infer_small.py \
  --in_json  /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir  /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/sweep_full \
  --use_warp --warp_radius 2 --warp_topk 200 --warp_decay 0.9 \
  --use_sources all \
  --match_thr 2.0 \
  --ignore_classes -1 \
  --save_best_json

LOG=/home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/sweep_full/run_$(date +%Y%m%d-%H%M%S).log

python -u scripts/sweep_infer_small.py \
  --in_json  /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir  /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/sweep_full \
  --use_warp --warp_radius 2 --warp_topk 200 --warp_decay 0.9 \
  --use_sources all \
  --match_thr 2.0 \
  --ignore_classes -1 \
  --save_best_json \
  2>&1 | tee "$LOG"


OUTDIR=/home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/sweep_full
mkdir -p "$OUTDIR"
LOG=$OUTDIR/run_$(date +%Y%m%d-%H%M%S).log
PIDFILE=$OUTDIR/run.pid

nohup python -u scripts/sweep_infer_small.py \
  --in_json  /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_dir  "$OUTDIR" \
  --use_warp --warp_radius 2 --warp_topk 200 --warp_decay 0.9 \
  --use_sources all \
  --match_thr 2.0 \
  --ignore_classes -1 \
  --save_best_json \
  > "$LOG" 2>&1 & echo $! > "$PIDFILE"

echo "log: $LOG"
echo "pid: $(cat $PIDFILE)"

tail -f "$LOG"
kill $(cat "$PIDFILE")
# 不行再强制：
kill -9 $(cat "$PIDFILE")



Notes:
- If you sweep warp params (radius/topk/decay), candidate building is re-run per warp config.
- If you only sweep infer params (keep_thr/nms_thr/topk), candidates are built once per warp config.
"""

import os
import sys
import csv
import json
import argparse
from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np

# Make repo root importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from json_io import load_root_and_samples, group_by_scene, sample_key_candidates, write_refined_json
from data import CandidateConfig, build_frames_by_scene
from infer import InferConfig, infer_frames_by_scene


# -----------------------------
# Helpers
# -----------------------------

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

def safe_np_boxes(xs, exp_dim=9) -> np.ndarray:
    arr = np.array(xs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, exp_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, exp_dim)
    return arr

def class_aware_greedy_tp(pred_xy: np.ndarray, pred_lab: np.ndarray,
                          gt_xy: np.ndarray, gt_lab: np.ndarray,
                          thr: float) -> int:
    """
    One-to-one greedy matching per class within XY distance threshold.
    Returns TP count.
    """
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return 0
    tp = 0
    classes = set(pred_lab.tolist()) | set(gt_lab.tolist())
    thr2 = float(thr) * float(thr)

    for c in classes:
        pi = np.where(pred_lab == c)[0]
        gi = np.where(gt_lab == c)[0]
        if pi.size == 0 or gi.size == 0:
            continue

        P = pred_xy[pi]  # [P,2]
        G = gt_xy[gi]    # [G,2]
        # dist^2 [P,G]
        dx = P[:, None, 0] - G[None, :, 0]
        dy = P[:, None, 1] - G[None, :, 1]
        d2 = dx * dx + dy * dy

        pairs: List[Tuple[float, int, int]] = []
        for ii in range(d2.shape[0]):
            for jj in range(d2.shape[1]):
                if d2[ii, jj] <= thr2:
                    pairs.append((float(d2[ii, jj]), int(pi[ii]), int(gi[jj])))

        if not pairs:
            continue
        pairs.sort(key=lambda x: x[0])

        used_p, used_g = set(), set()
        for _, p_idx, g_idx in pairs:
            if p_idx in used_p or g_idx in used_g:
                continue
            used_p.add(p_idx)
            used_g.add(g_idx)
            tp += 1

    return tp

def find_payload_for_sample(repl_map: Dict[str, Dict[str, Any]], sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for k in sample_key_candidates(sample):
        if k in repl_map:
            return repl_map[k]
    return None

def eval_repl_map(frames_by_scene: Dict[str, List[Any]],
                  repl_map: Dict[str, Dict[str, Any]],
                  match_thr_xy: float,
                  ignore_classes: Set[int]) -> Dict[str, Any]:
    """
    Evaluate P/R/F1 in-memory:
      - preds from repl_map payload: boxes_3d, labels_3d
      - GT from sample["gt"]: boxes_3d, labels_3d
    """
    tot_tp = tot_fp = tot_fn = 0
    n_frames = 0
    n_det = 0
    n_gt = 0
    n_frames_with_gt = 0

    for _, frames in frames_by_scene.items():
        for fr in frames:
            sample = fr.sample
            n_frames += 1

            gt = (sample.get("gt") or {})
            gt_boxes = safe_np_boxes(gt.get("boxes_3d", []) or [], 9)
            gt_lab = np.array(gt.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)

            # filter ignore on GT
            if gt_boxes.shape[0] > 0 and len(ignore_classes) > 0:
                keep = np.array([int(l) not in ignore_classes for l in gt_lab.tolist()], dtype=bool)
                gt_boxes = gt_boxes[keep]
                gt_lab = gt_lab[keep]

            G = int(gt_boxes.shape[0])
            n_gt += G
            if G > 0:
                n_frames_with_gt += 1

            payload = find_payload_for_sample(repl_map, sample)
            if payload is None:
                # no prediction for this sample
                if G > 0:
                    tot_fn += G
                continue

            pred_boxes = safe_np_boxes(payload.get("boxes_3d", []) or [], 9)
            pred_lab = np.array(payload.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)

            # filter ignore on preds
            if pred_boxes.shape[0] > 0 and len(ignore_classes) > 0:
                keepp = np.array([int(l) not in ignore_classes for l in pred_lab.tolist()], dtype=bool)
                pred_boxes = pred_boxes[keepp]
                pred_lab = pred_lab[keepp]

            Pn = int(pred_boxes.shape[0])
            n_det += Pn

            if G == 0:
                tot_fp += Pn
                continue
            if Pn == 0:
                tot_fn += G
                continue

            tp = class_aware_greedy_tp(
                pred_xy=pred_boxes[:, :2].astype(np.float32),
                pred_lab=pred_lab.astype(np.int64),
                gt_xy=gt_boxes[:, :2].astype(np.float32),
                gt_lab=gt_lab.astype(np.int64),
                thr=float(match_thr_xy),
            )
            fp = Pn - tp
            fn = G - tp

            tot_tp += tp
            tot_fp += fp
            tot_fn += fn

    P = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    R = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {
        "P": float(P), "R": float(R), "F1": float(F1),
        "tp": int(tot_tp), "fp": int(tot_fp), "fn": int(tot_fn),
        "frames": int(n_frames), "frames_with_gt": int(n_frames_with_gt),
        "det": int(n_det), "gt": int(n_gt),
    }


# -----------------------------
# Sweep grids (small, editable)
# -----------------------------

def default_sweep_grid():
    """
    Keep it small and meaningful (12–18 runs typical).
    You can edit these lists if you want.
    """
    keep_thrs = [0.50, 0.40, 0.3]
    nms_thrs  = [1.5, 1.0]
    topks     = [400]
    return keep_thrs, nms_thrs, topks


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--match_thr", type=float, default=2.0)
    ap.add_argument("--ignore_classes", type=str, default="-1")
    ap.add_argument("--use_sources", type=str, default="all", choices=["raw", "all"])
    ap.add_argument("--backend", type=str, default="heuristic", choices=["heuristic", "ebm"])


    # candidate building knobs
    ap.add_argument("--use_warp", action="store_true")
    ap.add_argument("--warp_radius", type=int, default=2)
    ap.add_argument("--warp_topk", type=int, default=200)
    ap.add_argument("--warp_decay", type=float, default=0.9)
    ap.add_argument("--raw_score_min", type=float, default=0.0)
    ap.add_argument("--max_per_frame", type=int, default=None)

    # runtime
    ap.add_argument("--limit_scenes", type=int, default=None, help="quick debug: only evaluate first K scenes")
    ap.add_argument("--save_best_json", action="store_true", help="write best det_refined json (one file)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ignore = parse_ignore(args.ignore_classes)

    print(f"[1/4] Load: {args.in_json}")
    res = load_root_and_samples(args.in_json, require_gt=True)
    samples = res.samples
    scenes = group_by_scene(samples)
    print(f"  samples={len(samples)} scenes={len(scenes)}")

    if args.limit_scenes is not None:
        keys = sorted(list(scenes.keys()))[: int(args.limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
        print(f"  limit_scenes={args.limit_scenes} -> using scenes={len(scenes)}")

    # Build candidates once for this warp config
    print("[2/4] Build candidates...")
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
    tot = tot_raw = tot_warp = 0
    for _, frames in frames_by_scene.items():
        for fr in frames:
            for c in fr.candidates:
                tot += 1
                if (getattr(c, "source", "") == "raw") and (getattr(c, "from_dt", 0) == 0):
                    tot_raw += 1
                elif (getattr(c, "from_dt", 0) != 0) or str(getattr(c, "source", "")).startswith("warp("):
                    tot_warp += 1
    print(f"  [cand stats] total={tot} raw={tot_raw} warp={tot_warp}")

    keep_thrs, nms_thrs, topks = default_sweep_grid()

    # Output CSV
    csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    print(f"[3/4] Sweep -> {csv_path}")

    best = None  # (F1, row_dict, repl_map)
    rows = []

    run_id = 0
    for keep_thr in keep_thrs:
        for nms_thr in nms_thrs:
            for topk in topks:
                run_id += 1
                infer_cfg = InferConfig(
                    backend=args.backend,  
                    use_sources=args.use_sources,
                    keep_thr=float(keep_thr),
                    nms_thr_xy=float(nms_thr),
                    topk=int(topk),
                    predict_attr=True,
                    include_sources=False,

                    two_stage=True,
                    seed_keep_thr=0.5,   # 或者单独 sweep
                    seed_sources="raw",
                    fill_keep_thr=float(keep_thr) ,
                    min_dt_support=2,
                    dt_cell_size=1.0,
                    min_dist_to_seed=0.8,
                    max_fill=300,
                )

                repl_map = infer_frames_by_scene(frames_by_scene, infer_cfg)
                m = eval_repl_map(frames_by_scene, repl_map, match_thr_xy=args.match_thr, ignore_classes=ignore)

                row = {
                    "run_id": run_id,
                    "use_warp": int(bool(args.use_warp)),
                    "warp_radius": int(args.warp_radius),
                    "warp_topk": int(args.warp_topk),
                    "warp_decay": float(args.warp_decay),
                    "use_sources": args.use_sources,
                    "backend": args.backend,
                    "keep_thr": float(keep_thr),
                    "nms_thr": float(nms_thr),
                    "topk": int(topk),
                    "P": m["P"], "R": m["R"], "F1": m["F1"],
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                    "det": m["det"], "gt": m["gt"],
                }
                rows.append(row)

                print(f"[{run_id:02d}] keep_thr={keep_thr:.2f} nms_thr={nms_thr:.1f} topk={topk:<4d} | "
                      f"P={m['P']:.4f} R={m['R']:.4f} F1={m['F1']:.4f} | det={m['det']}")

                if best is None or row["F1"] > best[0]:
                    best = (row["F1"], row, repl_map)

    # write CSV
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("[4/4] Done sweep.")
    if best is None:
        print("No runs executed.")
        return

    print("\n=== BEST (by F1) ===")
    print(json.dumps(best[1], indent=2))

    # optionally save best json
    if args.save_best_json:
        best_json_path = os.path.join(args.out_dir, "best_det_refined.json")
        print(f"\n[save_best_json] writing -> {best_json_path}")
        stats = write_refined_json(
            in_json_path=args.in_json,
            out_json_path=best_json_path,
            repl_map=best[2],
            mode="add",
            refined_key="det_refined",
            # Only keep essential fields to reduce file size
            fields=["boxes_3d", "labels_3d", "scores_3d", "attrs"],
            indent=2,
        )
        print(f"[writeback] matched={stats.matched} updated={stats.updated} visited={stats.visited}")


if __name__ == "__main__":
    main()
