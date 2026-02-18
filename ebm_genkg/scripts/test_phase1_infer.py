#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick end-to-end sanity test:
- load json
- build candidates (optionally with warp)
- run infer -> det_refined
- write out json
- evaluate raw(det) vs refined(det_refined)
heuristic/ebm


Example:
  python scripts/test_phase1_infer.py \
    --in_json  /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
    --out_json /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/out_refined.json \
    --use_warp --warp_radius 2 --warp_topk 200 --warp_decay 0.9 \
    --use_sources all \
    --backend ebm \
    --two_stage \
    --seed_keep_thr 0.50 \
    --fill_keep_thr 0.35 \
    --min_dt_support 2 \
    --dt_cell_size 1.0 \
    --min_dist_to_seed 0.8 \
    --max_fill 300 \
    --keep_thr 0.50 --nms_thr 1.5 --topk 200 \
    --match_thr 2.0 --ignore_classes -1

python scripts/test_phase1_infer.py \
  --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/out_refined_scene20.json \
  --limit_scenes 20 \
  --use_warp --warp_radius 2 --warp_topk 120 --warp_decay 0.9 \
  --raw_score_min 0.05 --max_per_frame 180 \
  --use_sources all \
  --backend ebm \
  --two_stage \
  --seed_keep_thr 0.50 \
  --fill_keep_thr 0.35 \
  --min_dt_support 2 \
  --dt_cell_size 1.0 \
  --min_dist_to_seed 0.8 \
  --max_fill 220 \
  --ebm_prefilter_topm_seed 500 \
  --ebm_prefilter_topm_fill 700 \
  --ebm_w_attr 0.0 \
  --keep_thr 0.50 --nms_thr 1.5 --topk 200 \
  --match_thr 2.0 --ignore_classes -1 \
  --log_every_scenes 2

"""

import os
import sys
import argparse
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# make repo root importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from json_io import load_root_and_samples, group_by_scene, write_refined_json
from data import CandidateConfig, build_frames_by_scene
from infer import InferConfig, infer_scene
from metrics import eval_json_path, format_metrics


def parse_ignore(s: str):
    ignore = set()
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            ignore.add(int(x))
    return ignore


def _safe_np_boxes(xs, exp_dim=9) -> np.ndarray:
    arr = np.array(xs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, exp_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, exp_dim)
    return arr


def _class_aware_greedy_tp(
    pred_xy: np.ndarray,
    pred_lab: np.ndarray,
    gt_xy: np.ndarray,
    gt_lab: np.ndarray,
    thr: float,
) -> int:
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return 0
    tp = 0
    thr2 = float(thr) * float(thr)
    classes = set(pred_lab.tolist()) | set(gt_lab.tolist())
    for c in classes:
        pi = np.where(pred_lab == c)[0]
        gi = np.where(gt_lab == c)[0]
        if pi.size == 0 or gi.size == 0:
            continue

        P = pred_xy[pi]
        G = gt_xy[gi]
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


def _find_payload_for_sample(repl_map: Dict[str, Dict[str, Any]], sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from json_io import sample_key_candidates

    for k in sample_key_candidates(sample):
        if k in repl_map:
            return repl_map[k]
    return None


def _eval_repl_map(
    frames_by_scene: Dict[str, List[Any]],
    repl_map: Dict[str, Dict[str, Any]],
    match_thr_xy: float,
    ignore_classes: Set[int],
) -> Dict[str, Any]:
    tot_tp = tot_fp = tot_fn = 0
    n_frames = n_det = n_gt = 0

    for _, frames in frames_by_scene.items():
        for fr in frames:
            n_frames += 1
            sample = fr.sample

            gt = (sample.get("gt") or {})
            gt_boxes = _safe_np_boxes(gt.get("boxes_3d", []) or [], 9)
            gt_lab = np.array(gt.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)
            if gt_boxes.shape[0] > 0 and ignore_classes:
                keep = np.array([int(l) not in ignore_classes for l in gt_lab.tolist()], dtype=bool)
                gt_boxes = gt_boxes[keep]
                gt_lab = gt_lab[keep]
            G = int(gt_boxes.shape[0])
            n_gt += G

            payload = _find_payload_for_sample(repl_map, sample)
            if payload is None:
                tot_fn += G
                continue

            pred_boxes = _safe_np_boxes(payload.get("boxes_3d", []) or [], 9)
            pred_lab = np.array(payload.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)
            if pred_boxes.shape[0] > 0 and ignore_classes:
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

            tp = _class_aware_greedy_tp(
                pred_xy=pred_boxes[:, :2].astype(np.float32),
                pred_lab=pred_lab.astype(np.int64),
                gt_xy=gt_boxes[:, :2].astype(np.float32),
                gt_lab=gt_lab.astype(np.int64),
                thr=float(match_thr_xy),
            )
            tot_tp += int(tp)
            tot_fp += int(Pn - tp)
            tot_fn += int(G - tp)

    P = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    R = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {
        "P": float(P),
        "R": float(R),
        "F1": float(F1),
        "tp": int(tot_tp),
        "fp": int(tot_fp),
        "fn": int(tot_fn),
        "frames": int(n_frames),
        "det": int(n_det),
        "gt": int(n_gt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)

    # candidate building
    ap.add_argument("--use_warp", action="store_true")
    ap.add_argument("--warp_radius", type=int, default=2)
    ap.add_argument("--warp_topk", type=int, default=200)
    ap.add_argument("--warp_decay", type=float, default=0.9)
    ap.add_argument("--raw_score_min", type=float, default=0.0)
    ap.add_argument("--max_per_frame", type=int, default=None)
    ap.add_argument("--limit_scenes", type=int, default=None, help="only run first K scenes for quick iteration")

    # inference (common)
    ap.add_argument("--backend", type=str, default="heuristic", choices=["heuristic", "ebm"],
                    help="infer backend in infer.py (you'll implement 'ebm' branch).")
    ap.add_argument("--keep_thr", type=float, default=0.5)
    ap.add_argument("--nms_thr", type=float, default=1.5)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--use_sources", type=str, default="all", choices=["raw", "all"])
    ap.add_argument("--include_sources", action="store_true",
                    help="write debug sources into det_refined (larger json).")

    # inference (two-stage fill)
    ap.add_argument("--two_stage", action="store_true")
    ap.add_argument("--seed_keep_thr", type=float, default=0.5)
    ap.add_argument("--seed_sources", type=str, default="raw", choices=["raw", "all"])
    ap.add_argument("--fill_keep_thr", type=float, default=0.35)
    ap.add_argument("--min_dt_support", type=int, default=2)
    ap.add_argument("--dt_cell_size", type=float, default=1.0)
    ap.add_argument("--min_dist_to_seed", type=float, default=0.8)
    ap.add_argument("--max_fill", type=int, default=300)
    ap.add_argument("--log_every_scenes", type=int, default=10)

    # ebm runtime knobs (packed into InferConfig.ebm_kwargs)
    ap.add_argument("--ebm_prefilter_topm_seed", type=int, default=800)
    ap.add_argument("--ebm_prefilter_topm_fill", type=int, default=1200)
    ap.add_argument("--ebm_enable_label_vote", action="store_true")
    ap.add_argument("--ebm_w_attr", type=float, default=0.0)

    # eval
    ap.add_argument("--match_thr", type=float, default=2.0)
    ap.add_argument("--ignore_classes", type=str, default="-1")
    args = ap.parse_args()

    # ensure out dir exists
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ignore = parse_ignore(args.ignore_classes)

    # 0) evaluate raw
    print("[0/4] Eval RAW det on input json...")
    raw_m = eval_json_path(
        args.in_json,
        det_field="det",
        match_thr_xy=args.match_thr,
        ignore_classes=ignore,
        pred_attr_field=None,
    )
    print("RAW :", format_metrics(raw_m))

    # 1) load + group scenes
    print("[1/4] Loading samples...")
    res = load_root_and_samples(args.in_json, require_gt=True)
    samples = res.samples
    scenes = group_by_scene(samples)
    if args.limit_scenes is not None:
        keys = sorted(list(scenes.keys()))[: int(args.limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
        print(f"  limit_scenes={args.limit_scenes} -> using scenes={len(scenes)}")
    print(f"  samples={len(samples)} scenes={len(scenes)}")

    # 2) build candidates
    print("[2/4] Building candidates...")
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

    # cand stats
    tot_all = tot_raw = tot_warp = 0
    for _, frames in frames_by_scene.items():
        for fr in frames:
            tot_all += len(fr.candidates)
            for c in fr.candidates:
                if (c.source == "raw") and (c.from_dt == 0):
                    tot_raw += 1
                elif (c.from_dt != 0) or str(c.source).startswith("warp("):
                    tot_warp += 1
    print(f"[cand stats] total={tot_all} raw={tot_raw} warp={tot_warp}")

    # 3) inference -> repl_map
    print("[3/4] Running inference...")
    infer_cfg = InferConfig(
        use_sources=args.use_sources,
        keep_thr=float(args.keep_thr),
        nms_thr_xy=float(args.nms_thr),
        topk=int(args.topk),
        predict_attr=True,
        include_sources=bool(args.include_sources),

        # backend selector (requires your infer.py to support it)
        backend=str(args.backend) if hasattr(InferConfig, "backend") else None,

        # two-stage fill knobs
        two_stage=bool(args.two_stage),
        seed_keep_thr=float(args.seed_keep_thr),
        seed_sources=str(args.seed_sources),
        fill_keep_thr=float(args.fill_keep_thr),
        min_dt_support=int(args.min_dt_support),
        dt_cell_size=float(args.dt_cell_size),
        min_dist_to_seed=float(args.min_dist_to_seed),
        max_fill=int(args.max_fill),
        ebm_kwargs={
            "prefilter_topm_seed": int(args.ebm_prefilter_topm_seed),
            "prefilter_topm_fill": int(args.ebm_prefilter_topm_fill),
            "enable_label_vote": bool(args.ebm_enable_label_vote),
            "w_attr": float(args.ebm_w_attr),
        },
    )

    # NOTE: if your InferConfig currently doesn't have "backend" field,
    # you can safely remove the backend=... line above.

    repl_map = {}
    scene_items = list(frames_by_scene.items())
    total_scenes = len(scene_items)
    total_frames = sum(len(frames) for _, frames in scene_items)
    done_frames = 0
    t_infer = time.time()
    for i, (_, frames) in enumerate(scene_items, start=1):
        repl_map.update(infer_scene(frames, infer_cfg))
        done_frames += len(frames)

        if args.log_every_scenes > 0 and (i % int(args.log_every_scenes) == 0 or i == total_scenes):
            elapsed = time.time() - t_infer
            fps = done_frames / max(elapsed, 1e-6)
            print(f"  progress: scenes={i}/{total_scenes}, frames={done_frames}/{total_frames}, "
                  f"elapsed={elapsed:.1f}s, fps={fps:.2f}")
    print(f"  repl_map size={len(repl_map)} (should be ~= samples)")

    # refined stats (if sources are included)
    if args.include_sources:
        ref_all = ref_raw = ref_warp = 0
        for _, payload in repl_map.items():
            srcs = payload.get("sources", []) or []
            ref_all += len(srcs)
            for s in srcs:
                if s == "raw":
                    ref_raw += 1
                elif str(s).startswith("warp("):
                    ref_warp += 1
        print(f"[refined stats] total={ref_all} raw={ref_raw} warp={ref_warp} "
              f"(warp_ratio={ref_warp / max(1, ref_all):.3f})")

    # 4) write out json with det_refined
    print("[4/4] Write det_refined -> out_json")
    stats = write_refined_json(
        in_json_path=args.in_json,
        out_json_path=args.out_json,
        repl_map=repl_map,
        refined_key="det_refined",
        mode="add",
        # 只写关键字段，避免 json 过大；如果你需要 sources/debug，再把它加回来
        fields=["boxes_3d", "labels_3d", "scores_3d", "attrs"] + (["sources"] if args.include_sources else []),
        indent=2,
    )
    print(f"[writeback] matched={stats.matched} updated={stats.updated} visited={stats.visited}")

    # eval refined
    if args.limit_scenes is not None:
        print("\n[Eval] det_refined on subset (frames_by_scene + repl_map)...")
        m = _eval_repl_map(
            frames_by_scene=frames_by_scene,
            repl_map=repl_map,
            match_thr_xy=float(args.match_thr),
            ignore_classes=ignore,
        )
        print(
            "REF_SUBSET : "
            f"P={m['P']:.4f} R={m['R']:.4f} F1={m['F1']:.4f} | "
            f"tp={m['tp']} fp={m['fp']} fn={m['fn']} | "
            f"frames={m['frames']} det={m['det']} gt={m['gt']}"
        )
    else:
        print("\n[Eval] det_refined on out_json...")
        ref_m = eval_json_path(
            args.out_json,
            det_field="det_refined",
            match_thr_xy=args.match_thr,
            ignore_classes=ignore,
            pred_attr_field="attrs",
        )
        print("REF :", format_metrics(ref_m))

    print("\nDone.")


if __name__ == "__main__":
    main()
