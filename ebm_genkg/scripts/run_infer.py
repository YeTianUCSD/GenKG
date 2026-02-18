#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified inference runner for ebm_genkg.

Features:
- Load dataset JSON, build candidates, run refinement (heuristic or ebm).
- Write results back to output JSON as det_refined (default).
- Support config file (json / yaml) + CLI override.
- Support quick iteration with --limit_scenes.

Example:
  python scripts/run_infer.py \
      --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
    --out_json /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/infer_refined_scene20.json \
    --backend ebm \
    --use_warp \
    --two_stage \
    --limit_scenes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from json_io import load_root_and_samples, group_by_scene, write_refined_json, dump_json
from data import CandidateConfig, build_frames_by_scene
from infer import InferConfig, infer_scene


def _parse_ignore(s: str):
    out = set()
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


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    flag = "--" + name.replace("_", "-")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return

    group = parser.add_mutually_exclusive_group()
    group.add_argument(flag, dest=name, action="store_true", help=help_text)
    group.add_argument("--no-" + name.replace("_", "-"), dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def _make_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ebm_genkg inference and write det_refined.")

    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")

    # IO
    p.add_argument("--in_json", type=str, default=cfg.get("in_json"), required=("in_json" not in cfg))
    p.add_argument("--out_json", type=str, default=cfg.get("out_json"), required=("out_json" not in cfg))
    p.add_argument("--require_gt", action="store_true", default=bool(cfg.get("require_gt", False)),
                   help="Require GT while loading samples.")
    p.add_argument("--limit_scenes", type=int, default=cfg.get("limit_scenes"),
                   help="Only run first K scenes for quick iteration.")
    p.add_argument("--log_every_scenes", type=int, default=int(cfg.get("log_every_scenes", 10)))

    # candidate build
    _add_bool_arg(p, "use_warp", bool(cfg.get("use_warp", False)), "Use warped candidates from neighbor frames")
    p.add_argument("--warp_radius", type=int, default=int(cfg.get("warp_radius", 2)))
    p.add_argument("--warp_topk", type=int, default=int(cfg.get("warp_topk", 200)))
    p.add_argument("--warp_decay", type=float, default=float(cfg.get("warp_decay", 0.9)))
    p.add_argument("--raw_score_min", type=float, default=float(cfg.get("raw_score_min", 0.0)))
    p.add_argument("--max_per_frame", type=int, default=cfg.get("max_per_frame"))
    p.add_argument("--ignore_classes", type=str, default=str(cfg.get("ignore_classes", "-1")))

    # infer
    p.add_argument("--backend", type=str, default=str(cfg.get("backend", "heuristic")), choices=["heuristic", "ebm"])
    p.add_argument("--use_sources", type=str, default=str(cfg.get("use_sources", "all")), choices=["raw", "all"])
    p.add_argument("--keep_thr", type=float, default=float(cfg.get("keep_thr", 0.5)))
    p.add_argument("--nms_thr", type=float, default=float(cfg.get("nms_thr", 1.5)))
    p.add_argument("--topk", type=int, default=int(cfg.get("topk", 200)))
    _add_bool_arg(p, "include_sources", bool(cfg.get("include_sources", False)), "Include debug sources in output")

    _add_bool_arg(p, "two_stage", bool(cfg.get("two_stage", False)), "Enable two-stage seed/fill")
    p.add_argument("--seed_keep_thr", type=float, default=float(cfg.get("seed_keep_thr", 0.5)))
    p.add_argument("--seed_sources", type=str, default=str(cfg.get("seed_sources", "raw")), choices=["raw", "all"])
    p.add_argument("--fill_keep_thr", type=float, default=float(cfg.get("fill_keep_thr", 0.35)))
    p.add_argument("--min_dt_support", type=int, default=int(cfg.get("min_dt_support", 2)))
    p.add_argument("--dt_cell_size", type=float, default=float(cfg.get("dt_cell_size", 1.0)))
    p.add_argument("--min_dist_to_seed", type=float, default=float(cfg.get("min_dist_to_seed", 0.8)))
    p.add_argument("--max_fill", type=int, default=int(cfg.get("max_fill", 300)))

    # EBM kwargs (packed into InferConfig.ebm_kwargs)
    p.add_argument("--ebm_prefilter_topm_seed", type=int, default=int(cfg.get("ebm_prefilter_topm_seed", 800)))
    p.add_argument("--ebm_prefilter_topm_fill", type=int, default=int(cfg.get("ebm_prefilter_topm_fill", 1200)))
    _add_bool_arg(p, "ebm_enable_label_vote", bool(cfg.get("ebm_enable_label_vote", False)), "Enable label vote in EBM")
    p.add_argument("--ebm_w_attr", type=float, default=float(cfg.get("ebm_w_attr", 0.0)))
    p.add_argument("--ebm_unary_ckpt_path", type=str, default=cfg.get("ebm_unary_ckpt_path"))
    _add_bool_arg(p, "ebm_unary_use_learned", bool(cfg.get("ebm_unary_use_learned", True)),
                  "Use learned unary ckpt in EBM when provided")

    # writeback
    p.add_argument("--write_mode", type=str, default=str(cfg.get("write_mode", "add")), choices=["add", "replace"])
    p.add_argument("--det_key", type=str, default=str(cfg.get("det_key", "det")))
    p.add_argument("--refined_key", type=str, default=str(cfg.get("refined_key", "det_refined")))
    p.add_argument(
        "--write_fields",
        type=str,
        default=str(cfg.get("write_fields", "boxes_3d,labels_3d,scores_3d,attrs")),
        help="Comma-separated fields to write from payload. Add 'sources' when needed.",
    )
    p.add_argument("--indent", type=int, default=int(cfg.get("indent", 2)))

    p.add_argument("--dump_repl_map", type=str, default=cfg.get("dump_repl_map"),
                   help="Optional path to dump replacement map JSON for debugging.")
    p.add_argument("--summary_json", type=str, default=cfg.get("summary_json"),
                   help="Optional path to write run summary JSON.")
    p.add_argument("--ckpt_path", type=str, default=cfg.get("ckpt_path"),
                   help="Optional checkpoint path metadata for experiment tracking.")

    return p


def main() -> None:
    # first pass: only for config path
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default=None)
    a0, _ = p0.parse_known_args()

    cfg = _load_cfg(a0.config)
    parser = _make_parser(cfg)
    args = parser.parse_args()

    ignore = _parse_ignore(args.ignore_classes)

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("[1/4] Loading samples...")
    t0 = time.time()
    res = load_root_and_samples(args.in_json, require_gt=bool(args.require_gt))
    scenes = group_by_scene(res.samples)
    if args.limit_scenes is not None:
        keys = sorted(list(scenes.keys()))[: int(args.limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
        print(f"  limit_scenes={args.limit_scenes} -> using scenes={len(scenes)}")
    print(f"  samples={len(res.samples)} scenes={len(scenes)} load_time={time.time()-t0:.1f}s")

    print("[2/4] Building candidates...")
    t1 = time.time()
    cand_cfg = CandidateConfig(
        use_warp=bool(args.use_warp),
        warp_radius=int(args.warp_radius),
        warp_topk=int(args.warp_topk),
        warp_score_decay=float(args.warp_decay),
        ignore_classes=ignore,
        raw_score_min=float(args.raw_score_min),
        max_per_frame=args.max_per_frame,
    )
    frames_by_scene = build_frames_by_scene(scenes, cand_cfg)
    tot_all = sum(len(fr.candidates) for fs in frames_by_scene.values() for fr in fs)
    tot_raw = sum(
        1
        for fs in frames_by_scene.values()
        for fr in fs
        for c in fr.candidates
        if c.source == "raw" and c.from_dt == 0
    )
    print(f"  candidate_total={tot_all} raw={tot_raw} warp_or_other={tot_all - tot_raw} build_time={time.time()-t1:.1f}s")

    print("[3/4] Running inference...")
    t2 = time.time()
    ebm_kwargs = dict(cfg.get("ebm_kwargs", {}) if isinstance(cfg.get("ebm_kwargs", {}), dict) else {})
    ebm_kwargs.update(
        {
            "prefilter_topm_seed": int(args.ebm_prefilter_topm_seed),
            "prefilter_topm_fill": int(args.ebm_prefilter_topm_fill),
            "enable_label_vote": bool(args.ebm_enable_label_vote),
            "w_attr": float(args.ebm_w_attr),
            "unary_use_learned": bool(args.ebm_unary_use_learned),
            "unary_ckpt_path": (str(args.ebm_unary_ckpt_path) if args.ebm_unary_ckpt_path else None),
        }
    )

    infer_cfg = InferConfig(
        backend=str(args.backend),
        use_sources=str(args.use_sources),
        keep_thr=float(args.keep_thr),
        nms_thr_xy=float(args.nms_thr),
        topk=int(args.topk),
        predict_attr=True,
        include_sources=bool(args.include_sources),
        two_stage=bool(args.two_stage),
        seed_keep_thr=float(args.seed_keep_thr),
        seed_sources=str(args.seed_sources),
        fill_keep_thr=float(args.fill_keep_thr),
        min_dt_support=int(args.min_dt_support),
        dt_cell_size=float(args.dt_cell_size),
        min_dist_to_seed=float(args.min_dist_to_seed),
        max_fill=int(args.max_fill),
        ebm_kwargs=ebm_kwargs,
    )

    repl_map: Dict[str, Dict[str, Any]] = {}
    scene_items = list(frames_by_scene.items())
    total_scenes = len(scene_items)
    total_frames = sum(len(frames) for _, frames in scene_items)
    done_frames = 0

    for i, (_, frames) in enumerate(scene_items, start=1):
        repl_map.update(infer_scene(frames, infer_cfg))
        done_frames += len(frames)

        if args.log_every_scenes > 0 and (i % int(args.log_every_scenes) == 0 or i == total_scenes):
            elapsed = time.time() - t2
            fps = done_frames / max(elapsed, 1e-6)
            print(
                f"  progress: scenes={i}/{total_scenes}, frames={done_frames}/{total_frames}, "
                f"elapsed={elapsed:.1f}s, fps={fps:.2f}"
            )

    print(f"  repl_map_size={len(repl_map)} infer_time={time.time()-t2:.1f}s")

    if args.dump_repl_map:
        dump_dir = os.path.dirname(os.path.abspath(args.dump_repl_map))
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        dump_json(repl_map, args.dump_repl_map, indent=2)
        print(f"  dumped repl_map -> {args.dump_repl_map}")

    print("[4/4] Writing output JSON...")
    t3 = time.time()
    fields = [x.strip() for x in str(args.write_fields).split(",") if x.strip()]
    stats = write_refined_json(
        in_json_path=args.in_json,
        out_json_path=args.out_json,
        repl_map=repl_map,
        mode=str(args.write_mode),
        det_key=str(args.det_key),
        refined_key=str(args.refined_key),
        fields=fields if len(fields) > 0 else None,
        indent=int(args.indent),
    )
    print(
        f"[writeback] matched={stats.matched} updated={stats.updated} visited={stats.visited} "
        f"total_time={time.time()-t0:.1f}s"
    )

    if args.summary_json:
        summ_dir = os.path.dirname(os.path.abspath(args.summary_json))
        if summ_dir:
            os.makedirs(summ_dir, exist_ok=True)

        summary: Dict[str, Any] = {
            "stage": "infer",
            "in_json": str(args.in_json),
            "out_json": str(args.out_json),
            "backend": str(args.backend),
            "use_warp": bool(args.use_warp),
            "limit_scenes": args.limit_scenes,
            "num_samples_loaded": int(len(res.samples)),
            "num_scenes_used": int(len(scenes)),
            "num_frames_used": int(total_frames),
            "num_candidates_total": int(tot_all),
            "num_candidates_raw": int(tot_raw),
            "num_candidates_warp_or_other": int(tot_all - tot_raw),
            "repl_map_size": int(len(repl_map)),
            "writeback": {
                "matched": int(stats.matched),
                "updated": int(stats.updated),
                "visited": int(stats.visited),
            },
            "timing_sec": {
                "load": float(t1 - t0),
                "build": float(t2 - t1),
                "infer": float(t3 - t2),
                "write": float(time.time() - t3),
                "total": float(time.time() - t0),
            },
            "ckpt_path": str(args.ckpt_path) if args.ckpt_path else None,
            "args": vars(args),
        }
        dump_json(summary, args.summary_json, indent=2)
        print(f"[summary] wrote {args.summary_json}")


if __name__ == "__main__":
    main()
