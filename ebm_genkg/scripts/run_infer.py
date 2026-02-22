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

from json_io import load_root_and_samples, group_by_scene, sample_key_candidates, dump_json
from data import CandidateConfig, build_candidates_for_scene
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
    alt_flag = "--" + name
    flags = [flag] if flag == alt_flag else [flag, alt_flag]
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(*flags, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return

    group = parser.add_mutually_exclusive_group()
    group.add_argument(*flags, dest=name, action="store_true", help=help_text)
    group.add_argument(
        "--no-" + name.replace("_", "-"),
        "--no-" + name,
        dest=name,
        action="store_false",
        help=f"Disable {help_text}",
    )
    parser.set_defaults(**{name: default})


def _load_ckpt_threshold(ckpt_path: str, field_name: str = "best_threshold") -> Optional[float]:
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = obj.get(str(field_name), None)
        if v is None:
            return None
        x = float(v)
        if not (x == x):  # NaN
            return None
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return None


def _apply_payload_to_sample(
    sample: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    mode: str,
    det_key: str,
    refined_key: str,
    fields: Optional[list[str]],
) -> None:
    if mode == "replace":
        target = sample.get(det_key, {}) or {}
    else:
        target = sample.get(refined_key, {}) or {}
    if not isinstance(target, dict):
        target = {}

    if fields is None:
        for kk, vv in payload.items():
            target[kk] = vv
    else:
        for kk in fields:
            if kk in payload:
                target[kk] = payload[kk]

    if mode == "replace":
        sample[det_key] = target
    else:
        sample[refined_key] = target


def _make_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ebm_genkg inference and write det_refined.")

    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")

    # IO
    p.add_argument("--in_json", type=str, default=cfg.get("in_json"), required=("in_json" not in cfg))
    p.add_argument("--out_json", type=str, default=cfg.get("out_json"), required=("out_json" not in cfg))
    _add_bool_arg(p, "require_gt", bool(cfg.get("require_gt", False)), "Require GT while loading samples.")
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
    _add_bool_arg(
        p,
        "ebm_adaptive_prefilter",
        bool(cfg.get("ebm_adaptive_prefilter", True)),
        "Use adaptive prefilter budget based on per-frame candidate count",
    )
    p.add_argument("--ebm_adaptive_prefilter_base", type=int, default=int(cfg.get("ebm_adaptive_prefilter_base", 128)))
    p.add_argument(
        "--ebm_adaptive_prefilter_sqrt_scale",
        type=float,
        default=float(cfg.get("ebm_adaptive_prefilter_sqrt_scale", 48.0)),
    )
    p.add_argument("--ebm_adaptive_prefilter_min", type=int, default=int(cfg.get("ebm_adaptive_prefilter_min", 256)))
    p.add_argument("--ebm_adaptive_prefilter_max", type=int, default=int(cfg.get("ebm_adaptive_prefilter_max", 4096)))
    p.add_argument(
        "--ebm_energy_mode",
        type=str,
        default=str(cfg.get("ebm_energy_mode", "four_term")),
        choices=["four_term", "legacy"],
    )
    p.add_argument("--ebm_energy_select_margin", type=float, default=float(cfg.get("ebm_energy_select_margin", 0.0)))
    p.add_argument("--ebm_energy_prob_gate", type=float, default=float(cfg.get("ebm_energy_prob_gate", 0.0)))
    _add_bool_arg(
        p,
        "ebm_energy_local_refine",
        bool(cfg.get("ebm_energy_local_refine", True)),
        "Enable local remove/add refinement for four_term energy solve",
    )
    p.add_argument(
        "--ebm_energy_local_refine_rounds",
        type=int,
        default=int(cfg.get("ebm_energy_local_refine_rounds", 2)),
    )
    p.add_argument("--ebm_w_keep", type=float, default=float(cfg.get("ebm_w_keep", 1.0)))
    p.add_argument("--ebm_w_pair", type=float, default=float(cfg.get("ebm_w_pair", 1.0)))
    _add_bool_arg(
        p,
        "ebm_enable_learned_pair",
        bool(cfg.get("ebm_enable_learned_pair", True)),
        "Enable learned pair-energy term from structured energy checkpoint",
    )
    _add_bool_arg(
        p,
        "ebm_hard_nms",
        bool(cfg.get("ebm_hard_nms", False)),
        "Use hard NMS conflict in pair energy",
    )
    _add_bool_arg(p, "ebm_enable_label_vote", bool(cfg.get("ebm_enable_label_vote", False)), "Enable label vote in EBM")
    p.add_argument("--ebm_w_attr", type=float, default=float(cfg.get("ebm_w_attr", 0.0)))
    _add_bool_arg(
        p,
        "ebm_enable_overlap_soft",
        bool(cfg.get("ebm_enable_overlap_soft", True)),
        "Enable soft overlap penalty term in EBM pair energy",
    )
    p.add_argument("--ebm_overlap_min_ratio", type=float, default=float(cfg.get("ebm_overlap_min_ratio", 0.10)))
    p.add_argument("--ebm_overlap_soft_scale", type=float, default=float(cfg.get("ebm_overlap_soft_scale", 1.00)))
    _add_bool_arg(
        p,
        "ebm_enable_temporal_pair",
        bool(cfg.get("ebm_enable_temporal_pair", True)),
        "Enable temporal pair consistency bonus term in EBM pair energy",
    )
    p.add_argument("--ebm_temporal_pair_radius", type=float, default=float(cfg.get("ebm_temporal_pair_radius", 1.5)))
    p.add_argument("--ebm_temporal_pair_bonus", type=float, default=float(cfg.get("ebm_temporal_pair_bonus", 0.30)))
    _add_bool_arg(
        p,
        "ebm_temporal_pair_warp_only",
        bool(cfg.get("ebm_temporal_pair_warp_only", True)),
        "Apply temporal pair bonus only when at least one candidate is warp",
    )
    _add_bool_arg(
        p,
        "ebm_soft_dt_support",
        bool(cfg.get("ebm_soft_dt_support", True)),
        "Use soft relation-energy penalty for dt support shortfall instead of hard filter",
    )
    p.add_argument(
        "--ebm_dt_shortfall_penalty",
        type=float,
        default=float(cfg.get("ebm_dt_shortfall_penalty", 0.35)),
        help="Energy penalty per unit dt-support shortfall when ebm_soft_dt_support is enabled.",
    )
    _add_bool_arg(
        p,
        "ebm_enable_context_density",
        bool(cfg.get("ebm_enable_context_density", True)),
        "Enable context-density unary relation energy term",
    )
    p.add_argument("--ebm_context_cell_xy", type=float, default=float(cfg.get("ebm_context_cell_xy", 1.0)))
    p.add_argument("--ebm_context_min_density", type=int, default=int(cfg.get("ebm_context_min_density", 2)))
    p.add_argument("--ebm_context_shortfall_penalty", type=float, default=float(cfg.get("ebm_context_shortfall_penalty", 0.15)))
    _add_bool_arg(
        p,
        "ebm_context_warp_only",
        bool(cfg.get("ebm_context_warp_only", True)),
        "Apply context-density penalty only to warp candidates",
    )
    p.add_argument("--ebm_unary_ckpt_path", type=str, default=cfg.get("ebm_unary_ckpt_path"))
    _add_bool_arg(p, "ebm_unary_use_learned", bool(cfg.get("ebm_unary_use_learned", True)),
                  "Use learned unary ckpt in EBM when provided")
    _add_bool_arg(
        p,
        "ebm_use_learned_class",
        bool(cfg.get("ebm_use_learned_class", True)),
        "Use learned class head from unary ckpt to relabel detections",
    )
    p.add_argument(
        "--ebm_learned_class_min_prob",
        type=float,
        default=float(cfg.get("ebm_learned_class_min_prob", 0.55)),
        help="Minimum class posterior to accept learned class relabel.",
    )
    _add_bool_arg(
        p,
        "ebm_use_learned_attr",
        bool(cfg.get("ebm_use_learned_attr", True)),
        "Use learned attr head from unary ckpt to predict attrs",
    )
    p.add_argument(
        "--ebm_learned_attr_min_prob",
        type=float,
        default=float(cfg.get("ebm_learned_attr_min_prob", 0.50)),
        help="Minimum attr posterior to accept learned attr prediction.",
    )
    _add_bool_arg(
        p,
        "ebm_dual_head_solver",
        bool(cfg.get("ebm_dual_head_solver", True)),
        "Enable staged keep/add dual-head solve in four_term mode",
    )
    _add_bool_arg(
        p,
        "ebm_dual_head_unified_context",
        bool(cfg.get("ebm_dual_head_unified_context", True)),
        "Use unified candidate pool for dual-head solve (avoid raw-only hard cutoff).",
    )
    p.add_argument("--ebm_keep_head_raw_bias", type=float, default=float(cfg.get("ebm_keep_head_raw_bias", 0.20)))
    p.add_argument("--ebm_keep_head_nonraw_bias", type=float, default=float(cfg.get("ebm_keep_head_nonraw_bias", -0.80)))
    p.add_argument("--ebm_add_head_warp_bias", type=float, default=float(cfg.get("ebm_add_head_warp_bias", 0.25)))
    p.add_argument("--ebm_add_head_nonwarp_bias", type=float, default=float(cfg.get("ebm_add_head_nonwarp_bias", -6.0)))
    p.add_argument("--ebm_add_head_support_gain", type=float, default=float(cfg.get("ebm_add_head_support_gain", 0.20)))
    p.add_argument("--ebm_add_head_potential_gain", type=float, default=float(cfg.get("ebm_add_head_potential_gain", 0.35)))
    p.add_argument("--ebm_add_head_class_gain", type=float, default=float(cfg.get("ebm_add_head_class_gain", 0.25)))
    p.add_argument("--ebm_stage_a_keep_thr", type=float, default=float(cfg.get("ebm_stage_a_keep_thr", 0.50)))
    p.add_argument("--ebm_stage_b_add_thr", type=float, default=float(cfg.get("ebm_stage_b_add_thr", 0.35)))
    p.add_argument(
        "--ebm_stage_b_min_potential_dist",
        type=float,
        default=float(cfg.get("ebm_stage_b_min_potential_dist", 1.2)),
    )
    p.add_argument("--ebm_stage_b_min_potential", type=float, default=float(cfg.get("ebm_stage_b_min_potential", 0.25)))
    p.add_argument("--ebm_stage_b_min_class_conf", type=float, default=float(cfg.get("ebm_stage_b_min_class_conf", 0.0)))
    _add_bool_arg(
        p,
        "ebm_stage_b_enforce_class_conf",
        bool(cfg.get("ebm_stage_b_enforce_class_conf", False)),
        "Require class confidence gate in Stage-B add pool",
    )
    p.add_argument("--ebm_stage_c_attr_scale", type=float, default=float(cfg.get("ebm_stage_c_attr_scale", 0.20)))
    p.add_argument("--ebm_stage_c_rel_scale", type=float, default=float(cfg.get("ebm_stage_c_rel_scale", 0.20)))
    _add_bool_arg(
        p,
        "ebm_recall_backfill_enabled",
        bool(cfg.get("ebm_recall_backfill_enabled", True)),
        "Enable recall-oriented backfill after EBM energy solve.",
    )
    p.add_argument(
        "--ebm_recall_backfill_min_keep_prob",
        type=float,
        default=float(cfg.get("ebm_recall_backfill_min_keep_prob", 0.15)),
    )
    p.add_argument(
        "--ebm_recall_backfill_min_per_frame",
        type=int,
        default=int(cfg.get("ebm_recall_backfill_min_per_frame", 120)),
    )
    p.add_argument(
        "--ebm_recall_backfill_raw_ratio",
        type=float,
        default=float(cfg.get("ebm_recall_backfill_raw_ratio", 0.75)),
    )
    p.add_argument(
        "--ebm_recall_backfill_pair_guard_scale",
        type=float,
        default=float(cfg.get("ebm_recall_backfill_pair_guard_scale", 0.50)),
    )
    p.add_argument(
        "--ebm_recall_backfill_min_class_conf",
        type=float,
        default=float(cfg.get("ebm_recall_backfill_min_class_conf", 0.65)),
    )
    p.add_argument(
        "--ebm_recall_backfill_min_support",
        type=int,
        default=int(cfg.get("ebm_recall_backfill_min_support", 2)),
    )
    _add_bool_arg(
        p,
        "ebm_auto_threshold_from_ckpt",
        bool(cfg.get("ebm_auto_threshold_from_ckpt", True)),
        "Auto align keep/seed thresholds with unary ckpt best_threshold",
    )
    p.add_argument(
        "--ebm_ckpt_threshold_field",
        type=str,
        default=str(cfg.get("ebm_ckpt_threshold_field", "best_threshold")),
        help="Field name in unary ckpt used for auto threshold alignment.",
    )

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
    auto_thr_info: Dict[str, Any] = {
        "enabled": bool(args.ebm_auto_threshold_from_ckpt),
        "applied": False,
        "threshold": None,
        "ckpt_path": str(args.ebm_unary_ckpt_path) if args.ebm_unary_ckpt_path else None,
        "field": str(args.ebm_ckpt_threshold_field),
    }

    if (
        str(args.backend) == "ebm"
        and bool(args.ebm_unary_use_learned)
        and bool(args.ebm_auto_threshold_from_ckpt)
        and args.ebm_unary_ckpt_path
    ):
        thr = _load_ckpt_threshold(str(args.ebm_unary_ckpt_path), field_name=str(args.ebm_ckpt_threshold_field))
        if thr is not None:
            old_keep = float(args.keep_thr)
            old_seed = float(args.seed_keep_thr)
            old_fill = float(args.fill_keep_thr)

            args.keep_thr = float(thr)
            if bool(args.two_stage):
                args.seed_keep_thr = float(thr)
                # Keep fill threshold no stricter than seed by default.
                if float(args.fill_keep_thr) > float(args.seed_keep_thr):
                    args.fill_keep_thr = float(args.seed_keep_thr)

            auto_thr_info.update(
                {
                    "applied": True,
                    "threshold": float(thr),
                    "old_keep_thr": old_keep,
                    "old_seed_keep_thr": old_seed,
                    "old_fill_keep_thr": old_fill,
                    "new_keep_thr": float(args.keep_thr),
                    "new_seed_keep_thr": float(args.seed_keep_thr),
                    "new_fill_keep_thr": float(args.fill_keep_thr),
                }
            )
            print(
                "[auto-thr] applied from ckpt: "
                f"keep_thr {old_keep:.3f}->{float(args.keep_thr):.3f}, "
                f"seed_keep_thr {old_seed:.3f}->{float(args.seed_keep_thr):.3f}, "
                f"fill_keep_thr {old_fill:.3f}->{float(args.fill_keep_thr):.3f}"
            )
        else:
            print(
                "[auto-thr] skipped: cannot load threshold "
                f"'{args.ebm_ckpt_threshold_field}' from {args.ebm_unary_ckpt_path}"
            )

    # Safety fallback: when using learned unary in EBM, allow --ckpt_path to act as unary ckpt.
    # This avoids silently using a stale ckpt path from config in ad-hoc inference runs.
    if (
        str(args.backend) == "ebm"
        and bool(args.ebm_unary_use_learned)
        and (not args.ebm_unary_ckpt_path)
        and bool(args.ckpt_path)
    ):
        args.ebm_unary_ckpt_path = str(args.ckpt_path)
        auto_thr_info["ckpt_path"] = str(args.ebm_unary_ckpt_path)
        print(f"[ckpt-fallback] use --ckpt_path as ebm_unary_ckpt_path: {args.ebm_unary_ckpt_path}")
        # If auto-threshold was enabled but skipped earlier due missing unary ckpt path,
        # re-try threshold alignment after fallback.
        if bool(args.ebm_auto_threshold_from_ckpt):
            thr = _load_ckpt_threshold(str(args.ebm_unary_ckpt_path), field_name=str(args.ebm_ckpt_threshold_field))
            if thr is not None:
                old_keep = float(args.keep_thr)
                old_seed = float(args.seed_keep_thr)
                old_fill = float(args.fill_keep_thr)
                args.keep_thr = float(thr)
                if bool(args.two_stage):
                    args.seed_keep_thr = float(thr)
                    if float(args.fill_keep_thr) > float(args.seed_keep_thr):
                        args.fill_keep_thr = float(args.seed_keep_thr)
                auto_thr_info.update(
                    {
                        "applied": True,
                        "threshold": float(thr),
                        "old_keep_thr": old_keep,
                        "old_seed_keep_thr": old_seed,
                        "old_fill_keep_thr": old_fill,
                        "new_keep_thr": float(args.keep_thr),
                        "new_seed_keep_thr": float(args.seed_keep_thr),
                        "new_fill_keep_thr": float(args.fill_keep_thr),
                    }
                )
                print(
                    "[auto-thr] applied from fallback ckpt: "
                    f"keep_thr {old_keep:.3f}->{float(args.keep_thr):.3f}, "
                    f"seed_keep_thr {old_seed:.3f}->{float(args.seed_keep_thr):.3f}, "
                    f"fill_keep_thr {old_fill:.3f}->{float(args.fill_keep_thr):.3f}"
                )

    ignore = _parse_ignore(args.ignore_classes)

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("[1/4] Loading samples...")
    t0 = time.time()
    res = load_root_and_samples(args.in_json, require_gt=bool(args.require_gt), keep_root=True)
    if res.root is None:
        raise RuntimeError("load_root_and_samples returned empty root unexpectedly.")
    scenes = group_by_scene(res.samples)
    if args.limit_scenes is not None:
        keys = sorted(list(scenes.keys()))[: int(args.limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
        print(f"  limit_scenes={args.limit_scenes} -> using scenes={len(scenes)}")
    print(f"  samples={len(res.samples)} scenes={len(scenes)} load_time={time.time()-t0:.1f}s")

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
    print("[2/4] Building candidates + running inference (stream by scene)...")
    t2 = time.time()
    ebm_kwargs = dict(cfg.get("ebm_kwargs", {}) if isinstance(cfg.get("ebm_kwargs", {}), dict) else {})
    ebm_kwargs.update(
        {
            "energy_mode": str(args.ebm_energy_mode),
            "energy_select_margin": float(args.ebm_energy_select_margin),
            "energy_prob_gate": float(args.ebm_energy_prob_gate),
            "energy_local_refine": bool(args.ebm_energy_local_refine),
            "energy_local_refine_rounds": int(args.ebm_energy_local_refine_rounds),
            "w_keep": float(args.ebm_w_keep),
            "w_pair": float(args.ebm_w_pair),
            "enable_learned_pair": bool(args.ebm_enable_learned_pair),
            "hard_nms": bool(args.ebm_hard_nms),
            "prefilter_topm_seed": int(args.ebm_prefilter_topm_seed),
            "prefilter_topm_fill": int(args.ebm_prefilter_topm_fill),
            "adaptive_prefilter": bool(args.ebm_adaptive_prefilter),
            "adaptive_prefilter_base": int(args.ebm_adaptive_prefilter_base),
            "adaptive_prefilter_sqrt_scale": float(args.ebm_adaptive_prefilter_sqrt_scale),
            "adaptive_prefilter_min": int(args.ebm_adaptive_prefilter_min),
            "adaptive_prefilter_max": int(args.ebm_adaptive_prefilter_max),
            "enable_label_vote": bool(args.ebm_enable_label_vote),
            "w_attr": float(args.ebm_w_attr),
            "enable_overlap_soft": bool(args.ebm_enable_overlap_soft),
            "overlap_min_ratio": float(args.ebm_overlap_min_ratio),
            "overlap_soft_scale": float(args.ebm_overlap_soft_scale),
            "enable_temporal_pair": bool(args.ebm_enable_temporal_pair),
            "temporal_pair_radius": float(args.ebm_temporal_pair_radius),
            "temporal_pair_bonus": float(args.ebm_temporal_pair_bonus),
            "temporal_pair_warp_only": bool(args.ebm_temporal_pair_warp_only),
            "soft_dt_support": bool(args.ebm_soft_dt_support),
            "dt_shortfall_penalty": float(args.ebm_dt_shortfall_penalty),
            "enable_context_density": bool(args.ebm_enable_context_density),
            "context_cell_xy": float(args.ebm_context_cell_xy),
            "context_min_density": int(args.ebm_context_min_density),
            "context_shortfall_penalty": float(args.ebm_context_shortfall_penalty),
            "context_warp_only": bool(args.ebm_context_warp_only),
            "unary_use_learned": bool(args.ebm_unary_use_learned),
            "unary_ckpt_path": (str(args.ebm_unary_ckpt_path) if args.ebm_unary_ckpt_path else None),
            "use_learned_class": bool(args.ebm_use_learned_class),
            "learned_class_min_prob": float(args.ebm_learned_class_min_prob),
            "use_learned_attr": bool(args.ebm_use_learned_attr),
            "learned_attr_min_prob": float(args.ebm_learned_attr_min_prob),
            "dual_head_solver": bool(args.ebm_dual_head_solver),
            "dual_head_unified_context": bool(args.ebm_dual_head_unified_context),
            "keep_head_raw_bias": float(args.ebm_keep_head_raw_bias),
            "keep_head_nonraw_bias": float(args.ebm_keep_head_nonraw_bias),
            "add_head_warp_bias": float(args.ebm_add_head_warp_bias),
            "add_head_nonwarp_bias": float(args.ebm_add_head_nonwarp_bias),
            "add_head_support_gain": float(args.ebm_add_head_support_gain),
            "add_head_potential_gain": float(args.ebm_add_head_potential_gain),
            "add_head_class_gain": float(args.ebm_add_head_class_gain),
            "stage_a_keep_thr": float(args.ebm_stage_a_keep_thr),
            "stage_b_add_thr": float(args.ebm_stage_b_add_thr),
            "stage_b_min_potential_dist": float(args.ebm_stage_b_min_potential_dist),
            "stage_b_min_potential": float(args.ebm_stage_b_min_potential),
            "stage_b_min_class_conf": float(args.ebm_stage_b_min_class_conf),
            "stage_b_enforce_class_conf": bool(args.ebm_stage_b_enforce_class_conf),
            "stage_c_attr_scale": float(args.ebm_stage_c_attr_scale),
            "stage_c_rel_scale": float(args.ebm_stage_c_rel_scale),
            "recall_backfill_enabled": bool(args.ebm_recall_backfill_enabled),
            "recall_backfill_min_keep_prob": float(args.ebm_recall_backfill_min_keep_prob),
            "recall_backfill_min_per_frame": int(args.ebm_recall_backfill_min_per_frame),
            "recall_backfill_raw_ratio": float(args.ebm_recall_backfill_raw_ratio),
            "recall_backfill_pair_guard_scale": float(args.ebm_recall_backfill_pair_guard_scale),
            "recall_backfill_min_class_conf": float(args.ebm_recall_backfill_min_class_conf),
            "recall_backfill_min_support": int(args.ebm_recall_backfill_min_support),
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
        dual_head_solver=bool(args.ebm_dual_head_solver),
        stage_a_keep_thr=float(args.ebm_stage_a_keep_thr),
        stage_b_add_thr=float(args.ebm_stage_b_add_thr),
        stage_b_min_potential_dist=float(args.ebm_stage_b_min_potential_dist),
        stage_b_min_potential=float(args.ebm_stage_b_min_potential),
        stage_c_attr_scale=float(args.ebm_stage_c_attr_scale),
        stage_c_rel_scale=float(args.ebm_stage_c_rel_scale),
        ebm_kwargs=ebm_kwargs,
    )

    repl_map: Optional[Dict[str, Dict[str, Any]]] = {} if args.dump_repl_map else None
    scene_keys = sorted(list(scenes.keys()))
    total_scenes = len(scene_keys)
    total_frames = sum(len(scenes[k]) for k in scene_keys)
    done_frames = 0
    tot_all = 0
    tot_raw = 0
    matched = 0
    updated = 0
    build_time_acc = 0.0
    infer_time_acc = 0.0
    fields = [x.strip() for x in str(args.write_fields).split(",") if x.strip()]
    fields_use = fields if len(fields) > 0 else None

    for i, sc in enumerate(scene_keys, start=1):
        tb = time.time()
        frames = build_candidates_for_scene(scenes[sc], cand_cfg)
        build_time_acc += time.time() - tb

        for fr in frames:
            n_c = len(fr.candidates)
            tot_all += n_c
            if n_c > 0:
                tot_raw += sum(1 for c in fr.candidates if c.source == "raw" and c.from_dt == 0)

        ti = time.time()
        scene_payloads = infer_scene(frames, infer_cfg)
        infer_time_acc += time.time() - ti
        if repl_map is not None:
            repl_map.update(scene_payloads)

        for fr in frames:
            keys = sample_key_candidates(fr.sample)
            chosen = None
            for k in keys:
                if k and str(k) != "":
                    chosen = str(k)
                    break
            if chosen is None:
                chosen = f"{fr.scene_token}:{fr.timestamp}"
            payload = scene_payloads.get(chosen, None)
            if payload is None:
                continue
            matched += 1
            _apply_payload_to_sample(
                fr.sample,
                payload,
                mode=str(args.write_mode),
                det_key=str(args.det_key),
                refined_key=str(args.refined_key),
                fields=fields_use,
            )
            updated += 1
        done_frames += len(frames)

        if args.log_every_scenes > 0 and (i % int(args.log_every_scenes) == 0 or i == total_scenes):
            elapsed = time.time() - t2
            fps = done_frames / max(elapsed, 1e-6)
            print(
                f"  progress: scenes={i}/{total_scenes}, frames={done_frames}/{total_frames}, "
                f"elapsed={elapsed:.1f}s, fps={fps:.2f}"
            )

    if repl_map is not None:
        print(f"  repl_map_size={len(repl_map)} infer_time={time.time()-t2:.1f}s")
    else:
        print(f"  infer_time={time.time()-t2:.1f}s")
    print(f"  candidate_total={tot_all} raw={tot_raw} warp_or_other={tot_all - tot_raw}")
    print(f"  build_time={build_time_acc:.1f}s infer_core_time={infer_time_acc:.1f}s")

    if args.dump_repl_map and repl_map is not None:
        dump_dir = os.path.dirname(os.path.abspath(args.dump_repl_map))
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        dump_json(repl_map, args.dump_repl_map, indent=2)
        print(f"  dumped repl_map -> {args.dump_repl_map}")

    print("[4/4] Writing output JSON...")
    t3 = time.time()
    dump_json(res.root, args.out_json, indent=int(args.indent))
    print(
        f"[writeback] matched={matched} updated={updated} visited={len(res.samples)} "
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
            "repl_map_size": int(len(repl_map)) if repl_map is not None else None,
            "writeback": {
                "matched": int(matched),
                "updated": int(updated),
                "visited": int(len(res.samples)),
            },
            "timing_sec": {
                "load": float(t1 - t0),
                "build": float(build_time_acc),
                "infer": float(infer_time_acc),
                "write": float(time.time() - t3),
                "total": float(time.time() - t0),
            },
            "ckpt_path": str(args.ckpt_path) if args.ckpt_path else None,
            "auto_threshold_from_ckpt": auto_thr_info,
            "args": vars(args),
        }
        dump_json(summary, args.summary_json, indent=2)
        print(f"[summary] wrote {args.summary_json}")


if __name__ == "__main__":
    main()
