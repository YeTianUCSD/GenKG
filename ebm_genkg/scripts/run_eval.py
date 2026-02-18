#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified evaluation runner for ebm_genkg.

Features:
- Evaluate det / det_refined with the same metrics implementation.
- Optional subset evaluation via --limit_scenes.
- Optional multi-threshold evaluation.
- Support config file (json / yaml) + CLI override.

Examples:
  python scripts/run_eval.py \
    --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json  \
    --eval_raw --eval_refined \
    --match_thrs 2.0 \
    --ignore_classes -1

  python scripts/run_eval.py \
    --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json  \
    --eval_refined \
    --limit_scenes 10 \
    --match_thrs 1.0,2.0,3.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from json_io import load_root_and_samples, group_by_scene
from metrics import evaluate_samples, format_metrics


def _parse_ignore(s: str) -> Set[int]:
    out: Set[int] = set()
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.add(int(x))
    return out


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    if not out:
        out = [2.0]
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
    p = argparse.ArgumentParser(description="Evaluate det/det_refined with unified metrics.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")

    p.add_argument("--in_json", type=str, default=cfg.get("in_json"), required=("in_json" not in cfg))
    p.add_argument("--require_gt", action="store_true", default=bool(cfg.get("require_gt", True)))

    _add_bool_arg(p, "eval_raw", bool(cfg.get("eval_raw", True)), "Evaluate det field")
    _add_bool_arg(p, "eval_refined", bool(cfg.get("eval_refined", True)), "Evaluate det_refined field")

    p.add_argument("--match_thrs", type=str, default=str(cfg.get("match_thrs", "2.0")),
                   help="Comma-separated XY thresholds in meters.")
    p.add_argument("--ignore_classes", type=str, default=str(cfg.get("ignore_classes", "-1")))

    p.add_argument("--raw_attr_field", type=str, default=cfg.get("raw_attr_field"),
                   help="Pred attr field under det for raw eval (optional).")
    p.add_argument("--refined_attr_field", type=str, default=str(cfg.get("refined_attr_field", "attrs")),
                   help="Pred attr field under det_refined for refined eval.")

    p.add_argument("--limit_scenes", type=int, default=cfg.get("limit_scenes"),
                   help="Only evaluate first K scenes.")
    _add_bool_arg(p, "print_delta", bool(cfg.get("print_delta", True)), "Print refined-raw delta")
    p.add_argument("--summary_json", type=str, default=cfg.get("summary_json"),
                   help="Optional path to write eval summary JSON.")

    return p


def _collect_samples(in_json: str, require_gt: bool, limit_scenes: Optional[int]) -> List[Dict[str, Any]]:
    lr = load_root_and_samples(in_json, require_gt=require_gt)
    scenes = group_by_scene(lr.samples)
    if limit_scenes is not None:
        keys = sorted(list(scenes.keys()))[: int(limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
        print(f"  limit_scenes={limit_scenes} -> using scenes={len(scenes)}")

    out: List[Dict[str, Any]] = []
    for _, frames in scenes.items():
        out.extend(frames)
    return out


def _eval_one(
    samples: List[Dict[str, Any]],
    det_field: str,
    match_thr: float,
    ignore_classes: Set[int],
    pred_attr_field: Optional[str],
):
    return evaluate_samples(
        samples,
        det_field=det_field,
        match_thr_xy=float(match_thr),
        ignore_classes=ignore_classes,
        pred_attr_field=pred_attr_field,
        attr_vocab=None,
    )


def main() -> None:
    # first pass for config
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default=None)
    a0, _ = p0.parse_known_args()

    cfg = _load_cfg(a0.config)
    parser = _make_parser(cfg)
    args = parser.parse_args()

    if (not args.eval_raw) and (not args.eval_refined):
        raise ValueError("At least one of --eval-raw/--eval-refined must be enabled.")

    ignore = _parse_ignore(args.ignore_classes)
    thrs = _parse_float_list(args.match_thrs)

    print("[1/2] Loading samples...")
    samples = _collect_samples(
        in_json=args.in_json,
        require_gt=bool(args.require_gt),
        limit_scenes=args.limit_scenes,
    )
    print(f"  samples={len(samples)}")

    print("[2/2] Evaluating...")
    results: List[Dict[str, Any]] = []
    for thr in thrs:
        print(f"\n=== match_thr_xy={thr:.3f} ===")
        raw_m = None
        ref_m = None

        if args.eval_raw:
            raw_m = _eval_one(
                samples=samples,
                det_field="det",
                match_thr=float(thr),
                ignore_classes=ignore,
                pred_attr_field=(args.raw_attr_field if args.raw_attr_field else None),
            )
            print("RAW :", format_metrics(raw_m))

        if args.eval_refined:
            pred_attr = args.refined_attr_field if args.refined_attr_field else None
            ref_m = _eval_one(
                samples=samples,
                det_field="det_refined",
                match_thr=float(thr),
                ignore_classes=ignore,
                pred_attr_field=pred_attr,
            )
            print("REF :", format_metrics(ref_m))

        if args.print_delta and (raw_m is not None) and (ref_m is not None):
            dP = ref_m.P - raw_m.P
            dR = ref_m.R - raw_m.R
            dF1 = ref_m.F1 - raw_m.F1
            print(f"DELTA(ref-raw): dP={dP:+.4f} dR={dR:+.4f} dF1={dF1:+.4f}")

        rec: Dict[str, Any] = {"match_thr_xy": float(thr)}
        if raw_m is not None:
            rec["raw"] = {
                "P": float(raw_m.P),
                "R": float(raw_m.R),
                "F1": float(raw_m.F1),
                "tp": int(raw_m.tp),
                "fp": int(raw_m.fp),
                "fn": int(raw_m.fn),
                "num_frames": int(raw_m.num_frames),
                "num_det": int(raw_m.num_det),
                "num_gt": int(raw_m.num_gt),
                "attr_acc": float(raw_m.attr_acc),
                "attr_macro_f1": float(raw_m.attr_macro_f1),
                "attr_num_tp_with_attr": int(raw_m.attr_num_tp_with_attr),
            }
        if ref_m is not None:
            rec["refined"] = {
                "P": float(ref_m.P),
                "R": float(ref_m.R),
                "F1": float(ref_m.F1),
                "tp": int(ref_m.tp),
                "fp": int(ref_m.fp),
                "fn": int(ref_m.fn),
                "num_frames": int(ref_m.num_frames),
                "num_det": int(ref_m.num_det),
                "num_gt": int(ref_m.num_gt),
                "attr_acc": float(ref_m.attr_acc),
                "attr_macro_f1": float(ref_m.attr_macro_f1),
                "attr_num_tp_with_attr": int(ref_m.attr_num_tp_with_attr),
            }
        if (raw_m is not None) and (ref_m is not None):
            rec["delta_ref_minus_raw"] = {
                "dP": float(ref_m.P - raw_m.P),
                "dR": float(ref_m.R - raw_m.R),
                "dF1": float(ref_m.F1 - raw_m.F1),
            }
        results.append(rec)

    if args.summary_json:
        out_dir = os.path.dirname(os.path.abspath(args.summary_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary = {
            "stage": "eval",
            "in_json": str(args.in_json),
            "limit_scenes": args.limit_scenes,
            "ignore_classes": sorted(list(ignore)),
            "eval_raw": bool(args.eval_raw),
            "eval_refined": bool(args.eval_refined),
            "match_thrs": [float(x) for x in thrs],
            "num_samples": int(len(samples)),
            "results": results,
            "args": vars(args),
        }
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] wrote {args.summary_json}")


if __name__ == "__main__":
    main()
