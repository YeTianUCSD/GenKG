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
from metrics import evaluate_samples, evaluate_samples_detailed, format_metrics


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


def _parse_bins_list(s: str, default_bins: List[float]) -> List[float]:
    out: List[float] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    if not out:
        out = list(default_bins)
    return sorted([float(v) for v in out])


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


def _make_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate det/det_refined with unified metrics.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")

    p.add_argument("--in_json", type=str, default=cfg.get("in_json"), required=("in_json" not in cfg))
    _add_bool_arg(p, "require_gt", bool(cfg.get("require_gt", True)), "Require GT while loading samples")

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
    _add_bool_arg(p, "decompose_keep_add", bool(cfg.get("decompose_keep_add", True)), "Evaluate raw/after_keep/after_add decomposition")
    p.add_argument("--distance_bins", type=str, default=str(cfg.get("distance_bins", "20,40,60")))
    p.add_argument("--speed_bins", type=str, default=str(cfg.get("speed_bins", "0.2,2.0,6.0")))
    p.add_argument("--summary_json", type=str, default=cfg.get("summary_json"),
                   help="Optional path to write eval summary JSON.")
    p.add_argument("--report_md", type=str, default=cfg.get("report_md"),
                   help="Optional path to write markdown report.")

    return p


def _collect_samples(in_json: str, require_gt: bool, limit_scenes: Optional[int]) -> List[Dict[str, Any]]:
    lr = load_root_and_samples(in_json, require_gt=require_gt, keep_root=False)
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


def _src_is_warp(src: Any) -> bool:
    s = str(src).lower()
    return ("warp" in s) or ("dt=" in s and "dt=0" not in s and "dt=+0" not in s and "dt=-0" not in s)


def _build_after_keep_samples(
    samples: List[Dict[str, Any]],
    refined_field: str = "det_refined",
    out_field: str = "det_after_keep",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    out_keep: List[Dict[str, Any]] = []
    out_base: List[Dict[str, Any]] = []
    skipped = 0
    for s in samples:
        # Shallow-copy top-level dict only; avoid deep-copying large nested objects.
        c = dict(s)
        det_ref = (c.get(refined_field, {}) or {})
        boxes = list(det_ref.get("boxes_3d", []) or [])
        labels = list(det_ref.get("labels_3d", []) or [])
        scores = list(det_ref.get("scores_3d", []) or [])
        attrs = list(det_ref.get("attrs", []) or [])
        srcs = det_ref.get("sources", None)
        if not isinstance(srcs, list) or len(srcs) != len(boxes):
            skipped += 1
            continue
        keep = [not _src_is_warp(srcs[i]) for i in range(len(srcs))]
        c[out_field] = {
            "boxes_3d": [boxes[i] for i, m in enumerate(keep) if m],
            "labels_3d": [labels[i] for i, m in enumerate(keep) if m],
            "scores_3d": [scores[i] for i, m in enumerate(keep) if m],
            "attrs": [attrs[i] for i, m in enumerate(keep) if m] if len(attrs) == len(boxes) else [],
            "sources": [srcs[i] for i, m in enumerate(keep) if m],
        }
        out_keep.append(c)
        out_base.append(s)
    return out_keep, out_base, skipped


def _metric_triplet(
    samples: List[Dict[str, Any]],
    thr: float,
    ignore: Set[int],
    raw_attr_field: Optional[str],
    refined_attr_field: Optional[str],
    dist_bins: List[float],
    speed_bins: List[float],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    keep_samples, base_samples, skipped = _build_after_keep_samples(samples, refined_field="det_refined", out_field="det_after_keep")
    if len(base_samples) == 0:
        return {"raw": None, "after_keep": None, "after_add": None}, {
            "valid_samples": 0,
            "skipped_samples": int(skipped),
            "skip_ratio": 1.0,
            "fully_valid": False,
        }

    raw_d = evaluate_samples_detailed(
        base_samples,
        det_field="det",
        match_thr_xy=thr,
        ignore_classes=ignore,
        pred_attr_field=raw_attr_field,
        distance_bins=dist_bins,
        speed_bins=speed_bins,
    )
    keep_d = evaluate_samples_detailed(
        keep_samples,
        det_field="det_after_keep",
        match_thr_xy=thr,
        ignore_classes=ignore,
        pred_attr_field=refined_attr_field,
        distance_bins=dist_bins,
        speed_bins=speed_bins,
    )
    add_d = evaluate_samples_detailed(
        base_samples,
        det_field="det_refined",
        match_thr_xy=thr,
        ignore_classes=ignore,
        pred_attr_field=refined_attr_field,
        distance_bins=dist_bins,
        speed_bins=speed_bins,
    )
    total = max(1, len(samples))
    meta = {
        "valid_samples": int(len(base_samples)),
        "skipped_samples": int(skipped),
        "skip_ratio": float(skipped / total),
        "fully_valid": bool(skipped == 0),
    }
    return {"raw": raw_d, "after_keep": keep_d, "after_add": add_d}, meta


def _render_report_md(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Eval Report")
    lines.append("")
    lines.append(f"- in_json: `{summary.get('in_json', '')}`")
    lines.append(f"- num_samples: `{summary.get('num_samples', 0)}`")
    lines.append("")
    for rec in summary.get("results", []):
        thr = float(rec.get("match_thr_xy", 0.0))
        lines.append(f"## match_thr_xy = {thr:.3f}")
        raw = rec.get("raw")
        ref = rec.get("refined")
        if raw and ref:
            lines.append("")
            lines.append("| split | P | R | F1 | FP |")
            lines.append("|---|---:|---:|---:|---:|")
            lines.append(f"| raw | {raw.get('P',0):.4f} | {raw.get('R',0):.4f} | {raw.get('F1',0):.4f} | {raw.get('fp',0)} |")
            lines.append(f"| refined | {ref.get('P',0):.4f} | {ref.get('R',0):.4f} | {ref.get('F1',0):.4f} | {ref.get('fp',0)} |")
        dec = rec.get("decompose_keep_add")
        if dec:
            rr = dec["raw"]["overall"]
            rk = dec["after_keep"]["overall"]
            ra = dec["after_add"]["overall"]
            lines.append("")
            lines.append("| stage | R | FP |")
            lines.append("|---|---:|---:|")
            lines.append(f"| raw | {rr.get('R',0):.4f} | {rr.get('fp',0)} |")
            lines.append(f"| after_keep | {rk.get('R',0):.4f} | {rk.get('fp',0)} |")
            lines.append(f"| after_add | {ra.get('R',0):.4f} | {ra.get('fp',0)} |")
        lines.append("")
    return "\n".join(lines)


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
    dist_bins = _parse_bins_list(args.distance_bins, [20.0, 40.0, 60.0])
    speed_bins = _parse_bins_list(args.speed_bins, [0.2, 2.0, 6.0])

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
        raw_d = None
        ref_d = None

        use_detailed_once = bool(args.decompose_keep_add) and bool(args.eval_raw) and bool(args.eval_refined)
        if use_detailed_once:
            raw_d = evaluate_samples_detailed(
                samples,
                det_field="det",
                match_thr_xy=float(thr),
                ignore_classes=ignore,
                pred_attr_field=(args.raw_attr_field if args.raw_attr_field else None),
                distance_bins=dist_bins,
                speed_bins=speed_bins,
            )
            ref_d = evaluate_samples_detailed(
                samples,
                det_field="det_refined",
                match_thr_xy=float(thr),
                ignore_classes=ignore,
                pred_attr_field=(args.refined_attr_field if args.refined_attr_field else None),
                distance_bins=dist_bins,
                speed_bins=speed_bins,
            )
            print(
                "RAW : "
                f"P={raw_d['overall']['P']:.4f} R={raw_d['overall']['R']:.4f} F1={raw_d['overall']['F1']:.4f} | "
                f"tp={raw_d['overall']['tp']} fp={raw_d['overall']['fp']} fn={raw_d['overall']['fn']}"
            )
            print(
                "REF : "
                f"P={ref_d['overall']['P']:.4f} R={ref_d['overall']['R']:.4f} F1={ref_d['overall']['F1']:.4f} | "
                f"tp={ref_d['overall']['tp']} fp={ref_d['overall']['fp']} fn={ref_d['overall']['fn']}"
            )
        elif args.eval_raw:
            raw_m = _eval_one(
                samples=samples,
                det_field="det",
                match_thr=float(thr),
                ignore_classes=ignore,
                pred_attr_field=(args.raw_attr_field if args.raw_attr_field else None),
            )
            print("RAW :", format_metrics(raw_m))

        if (not use_detailed_once) and args.eval_refined:
            pred_attr = args.refined_attr_field if args.refined_attr_field else None
            ref_m = _eval_one(
                samples=samples,
                det_field="det_refined",
                match_thr=float(thr),
                ignore_classes=ignore,
                pred_attr_field=pred_attr,
            )
            print("REF :", format_metrics(ref_m))

        if args.print_delta:
            if use_detailed_once and raw_d is not None and ref_d is not None:
                dP = float(ref_d["overall"]["P"] - raw_d["overall"]["P"])
                dR = float(ref_d["overall"]["R"] - raw_d["overall"]["R"])
                dF1 = float(ref_d["overall"]["F1"] - raw_d["overall"]["F1"])
                print(f"DELTA(ref-raw): dP={dP:+.4f} dR={dR:+.4f} dF1={dF1:+.4f}")
            elif (raw_m is not None) and (ref_m is not None):
                dP = ref_m.P - raw_m.P
                dR = ref_m.R - raw_m.R
                dF1 = ref_m.F1 - raw_m.F1
                print(f"DELTA(ref-raw): dP={dP:+.4f} dR={dR:+.4f} dF1={dF1:+.4f}")

        rec: Dict[str, Any] = {"match_thr_xy": float(thr)}
        if use_detailed_once and raw_d is not None:
            ro = raw_d["overall"]
            rec["raw"] = {
                "P": float(ro["P"]),
                "R": float(ro["R"]),
                "F1": float(ro["F1"]),
                "tp": int(ro["tp"]),
                "fp": int(ro["fp"]),
                "fn": int(ro["fn"]),
                "num_frames": int(ro["num_frames"]),
                "num_det": int(ro["num_det"]),
                "num_gt": int(ro["num_gt"]),
                "attr_acc": float(ro["attr_acc"]),
                "attr_macro_f1": float(ro["attr_macro_f1"]),
                "attr_num_tp_with_attr": int(ro["attr_num_tp_with_attr"]),
            }
            rec["raw_detailed"] = raw_d
        elif raw_m is not None:
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
        if use_detailed_once and ref_d is not None:
            ro = ref_d["overall"]
            rec["refined"] = {
                "P": float(ro["P"]),
                "R": float(ro["R"]),
                "F1": float(ro["F1"]),
                "tp": int(ro["tp"]),
                "fp": int(ro["fp"]),
                "fn": int(ro["fn"]),
                "num_frames": int(ro["num_frames"]),
                "num_det": int(ro["num_det"]),
                "num_gt": int(ro["num_gt"]),
                "attr_acc": float(ro["attr_acc"]),
                "attr_macro_f1": float(ro["attr_macro_f1"]),
                "attr_num_tp_with_attr": int(ro["attr_num_tp_with_attr"]),
            }
            rec["refined_detailed"] = ref_d
        elif ref_m is not None:
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
        if use_detailed_once and raw_d is not None and ref_d is not None:
            rec["delta_ref_minus_raw"] = {
                "dP": float(ref_d["overall"]["P"] - raw_d["overall"]["P"]),
                "dR": float(ref_d["overall"]["R"] - raw_d["overall"]["R"]),
                "dF1": float(ref_d["overall"]["F1"] - raw_d["overall"]["F1"]),
            }
        elif (raw_m is not None) and (ref_m is not None):
            rec["delta_ref_minus_raw"] = {
                "dP": float(ref_m.P - raw_m.P),
                "dR": float(ref_m.R - raw_m.R),
                "dF1": float(ref_m.F1 - raw_m.F1),
            }
        if bool(args.decompose_keep_add) and args.eval_raw and args.eval_refined:
            tri, tri_meta = _metric_triplet(
                samples=samples,
                thr=float(thr),
                ignore=ignore,
                raw_attr_field=(args.raw_attr_field if args.raw_attr_field else None),
                refined_attr_field=(args.refined_attr_field if args.refined_attr_field else None),
                dist_bins=dist_bins,
                speed_bins=speed_bins,
            )
            rec["decompose_valid"] = bool(tri_meta["valid_samples"] > 0)
            rec["decompose_meta"] = tri_meta
            if tri_meta["valid_samples"] > 0:
                rec["decompose_keep_add"] = tri
                rec["decompose_summary"] = {
                    "R_raw": float(tri["raw"]["overall"]["R"]),
                    "R_after_keep": float(tri["after_keep"]["overall"]["R"]),
                    "R_after_add": float(tri["after_add"]["overall"]["R"]),
                    "FP_raw": int(tri["raw"]["overall"]["fp"]),
                    "FP_after_keep": int(tri["after_keep"]["overall"]["fp"]),
                    "FP_after_add": int(tri["after_add"]["overall"]["fp"]),
                }
                print(
                    "DECOMP: "
                    f"R_raw={rec['decompose_summary']['R_raw']:.4f} "
                    f"R_after_keep={rec['decompose_summary']['R_after_keep']:.4f} "
                    f"R_after_add={rec['decompose_summary']['R_after_add']:.4f} | "
                    f"FP_raw={rec['decompose_summary']['FP_raw']} "
                    f"FP_after_keep={rec['decompose_summary']['FP_after_keep']} "
                    f"FP_after_add={rec['decompose_summary']['FP_after_add']}"
                )
                if int(tri_meta["skipped_samples"]) > 0:
                    print(
                        f"DECOMP(partial): valid_samples={tri_meta['valid_samples']} "
                        f"skipped_samples={tri_meta['skipped_samples']} skip_ratio={tri_meta['skip_ratio']:.3f}"
                    )
            else:
                rec["decompose_keep_add"] = None
                rec["decompose_summary"] = None
                rec["decompose_warning"] = "All samples missing/invalid det_refined.sources; keep/add decomposition skipped."
                print("DECOMP: skipped (all samples missing/invalid det_refined.sources)")
        results.append(rec)

    summary: Dict[str, Any] = {
        "stage": "eval",
        "in_json": str(args.in_json),
        "limit_scenes": args.limit_scenes,
        "ignore_classes": sorted(list(ignore)),
        "eval_raw": bool(args.eval_raw),
        "eval_refined": bool(args.eval_refined),
        "decompose_keep_add": bool(args.decompose_keep_add),
        "distance_bins": [float(x) for x in dist_bins],
        "speed_bins": [float(x) for x in speed_bins],
        "match_thrs": [float(x) for x in thrs],
        "num_samples": int(len(samples)),
        "results": results,
        "args": vars(args),
    }

    if args.summary_json:
        out_dir = os.path.dirname(os.path.abspath(args.summary_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] wrote {args.summary_json}")
    if args.report_md:
        out_dir = os.path.dirname(os.path.abspath(args.report_md))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        md = _render_report_md(summary)
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[report] wrote {args.report_md}")


if __name__ == "__main__":
    main()
