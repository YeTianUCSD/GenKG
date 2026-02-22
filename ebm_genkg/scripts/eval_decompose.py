#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decompose refined gains into keep/add/filter contributions.

Requires input JSON containing both:
- det (raw)
- det_refined (refined)

If det_refined.sources exists and aligns with detections, also reports TP source split:
- tp_from_raw_source
- tp_from_warp_source
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
from metrics import parse_det_from_sample, parse_gt_from_sample, apply_ignore_classes, class_aware_greedy_match


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


def _parse_ignore(s: str) -> Set[int]:
    out: Set[int] = set()
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.add(int(x))
    return out


def _load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if ext in (".yaml", ".yml"):
        import yaml  # type: ignore
        obj = yaml.safe_load(txt)
    else:
        obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("config root must be dict")
    return obj


def _build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Decompose refined gains into add/filter parts.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--in_json", type=str, default=cfg.get("in_json"), required=("in_json" not in cfg))
    p.add_argument("--summary_json", type=str, default=cfg.get("summary_json"))
    _add_bool_arg(p, "require_gt", bool(cfg.get("require_gt", True)), "Require GT while loading samples")
    p.add_argument("--limit_scenes", type=int, default=cfg.get("limit_scenes"))
    p.add_argument("--match_thr", type=float, default=float(cfg.get("match_thr", 2.0)))
    p.add_argument("--ignore_classes", type=str, default=str(cfg.get("ignore_classes", "-1")))
    return p


def _is_warp_source(src: str) -> bool:
    s = str(src).lower()
    return ("warp" in s) or ("dt=" in s and "dt=0" not in s and "dt=+0" not in s and "dt=-0" not in s)


def main() -> None:
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default=None)
    a0, _ = p0.parse_known_args()
    cfg = _load_cfg(a0.config)
    args = _build_parser(cfg).parse_args()

    ignore = _parse_ignore(args.ignore_classes)
    lr = load_root_and_samples(args.in_json, require_gt=bool(args.require_gt))
    scenes = group_by_scene(lr.samples)
    if args.limit_scenes is not None:
        keys = sorted(list(scenes.keys()))[: int(args.limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
    samples: List[Dict[str, Any]] = []
    for _, fs in scenes.items():
        samples.extend(fs)

    gt_total = 0
    raw_tp = raw_fp = raw_fn = 0
    ref_tp = ref_fp = ref_fn = 0
    add_gt = 0
    lost_gt = 0
    ref_tp_from_raw_source = 0
    ref_tp_from_warp_source = 0
    ref_tp_source_known = 0

    for s in samples:
        r_boxes, r_scores, r_labels = parse_det_from_sample(s, det_field="det")
        f_boxes, f_scores, f_labels = parse_det_from_sample(s, det_field="det_refined")
        g_boxes, g_labels, _ = parse_gt_from_sample(s)

        r_boxes, r_labels, r_scores, _ = apply_ignore_classes(r_boxes, r_labels, r_scores, None, ignore)
        f_boxes, f_labels, f_scores, _ = apply_ignore_classes(f_boxes, f_labels, f_scores, None, ignore)
        g_boxes, g_labels, _, _ = apply_ignore_classes(g_boxes, g_labels, None, None, ignore)

        mr_raw = class_aware_greedy_match(
            det_xyz=r_boxes[:, :3], det_labels=r_labels,
            gt_xyz=g_boxes[:, :3], gt_labels=g_labels,
            thr_xy=float(args.match_thr),
        )
        mr_ref = class_aware_greedy_match(
            det_xyz=f_boxes[:, :3], det_labels=f_labels,
            gt_xyz=g_boxes[:, :3], gt_labels=g_labels,
            thr_xy=float(args.match_thr),
        )

        raw_tp += int(mr_raw.tp)
        raw_fp += int(mr_raw.fp)
        raw_fn += int(mr_raw.fn)
        ref_tp += int(mr_ref.tp)
        ref_fp += int(mr_ref.fp)
        ref_fn += int(mr_ref.fn)
        gt_total += int(g_labels.shape[0])

        raw_matched = set([int(x) for x in mr_raw.det_to_gt.tolist() if int(x) >= 0])
        ref_matched = set([int(x) for x in mr_ref.det_to_gt.tolist() if int(x) >= 0])
        add_gt += int(len(ref_matched - raw_matched))
        lost_gt += int(len(raw_matched - ref_matched))

        det_refined = s.get("det_refined", {}) or {}
        srcs = det_refined.get("sources", None)
        if isinstance(srcs, list) and len(srcs) == int(f_labels.shape[0]):
            for i_det, j_gt in enumerate(mr_ref.det_to_gt.tolist()):
                if int(j_gt) < 0:
                    continue
                ref_tp_source_known += 1
                if _is_warp_source(str(srcs[i_det])):
                    ref_tp_from_warp_source += 1
                else:
                    ref_tp_from_raw_source += 1

    out = {
        "num_samples": int(len(samples)),
        "num_gt": int(gt_total),
        "match_thr_xy": float(args.match_thr),
        "raw": {
            "tp": int(raw_tp),
            "fp": int(raw_fp),
            "fn": int(raw_fn),
            "P": float(raw_tp / max(1, raw_tp + raw_fp)),
            "R": float(raw_tp / max(1, raw_tp + raw_fn)),
        },
        "refined": {
            "tp": int(ref_tp),
            "fp": int(ref_fp),
            "fn": int(ref_fn),
            "P": float(ref_tp / max(1, ref_tp + ref_fp)),
            "R": float(ref_tp / max(1, ref_tp + ref_fn)),
        },
        "decompose": {
            "add_gt_covered": int(add_gt),
            "lost_gt_covered": int(lost_gt),
            "add_recall_gain": float(add_gt / max(1, gt_total)),
            "lost_recall_drop": float(lost_gt / max(1, gt_total)),
            "fp_reduced": int(max(0, raw_fp - ref_fp)),
            "fp_increased": int(max(0, ref_fp - raw_fp)),
        },
        "refined_tp_source_split": {
            "tp_source_known": int(ref_tp_source_known),
            "tp_from_raw_source": int(ref_tp_from_raw_source),
            "tp_from_warp_source": int(ref_tp_from_warp_source),
            "raw_source_ratio": float(ref_tp_from_raw_source / max(1, ref_tp_source_known)),
            "warp_source_ratio": float(ref_tp_from_warp_source / max(1, ref_tp_source_known)),
        },
        "args": vars(args),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[summary] wrote {args.summary_json}")


if __name__ == "__main__":
    main()
