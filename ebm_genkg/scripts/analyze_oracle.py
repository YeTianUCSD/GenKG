#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oracle recall analysis for candidate construction quality.

Report:
- raw recall upper bound (raw dt=0 candidates only)
- candidate oracle recall upper bound (raw + warp candidates)
- warp contribution (oracle - raw)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Set

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from json_io import load_root_and_samples, group_by_scene
from data import CandidateConfig, build_frames_by_scene, build_candidate_gt_links_xy


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    flag = "--" + name.replace("_", "-")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return
    group = parser.add_mutually_exclusive_group()
    group.add_argument(flag, dest=name, action="store_true", help=help_text)
    group.add_argument("--no-" + name.replace("_", "-"), dest=name, action="store_false", help=f"Disable {help_text}")
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
    p = argparse.ArgumentParser(description="Analyze oracle recall of raw/all candidates.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--in_json", type=str, default=cfg.get("in_json"), required=("in_json" not in cfg))
    p.add_argument("--summary_json", type=str, default=cfg.get("summary_json"))
    _add_bool_arg(p, "require_gt", bool(cfg.get("require_gt", True)), "Require GT while loading")
    p.add_argument("--limit_scenes", type=int, default=cfg.get("limit_scenes"))
    p.add_argument("--match_thr", type=float, default=float(cfg.get("match_thr", 2.0)))
    p.add_argument("--ignore_classes", type=str, default=str(cfg.get("ignore_classes", "-1")))
    _add_bool_arg(p, "use_warp", bool(cfg.get("use_warp", True)), "Use warp candidates in oracle pool")
    p.add_argument("--warp_radius", type=int, default=int(cfg.get("warp_radius", 2)))
    p.add_argument("--warp_topk", type=int, default=int(cfg.get("warp_topk", 200)))
    p.add_argument("--warp_decay", type=float, default=float(cfg.get("warp_decay", 0.9)))
    p.add_argument("--raw_score_min", type=float, default=float(cfg.get("raw_score_min", 0.05)))
    p.add_argument("--max_per_frame", type=int, default=cfg.get("max_per_frame", 180))
    return p


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

    cand_cfg = CandidateConfig(
        raw_score_min=float(args.raw_score_min),
        max_per_frame=args.max_per_frame,
        use_warp=bool(args.use_warp),
        warp_radius=int(args.warp_radius),
        warp_topk=int(args.warp_topk),
        warp_score_decay=float(args.warp_decay),
        ignore_classes=ignore,
        match_thr_xy=float(args.match_thr),
    )
    frames_by_scene = build_frames_by_scene(scenes, cand_cfg)

    gt_total = 0
    raw_cov = 0
    all_cov = 0
    frames = 0
    for _, frs in frames_by_scene.items():
        for fr in frs:
            frames += 1
            gtb = fr.gt_boxes[:, :3].astype(np.float32, copy=False)
            gtl = fr.gt_labels.astype(np.int64, copy=False)
            gt_total += int(gtb.shape[0])
            if gtb.shape[0] == 0:
                continue

            raw = [c for c in fr.candidates if c.source == "raw" and c.from_dt == 0]
            all_cands = list(fr.candidates)

            if len(raw) > 0:
                rxyz = np.stack([c.box for c in raw], axis=0).astype(np.float32)[:, :3]
                rlab = np.asarray([int(c.label) for c in raw], dtype=np.int64)
                rscore = np.asarray([float(c.score) for c in raw], dtype=np.float32)
                rlinks = build_candidate_gt_links_xy(rxyz, rlab, gtb, gtl, thr_xy=float(args.match_thr), det_scores=rscore)
                raw_cov += int(np.count_nonzero(rlinks.gt_best_cand >= 0))

            if len(all_cands) > 0:
                axyz = np.stack([c.box for c in all_cands], axis=0).astype(np.float32)[:, :3]
                alab = np.asarray([int(c.label) for c in all_cands], dtype=np.int64)
                ascore = np.asarray([float(c.score) for c in all_cands], dtype=np.float32)
                alinks = build_candidate_gt_links_xy(axyz, alab, gtb, gtl, thr_xy=float(args.match_thr), det_scores=ascore)
                all_cov += int(np.count_nonzero(alinks.gt_best_cand >= 0))

    raw_recall = float(raw_cov / max(1, gt_total))
    all_recall = float(all_cov / max(1, gt_total))
    out = {
        "num_frames": int(frames),
        "num_gt": int(gt_total),
        "raw_oracle_cover": int(raw_cov),
        "all_oracle_cover": int(all_cov),
        "raw_oracle_recall": raw_recall,
        "all_oracle_recall": all_recall,
        "warp_gain_recall": float(all_recall - raw_recall),
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
