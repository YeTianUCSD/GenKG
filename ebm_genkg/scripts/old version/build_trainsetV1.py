#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build training dataset for learned EBM unary terms.

Output:
- NPZ fields:
  X            float32 [N, D]
  y_keep       int64   [N]   (0/1)
  y_cls        int64   [N]   (matched class, -100 for unmatched)
  y_attr       int64   [N]   (attr id, -1 unknown/unmatched)
  cand_label   int64   [N]   (candidate original label)
  cand_score   float32 [N]
  from_dt      int32   [N]
  scene_idx    int32   [N]
  frame_idx    int32   [N]
  feature_names object [D]

- Meta JSON:
  dataset stats + attr vocab + args snapshot.

Example:
  python scripts/build_trainset.py \
    --in_json /path/to/train.json \
    --out_npz /path/to/trainset.npz \
    --use_warp --warp_radius 2 --warp_topk 120 --warp_decay 0.9 \
    --target_candidates all \
    --match_thr 2.0 --ignore_classes -1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from json_io import load_root_and_samples, group_by_scene
from data import (
    Candidate,
    CandidateConfig,
    build_frames_by_scene,
    match_all_candidates_to_gt,
    match_raw_det_to_gt,
)


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


def _cand_speed(c: Candidate) -> float:
    b = np.asarray(c.box, dtype=np.float32)
    if b.shape[0] < 9:
        return 0.0
    vx, vy = float(b[7]), float(b[8])
    if not (np.isfinite(vx) and np.isfinite(vy)):
        return 0.0
    return float(np.hypot(vx, vy))


def _source_flags(c: Candidate) -> Tuple[float, float, float]:
    src = str(getattr(c, "source", ""))
    dt = int(getattr(c, "from_dt", 0))
    is_raw = 1.0 if (src == "raw" and dt == 0) else 0.0
    is_warp = 1.0 if (src.startswith("warp(") or dt != 0) else 0.0
    is_other = 1.0 - min(1.0, is_raw + is_warp)
    return is_raw, is_warp, is_other


def _support_stats(
    cands: List[Candidate],
    cell: float,
) -> Tuple[
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int], int],
]:
    """
    Return:
      - count_map: keyed by (label, gx, gy)
      - unique_dt_count_map: keyed by (label, gx, gy)
      - local_density_map: keyed by (gx, gy), counts all labels in the same cell
    """
    cs = float(max(cell, 1e-6))
    count_map: Dict[Tuple[int, int, int], int] = {}
    dt_map: Dict[Tuple[int, int, int], Set[int]] = {}
    local_density_map: Dict[Tuple[int, int], int] = {}

    for c in cands:
        b = np.asarray(c.box, dtype=np.float32)
        x, y = float(b[0]), float(b[1])
        gx = int(np.round(x / cs))
        gy = int(np.round(y / cs))
        k = (int(c.label), gx, gy)
        count_map[k] = count_map.get(k, 0) + 1
        dt_map.setdefault(k, set()).add(int(c.from_dt))
        k2 = (gx, gy)
        local_density_map[k2] = local_density_map.get(k2, 0) + 1

    unique_dt_count = {k: len(v) for k, v in dt_map.items()}
    return count_map, unique_dt_count, local_density_map


def _feature_row(
    c: Candidate,
    count_map: Dict[Tuple[int, int, int], int],
    unique_dt_count_map: Dict[Tuple[int, int, int], int],
    local_density_map: Dict[Tuple[int, int], int],
    cell: float,
) -> List[float]:
    b = np.asarray(c.box, dtype=np.float32)
    if b.shape[0] < 9:
        b = np.pad(b, (0, max(0, 9 - b.shape[0])), mode="constant")

    score = float(c.score) if np.isfinite(float(c.score)) else 0.0
    label = float(int(c.label))
    from_dt = float(int(c.from_dt))
    abs_dt = float(abs(int(c.from_dt)))
    speed = _cand_speed(c)

    is_raw, is_warp, is_other = _source_flags(c)

    x, y, z, dx, dy, dz, yaw, vx, vy = [float(v) if np.isfinite(float(v)) else 0.0 for v in b[:9]]

    cs = float(max(cell, 1e-6))
    gx = int(np.round(x / cs))
    gy = int(np.round(y / cs))
    k = (int(c.label), gx, gy)
    support_count = float(count_map.get(k, 1))
    support_unique_dt = float(unique_dt_count_map.get(k, 1))
    temporal_stability = float(support_unique_dt / max(1.0, support_count))
    local_density = float(local_density_map.get((gx, gy), 1))

    return [
        score,
        label,
        is_raw,
        is_warp,
        is_other,
        from_dt,
        abs_dt,
        speed,
        x,
        y,
        z,
        dx,
        dy,
        dz,
        yaw,
        vx,
        vy,
        support_count,
        support_unique_dt,
        temporal_stability,
        local_density,
    ]


def _feature_names() -> List[str]:
    return [
        "score",
        "label",
        "is_raw",
        "is_warp",
        "is_other",
        "from_dt",
        "abs_dt",
        "speed",
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
        "yaw",
        "vx",
        "vy",
        "support_count",
        "support_unique_dt",
        "temporal_stability",
        "local_density",
    ]


def _build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build NPZ training set from scene JSON for EBM training.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")

    p.add_argument("--in_json", type=str, default=cfg.get("in_json"), required=("in_json" not in cfg))
    p.add_argument("--out_npz", type=str, default=cfg.get("out_npz"), required=("out_npz" not in cfg))
    p.add_argument("--out_meta", type=str, default=cfg.get("out_meta"), help="Optional meta json path.")

    p.add_argument("--require_gt", action="store_true", default=bool(cfg.get("require_gt", True)))
    p.add_argument("--limit_scenes", type=int, default=cfg.get("limit_scenes"))

    p.add_argument("--target_candidates", type=str, default=str(cfg.get("target_candidates", "all")),
                   choices=["all", "raw"], help="Which candidates to export/match as training targets.")

    p.add_argument("--match_thr", type=float, default=float(cfg.get("match_thr", 2.0)))
    p.add_argument("--ignore_classes", type=str, default=str(cfg.get("ignore_classes", "-1")))

    p.add_argument("--use_warp", action="store_true", default=bool(cfg.get("use_warp", False)))
    p.add_argument("--warp_radius", type=int, default=int(cfg.get("warp_radius", 2)))
    p.add_argument("--warp_topk", type=int, default=int(cfg.get("warp_topk", 200)))
    p.add_argument("--warp_decay", type=float, default=float(cfg.get("warp_decay", 0.9)))
    p.add_argument("--raw_score_min", type=float, default=float(cfg.get("raw_score_min", 0.0)))
    p.add_argument("--max_per_frame", type=int, default=cfg.get("max_per_frame"))

    p.add_argument("--support_cell", type=float, default=float(cfg.get("support_cell", 1.0)),
                   help="Cell size for support_count/support_unique_dt features.")

    return p


def main() -> None:
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default=None)
    a0, _ = p0.parse_known_args()

    cfg = _load_cfg(a0.config)
    parser = _build_parser(cfg)
    args = parser.parse_args()

    ignore = _parse_ignore(args.ignore_classes)

    out_npz = os.path.abspath(args.out_npz)
    out_meta = os.path.abspath(args.out_meta) if args.out_meta else (os.path.splitext(out_npz)[0] + ".meta.json")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)

    print("[1/4] Loading & grouping samples...")
    lr = load_root_and_samples(args.in_json, require_gt=bool(args.require_gt))
    scenes = group_by_scene(lr.samples)
    if args.limit_scenes is not None:
        keys = sorted(list(scenes.keys()))[: int(args.limit_scenes)]
        scenes = {k: scenes[k] for k in keys}
        print(f"  limit_scenes={args.limit_scenes} -> using scenes={len(scenes)}")
    print(f"  samples_loaded={len(lr.samples)} scenes_used={len(scenes)}")

    print("[2/4] Building candidates...")
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

    print("[3/4] Extracting features & targets...")
    X: List[List[float]] = []
    y_keep: List[int] = []
    y_cls: List[int] = []
    y_attr_str: List[str] = []
    cand_label: List[int] = []
    cand_score: List[float] = []
    from_dt: List[int] = []
    scene_idx: List[int] = []
    frame_idx: List[int] = []

    attr_counter: Counter = Counter()
    total_frames = 0
    total_candidates = 0
    total_pos = 0

    scene_keys = sorted(list(frames_by_scene.keys()))
    for si, sc in enumerate(scene_keys):
        frames = frames_by_scene[sc]
        for fr in frames:
            total_frames += 1

            if args.target_candidates == "all":
                cands = list(fr.candidates)
                mt = match_all_candidates_to_gt(fr, cand_cfg)
            else:
                cands = [c for c in fr.candidates if c.source == "raw" and c.from_dt == 0]
                mt = match_raw_det_to_gt(fr, cand_cfg)

            if len(cands) == 0:
                continue

            if len(cands) != len(mt.keep):
                n = min(len(cands), len(mt.keep))
                cands = cands[:n]
                keep_arr = mt.keep[:n]
                cls_arr = mt.cls_tgt[:n]
                attr_list = mt.attr_tgt[:n]
            else:
                keep_arr = mt.keep
                cls_arr = mt.cls_tgt
                attr_list = mt.attr_tgt

            count_map, unique_dt_count_map, local_density_map = _support_stats(cands, cell=float(args.support_cell))

            for i, c in enumerate(cands):
                feat = _feature_row(
                    c,
                    count_map,
                    unique_dt_count_map,
                    local_density_map,
                    cell=float(args.support_cell),
                )
                X.append(feat)

                k = int(keep_arr[i])
                cl = int(cls_arr[i])
                at = str(attr_list[i]) if attr_list[i] is not None else ""

                y_keep.append(k)
                y_cls.append(cl)
                y_attr_str.append(at)
                cand_label.append(int(c.label))
                cand_score.append(float(c.score))
                from_dt.append(int(c.from_dt))
                scene_idx.append(int(si))
                frame_idx.append(int(fr.index_in_scene))

                total_candidates += 1
                total_pos += int(k)
                if at != "":
                    attr_counter[at] += 1

    # attr vocab: -1 reserved for unknown/empty
    attr_vocab: Dict[str, int] = {"": -1}
    next_id = 0
    for a, _ in attr_counter.most_common():
        if a not in attr_vocab:
            attr_vocab[a] = next_id
            next_id += 1
    y_attr = np.array([attr_vocab.get(a, -1) for a in y_attr_str], dtype=np.int64)

    X_arr = np.asarray(X, dtype=np.float32)
    y_keep_arr = np.asarray(y_keep, dtype=np.int64)
    y_cls_arr = np.asarray(y_cls, dtype=np.int64)

    print("[4/4] Saving dataset...")
    np.savez_compressed(
        out_npz,
        X=X_arr,
        y_keep=y_keep_arr,
        y_cls=y_cls_arr,
        y_attr=y_attr,
        cand_label=np.asarray(cand_label, dtype=np.int64),
        cand_score=np.asarray(cand_score, dtype=np.float32),
        from_dt=np.asarray(from_dt, dtype=np.int32),
        scene_idx=np.asarray(scene_idx, dtype=np.int32),
        frame_idx=np.asarray(frame_idx, dtype=np.int32),
        feature_names=np.array(_feature_names(), dtype=object),
    )

    meta = {
        "in_json": str(args.in_json),
        "out_npz": out_npz,
        "target_candidates": str(args.target_candidates),
        "ignore_classes": sorted(list(ignore)),
        "limit_scenes": args.limit_scenes,
        "num_samples_loaded": int(len(lr.samples)),
        "num_scenes_used": int(len(scenes)),
        "num_frames_used": int(total_frames),
        "num_rows": int(X_arr.shape[0]),
        "num_features": int(X_arr.shape[1] if X_arr.ndim == 2 else 0),
        "num_positive": int(total_pos),
        "pos_rate": float(total_pos / max(1, total_candidates)),
        "feature_names": _feature_names(),
        "attr_vocab": attr_vocab,
        "args": vars(args),
    }

    with open(out_meta, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  npz: {out_npz}")
    print(f"  meta: {out_meta}")
    print(f"  rows={meta['num_rows']} pos_rate={meta['pos_rate']:.4f} features={meta['num_features']}")


if __name__ == "__main__":
    main()
