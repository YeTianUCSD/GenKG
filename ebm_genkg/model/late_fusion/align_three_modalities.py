#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align three detection JSON files by sample_token and keep strict intersection.

Input expectations:
1) IS-Fusion file:
   {
     "scenes": [
       {"scene_token": ..., "samples": [{"sample_token": ..., "det": ..., "gt": ...}, ...]},
       ...
     ]
   }
2) Camera/Lidar file:
   [
     {"sample_token": "...", "detections": [ ... ]},
     ...
   ]

Output:
- A filtered IS-Fusion style JSON, keeping only frames that exist in all 3 modalities.
- For each kept sample, add:
    - det_isfusion (copy of original det)
    - det_camera_raw (raw detection list from camera file)
    - det_lidar_raw (raw detection list from lidar file)
- A summary JSON with alignment stats.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple


@dataclass
class DetectorLoadStats:
    rows_total: int
    rows_with_token: int
    unique_tokens: int
    duplicate_tokens: int
    max_dup_freq: int


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _detector_to_map(path: str) -> Tuple[Dict[str, List[Dict[str, Any]]], DetectorLoadStats]:
    obj = _load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"Detector JSON must be a list at root: {path}")

    token_to_dets: Dict[str, List[Dict[str, Any]]] = {}
    freq: Dict[str, int] = {}
    rows_with_token = 0

    for row in obj:
        if not isinstance(row, dict):
            continue
        tok = row.get("sample_token")
        if not isinstance(tok, str) or tok == "":
            continue
        rows_with_token += 1
        freq[tok] = int(freq.get(tok, 0) + 1)
        if tok in token_to_dets:
            # Keep first occurrence to avoid silent replacement instability.
            continue

        dets = row.get("detections", [])
        if not isinstance(dets, list):
            dets = []
        token_to_dets[tok] = dets

    duplicate_tokens = sum(1 for _, v in freq.items() if v > 1)
    max_dup_freq = max(freq.values()) if freq else 0
    stats = DetectorLoadStats(
        rows_total=len(obj),
        rows_with_token=rows_with_token,
        unique_tokens=len(token_to_dets),
        duplicate_tokens=duplicate_tokens,
        max_dup_freq=max_dup_freq,
    )
    return token_to_dets, stats


def _collect_is_tokens(root: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    scenes = root.get("scenes", [])
    if not isinstance(scenes, list):
        return out
    for sc in scenes:
        if not isinstance(sc, dict):
            continue
        samples = sc.get("samples", [])
        if not isinstance(samples, list):
            continue
        for smp in samples:
            if not isinstance(smp, dict):
                continue
            tok = smp.get("sample_token")
            if isinstance(tok, str) and tok != "":
                out.append(tok)
    return out


def _align_isfusion_root(
    is_root: Dict[str, Any],
    camera_map: Dict[str, List[Dict[str, Any]]],
    lidar_map: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    scenes = is_root.get("scenes", [])
    if not isinstance(scenes, list):
        raise ValueError("IS-Fusion JSON missing list field: scenes")

    keep_tokens = set(camera_map.keys()) & set(lidar_map.keys())
    total_samples = 0
    kept_samples = 0
    dropped_missing_any = 0
    dropped_tokens: List[str] = []
    out_scenes: List[Dict[str, Any]] = []

    for sc in scenes:
        if not isinstance(sc, dict):
            continue
        samples = sc.get("samples", [])
        if not isinstance(samples, list):
            samples = []

        kept_scene_samples: List[Dict[str, Any]] = []
        for smp in samples:
            if not isinstance(smp, dict):
                continue
            total_samples += 1
            tok = smp.get("sample_token")
            if not isinstance(tok, str) or tok == "":
                dropped_missing_any += 1
                continue
            if tok not in keep_tokens:
                dropped_missing_any += 1
                dropped_tokens.append(tok)
                continue

            new_smp = copy.deepcopy(smp)
            new_smp["det_isfusion"] = copy.deepcopy(new_smp.get("det", {}))
            new_smp["det_camera_raw"] = copy.deepcopy(camera_map.get(tok, []))
            new_smp["det_lidar_raw"] = copy.deepcopy(lidar_map.get(tok, []))
            kept_scene_samples.append(new_smp)
            kept_samples += 1

        if kept_scene_samples:
            new_sc = copy.deepcopy(sc)
            new_sc["samples"] = kept_scene_samples
            new_sc["num_samples"] = len(kept_scene_samples)
            out_scenes.append(new_sc)

    out_root = copy.deepcopy(is_root)
    out_root["scenes"] = out_scenes
    if isinstance(out_root.get("scene_count"), int):
        out_root["scene_count"] = len(out_scenes)
    out_root["alignment_meta"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "strict_intersection_by_sample_token",
        "added_fields_per_sample": ["det_isfusion", "det_camera_raw", "det_lidar_raw"],
    }

    summary = {
        "total_samples_in_isfusion": int(total_samples),
        "kept_samples": int(kept_samples),
        "dropped_missing_any_modality": int(dropped_missing_any),
        "kept_ratio": float(kept_samples / max(1, total_samples)),
        "total_scenes_in_isfusion": int(len(scenes)),
        "kept_scenes": int(len(out_scenes)),
        "dropped_sample_tokens": sorted(set(dropped_tokens)),
    }
    return out_root, summary


def _default_output_name(isfusion_path: str) -> str:
    stem = os.path.splitext(os.path.basename(isfusion_path))[0]
    return f"{stem}_3_modality_aligned.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Align IS-Fusion + camera + lidar JSON by sample_token.")
    p.add_argument(
        "--isfusion_json",
        type=str,
        default="/home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json",
        help="IS-Fusion JSON with scenes/samples and GT.",
    )
    p.add_argument(
        "--camera_json",
        type=str,
        default="/home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/camera/nuscenes_val_pre.json",
        help="Camera detector JSON (list root).",
    )
    p.add_argument(
        "--lidar_json",
        type=str,
        default="/home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/lidar/nuscenes_val_focalformer.json",
        help="Lidar detector JSON (list root).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/model/late_fusion/output",
        help="Directory for aligned outputs.",
    )
    p.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional output JSON filename. Default: <isfusion_stem>_3_modality_aligned.json",
    )
    p.add_argument(
        "--summary_name",
        type=str,
        default=None,
        help="Optional summary JSON filename. Default: <output_stem>.summary.json",
    )
    p.add_argument("--indent", type=int, default=2)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    _ensure_dir(args.output_dir)
    out_name = args.output_name if args.output_name else _default_output_name(args.isfusion_json)
    out_path = os.path.join(args.output_dir, out_name)
    out_stem = os.path.splitext(out_name)[0]
    summary_name = args.summary_name if args.summary_name else f"{out_stem}.summary.json"
    summary_path = os.path.join(args.output_dir, summary_name)

    print("[1/4] Loading camera detections...")
    camera_map, camera_stats = _detector_to_map(args.camera_json)
    print(f"  camera tokens={camera_stats.unique_tokens} rows={camera_stats.rows_total}")

    print("[2/4] Loading lidar detections...")
    lidar_map, lidar_stats = _detector_to_map(args.lidar_json)
    print(f"  lidar tokens={lidar_stats.unique_tokens} rows={lidar_stats.rows_total}")

    print("[3/4] Loading IS-Fusion root and aligning...")
    is_root = _load_json(args.isfusion_json)
    if not isinstance(is_root, dict):
        raise ValueError("IS-Fusion JSON root must be an object/dict.")
    is_tokens = _collect_is_tokens(is_root)

    out_root, align_summary = _align_isfusion_root(
        is_root=is_root,
        camera_map=camera_map,
        lidar_map=lidar_map,
    )

    print("[4/4] Writing outputs...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_root, f, ensure_ascii=False, indent=int(args.indent))

    inter_all = len(set(is_tokens) & set(camera_map.keys()) & set(lidar_map.keys()))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "isfusion_json": os.path.abspath(args.isfusion_json),
            "camera_json": os.path.abspath(args.camera_json),
            "lidar_json": os.path.abspath(args.lidar_json),
        },
        "outputs": {
            "aligned_json": os.path.abspath(out_path),
            "summary_json": os.path.abspath(summary_path),
        },
        "stats": {
            "isfusion_tokens_total": len(is_tokens),
            "camera_tokens_total": len(camera_map),
            "lidar_tokens_total": len(lidar_map),
            "intersection_3_modalities": int(inter_all),
            "camera_loader": camera_stats.__dict__,
            "lidar_loader": lidar_stats.__dict__,
            **align_summary,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=int(args.indent))

    print(f"  aligned_json: {out_path}")
    print(f"  summary_json: {summary_path}")
    print(
        "  kept_samples="
        f"{summary['stats']['kept_samples']}/{summary['stats']['total_samples_in_isfusion']} "
        f"(intersection={summary['stats']['intersection_3_modalities']})"
    )


if __name__ == "__main__":
    main()
