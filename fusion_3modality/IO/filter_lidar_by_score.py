#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/filter_lidar_by_score.py \
  --json_path /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/lidar/nuscenes_val_focalformer.json \
  --out_path  /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/output/lidar__score0p05.json \
  --score_thr 0.05
'''
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


JsonType = Union[List[Dict[str, Any]], Dict[str, Any]]


def load_json(path: Path) -> JsonType:
    # Load a JSON file from disk
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    # Save JSON to disk with pretty formatting
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_records(obj: JsonType) -> Tuple[List[Dict[str, Any]], bool]:
    # Normalize JSON to a list of dict records; return (records, was_single_dict)
    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            bad = next((type(x) for x in obj if not isinstance(x, dict)), None)
            raise TypeError(f"Top-level list contains a non-dict element: {bad}")
        return obj, False

    if isinstance(obj, dict):
        return [obj], True

    raise TypeError(f"Unsupported top-level JSON type: {type(obj)}")


def get_score(det: Dict[str, Any], score_key: str) -> float:
    # Extract score value from a detection dict
    v = det.get(score_key, None)
    if isinstance(v, (int, float)):
        return float(v)
    return float("-inf")


def filter_detections(records: List[Dict[str, Any]], score_thr: float, score_key: str) -> Tuple[int, int, int]:
    # Filter detections in-place; return (num_records, total_before, total_after)
    total_before = 0
    total_after = 0

    for rec in records:
        dets = rec.get("detections", None)
        if not isinstance(dets, list):
            continue

        total_before += len(dets)

        kept = []
        for det in dets:
            if not isinstance(det, dict):
                continue
            s = get_score(det, score_key)
            # Keep detections with score >= threshold
            if s >= score_thr:
                kept.append(det)

        rec["detections"] = kept
        total_after += len(kept)

    return len(records), total_before, total_after


def unique_sample_tokens(records: List[Dict[str, Any]]) -> int:
    # Count unique sample_token across records
    tokens = set()
    for rec in records:
        t = rec.get("sample_token", None)
        if isinstance(t, str) and t:
            tokens.add(t)
    return len(tokens)


def main():
    parser = argparse.ArgumentParser(description="Filter detection results by score threshold and save JSON.")
    parser.add_argument("--json_path", required=True, type=str, help="Path to input JSON file")
    parser.add_argument("--out_path", required=True, type=str, help="Path to output filtered JSON file")
    parser.add_argument("--score_thr", default=0.5, type=float, help="Score threshold (default: 0.5)")
    parser.add_argument("--score_key", default="detection_score", type=str, help="Score field name (default: detection_score)")
    args = parser.parse_args()

    in_path = Path(args.json_path)
    out_path = Path(args.out_path)

    obj = load_json(in_path)
    records, was_single = normalize_records(obj)

    n_samples_before = unique_sample_tokens(records)

    n_records, total_before, total_after = filter_detections(records, args.score_thr, args.score_key)

    # Restore original top-level structure
    out_obj: Any = records[0] if was_single else records
    save_json(out_obj, out_path)

    n_samples_after = unique_sample_tokens(records)
    avg_before = (total_before / n_samples_before) if n_samples_before > 0 else 0.0
    avg_after = (total_after / n_samples_after) if n_samples_after > 0 else 0.0

    # English outputs
    print(f"Input file: {in_path}")
    print(f"Output file: {out_path}")
    print(f"Score key: {args.score_key}")
    print(f"Score threshold: {args.score_thr}")
    print(f"Records processed: {n_records}")
    print(f"Unique samples (before): {n_samples_before}")
    print(f"Unique samples (after):  {n_samples_after}")
    print(f"Detections kept: {total_after} / {total_before} ({(total_after/total_before*100.0) if total_before>0 else 0.0:.2f}%)")
    print(f"Average detections per sample (before): {avg_before:.3f}")
    print(f"Average detections per sample (after):  {avg_after:.3f}")


if __name__ == "__main__":
    main()