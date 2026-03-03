#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/read_lidar_json.py \
  --json_path /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/output/lidar__score0p1.json \
  --log_path  /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/output/lidar_detection__score0p1.log
'''


import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Any:
    # Load a JSON file from disk
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_records(obj: Any) -> List[Dict[str, Any]]:
    # Ensure the top-level JSON is a list of dict records
    if not isinstance(obj, list):
        raise TypeError(f"Expected top-level JSON to be a list, but got: {type(obj)}")
    if len(obj) == 0:
        raise ValueError("Top-level JSON list is empty.")
    if not all(isinstance(x, dict) for x in obj):
        bad = next((type(x) for x in obj if not isinstance(x, dict)), None)
        raise TypeError(f"Top-level JSON list contains a non-dict element: {bad}")
    return obj


def setup_logger(log_path: Path) -> logging.Logger:
    # Create a file logger (overwrite mode)
    logger = logging.getLogger("lidar_dump_and_stats")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def dump_first_record(logger: logging.Logger, records: List[Dict[str, Any]]) -> None:
    # Dump the first record to the log file as pretty JSON
    pretty = json.dumps(records[0], ensure_ascii=False, indent=2)
    logger.info("First record dump begin")
    logger.info("\n%s", pretty)
    logger.info("First record dump end")


def compute_stats(records: List[Dict[str, Any]]) -> Tuple[int, float]:
    # Aggregate detection counts per sample_token (merge if duplicated tokens exist)
    token_to_detcount: Dict[str, int] = {}

    for rec in records:
        token = rec.get("sample_token", None)
        if token is None:
            # Skip records without sample_token
            continue

        dets = rec.get("detections", [])
        det_count = len(dets) if isinstance(dets, list) else 0

        token_to_detcount[token] = token_to_detcount.get(token, 0) + det_count

    num_samples = len(token_to_detcount)
    avg_det = (sum(token_to_detcount.values()) / num_samples) if num_samples > 0 else 0.0
    return num_samples, avg_det


def main():
    parser = argparse.ArgumentParser(description="Dump first LiDAR record and compute detection statistics.")
    parser.add_argument("--json_path", required=True, type=str, help="Path to the LiDAR JSON file")
    parser.add_argument("--log_path", required=True, type=str, help="Path to output log file")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    log_path = Path(args.log_path)

    obj = load_json(json_path)
    records = ensure_records(obj)

    logger = setup_logger(log_path)

    # 1) Dump first record
    dump_first_record(logger, records)

    # 2) Compute stats
    num_samples, avg_det = compute_stats(records)

    # English output
    print(f"Number of samples (unique sample_token): {num_samples}")
    print(f"Average #detections per sample: {avg_det:.3f}")
    print(f"Log file written to: {log_path}")

    # Also write stats into log
    logger.info("Number of samples (unique sample_token): %d", num_samples)
    logger.info("Average #detections per sample: %.3f", avg_det)


if __name__ == "__main__":
    main()