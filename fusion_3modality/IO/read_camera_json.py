#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/read_camera_json.py \
  --json_path /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/output/camera__score0p1.json \
  --log_path  /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/output/camera_detection__score0p1.log
'''

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple


JsonType = Union[List[Dict[str, Any]], Dict[str, Any]]


def load_json(path: Path) -> Any:
    # Load JSON file from disk
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_records(obj: Any) -> List[Dict[str, Any]]:
    # Normalize input JSON to a list of dict records
    if isinstance(obj, list):
        if len(obj) == 0:
            raise ValueError("Top-level JSON is a list but empty.")
        if not all(isinstance(x, dict) for x in obj):
            bad = next((type(x) for x in obj if not isinstance(x, dict)), None)
            raise TypeError(f"Top-level JSON is a list but contains non-dict element: {bad}")
        return obj

    if isinstance(obj, dict):
        return [obj]

    raise TypeError(f"Unsupported top-level JSON type: {type(obj)}")


def setup_logger(log_path: Path) -> logging.Logger:
    # Setup a simple file logger
    logger = logging.getLogger("camera_json_stats")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def dump_first_record(logger: logging.Logger, first_record: Dict[str, Any]) -> None:
    # Pretty dump the first record into log
    pretty = json.dumps(first_record, ensure_ascii=False, indent=2)
    logger.info("First record dump begin")
    logger.info("\n%s", pretty)
    logger.info("First record dump end")


def compute_stats(records: List[Dict[str, Any]]) -> Tuple[int, float, Dict[str, int]]:
    # Aggregate detections count per sample_token (merge if duplicated tokens exist)
    token_to_detcount: Dict[str, int] = {}

    for rec in records:
        token = rec.get("sample_token", None)
        if token is None:
            # Skip records without sample_token
            continue

        dets = rec.get("detections", [])
        # If detections is missing or not a list, treat as 0
        det_count = len(dets) if isinstance(dets, list) else 0

        token_to_detcount[token] = token_to_detcount.get(token, 0) + det_count

    num_samples = len(token_to_detcount)
    avg_det = (sum(token_to_detcount.values()) / num_samples) if num_samples > 0 else 0.0
    return num_samples, avg_det, token_to_detcount


def main():
    parser = argparse.ArgumentParser(
        description="Dump first record and compute sample_token/detection statistics."
    )
    parser.add_argument("--json_path", required=True, type=str, help="Path to input JSON file")
    parser.add_argument("--log_path", required=True, type=str, help="Path to output log file")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    log_path = Path(args.log_path)

    obj = load_json(json_path)
    records = normalize_records(obj)

    logger = setup_logger(log_path)

    # Dump first record
    dump_first_record(logger, records[0])

    # Compute stats
    num_samples, avg_det, token_to_detcount = compute_stats(records)

    msg = f"统计结果：sample_token 数量 = {num_samples}，平均每个 sample_token 检测框数量 = {avg_det:.3f}"
    print(msg)
    logger.info(msg)

    # Optional: also log a short summary of min/max for debugging
    if num_samples > 0:
        min_token = min(token_to_detcount, key=token_to_detcount.get)
        max_token = max(token_to_detcount, key=token_to_detcount.get)
        logger.info(
            "Detections per token: min=%d (token=%s), max=%d (token=%s)",
            token_to_detcount[min_token], min_token,
            token_to_detcount[max_token], max_token
        )


if __name__ == "__main__":
    main()