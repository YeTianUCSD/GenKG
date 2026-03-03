#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/read_isfusion_json.py \
  --json_path /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --log_path  /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/IO/output/isfusion_detection.log
'''

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    # Load a JSON file from disk
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected top-level JSON to be a dict, but got: {type(obj)}")
    return obj


def setup_logger(log_path: Path) -> logging.Logger:
    # Create a file logger (overwrite mode)
    logger = logging.getLogger("isfusion_dump_and_stats")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def flatten_samples(isfusion_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Flatten all samples across all scenes into a single list
    scenes = isfusion_obj.get("scenes", [])
    if not isinstance(scenes, list):
        raise TypeError(f'Expected "scenes" to be a list, but got: {type(scenes)}')

    all_samples: List[Dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        samples = scene.get("samples", [])
        if isinstance(samples, list):
            all_samples.extend([s for s in samples if isinstance(s, dict)])
    return all_samples


def safe_len_boxes(sample: Dict[str, Any], key: str) -> int:
    # Return number of 3D boxes for sample[key]["boxes_3d"], robust to missing fields
    block = sample.get(key, {})
    if not isinstance(block, dict):
        return 0
    boxes = block.get("boxes_3d", [])
    return len(boxes) if isinstance(boxes, list) else 0


def dump_first_sample(logger: logging.Logger, samples: List[Dict[str, Any]]) -> None:
    # Dump the first sample (scene[0].samples[0] after flattening) to the log file
    if len(samples) == 0:
        logger.info("No samples found. Nothing to dump.")
        return

    pretty = json.dumps(samples[0], ensure_ascii=False, indent=2)
    logger.info("First sample dump begin")
    logger.info("\n%s", pretty)
    logger.info("First sample dump end")


def compute_stats(samples: List[Dict[str, Any]]) -> Tuple[int, float, float]:
    # Compute number of samples, average number of det boxes, and average number of GT boxes
    n = len(samples)
    if n == 0:
        return 0, 0.0, 0.0

    det_counts = [safe_len_boxes(s, "det") for s in samples]
    gt_counts = [safe_len_boxes(s, "gt") for s in samples]

    avg_det = sum(det_counts) / n
    avg_gt = sum(gt_counts) / n
    return n, avg_det, avg_gt


def main():
    parser = argparse.ArgumentParser(description="Dump first isfusion sample and compute det/gt statistics.")
    parser.add_argument("--json_path", required=True, type=str, help="Path to the isfusion JSON file")
    parser.add_argument("--log_path", required=True, type=str, help="Path to output log file")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    log_path = Path(args.log_path)

    obj = load_json(json_path)
    logger = setup_logger(log_path)

    samples = flatten_samples(obj)

    # 1) Dump the first sample
    dump_first_sample(logger, samples)

    # 2) Compute stats
    num_samples, avg_det, avg_gt = compute_stats(samples)

    # Print results to stdout (English output)
    print(f"Number of samples: {num_samples}")
    print(f"Average #det objects per sample: {avg_det:.3f}")
    print(f"Average #gt objects per sample: {avg_gt:.3f}")
    print(f"Log file written to: {log_path}")

    # Also write stats into log
    logger.info("Number of samples: %d", num_samples)
    logger.info("Average #det objects per sample: %.3f", avg_det)
    logger.info("Average #gt objects per sample: %.3f", avg_gt)


if __name__ == "__main__":
    main()