#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read the n-th entry from an IS-Fusion results JSON.

Usage examples:
  # 打印第 0 条（0-based）
  python /home/code/3Ddetection/IS-Fusion/GenKG/code/read_isfusionres_json.py \
      --json /home/code/3Ddetection/IS-Fusion/GenKG/data/ISFUSIONres_with_tokens.json \
      --index 0

  # 打印第 100 条（1-based）
  python /home/code/3Ddetection/IS-Fusion/GenKG/code/read_isfusionres_json.py \
      --json /home/code/3Ddetection/IS-Fusion/GenKG/data/ISFUSIONres_and_GT.json \
      --index 100 --one-based

   python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_isfusion_GT_json.py  \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/ISFUSIONres_GT_attributions.json\
  --index 0  \
  > /home/code/3Ddetection/IS-Fusion/GenKG/data/read_isfusion_GT__attributions_json.log

  # 仅打印摘要信息
  python /home/code/3Ddetection/IS-Fusion/GenKG/code/read_isfusionres_json.py \
      --json /home/code/3Ddetection/IS-Fusion/GenKG/data/ISFUSIONres_with_tokens.json \
      --index 0 --summary
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

def main():
    parser = argparse.ArgumentParser(description="Print the n-th item from an IS-Fusion result JSON list.")
    parser.add_argument("--json", required=True, help="Path to results JSON (a list)")
    parser.add_argument("--index", type=int, required=True, help="Index of the item to print (0-based by default)")
    parser.add_argument("--one-based", action="store_true", help="Treat --index as 1-based")
    parser.add_argument("--summary", action="store_true", help="Print a brief summary instead of the full JSON")
    args = parser.parse_args()

    path = args.json
    if not os.path.exists(path):
        print(f"[ERR ] JSON not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Load JSON (expects a list)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERR ] Failed to load JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("[ERR ] The JSON root must be a list.", file=sys.stderr)
        sys.exit(1)

    idx = args.index - 1 if args.one_based else args.index
    if not (0 <= idx < len(data)):
        print(f"[ERR ] Index out of range: {idx} (len={len(data)})", file=sys.stderr)
        sys.exit(1)

    item = data[idx]

    print(f"[INFO] File: {path}")
    print(f"[INFO] Length: {len(data)}")
    print(f"[INFO] Printing item at index {idx} ({'1-based ' + str(args.index) if args.one_based else '0-based'})")

    if args.summary and isinstance(item, dict):
        # Try to summarize common IS-Fusion fields
        sample_token = item.get("sample_token")
        sdt = item.get("sample_data_token")
        ts = item.get("timestamp")
        boxes = item.get("boxes_3d")
        scores = item.get("scores_3d")
        labels = item.get("labels_3d")
        n_boxes = len(boxes) if isinstance(boxes, list) else 0
        n_scores = len(scores) if isinstance(scores, list) else 0
        n_labels = len(labels) if isinstance(labels, list) else 0

        print("----- SUMMARY -----")
        print(f"sample_token      : {sample_token}")
        print(f"sample_data_token : {sdt}")
        print(f"timestamp         : {ts}")
        print(f"#boxes/scores/labels: {n_boxes} / {n_scores} / {n_labels}")
        # 如果需要更多字段，可在此继续添加
    else:
        # Pretty print full item
        try:
            txt = json.dumps(item, ensure_ascii=False, indent=2)
        except TypeError:
            # 有极端情况包含不可序列化对象（不太可能，因为我们之前生成的是纯JSON）
            txt = str(item)
        print(txt)

if __name__ == "__main__":
    main()
