#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 IS-Fusion 检测结果在 GT 上的“天花板召回率” + 距离误差统计：

- 对每个 GT，找同类 det 的最近距离 d_min：
    * 如果在平面 (x,y) 上 d_min <= 某个 match_thr，就认为在该阈值下“可被覆盖”。

- 输出：
    * 多个距离阈值下的覆盖率（上限召回率）
    * 所有 d_min 的均值 / 中位数 / 各百分位（反映定位误差分布）

用法示例：

python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/check_detection_max.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --match_thr 0.5 1 2 3 4 5 6 7 8 9 10 \
  --ignore_classes -1
"""

import argparse
import json
from typing import Any, Dict, Iterable, List

import numpy as np


# --------- 工具：递归遍历 JSON，找到所有 sample ---------

def iter_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    与训练脚本里的 iter_samples 一致：
    只要一个 dict 里同时包含 det / gt，且 det.boxes_3d 是 list，
    就认为它是一条 sample。
    """
    if isinstance(obj, dict):
        det = obj.get("det")
        gt = obj.get("gt")
        if isinstance(det, dict) and isinstance(gt, dict) and isinstance(det.get("boxes_3d"), list):
            yield obj
        for v in obj.values():
            yield from iter_samples(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_samples(it)


def safe_stack_boxes(xs: List[List[float]], exp_dim: int = 9) -> np.ndarray:
    """
    把 boxes_3d 安全地变成 [N, exp_dim] 的 numpy 数组
    """
    arr = np.array(xs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, exp_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, exp_dim)
    return arr


# --------- 主逻辑 ---------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="带 det 和 gt 的 JSON 路径")
    parser.add_argument(
        "--match_thr",
        type=float,
        nargs="+",
        default=[2.0],
        help="匹配 det 和 gt 的 xy 距离阈值（米），可传多个值，例如: --match_thr 0.5 1 2 3"
    )
    parser.add_argument(
        "--ignore_classes",
        type=str,
        default="-1",
        help="用逗号分隔的要忽略的类别 id，例如 '10,11'，默认只忽略 -1"
    )
    args = parser.parse_args()

    ignore = set(
        int(x.strip())
        for x in args.ignore_classes.split(",")
        if x.strip()
    )

    print(f"[Info] Loading JSON from: {args.json}")
    with open(args.json, "r") as f:
        root = json.load(f)

    samples = list(iter_samples(root))
    print(f"[Info] Found {len(samples)} samples with det+gt.")

    # 多个阈值
    match_thrs = sorted(set(args.match_thr))

    total_gt = 0
    # 每个阈值一套 covered 计数
    covered_gt: Dict[float, int] = {thr: 0 for thr in match_thrs}

    # 按类别统计：总 GT 与每个阈值下的 covered
    cls_total: Dict[int, int] = {}
    cls_covered: Dict[float, Dict[int, int]] = {thr: {} for thr in match_thrs}

    # 记录所有 GT 的最近距离 d_min（仅对“有同类 det”的 GT）
    all_d_min: List[float] = []

    for s in samples:
        det = s.get("det", {}) or {}
        gt = s.get("gt", {}) or {}

        det_boxes = safe_stack_boxes(det.get("boxes_3d", []) or [], exp_dim=9)
        gt_boxes = safe_stack_boxes(gt.get("boxes_3d", []) or [], exp_dim=9)

        det_labels = np.array(det.get("labels_3d", []) or [], dtype=np.int64)
        gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64)

        # 长度对不上也要截一下，防止奇怪错误
        if det_boxes.shape[0] != det_labels.shape[0]:
            n = min(det_boxes.shape[0], det_labels.shape[0])
            det_boxes = det_boxes[:n]
            det_labels = det_labels[:n]
        if gt_boxes.shape[0] != gt_labels.shape[0]:
            n = min(gt_boxes.shape[0], gt_labels.shape[0])
            gt_boxes = gt_boxes[:n]
            gt_labels = gt_labels[:n]

        # 过滤 ignore 类别
        if ignore:
            if det_labels.size > 0:
                mask_det = np.array([int(c) not in ignore for c in det_labels], dtype=bool)
                det_boxes = det_boxes[mask_det]
                det_labels = det_labels[mask_det]
            if gt_labels.size > 0:
                mask_gt = np.array([int(c) not in ignore for c in gt_labels], dtype=bool)
                gt_boxes = gt_boxes[mask_gt]
                gt_labels = gt_labels[mask_gt]

        if gt_boxes.shape[0] == 0:
            continue  # 这一帧没 GT

        total_gt += gt_boxes.shape[0]

        for cls_id in gt_labels.tolist():
            cls_total[cls_id] = cls_total.get(cls_id, 0) + 1

        # 没有任何 det，就全算没覆盖
        if det_boxes.shape[0] == 0:
            continue

        # 只看 xy 平面
        det_xy = det_boxes[:, :2]  # [N_det, 2]
        gt_xy = gt_boxes[:, :2]    # [N_gt, 2]

        # [N_gt, N_det] 的距离矩阵
        diff = gt_xy[:, None, :] - det_xy[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        # 对每个 GT，看有没有同类 & 最近距离 d_min
        for i in range(gt_boxes.shape[0]):
            c = int(gt_labels[i])
            same_cls = np.where(det_labels == c)[0]
            if same_cls.size == 0:
                # 没有同类 det，这个 GT 在所有阈值下都算没覆盖
                continue

            d_min = float(dist[i, same_cls].min())
            all_d_min.append(d_min)

            # 对每个阈值分别判断
            for thr in match_thrs:
                if d_min <= thr:
                    covered_gt[thr] += 1
                    cls_cov_dict = cls_covered[thr]
                    cls_cov_dict[c] = cls_cov_dict.get(c, 0) + 1

    if total_gt == 0:
        print("[Warn] No GT boxes found after filtering. Nothing to compute.")
        return

    print("========== Detection Ceiling (GT Coverage) ==========")
    print(f"Total GT boxes (after ignore):     {total_gt}")
    print(f"Ignore classes:                    {sorted(ignore) if ignore else 'None'}")
    print(f"Match thresholds (xy distance, m): {match_thrs}")

    for thr in match_thrs:
        cov = covered_gt[thr] / float(total_gt)
        print(f"\n--- Threshold = {thr:.3f} m ---")
        print(f"Covered GT boxes:  {covered_gt[thr]}")
        print(f"Overall coverage:  {cov:.4f}  ({cov*100:.2f}%)")

        print("Per-class coverage (class_id: covered / total = rate):")
        for c in sorted(cls_total.keys()):
            tot_c = cls_total[c]
            cov_c = cls_covered[thr].get(c, 0)
            rate_c = cov_c / float(tot_c) if tot_c > 0 else 0.0
            print(f"  {c:3d}: {cov_c:6d} / {tot_c:6d} = {rate_c:.4f}")

    # --------- 额外：距离误差分布统计 ---------
    if len(all_d_min) > 0:
        d_arr = np.array(all_d_min, dtype=np.float32)
        print("\n========== Distance Error Stats (for GT with at least one same-class det) ==========")
        print(f"Count of such GT:  {d_arr.shape[0]}")
        print(f"Mean distance:     {d_arr.mean():.3f} m")
        print(f"Median distance:   {np.median(d_arr):.3f} m")
        for p in [50, 75, 90, 95, 99]:
            print(f"{p:2d}th percentile:   {np.percentile(d_arr, p):.3f} m")


if __name__ == "__main__":
    main()
