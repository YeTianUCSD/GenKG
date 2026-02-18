#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时序版“检测天花板召回率”统计脚本：

以“当前帧的 GT”为基准，在同一 scene 内：
- 取当前帧前后 frame_radius 帧（共 2*frame_radius+1 帧）的 IS-Fusion 检测结果
- 在这些帧上的 det 里，寻找与当前 GT 同类别、且 XY 平面距离 <= match_thr 的框
- 如果存在，则认为这个 GT 在该时序窗口内“理论上可以被检测/利用到”

也就是说：
- 之前是只看“同一帧上的 det 能不能匹配到这个 GT”
- 现在是看“前后若干帧整体上有没有任何一个 det 匹配到这个 GT”

用法示例：

python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/check_detection_maxV2.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --match_thr 0.5 1 2 3 5 8 10 \
  --frame_radius 9 \
  --ignore_classes -1
"""

import argparse
import json
import math
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


# --------- 位姿与坐标变换工具 ---------

def quat_to_rot(q: List[float]) -> np.ndarray:
    """
    四元数 [w, x, y, z] -> 3x3 旋转矩阵
    """
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def make_T(translation: List[float], quat_wxyz: List[float]) -> np.ndarray:
    """
    给定平移和四元数，构造 4x4 齐次变换矩阵
    """
    R = quat_to_rot(quat_wxyz)
    t = np.array(translation, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    使用 4x4 齐次变换矩阵 T 变换点云 pts（N,3）
    """
    if pts.size == 0:
        return pts
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1), dtype=pts.dtype)])  # [N,4]
    out = (T @ homo.T).T[:, :3]  # [N,3]
    return out


# --------- 主逻辑 ---------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="带 det 和 gt 的 JSON 路径")
    parser.add_argument(
        "--match_thr",
        type=float,
        nargs="+",
        default=[2.0],
        help="匹配 det 和 gt 的 xy 距离阈值（米），可传多个值，例如: --match_thr 0.5 1 2 3",
    )
    parser.add_argument(
        "--frame_radius",
        type=int,
        default=5,
        help="时序窗口半径：前后多少帧一起参与匹配，例如 5 表示 [-5,...,0,...,+5] 共 11 帧",
    )
    parser.add_argument(
        "--ignore_classes",
        type=str,
        default="-1",
        help="用逗号分隔的要忽略的类别 id，例如 '10,11'，默认只忽略 -1",
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

    # --------- 按 scene 分组，并预先计算全局坐标系下的 det/gt xy ---------

    scenes: Dict[str, List[Dict[str, Any]]] = {}

    for s in samples:
        scene_token = s.get("scene_token", "")
        if not scene_token:
            continue
        ts = int(s.get("timestamp", s.get("timestamp_us", 0)))

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

        # 位姿：lidar -> ego -> global
        lidar_info = s.get("lidar2ego", None)
        ego_global_info = s.get("ego2global", s.get("ego_pose", None))

        if (lidar_info is None) or (ego_global_info is None):
            # 缺少位姿信息，直接跳过这一帧
            continue

        Tl2e = make_T(
            lidar_info["translation"],
            lidar_info["rotation"],
        )
        Te2g = make_T(
            ego_global_info["translation"],
            ego_global_info["rotation"],
        )
        T_lidar2global = Te2g @ Tl2e  # p_global = Te2g * Tl2e * p_lidar

        # 变换到全局坐标系
        det_xyz_global = transform_points(T_lidar2global, det_boxes[:, :3]) if det_boxes.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
        gt_xyz_global = transform_points(T_lidar2global, gt_boxes[:, :3]) if gt_boxes.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

        # 提取 xy
        det_xy_global = det_xyz_global[:, :2] if det_xyz_global.size > 0 else np.zeros((0, 2), dtype=np.float32)
        gt_xy_global = gt_xyz_global[:, :2] if gt_xyz_global.size > 0 else np.zeros((0, 2), dtype=np.float32)

        # 过滤 ignore 类别
        if ignore:
            if det_labels.size > 0:
                mask_det = np.array([int(c) not in ignore for c in det_labels], dtype=bool)
                det_xy_global = det_xy_global[mask_det]
                det_labels = det_labels[mask_det]
            if gt_labels.size > 0:
                mask_gt = np.array([int(c) not in ignore for c in gt_labels], dtype=bool)
                gt_xy_global = gt_xy_global[mask_gt]
                gt_labels = gt_labels[mask_gt]

        frame_info = {
            "timestamp": ts,
            "det_xy": det_xy_global,     # [N_det, 2]
            "det_labels": det_labels,    # [N_det]
            "gt_xy": gt_xy_global,       # [N_gt, 2]
            "gt_labels": gt_labels,      # [N_gt]
        }

        scenes.setdefault(scene_token, []).append(frame_info)

    # 对每个 scene 按时间排序
    for sc in scenes:
        scenes[sc].sort(key=lambda x: x["timestamp"])

    print(f"[Info] Scenes loaded: {len(scenes)}")

    # --------- 统计：以当前帧 GT 为基准，在时序窗口内看 det ---------

    total_gt = 0
    covered_gt: Dict[float, int] = {thr: 0 for thr in match_thrs}

    cls_total: Dict[int, int] = {}
    cls_covered: Dict[float, Dict[int, int]] = {thr: {} for thr in match_thrs}

    # 记录所有“有同类 det 在窗口内”的 GT 的最近距离 d_min
    all_d_min: List[float] = []

    R = args.frame_radius

    for sc, frames in scenes.items():
        n_frames = len(frames)
        if n_frames == 0:
            continue

        for idx in range(n_frames):
            center = frames[idx]
            gt_xy = center["gt_xy"]
            gt_labels = center["gt_labels"]

            if gt_xy.shape[0] == 0:
                continue  # 当前帧没有 GT

            total_gt += gt_xy.shape[0]

            for cls_id in gt_labels.tolist():
                cls_total[cls_id] = cls_total.get(cls_id, 0) + 1

            # 构造窗口内的所有 det（前后 R 帧 + 当前帧）
            left = max(0, idx - R)
            right = min(n_frames - 1, idx + R)

            det_xy_list = []
            det_labels_list = []

            for j in range(left, right + 1):
                det_xy_j = frames[j]["det_xy"]
                det_lab_j = frames[j]["det_labels"]
                if det_xy_j.shape[0] == 0:
                    continue
                det_xy_list.append(det_xy_j)
                det_labels_list.append(det_lab_j)

            if not det_xy_list:
                # 窗口内完全没有 det，则这些 GT 在所有阈值下都覆盖不到
                continue

            det_xy_win = np.concatenate(det_xy_list, axis=0)      # [M,2]
            det_labels_win = np.concatenate(det_labels_list, axis=0)  # [M]

            # [N_gt, M] 的距离矩阵（全局 XY 差）
            diff = gt_xy[:, None, :] - det_xy_win[None, :, :]
            dist = np.linalg.norm(diff, axis=2)  # [N_gt, M]

            # 对当前帧的每个 GT，找窗口内最近的同类 det
            for i in range(gt_xy.shape[0]):
                c = int(gt_labels[i])
                same_cls = np.where(det_labels_win == c)[0]
                if same_cls.size == 0:
                    # 窗口内根本没有同类 det
                    continue

                d_min = float(dist[i, same_cls].min())
                all_d_min.append(d_min)

                for thr in match_thrs:
                    if d_min <= thr:
                        covered_gt[thr] += 1
                        cls_cov_dict = cls_covered[thr]
                        cls_cov_dict[c] = cls_cov_dict.get(c, 0) + 1

    # --------- 输出结果 ---------

    if total_gt == 0:
        print("[Warn] No GT boxes found after filtering. Nothing to compute.")
        return

    print("========== Temporal Detection Ceiling (GT Coverage over window) ==========")
    print(f"Total GT boxes (after ignore):     {total_gt}")
    print(f"Ignore classes:                    {sorted(ignore) if ignore else 'None'}")
    print(f"Match thresholds (xy distance, m): {match_thrs}")
    print(f"Frame radius:                      {R} (window = {2*R+1} frames)")

    for thr in match_thrs:
        cov = covered_gt[thr] / float(total_gt)
        print(f"\n--- Threshold = {thr:.3f} m ---")
        print(f"Covered GT boxes (temporal):  {covered_gt[thr]}")
        print(f"Overall coverage (temporal):  {cov:.4f}  ({cov*100:.2f}%)")

        print("Per-class coverage (class_id: covered / total = rate):")
        for c in sorted(cls_total.keys()):
            tot_c = cls_total[c]
            cov_c = cls_covered[thr].get(c, 0)
            rate_c = cov_c / float(tot_c) if tot_c > 0 else 0.0
            print(f"  {c:3d}: {cov_c:6d} / {tot_c:6d} = {rate_c:.4f}")

    # 距离分布（仅统计“窗口内至少有同类 det”的 GT）
    if len(all_d_min) > 0:
        d_arr = np.array(all_d_min, dtype=np.float32)
        print("\n========== Distance Error Stats over temporal window ==========")
        print(f"Count of GT with same-class det in window:  {d_arr.shape[0]}")
        print(f"Mean distance:     {d_arr.mean():.3f} m")
        print(f"Median distance:   {np.median(d_arr):.3f} m")
        for p in [50, 75, 90, 95, 99]:
            print(f"{p:2d}th percentile:   {np.percentile(d_arr, p):.3f} m")


if __name__ == "__main__":
    main()
