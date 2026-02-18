#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时序版“检测天花板召回率 + 检测式 P/R/F1”统计脚本（带 max_per_frame 截断）：

以“当前帧的 GT”为基准，在同一 scene 内：
- 取当前帧前后 frame_radius 帧（共 2*frame_radius+1 帧）的 IS-Fusion 检测结果
- 在这些帧上的 det 里，寻找与当前 GT 同类别、且 XY 平面距离 <= match_thr 的框

1）Recall 天花板（原逻辑）：
   - 只要窗口内存在一个满足条件的 det，就认为该 GT 被“理论上可用的 det”覆盖。

2）检测式 P/R/F1（新增）：
   - 对当前帧所有 GT 和窗口内所有 det，做“类别感知 + 距离阈值”的一对一贪心匹配：
       * TP: 被匹配到的 GT 数量（匹配对数）
       * FP: 窗口内 det 总数 - TP（未匹配上的 det）
       * FN: 当前帧 GT 总数 - TP（未被任何 det 匹配上的 GT）
   - 统计全局 micro P/R/F1 和 per-class P/R/F1

注意：这里的“预测框”是“时序窗口内所有 det 的并集”，
所以这是一个“利用所有时间窗口的理论上限”的检测指标，而不是现实模型的结果。

用法示例：

python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/check_detection_maxV4.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --match_thr 0.5 1 2 3 5 8 10 \
  --frame_radius 0 \
  --max_per_frame 500 \
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
        "--max_per_frame",
        type=int,
        default=None,
        help="每帧最多保留多少个 det（按 scores_3d 排序截断）；"
             "None 或 0 表示不截断，使用该帧所有 det",
    )
    parser.add_argument(
        "--ignore_classes",
        type=str,
        default="-1",
        help="用逗号分隔的要忽略的类别 id，例如 '10,11'，默认只忽略 -1",
    )
    args = parser.parse_args()

    max_per_frame = args.max_per_frame
    if max_per_frame is not None and max_per_frame <= 0:
        max_per_frame = None  # 统一处理：None = 不截断

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

        # 读取 det / gt
        det_boxes = safe_stack_boxes(det.get("boxes_3d", []) or [], exp_dim=9)
        gt_boxes  = safe_stack_boxes(gt.get("boxes_3d", [])  or [], exp_dim=9)

        det_labels = np.array(det.get("labels_3d", []) or [], dtype=np.int64)
        gt_labels  = np.array(gt.get("labels_3d", []) or [], dtype=np.int64)

        det_scores = np.array(det.get("scores_3d", []) or [], dtype=np.float32).reshape(-1)

        # --------- 对 det 做长度对齐 + max_per_frame 截断（按 score） ---------

        if det_boxes.shape[0] > 0:
            n_det = min(det_boxes.shape[0], det_labels.shape[0], det_scores.shape[0])
            det_boxes = det_boxes[:n_det]
            det_labels = det_labels[:n_det]
            det_scores = det_scores[:n_det]

            if (max_per_frame is not None) and (n_det > max_per_frame):
                idxs = np.argsort(-det_scores)[:max_per_frame]
                det_boxes = det_boxes[idxs]
                det_labels = det_labels[idxs]
                det_scores = det_scores[idxs]
        else:
            det_boxes = np.zeros((0, 9), dtype=np.float32)
            det_labels = np.zeros((0,), dtype=np.int64)
            det_scores = np.zeros((0,), dtype=np.float32)

        # GT 长度对齐（没 scores）
        if gt_boxes.shape[0] > 0:
            n_gt = min(gt_boxes.shape[0], gt_labels.shape[0])
            gt_boxes = gt_boxes[:n_gt]
            gt_labels = gt_labels[:n_gt]
        else:
            gt_boxes = np.zeros((0, 9), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)

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
        gt_xyz_global  = transform_points(T_lidar2global, gt_boxes[:, :3])  if gt_boxes.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

        # 提取 xy
        det_xy_global = det_xyz_global[:, :2] if det_xyz_global.size > 0 else np.zeros((0, 2), dtype=np.float32)
        gt_xy_global  = gt_xyz_global[:, :2]  if gt_xyz_global.size > 0  else np.zeros((0, 2), dtype=np.float32)

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
    cls_total: Dict[int, int] = {}

    # recall 天花板
    covered_gt: Dict[float, int] = {thr: 0 for thr in match_thrs}
    cls_covered: Dict[float, Dict[int, int]] = {thr: {} for thr in match_thrs}

    # 检测式 P/R/F1 统计
    prf_stats: Dict[float, Dict[str, Any]] = {}
    for thr in match_thrs:
        prf_stats[thr] = {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "cls_tp": {},
            "cls_fp": {},
            "cls_fn": {},
        }

    # 记录所有“有同类 det 在窗口内”的 GT 的最近距离 d_min（用于距离分布）
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

            N_gt = gt_xy.shape[0]
            total_gt += N_gt

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
                # 窗口内完全没有 det：所有 GT 都是 FN（在任何 thr 下）
                for thr in match_thrs:
                    stat = prf_stats[thr]
                    stat["fn"] += float(N_gt)
                    for c in gt_labels.tolist():
                        stat["cls_fn"][int(c)] = stat["cls_fn"].get(int(c), 0) + 1
                continue

            det_xy_win = np.concatenate(det_xy_list, axis=0)          # [M,2]
            det_labels_win = np.concatenate(det_labels_list, axis=0)  # [M]
            N_det = det_labels_win.shape[0]

            # [N_gt, M] 的距离矩阵（全局 XY 差）
            diff = gt_xy[:, None, :] - det_xy_win[None, :, :]
            dist = np.linalg.norm(diff, axis=2)  # [N_gt, M]

            # ---------- 先做 coverage + d_min 记录 ----------

            for i in range(N_gt):
                c = int(gt_labels[i])
                same_cls = np.where(det_labels_win == c)[0]
                if same_cls.size == 0:
                    continue
                d_min = float(dist[i, same_cls].min())
                all_d_min.append(d_min)
                for thr in match_thrs:
                    if d_min <= thr:
                        covered_gt[thr] += 1
                        cov_dict = cls_covered[thr]
                        cov_dict[c] = cov_dict.get(c, 0) + 1

            # ---------- 再做检测式 P/R/F1 的一对一匹配 ----------

            for thr in match_thrs:
                stat = prf_stats[thr]

                if N_det == 0 and N_gt == 0:
                    continue
                if N_det == 0 and N_gt > 0:
                    stat["fn"] += float(N_gt)
                    for c in gt_labels.tolist():
                        stat["cls_fn"][int(c)] = stat["cls_fn"].get(int(c), 0) + 1
                    continue
                if N_det > 0 and N_gt == 0:
                    stat["fp"] += float(N_det)
                    for c in det_labels_win.tolist():
                        stat["cls_fp"][int(c)] = stat["cls_fp"].get(int(c), 0) + 1
                    continue

                pairs = []
                # 构造所有“同类且距离 <= thr”的 (d, gt_idx, det_idx)
                for gi in range(N_gt):
                    c = int(gt_labels[gi])
                    same_cls = np.where(det_labels_win == c)[0]
                    if same_cls.size == 0:
                        continue
                    dists_g = dist[gi, same_cls]
                    for k, dj in enumerate(same_cls):
                        d_ij = float(dists_g[k])
                        if d_ij <= thr:
                            pairs.append((d_ij, gi, dj))

                pairs.sort(key=lambda x: x[0])  # 按距离从小到大贪心

                used_gt = set()
                used_det = set()
                matches = []

                for d_ij, gi, dj in pairs:
                    if gi in used_gt or dj in used_det:
                        continue
                    used_gt.add(gi)
                    used_det.add(dj)
                    matches.append((gi, dj))

                tp = float(len(matches))
                fp = float(N_det - len(matches))
                fn = float(N_gt - len(matches))

                stat["tp"] += tp
                stat["fp"] += fp
                stat["fn"] += fn

                # per-class 统计
                for gi, dj in matches:
                    c = int(gt_labels[gi])
                    stat["cls_tp"][c] = stat["cls_tp"].get(c, 0) + 1

                for dj in range(N_det):
                    if dj in used_det:
                        continue
                    c = int(det_labels_win[dj])
                    stat["cls_fp"][c] = stat["cls_fp"].get(c, 0) + 1

                for gi in range(N_gt):
                    if gi in used_gt:
                        continue
                    c = int(gt_labels[gi])
                    stat["cls_fn"][c] = stat["cls_fn"].get(c, 0) + 1

    # --------- 输出：Recall 天花板（原逻辑） ---------

    if total_gt == 0:
        print("[Warn] No GT boxes found after filtering. Nothing to compute.")
        return

    print("========== Temporal Detection Ceiling (GT Coverage over window) ==========")
    print(f"Total GT boxes (after ignore):     {total_gt}")
    print(f"Ignore classes:                    {sorted(ignore) if ignore else 'None'}")
    print(f"Match thresholds (xy distance, m): {match_thrs}")
    print(f"Frame radius:                      {R} (window = {2*R+1} frames)")
    print(f"Max det per frame (Top-K by score): {max_per_frame if max_per_frame is not None else 'no limit'}")

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

    # --------- 输出：检测式 P/R/F1（时序窗口版） ---------

    print("\n\n========== Temporal-window Detection Metrics (P/R/F1 over GT frames) ==========")
    for thr in match_thrs:
        stat = prf_stats[thr]
        tp = stat["tp"]
        fp = stat["fp"]
        fn = stat["fn"]

        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

        print("\n" + "=" * 70)
        print(f"Distance threshold = {thr:.3f} meters")
        print("-" * 70)
        print("class |       TP       FP       FN ||   Prec    Rec     F1")
        print("-" * 70)

        # per-class 行
        all_classes = sorted(cls_total.keys())
        for c in all_classes:
            tp_c = float(stat["cls_tp"].get(c, 0))
            fp_c = float(stat["cls_fp"].get(c, 0))
            fn_c = float(stat["cls_fn"].get(c, 0))
            if tp_c + fp_c > 0:
                prec_c = tp_c / (tp_c + fp_c)
            else:
                prec_c = 0.0
            if tp_c + fn_c > 0:
                rec_c = tp_c / (tp_c + fn_c)
            else:
                rec_c = 0.0
            if prec_c + rec_c > 0:
                f1_c = 2 * prec_c * rec_c / (prec_c + rec_c)
            else:
                f1_c = 0.0

            print(f"{c:4d} | {int(tp_c):7d} {int(fp_c):7d} {int(fn_c):7d} ||"
                  f" {prec_c:6.3f} {rec_c:6.3f} {f1_c:6.3f}")

        print("-" * 70)
        print(f" MICRO| {int(tp):7d} {int(fp):7d} {int(fn):7d} ||"
              f" {P:6.3f} {R:6.3f} {F1:6.3f}")

        # 简单 macro-F1（对有 GT 的类别平均）
        macro_P = []
        macro_R = []
        macro_F1 = []
        for c in all_classes:
            tp_c = float(stat["cls_tp"].get(c, 0))
            fp_c = float(stat["cls_fp"].get(c, 0))
            fn_c = float(stat["cls_fn"].get(c, 0))
            if cls_total.get(c, 0) == 0:
                continue
            if tp_c + fp_c > 0:
                p_c = tp_c / (tp_c + fp_c)
            else:
                p_c = 0.0
            if tp_c + fn_c > 0:
                r_c = tp_c / (tp_c + fn_c)
            else:
                r_c = 0.0
            if p_c + r_c > 0:
                f1_c = 2 * p_c * r_c / (p_c + r_c)
            else:
                f1_c = 0.0
            macro_P.append(p_c)
            macro_R.append(r_c)
            macro_F1.append(f1_c)

        if len(macro_P) > 0:
            mP = float(np.mean(macro_P))
            mR = float(np.mean(macro_R))
            mF1 = float(np.mean(macro_F1))
        else:
            mP = mR = mF1 = 0.0

        print(f" MACRO|    -       -       -   || {mP:6.3f} {mR:6.3f} {mF1:6.3f}")


if __name__ == "__main__":
    main()
