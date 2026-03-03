#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal detection ceiling evaluator for multiple det fields in one JSON.

Typical use:

python /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/model/late_fusion/check_detection_ceiling_multi_det.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/model/late_fusion/output/V2/sorted_by_scene_ISFUSIONandGTattr_val_3_modality_aligned_fused_candidates.json \
  --det_fields det_isfusion,det \
  --report_names isfusion,fused \
  --match_thr 2.0 \
  --frame_radius 0 \
  --max_per_frame 0 \
  --ignore_classes -1

"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def iter_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        gt = obj.get("gt")
        if isinstance(gt, dict) and ("scene_token" in obj):
            ts = obj.get("timestamp", obj.get("timestamp_us", 0))
            if ts:
                yield obj
        for v in obj.values():
            yield from iter_samples(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_samples(it)


def safe_stack_boxes(xs: List[List[float]], exp_dim: int = 9) -> np.ndarray:
    arr = np.array(xs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, exp_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < exp_dim:
        pad = np.zeros((arr.shape[0], exp_dim - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > exp_dim:
        arr = arr[:, :exp_dim]
    return arr.astype(np.float32, copy=False)


def quat_to_rot(q: List[float]) -> np.ndarray:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_T(translation: List[float], quat_wxyz: List[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(quat_wxyz)
    t = np.array(translation, dtype=np.float64).reshape(3,)
    T[:3, 3] = t
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    homo = np.hstack([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)])
    out = (T @ homo.T).T[:, :3]
    return out


def parse_ignore(ignore_str: str) -> set[int]:
    out: set[int] = set()
    for x in (ignore_str or "").split(","):
        x = x.strip()
        if x:
            out.add(int(x))
    return out


def _parse_det_field(
    sample: Dict[str, Any],
    field_name: str,
    max_per_frame: int | None,
    ignore: set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    det = sample.get(field_name, {}) or {}
    if not isinstance(det, dict):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes = safe_stack_boxes(det.get("boxes_3d", []) or [], exp_dim=9)
    labels = np.array(det.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)
    scores = np.array(det.get("scores_3d", []) or [], dtype=np.float32).reshape(-1)

    if boxes.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    n = min(boxes.shape[0], labels.shape[0], scores.shape[0] if scores.size > 0 else boxes.shape[0])
    boxes = boxes[:n]
    labels = labels[:n]
    if scores.size > 0:
        scores = scores[:n]
    else:
        scores = np.ones((n,), dtype=np.float32)

    if ignore:
        m = np.array([int(c) not in ignore for c in labels.tolist()], dtype=bool)
        boxes = boxes[m]
        labels = labels[m]
        scores = scores[m]

    if max_per_frame is not None and boxes.shape[0] > max_per_frame:
        idx = np.argsort(-scores)[:max_per_frame]
        boxes = boxes[idx]
        labels = labels[idx]

    return boxes[:, :2].astype(np.float32, copy=False), labels.astype(np.int64, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate temporal detection ceiling for multiple det fields.")
    ap.add_argument("--json", required=True, type=str)
    ap.add_argument("--det_fields", type=str, default="det_isfusion,det",
                    help="Comma-separated det field names, e.g. det_isfusion,det")
    ap.add_argument("--report_names", type=str, default="",
                    help="Optional comma-separated display names aligned with det_fields.")
    ap.add_argument("--match_thr", type=float, nargs="+", default=[2.0])
    ap.add_argument("--frame_radius", type=int, default=0)
    ap.add_argument("--max_per_frame", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--ignore_classes", type=str, default="-1")
    args = ap.parse_args()

    det_fields = [x.strip() for x in args.det_fields.split(",") if x.strip()]
    if not det_fields:
        raise ValueError("det_fields is empty.")
    report_names = [x.strip() for x in args.report_names.split(",") if x.strip()]
    if report_names and len(report_names) != len(det_fields):
        raise ValueError("report_names length must match det_fields length.")
    if not report_names:
        report_names = list(det_fields)

    max_per_frame = None if int(args.max_per_frame) <= 0 else int(args.max_per_frame)
    ignore = parse_ignore(args.ignore_classes)
    thrs = sorted(set(float(x) for x in args.match_thr))
    R = int(max(0, args.frame_radius))

    print(f"[Info] Loading JSON: {args.json}")
    with open(args.json, "r", encoding="utf-8") as f:
        root = json.load(f)
    samples = list(iter_samples(root))
    print(f"[Info] Samples found: {len(samples)}")

    scenes: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        sc = str(s.get("scene_token", "") or "")
        if sc == "":
            continue
        ts = int(s.get("timestamp", s.get("timestamp_us", 0)))
        if ts == 0:
            continue
        lidar_info = s.get("lidar2ego", None)
        ego_global = s.get("ego2global", s.get("ego_pose", None))
        if not isinstance(lidar_info, dict) or not isinstance(ego_global, dict):
            continue
        if ("translation" not in lidar_info) or ("rotation" not in lidar_info):
            continue
        if ("translation" not in ego_global) or ("rotation" not in ego_global):
            continue

        gt = s.get("gt", {}) or {}
        gt_boxes = safe_stack_boxes(gt.get("boxes_3d", []) or [], exp_dim=9)
        gt_labels = np.array(gt.get("labels_3d", []) or [], dtype=np.int64).reshape(-1)
        n_gt = min(gt_boxes.shape[0], gt_labels.shape[0])
        gt_boxes = gt_boxes[:n_gt]
        gt_labels = gt_labels[:n_gt]
        if ignore and n_gt > 0:
            mgt = np.array([int(c) not in ignore for c in gt_labels.tolist()], dtype=bool)
            gt_boxes = gt_boxes[mgt]
            gt_labels = gt_labels[mgt]

        T_l2e = make_T(lidar_info["translation"], lidar_info["rotation"])
        T_e2g = make_T(ego_global["translation"], ego_global["rotation"])
        T_l2g = T_e2g @ T_l2e

        gt_xyz_g = transform_points(T_l2g, gt_boxes[:, :3]) if gt_boxes.shape[0] > 0 else np.zeros((0, 3), dtype=np.float64)
        frame: Dict[str, Any] = {
            "timestamp": ts,
            "gt_xy": gt_xyz_g[:, :2].astype(np.float32, copy=False),
            "gt_labels": gt_labels.astype(np.int64, copy=False),
            "det_by_field": {},
        }

        for f_name in det_fields:
            det_xy_l, det_lb = _parse_det_field(s, f_name, max_per_frame=max_per_frame, ignore=ignore)
            det_xyz_l = np.zeros((det_xy_l.shape[0], 3), dtype=np.float64)
            det_xyz_l[:, :2] = det_xy_l.astype(np.float64)
            det_xyz_g = transform_points(T_l2g, det_xyz_l) if det_xyz_l.shape[0] > 0 else np.zeros((0, 3), dtype=np.float64)
            frame["det_by_field"][f_name] = {
                "xy": det_xyz_g[:, :2].astype(np.float32, copy=False),
                "labels": det_lb.astype(np.int64, copy=False),
            }

        scenes.setdefault(sc, []).append(frame)

    for sc in scenes:
        scenes[sc].sort(key=lambda x: x["timestamp"])
    print(f"[Info] Scenes loaded: {len(scenes)}")

    total_gt = 0
    cls_total: Dict[int, int] = {}
    covered: Dict[str, Dict[float, int]] = {f: {t: 0 for t in thrs} for f in det_fields}
    cls_cov: Dict[str, Dict[float, Dict[int, int]]] = {f: {t: {} for t in thrs} for f in det_fields}

    for _, frames in scenes.items():
        n = len(frames)
        for i in range(n):
            gt_xy = frames[i]["gt_xy"]
            gt_labels = frames[i]["gt_labels"]
            if gt_xy.shape[0] == 0:
                continue
            total_gt += int(gt_xy.shape[0])
            for c in gt_labels.tolist():
                cls_total[int(c)] = cls_total.get(int(c), 0) + 1

            l = max(0, i - R)
            r = min(n - 1, i + R)
            for f_name in det_fields:
                det_xy_list: List[np.ndarray] = []
                det_lb_list: List[np.ndarray] = []
                for j in range(l, r + 1):
                    d = frames[j]["det_by_field"][f_name]
                    if d["xy"].shape[0] == 0:
                        continue
                    det_xy_list.append(d["xy"])
                    det_lb_list.append(d["labels"])
                if not det_xy_list:
                    continue
                det_xy = np.concatenate(det_xy_list, axis=0)
                det_lb = np.concatenate(det_lb_list, axis=0)

                diff = gt_xy[:, None, :] - det_xy[None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                for gi in range(gt_xy.shape[0]):
                    c = int(gt_labels[gi])
                    same_cls = np.where(det_lb == c)[0]
                    if same_cls.size == 0:
                        continue
                    dmin = float(dist[gi, same_cls].min())
                    for t in thrs:
                        if dmin <= t:
                            covered[f_name][t] += 1
                            cls_cov[f_name][t][c] = cls_cov[f_name][t].get(c, 0) + 1

    print("\n========== Temporal Detection Ceiling (multi det fields) ==========")
    print(f"Total GT boxes (after ignore): {total_gt}")
    print(f"Ignore classes: {sorted(ignore) if ignore else 'None'}")
    print(f"Thresholds: {thrs}")
    print(f"Frame radius: {R} (window={2*R+1})")
    print(f"Max det per frame: {max_per_frame if max_per_frame is not None else 'no limit'}")

    if total_gt == 0:
        print("[Warn] No GT found after filtering.")
        return

    for f_name, disp in zip(det_fields, report_names):
        print("\n" + "=" * 72)
        print(f"Field: {f_name}  |  Name: {disp}")
        for t in thrs:
            c = int(covered[f_name][t])
            rate = float(c / max(1, total_gt))
            print(f"  thr={t:.3f}m: covered={c} / {total_gt} => {rate:.4f} ({rate*100:.2f}%)")
            print("    per-class:")
            for cls_id in sorted(cls_total.keys()):
                tot = int(cls_total[cls_id])
                got = int(cls_cov[f_name][t].get(cls_id, 0))
                rr = float(got / max(1, tot))
                print(f"      {cls_id:3d}: {got:6d}/{tot:6d} = {rr:.4f}")


if __name__ == "__main__":
    main()

