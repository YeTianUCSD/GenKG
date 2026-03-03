#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fuse tri-modality detections into one training/inference JSON.

Input:
- JSON from align_three_modalities.py, with per-sample fields:
  - det_isfusion
  - det_camera_raw
  - det_lidar_raw

Output:
- Same scene/sample structure, with sample["det"] replaced by fused candidates.
- Optional debug fields for traceability (source membership / cluster size).

Fusion pipeline per sample:
1) Normalize all modalities to [x,y,z,dx,dy,dz,yaw,vx,vy] + label + score
2) Coordinate handling for camera/lidar raw:
   - lidar: keep as-is
   - global: convert to lidar frame using sample pose
   - auto: choose between the two using nearest-distance to ISFusion candidates
3) Per-source score calibration (temperature + bias in logit space)
4) Class-aware spatial clustering merge (XY radius + optional Z gate)
5) Cluster aggregation:
   - weighted box average
   - fused score = noisy-or(source-weighted member scores) + consensus bonus


python /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/model/late_fusion/merge_three_modality_candidates.py \
  --aligned_json /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/model/late_fusion/output/sorted_by_scene_ISFUSIONandGTattr_val_3_modality_aligned.json \
  --output_dir /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/model/late_fusion/output \
  --second_dedup \
  --prune_single_source \
  --single_source_score_thr 0.08 \
  --single_source_isolated_score_thr 0.12


"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_LABEL_MAP: Dict[str, int] = {
    "car": 0,
    "truck": 1,
    "construction_vehicle": 2,
    "bus": 3,
    "trailer": 4,
    "barrier": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "pedestrian": 8,
    "traffic_cone": 9,
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if math.isnan(x) or math.isinf(x):
        return float(default)
    return float(x)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _sigmoid(x: float) -> float:
    x = max(-40.0, min(40.0, float(x)))
    return float(1.0 / (1.0 + math.exp(-x)))


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    pp = max(eps, min(1.0 - eps, float(p)))
    return float(math.log(pp) - math.log(1.0 - pp))


def _quat_to_yaw(q: List[float]) -> float:
    if len(q) < 4:
        return 0.0
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(t0, t1))


def _quat_to_rot(q: List[float]) -> np.ndarray:
    if len(q) < 4:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = [float(v) for v in q[:4]]
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


def _make_T(translation: List[float], quat_wxyz: List[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_rot(quat_wxyz)
    t = [0.0, 0.0, 0.0]
    if isinstance(translation, list):
        for i in range(min(3, len(translation))):
            t[i] = _safe_float(translation[i], 0.0)
    T[:3, 3] = np.array(t, dtype=np.float64)
    return T


def _inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def _transform_xyz(T: np.ndarray, xyz: List[float]) -> List[float]:
    p = np.ones((4,), dtype=np.float64)
    p[0] = _safe_float(xyz[0] if len(xyz) > 0 else 0.0, 0.0)
    p[1] = _safe_float(xyz[1] if len(xyz) > 1 else 0.0, 0.0)
    p[2] = _safe_float(xyz[2] if len(xyz) > 2 else 0.0, 0.0)
    q = T @ p
    return [float(q[0]), float(q[1]), float(q[2])]


def _get_T_global_to_lidar(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    lidar_info = sample.get("lidar2ego")
    ego_global_info = sample.get("ego2global", sample.get("ego_pose"))
    if not isinstance(lidar_info, dict) or not isinstance(ego_global_info, dict):
        return None
    if "translation" not in lidar_info or "rotation" not in lidar_info:
        return None
    if "translation" not in ego_global_info or "rotation" not in ego_global_info:
        return None
    T_l2e = _make_T(lidar_info["translation"], lidar_info["rotation"])
    T_e2g = _make_T(ego_global_info["translation"], ego_global_info["rotation"])
    T_l2g = T_e2g @ T_l2e
    return _inv_se3(T_l2g)


def _calibrate_score(score: float, temp: float, bias: float) -> float:
    s = _clamp01(score)
    t = float(temp) if float(temp) > 1e-6 else 1.0
    z = _safe_logit(s) / t + float(bias)
    return _clamp01(_sigmoid(z))


def _xy_dist2(a: List[float], b: List[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float(dx * dx + dy * dy)


def _nearest_xy_dist(cands_a: List[Dict[str, Any]], cands_ref: List[Dict[str, Any]]) -> float:
    if len(cands_a) == 0 or len(cands_ref) == 0:
        return float("inf")
    by_label: Dict[int, List[Dict[str, Any]]] = {}
    for c in cands_ref:
        by_label.setdefault(int(c["label"]), []).append(c)
    d_list: List[float] = []
    for c in cands_a:
        ref = by_label.get(int(c["label"]), cands_ref)
        if len(ref) == 0:
            continue
        d2 = min(_xy_dist2(c["box"], r["box"]) for r in ref)
        d_list.append(math.sqrt(max(0.0, d2)))
    if len(d_list) == 0:
        return float("inf")
    d_list.sort()
    return float(d_list[len(d_list) // 2])  # median


def _normalize_isfusion_det(
    det_obj: Dict[str, Any],
    *,
    score_temp: float,
    score_bias: float,
) -> List[Dict[str, Any]]:
    boxes = det_obj.get("boxes_3d", []) or []
    labels = det_obj.get("labels_3d", []) or []
    scores = det_obj.get("scores_3d", []) or []
    attrs = det_obj.get("attrs", []) or []
    n = min(len(boxes), len(labels), len(scores))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        b = boxes[i]
        if not isinstance(b, list):
            continue
        if len(b) < 9:
            b = list(b) + [0.0] * (9 - len(b))
        box9 = [_safe_float(x, 0.0) for x in b[:9]]
        score_raw = _clamp01(_safe_float(scores[i], 0.0))
        out.append(
            {
                "box": box9,
                "label": int(labels[i]),
                "score_raw": float(score_raw),
                "score": _calibrate_score(score_raw, score_temp, score_bias),
                "attr": str(attrs[i]) if i < len(attrs) and attrs[i] is not None else "",
                "source": "isfusion",
                "coord_frame": "lidar",
            }
        )
    return out


def _build_raw_candidate(
    row: Dict[str, Any],
    source: str,
    label_map: Dict[str, int],
    unknown_label_id: int,
    score_min: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(row, dict):
        return None, "skip_non_dict"
    dname = str(row.get("detection_name", "")).strip()
    if dname == "":
        return None, "skip_no_name"
    if dname not in label_map:
        if unknown_label_id < 0:
            return None, "skip_unknown_name"
        label = int(unknown_label_id)
    else:
        label = int(label_map[dname])

    score_raw = _clamp01(_safe_float(row.get("detection_score", 0.0), 0.0))
    if score_raw < float(score_min):
        return None, "skip_low_score"

    tr = row.get("translation", []) or []
    sz = row.get("size", []) or []
    rt = row.get("rotation", []) or []
    vel = row.get("velocity", []) or []
    if len(tr) < 3 or len(sz) < 3:
        return None, "skip_invalid_geom"

    x, y, z = _safe_float(tr[0]), _safe_float(tr[1]), _safe_float(tr[2])
    dx, dy, dz = _safe_float(sz[0]), _safe_float(sz[1]), _safe_float(sz[2])
    yaw = _quat_to_yaw(rt if isinstance(rt, list) else [])
    vx = _safe_float(vel[0], 0.0) if len(vel) > 0 else 0.0
    vy = _safe_float(vel[1], 0.0) if len(vel) > 1 else 0.0
    attr = str(row.get("attribute_name", "") or "")

    return (
        {
            "box": [x, y, z, dx, dy, dz, yaw, vx, vy],
            "label": label,
            "score_raw": float(score_raw),
            "score": float(score_raw),  # calibrated later
            "attr": attr,
            "source": source,
            "coord_frame": "unknown",
        },
        "ok",
    )


def _normalize_detector_raw(
    det_list: List[Dict[str, Any]],
    *,
    sample: Dict[str, Any],
    source: str,
    label_map: Dict[str, int],
    unknown_label_id: int,
    score_min: float,
    score_temp: float,
    score_bias: float,
    raw_coord_frame: str,
    isf_ref: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Counter]:
    stats: Counter = Counter()
    parsed: List[Dict[str, Any]] = []
    for row in det_list:
        cand, tag = _build_raw_candidate(
            row=row,
            source=source,
            label_map=label_map,
            unknown_label_id=unknown_label_id,
            score_min=score_min,
        )
        stats[tag] += 1
        if cand is not None:
            parsed.append(cand)

    if len(parsed) == 0:
        return [], stats

    T_g2l = _get_T_global_to_lidar(sample)
    cands_as_is = copy.deepcopy(parsed)
    cands_g2l: List[Dict[str, Any]] = []
    if T_g2l is not None:
        for c in parsed:
            cc = copy.deepcopy(c)
            xyz_l = _transform_xyz(T_g2l, cc["box"][:3])
            cc["box"] = [xyz_l[0], xyz_l[1], xyz_l[2]] + [float(v) for v in cc["box"][3:]]
            cc["coord_frame"] = "global_to_lidar"
            cands_g2l.append(cc)

    selected: List[Dict[str, Any]]
    mode = str(raw_coord_frame).lower().strip()
    if mode == "lidar":
        selected = cands_as_is
        for c in selected:
            c["coord_frame"] = "lidar_as_is"
        stats["coord_lidar_as_is"] += len(selected)
    elif mode == "global":
        if T_g2l is None:
            selected = cands_as_is
            for c in selected:
                c["coord_frame"] = "fallback_lidar_as_is"
            stats["coord_global_no_pose_fallback"] += len(selected)
        else:
            selected = cands_g2l
            stats["coord_global_to_lidar"] += len(selected)
    else:  # auto
        if T_g2l is None or len(cands_g2l) == 0 or len(isf_ref) == 0:
            selected = cands_as_is
            for c in selected:
                c["coord_frame"] = "auto_fallback_lidar_as_is"
            stats["coord_auto_fallback"] += len(selected)
        else:
            d_as_is = _nearest_xy_dist(cands_as_is, isf_ref)
            d_g2l = _nearest_xy_dist(cands_g2l, isf_ref)
            if d_g2l < d_as_is:
                selected = cands_g2l
                stats["coord_auto_choose_global_to_lidar"] += len(selected)
            else:
                selected = cands_as_is
                for c in selected:
                    c["coord_frame"] = "auto_choose_lidar_as_is"
                stats["coord_auto_choose_lidar_as_is"] += len(selected)

    for c in selected:
        c["score"] = _calibrate_score(c["score_raw"], score_temp, score_bias)
    stats["kept"] += len(selected)
    return selected, stats


def _fuse_score_noisy_or(
    members: List[Dict[str, Any]],
    source_weights: Dict[str, float],
    consensus_bonus: float,
) -> float:
    p_fail = 1.0
    uniq_sources = set()
    for m in members:
        src = str(m.get("source", ""))
        uniq_sources.add(src)
        sw = float(source_weights.get(src, 1.0))
        p = _clamp01(float(m["score"]) * sw)
        p_fail *= float(1.0 - p)
    fused = float(1.0 - p_fail)
    if len(uniq_sources) > 1:
        fused += float(consensus_bonus) * float(len(uniq_sources) - 1)
    return _clamp01(fused)


def _cluster_and_merge(
    cands: List[Dict[str, Any]],
    *,
    merge_dist_xy: float,
    merge_dist_z: float,
    source_weights: Dict[str, float],
    consensus_bonus: float,
) -> Tuple[List[Dict[str, Any]], Counter]:
    stats: Counter = Counter()
    if len(cands) == 0:
        return [], stats

    thr2 = float(max(merge_dist_xy, 1e-6)) ** 2
    use_z = float(merge_dist_z) > 0.0
    order = sorted(range(len(cands)), key=lambda i: float(cands[i]["score"]), reverse=True)
    clusters: List[Dict[str, Any]] = []

    for idx in order:
        c = cands[idx]
        lab = int(c["label"])
        best_k = -1
        best_d2 = float("inf")
        for k, clu in enumerate(clusters):
            if int(clu["label"]) != lab:
                continue
            d2 = _xy_dist2(c["box"], clu["center"])
            if d2 > thr2:
                continue
            if use_z and abs(float(c["box"][2]) - float(clu["center"][2])) > float(merge_dist_z):
                continue
            if d2 < best_d2:
                best_d2 = d2
                best_k = k

        if best_k < 0:
            clusters.append(
                {
                    "label": int(lab),
                    "members": [c],
                    "center": [float(v) for v in c["box"][:3]],
                }
            )
            stats["new_cluster"] += 1
        else:
            clusters[best_k]["members"].append(c)
            mm = clusters[best_k]["members"]
            wsum = sum(max(1e-4, float(x["score"])) for x in mm)
            cx = sum(float(x["box"][0]) * max(1e-4, float(x["score"])) for x in mm) / wsum
            cy = sum(float(x["box"][1]) * max(1e-4, float(x["score"])) for x in mm) / wsum
            cz = sum(float(x["box"][2]) * max(1e-4, float(x["score"])) for x in mm) / wsum
            clusters[best_k]["center"] = [float(cx), float(cy), float(cz)]
            stats["attach_cluster"] += 1

    merged: List[Dict[str, Any]] = []
    for clu in clusters:
        mm = clu["members"]
        w = np.asarray([max(1e-4, float(x["score"])) for x in mm], dtype=np.float64)
        b = np.asarray([x["box"] for x in mm], dtype=np.float64)
        box = (w[:, None] * b).sum(axis=0) / max(1e-8, float(w.sum()))
        srcs = sorted(set(str(x["source"]) for x in mm))
        best_idx = int(np.argmax(np.asarray([float(x["score"]) for x in mm], dtype=np.float64)))
        best_attr = str(mm[best_idx].get("attr", ""))
        fused_score = _fuse_score_noisy_or(
            mm,
            source_weights=source_weights,
            consensus_bonus=float(consensus_bonus),
        )
        merged.append(
            {
                "box": [float(v) for v in box.tolist()],
                "label": int(clu["label"]),
                "score": float(fused_score),
                "attr": best_attr,
                "sources_list": srcs,
                "source": "+".join(srcs),
                "num_sources": int(len(srcs)),
                "cluster_size": int(len(mm)),
            }
        )
        stats["merged_cluster"] += 1

    merged.sort(key=lambda x: float(x["score"]), reverse=True)
    stats["merged_total"] = len(merged)
    return merged, stats


def _to_det_field(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "boxes_3d": [c["box"] for c in cands],
        "labels_3d": [int(c["label"]) for c in cands],
        "scores_3d": [float(c["score"]) for c in cands],
        "attrs": [str(c.get("attr", "")) for c in cands],
        "sources": [str(c.get("source", "")) for c in cands],
        "num_sources": [int(c.get("num_sources", 1)) for c in cands],
        "cluster_size": [int(c.get("cluster_size", 1)) for c in cands],
    }


def _parse_class_float_map(s: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    txt = str(s or "").strip()
    if txt == "":
        return out
    for seg in txt.split(","):
        seg = seg.strip()
        if seg == "" or ":" not in seg:
            continue
        k, v = seg.split(":", 1)
        try:
            out[int(k.strip())] = float(v.strip())
        except Exception:
            continue
    return out


def _classwise_second_dedup(
    cands: List[Dict[str, Any]],
    *,
    default_dist_xy: float,
    class_dist_xy: Dict[int, float],
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Class-wise secondary dedup on merged candidates.
    Keep priority:
      num_sources (desc), score (desc), cluster_size (desc)
    """
    stats: Counter = Counter()
    if len(cands) <= 1:
        stats["dedup_kept"] = len(cands)
        return cands, stats

    order = sorted(
        range(len(cands)),
        key=lambda i: (
            int(cands[i].get("num_sources", 1)),
            float(cands[i].get("score", 0.0)),
            int(cands[i].get("cluster_size", 1)),
        ),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    kept_xy: Dict[int, List[Tuple[float, float]]] = {}
    default_r = float(max(default_dist_xy, 1e-6))

    for idx in order:
        c = cands[idx]
        cls = int(c.get("label", -1))
        b = c.get("box", [0.0, 0.0, 0.0])
        x, y = float(b[0]), float(b[1])
        rr = float(class_dist_xy.get(cls, default_r))
        thr2 = rr * rr

        dup = False
        for kx, ky in kept_xy.get(cls, []):
            d2 = (x - kx) * (x - kx) + (y - ky) * (y - ky)
            if d2 <= thr2:
                dup = True
                break
        if dup:
            stats["dedup_drop"] += 1
            continue
        kept.append(c)
        kept_xy.setdefault(cls, []).append((x, y))
        stats["dedup_kept"] += 1

    kept.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return kept, stats


def _prune_low_quality_single_source(
    cands: List[Dict[str, Any]],
    *,
    single_score_thr: float,
    single_isolated_score_thr: float,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Prune weak single-source candidates:
      1) num_sources==1 and score < single_score_thr
      2) num_sources==1 and cluster_size<=1 and score < single_isolated_score_thr
    """
    stats: Counter = Counter()
    if len(cands) == 0:
        return cands, stats
    out: List[Dict[str, Any]] = []
    for c in cands:
        ns = int(c.get("num_sources", 1))
        sc = float(c.get("score", 0.0))
        cs = int(c.get("cluster_size", 1))
        if ns == 1 and sc < float(single_score_thr):
            stats["single_drop_low_score"] += 1
            continue
        if ns == 1 and cs <= 1 and sc < float(single_isolated_score_thr):
            stats["single_drop_low_isolated"] += 1
            continue
        out.append(c)
        stats["single_keep"] += 1
    return out, stats


def _default_output_name(aligned_json: str) -> str:
    stem = os.path.splitext(os.path.basename(aligned_json))[0]
    return f"{stem}_fused_candidates.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fuse ISFusion/camera/lidar detections per sample.")
    p.add_argument(
        "--aligned_json",
        type=str,
        default="/home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/model/late_fusion/output/sorted_by_scene_ISFUSIONandGTattr_val_3_modality_aligned.json",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/model/late_fusion/output",
    )
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--summary_name", type=str, default=None)
    p.add_argument("--label_map_json", type=str, default=None)
    p.add_argument("--unknown_label_id", type=int, default=-1)

    p.add_argument("--camera_score_min", type=float, default=0.02)
    p.add_argument("--lidar_score_min", type=float, default=0.02)
    p.add_argument("--isf_score_min", type=float, default=0.0)

    p.add_argument("--isf_score_temp", type=float, default=1.0)
    p.add_argument("--camera_score_temp", type=float, default=1.0)
    p.add_argument("--lidar_score_temp", type=float, default=1.0)
    p.add_argument("--isf_score_bias", type=float, default=0.0)
    p.add_argument("--camera_score_bias", type=float, default=0.0)
    p.add_argument("--lidar_score_bias", type=float, default=0.0)

    p.add_argument("--raw_coord_frame", type=str, default="auto", choices=["auto", "global", "lidar"])
    p.add_argument("--merge_dist_xy", type=float, default=0.9)
    p.add_argument("--merge_dist_z", type=float, default=1.5)
    p.add_argument("--consensus_bonus", type=float, default=0.02)
    p.add_argument("--source_weight_isfusion", type=float, default=1.0)
    p.add_argument("--source_weight_camera", type=float, default=1.0)
    p.add_argument("--source_weight_lidar", type=float, default=1.0)
    p.add_argument("--second_dedup", action="store_true",
                   help="Enable class-wise secondary dedup after merge.")
    p.add_argument("--second_dedup_dist_xy", type=float, default=0.6,
                   help="Default class-wise secondary dedup radius (meters).")
    p.add_argument("--second_dedup_class_dist_xy", type=str,
                   default="0:0.8,1:0.9,2:0.9,3:1.0,4:1.0,5:0.4,6:0.45,7:0.45,8:0.45,9:0.35",
                   help="Per-class dedup radius map, e.g. '0:1.0,8:0.6,9:0.5'.")
    p.add_argument("--prune_single_source", action="store_true",
                   help="Enable low-quality single-source pruning.")
    p.add_argument("--single_source_score_thr", type=float, default=0.02,
                   help="Drop if num_sources==1 and score below this.")
    p.add_argument("--single_source_isolated_score_thr", type=float, default=0.04,
                   help="Drop if num_sources==1 and cluster_size==1 and score below this.")

    p.add_argument("--max_out_per_frame", type=int, default=0, help="0 means no cap.")
    p.add_argument("--keep_original_det", action="store_true")
    p.add_argument("--indent", type=int, default=2)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    _ensure_dir(args.output_dir)
    out_name = args.output_name if args.output_name else _default_output_name(args.aligned_json)
    out_path = os.path.join(args.output_dir, out_name)
    out_stem = os.path.splitext(out_name)[0]
    summary_name = args.summary_name if args.summary_name else f"{out_stem}.summary.json"
    summary_path = os.path.join(args.output_dir, summary_name)

    label_map = dict(DEFAULT_LABEL_MAP)
    if args.label_map_json:
        user_map = _load_json(args.label_map_json)
        if not isinstance(user_map, dict):
            raise ValueError("label_map_json must be dict.")
        label_map = {str(k): int(v) for k, v in user_map.items()}

    root = _load_json(args.aligned_json)
    if not isinstance(root, dict):
        raise ValueError("aligned_json root must be dict.")
    scenes = root.get("scenes", [])
    if not isinstance(scenes, list):
        raise ValueError("aligned_json missing scenes list.")

    global_stats: Counter = Counter()
    source_counter: Counter = Counter()
    label_counter: Counter = Counter()
    kept_scene_count = 0
    total_samples = 0

    source_weights = {
        "isfusion": float(args.source_weight_isfusion),
        "camera": float(args.source_weight_camera),
        "lidar": float(args.source_weight_lidar),
    }
    dedup_class_dist = _parse_class_float_map(str(args.second_dedup_class_dist_xy))

    out_root = copy.deepcopy(root)
    out_scenes: List[Dict[str, Any]] = []

    for sc in scenes:
        if not isinstance(sc, dict):
            continue
        samples = sc.get("samples", [])
        if not isinstance(samples, list):
            samples = []
        new_samples: List[Dict[str, Any]] = []
        for smp in samples:
            if not isinstance(smp, dict):
                continue
            total_samples += 1
            det_isf = smp.get("det_isfusion", smp.get("det", {}))
            det_cam_raw = smp.get("det_camera_raw", [])
            det_lid_raw = smp.get("det_lidar_raw", [])

            isf_norm = _normalize_isfusion_det(
                det_isf if isinstance(det_isf, dict) else {},
                score_temp=float(args.isf_score_temp),
                score_bias=float(args.isf_score_bias),
            )
            if float(args.isf_score_min) > 0.0:
                isf_norm = [c for c in isf_norm if float(c["score_raw"]) >= float(args.isf_score_min)]

            cam_norm, cam_stats = _normalize_detector_raw(
                det_cam_raw if isinstance(det_cam_raw, list) else [],
                sample=smp,
                source="camera",
                label_map=label_map,
                unknown_label_id=int(args.unknown_label_id),
                score_min=float(args.camera_score_min),
                score_temp=float(args.camera_score_temp),
                score_bias=float(args.camera_score_bias),
                raw_coord_frame=str(args.raw_coord_frame),
                isf_ref=isf_norm,
            )
            lid_norm, lid_stats = _normalize_detector_raw(
                det_lid_raw if isinstance(det_lid_raw, list) else [],
                sample=smp,
                source="lidar",
                label_map=label_map,
                unknown_label_id=int(args.unknown_label_id),
                score_min=float(args.lidar_score_min),
                score_temp=float(args.lidar_score_temp),
                score_bias=float(args.lidar_score_bias),
                raw_coord_frame=str(args.raw_coord_frame),
                isf_ref=isf_norm,
            )

            global_stats.update({f"cam_{k}": int(v) for k, v in cam_stats.items()})
            global_stats.update({f"lid_{k}": int(v) for k, v in lid_stats.items()})
            global_stats["isf_kept"] += int(len(isf_norm))

            merged, merge_stats = _cluster_and_merge(
                isf_norm + cam_norm + lid_norm,
                merge_dist_xy=float(args.merge_dist_xy),
                merge_dist_z=float(args.merge_dist_z),
                source_weights=source_weights,
                consensus_bonus=float(args.consensus_bonus),
            )
            global_stats.update({f"merge_{k}": int(v) for k, v in merge_stats.items()})

            if bool(args.second_dedup):
                merged, d_stats = _classwise_second_dedup(
                    merged,
                    default_dist_xy=float(args.second_dedup_dist_xy),
                    class_dist_xy=dedup_class_dist,
                )
                global_stats.update({f"post_{k}": int(v) for k, v in d_stats.items()})

            if bool(args.prune_single_source):
                merged, p_stats = _prune_low_quality_single_source(
                    merged,
                    single_score_thr=float(args.single_source_score_thr),
                    single_isolated_score_thr=float(args.single_source_isolated_score_thr),
                )
                global_stats.update({f"post_{k}": int(v) for k, v in p_stats.items()})

            if int(args.max_out_per_frame) > 0 and len(merged) > int(args.max_out_per_frame):
                merged = merged[: int(args.max_out_per_frame)]
                global_stats["merge_truncated_frames"] += 1

            for c in merged:
                source_counter[str(c["source"])] += 1
                label_counter[int(c["label"])] += 1

            new_smp = copy.deepcopy(smp)
            if bool(args.keep_original_det):
                new_smp["det_before_merge"] = copy.deepcopy(new_smp.get("det", {}))
            new_smp["det"] = _to_det_field(merged)
            new_samples.append(new_smp)

        if len(new_samples) > 0:
            kept_scene_count += 1
            new_sc = copy.deepcopy(sc)
            new_sc["samples"] = new_samples
            new_sc["num_samples"] = len(new_samples)
            out_scenes.append(new_sc)

    out_root["scenes"] = out_scenes
    if isinstance(out_root.get("scene_count"), int):
        out_root["scene_count"] = int(kept_scene_count)
    out_root["merge_meta"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "cluster_fusion_isfusion_camera_lidar",
        "raw_coord_frame": str(args.raw_coord_frame),
        "merge_dist_xy": float(args.merge_dist_xy),
        "merge_dist_z": float(args.merge_dist_z),
        "consensus_bonus": float(args.consensus_bonus),
        "source_weights": source_weights,
        "score_calibration": {
            "isfusion": {"temp": float(args.isf_score_temp), "bias": float(args.isf_score_bias)},
            "camera": {"temp": float(args.camera_score_temp), "bias": float(args.camera_score_bias)},
            "lidar": {"temp": float(args.lidar_score_temp), "bias": float(args.lidar_score_bias)},
        },
        "post_filter": {
            "second_dedup": bool(args.second_dedup),
            "second_dedup_dist_xy": float(args.second_dedup_dist_xy),
            "second_dedup_class_dist_xy": dedup_class_dist,
            "prune_single_source": bool(args.prune_single_source),
            "single_source_score_thr": float(args.single_source_score_thr),
            "single_source_isolated_score_thr": float(args.single_source_isolated_score_thr),
        },
    }

    _save_json(out_path, out_root, indent=int(args.indent))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "aligned_json": os.path.abspath(args.aligned_json),
            "label_map_json": os.path.abspath(args.label_map_json) if args.label_map_json else None,
        },
        "outputs": {
            "fused_json": os.path.abspath(out_path),
            "summary_json": os.path.abspath(summary_path),
        },
        "config": {
            "camera_score_min": float(args.camera_score_min),
            "lidar_score_min": float(args.lidar_score_min),
            "isf_score_min": float(args.isf_score_min),
            "raw_coord_frame": str(args.raw_coord_frame),
            "merge_dist_xy": float(args.merge_dist_xy),
            "merge_dist_z": float(args.merge_dist_z),
            "consensus_bonus": float(args.consensus_bonus),
            "second_dedup": bool(args.second_dedup),
            "second_dedup_dist_xy": float(args.second_dedup_dist_xy),
            "second_dedup_class_dist_xy": dedup_class_dist,
            "prune_single_source": bool(args.prune_single_source),
            "single_source_score_thr": float(args.single_source_score_thr),
            "single_source_isolated_score_thr": float(args.single_source_isolated_score_thr),
            "max_out_per_frame": int(args.max_out_per_frame),
            "keep_original_det": bool(args.keep_original_det),
        },
        "stats": {
            "scenes": int(kept_scene_count),
            "samples": int(total_samples),
            "source_counts_after_merge": {k: int(v) for k, v in sorted(source_counter.items())},
            "label_counts_after_merge": {str(k): int(v) for k, v in sorted(label_counter.items(), key=lambda x: x[0])},
            **{str(k): int(v) for k, v in sorted(global_stats.items(), key=lambda x: str(x[0]))},
        },
    }
    _save_json(summary_path, summary, indent=int(args.indent))

    print(f"fused_json: {out_path}")
    print(f"summary_json: {summary_path}")
    print(
        f"samples={summary['stats']['samples']} "
        f"merged_total={summary['stats'].get('merge_merged_total', 0)} "
        f"sources={summary['stats']['source_counts_after_merge']}"
    )


if __name__ == "__main__":
    main()
