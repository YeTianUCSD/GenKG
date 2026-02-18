#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 IS-Fusion det 的简单 3D tracking + 插值/外推脚本：

- 输入：一个带有场景结构的 JSON（包含 det + gt + 位姿等），结构不改，只更新 det 字段：
    det = {
        "boxes_3d": [[x,y,z,dx,dy,dz,yaw,vx,vy], ...],
        "scores_3d": [...],
        "labels_3d": [...]
    }

- 对每个 scene：
    1) 按 timestamp 排好所有帧；
    2) 在「全局坐标系」下，用 det 的中心点 (x,y) + 类别 做简单的 greedy tracking；
    3) 对每条 track 上的相邻观测，如果中间缺少 <= max_gap 帧：
        - 在缺失帧的 timestamp 处，对全局中心做线性插值；
        - 再用该帧的位姿把插值得到的全局中心变回当前帧的 lidar 坐标；
        - 拿前/后观测的 box 作为模板，只替换 x,y,z，生成一个插值 box；
        - score 取两端 score 的较小值，label 用 track 的类别。
    4) 把这些插值 box 追加到对应帧的 det 里（原来的 det 不动，只是增加）。

- 输出：结构完全保持的 JSON，新插值框已经写入每个 sample 的 det 中。

注意：
- tracking 只用 det 的中心点 + 类别，在「全局坐标」下做欧式距离匹配；
- 插值只保证中心点在全局系里连续，dx,dy,dz,yaw,vx,vy 直接沿用邻近观测的模板，
  不影响你之后用“中心点 + 类别”度量 P/R/F1 的逻辑。

python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/model/track_and_interpV2.py \
  --in_json  /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/sorted_by_scene_ISFUSIONres_GT_attributions_train.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/isfusionres_tracked/ISFUSIONandGTattr_train_40_match2.json \
  --match_thr 2 \
  --max_age 40 \
  --max_gap 40


python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/model/track_and_interpV2.py \
  --in_json  /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/isfusionres_tracked/ISFUSIONandGTattr_val_40_match2.json \
  --match_thr 2 \
  --max_age 40 \
  --max_gap 40
  
"""

import argparse
import json
import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


# ============================= 几何/位姿工具 =============================

def quat_to_rot(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵"""
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def make_T(translation, quat_wxyz):
    """平移 + 四元数 -> 4x4 齐次变换矩阵"""
    R = quat_to_rot(quat_wxyz)
    t = np.array(translation, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def T_inv(T):
    """4x4 齐次变换矩阵求逆"""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T, pts):
    """用 4x4 变换矩阵 T 变换 Nx3 点"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(-1, 3)
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])
    out = (T @ homo.T).T
    return out[:, :3]


# ============================= JSON 解析 =============================

def iter_samples(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    在 JSON 结构中递归查找“样本节点”：
    简单规则：同时拥有 det/gt 且 det.boxes_3d 是 list 的 dict。
    """
    if isinstance(obj, dict):
        det = obj.get("det", None)
        gt = obj.get("gt", None)
        if isinstance(det, dict) and isinstance(gt, dict) and isinstance(det.get("boxes_3d"), list):
            yield obj
        for v in obj.values():
            yield from iter_samples(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_samples(it)


def get_timestamp(sample: Dict[str, Any]) -> int:
    """统一取时间戳（优先 timestamp，其次 timestamp_us）"""
    if "timestamp" in sample:
        return int(sample["timestamp"])
    if "timestamp_us" in sample:
        return int(sample["timestamp_us"])
    # 兜底
    return int(sample.get("timestamp_us", 0))


def group_by_scene(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    scenes: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        scene_tok = s.get("scene_token", None)
        if scene_tok is None:
            continue
        scenes.setdefault(scene_tok, []).append(s)
    # 每个 scene 内按时间排序
    for k, lst in scenes.items():
        lst.sort(key=get_timestamp)
    return scenes


# ============================= tracking 相关 =============================

class Track:
    def __init__(self, tid: int, label: int,
                 frame_idx: int, center_global: np.ndarray,
                 score: float, timestamp: int, det_idx: int):
        self.id = tid
        self.label = int(label)
        # obs: (frame_idx, det_idx, center_global(3,), score, timestamp)
        self.obs: List[Tuple[int, int, np.ndarray, float, int]] = []
        self.add_observation(frame_idx, det_idx, center_global, score, timestamp)
        self.last_frame = frame_idx
        self.last_center = center_global.copy()
        self.last_timestamp = timestamp
        self.miss_count = 0
        self.active = True

    def add_observation(self, frame_idx: int, det_idx: int,
                        center_global: np.ndarray, score: float, timestamp: int):
        self.obs.append((frame_idx, det_idx, center_global.copy(), float(score), int(timestamp)))
        self.last_frame = frame_idx
        self.last_center = center_global.copy()
        self.last_timestamp = int(timestamp)
        self.miss_count = 0

    def mark_missed(self):
        self.miss_count += 1


def match_tracks_to_dets(track_centers_xy: np.ndarray,
                         track_labels: np.ndarray,
                         det_centers_xy: np.ndarray,
                         det_labels: np.ndarray,
                         thr: float) -> Tuple[List[int], List[int]]:
    """
    类别感知 + 2D 欧氏距离 greedy 匹配：
    - track_centers_xy: [T,2]
    - det_centers_xy:   [N,2]
    - track_labels:     [T]
    - det_labels:       [N]
    返回：
    - track_to_det: 长度 T，若匹配则为 det 索引，否则为 -1
    - unmatched_det_idxs: list，未被匹配上的 det 索引
    """
    T = track_centers_xy.shape[0]
    N = det_centers_xy.shape[0]
    if T == 0 or N == 0:
        return [-1] * T, list(range(N))

    # [T,N] 距离矩阵
    a = track_centers_xy[:, None, :]  # [T,1,2]
    b = det_centers_xy[None, :, :]    # [1,N,2]
    dist = np.linalg.norm(a - b, axis=2)  # [T,N]

    # 类别不一致的设为很大
    label_equal = (track_labels[:, None] == det_labels[None, :])
    dist[~label_equal] = 1e9

    # 只考虑 dist <= thr 的候选
    pairs = []
    for i in range(T):
        for j in range(N):
            d = dist[i, j]
            if d <= thr:
                pairs.append((float(d), i, j))
    pairs.sort(key=lambda x: x[0])

    track_to_det = [-1] * T
    used_tracks = set()
    used_dets = set()

    for d, ti, dj in pairs:
        if ti in used_tracks or dj in used_dets:
            continue
        track_to_det[ti] = dj
        used_tracks.add(ti)
        used_dets.add(dj)

    unmatched = [j for j in range(N) if j not in used_dets]
    return track_to_det, unmatched


def build_tracks_for_scene(frames_meta: List[Dict[str, Any]],
                           match_thr: float = 2.0,
                           max_age: int = 5) -> List[Track]:
    """
    在一个 scene 内做简单 3D tracking（实际用的是全局坐标系下的 2D 中心点 + 类别）。
    frames_meta: 每帧的信息 list（见 main 中构造）
    """
    next_tid = 0
    active_tracks: List[Track] = []
    all_tracks: List[Track] = []

    for f_idx, fm in enumerate(frames_meta):
        centers_g = fm["centers_global"]  # [Ni,3]
        labels = fm["labels"]             # [Ni]
        scores = fm["scores"]             # [Ni]
        ts = fm["timestamp"]

        # 当前帧的 det
        Ni = centers_g.shape[0]

        if len(active_tracks) > 0 and Ni > 0:
            track_centers_xy = np.stack([t.last_center[:2] for t in active_tracks], axis=0)
            track_labels = np.array([t.label for t in active_tracks], dtype=np.int64)
            det_centers_xy = centers_g[:, :2]
            det_labels = labels

            track_to_det, unmatched_det_idxs = match_tracks_to_dets(
                track_centers_xy, track_labels, det_centers_xy, det_labels, thr=match_thr
            )
        else:
            track_to_det = []
            unmatched_det_idxs = list(range(Ni))

        # 先更新已有 track
        for ti, trk in enumerate(active_tracks):
            if Ni > 0 and ti < len(track_to_det) and track_to_det[ti] != -1:
                dj = track_to_det[ti]
                trk.add_observation(
                    frame_idx=f_idx,
                    det_idx=int(dj),
                    center_global=centers_g[dj],
                    score=float(scores[dj]),
                    timestamp=int(ts),
                )
            else:
                trk.mark_missed()

        # 结束太久没匹配的 track
        still_active = []
        for trk in active_tracks:
            if trk.miss_count > max_age:
                trk.active = False
                all_tracks.append(trk)
            else:
                still_active.append(trk)
        active_tracks = still_active

        # 对未匹配的 det，新开 track
        for dj in unmatched_det_idxs:
            if Ni == 0:
                break
            new_trk = Track(
                tid=next_tid,
                label=int(labels[dj]),
                frame_idx=f_idx,
                center_global=centers_g[dj],
                score=float(scores[dj]),
                timestamp=int(ts),
                det_idx=int(dj),
            )
            active_tracks.append(new_trk)
            next_tid += 1

    # 剩余仍活跃的也加入 all_tracks
    all_tracks.extend(active_tracks)
    return all_tracks


# ============================= 插值 / 外推 =============================

def interpolate_tracks_for_scene(
    frames_meta: List[Dict[str, Any]],
    tracks: List[Track],
    max_gap: int = 2,
) -> Dict[int, List[Tuple[np.ndarray, float, int]]]:
    """
    对每条 track，在相邻观测之间插值，生成新 box：
    返回：frame_idx -> list[(new_box(9,), new_score, new_label)]
    """
    frame_new_dets: Dict[int, List[Tuple[np.ndarray, float, int]]] = {}

    for trk in tracks:
        if len(trk.obs) < 2:
            continue
        # 按 frame_idx 排序
        obs_sorted = sorted(trk.obs, key=lambda o: o[0])  # (frame_idx, det_idx, center_g, score, ts)

        for k in range(len(obs_sorted) - 1):
            f0, d0, c0, s0, t0 = obs_sorted[k]
            f1, d1, c1, s1, t1 = obs_sorted[k + 1]

            # 中间没有缺帧
            if f1 <= f0 + 1:
                continue

            gap = f1 - f0 - 1
            if gap > max_gap:
                # 缺太多帧就不插值，避免瞎补
                continue

            # 遍历中间缺失的帧：f = f0+1 ... f1-1
            for f in range(f0 + 1, f1):
                fm = frames_meta[f]
                tf = fm["timestamp"]
                if t1 == t0:
                    alpha = 0.5
                else:
                    alpha = float(tf - t0) / float(t1 - t0)
                    alpha = max(0.0, min(1.0, alpha))

                # 全局中心线性插值
                center_g = (1.0 - alpha) * c0 + alpha * c1  # [3]

                # 变回当前帧的 lidar 坐标
                T_g2l = fm["T_g2l"]
                center_l = transform_points(T_g2l, center_g.reshape(1, 3))[0]

                # 选择模板 box：距离哪端时间更近就用哪端
                if abs(tf - t0) <= abs(t1 - tf):
                    tpl_frame_idx, tpl_det_idx, tpl_score = f0, d0, s0
                else:
                    tpl_frame_idx, tpl_det_idx, tpl_score = f1, d1, s1

                tpl_boxes = frames_meta[tpl_frame_idx]["boxes"]
                if tpl_boxes.shape[0] == 0:
                    continue
                tpl_box = tpl_boxes[tpl_det_idx].copy()  # [9]

                # 替换中心
                tpl_box[:3] = center_l.astype(np.float32)

                # score 保守一点，用两端的最小值
                new_score = float(min(s0, s1))
                new_label = trk.label

                frame_new_dets.setdefault(f, []).append((tpl_box, new_score, new_label))

    return frame_new_dets


# ============================= 主流程 =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="输入 JSON 路径（含 det + gt + 位姿）")
    ap.add_argument("--out_json", required=True, help="输出 JSON 路径（写入插值后的 det）")
    ap.add_argument("--match_thr", type=float, default=2.0,
                    help="tracking 时，在全局坐标系下的 XY 匹配距离阈值（米）")
    ap.add_argument("--max_age", type=int, default=5,
                    help="track 允许连续多少帧没匹配到，再认为终止")
    ap.add_argument("--max_gap", type=int, default=2,
                    help="只对 gap <= max_gap 的缺帧做插值（单位：帧数）")
    args = ap.parse_args()

    print(f"[1/4] Loading JSON from {args.in_json} ...")
    with open(args.in_json, "r") as f:
        root = json.load(f)

    print("[2/4] Collecting samples...")
    samples = list(iter_samples(root))
    print(f"  -> Found {len(samples)} samples with det+gt.")

    # 按 scene 分组并排序
    scenes = group_by_scene(samples)
    print(f"[2/4] Scenes: {len(scenes)}")

    # 逐个 scene 做 tracking + 插值
    for si, (scene_token, scene_frames) in enumerate(scenes.items(), 1):
        print(f"[3/4] Processing scene {si}/{len(scenes)}: {scene_token}  (#frames={len(scene_frames)})")

        # 为该 scene 的每一帧预计算一些信息
        frames_meta: List[Dict[str, Any]] = []
        for idx, s in enumerate(scene_frames):
            ts = get_timestamp(s)

            det = s.get("det") or {}
            boxes = np.array(det.get("boxes_3d") or [], dtype=np.float32)
            scores = np.array(det.get("scores_3d") or [], dtype=np.float32)
            labels = np.array(det.get("labels_3d") or [], dtype=np.int64)

            if boxes.ndim == 1:
                boxes = boxes.reshape(-1, 9)
            if scores.ndim == 0 and boxes.shape[0] > 0:
                scores = np.full((boxes.shape[0],), float(scores), dtype=np.float32)
            if labels.ndim == 0 and boxes.shape[0] > 0:
                labels = np.full((boxes.shape[0],), int(labels), dtype=np.int64)

            # 计算 lidar->global 以及其逆
            lidar2ego = s.get("lidar2ego", {})
            ego2global = s.get("ego2global", {})

            T_l2e = make_T(lidar2ego.get("translation", [0, 0, 0]),
                           lidar2ego.get("rotation", [1, 0, 0, 0]))
            T_e2g = make_T(ego2global.get("translation", [0, 0, 0]),
                           ego2global.get("rotation", [1, 0, 0, 0]))
            T_l2g = T_e2g @ T_l2e
            T_g2l = T_inv(T_l2g)

            if boxes.shape[0] > 0:
                centers_l = boxes[:, :3]  # lidar 中心
                centers_g = transform_points(T_l2g, centers_l)  # global 中心
            else:
                centers_g = np.zeros((0, 3), dtype=np.float64)

            frames_meta.append({
                "sample": s,
                "timestamp": ts,
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "centers_global": centers_g,
                "T_g2l": T_g2l,
            })

        # 1) 先 build tracks（用原始 det）
        tracks = build_tracks_for_scene(
            frames_meta,
            match_thr=args.match_thr,
            max_age=args.max_age,
        )

        # 2) 对 track 中的小 gap 做插值，得到每帧要新增的 det
        frame_new_dets = interpolate_tracks_for_scene(
            frames_meta,
            tracks,
            max_gap=args.max_gap,
        )

        # 3) 把插值 det 追加写回到 JSON 中（只改 det，其他字段不动）
        for f_idx, fm in enumerate(frames_meta):
            new_list = frame_new_dets.get(f_idx, [])
            if not new_list:
                continue

            s = fm["sample"]
            det = s.get("det")
            if det is None:
                det = {}
                s["det"] = det

            boxes_3d = det.get("boxes_3d") or []
            scores_3d = det.get("scores_3d") or []
            labels_3d = det.get("labels_3d") or []

            # 确保是 list
            boxes_3d = list(boxes_3d)
            scores_3d = list(scores_3d)
            labels_3d = list(labels_3d)

            for box, score, label in new_list:
                boxes_3d.append([float(x) for x in box.tolist()])
                scores_3d.append(float(score))
                labels_3d.append(int(label))

            det["boxes_3d"] = boxes_3d
            det["scores_3d"] = scores_3d
            det["labels_3d"] = labels_3d

        print(f"  -> scene {scene_token}: added interpolated dets to {len(frame_new_dets)} frames.")

    print(f"[4/4] Writing output to {args.out_json} ...")
    with open(args.out_json, "w") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
