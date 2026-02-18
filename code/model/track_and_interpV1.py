#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 IS-Fusion 3D det 的简单 tracking + 线性插值补洞脚本：

- 输入：一个包含 det/gt/位姿等信息的大 JSON（结构跟你现有的训练/评估 JSON 一致）
- 处理：
    1）按 scene_token 分组，并在每个 scene 内按 timestamp 排序
    2）在每个 scene 内，对 det 做基于「global 坐标中心点 + label」的多目标 tracking（贪心匹配）
    3）对每条 track 中间的小空洞（gap <= max_interp_gap）做线性插值，生成新的 3D 框
    4）把这些插值出来的框 append 回对应帧的 det:
         det["boxes_3d"] / det["scores_3d"] / det["labels_3d"]

- 输出：一个新的 JSON 文件，原有结构完全保留，只是各帧的 det 里可能多了一些插值框

python /home/code/3Ddetection/IS-Fusion/GenKG/code/model/track_and_interpV1.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/isfusionres_tracked/isfusionres_tracked_interp.json \
  --match_dist 2.0 \
  --max_age 9 \
  --max_interp_gap 9

"""

from typing import List, Dict, Any, NamedTuple
from collections import defaultdict
import argparse
import json
import numpy as np


# ======================= Track 数据结构 =======================

class TrackNode(NamedTuple):
    scene_frame_idx: int   # 在本 scene 中的帧序号（0..N-1）
    sample_idx: int        # 在 samples 列表中的全局下标
    det_idx: int           # 这一帧 det 里的 index
    center_global: np.ndarray  # (3,)
    score: float


class Track:
    def __init__(self, track_id: int, label: int):
        self.id = track_id
        self.label = int(label)
        self.nodes: List[TrackNode] = []
        self.last_scene_frame_idx: int = -1
        self.missed: int = 0  # 连续多少帧没匹配到

    def add_node(self, node: TrackNode):
        self.nodes.append(node)
        self.last_scene_frame_idx = node.scene_frame_idx
        self.missed = 0

    def last_center_global(self) -> np.ndarray:
        return self.nodes[-1].center_global


# ======================= 位姿相关工具 =======================

def quat_to_rot(q: List[float]) -> np.ndarray:
    """四元数(w,x,y,z) -> 3x3 旋转矩阵"""
    w, x, y, z = q
    n = (w*w + x*x + y*y + z*z) ** 0.5
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [    2*(x*z - y*w), 2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def make_T(translation: List[float], quat_wxyz: List[float]) -> np.ndarray:
    """平移 + 四元数 -> 4x4 齐次变换矩阵"""
    R = quat_to_rot(quat_wxyz)
    t = np.array(translation, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """用 4x4 齐次变换矩阵变换一组 3D 点"""
    if pts.size == 0:
        return pts
    pts = np.asarray(pts, dtype=np.float64)
    N = pts.shape[0]
    homo = np.concatenate([pts, np.ones((N, 1), dtype=np.float64)], axis=1)
    out = (T @ homo.T).T
    return out[:, :3]


# ======================= JSON 样本遍历 =======================

def iter_samples(obj: Any):
    """
    递归遍历 JSON，找到包含 det/gt 的“样本”节点。
    这里的逻辑和你训练脚本里的 iter_samples 保持一致：
      - obj 是 dict 且包含 det / gt 且 det.boxes_3d 是 list -> 视为一个 sample
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


def get_timestamp(sample: Dict[str, Any]) -> int:
    """统一取时间戳（优先 timestamp，没有就用 timestamp_us）"""
    if "timestamp" in sample:
        return int(sample["timestamp"])
    if "timestamp_us" in sample:
        return int(sample["timestamp_us"])
    return 0


def get_T_lidar_to_global(sample: Dict[str, Any]) -> np.ndarray:
    """从 sample 中取出 lidar2ego / ego2global，组成 lidar -> global 的变换"""
    lidar2ego = sample["lidar2ego"]
    ego2global = sample["ego2global"]
    Tl2e = make_T(lidar2ego["translation"], lidar2ego["rotation"])
    Te2g = make_T(ego2global["translation"], ego2global["rotation"])
    return Te2g @ Tl2e   # lidar -> ego -> global


# ======================= 单个 scene 的 tracking =======================

def build_tracks_for_scene(scene_token: str,
                           scene_frame_indices: List[int],
                           samples: List[Dict[str, Any]],
                           match_dist: float = 2.0,
                           max_age: int = 3) -> List[Track]:
    """
    对单个 scene 里的所有帧做基于 3D 中心点的简单 tracking：
      - 先把每一帧的 det 中心点从 lidar 坐标变到 global 坐标
      - 用 label + XY 距离（<= match_dist）在相邻帧之间贪心匹配
      - track 连续 max_age 帧没有匹配到新的 det 就终止

    返回该 scene 中所有的 track 列表。
    """

    next_track_id = 0
    active_tracks: List[Track] = []
    finished_tracks: List[Track] = []

    # 缓存这一 scene 各帧的 det 信息（为了重复使用）
    frame_det_centers_g: Dict[int, np.ndarray] = {}
    frame_det_labels: Dict[int, np.ndarray] = {}
    frame_det_scores: Dict[int, np.ndarray] = {}

    for local_f_idx, sample_idx in enumerate(scene_frame_indices):
        s = samples[sample_idx]
        det = s.get("det") or {}
        boxes = np.asarray(det.get("boxes_3d") or [], dtype=np.float64)
        labels = np.asarray(det.get("labels_3d") or [], dtype=np.int64)
        scores = np.asarray(det.get("scores_3d") or [], dtype=np.float32)

        if boxes.shape[0] == 0:
            centers_g = np.zeros((0, 3), dtype=np.float64)
        else:
            T_l2g = get_T_lidar_to_global(s)
            centers_l = boxes[:, :3]
            centers_g = transform_points(T_l2g, centers_l)

        frame_det_centers_g[local_f_idx] = centers_g
        frame_det_labels[local_f_idx] = labels
        frame_det_scores[local_f_idx] = scores

    # 正式 tracking
    for local_f_idx, sample_idx in enumerate(scene_frame_indices):
        centers_g = frame_det_centers_g[local_f_idx]
        labels = frame_det_labels[local_f_idx]
        scores = frame_det_scores[local_f_idx]
        num_det = centers_g.shape[0]

        # 这一帧没有任何 det：只更新 active_tracks 的 miss 计数
        if num_det == 0:
            still_active: List[Track] = []
            for trk in active_tracks:
                trk.missed += 1
                if trk.missed <= max_age:
                    still_active.append(trk)
                else:
                    finished_tracks.append(trk)
            active_tracks = still_active
            continue

        # === 计算 active_tracks 与当前帧 det 的距离矩阵 ===
        num_trk = len(active_tracks)
        assigned_tracks = set()
        assigned_dets = set()

        if num_trk > 0:
            trk_centers = np.stack([t.last_center_global() for t in active_tracks], axis=0)  # [T,3]
            # 只用 XY 距离
            diff = trk_centers[:, None, :2] - centers_g[None, :, :2]  # [T,D,2]
            dist = np.linalg.norm(diff, axis=2)  # [T,D]

            # label 不同的直接屏蔽（设置成一个很大值）
            trk_labels = np.array([t.label for t in active_tracks], dtype=np.int64)
            det_labels = labels
            for ti in range(num_trk):
                for di in range(num_det):
                    if trk_labels[ti] != det_labels[di]:
                        dist[ti, di] = 1e9

            # 收集所有 <= match_dist 的候选对，按距离升序排序，做一对一贪心匹配
            candidates = []
            for ti in range(num_trk):
                for di in range(num_det):
                    d = float(dist[ti, di])
                    if d <= match_dist:
                        candidates.append((d, ti, di))
            candidates.sort(key=lambda x: x[0])

            for d, ti, di in candidates:
                if ti in assigned_tracks or di in assigned_dets:
                    continue
                assigned_tracks.add(ti)
                assigned_dets.add(di)
                trk = active_tracks[ti]
                node = TrackNode(
                    scene_frame_idx=local_f_idx,
                    sample_idx=sample_idx,
                    det_idx=int(di),
                    center_global=centers_g[di].copy(),
                    score=float(scores[di]) if scores.size > 0 else 1.0,
                )
                trk.add_node(node)

        # 没有匹配到 det 的 track：miss+1，超龄则结束
        still_active: List[Track] = []
        for ti, trk in enumerate(active_tracks):
            if ti in assigned_tracks:
                continue
            trk.missed += 1
            if trk.missed <= max_age:
                still_active.append(trk)
            else:
                finished_tracks.append(trk)
        active_tracks = still_active

        # 当前帧未被任何 track 使用的 det：各自新开一个 track
        for di in range(num_det):
            if di in assigned_dets:
                continue
            label = int(labels[di]) if labels.size > 0 else -1
            trk = Track(track_id=next_track_id, label=label)
            next_track_id += 1
            node = TrackNode(
                scene_frame_idx=local_f_idx,
                sample_idx=sample_idx,
                det_idx=int(di),
                center_global=centers_g[di].copy(),
                score=float(scores[di]) if scores.size > 0 else 1.0,
            )
            trk.add_node(node)
            active_tracks.append(trk)

    # 所有 track 收集到一起
    finished_tracks.extend(active_tracks)
    return finished_tracks


# ======================= 单个 scene 的插值补洞 =======================

def interpolate_tracks_for_scene(scene_token: str,
                                 scene_frame_indices: List[int],
                                 samples: List[Dict[str, Any]],
                                 tracks: List[Track],
                                 max_interp_gap: int,
                                 extra_det) -> None:
    """
    在单个 scene 内，对每条 track 的中间小空洞做线性插值：
      - 如果两个节点之间的 gap = n2.scene_frame_idx - n1.scene_frame_idx - 1
      - 且 0 < gap <= max_interp_gap，则对中间这些帧补框

    补出来的框写到 extra_det[sample_idx] 里：
      extra_det: Dict[int, Dict[str, List]]
        key: sample_idx
        value: {"boxes": [...], "scores": [...], "labels": [...]}
    """

    # 预先把 timestamp 取出来
    frame_ts = {local_f_idx: get_timestamp(samples[sample_idx])
                for local_f_idx, sample_idx in enumerate(scene_frame_indices)}

    for trk in tracks:
        if len(trk.nodes) < 2:
            continue

        # 按 scene_frame_idx 排一下（理论上本来就递增，这里只是保险）
        nodes = sorted(trk.nodes, key=lambda n: n.scene_frame_idx)

        for i in range(len(nodes) - 1):
            n1 = nodes[i]
            n2 = nodes[i + 1]
            gap = n2.scene_frame_idx - n1.scene_frame_idx - 1
            if gap <= 0:
                continue
            if gap > max_interp_gap:
                continue  # 空洞太长就不补，避免瞎补

            t1 = frame_ts[n1.scene_frame_idx]
            t2 = frame_ts[n2.scene_frame_idx]
            if t2 == t1:
                # 时间戳异常，就退回用帧序号来插值
                t1 = n1.scene_frame_idx
                t2 = n2.scene_frame_idx

            for offset in range(1, gap + 1):
                mid_f_idx = n1.scene_frame_idx + offset
                sample_idx_mid = scene_frame_indices[mid_f_idx]
                s_mid = samples[sample_idx_mid]
                tm = frame_ts[mid_f_idx]

                alpha = (tm - t1) / (t2 - t1) if t2 != t1 else \
                        (mid_f_idx - n1.scene_frame_idx) / (n2.scene_frame_idx - n1.scene_frame_idx)

                # 1）在 global 坐标系下线性插值中心点
                center_g = (1.0 - alpha) * n1.center_global + alpha * n2.center_global

                # 2）根据离哪端更近，选一个模板框，拷贝 dx,dy,dz,yaw,vx,vy 等属性
                if abs(tm - t1) <= abs(t2 - tm):
                    template_node = n1
                else:
                    template_node = n2

                s_tmp = samples[template_node.sample_idx]
                det_tmp = s_tmp.get("det") or {}
                boxes_tmp = det_tmp.get("boxes_3d") or []
                if not boxes_tmp or template_node.det_idx >= len(boxes_tmp):
                    continue
                box_tmp = np.asarray(boxes_tmp[template_node.det_idx], dtype=np.float64)
                # box_tmp: [x,y,z,dx,dy,dz,yaw,vx,vy]
                attrs = box_tmp[3:]  # [dx,dy,dz,yaw,vx,vy]

                # 3）把 global center 变回当前帧的 lidar 坐标系
                T_l2g_mid = get_T_lidar_to_global(s_mid)
                T_g2l_mid = np.linalg.inv(T_l2g_mid)
                center_l = transform_points(T_g2l_mid, center_g.reshape(1, 3))[0]

                # 4）组成新的 3D 框
                new_box = np.concatenate([center_l, attrs], axis=0)
                new_score = min(n1.score, n2.score)  # 保守一点，取两端 score 的较小值
                new_label = trk.label

                ed = extra_det[sample_idx_mid]
                ed["boxes"].append(new_box.tolist())
                ed["scores"].append(float(new_score))
                ed["labels"].append(int(new_label))


# ======================= 主流程 =======================

def main():
    ap = argparse.ArgumentParser(description="基于 3D det 的简单 tracking + 线性插值补洞")
    ap.add_argument("--json", required=True, help="输入 JSON 文件路径（包含 det/gt 等结构）")
    ap.add_argument("--out_json", required=True, help="输出 JSON 文件路径")
    ap.add_argument("--match_dist", type=float, default=2.0,
                    help="帧间匹配的最大 XY 距离阈值（米）")
    ap.add_argument("--max_age", type=int, default=3,
                    help="track 连续多少帧没匹配到就终止")
    ap.add_argument("--max_interp_gap", type=int, default=1,
                    help="只对空洞长度 <= max_interp_gap 的地方做插值，例如 1 表示只补单帧缺失")
    args = ap.parse_args()

    # 1) 读 JSON
    with open(args.json, "r") as f:
        root = json.load(f)

    # 2) 收集所有 sample（保持对原始结构的引用）
    samples = list(iter_samples(root))
    if not samples:
        print("没有在 JSON 中找到任何包含 det/gt 的样本节点，直接写回原文件结构。")
        with open(args.out_json, "w") as f:
            json.dump(root, f, ensure_ascii=False, indent=2)
        return

    # 给每个样本加一个 _ts 字段方便排序
    for s in samples:
        s["_ts"] = get_timestamp(s)

    # 3) 按 scene_token 分组，并在每个 scene 内按时间排序
    scene_to_frame_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        scene = s.get("scene_token", "unknown_scene")
        scene_to_frame_indices[scene].append(idx)

    for scene, idx_list in scene_to_frame_indices.items():
        idx_list.sort(key=lambda i: samples[i]["_ts"])

    # 4) 对每个 scene 做 tracking + 插值，记录需要额外插入的 det
    extra_det: Dict[int, Dict[str, List]] = defaultdict(lambda: {"boxes": [], "scores": [], "labels": []})

    for scene, idx_list in scene_to_frame_indices.items():
        tracks = build_tracks_for_scene(scene, idx_list, samples,
                                        match_dist=args.match_dist,
                                        max_age=args.max_age)
        interpolate_tracks_for_scene(scene, idx_list, samples, tracks,
                                     max_interp_gap=args.max_interp_gap,
                                     extra_det=extra_det)

    # 5) 把插值出来的框 append 回原来的 det 结构中
    for sample_idx, extra in extra_det.items():
        if not extra["boxes"]:
            continue
        s = samples[sample_idx]
        det = s.get("det")
        if det is None:
            det = {}
            s["det"] = det

        boxes = det.get("boxes_3d") or []
        scores = det.get("scores_3d") or []
        labels = det.get("labels_3d") or []

        boxes = list(boxes)
        scores = list(scores)
        labels = list(labels)

        boxes.extend(extra["boxes"])
        scores.extend(extra["scores"])
        labels.extend(extra["labels"])

        det["boxes_3d"] = boxes
        det["scores_3d"] = scores
        det["labels_3d"] = labels

    # 6) 写回 JSON，保持原始结构，其它字段完全不动
    with open(args.out_json, "w") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
