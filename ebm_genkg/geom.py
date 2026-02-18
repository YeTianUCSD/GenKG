# ebm_refine/geom.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geometry / pose utilities for autonomous-driving JSON frames.

Conventions:
- Quaternions are assumed to be in (w, x, y, z) order (nuScenes style).
- 4x4 transforms are homogeneous matrices, acting on column vectors in homogeneous coords.
- Points are row-major Nx3 arrays; internally we convert to homogeneous Nx4.

Core helpers:
- quat_to_rot, make_T, T_inv
- transform_points, transform_vectors
- rotate_yaw, rotate_vel_xy
- T_l2g_from_sample, T_k2ref (lidar_k -> lidar_ref)
- lidar <-> global convenience wrappers
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


# -----------------------------
# Basic math helpers
# -----------------------------

def _as_np(x: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def wrap_angle_rad(a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def angle_diff_rad(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute wrapped difference a - b in (-pi, pi]."""
    return wrap_angle_rad(a - b)


# -----------------------------
# Quaternion / Transform
# -----------------------------

def quat_normalize_wxyz(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """Normalize quaternion (w,x,y,z). Returns float64 (4,)."""
    q = _as_np(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_to_rot(q_wxyz: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Convert quaternion (w,x,y,z) to 3x3 rotation matrix.

    Uses the standard Hamilton convention (as in nuScenes).
    """
    w, x, y, z = quat_normalize_wxyz(q_wxyz)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def make_T(translation_xyz: Union[List[float], np.ndarray],
           quat_wxyz: Union[List[float], np.ndarray]) -> np.ndarray:
    """Create 4x4 homogeneous transform from translation (x,y,z) and quaternion (w,x,y,z)."""
    t = _as_np(translation_xyz, dtype=np.float64).reshape(3)
    R = quat_to_rot(quat_wxyz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform."""
    T = _as_np(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, pts: Union[np.ndarray, List[List[float]], List[float]]) -> np.ndarray:
    """
    Transform points with a 4x4 transform.

    Args:
        T: (4,4)
        pts: (N,3) or (3,)
    Returns:
        (N,3)
    """
    T = _as_np(T, dtype=np.float64).reshape(4, 4)
    pts = _as_np(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(-1, 3)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])
    out = (T @ homo.T).T
    return out[:, :3]


def transform_vectors(R_or_T: np.ndarray, vecs: Union[np.ndarray, List[List[float]], List[float]]) -> np.ndarray:
    """
    Rotate vectors (no translation). Accepts:
      - R (3,3) rotation matrix
      - T (4,4) transform (uses T[:3,:3])
    """
    M = _as_np(R_or_T, dtype=np.float64)
    if M.shape == (4, 4):
        R = M[:3, :3]
    else:
        R = M.reshape(3, 3)

    v = _as_np(vecs, dtype=np.float64)
    if v.size == 0:
        return v.reshape(-1, 3)
    if v.ndim == 1:
        v = v.reshape(1, -1)

    if v.shape[1] == 2:
        v3 = np.hstack([v, np.zeros((v.shape[0], 1), dtype=np.float64)])
        out3 = (R @ v3.T).T
        return out3[:, :2]
    elif v.shape[1] == 3:
        out = (R @ v.T).T
        return out
    else:
        raise ValueError(f"transform_vectors expects vec dim 2 or 3, got {v.shape}")


# -----------------------------
# Yaw / velocity rotation
# -----------------------------

def yaw_to_unitvec_xy(yaw: np.ndarray) -> np.ndarray:
    """yaw radians -> unit vector (cos, sin) in XY plane."""
    yaw = _as_np(yaw, dtype=np.float64).reshape(-1)
    return np.stack([np.cos(yaw), np.sin(yaw)], axis=1)


def unitvec_xy_to_yaw(vxy: np.ndarray) -> np.ndarray:
    """(vx, vy) -> yaw = atan2(vy, vx)."""
    vxy = _as_np(vxy, dtype=np.float64)
    if vxy.ndim == 1:
        vxy = vxy.reshape(1, 2)
    return np.arctan2(vxy[:, 1], vxy[:, 0])


def rotate_yaw(T_or_R: np.ndarray, yaw: Union[np.ndarray, List[float], float]) -> np.ndarray:
    """
    Rotate yaw angles by applying the rotation in T_or_R to the corresponding heading vectors.

    Args:
        T_or_R: (4,4) or (3,3)
        yaw: (N,) or scalar
    Returns:
        yaw_rot: (N,)
    """
    yaw = _as_np(yaw, dtype=np.float64)
    if yaw.ndim == 0:
        yaw = yaw.reshape(1)
    vxy = yaw_to_unitvec_xy(yaw)           # (N,2)
    vxy2 = transform_vectors(T_or_R, vxy)  # (N,2)
    yaw2 = unitvec_xy_to_yaw(vxy2)
    return wrap_angle_rad(yaw2)


def rotate_vel_xy(T_or_R: np.ndarray, vxy: Union[np.ndarray, List[List[float]], List[float]]) -> np.ndarray:
    """
    Rotate planar velocity vectors (vx, vy) using the rotation part of T_or_R.
    """
    vxy = _as_np(vxy, dtype=np.float64)
    if vxy.size == 0:
        return vxy.reshape(-1, 2)
    if vxy.ndim == 1:
        vxy = vxy.reshape(1, 2)
    return transform_vectors(T_or_R, vxy)  # returns (N,2)


# -----------------------------
# Pose helpers for your JSON schema
# -----------------------------

def _get_pose_block(sample: Dict[str, Any], key: str) -> Tuple[List[float], List[float]]:
    """
    Read translation/rotation blocks like sample["lidar2ego"] or sample["ego2global"].
    Returns (translation_xyz, quat_wxyz) with sensible defaults.
    """
    blk = sample.get(key, {}) or {}
    t = blk.get("translation", [0.0, 0.0, 0.0])
    q = blk.get("rotation", [1.0, 0.0, 0.0, 0.0])
    return t, q


def T_l2g_from_sample(sample: Dict[str, Any]) -> np.ndarray:
    """
    Compute lidar->global transform for a sample:
        T_l2g = T_e2g @ T_l2e
    """
    t_l2e, q_l2e = _get_pose_block(sample, "lidar2ego")
    t_e2g, q_e2g = _get_pose_block(sample, "ego2global")
    T_l2e = make_T(t_l2e, q_l2e)
    T_e2g = make_T(t_e2g, q_e2g)
    return T_e2g @ T_l2e


def T_g2l_from_sample(sample: Dict[str, Any]) -> np.ndarray:
    """Compute global->lidar inverse transform for a sample."""
    return T_inv(T_l2g_from_sample(sample))


def T_k2ref(ref_sample: Dict[str, Any], k_sample: Dict[str, Any]) -> np.ndarray:
    """
    Compute lidar_k -> lidar_ref transform.

    Derivation:
      p_g = (T_e2g^k @ T_l2e^k) p_lk
      p_lr = (T_l2e^ref)^{-1} (T_e2g^ref)^{-1} p_g

    So:
      T_k2ref = (T_l2e^ref)^{-1} (T_e2g^ref)^{-1} (T_e2g^k) (T_l2e^k)
    """
    t_l2e_ref, q_l2e_ref = _get_pose_block(ref_sample, "lidar2ego")
    t_e2g_ref, q_e2g_ref = _get_pose_block(ref_sample, "ego2global")
    t_l2e_k,   q_l2e_k   = _get_pose_block(k_sample, "lidar2ego")
    t_e2g_k,   q_e2g_k   = _get_pose_block(k_sample, "ego2global")

    T_l2e_ref = make_T(t_l2e_ref, q_l2e_ref)
    T_e2g_ref = make_T(t_e2g_ref, q_e2g_ref)
    T_l2e_k   = make_T(t_l2e_k,   q_l2e_k)
    T_e2g_k   = make_T(t_e2g_k,   q_e2g_k)

    return T_inv(T_l2e_ref) @ T_inv(T_e2g_ref) @ T_e2g_k @ T_l2e_k


# -----------------------------
# Convenience wrappers: lidar <-> global
# -----------------------------

def lidar_to_global(sample: Dict[str, Any], xyz_lidar: Union[np.ndarray, List[List[float]], List[float]]) -> np.ndarray:
    """Transform lidar-frame points to global frame."""
    return transform_points(T_l2g_from_sample(sample), xyz_lidar)


def global_to_lidar(sample: Dict[str, Any], xyz_global: Union[np.ndarray, List[List[float]], List[float]]) -> np.ndarray:
    """Transform global-frame points to lidar frame."""
    return transform_points(T_g2l_from_sample(sample), xyz_global)


# -----------------------------
# Box helpers (optional but handy)
# -----------------------------

def transform_boxes_center(
    T: np.ndarray,
    boxes_3d: np.ndarray,
    *,
    rotate_yaw_flag: bool = True,
    rotate_vel_flag: bool = True,
) -> np.ndarray:
    """
    Transform boxes by transforming their centers (x,y,z) and optionally rotating yaw and velocity.
    This does NOT change dimensions (dx,dy,dz).

    Expected box format: [x,y,z, dx,dy,dz, yaw, vx,vy] (9,)
    """
    boxes = _as_np(boxes_3d, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(-1, 9)
    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 9)

    out = boxes.copy()
    xyz = boxes[:, :3]
    out[:, :3] = transform_points(T, xyz)

    if rotate_yaw_flag:
        yaw = boxes[:, 6]
        out[:, 6] = rotate_yaw(T, yaw)

    if rotate_vel_flag:
        vxy = boxes[:, 7:9]
        out[:, 7:9] = rotate_vel_xy(T, vxy)

    return out.astype(np.float64)
