# ebm_refine/io.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
I/O utilities for scene-style JSON that contains per-frame `det` (+ optional `gt`) fields.

Goals:
- Robustly find "sample nodes" in arbitrary nested JSON.
- Load root + flattened sample list with normalized timestamps.
- Group samples by scene and sort by time.
- Write refined detections back to the original JSON structure, without breaking it.

Key design:
- We identify a sample by a set of candidate keys (sample_token, sample_data_token, timestamp, etc.).
- A replacement map `repl_map` can use any of those keys.
- Writeback supports two modes:
    1) replace: overwrite `sample["det"]` fields
    2) add: write to `sample["det_refined"]` (recommended to keep original det)

Payload schema (flexible):
- Required if present in payload: "boxes_3d", "labels_3d", "scores_3d"
- Optional extras: "attrs", "attr_scores", "track_ids", "exist_prob", ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


# -----------------------------
# Timestamp helpers
# -----------------------------

def get_timestamp(sample: Dict[str, Any]) -> int:
    """Return an integer timestamp. Prefers `timestamp`, else `timestamp_us`, else 0."""
    if "timestamp" in sample:
        try:
            return int(sample["timestamp"])
        except Exception:
            pass
    if "timestamp_us" in sample:
        try:
            return int(sample["timestamp_us"])
        except Exception:
            pass
    return 0


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str) and len(x) > 0:
        return x
    try:
        s = str(x)
        return s if len(s) > 0 else None
    except Exception:
        return None


def sample_key_candidates(sample: Dict[str, Any]) -> List[str]:
    """
    Generate a ranked list of possible keys for this sample.
    These keys are used to look up entries in `repl_map`.
    """
    scene = _safe_str(sample.get("scene_token"))
    st = _safe_str(sample.get("sample_token"))
    sdt = _safe_str(sample.get("sample_data_token"))
    ts = get_timestamp(sample)
    ts_s = _safe_str(ts)

    keys: List[str] = []
    # Highest priority: stable tokens
    if st:
        keys.append(st)
    if sdt:
        keys.append(sdt)

    # Fallbacks: time-based keys
    if ts_s:
        keys.append(ts_s)
        if scene:
            keys.append(f"{scene}:{ts_s}")
        if sdt:
            keys.append(f"{sdt}:{ts_s}")
        if st:
            keys.append(f"{st}:{ts_s}")

    return keys


# -----------------------------
# Sample discovery in nested JSON
# -----------------------------

def iter_samples(
    obj: Any,
    require_det: bool = True,
    require_gt: bool = False,
) -> Iterable[Dict[str, Any]]:
    """
    Recursively traverse an arbitrary nested JSON-like object (dict/list)
    and yield dict nodes that look like per-frame samples.

    Default heuristic:
      - has a `det` dict
      - and `det["boxes_3d"]` is a list
      - optionally require `gt` dict when require_gt=True
    """
    if isinstance(obj, dict):
        det = obj.get("det", None)
        gt = obj.get("gt", None)

        ok = True
        if require_det:
            ok = ok and isinstance(det, dict) and isinstance(det.get("boxes_3d"), list)
        if require_gt:
            ok = ok and isinstance(gt, dict)

        if ok:
            yield obj

        for v in obj.values():
            yield from iter_samples(v, require_det=require_det, require_gt=require_gt)

    elif isinstance(obj, list):
        for it in obj:
            yield from iter_samples(it, require_det=require_det, require_gt=require_gt)


# -----------------------------
# Loading / grouping
# -----------------------------

@dataclass
class LoadResult:
    root: Any
    samples: List[Dict[str, Any]]


def load_root_and_samples(
    json_path: str,
    require_gt: bool = True,
) -> LoadResult:
    """
    Load JSON and collect sample nodes.

    - Adds a normalized integer timestamp field `_ts` into each sample dict.
    - Filters out nodes missing `scene_token` or timestamp.
    """
    with open(json_path, "r") as f:
        root = json.load(f)

    samples = list(iter_samples(root, require_det=True, require_gt=require_gt))

    # normalize / filter
    out: List[Dict[str, Any]] = []
    for s in samples:
        if "scene_token" not in s:
            continue
        ts = get_timestamp(s)
        if ts == 0:
            continue
        s["_ts"] = int(ts)
        out.append(s)

    return LoadResult(root=root, samples=out)


def group_by_scene(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group samples by scene_token and sort each scene list by `_ts`."""
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        sc = s.get("scene_token", None)
        if sc is None:
            continue
        by_scene.setdefault(sc, []).append(s)
    for sc, lst in by_scene.items():
        lst.sort(key=lambda x: int(x.get("_ts", get_timestamp(x))))
    return by_scene


# -----------------------------
# Replacement map utilities
# -----------------------------

def build_repl_map_from_flat(
    flat_list: List[Dict[str, Any]],
    prefer_key: str = "sample_token",
) -> Dict[str, Dict[str, Any]]:
    """
    Convert a flat list of per-frame outputs into a repl_map.

    Expected minimal structure per item:
      {
        "scene_token": "...",
        "sample_token": "...",            (optional)
        "sample_data_token": "...",       (optional)
        "timestamp": 123,                (optional)
        "det": { "boxes_3d": ..., "labels_3d": ..., "scores_3d": ..., ... }
      }

    Key choice:
      - if prefer_key exists and non-empty, use it
      - else fallback: sample_token -> sample_data_token -> timestamp -> scene:timestamp
    """
    repl: Dict[str, Dict[str, Any]] = {}
    for it in flat_list:
        det = it.get("det", {}) or {}
        if not isinstance(det, dict):
            continue
        st = _safe_str(it.get("sample_token"))
        sdt = _safe_str(it.get("sample_data_token"))
        ts = it.get("timestamp", None)
        ts_s = _safe_str(ts) if ts is not None else None
        scene = _safe_str(it.get("scene_token"))

        chosen: Optional[str] = None
        if prefer_key == "sample_token" and st:
            chosen = st
        elif prefer_key == "sample_data_token" and sdt:
            chosen = sdt
        elif prefer_key == "timestamp" and ts_s:
            chosen = ts_s

        if chosen is None:
            chosen = st or sdt or ts_s or (f"{scene}:{ts_s}" if (scene and ts_s) else None)

        if chosen is None:
            continue

        repl[chosen] = det
    return repl


def merge_repl_maps(repl_maps: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Merge multiple repl maps (e.g., DDP shard outputs). Later maps override earlier ones.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for m in repl_maps:
        out.update(m)
    return out


# -----------------------------
# Writing back into the full JSON structure
# -----------------------------

@dataclass
class WritebackStats:
    matched: int
    updated: int
    visited: int


def apply_det_replacements(
    root_obj: Any,
    repl_map: Dict[str, Dict[str, Any]],
    *,
    mode: str = "add",                 # "replace" or "add"
    det_key: str = "det",
    refined_key: str = "det_refined",
    fields: Optional[List[str]] = None,
) -> WritebackStats:
    """
    Recursively traverse root_obj and apply det replacements.

    Args:
      repl_map: maps one of sample keys -> payload dict (e.g., {"boxes_3d":..., ...})
      mode:
        - "replace": write into sample[det_key] fields
        - "add":     write into sample[refined_key] (recommended)
      fields:
        If provided, only copy these fields from payload.
        If None, copy all fields in payload dict.

    Returns:
      WritebackStats counts.
    """
    if mode not in ("replace", "add"):
        raise ValueError(f"mode must be 'replace' or 'add', got: {mode}")

    stats = WritebackStats(matched=0, updated=0, visited=0)

    def _apply(node: Any):
        if isinstance(node, dict):
            stats.visited += 1

            # Only attempt match if it's a sample-like dict with det present
            det = node.get(det_key, None)
            if isinstance(det, dict):
                keys = sample_key_candidates(node)
                chosen: Optional[str] = None
                for k in keys:
                    if k in repl_map:
                        chosen = k
                        break

                if chosen is not None:
                    stats.matched += 1
                    payload = repl_map[chosen]
                    if isinstance(payload, dict):
                        target = det if mode == "replace" else (node.get(refined_key) or {})
                        if not isinstance(target, dict):
                            target = {}

                        if fields is None:
                            for kk, vv in payload.items():
                                target[kk] = vv
                        else:
                            for kk in fields:
                                if kk in payload:
                                    target[kk] = payload[kk]

                        if mode == "replace":
                            node[det_key] = target
                        else:
                            node[refined_key] = target

                        stats.updated += 1

            for v in node.values():
                _apply(v)

        elif isinstance(node, list):
            for it in node:
                _apply(it)

    _apply(root_obj)
    return stats


def dump_json(obj: Any, out_path: str, indent: int = 2) -> None:
    """Write JSON with stable formatting."""
    with open(out_path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_refined_json(
    in_json_path: str,
    out_json_path: str,
    repl_map: Dict[str, Dict[str, Any]],
    *,
    mode: str = "add",
    det_key: str = "det",
    refined_key: str = "det_refined",
    fields: Optional[List[str]] = None,
    indent: int = 2,
) -> WritebackStats:
    """
    Convenience helper: load -> apply replacements -> dump.
    """
    root = load_json(in_json_path)
    stats = apply_det_replacements(
        root,
        repl_map,
        mode=mode,
        det_key=det_key,
        refined_key=refined_key,
        fields=fields,
    )
    dump_json(root, out_json_path, indent=indent)
    return stats
