#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dump one entry from nuscenes_infos_*.pkl as full JSON to stdout.

Usage:
  python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_pkl_print_nusence.py \
    --path /home/dataset/nuscene/nuscenes_infos_val.pkl \
    --index 0 \
    > /home/code/3Ddetection/IS-Fusion/GenKG/data/read_json.log 2>&1
"""

import argparse, os, sys, pickle, json

def load_pickle(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def find_info_list(container):
    if isinstance(container, list):
        return container
    if isinstance(container, dict):
        # 常见 key：infos / data_list / samples
        for k in ("infos", "data_list", "samples"):
            v = container.get(k, None)
            if isinstance(v, list):
                return v
        # 兜底：找第一个 list
        for v in container.values():
            if isinstance(v, list):
                return v
    return None

def safe_to_builtin(obj, _depth=0, _max_depth=100):
    """递归把对象转成可 JSON 序列化的内置类型。"""
    # 防止异常深递归
    if _depth > _max_depth:
        return f"<<max_depth_exceeded:{type(obj).__name__}>>"

    # None / 基础标量
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy 标量
    try:
        import numpy as np
        if isinstance(obj, (np.generic,)):
            return obj.item()
    except Exception:
        np = None

    # numpy 数组 / Tensor / 具有 .tolist()
    for attr in ("tolist",):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass

    # mmdet3d Boxes（带 .tensor）
    if hasattr(obj, "tensor"):
        try:
            return obj.tensor.tolist()
        except Exception:
            return f"<<unserializable:{type(obj).__name__}>>"

    # 字典
    if isinstance(obj, dict):
        return {str(k): safe_to_builtin(v, _depth+1, _max_depth) for k, v in obj.items()}

    # 列表/元组
    if isinstance(obj, (list, tuple)):
        return [safe_to_builtin(v, _depth+1, _max_depth) for v in obj]

    # 其它不可序列化对象，用字符串表示
    return f"<<unserializable:{type(obj).__name__}>>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="nuscenes_infos_*.pkl 路径")
    ap.add_argument("--index", type=int, required=True, help="info 索引（0 开始）")
    ap.add_argument("--one-based", action="store_true", help="将 --index 视为 1-based")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"[ERR] 文件不存在: {args.path}", file=sys.stderr)
        sys.exit(1)

    container = load_pickle(args.path)
    infos = find_info_list(container)
    if infos is None:
        print("[ERR] 未找到 info 列表（如 'infos' 或 'data_list'）。", file=sys.stderr)
        sys.exit(1)

    idx = args.index - 1 if args.one_based else args.index
    if not (0 <= idx < len(infos)):
        print(f"[ERR] 索引越界：{idx} / {len(infos)}", file=sys.stderr)
        sys.exit(1)

    info = infos[idx]
    data = safe_to_builtin(info)

    # 打印为 JSON 到 stdout（其余信息走 stderr，便于重定向一起写入 2>&1）
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[OK] dumped index {idx} of {len(infos)} from {args.path}", file=sys.stderr)

if __name__ == "__main__":
    main()
