#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：读取 IS-Fusion 结果 pkl，打印里面有多少条 sample 数据（len(data)）。

用法示例：
  # 使用默认路径
  python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_pkl_isfusionres_number.py

  # 指定自定义 pkl 路径
  python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_pkl_isfusionres_number.py \
      --path /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/nuscenes_train_results.pkl
"""

import argparse
import os
import sys
import types
import io
import pickle

# 默认 pkl 路径（按你现在的设置）
DEFAULT_PATH = "/home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/isfusion_nusece_train_res.pkl"


# --- 为了解决 mmdet3d 等模块缺失导致的 unpickle 失败，做一些“空壳 stub” ---
def _install_stubs():
    """给常见的 mmdet3d/mmdet/mmcv/mmengine 命名空间打空壳，防止 import 报错。"""
    for name in [
        "mmdet3d",
        "mmdet3d.core",
        "mmdet3d.core.bbox",
        "mmdet3d.core.bbox.box_np_ops",
        "mmdet3d.ops",
        "mmdet3d.ops.iou3d",
        "mmdet",
        "mmcv",
        "mmengine",
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


class SafeUnpickler(pickle.Unpickler):
    """拦截老项目里的类引用，统一换成空壳类，防止没有这些包时报错。"""

    def find_class(self, module, name):
        if module.startswith(("mmdet3d", "mmdet", "mmcv", "mmengine")):
            return type(name, (object,), {})
        return super().find_class(module, name)


def load_pickle(path):
    """尽量安全地 load pkl（支持 stub + latin1 回退）。"""
    _install_stubs()
    with open(path, "rb") as f:
        try:
            return SafeUnpickler(f).load()
        except Exception:
            # 有些是 python2 存的，需要 latin1
            f.seek(0)
            buf = io.BytesIO(f.read())
            return SafeUnpickler(buf).load()


def main():
    ap = argparse.ArgumentParser(description="统计 IS-Fusion 结果 pkl 里有多少条 sample。")
    ap.add_argument(
        "--path",
        type=str,
        default=DEFAULT_PATH,
        help="pkl 文件路径（默认：%(default)s）",
    )
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"文件不存在: {args.path}")
        sys.exit(1)

    data = load_pickle(args.path)

    # 最常见情况：顶层是 list，每个元素就是一个 sample
    if isinstance(data, list):
        n = len(data)
        print(f"该 pkl 顶层是 list，包含 {n} 条 sample 数据。")
        return

    # 如果不是 list，给点提示信息，方便你排查
    print(f"该 pkl 顶层类型是: {type(data).__name__}")
    try:
        # 有些可能是 dict，里边某个 key 才是真正的 list
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    print(f"  其中键 '{k}' 对应的 list 长度为 {len(v)}")
        else:
            print("不是 list，无法直接统计“条数”（len(data)）作为 sample 数。")
    except Exception as e:
        print("尝试分析结构时出错：", repr(e))


if __name__ == "__main__":
    main()
