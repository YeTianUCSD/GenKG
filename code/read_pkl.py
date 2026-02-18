# -*- coding: utf-8 -*-
import pickle
import os

# 你的文件路径
PKL_PATH = "/home/code/3Ddetection/IS-Fusion/GenKG/data/res_from_ISFUSION.pkl"

def load_pickle(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)               # 优先用默认方式
        except Exception:
            f.seek(0)
            # 兼容 Python2 存的 pickle
            return pickle.load(f, encoding="latin1")

def preview(obj, n=3):
    import numpy as np

    print("=== PKL 顶层对象类型:", type(obj).__name__, "===\n")

    # DataFrame 优先：打印 head(3)
    if hasattr(obj, "head") and callable(getattr(obj, "head")):
        try:
            print(obj.head(n))
            return
        except Exception:
            pass

    # dict：打印前三个键值
    if isinstance(obj, dict):
        print(f"(dict) 共 {len(obj)} 个键，展示前 {n} 个：\n")
        for i, (k, v) in enumerate(list(obj.items())[:n], 1):
            print(f"[{i}] key={repr(k)[:120]}")
            print(f"    value_type={type(v).__name__}")
            if isinstance(v, (list, tuple)):
                print(f"    value_preview(list/tuple 前几项): {v[:min(3, len(v))]}")
            elif isinstance(v, np.ndarray):
                flat = v.ravel()
                print(f"    ndarray shape={v.shape}, dtype={v.dtype}, preview={flat[:min(6, flat.size)]}")
            else:
                print(f"    value_preview: {repr(v)[:300]}")
            print()
        return

    # list / tuple：打印前三项
    if isinstance(obj, (list, tuple)):
        print(f"({type(obj).__name__}) 共 {len(obj)} 项，展示前 {n} 项：\n")
        for i, item in enumerate(obj[:n], 1):
            print(f"[{i}] type={type(item).__name__}")
            print(repr(item)[:1000])  # 避免过长
            print()
        return

    # numpy 数组
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            print(f"(ndarray) shape={obj.shape}, dtype={obj.dtype}\n")
            # 尽量打印前 n 行（或前 n 个元素）
            if obj.ndim == 1:
                print(obj[:n])
            else:
                print(obj[:n])
            return
    except Exception:
        pass

    # 其他类型：直接打印 repr
    print("(other) 直接预览：\n")
    print(repr(obj)[:1000])

def main():
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"文件不存在：{PKL_PATH}")

    obj = load_pickle(PKL_PATH)
    preview(obj, n=3)

if __name__ == "__main__":
    main()
