# -*- coding: utf-8 -*-
"""
用法：
  # 读取第 0 个 sample
  python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_pkl_print_isfusionres.py 0

  # 指定自定义 pkl 路径与 sample 索引
  python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_pkl_print_isfusionres.py 15 \
      --path /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/nuscenes_infos_train.pkl
"""
#import argparse, os, pickle, sys

import argparse, os, sys, types, io, pickle

def _stub(modname: str):
    if modname not in sys.modules:
         sys.modules[modname] = types.ModuleType(modname)
    return sys.modules[modname]

    # 常见会被引用的命名空间，先全部打壳
    for name in [
        "mmdet3d", "mmdet3d.core", "mmdet3d.core.bbox", "mmdet3d.core.bbox.box_np_ops",
        "mmdet3d.ops", "mmdet3d.ops.iou3d",
        "mmdet", "mmcv", "mmengine",
    ]:
        _stub(name)

    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # 序列化里凡是引用到这些老包的类，一律返回“空壳类”，避免导入失败
            if module.startswith(("mmdet3d", "mmdet", "mmcv", "mmengine")):
                return type(name, (object,), {})
            return super().find_class(module, name)

    with open(path, "rb") as f:
        try:
            return SafeUnpickler(f).load()
        except Exception:
            # 有些 pkl 需要 latin1；或流读指针需要回退
            f.seek(0)
            buf = io.BytesIO(f.read())
            return SafeUnpickler(buf).load()

DEFAULT_PATH = "/home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/isfusion_nusece_train_res.pkl"

def load_pickle(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def to_list(x, max_items=None):
    # 将 tensor/ndarray/list 转成 python list，并可选截断
    try:
        if hasattr(x, "tolist"):
            x = x.tolist()
    except Exception:
        pass
    if isinstance(x, (list, tuple)) and max_items is not None and len(x) > max_items:
        return x[:max_items] + [f"... ({len(x)} total)"]
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int, help="要读取的 sample 索引（从 0 开始）")
    ap.add_argument("--path", type=str, default=DEFAULT_PATH, help="pkl 文件路径")
    ap.add_argument("--max_print", type=int, default=20, help="scores/labels 最多打印多少个元素")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"文件不存在: {args.path}")
        sys.exit(1)

    data = load_pickle(args.path)
    if not isinstance(data, list):
        print(f"期望顶层是 list，但得到 {type(data).__name__}")
        sys.exit(1)

    n = len(data)
    if args.index < 0 or args.index >= n:
        print(f"索引越界: {args.index}（共有 {n} 个 sample，索引范围 0~{n-1}）")
        sys.exit(1)

    sample = data[args.index]

    # 兼容不同键命名：优先 pts_bbox，其次 img_bbox，否则直接用 sample
    det = sample.get("pts_bbox") if isinstance(sample, dict) else None
    if det is None and isinstance(sample, dict):
        det = sample.get("img_bbox")
    if det is None:
        det = sample

    boxes = det.get("boxes_3d") if isinstance(det, dict) else None
    scores = det.get("scores_3d") if isinstance(det, dict) else None

    # labels 可能有不同命名，做个兜底搜索
    label_key = None
    for k in ["labels_3d", "labels", "cls_labels", "pred_labels"]:
        if isinstance(det, dict) and k in det:
            label_key = k
            break
    labels = det.get(label_key) if (isinstance(det, dict) and label_key) else None

    # --- 打印 ---
    print("column order: [x, y, z, l, w, h, yaw, vx, vy]")

    # 取第一条 box
    first_box = None
    if boxes is None:
        print("first box: <boxes_3d not found>")
    else:
        # 兼容 LiDARInstance3DBoxes 或 直接的 ndarray/list
        vec = None
        try:
            vec = boxes.tensor[0]
        except Exception:
            try:
                vec = boxes[0]
            except Exception:
                pass
        if vec is None:
            print("first box: <cannot access first box>")
        else:
            first_box = to_list(vec)
            print("first box:", first_box)

    # scores_3d
    if scores is None:
        print("scores_3d : <not found>")
    else:
        try:
            out_scores = to_list(scores, max_items=args.max_print)
            print("scores_3d :", out_scores)
        except Exception:
            print("scores_3d : <unprintable>")

    # labels_3d
    if labels is None:
        if isinstance(det, dict):
            print(f"labels_3d : <not found> (available keys: {list(det.keys())})")
        else:
            print("labels_3d : <not found>")
    else:
        try:
            out_labels = to_list(labels, max_items=args.max_print)
            print("labels_3d :", out_labels)
        except Exception:
            print("labels_3d : <unprintable>")
    
    # 假设 det = sample["pts_bbox"]
    boxes = det["boxes_3d"]
    scores = det["scores_3d"]
    labels = det["labels_3d"]

# 取 boxes 的数量
    try:
        n_boxes = boxes.tensor.shape[0]   # LiDARInstance3DBoxes
    except Exception:
        n_boxes = len(boxes)              # 若是list/ndarray

    print("N_boxes =", n_boxes)
    print("N_scores =", len(scores))
    print("N_labels =", len(labels))
    assert n_boxes == len(scores) == len(labels), "数量不一致！"


if __name__ == "__main__":
    main()


