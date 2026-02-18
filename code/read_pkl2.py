# -*- coding: utf-8 -*-
import pickle, os

PKL = "/home/code/3Ddetection/IS-Fusion/GenKG/isfusion_output/tianyetest.pkl"

# 若你有自己的类别顺序，按需改这行；否则留空让脚本只打印 label id
NU_CLASSES = [
    "car","truck","bus","trailer","construction_vehicle",
    "pedestrian","motorcycle","bicycle","traffic_cone","barrier"
]

def load_pkl(p):
    with open(p, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

res = load_pkl(PKL)
print(f"Top-level type={type(res).__name__}, len={len(res)}")

i = 0  # 看第 i 个 sample
sample = res[i]
print("\nKeys in one sample:", list(sample.keys()))

pred = sample.get("pts_bbox", sample)
boxes = pred.get("boxes_3d", None)
scores = pred.get("scores_3d", None)
labels = pred.get("labels_3d", None)

print("\n--- pts_bbox summary ---")
if boxes is not None:
    print("boxes_3d type:", type(boxes).__name__)
    print("tensor shape:", getattr(boxes, "tensor").shape)
    print("column order: [x, y, z, l, w, h, yaw, vx, vy]")
    print("first box:", boxes.tensor[0].tolist())
if scores is not None:
    print("scores_3d shape:", scores.shape, "first5:", scores[:5].tolist())
if labels is not None:
    print("labels_3d shape:", labels.shape, "first5 id:", labels[:5].tolist())
    if NU_CLASSES and int(labels[0]) < len(NU_CLASSES):
        names = [NU_CLASSES[int(x)] if int(x) < len(NU_CLASSES) else f"id{int(x)}" for x in labels[:5].tolist()]
        print("first5 name:", names)

# 看一眼是否还含有 img_bbox
if "img_bbox" in sample:
    print("\n(img_bbox present) keys:", list(sample["img_bbox"].keys()))
