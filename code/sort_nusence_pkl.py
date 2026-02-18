import pickle

#pkl_path = "/home/dataset/nuscene/nuscenes_infos_val.pkl"
#out_path = "/home/code/3Ddetection/IS-Fusion/GenKG/data/nuscenes_infos_val_sorted.pkl"

pkl_path = "/home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/nuscenes_infos_train.pkl"
out_path = "/home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/nuscenes_infos_train_sorted.pkl"


# 读取原始 pkl
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# 提取 infos
if isinstance(data, dict):
    if "infos" in data and isinstance(data["infos"], (list, tuple)):
        infos = data["infos"]
    else:
        raise ValueError("找不到 infos")
else:
    raise ValueError("pkl 格式不是 dict")

# 按 timestamp 排序
infos = sorted(infos, key=lambda e: e["timestamp"])

# 更新 data 并保存到新文件
data["infos"] = infos
with open(out_path, "wb") as f:
    pickle.dump(data, f)

print(f"已保存排序后的文件到: {out_path}")
