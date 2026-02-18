'''
# 你的“排序后的 JSON”顶层是 dict（有 scenes 字段）
# 1) 打印第 0 个场景里的第 0 个样本
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_isfusion_GT_Sort_json.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/filter_json/sorted_by_scene_ISFUSIONandGTattr_filter0p5.json \
  --scene-index 5 --sample-index 3 --full \
  > /home/code/3Ddetection/IS-Fusion/GenKG/data/read_sorted_by_scene_ISFUSIONandGTattr_filter0p5.log

python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_isfusion_GT_Sort_json.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/filter_json/sorted_by_scene_ISFUSIONandGTattr_filter0p1.json \
  --scene-index 15 --sample-index 3 --pretty \
  > /home/code/3Ddetection/IS-Fusion/GenKG/data/read_json_is_GT_sorted_json.log


# 2) 扁平化后按全局顺序取第 n 条
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_isfusionres_json.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/sorted_by_scene_ISFUSIONandGT.json \
  --flatten --index 0

# 3) 直接用 sample_token 定位
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_isfusionres_json.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/sorted_by_scene_ISFUSIONandGT.json \
  --by-sample-token b10f0cd792b64d16a1a5e8349b20504c --pretty

'''

# -*- coding: utf-8 -*-
import argparse, json, os, sys

def load_json_auto(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            f.seek(0)
            lines = [ln.strip() for ln in f if ln.strip()]
            return [json.loads(ln) for ln in lines]

def norm_ts_us(rec):
    # 规范化时间戳，优先 timestamp_us / timestamp（μs），次选 ego_pose_timestamp（s）
    if "timestamp_us" in rec:
        return int(rec["timestamp_us"])
    if "timestamp" in rec:
        ts = rec["timestamp"]
        return int(ts) if isinstance(ts, int) and ts > 1_000_000_000 else int(float(ts) * 1e6)
    if "ego_pose_timestamp" in rec:
        return int(float(rec["ego_pose_timestamp"]) * 1e6)
    return 0

def flatten_scenes(scenes):
    flat = []
    for sc in scenes:
        st = sc.get("scene_token")
        for i, r in enumerate(sc.get("samples", [])):
            r2 = dict(r)
            r2.setdefault("scene_token", st)
            r2["_scene_idx"] = i
            r2["_ts_us"] = norm_ts_us(r2)
            flat.append(r2)
    flat.sort(key=lambda x: (x.get("scene_token",""), x["_scene_idx"]))
    return flat

def count_det(rec):
    det = rec.get("det") or {}
    n = None
    if "scores_3d" in det and isinstance(det["scores_3d"], list):
        n = len(det["scores_3d"])
    elif "labels_3d" in det and isinstance(det["labels_3d"], list):
        n = len(det["labels_3d"])
    elif "boxes_3d" in det and isinstance(det["boxes_3d"], list):
        n = len(det["boxes_3d"])
    return n

def count_gt(rec):
    gt = rec.get("gt") or {}
    if "names" in gt and isinstance(gt["names"], list):
        return len(gt["names"])
    if "labels_3d" in gt and isinstance(gt["labels_3d"], list):
        return len(gt["labels_3d"])
    if "boxes_3d" in gt and isinstance(gt["boxes_3d"], list):
        return len(gt["boxes_3d"])
    return None

def print_sample(rec, pretty=False, max_boxes=1):
    basic = {
        "scene_token": rec.get("scene_token"),
        "sample_token": rec.get("sample_token"),
        "sample_data_token": rec.get("sample_data_token"),
        "timestamp_us": norm_ts_us(rec),
    }
    print(json.dumps(basic, ensure_ascii=False, indent=2))

    # 统计 det / gt 数量
    nd = count_det(rec)
    ng = count_gt(rec)
    print(f"det_count: {nd}    gt_count: {ng}")

    # 打印 det 的第一框/前若干分数 & 标签（如果有）
    det = rec.get("det") or {}
    boxes = det.get("boxes_3d")
    scores = det.get("scores_3d")
    labels = det.get("labels_3d")

    print("\n[det preview]")
    if boxes and isinstance(boxes, list) and len(boxes) > 0:
        print("column order: [x, y, z, l, w, h, yaw, vx, vy]")
        print("first box:", boxes[0])
    else:
        print("first box: <none>")

    if scores and isinstance(scores, list):
        preview = scores[:20] + (["... (%d total)" % len(scores)] if len(scores) > 20 else [])
        print("scores_3d:", preview)
    else:
        print("scores_3d: <none>")

    if labels and isinstance(labels, list):
        preview = labels[:20] + (["... (%d total)" % len(labels)] if len(labels) > 20 else [])
        print("labels_3d:", preview)
    else:
        print("labels_3d: <none>")

    if pretty:
        print("\n[raw record]")
        print(json.dumps(rec, ensure_ascii=False, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="输入 JSON 路径（排序后的文件）")
    ap.add_argument("--index", type=int, help="当顶层为 list 或 --flatten 时，选择第 n 条（0 起始）")
    ap.add_argument("--scene-index", type=int, default=0, help="当顶层含 scenes 时，选择第几个 scene（0 起始）")
    ap.add_argument("--sample-index", type=int, default=0, help="当顶层含 scenes 时，选择该 scene 的第几个样本（0 起始）")
    ap.add_argument("--flatten", action="store_true", help="把所有 scene 样本拍平成一条大列表，再用 --index 选择")
    ap.add_argument("--by-sample-token", type=str, default=None, help="直接按 sample_token 精确查找")
    ap.add_argument("--pretty", action="store_true", help="额外打印完整 JSON 记录（在摘要与预览之后）")
    ap.add_argument("--full", action="store_true", help="只打印完整 JSON 记录（不输出摘要/预览）")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        print(f"[ERR ] File not found: {args.json}")
        sys.exit(1)

    obj = load_json_auto(args.json)

    # 情况 1：顶层是 list
    if isinstance(obj, list):
        total = len(obj)
        if args.by_sample_token:
            rec = next((r for r in obj if r.get("sample_token") == args.by_sample_token), None)
            if not rec:
                print(f"[ERR ] sample_token not found: {args.by_sample_token}")
                sys.exit(1)
            if args.full:
                print(json.dumps(rec, ensure_ascii=False, indent=2))
            else:
                print_sample(rec, pretty=args.pretty)
            return
        if args.index is None:
            print(f"[ERR ] The JSON root is a list. Please provide --index (0~{total-1}).")
            sys.exit(1)
        if not (0 <= args.index < total):
            print(f"[ERR ] index out of range: {args.index} / {total}")
            sys.exit(1)
        rec = obj[args.index]
        if args.full:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
        else:
            print_sample(rec, pretty=args.pretty)
        return

    # 情况 2：顶层是 dict（带 scenes）
    if isinstance(obj, dict) and "scenes" in obj and isinstance(obj["scenes"], list):
        scenes = obj["scenes"]
        total_scenes = len(scenes)
        total_samples = sum(len(s.get("samples", [])) for s in scenes)
        print(f"scenes: {total_scenes}   total_samples: {total_samples}")

        # 按 token 精确查
        if args.by_sample_token:
            for sc in scenes:
                for r in sc.get("samples", []):
                    if r.get("sample_token") == args.by_sample_token:
                        if args.full:
                            print(json.dumps(r, ensure_ascii=False, indent=2))
                        else:
                            print_sample(r, pretty=args.pretty)
                        return
            print(f"[ERR ] sample_token not found: {args.by_sample_token}")
            sys.exit(1)

        # 扁平模式
        if args.flatten:
            flat = flatten_scenes(scenes)
            if args.index is None:
                print(f"[ERR ] Use --index with --flatten (0~{len(flat)-1}).")
                sys.exit(1)
            if not (0 <= args.index < len(flat)):
                print(f"[ERR ] index out of range: {args.index} / {len(flat)}")
            rec = flat[args.index]
            if args.full:
                print(json.dumps(rec, ensure_ascii=False, indent=2))
            else:
                print_sample(rec, pretty=args.pretty)
            return

        # 按 scene + sample 索引
        if not (0 <= args.scene_index < total_scenes):
            print(f"[ERR ] scene-index out of range: {args.scene_index} / {total_scenes}")
            sys.exit(1)
        scene = scenes[args.scene_index]
        samples = scene.get("samples", [])
        if not samples:
            print("[ERR ] selected scene has no samples.")
            sys.exit(1)
        if not (0 <= args.sample_index < len(samples)):
            print(f"[ERR ] sample-index out of range: {args.sample_index} / {len(samples)}")
            sys.exit(1)
        rec = samples[args.sample_index]
        if args.full:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
        else:
            print_sample(rec, pretty=args.pretty)
        return

    print("[ERR ] Unsupported JSON structure. Expect a list OR a dict with 'scenes'.")

if __name__ == "__main__":
    main()

