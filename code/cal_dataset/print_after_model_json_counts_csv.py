'''
# 1) dict(scenes=...) 结构
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/print_after_model_json_counts_csv.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/filter_json/sorted_by_scene_ISFUSIONandGTattr_filter0p5.json \
  > /home/code/3Ddetection/GenKG/counts_filter0p5.csv

# 2) 顶层 list / NDJSON（逐条样本）
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/cal_dataset/print_after_model_json_counts_csv.py \
  --json /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/isfusionres_tracked/ISFUSIONandGTattr_val_40.json \
  > /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_val/isfusionres_tracked/ISFUSIONandGTattr_val_40.csv

'''

# -*- coding: utf-8 -*-
import argparse, json, os, sys, csv

def load_json_auto(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)  # 普通 JSON
        except Exception:
            f.seek(0)            # NDJSON / JSON Lines
            lines = [ln.strip() for ln in f if ln.strip()]
            return [json.loads(ln) for ln in lines]

def iter_samples(obj):
    """
    统一遍历所有样本：
    - 顶层 list：逐条视为一个样本（scene_idx=-1, sample_idx=序号）
    - 顶层 dict 且含 scenes：遍历每个 scene 的 samples（保留 scene/sample 索引）
    """
    if isinstance(obj, list):
        for i, rec in enumerate(obj):
            yield -1, i, rec
        return

    if isinstance(obj, dict) and isinstance(obj.get("scenes"), list):
        for si, sc in enumerate(obj["scenes"]):
            st = sc.get("scene_token")
            for sj, rec in enumerate(sc.get("samples", [])):
                # 注入 scene_token（若样本没带）
                if "scene_token" not in rec and st is not None:
                    rec = dict(rec)
                    rec["scene_token"] = st
                yield si, sj, rec
        return

    raise SystemExit("[ERR ] Unsupported JSON structure. Expect a list OR a dict with 'scenes'.")

def len_if_list(x):
    return len(x) if isinstance(x, list) else 0

def count_fields(rec):
    det = rec.get("det") or {}
    gt  = rec.get("gt")  or {}

    det_boxes  = len_if_list(det.get("boxes_3d"))
    det_scores = len_if_list(det.get("scores_3d"))
    det_labels = len_if_list(det.get("labels_3d"))

    # 注意：按你的要求，只统计 gt 里的 boxes_3d / labels_3d
    gt_boxes   = len_if_list(gt.get("boxes_3d"))
    gt_labels  = len_if_list(gt.get("labels_3d"))

    return det_boxes, det_scores, det_labels, gt_boxes, gt_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="输入 JSON 路径（支持 dict(scenes) / list / NDJSON）")
    ap.add_argument("--no-header", action="store_true", help="不输出表头")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        raise SystemExit(f"[ERR ] File not found: {args.json}")

    obj = load_json_auto(args.json)

    # 统一写 CSV 到 stdout（重定向到文件即可）
    writer = csv.writer(sys.stdout)
    header = ["scene_idx","sample_idx","scene_token","sample_token",
              "det_boxes","det_scores","det_labels","gt_boxes","gt_labels"]
    if not args.no_header:
        writer.writerow(header)

    any_row = False
    for si, sj, rec in iter_samples(obj):
        det_boxes, det_scores, det_labels, gt_boxes, gt_labels = count_fields(rec)
        writer.writerow([
            si, sj,
            rec.get("scene_token",""),
            rec.get("sample_token",""),
            det_boxes, det_scores, det_labels, gt_boxes, gt_labels
        ])
        any_row = True

    if not any_row:
        print("[WARN] No samples found.", file=sys.stderr)

if __name__ == "__main__":
    main()
