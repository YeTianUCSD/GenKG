'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/convert_camera_to_nuscenes_submission.py \
  --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/camera/nuscenes_val_pre.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/camera_submission.json \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --eval_set val \
  --quat_order wxyz
 

python -m nuscenes.eval.detection.evaluate \
  /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/camera_submission_wxyz.json \
  --output_dir /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/eval_out \
  --eval_set val \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --plot_examples 0 \
  --render_curves 0   
'''

import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List

def load_records(path: str):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "results" in obj and "meta" in obj:
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise TypeError(f"Unsupported json top type: {type(obj)}")

def collect_split_sample_tokens(dataroot: str, version: str, eval_set: str):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    split_scenes = set(create_splits_scenes()[eval_set])

    tokens = []
    for scene in nusc.scene:
        if scene["name"] not in split_scenes:
            continue
        tok = scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            sample = nusc.get("sample", tok)
            tok = sample["next"] if sample["next"] != "" else None
    return tokens

def reorder_quat(q, quat_order: str):
    if q is None:
        return q
    assert len(q) == 4
    if quat_order.lower() == "wxyz":
        return q
    if quat_order.lower() == "xyzw":
        x, y, z, w = q
        return [w, x, y, z]
    raise ValueError("quat_order must be wxyz or xyzw")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--eval_set", default="val", choices=["train", "val", "test"])
    ap.add_argument("--quat_order", default="xyzw", choices=["xyzw", "wxyz"])
    ap.add_argument("--max_dets_per_sample", type=int, default=500)
    ap.add_argument("--use_camera", type=int, default=1)
    ap.add_argument("--use_lidar", type=int, default=0)
    ap.add_argument("--use_radar", type=int, default=0)
    ap.add_argument("--use_map", type=int, default=0)
    ap.add_argument("--use_external", type=int, default=0)
    args = ap.parse_args()

    raw = load_records(args.in_json)
    if isinstance(raw, dict) and "results" in raw and "meta" in raw:
        with open(args.out_json, "w") as f:
            json.dump(raw, f)
        print("Input already is nuScenes submission. Saved as-is.")
        return

    all_tokens = collect_split_sample_tokens(args.dataroot, args.version, args.eval_set)
    all_tokens_set = set(all_tokens)

    results: Dict[str, List[Dict]] = defaultdict(list)

    for rec in raw:
        dets = rec.get("detections", [])
        sample_token = rec.get("sample_token") or (dets[0].get("sample_token") if dets else None)
        if sample_token is None or sample_token not in all_tokens_set:
            continue

        for d in dets:
            one = {
                "sample_token": sample_token,
                "translation": d.get("translation"),
                "size": d.get("size"),
                "rotation": reorder_quat(d.get("rotation"), args.quat_order),
                "velocity": d.get("velocity", [0.0, 0.0]),
                "detection_name": d.get("detection_name"),
                "detection_score": float(d.get("detection_score", 0.0)),
                "attribute_name": d.get("attribute_name") or "",
            }
            results[sample_token].append(one)

    # 补齐所有 sample_token，并做 topK=500
    for tok in all_tokens:
        det_list = results.get(tok, [])
        det_list.sort(key=lambda x: x["detection_score"], reverse=True)
        results[tok] = det_list[: args.max_dets_per_sample]

    submission = {
        "meta": {
            "use_camera": bool(args.use_camera),
            "use_lidar": bool(args.use_lidar),
            "use_radar": bool(args.use_radar),
            "use_map": bool(args.use_map),
            "use_external": bool(args.use_external),
        },
        "results": dict(results),
    }

    with open(args.out_json, "w") as f:
        json.dump(submission, f)
    print(f"Wrote: {args.out_json}")
    print(f"Split samples: {len(all_tokens)}")
    print(f"Samples with preds: {sum(1 for _,v in results.items() if len(v)>0)}")

if __name__ == "__main__":
    main()