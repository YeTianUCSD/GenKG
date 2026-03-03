'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/convert_lidar_global_fast.py \
  --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/lidar/nuscenes_val_focalformer.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/pred_lidar_submission_val.json \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --eval_set val \
  --quat_order wxyz

export MPLBACKEND=Agg
python -m nuscenes.eval.detection.evaluate \
  /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/pred_lidar_submission_val.json \
  --output_dir /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/eval_out_lidar \
  --eval_set val \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --plot_examples 0 \
  --render_curves 0
'''

import os, json, argparse
from collections import defaultdict
from nuscenes.utils.splits import create_splits_scenes

def reorder_quat_to_wxyz(q, in_order: str):
    if q is None: return None
    assert len(q) == 4
    in_order = in_order.lower()
    if in_order == "wxyz": return q
    if in_order == "xyzw":
        x, y, z, w = q
        return [w, x, y, z]
    raise ValueError("quat_order must be wxyz or xyzw")

def collect_split_sample_tokens_fast(dataroot: str, version: str, eval_set: str):
    meta_dir = os.path.join(dataroot, version)
    scene_path = os.path.join(meta_dir, "scene.json")
    sample_path = os.path.join(meta_dir, "sample.json")

    scenes = json.load(open(scene_path, "r"))
    samples = json.load(open(sample_path, "r"))

    next_map = {s["token"]: (s["next"] if s["next"] != "" else None) for s in samples}

    split_scene_names = set(create_splits_scenes()[eval_set])
    tokens = []
    for sc in scenes:
        if sc["name"] not in split_scene_names:
            continue
        tok = sc["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = next_map.get(tok)
    return tokens

def load_raw(path: str):
    obj = json.load(open(path, "r"))
    if isinstance(obj, list): return obj
    if isinstance(obj, dict) and "detections" in obj: return [obj]
    if isinstance(obj, dict) and "results" in obj and "meta" in obj:
        return obj  # already submission
    raise TypeError(f"Unsupported top-level type: {type(obj)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--eval_set", default="val", choices=["train","val","test"])
    ap.add_argument("--quat_order", default="wxyz", choices=["wxyz","xyzw"])
    ap.add_argument("--max_dets_per_sample", type=int, default=500)
    ap.add_argument("--use_camera", type=int, default=0)
    ap.add_argument("--use_lidar", type=int, default=1)
    ap.add_argument("--use_radar", type=int, default=0)
    ap.add_argument("--use_map", type=int, default=0)
    ap.add_argument("--use_external", type=int, default=0)
    args = ap.parse_args()

    raw = load_raw(args.in_json)
    if isinstance(raw, dict) and "results" in raw and "meta" in raw:
        json.dump(raw, open(args.out_json, "w"))
        print("Input already submission; saved as-is.")
        return

    # 1) get all tokens in split (fast, no NuScenes init)
    all_tokens = collect_split_sample_tokens_fast(args.dataroot, args.version, args.eval_set)
    all_set = set(all_tokens)

    # 2) aggregate predictions
    results = defaultdict(list)
    for rec in raw:
        dets = rec.get("detections", [])
        if not dets:
            continue
        sample_token = rec.get("sample_token") or dets[0].get("sample_token")
        if sample_token is None or sample_token not in all_set:
            continue
        for d in dets:
            q = reorder_quat_to_wxyz(d.get("rotation"), args.quat_order)
            if q is None:
                continue
            results[sample_token].append({
                "sample_token": sample_token,
                "translation": d.get("translation"),
                "size": d.get("size"),
                "rotation": q,  # wxyz
                "velocity": d.get("velocity", [0.0, 0.0]),
                "detection_name": d.get("detection_name"),
                "detection_score": float(d.get("detection_score", 0.0)),
                "attribute_name": d.get("attribute_name") or "",
            })

    # 3) fill empty + sort + topK
    for tok in all_tokens:
        det_list = results.get(tok, [])
        det_list.sort(key=lambda x: x["detection_score"], reverse=True)
        results[tok] = det_list[:args.max_dets_per_sample]

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
    json.dump(submission, open(args.out_json, "w"))
    nonempty = sum(1 for _,v in submission["results"].items() if len(v)>0)
    print("Wrote:", args.out_json)
    print("Split samples:", len(all_tokens), "nonempty:", nonempty)

if __name__ == "__main__":
    main()