'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/convert_lidar_to_nuscenes_submission.py \
  --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/lidar/nuscenes_val_focalformer.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/pred_lidar_submission_val.json \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --eval_set val \
  --frame global \
  --quat_order wxyz
'''


import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional

def load_records(path: str):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "results" in obj and "meta" in obj:
        return obj  # already submission
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise TypeError(f"Unsupported json top-level type: {type(obj)}")

def reorder_quat_to_wxyz(q, in_order: str):
    """Return quaternion in wxyz order."""
    if q is None:
        return None
    assert len(q) == 4
    in_order = in_order.lower()
    if in_order == "wxyz":
        return q
    if in_order == "xyzw":
        x, y, z, w = q
        return [w, x, y, z]
    raise ValueError("quat_order must be wxyz or xyzw")

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

def infer_frame_from_translation(det_translation) -> str:
    """Very rough heuristic."""
    if det_translation is None or len(det_translation) < 2:
        return "unknown"
    x, y = det_translation[0], det_translation[1]
    if abs(x) > 200 or abs(y) > 200:
        return "global"
    return "lidar"  # could be ego as well; user should specify if needed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="your raw lidar prediction json")
    ap.add_argument("--out_json", required=True, help="output nuscenes submission json")
    ap.add_argument("--dataroot", required=True, help="nuscenes dataroot")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--eval_set", default="val", choices=["train", "val", "test"])
    ap.add_argument("--quat_order", default="wxyz", choices=["wxyz", "xyzw"],
                    help="input quaternion order in your file")
    ap.add_argument("--frame", default="auto", choices=["auto", "global", "ego", "lidar"],
                    help="coordinate frame of translation/rotation/velocity in your file")
    ap.add_argument("--max_dets_per_sample", type=int, default=500)
    ap.add_argument("--no_fill_empty", action="store_true",
                    help="do not fill missing sample_token with [] (eval may assert if missing)")
    # meta flags
    ap.add_argument("--use_camera", type=int, default=0)
    ap.add_argument("--use_lidar", type=int, default=1)
    ap.add_argument("--use_radar", type=int, default=0)
    ap.add_argument("--use_map", type=int, default=0)
    ap.add_argument("--use_external", type=int, default=0)
    args = ap.parse_args()

    raw = load_records(args.in_json)
    if isinstance(raw, dict) and "results" in raw and "meta" in raw:
        with open(args.out_json, "w") as f:
            json.dump(raw, f)
        print("Input already is a nuScenes submission JSON. Saved as-is.")
        return

    # Load nusc only if we need split tokens or coordinate transform
    need_nusc = (not args.no_fill_empty) or (args.frame in ["ego", "lidar"])
    nusc = None
    all_tokens = None
    all_tokens_set = None

    if need_nusc:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
        all_tokens = collect_split_sample_tokens(args.dataroot, args.version, args.eval_set)
        all_tokens_set = set(all_tokens)
    else:
        all_tokens_set = None

    # helpers for transforms
    from pyquaternion import Quaternion

    def get_pose_and_calib(sample_token: str):
        """Return ego_pose (global) and calibrated_sensor (ego<-sensor) for LIDAR_TOP of this sample."""
        sample = nusc.get("sample", sample_token)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        return ego_pose, calib

    results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Convert each record
    for rec in raw:
        dets = rec.get("detections", [])
        if not dets:
            continue
        sample_token = rec.get("sample_token") or dets[0].get("sample_token")
        if sample_token is None:
            continue

        # if evaluating a split, drop samples not in split
        if all_tokens_set is not None and sample_token not in all_tokens_set:
            continue

        # infer frame if auto
        frame = args.frame
        if frame == "auto":
            frame = infer_frame_from_translation(dets[0].get("translation"))
            if frame == "unknown":
                frame = "global"  # default fallback
        # prefetch transforms if needed
        if frame in ["ego", "lidar"]:
            ego_pose, calib = get_pose_and_calib(sample_token)
            q_ego = Quaternion(ego_pose["rotation"])          # wxyz
            t_ego = ego_pose["translation"]                  # global
            q_cal = Quaternion(calib["rotation"])            # wxyz, sensor->ego rotation
            t_cal = calib["translation"]                     # sensor->ego translation
            # For lidar frame: global = ego_pose ⊗ (calib ⊗ lidar)
            # For ego frame:  global = ego_pose ⊗ ego
        else:
            q_ego = t_ego = q_cal = t_cal = None

        for d in dets:
            q_in = reorder_quat_to_wxyz(d.get("rotation"), args.quat_order)
            if q_in is None:
                continue

            translation = d.get("translation")
            size = d.get("size")
            velocity = d.get("velocity", [0.0, 0.0])

            if frame == "global":
                q_out = q_in
                t_out = translation
                v_out = velocity
            elif frame == "ego":
                # t_global = t_ego + q_ego.rotate(t_ego_frame)
                # q_global = q_ego * q_box
                q_box = Quaternion(q_in)
                q_g = q_ego * q_box
                t_rot = q_ego.rotate(translation)
                t_out = [t_ego[0] + t_rot[0], t_ego[1] + t_rot[1], t_ego[2] + t_rot[2]]
                q_out = [q_g.w, q_g.x, q_g.y, q_g.z]
                # velocity rotate to global xy
                v3 = q_ego.rotate([velocity[0], velocity[1], 0.0])
                v_out = [v3[0], v3[1]]
            elif frame == "lidar":
                # sensor(lidar) -> ego: t_ego = t_cal + q_cal.rotate(t_lidar)
                # ego -> global: t_global = t_ego_pose + q_ego.rotate(t_ego)
                q_box = Quaternion(q_in)
                q_g = q_ego * q_cal * q_box
                t_ego_frame = [t_cal[0], t_cal[1], t_cal[2]]
                t_cal_rot = q_cal.rotate(translation)
                t_ego_frame = [t_ego_frame[0] + t_cal_rot[0], t_ego_frame[1] + t_cal_rot[1], t_ego_frame[2] + t_cal_rot[2]]
                t_g_rot = q_ego.rotate(t_ego_frame)
                t_out = [t_ego[0] + t_g_rot[0], t_ego[1] + t_g_rot[1], t_ego[2] + t_g_rot[2]]
                q_out = [q_g.w, q_g.x, q_g.y, q_g.z]
                # velocity: lidar -> ego -> global
                v_ego3 = q_cal.rotate([velocity[0], velocity[1], 0.0])
                v_g3 = q_ego.rotate(v_ego3)
                v_out = [v_g3[0], v_g3[1]]
            else:
                raise ValueError(f"Unknown frame: {frame}")

            one = {
                "sample_token": sample_token,
                "translation": t_out,
                "size": size,
                "rotation": q_out,              # must be wxyz for nuScenes json
                "velocity": v_out,
                "detection_name": d.get("detection_name"),
                "detection_score": float(d.get("detection_score", 0.0)),
                "attribute_name": d.get("attribute_name") or "",
            }
            results[sample_token].append(one)

    # Sort + TopK + (optional) fill empty tokens
    if all_tokens is not None and (not args.no_fill_empty):
        for tok in all_tokens:
            det_list = results.get(tok, [])
            det_list.sort(key=lambda x: x["detection_score"], reverse=True)
            results[tok] = det_list[: args.max_dets_per_sample]
    else:
        # Only for tokens present
        for tok, det_list in list(results.items()):
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
    print(f"Wrote submission: {args.out_json}")
    print(f"Num samples in results: {len(submission['results'])}")
    # small sanity
    nonempty = sum(1 for _,v in submission["results"].items() if len(v)>0)
    print(f"Non-empty samples: {nonempty}")

if __name__ == "__main__":
    main()