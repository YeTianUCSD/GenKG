'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/convert_isfusion_to_nuscenes_submission_fast.py \
  --in_json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json \
  --out_json /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/pred_isfusion_submission_val.json \
  --attribute_mode empty

  export MPLBACKEND=Agg
python -m nuscenes.eval.detection.evaluate \
  /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/pred_isfusion_submission_val.json \
  --output_dir /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/eval_out_isfusion \
  --eval_set val \
  --dataroot /home/dataset/nuscene \
  --version v1.0-trainval \
  --plot_examples 0 \
  --render_curves 0 
  
'''

import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional
from pyquaternion import Quaternion

# 默认 label -> class 名映射（mmdet3d / bevformer 常见）
DEFAULT_LABEL_MAP = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

def yaw_to_quat_wxyz(yaw: float) -> List[float]:
    # yaw about +Z axis in lidar frame
    q = Quaternion(axis=[0, 0, 1], radians=yaw)
    return [q.w, q.x, q.y, q.z]

def quat_from_wxyz(q: List[float]) -> Quaternion:
    # input is [w,x,y,z]
    return Quaternion(q)

def rotate_and_translate(q: Quaternion, t: List[float], p: List[float]) -> List[float]:
    pr = q.rotate(p)
    return [pr[0] + t[0], pr[1] + t[1], pr[2] + t[2]]

def rotate_vec_xy(q: Quaternion, vxy: List[float]) -> List[float]:
    v3 = q.rotate([vxy[0], vxy[1], 0.0])
    return [v3[0], v3[1]]

def load_label_map(path: Optional[str]) -> List[str]:
    if path is None:
        return DEFAULT_LABEL_MAP
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # allow {"0":"car", "1":"truck", ...}
        m = []
        for k in sorted(obj.keys(), key=lambda x: int(x)):
            m.append(obj[k])
        return m
    raise TypeError("label_map must be a json list or dict")

def get_class_name(det: Dict[str, Any], idx: int, label_map: List[str]) -> str:
    # if already has name
    if "detection_name" in det:
        return det["detection_name"]
    labels = det.get("labels_3d", None)
    if labels is None:
        # fallback
        return "car"
    lab = int(labels[idx])
    if 0 <= lab < len(label_map):
        return label_map[lab]
    return "car"

def get_score(det: Dict[str, Any], idx: int) -> float:
    if "detection_score" in det:
        return float(det["detection_score"])
    scores = det.get("scores_3d", None)
    if scores is None:
        return 1.0
    return float(scores[idx])

def heuristic_attribute(cls: str, vxy: List[float]) -> str:
    # very simple, enough for not breaking eval; AAE may not match leaderboard exactly
    speed = (vxy[0] ** 2 + vxy[1] ** 2) ** 0.5
    if cls in ["car", "truck", "bus", "trailer", "construction_vehicle"]:
        return "vehicle.parked" if speed < 0.2 else "vehicle.moving"
    if cls == "pedestrian":
        return "pedestrian.standing" if speed < 0.2 else "pedestrian.walking"
    if cls in ["bicycle", "motorcycle"]:
        return "cycle.without_rider" if speed < 0.2 else "cycle.with_rider"
    # barrier / cone etc.
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="IS-Fusion json (contains scenes->samples->det)")
    ap.add_argument("--out_json", required=True, help="nuScenes submission json")
    ap.add_argument("--label_map", default=None, help="optional json file: list of class names by label id")
    ap.add_argument("--max_dets_per_sample", type=int, default=500)
    ap.add_argument("--attribute_mode", default="heuristic", choices=["heuristic", "empty"])
    # meta
    ap.add_argument("--use_camera", type=int, default=0)
    ap.add_argument("--use_lidar", type=int, default=1)
    ap.add_argument("--use_radar", type=int, default=0)
    ap.add_argument("--use_map", type=int, default=0)
    ap.add_argument("--use_external", type=int, default=0)
    args = ap.parse_args()

    label_map = load_label_map(args.label_map)

    with open(args.in_json, "r") as f:
        root = json.load(f)

    scenes = root.get("scenes", [])
    results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # iterate all samples in json (this json looks like full val scenes already)
    for sc in scenes:
        for sample in sc.get("samples", []):
            sample_token = sample.get("sample_token")
            if not sample_token:
                continue

            ego2global = sample.get("ego2global") or sample.get("ego_pose")  # tolerate naming
            lidar2ego = sample.get("lidar2ego")

            if ego2global is None or lidar2ego is None:
                # cannot transform
                continue

            q_ego = quat_from_wxyz(ego2global["rotation"])    # ego->global
            t_ego = ego2global["translation"]

            q_l2e = quat_from_wxyz(lidar2ego["rotation"])     # lidar->ego
            t_l2e = lidar2ego["translation"]

            det = sample.get("det", {})
            boxes = det.get("boxes_3d", [])
            if not boxes:
                # keep empty; we'll fill later
                continue

            # scores/labels can be in det
            for i, b in enumerate(boxes):
                # expected: [x,y,z,dx,dy,dz,yaw,vx,vy]
                if b is None or len(b) < 7:
                    continue

                x, y, z = float(b[0]), float(b[1]), float(b[2])
                dx, dy, dz = float(b[3]), float(b[4]), float(b[5])  # assume already [w,l,h]
                yaw = float(b[6])
                vx = float(b[7]) if len(b) > 7 else 0.0
                vy = float(b[8]) if len(b) > 8 else 0.0

                cls = get_class_name(det, i, label_map)
                score = get_score(det, i)

                # center: lidar -> ego -> global
                p_ego = rotate_and_translate(q_l2e, t_l2e, [x, y, z])
                p_g = rotate_and_translate(q_ego, t_ego, p_ego)

                # rotation: global = q_ego * q_l2e * q_yaw(lidar)
                q_yaw = Quaternion(yaw_to_quat_wxyz(yaw))
                q_g = q_ego * q_l2e * q_yaw
                q_g = q_g.normalised

                # velocity: lidar -> ego -> global (xy only)
                v_ego = rotate_vec_xy(q_l2e, [vx, vy])
                v_g = rotate_vec_xy(q_ego, v_ego)

                if args.attribute_mode == "heuristic":
                    attr = heuristic_attribute(cls, v_g)
                else:
                    attr = ""

                results[sample_token].append({
                    "sample_token": sample_token,
                    "translation": [p_g[0], p_g[1], p_g[2]],
                    "size": [dx, dy, dz],
                    "rotation": [q_g.w, q_g.x, q_g.y, q_g.z],   # wxyz
                    "velocity": [v_g[0], v_g[1]],
                    "detection_name": cls,
                    "detection_score": float(score),
                    "attribute_name": attr,
                })

    # ensure all samples appear in results (including empty)
    # since your json is already a full val set (scene_count=150), we can fill empties from it
    all_tokens = []
    for sc in scenes:
        for sample in sc.get("samples", []):
            tok = sample.get("sample_token")
            if tok:
                all_tokens.append(tok)

    # sort and topK
    for tok in all_tokens:
        dets = results.get(tok, [])
        dets.sort(key=lambda x: x["detection_score"], reverse=True)
        results[tok] = dets[:args.max_dets_per_sample]

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

    nonempty = sum(1 for _, v in submission["results"].items() if len(v) > 0)
    print("Wrote:", args.out_json)
    print("Samples:", len(submission["results"]), "Non-empty:", nonempty)

if __name__ == "__main__":
    main()