'''
python /home/code/3Ddetection/IS-Fusion/GenKG/fusion_3modality/cal_mAP/detect_quat_order.py \
--pred /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/lidar/nuscenes_val_focalformer.json \
--max_n 2000
'''

import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def iter_dets(obj):
    # obj can be list of records or dict(single record)
    if isinstance(obj, dict):
        obj = [obj]
    for rec in obj:
        for d in rec.get("detections", []):
            q = d.get("rotation", None)
            if q is None or len(q) != 4:
                continue
            yield q

def score(quats, assume):
    # assume: "wxyz" means input q=[w,x,y,z]
    # assume: "xyzw" means input q=[x,y,z,w]
    vals = []
    for q in quats:
        if assume == "wxyz":
            w, x, y, z = q
            quat_xyzw = [x, y, z, w]   # scipy expects [x,y,z,w]
        else:
            x, y, z, w = q
            quat_xyzw = [x, y, z, w]

        rot = R.from_quat(quat_xyzw)
        roll, pitch, yaw = rot.as_euler("xyz", degrees=False)
        vals.append(abs(roll) + abs(pitch))

    if not vals:
        return None
    return float(np.median(vals)), float(np.mean(vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="your raw prediction json, e.g. pred_raw.json")
    ap.add_argument("--max_n", type=int, default=5000, help="max number of boxes to sample")
    args = ap.parse_args()

    with open(args.pred, "r") as f:
        obj = json.load(f)

    quats = []
    for q in iter_dets(obj):
        quats.append(q)
        if len(quats) >= args.max_n:
            break

    print(f"Loaded {len(quats)} quaternions (sampled up to {args.max_n}).")
    s_wxyz = score(quats, "wxyz")
    s_xyzw = score(quats, "xyzw")

    print("Assume input is wxyz  => median/mean(|roll|+|pitch|):", s_wxyz)
    print("Assume input is xyzw  => median/mean(|roll|+|pitch|):", s_xyzw)

    if s_wxyz is None or s_xyzw is None:
        print("Not enough quaternions to decide.")
        return

    # smaller is better (closer to pure-yaw)
    if s_wxyz[0] < s_xyzw[0]:
        print("\n=> Likely your input rotation order is: wxyz")
        print("Use: --quat_order wxyz  (no reordering needed)")
    else:
        print("\n=> Likely your input rotation order is: xyzw")
        print("Use: --quat_order xyzw  (will convert to wxyz for nuScenes)")

if __name__ == "__main__":
    main()