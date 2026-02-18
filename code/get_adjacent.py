# -*- coding: utf-8 -*-
import argparse, json, os, sys
from nuscenes.nuscenes import NuScenes


'''

# 只看 sample 级前后各 1 帧
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/get_adjacent.py \
  --dataroot /home/dataset/nuscene --version v1.0-trainval \
  --sample_token b10f0cd792b64d16a1a5e8349b20504c

# sample 级和 LIDAR_TOP 的 sample_data 级前后各 3 帧，并输出绝对路径
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/get_adjacent.py \
  --dataroot /home/dataset/nuscene --version v1.0-trainval \
  --sample_token b10f0cd792b64d16a1a5e8349b20504c \
  --sensor LIDAR_TOP --k 3 --abs-path

'''

def step_links(nusc: NuScenes, table: str, start_token: str, direction: str, k: int):
    """沿 prev/next 链走 k 步，返回 token 列表。table: 'sample' or 'sample_data'"""
    assert direction in ("prev", "next")
    out, curr_tok = [], start_token
    for _ in range(max(0, k)):
        rec = nusc.get(table, curr_tok)
        nxt = rec[direction]
        if not nxt:
            break
        out.append(nxt)
        curr_tok = nxt
    return out

def sample_brief(nusc: NuScenes, sample_token: str, dataroot: str = None, abs_path: bool = False):
    s = nusc.get('sample', sample_token)
    brief = {
        "sample_token": sample_token,
        "scene_token": s["scene_token"],
        "timestamp": s["timestamp"],  # μs
        "data_channels": list(s["data"].keys())
    }
    # 顺带给出 LIDAR_TOP 的文件和时间（若存在）
    lid_tok = s["data"].get("LIDAR_TOP")
    if lid_tok:
        sd = nusc.get("sample_data", lid_tok)
        brief["lidar_file"] = os.path.join(dataroot, sd["filename"]) if abs_path else sd["filename"]
        brief["lidar_timestamp"] = sd["timestamp"]
    return brief

def sd_brief(nusc: NuScenes, sd_token: str, dataroot: str = None, abs_path: bool = False):
    sd = nusc.get("sample_data", sd_token)
    return {
        "sample_data_token": sd_token,
        "sample_token": sd["sample_token"],
        "channel": sd["channel"],
        "is_key_frame": sd["is_key_frame"],
        "timestamp": sd["timestamp"],  # μs
        "filename": os.path.join(dataroot, sd["filename"]) if abs_path else sd["filename"]
    }

def get_adjacent_sample_tokens(nusc: NuScenes, sample_token: str):
    s = nusc.get('sample', sample_token)
    return (s['prev'] or None), (s['next'] or None)

def get_adjacent_sample_data(nusc: NuScenes, sample_token: str, sensor_channel: str):
    s = nusc.get('sample', sample_token)
    if sensor_channel not in s['data']:
        raise KeyError(f"Sample has no sensor channel '{sensor_channel}'. Available: {list(s['data'].keys())}")
    sd_token = s['data'][sensor_channel]
    sd = nusc.get('sample_data', sd_token)
    return (sd['prev'] or None), (sd['next'] or None)

def main():
    parser = argparse.ArgumentParser(description="Get prev/next for nuScenes sample and sensor sample_data.")
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes dataroot (folder that contains v1.0-*)")
    parser.add_argument("--version", default="v1.0-trainval",
                        help="nuScenes version, e.g., v1.0-trainval / v1.0-mini / v1.0-test")
    parser.add_argument("--sample_token", required=True, help="Input sample token")
    parser.add_argument("--sensor", default=None,
                        help="Optional sensor channel (e.g., CAM_FRONT, LIDAR_TOP) to also print sample_data prev/next")
    parser.add_argument("--k", type=int, default=1, help="Steps to walk for prev and next (default: 1)")
    parser.add_argument("--abs-path", action="store_true", help="Print absolute file paths for sample_data")
    args = parser.parse_args()

    try:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    except Exception as e:
        print(f"Failed to initialize NuScenes: {e}")
        sys.exit(1)

    # 当前 sample 概览
    try:
        out = {"current": sample_brief(nusc, args.sample_token, args.dataroot, args.abs_path)}
    except KeyError:
        print(f"Invalid sample token: {args.sample_token}")
        sys.exit(1)

    # sample 级 prev/next 链（各走 k 步）
    prev_samples = step_links(nusc, "sample", args.sample_token, "prev", args.k)
    next_samples = step_links(nusc, "sample", args.sample_token, "next", args.k)
    out["sample_prev_chain"] = [sample_brief(nusc, tok, args.dataroot, args.abs_path) for tok in prev_samples]
    out["sample_next_chain"] = [sample_brief(nusc, tok, args.dataroot, args.abs_path) for tok in next_samples]

    # 可选：某个传感器的 sample_data 链
    if args.sensor:
        s = nusc.get('sample', args.sample_token)
        if args.sensor not in s['data']:
            out["sensor_error"] = f"Sample has no sensor channel '{args.sensor}'. Available: {list(s['data'].keys())}"
        else:
            sd0 = s['data'][args.sensor]
            out["sample_data_current"] = sd_brief(nusc, sd0, args.dataroot, args.abs_path)
            prev_sds = step_links(nusc, "sample_data", sd0, "prev", args.k)
            next_sds = step_links(nusc, "sample_data", sd0, "next", args.k)
            out["sample_data_prev_chain"] = [sd_brief(nusc, tok, args.dataroot, args.abs_path) for tok in prev_sds]
            out["sample_data_next_chain"] = [sd_brief(nusc, tok, args.dataroot, args.abs_path) for tok in next_sds]

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
