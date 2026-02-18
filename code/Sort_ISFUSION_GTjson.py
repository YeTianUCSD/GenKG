# -*- coding: utf-8 -*-
"""
按 scene 分组并在 scene 内按时间排序你的 per-sample JSON，
输出格式：{"version": "...", "scene_count": N, "scenes":[
  {"scene_token": "xxx", "num_samples": M, "samples":[...按时间升序...]}, ...
]}

用法示例：
  python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/Sort_ISFUSION_GTjson.py \
    --in /home/code/3Ddetection/IS-Fusion/GenKG/data/nusence_train/ISFUSIONres_GT_attributions_train.json \
    --out /home/code/3Ddetection/IS-Fusion/GenKG/data/sorted_by_scene_ISFUSIONres_GT_attributions_train.json \
    --dataroot /home/dataset/nuscene --version v1.0-trainval --pretty
"""
import argparse, json, os, sys
from collections import defaultdict

def load_json_auto(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)  # 普通 JSON: list 或 dict
        except Exception:
            # 尝试 JSONL
            f.seek(0)
            lines = [ln.strip() for ln in f if ln.strip()]
            return [json.loads(ln) for ln in lines]

def build_sample_to_scene_map(dataroot, version):
    """从官方 sample.json 构建 sample_token->scene_token 映射。"""
    sp = os.path.join(dataroot, version, "sample.json")
    if not os.path.exists(sp):
        raise FileNotFoundError(f"未找到 {sp}")
    with open(sp, "r") as f:
        arr = json.load(f)  # list[dict]
    return {r["token"]: r["scene_token"] for r in arr}

def get_timestamp_us(rec):
    """
    统一返回用于排序的微秒时间戳：
    - 优先使用 rec['timestamp']（若是秒则 *1e6）
    - 否则回退 rec['ego_pose_timestamp']（秒）
    """
    ts = rec.get("timestamp", None)
    if ts is not None:
        # 可能是 int 微秒，也可能是 float 秒
        if isinstance(ts, (int,)) and ts >= 1_000_000_000_000:
            return int(ts)
        try:
            # float 或较小的 int，按秒处理
            return int(float(ts) * 1e6)
        except Exception:
            pass
    ep = rec.get("ego_pose_timestamp", None)
    if ep is not None:
        return int(float(ep) * 1e6)
    # 若都没有，最后兜底为 0（不建议，但避免崩）
    return 0

def ensure_scene_token(rec, s2scene, require=True):
    st = rec.get("scene_token")
    if st:
        return st
    tok = rec.get("sample_token")
    if tok and tok in s2scene:
        st = s2scene[tok]
        rec["scene_token"] = st  # 顺便补回去，后续更方便
        return st
    if require:
        raise KeyError("记录缺少 scene_token，且无法通过 sample_token 映射得到。"
                       f" sample_token={tok}")
    return None

def main():
    ap = argparse.ArgumentParser(description="Group by scene & sort by time for nuScenes per-sample JSON.")
    ap.add_argument("--in", dest="inp", required=True, help="输入 JSON（list 或 dict 或 JSONL）")
    ap.add_argument("--out", dest="out", required=True, help="输出 JSON 路径")
    ap.add_argument("--dataroot", default=None, help="nuScenes 数据根目录（用于补全 scene_token）")
    ap.add_argument("--version", default="v1.0-trainval", help="如 v1.0-trainval / v1.0-mini")
    ap.add_argument("--pretty", action="store_true", help="缩进美化输出")
    args = ap.parse_args()

    obj = load_json_auto(args.inp)

    # 统一得到样本列表
    if isinstance(obj, dict):
        # 顶层是 dict，则把它的 value 当样本（若 key 是 sample_token 也顺便写回）
        samples = []
        for k, v in obj.items():
            if isinstance(v, dict) and "sample_token" not in v:
                v["sample_token"] = v.get("sample_token", k)
            samples.append(v)
    elif isinstance(obj, list):
        samples = obj
    else:
        print(f"不支持的顶层类型：{type(obj).__name__}")
        sys.exit(1)

    # 若样本中缺 scene_token，则需要官方映射
    need_scene_map = any("scene_token" not in r for r in samples)
    s2scene = {}
    if need_scene_map:
        if not args.dataroot:
            print("样本缺少 scene_token，且未提供 --dataroot 无法补全。")
            sys.exit(1)
        s2scene = build_sample_to_scene_map(args.dataroot, args.version)

    # 分组
    groups = defaultdict(list)  # scene_token -> list[record]
    missing_scene = 0
    for rec in samples:
        try:
            st = ensure_scene_token(rec, s2scene, require=True)
        except Exception as e:
            missing_scene += 1
            continue
        rec["_ts_us"] = get_timestamp_us(rec)  # 排序用的规范化时间
        groups[st].append(rec)
    if missing_scene:
        print(f"警告：{missing_scene} 条记录无法确定 scene_token，已跳过。")

    # 按场景的“首帧时间”对场景排序；场景内按时间升序
    scene_order = []
    for st, arr in groups.items():
        arr.sort(key=lambda r: r["_ts_us"])
        first_ts = arr[0]["_ts_us"] if arr else 0
        scene_order.append((first_ts, st))
    scene_order.sort(key=lambda x: x[0])

    out = {
        "version": args.version,
        "scene_count": len(scene_order),
        "scenes": []
    }

    for _, st in scene_order:
        arr = groups[st]
        # 清理辅助字段
        for r in arr:
            if "_ts_us" in r:
                r["timestamp_us"] = r["_ts_us"]  # 顺便保留一个规范化微秒时间戳
                del r["_ts_us"]
        out["scenes"].append({
            "scene_token": st,
            "num_samples": len(arr),
            "samples": arr
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2 if args.pretty else None)

    print(f"[OK] 已写出：{args.out}")
    print(f"  scenes: {out['scene_count']}  samples: {sum(s['num_samples'] for s in out['scenes'])}")

if __name__ == "__main__":
    main()
