# -*- coding: utf-8 -*-
import argparse, json, os, sys

'''
# 打印总数，并查看第 2 条（0 起始），保持插入顺序
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_nusence_json.py \
  --path /home/code/3Ddetection/IS-Fusion/GenKG/data/extracted_annotations_rxy.json --index 2 --pretty

# 若你的 JSON 顶层是 dict，想按键名排序后再取第 2 条
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_nusence_json.py \
  --path /home/code/3Ddetection/IS-Fusion/GenKG/data/extracted_annotations_rxy.json --index 2 --sort-keys --pretty

# 直接按 sample_token（顶层键）取值
python -u /home/code/3Ddetection/IS-Fusion/GenKG/code/read_nusence_json.py \
  --path /home/code/3Ddetection/IS-Fusion/GenKG/data/extracted_annotations_rxy.json --index 0 \
  --by-key fd8420396768425eabec9bdddf7e64b6 --pretty

'''

def load_json_auto(path):
    """先按普通 JSON 解析，失败则尝试 JSON Lines（每行一个 JSON）。"""
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            f.seek(0)
            lines = [ln.strip() for ln in f if ln.strip()]
            try:
                return [json.loads(ln) for ln in lines]
            except Exception as e:
                raise RuntimeError(f"无法解析为 JSON 或 JSONL：{e}")

def main():
    ap = argparse.ArgumentParser(description="Preview nuScenes JSON")
    ap.add_argument("--path", required=True, help="JSON 路径（如 extracted_annotations_rxy.json）")
    ap.add_argument("--index", type=int, required=True, help="要打印的第 n 条（0 起始）")
    ap.add_argument("--sort-keys", action="store_true",
                    help="若为 dict，先按键名排序后再取第 n 条")
    ap.add_argument("--by-key", type=str, default=None,
                    help="直接按键名（如 sample_token）取值，忽略 --index")
    ap.add_argument("--pretty", action="store_true",
                    help="使用缩进美化输出（可能很长）")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"文件不存在：{args.path}")
        sys.exit(1)

    obj = load_json_auto(args.path)

    # 顶层是 dict 或 list 都支持
    if isinstance(obj, dict):
        keys = list(obj.keys())
        total = len(keys)
        print(f"Top-level type: dict")
        print(f"Total entries: {total}")

        if args.by_key is not None:
            key = args.by_key
            if key not in obj:
                print(f"指定 key 不存在：{key}")
                print(f"可用示例 key（前 5 个）：{keys[:5]}")
                sys.exit(1)
        else:
            if args.sort_keys:
                keys = sorted(keys)
            if not (0 <= args.index < total):
                print(f"索引越界：{args.index} / {total}")
                sys.exit(1)
            key = keys[args.index]

        val = obj[key]
        print("\n=== Entry ===")
        print(f"key: {key}")
        # 如果 value 是 list，顺便打印里面元素数量
        if isinstance(val, list):
            print(f"value_type: list  (len={len(val)})")
        else:
            print(f"value_type: {type(val).__name__}")

        if args.pretty:
            print(json.dumps(val, ensure_ascii=False, indent=2))
        else:
            # 非 pretty 输出一行，防止太长
            s = json.dumps(val, ensure_ascii=False)
            print(s)
    elif isinstance(obj, list):
        total = len(obj)
        print(f"Top-level type: list")
        print(f"Total entries: {total}")

        if not (0 <= args.index < total):
            print(f"索引越界：{args.index} / {total}")
            sys.exit(1)

        val = obj[args.index]
        print("\n=== Entry ===")
        print(f"index: {args.index}")
        print(f"value_type: {type(val).__name__}")
        if args.pretty:
            print(json.dumps(val, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(val, ensure_ascii=False))
    else:
        print(f"不支持的顶层类型：{type(obj).__name__}")

if __name__ == "__main__":
    main()
