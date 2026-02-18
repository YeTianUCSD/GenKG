#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格搜索超参组合：
  (frame_radius, match_thr, weight_decay)

每组参数：
  - 启动 train_det_transformer_V4.py
  - 训练若干个 epoch
  - 解析 log 中每一行 "[E{epoch}] Val: P=.. R=.. F1=.."
    取 F1 最大的那一行作为该组参数的 best 结果

同时：
  - 每组参数的训练输出都会「实时」打印到终端，
    并写到该组合自己的 log 文件里：
      BASE_OUT_DIR/r{R}_mt{MT}_wd{WD}/train_r{R}_mt{MT}_wd{WD}.log
  - 全局维护一个 best F1：
    只要出现新的全局 best，就把该 run 下的 best.pt 复制到：
      BASE_OUT_DIR/best_overall.pt
    并写 BASE_OUT_DIR/best_overall_meta.json

    python /home/code/3Ddetection/IS-Fusion/GenKG/code/model/search_hparams_V6.py

    cd /home/code/3Ddetection/IS-Fusion/GenKG/code/model

永久执行：
PIDFILE=/home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_v6.pid
LOGFILE=/home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_v6.log
setsid nohup python /home/code/3Ddetection/IS-Fusion/GenKG/code/model/search_hparams_V6.py \
  >> "$LOGFILE" 2>&1 & printf '%s\n' "$!" > "$PIDFILE"


删除：
pkill -TERM -f search_hparams_V6.py
train_det_transformer_V5.py
search_hparams_V6.py

# 1) 杀掉搜索脚本本身
kill "$(cat /home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_v6.pid)"
ps aux | grep train_det_transformer_V5.py | grep -v grep
pkill -TERM -f torchrun
"""




import os
import re
import json
import shutil
import subprocess
from itertools import product

# ======== 根据自己环境修改这里几个路径/参数 ========

TRAIN_SCRIPT = "/home/code/3Ddetection/IS-Fusion/GenKG/code/model/train_det_transformer_V6.py"
TRAIN_JSON   = "/home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONres_GT_attributions_train.json"
VAL_JSON     = "/home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json"

# 所有搜索结果统一放在这个目录下面
BASE_OUT_DIR = "/home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV6/search_res_V1"

# 使用多少卡（跟你平时 torchrun 一致）
NPROC  = 2
EPOCHS = 20
KEEP_THR        = 0.5
IGNORE_CLASSES  = "-1"
MAX_PER_FRAME   = 80
LOG_INTERVAL    = 50
LR              = 2e-4

GRAD_CLIP       = 5.0
EMA_MOMENTUM    = 0.9

# ======== 超参搜索空间 ========

FRAME_RADIUS_LIST = [2]                 # 时间窗口半径
MATCH_THR_LIST    = [1, 2, 3, 5]        # 匹配距离阈值
WEIGHT_DECAY_LIST = [1e-3]              # AdamW 正则强度

# ✨ 新增：merge_radius / max_tokens 的搜索空间
MERGE_RADIUS_LIST = [2]          # 0.0 = 不合并；>0 开启合并
MAX_TOKENS_LIST   = [1024]         # 单个 clip 的 token 上限

# ======== 日志解析：匹配 [E{epoch}] Val: P=.. R=.. F1=.. ========

VAL_LINE_PATTERN = re.compile(
    r"\[E(\d+)\]\s+Val:\s+P=([0-9.]+)\s+R=([0-9.]+)\s+F1=([0-9.]+)"
)


def run_one_trial(frame_radius: int,
                  match_thr: float,
                  weight_decay: float,
                  merge_radius: float,
                  max_tokens: int):
    """
    跑一组 (frame_radius, match_thr, weight_decay, merge_radius, max_tokens) 组合。
    返回：
      {
        "best_f1": ...,
        "best_epoch": ...,
        "best_P": ...,
        "best_R": ...,
        "log_path": ...,
        "out_dir": ...,
        "returncode": ...,
      }
    """
    # 浮点转 tag：把 '.' 换成 'p'
    mt_tag = f"{match_thr}".replace(".", "p")
    wd_tag = f"{weight_decay}".replace(".", "p")
    mr_tag = f"{merge_radius}".replace(".", "p")
    tok_tag = f"{max_tokens}"

    out_dir = os.path.join(
        BASE_OUT_DIR,
        f"r{frame_radius}_mt{mt_tag}_wd{wd_tag}_mr{mr_tag}_tok{tok_tag}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # log 文件名里带上参数，方便区分
    log_filename = f"train_r{frame_radius}_mt{mt_tag}_wd{wd_tag}_mr{mr_tag}_tok{tok_tag}.log"
    log_path = os.path.join(out_dir, log_filename)

    cmd = [
        "torchrun", f"--nproc_per_node={NPROC}",
        TRAIN_SCRIPT,
        "--train_json", TRAIN_JSON,
        "--val_json",   VAL_JSON,
        "--out_dir",    out_dir,
        "--epochs",     str(EPOCHS),
        "--frame_radius", str(frame_radius),
        "--match_thr",    str(match_thr),
        "--keep_thr",     str(KEEP_THR),
        "--ignore_classes", IGNORE_CLASSES,
        "--max_per_frame", str(MAX_PER_FRAME),
        "--log_interval",  str(LOG_INTERVAL),
        "--lr",            str(LR),
        "--weight_decay",  str(weight_decay),
        "--grad_clip",     str(GRAD_CLIP),
        "--ema_momentum",  str(EMA_MOMENTUM),
        # ✨ 新增的两个参数
        "--merge_radius",  str(merge_radius),
        "--max_tokens",    str(max_tokens),
    ]

    print(f"\n=== Run: frame_radius={frame_radius}, match_thr={match_thr}, "
          f"weight_decay={weight_decay}, merge_radius={merge_radius}, "
          f"max_tokens={max_tokens} ===")
    print("Out dir :", out_dir)
    print("Log file:", log_path)
    print("Cmd:", " ".join(cmd))

    # 用 Popen + stdout 管道，实现「终端实时打印 + 写 log 文件」
    lines = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    with open(log_path, "w") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)

    returncode = proc.wait()
    full_log_text = "".join(lines)

    # 解析 [E{epoch}] Val: P=.. R=.. F1=.. 行，找本 run 内 best
    matches = VAL_LINE_PATTERN.findall(full_log_text)

    best_f1 = 0.0
    best_epoch = -1
    best_P = 0.0
    best_R = 0.0

    for ep_str, P_str, R_str, F1_str in matches:
        ep = int(ep_str)
        P = float(P_str)
        R = float(R_str)
        F1 = float(F1_str)
        if F1 > best_f1:
            best_f1 = F1
            best_epoch = ep
            best_P = P
            best_R = R

    print(f"  -> best F1 from log: {best_f1:.4f} (epoch {best_epoch}), "
          f"P={best_P:.3f}, R={best_R:.3f}, returncode={returncode}")

    info = {
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "best_P": best_P,
        "best_R": best_R,
        "log_path": log_path,
        "out_dir": out_dir,
        "returncode": returncode,
    }
    return info


def main():
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    all_results = []

    global_best_f1 = 0.0
    global_best_info = None
    best_model_path = os.path.join(BASE_OUT_DIR, "best_overall.pt")
    best_meta_path  = os.path.join(BASE_OUT_DIR, "best_overall_meta.json")

    # ✨ product 里加上 merge_radius / max_tokens
    for frame_radius, match_thr, weight_decay, merge_radius, max_tokens in product(
        FRAME_RADIUS_LIST, MATCH_THR_LIST, WEIGHT_DECAY_LIST,
        MERGE_RADIUS_LIST, MAX_TOKENS_LIST
    ):
        trial_info = run_one_trial(frame_radius, match_thr, weight_decay,
                                   merge_radius, max_tokens)

        result = {
            "frame_radius": frame_radius,
            "match_thr": match_thr,
            "weight_decay": weight_decay,
            "merge_radius": merge_radius,
            "max_tokens": max_tokens,
            **trial_info,
        }
        all_results.append(result)

        # 更新“全局 best”
        best_f1 = trial_info["best_f1"]
        best_epoch = trial_info["best_epoch"]

        if best_epoch >= 0 and best_f1 > global_best_f1:
            print("\n*** New GLOBAL BEST found! ***")
            print(f"Old global best F1: {global_best_f1:.4f}")
            print(f"New global best F1: {best_f1:.4f} "
                  f"(radius={frame_radius}, match_thr={match_thr}, "
                  f"weight_decay={weight_decay}, merge_radius={merge_radius}, "
                  f"max_tokens={max_tokens}, epoch={best_epoch})")

            global_best_f1 = best_f1
            global_best_info = result

            # 拷贝当前 out_dir 下的 best.pt
            src_best = os.path.join(trial_info["out_dir"], "best.pt")
            if os.path.exists(src_best):
                shutil.copy2(src_best, best_model_path)
                print(f"  -> Copied model from {src_best} to {best_model_path}")
                with open(best_meta_path, "w") as f:
                    json.dump(global_best_info, f, indent=2)
                print(f"  -> Wrote global best meta to {best_meta_path}")
            else:
                print(f"  [WARN] best.pt not found in {trial_info['out_dir']}, "
                      f"cannot copy global best model.")

    # 按 F1 从大到小排序所有结果
    all_results.sort(key=lambda x: x["best_f1"], reverse=True)

    # 保存全部搜索结果
    result_json = os.path.join(BASE_OUT_DIR, "search_results.json")
    with open(result_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n=== All trials finished. Results saved to:", result_json)

    # 打印 Top 5
    print("\n=== Top 5 configs ===")
    for i, r in enumerate(all_results[:5]):
        print(
            f"#{i+1}: F1={r['best_f1']:.4f} (epoch {r['best_epoch']}) | "
            f"P={r['best_P']:.3f} R={r['best_R']:.3f} | "
            f"radius={r['frame_radius']} match_thr={r['match_thr']} "
            f"weight_decay={r['weight_decay']} merge_radius={r['merge_radius']} "
            f"max_tokens={r['max_tokens']} | log={r['log_path']}"
        )

    # 全局 best 汇总
    if global_best_info is not None:
        print("\n=== Global best summary ===")
        print(
            f"F1={global_best_info['best_f1']:.4f} "
            f"(epoch {global_best_info['best_epoch']}) | "
            f"P={global_best_info['best_P']:.3f} R={global_best_info['best_R']:.3f} | "
            f"radius={global_best_info['frame_radius']} "
            f"match_thr={global_best_info['match_thr']} "
            f"weight_decay={global_best_info['weight_decay']} "
            f"merge_radius={global_best_info['merge_radius']} "
            f"max_tokens={global_best_info['max_tokens']}"
        )
        print("Model path:", best_model_path)
        print("Meta path :", best_meta_path)
    else:
        print("\n[WARN] No valid epoch found in any trial; global best not set.")


if __name__ == "__main__":
    main()
