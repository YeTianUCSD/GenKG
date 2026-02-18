#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 log 文件中解析每个 epoch 的 P/R/F1，并画三条折线图。

日志示例：
[Epoch 1] step 2400/2407 | loss=0.1893 keep=0.1867 cls=0.0026
[E1] P=0.757 R=0.639 F1=0.693 (tp=19436, fp=6251, fn=10986)

python /home/code/3Ddetection/IS-Fusion/GenKG/code/model/picture_showEpochValue.py \
  --log /home/code/3Ddetection/IS-Fusion/GenKG/code/model/traincopy.log \
  --out /home/code/3Ddetection/IS-Fusion/GenKG/code/model/outV3/prf_curve.png



...
"""

import re
import argparse
import matplotlib.pyplot as plt


def parse_log(log_path):
    """
    解析 log 文件，返回：
      epochs: [1, 2, 3, ...]
      P_list: [0.757, ...]
      R_list
      F1_list
      tp_list, fp_list, fn_list
    只会匹配类似：
      [E1] P=0.757 R=0.639 F1=0.693 (tp=19436, fp=6251, fn=10986)
    的行，其它训练 step 行全部忽略。
    """
    pattern = re.compile(
        r"\[E(\d+)\]\s+P=([0-9.]+)\s+R=([0-9.]+)\s+F1=([0-9.]+)\s*"
        r"\(tp=(\d+),\s*fp=(\d+),\s*fn=(\d+)\)"
    )

    epochs, P_list, R_list, F1_list = [], [], [], []
    tp_list, fp_list, fn_list = [], [], []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue  # 不符合 [E..] 这种格式的行直接跳过

            epoch = int(m.group(1))
            P = float(m.group(2))
            R = float(m.group(3))
            F1 = float(m.group(4))
            tp = int(m.group(5))
            fp = int(m.group(6))
            fn = int(m.group(7))

            epochs.append(epoch)
            P_list.append(P)
            R_list.append(R)
            F1_list.append(F1)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

    if not epochs:
        return [], [], [], [], [], [], []

    # 按 epoch 排序，防止日志顺序被打乱
    zipped = list(zip(epochs, P_list, R_list, F1_list, tp_list, fp_list, fn_list))
    zipped.sort(key=lambda x: x[0])

    epochs, P_list, R_list, F1_list, tp_list, fp_list, fn_list = zip(*zipped)

    return list(epochs), list(P_list), list(R_list), list(F1_list), list(tp_list), list(fp_list), list(fn_list)


def plot_prf(epochs, P_list, R_list, F1_list, out_path=None):
    # 如果你之前设置过别的全局 style，这行可以强制回默认
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(8, 5))

    # linewidth 控制线条粗细，markersize 控制点的大小
    ax.plot(epochs, P_list,
            linewidth=0.8, label="Precision (P)")
    ax.plot(epochs, R_list,
            linewidth=0.8, label="Recall (R)")
    ax.plot(epochs, F1_list,
            linewidth=0.8, label="F1")

    unique_epochs = sorted(set(epochs))
    ax.set_xticks(unique_epochs[::5])
    ax.set_ylabel("Score")
    ax.set_title("Validation P/R/F1 per Epoch")

    # 去掉灰色背景网格
    # 如果你想要纯白背景 + 没有网格，就把 grid 关掉
    ax.grid(False)
    # 再保险一点，把背景色设为白色
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.set_xticks(epochs)
    ax.legend()
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"图像已保存到: {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="log 文件路径")
    parser.add_argument("--out", default=None, help="输出图片路径，如 prf_curve.png；不填则直接显示")
    args = parser.parse_args()

    epochs, P_list, R_list, F1_list, tp_list, fp_list, fn_list = parse_log(args.log)

    if not epochs:
        print("没有匹配到任何 [E*] P/R/F1 行，请检查 log 格式或正则。")
        return

    print(f"共解析到 {len(epochs)} 个 epoch：", epochs)
    plot_prf(epochs, P_list, R_list, F1_list, args.out)


if __name__ == "__main__":
    main()
