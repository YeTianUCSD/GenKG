#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-command training pipeline:
  build_trainset.py -> train_unary.py
and archive artifacts under experiments/<run_id>/.

Artifacts per run:
- config/trainset_config.(yaml|json)
- config/train_unary_config.(yaml|json)
- config/infer_config.(yaml|json)                (optional, when eval_after_train)
- config/eval_config.(yaml|json)                 (optional, when eval_after_train)
- logs/build_trainset.log
- logs/train_unary.log
- logs/infer.log                                 (optional)
- logs/eval.log                                  (optional)
- metrics/trainset.meta.json
- metrics/train_unary.summary.json
- metrics/infer_summary.json                     (optional)
- metrics/eval_summary.json                      (optional)
- checkpoints/unary_logreg.json
- artifacts/out_refined.json                     (optional)
- manifest.json

Example:
  python scripts/run_train.py \
    --trainset_config configs/trainset.yaml \
    --train_unary_config configs/train_unary.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_and_tee(cmd: List[str], log_path: str, cwd: str) -> int:
    _ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        p.wait()
        f.write(f"\n[exit_code] {p.returncode}\n")
        return int(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run build_trainset + train_unary and archive artifacts.")
    ap.add_argument("--trainset_config", type=str, required=True, help="Path to trainset config (yaml/json)")
    ap.add_argument("--train_unary_config", type=str, required=True, help="Path to train_unary config (yaml/json)")

    ap.add_argument("--run_name", type=str, default=None,
                    help="Optional run folder name. Default: train_<timestamp>")
    ap.add_argument("--experiments_root", type=str, default=EXPERIMENTS_DIR)

    # optional overrides
    ap.add_argument("--limit_scenes", type=int, default=None,
                    help="Override limit_scenes for build_trainset stage")
    ap.add_argument("--max_train_rows", type=int, default=None,
                    help="Override max_train_rows for train_unary stage")
    ap.add_argument("--eval_after_train", action="store_true",
                    help="After training, run infer+eval using the trained ckpt.")
    ap.add_argument("--infer_config", type=str, default=None,
                    help="Infer config path (required when --eval_after_train).")
    ap.add_argument("--eval_config", type=str, default=None,
                    help="Eval config path (required when --eval_after_train).")
    ap.add_argument("--eval_limit_scenes", type=int, default=None,
                    help="Optional limit_scenes override for infer/eval stages.")

    ap.add_argument("--skip_train", action="store_true", help="Only run dataset build stage")
    args = ap.parse_args()

    if args.eval_after_train and args.skip_train:
        raise ValueError("--eval_after_train requires training. Do not use with --skip_train.")
    if args.eval_after_train and (not args.infer_config or not args.eval_config):
        raise ValueError("--eval_after_train requires --infer_config and --eval_config.")

    run_id = args.run_name if args.run_name else f"train_{_ts()}"
    exp_dir = os.path.join(args.experiments_root, run_id)
    cfg_dir = os.path.join(exp_dir, "config")
    log_dir = os.path.join(exp_dir, "logs")
    metrics_dir = os.path.join(exp_dir, "metrics")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    artifacts_dir = os.path.join(exp_dir, "artifacts")

    for d in [exp_dir, cfg_dir, log_dir, metrics_dir, ckpt_dir, artifacts_dir]:
        _ensure_dir(d)

    # Save config snapshots
    trainset_cfg_dst = os.path.join(cfg_dir, os.path.basename(args.trainset_config))
    train_unary_cfg_dst = os.path.join(cfg_dir, os.path.basename(args.train_unary_config))
    shutil.copy2(args.trainset_config, trainset_cfg_dst)
    shutil.copy2(args.train_unary_config, train_unary_cfg_dst)
    infer_cfg_dst = None
    eval_cfg_dst = None
    if args.eval_after_train:
        infer_cfg_dst = os.path.join(cfg_dir, os.path.basename(args.infer_config))
        eval_cfg_dst = os.path.join(cfg_dir, os.path.basename(args.eval_config))
        shutil.copy2(args.infer_config, infer_cfg_dst)
        shutil.copy2(args.eval_config, eval_cfg_dst)

    trainset_npz = os.path.join(metrics_dir, "trainset.npz")
    trainset_meta = os.path.join(metrics_dir, "trainset.meta.json")
    unary_ckpt = os.path.join(ckpt_dir, "unary_logreg.json")
    unary_summary = os.path.join(metrics_dir, "train_unary.summary.json")
    infer_summary = os.path.join(metrics_dir, "infer_summary.json")
    eval_summary = os.path.join(metrics_dir, "eval_summary.json")
    out_refined_json = os.path.join(artifacts_dir, "out_refined.json")

    py = sys.executable

    build_cmd: List[str] = [
        py,
        os.path.join(SCRIPTS_DIR, "build_trainset.py"),
        "--config",
        args.trainset_config,
        "--out_npz",
        trainset_npz,
        "--out_meta",
        trainset_meta,
    ]
    if args.limit_scenes is not None:
        build_cmd.extend(["--limit_scenes", str(args.limit_scenes)])

    print(f"[train_pipeline] run_id={run_id}")
    print("[train_pipeline] build_trainset stage...")
    build_rc = _run_and_tee(build_cmd, os.path.join(log_dir, "build_trainset.log"), cwd=REPO_ROOT)

    train_rc: Optional[int] = None
    if build_rc == 0 and not args.skip_train:
        train_cmd: List[str] = [
            py,
            os.path.join(SCRIPTS_DIR, "train_unary.py"),
            "--config",
            args.train_unary_config,
            "--in_npz",
            trainset_npz,
            "--out_ckpt",
            unary_ckpt,
            "--out_summary",
            unary_summary,
        ]
        if args.max_train_rows is not None:
            train_cmd.extend(["--max_train_rows", str(args.max_train_rows)])

        print("[train_pipeline] train_unary stage...")
        train_rc = _run_and_tee(train_cmd, os.path.join(log_dir, "train_unary.log"), cwd=REPO_ROOT)

    infer_rc: Optional[int] = None
    eval_rc: Optional[int] = None
    if args.eval_after_train and build_rc == 0 and train_rc == 0:
        infer_cmd: List[str] = [
            py,
            os.path.join(SCRIPTS_DIR, "run_infer.py"),
            "--config",
            args.infer_config,
            "--out_json",
            out_refined_json,
            "--summary_json",
            infer_summary,
            "--ckpt_path",
            unary_ckpt,
            "--ebm_unary_ckpt_path",
            unary_ckpt,
        ]
        if args.eval_limit_scenes is not None:
            infer_cmd.extend(["--limit_scenes", str(args.eval_limit_scenes)])

        print("[train_pipeline] infer stage (post-train)...")
        infer_rc = _run_and_tee(infer_cmd, os.path.join(log_dir, "infer.log"), cwd=REPO_ROOT)

        if infer_rc == 0:
            eval_cmd: List[str] = [
                py,
                os.path.join(SCRIPTS_DIR, "run_eval.py"),
                "--config",
                args.eval_config,
                "--in_json",
                out_refined_json,
                "--summary_json",
                eval_summary,
            ]
            if args.eval_limit_scenes is not None:
                eval_cmd.extend(["--limit_scenes", str(args.eval_limit_scenes)])

            print("[train_pipeline] eval stage (post-train)...")
            eval_rc = _run_and_tee(eval_cmd, os.path.join(log_dir, "eval.log"), cwd=REPO_ROOT)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "repo_root": REPO_ROOT,
        "experiments_root": os.path.abspath(args.experiments_root),
        "trainset_config": os.path.abspath(args.trainset_config),
        "train_unary_config": os.path.abspath(args.train_unary_config),
        "trainset_config_snapshot": trainset_cfg_dst,
        "train_unary_config_snapshot": train_unary_cfg_dst,
        "eval_after_train": bool(args.eval_after_train),
        "infer_config": os.path.abspath(args.infer_config) if args.infer_config else None,
        "eval_config": os.path.abspath(args.eval_config) if args.eval_config else None,
        "infer_config_snapshot": infer_cfg_dst,
        "eval_config_snapshot": eval_cfg_dst,
        "overrides": {
            "limit_scenes": args.limit_scenes,
            "max_train_rows": args.max_train_rows,
            "eval_limit_scenes": args.eval_limit_scenes,
        },
        "build_exit_code": build_rc,
        "train_exit_code": train_rc,
        "infer_exit_code": infer_rc,
        "eval_exit_code": eval_rc,
        "paths": {
            "build_log": os.path.join(log_dir, "build_trainset.log"),
            "train_log": os.path.join(log_dir, "train_unary.log"),
            "infer_log": os.path.join(log_dir, "infer.log"),
            "eval_log": os.path.join(log_dir, "eval.log"),
            "trainset_npz": trainset_npz,
            "trainset_meta": trainset_meta,
            "unary_ckpt": unary_ckpt,
            "unary_summary": unary_summary,
            "out_refined_json": out_refined_json,
            "infer_summary": infer_summary,
            "eval_summary": eval_summary,
        },
    }

    manifest_path = os.path.join(exp_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[train_pipeline] manifest -> {manifest_path}")
    print(
        f"[train_pipeline] done. build_rc={build_rc} train_rc={train_rc} "
        f"infer_rc={infer_rc} eval_rc={eval_rc}"
    )

    if build_rc != 0:
        raise SystemExit(build_rc)
    if (train_rc is not None) and (train_rc != 0):
        raise SystemExit(train_rc)
    if (infer_rc is not None) and (infer_rc != 0):
        raise SystemExit(infer_rc)
    if (eval_rc is not None) and (eval_rc != 0):
        raise SystemExit(eval_rc)


if __name__ == "__main__":
    main()
