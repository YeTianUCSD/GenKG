#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-command pipeline runner:
  run_infer.py -> run_eval.py
and archive artifacts under experiments/<run_id>/.

Artifacts per run:
- config/infer_config.(yaml|json)
- config/eval_config.(yaml|json)
- logs/infer.log
- logs/eval.log
- metrics/infer_summary.json
- metrics/eval_summary.json
- manifest.json

Example:
  python scripts/run_pipeline.py \
    --infer_config configs/infer_ebm.yaml \
    --eval_config configs/eval.yaml
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
    ap = argparse.ArgumentParser(description="Run infer+eval and archive experiment artifacts.")
    ap.add_argument("--infer_config", type=str, required=True, help="Path to infer config (yaml/json)")
    ap.add_argument("--eval_config", type=str, required=True, help="Path to eval config (yaml/json)")

    ap.add_argument("--run_name", type=str, default=None,
                    help="Optional run folder name. Default: run_<timestamp>")
    ap.add_argument("--experiments_root", type=str, default=EXPERIMENTS_DIR)

    ap.add_argument("--limit_scenes", type=int, default=None,
                    help="Override limit_scenes for both infer/eval at runtime")
    ap.add_argument("--ckpt_path", type=str, default=None,
                    help="Optional ckpt path metadata to store in manifest and pass to infer summary")

    ap.add_argument("--skip_eval", action="store_true", help="Only run infer stage")
    args = ap.parse_args()

    run_id = args.run_name if args.run_name else f"run_{_ts()}"
    exp_dir = os.path.join(args.experiments_root, run_id)
    cfg_dir = os.path.join(exp_dir, "config")
    log_dir = os.path.join(exp_dir, "logs")
    metrics_dir = os.path.join(exp_dir, "metrics")

    for d in [exp_dir, cfg_dir, log_dir, metrics_dir]:
        _ensure_dir(d)

    # Save config snapshots
    infer_cfg_dst = os.path.join(cfg_dir, os.path.basename(args.infer_config))
    eval_cfg_dst = os.path.join(cfg_dir, os.path.basename(args.eval_config))
    shutil.copy2(args.infer_config, infer_cfg_dst)
    shutil.copy2(args.eval_config, eval_cfg_dst)

    infer_summary = os.path.join(metrics_dir, "infer_summary.json")
    eval_summary = os.path.join(metrics_dir, "eval_summary.json")

    py = sys.executable

    infer_cmd: List[str] = [
        py,
        os.path.join(SCRIPTS_DIR, "run_infer.py"),
        "--config",
        args.infer_config,
        "--summary_json",
        infer_summary,
    ]
    if args.limit_scenes is not None:
        infer_cmd.extend(["--limit_scenes", str(args.limit_scenes)])
    if args.ckpt_path:
        infer_cmd.extend(["--ckpt_path", str(args.ckpt_path)])
        infer_cmd.extend(["--ebm_unary_ckpt_path", str(args.ckpt_path)])

    print(f"[pipeline] run_id={run_id}")
    print("[pipeline] infer stage...")
    infer_rc = _run_and_tee(infer_cmd, os.path.join(log_dir, "infer.log"), cwd=REPO_ROOT)

    eval_rc: Optional[int] = None
    if infer_rc == 0 and not args.skip_eval:
        eval_cmd: List[str] = [
            py,
            os.path.join(SCRIPTS_DIR, "run_eval.py"),
            "--config",
            args.eval_config,
            "--summary_json",
            eval_summary,
        ]
        if args.limit_scenes is not None:
            eval_cmd.extend(["--limit_scenes", str(args.limit_scenes)])

        print("[pipeline] eval stage...")
        eval_rc = _run_and_tee(eval_cmd, os.path.join(log_dir, "eval.log"), cwd=REPO_ROOT)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "repo_root": REPO_ROOT,
        "experiments_root": os.path.abspath(args.experiments_root),
        "infer_config": os.path.abspath(args.infer_config),
        "eval_config": os.path.abspath(args.eval_config),
        "infer_config_snapshot": infer_cfg_dst,
        "eval_config_snapshot": eval_cfg_dst,
        "limit_scenes_override": args.limit_scenes,
        "ckpt_path": args.ckpt_path,
        "infer_exit_code": infer_rc,
        "eval_exit_code": eval_rc,
        "paths": {
            "infer_log": os.path.join(log_dir, "infer.log"),
            "eval_log": os.path.join(log_dir, "eval.log"),
            "infer_summary": infer_summary,
            "eval_summary": eval_summary,
        },
    }

    manifest_path = os.path.join(exp_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[pipeline] manifest -> {manifest_path}")
    print(f"[pipeline] done. infer_rc={infer_rc} eval_rc={eval_rc}")

    if infer_rc != 0:
        raise SystemExit(infer_rc)
    if (eval_rc is not None) and (eval_rc != 0):
        raise SystemExit(eval_rc)


if __name__ == "__main__":
    main()
