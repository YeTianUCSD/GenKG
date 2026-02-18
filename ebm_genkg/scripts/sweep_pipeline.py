#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter sweep runner built on top of run_pipeline.py.

What it does:
- Loads base infer/eval config.
- Creates parameter combinations (grid search).
- Runs pipeline per combo (compact mode by default).
- Collects refined metrics from eval_summary.json.
- Writes a CSV leaderboard sorted by refined F1.

Example:
  python scripts/sweep_pipeline.py \
    --infer_config configs/infer_ebm.yaml \
    --eval_config configs/eval.yaml \
    --limit_scenes 10 \
    --keep_thrs 0.40,0.50 \
    --fill_keep_thrs 0.30,0.35,0.40 \
    --nms_thrs 1.0,1.5 \
    --out_dir experiments/sweeps/sweep_debug
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def _load_cfg(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        txt = f.read()

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml. Install it or use JSON config.") from e
        obj = yaml.safe_load(txt)
    else:
        obj = json.loads(txt)

    if not isinstance(obj, dict):
        raise ValueError(f"Config root must be dict: {path}")
    return obj


def _dump_cfg(obj: Dict[str, Any], path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    _ensure_dir(os.path.dirname(path))

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml. Install it or use JSON config.") from e
        with open(path, "w") as f:
            yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    else:
        with open(path, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def _extract_refined_f1(eval_summary_path: str, match_thr: float) -> Dict[str, float]:
    with open(eval_summary_path, "r") as f:
        obj = json.load(f)

    results = obj.get("results", []) or []
    best = None
    for r in results:
        thr = float(r.get("match_thr_xy", -1.0))
        if abs(thr - float(match_thr)) > 1e-9:
            continue
        ref = r.get("refined", None)
        if not isinstance(ref, dict):
            continue
        best = ref
        break

    if best is None:
        return {"P": 0.0, "R": 0.0, "F1": 0.0}

    return {
        "P": float(best.get("P", 0.0)),
        "R": float(best.get("R", 0.0)),
        "F1": float(best.get("F1", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid-search inference params via run_pipeline.py")
    ap.add_argument("--infer_config", type=str, required=True)
    ap.add_argument("--eval_config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=os.path.join(REPO_ROOT, "experiments", "sweeps", f"sweep_{_ts()}"))

    ap.add_argument("--limit_scenes", type=int, default=None, help="Override for faster debug")
    ap.add_argument("--match_thr", type=float, default=2.0, help="Which threshold to rank by from eval summary")
    ap.add_argument("--artifact_mode", type=str, default="compact", choices=["compact", "full"],
                    help="compact: do not keep per-run folders; full: keep run_pipeline artifacts per run.")
    ap.add_argument("--save_all_run_configs", action="store_true",
                    help="Save per-run infer/eval configs under out_dir/configs in compact mode.")
    ap.add_argument("--save_best_artifacts", action="store_true",
                    help="In compact mode, save configs and summaries for best run.")
    ap.add_argument("--save_best_out_json", action="store_true",
                    help="In compact mode, also save best refined out_json (large file).")

    # grid knobs
    ap.add_argument("--keep_thrs", type=str, default="0.50")
    ap.add_argument("--fill_keep_thrs", type=str, default="0.35")
    ap.add_argument("--nms_thrs", type=str, default="1.5")
    ap.add_argument("--topks", type=str, default="200")
    ap.add_argument("--min_dt_supports", type=str, default="2")
    ap.add_argument("--max_fills", type=str, default="220")
    ap.add_argument("--prefilter_seed", type=str, default="500")
    ap.add_argument("--prefilter_fill", type=str, default="700")

    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    infer_base = _load_cfg(args.infer_config)
    eval_base = _load_cfg(args.eval_config)

    keep_thrs = _parse_float_list(args.keep_thrs)
    fill_keep_thrs = _parse_float_list(args.fill_keep_thrs)
    nms_thrs = _parse_float_list(args.nms_thrs)
    topks = _parse_int_list(args.topks)
    min_dt_supports = _parse_int_list(args.min_dt_supports)
    max_fills = _parse_int_list(args.max_fills)
    pre_seed = _parse_int_list(args.prefilter_seed)
    pre_fill = _parse_int_list(args.prefilter_fill)

    grid = list(
        itertools.product(
            keep_thrs,
            fill_keep_thrs,
            nms_thrs,
            topks,
            min_dt_supports,
            max_fills,
            pre_seed,
            pre_fill,
        )
    )

    _ensure_dir(args.out_dir)
    cfg_dir = os.path.join(args.out_dir, "configs")
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    best_dir = os.path.join(args.out_dir, "best")
    _ensure_dir(tmp_dir)
    if args.artifact_mode == "full" or bool(getattr(args, "save_all_run_configs", False)):
        _ensure_dir(cfg_dir)
    if args.save_best_artifacts:
        _ensure_dir(best_dir)

    print(f"[sweep] out_dir={args.out_dir}")
    print(f"[sweep] artifact_mode={args.artifact_mode}")
    print(f"[sweep] total_runs={len(grid)}")

    py = sys.executable
    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] = {}
    best_f1 = -1.0

    for i, (keep_thr, fill_keep_thr, nms_thr, topk, min_dt_support, max_fill, pseed, pfill) in enumerate(grid, start=1):
        run_name = f"sweep_{i:03d}"

        infer_cfg = dict(infer_base)
        eval_cfg = dict(eval_base)

        infer_cfg["keep_thr"] = float(keep_thr)
        infer_cfg["fill_keep_thr"] = float(fill_keep_thr)
        infer_cfg["nms_thr"] = float(nms_thr)
        infer_cfg["topk"] = int(topk)
        infer_cfg["min_dt_support"] = int(min_dt_support)
        infer_cfg["max_fill"] = int(max_fill)
        infer_cfg["ebm_prefilter_topm_seed"] = int(pseed)
        infer_cfg["ebm_prefilter_topm_fill"] = int(pfill)

        if args.limit_scenes is not None:
            infer_cfg["limit_scenes"] = int(args.limit_scenes)
            eval_cfg["limit_scenes"] = int(args.limit_scenes)

        if args.artifact_mode == "full" or bool(getattr(args, "save_all_run_configs", False)):
            infer_cfg_path = os.path.join(cfg_dir, f"{run_name}.infer.yaml")
            eval_cfg_path = os.path.join(cfg_dir, f"{run_name}.eval.yaml")
        else:
            # Compact mode: reuse two temp config files to avoid config-file explosion.
            infer_cfg_path = os.path.join(tmp_dir, "infer_cfg.yaml")
            eval_cfg_path = os.path.join(tmp_dir, "eval_cfg.yaml")
        _dump_cfg(infer_cfg, infer_cfg_path)
        _dump_cfg(eval_cfg, eval_cfg_path)

        print(
            f"\n[{i}/{len(grid)}] keep={keep_thr:.3f} fill={fill_keep_thr:.3f} "
            f"nms={nms_thr:.2f} topk={topk} min_dt={min_dt_support} max_fill={max_fill} "
            f"pre=({pseed},{pfill})"
        )

        rc = 0
        if not args.dry_run:
            if args.artifact_mode == "full":
                cmd = [
                    py,
                    os.path.join(SCRIPTS_DIR, "run_pipeline.py"),
                    "--infer_config",
                    infer_cfg_path,
                    "--eval_config",
                    eval_cfg_path,
                    "--run_name",
                    run_name,
                    "--experiments_root",
                    args.out_dir,
                ]
                rc = subprocess.call(cmd, cwd=REPO_ROOT)
            else:
                # compact mode: run infer/eval directly and reuse tmp files
                tmp_out_json = os.path.join(tmp_dir, "out_refined.json")
                tmp_infer_summary = os.path.join(tmp_dir, "infer_summary.json")
                tmp_eval_summary = os.path.join(tmp_dir, "eval_summary.json")

                infer_cmd = [
                    py,
                    os.path.join(SCRIPTS_DIR, "run_infer.py"),
                    "--config",
                    infer_cfg_path,
                    "--out_json",
                    tmp_out_json,
                    "--summary_json",
                    tmp_infer_summary,
                ]
                eval_cmd = [
                    py,
                    os.path.join(SCRIPTS_DIR, "run_eval.py"),
                    "--config",
                    eval_cfg_path,
                    "--in_json",
                    tmp_out_json,
                    "--summary_json",
                    tmp_eval_summary,
                ]

                rc_infer = subprocess.call(infer_cmd, cwd=REPO_ROOT)
                rc_eval = 0
                if rc_infer == 0:
                    rc_eval = subprocess.call(eval_cmd, cwd=REPO_ROOT)
                rc = rc_infer if rc_infer != 0 else rc_eval

        row: Dict[str, Any] = {
            "run_name": run_name,
            "keep_thr": keep_thr,
            "fill_keep_thr": fill_keep_thr,
            "nms_thr": nms_thr,
            "topk": topk,
            "min_dt_support": min_dt_support,
            "max_fill": max_fill,
            "prefilter_topm_seed": pseed,
            "prefilter_topm_fill": pfill,
            "exit_code": rc,
            "P": 0.0,
            "R": 0.0,
            "F1": 0.0,
        }

        if (not args.dry_run) and rc == 0:
            if args.artifact_mode == "full":
                eval_summary = os.path.join(args.out_dir, run_name, "metrics", "eval_summary.json")
                infer_summary = os.path.join(args.out_dir, run_name, "metrics", "infer_summary.json")
                out_refined_json = os.path.join(args.out_dir, run_name, "artifacts", "out_refined.json")
            else:
                eval_summary = os.path.join(tmp_dir, "eval_summary.json")
                infer_summary = os.path.join(tmp_dir, "infer_summary.json")
                out_refined_json = os.path.join(tmp_dir, "out_refined.json")

            if os.path.exists(eval_summary):
                m = _extract_refined_f1(eval_summary, match_thr=float(args.match_thr))
                row.update(m)
                print(f"  -> REF@{args.match_thr:.2f}: P={m['P']:.4f} R={m['R']:.4f} F1={m['F1']:.4f}")

                if args.artifact_mode == "compact" and args.save_best_artifacts and float(m["F1"]) > best_f1:
                    best_f1 = float(m["F1"])
                    best_row = dict(row)
                    shutil.copy2(infer_cfg_path, os.path.join(best_dir, "best.infer.yaml"))
                    shutil.copy2(eval_cfg_path, os.path.join(best_dir, "best.eval.yaml"))
                    if os.path.exists(infer_summary):
                        shutil.copy2(infer_summary, os.path.join(best_dir, "best.infer_summary.json"))
                    if os.path.exists(eval_summary):
                        shutil.copy2(eval_summary, os.path.join(best_dir, "best.eval_summary.json"))
                    if args.save_best_out_json and os.path.exists(out_refined_json):
                        shutil.copy2(out_refined_json, os.path.join(best_dir, "best.out_refined.json"))
            else:
                print("  -> warning: missing eval_summary.json")
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: float(r.get("F1", 0.0)), reverse=True)

    out_csv = os.path.join(args.out_dir, "results.csv")
    if rows_sorted:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            writer.writeheader()
            for r in rows_sorted:
                writer.writerow(r)
    else:
        with open(out_csv, "w", newline="") as f:
            f.write("run_name\n")

    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print(f"\n[sweep] wrote {out_csv}")
    print(f"[sweep] wrote {out_json}")

    if args.artifact_mode == "compact" and args.save_best_artifacts and best_row:
        with open(os.path.join(best_dir, "best.row.json"), "w") as f:
            json.dump(best_row, f, ensure_ascii=False, indent=2)
        print(f"[sweep] wrote compact best artifacts under {best_dir}")

    if rows_sorted:
        b = rows_sorted[0]
        print(
            "[sweep] best: "
            f"run={b['run_name']} F1={float(b['F1']):.4f} "
            f"(keep={b['keep_thr']}, fill={b['fill_keep_thr']}, nms={b['nms_thr']}, "
            f"topk={b['topk']}, min_dt={b['min_dt_support']}, max_fill={b['max_fill']}, "
            f"pre=({b['prefilter_topm_seed']},{b['prefilter_topm_fill']}))"
        )


if __name__ == "__main__":
    main()
