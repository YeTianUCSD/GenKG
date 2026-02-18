#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-tune inference hyperparameters with a recall-first objective.

This script runs a bounded search over infer config knobs by repeatedly calling:
  scripts/run_infer.py -> scripts/run_eval.py

It selects the best trial using:
  - primary: satisfy R >= target_recall and maximize precision
  - fallback (if none satisfy): maximize recall, then precision
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_cfg(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml.") from e
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
            raise RuntimeError("YAML config requires pyyaml.") from e
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


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


def _extract_refined_metrics(eval_summary_path: str) -> Optional[Dict[str, float]]:
    try:
        with open(eval_summary_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        results = obj.get("results", []) or []
        for rec in results:
            ref = rec.get("refined", None)
            if isinstance(ref, dict):
                return {
                    "P": float(ref.get("P", 0.0)),
                    "R": float(ref.get("R", 0.0)),
                    "F1": float(ref.get("F1", 0.0)),
                }
        return None
    except Exception:
        return None


def _is_better(
    cand: Dict[str, Any],
    best: Optional[Dict[str, Any]],
    target_recall: float,
) -> bool:
    if best is None:
        return True

    c_ok = float(cand.get("R", 0.0)) >= float(target_recall)
    b_ok = float(best.get("R", 0.0)) >= float(target_recall)
    if c_ok != b_ok:
        return c_ok

    if c_ok:
        c_key = (float(cand.get("P", 0.0)), float(cand.get("F1", 0.0)), float(cand.get("R", 0.0)))
        b_key = (float(best.get("P", 0.0)), float(best.get("F1", 0.0)), float(best.get("R", 0.0)))
        return c_key > b_key

    c_key = (float(cand.get("R", 0.0)), float(cand.get("P", 0.0)), float(cand.get("F1", 0.0)))
    b_key = (float(best.get("R", 0.0)), float(best.get("P", 0.0)), float(best.get("F1", 0.0)))
    return c_key > b_key


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tune inference params under recall constraint.")
    ap.add_argument("--infer_config", type=str, required=True)
    ap.add_argument("--eval_config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--best_infer_config_out", type=str, required=True)
    ap.add_argument("--summary_json", type=str, required=True)

    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--limit_scenes", type=int, default=None)
    ap.add_argument("--target_recall", type=float, default=0.76)
    ap.add_argument("--max_trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search_profile", type=str, default="core", choices=["core", "full"],
                    help="core: search only key params; full: include prefilter params too.")

    ap.add_argument("--seed_keep_thrs", type=str, default="0.40,0.45,0.50")
    ap.add_argument("--fill_keep_thrs", type=str, default="0.20,0.25,0.30,0.35")
    ap.add_argument("--min_dt_supports", type=str, default="1,2")
    ap.add_argument("--min_dist_to_seeds", type=str, default="0.30,0.50,0.80")
    ap.add_argument("--max_fills", type=str, default="220,320,450")
    ap.add_argument("--nms_thrs", type=str, default="1.2,1.5")
    ap.add_argument("--prefilter_topm_seeds", type=str, default="500,800",
                    help="Only used in full profile.")
    ap.add_argument("--prefilter_topm_fills", type=str, default="700,1200",
                    help="Only used in full profile.")
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    _ensure_dir(args.out_dir)
    trials_dir = os.path.join(args.out_dir, "trials")
    _ensure_dir(trials_dir)

    infer_base = _load_cfg(args.infer_config)
    eval_base = _load_cfg(args.eval_config)
    py = sys.executable

    core_space = list(
        itertools.product(
            _parse_float_list(args.seed_keep_thrs),
            _parse_float_list(args.fill_keep_thrs),
            _parse_int_list(args.min_dt_supports),
            _parse_float_list(args.min_dist_to_seeds),
            _parse_int_list(args.max_fills),
            _parse_float_list(args.nms_thrs),
        )
    )
    if args.search_profile == "full":
        full_space = list(
            itertools.product(
                _parse_float_list(args.seed_keep_thrs),
                _parse_float_list(args.fill_keep_thrs),
                _parse_int_list(args.min_dt_supports),
                _parse_float_list(args.min_dist_to_seeds),
                _parse_int_list(args.max_fills),
                _parse_float_list(args.nms_thrs),
                _parse_int_list(args.prefilter_topm_seeds),
                _parse_int_list(args.prefilter_topm_fills),
            )
        )
        space = full_space
    else:
        # Fix prefilter params to base infer config; focus on key knobs.
        base_seed = int(infer_base.get("ebm_prefilter_topm_seed", 500))
        base_fill = int(infer_base.get("ebm_prefilter_topm_fill", 700))
        space = [
            (a, b, c, d, e, f, base_seed, base_fill)
            for (a, b, c, d, e, f) in core_space
        ]

    if len(space) > int(args.max_trials):
        # Sample to bound runtime.
        rng = random.Random(int(args.seed))
        idx = list(range(len(space)))
        rng.shuffle(idx)
        idx = idx[: int(args.max_trials)]
        idx.sort()
        space = [space[i] for i in idx]

    print(
        f"[tune] profile={args.search_profile} trials={len(space)} "
        f"target_recall={args.target_recall:.4f} max_trials={args.max_trials}"
    )

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_cfg: Optional[Dict[str, Any]] = None

    for i, (seed_keep, fill_keep, min_dt, min_dist, max_fill, nms_thr, topm_seed, topm_fill) in enumerate(space, start=1):
        tag = f"trial_{i:03d}"
        print(
            f"\n[{i}/{len(space)}] seed={seed_keep:.2f} fill={fill_keep:.2f} min_dt={min_dt} "
            f"min_dist={min_dist:.2f} max_fill={max_fill} nms={nms_thr:.2f} "
            f"topm=({topm_seed},{topm_fill})"
        )

        infer_cfg = dict(infer_base)
        infer_cfg["seed_keep_thr"] = float(seed_keep)
        infer_cfg["fill_keep_thr"] = float(fill_keep)
        infer_cfg["min_dt_support"] = int(min_dt)
        infer_cfg["min_dist_to_seed"] = float(min_dist)
        infer_cfg["max_fill"] = int(max_fill)
        infer_cfg["nms_thr"] = float(nms_thr)
        infer_cfg["ebm_prefilter_topm_seed"] = int(topm_seed)
        infer_cfg["ebm_prefilter_topm_fill"] = int(topm_fill)
        if args.limit_scenes is not None:
            infer_cfg["limit_scenes"] = int(args.limit_scenes)

        eval_cfg = dict(eval_base)
        if args.limit_scenes is not None:
            eval_cfg["limit_scenes"] = int(args.limit_scenes)

        infer_cfg_path = os.path.join(trials_dir, f"{tag}.infer.yaml")
        eval_cfg_path = os.path.join(trials_dir, f"{tag}.eval.yaml")
        out_json = os.path.join(trials_dir, f"{tag}.out_refined.json")
        infer_summary = os.path.join(trials_dir, f"{tag}.infer_summary.json")
        eval_summary = os.path.join(trials_dir, f"{tag}.eval_summary.json")
        infer_log = os.path.join(trials_dir, f"{tag}.infer.log")
        eval_log = os.path.join(trials_dir, f"{tag}.eval.log")

        _dump_cfg(infer_cfg, infer_cfg_path)
        _dump_cfg(eval_cfg, eval_cfg_path)

        infer_cmd: List[str] = [
            py,
            os.path.join(SCRIPTS_DIR, "run_infer.py"),
            "--config",
            infer_cfg_path,
            "--out_json",
            out_json,
            "--summary_json",
            infer_summary,
        ]
        if args.ckpt_path:
            infer_cmd.extend(["--ckpt_path", str(args.ckpt_path)])
            infer_cmd.extend(["--ebm_unary_ckpt_path", str(args.ckpt_path)])
        if args.limit_scenes is not None:
            infer_cmd.extend(["--limit_scenes", str(args.limit_scenes)])

        rc_infer = _run_and_tee(infer_cmd, infer_log, cwd=REPO_ROOT)
        rc_eval = -1
        metrics: Optional[Dict[str, float]] = None
        if rc_infer == 0:
            eval_cmd: List[str] = [
                py,
                os.path.join(SCRIPTS_DIR, "run_eval.py"),
                "--config",
                eval_cfg_path,
                "--in_json",
                out_json,
                "--summary_json",
                eval_summary,
            ]
            if args.limit_scenes is not None:
                eval_cmd.extend(["--limit_scenes", str(args.limit_scenes)])
            rc_eval = _run_and_tee(eval_cmd, eval_log, cwd=REPO_ROOT)
            if rc_eval == 0:
                metrics = _extract_refined_metrics(eval_summary)

        row: Dict[str, Any] = {
            "trial": tag,
            "seed_keep_thr": float(seed_keep),
            "fill_keep_thr": float(fill_keep),
            "min_dt_support": int(min_dt),
            "min_dist_to_seed": float(min_dist),
            "max_fill": int(max_fill),
            "nms_thr": float(nms_thr),
            "ebm_prefilter_topm_seed": int(topm_seed),
            "ebm_prefilter_topm_fill": int(topm_fill),
            "infer_exit_code": int(rc_infer),
            "eval_exit_code": int(rc_eval),
            "P": 0.0,
            "R": 0.0,
            "F1": 0.0,
            "infer_cfg": infer_cfg_path,
            "eval_cfg": eval_cfg_path,
            "out_json": out_json,
            "infer_summary": infer_summary,
            "eval_summary": eval_summary,
        }

        if metrics is not None:
            row.update(metrics)
            print(
                f"  -> REF: P={row['P']:.4f} R={row['R']:.4f} F1={row['F1']:.4f} "
                f"(target R>={args.target_recall:.2f})"
            )
            if _is_better(row, best, target_recall=float(args.target_recall)):
                best = dict(row)
                best_cfg = dict(infer_cfg)
        else:
            print(f"  -> failed: infer_rc={rc_infer} eval_rc={rc_eval}")

        rows.append(row)

    if best is None or best_cfg is None:
        raise SystemExit("[tune] no successful trial. Check trial logs under out_dir.")

    _dump_cfg(best_cfg, args.best_infer_config_out)

    summary = {
        "stage": "tune_infer_params",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target_recall": float(args.target_recall),
        "max_trials": int(args.max_trials),
        "num_trials": int(len(rows)),
        "best": best,
        "best_infer_config_out": os.path.abspath(args.best_infer_config_out),
        "args": vars(args),
        "trials": rows,
    }
    _ensure_dir(os.path.dirname(os.path.abspath(args.summary_json)))
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[tune] best config -> {args.best_infer_config_out}")
    print(
        f"[tune] best REF: P={best['P']:.4f} R={best['R']:.4f} F1={best['F1']:.4f} "
        f"(trial={best['trial']})"
    )
    print(f"[tune] summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
