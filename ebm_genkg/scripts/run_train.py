#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-command training pipeline:
  build_trainset.py -> train_unary.py / train_energy.py / train_energy_structured.py
and archive artifacts under experiments/<run_id>/.

Artifacts per run:
- config/trainset_config.(yaml|json)
- config/train_config.(yaml|json)
- config/infer_config.(yaml|json)                (optional, when eval_after_train)
- config/eval_config.(yaml|json)                 (optional, when eval_after_train)
- logs/build_trainset.log
- logs/train_unary.log / logs/train_energy.log
- logs/infer.log                                 (optional)
- logs/eval.log                                  (optional)
- logs/tune_infer.log                            (optional)
- metrics/trainset.meta.json
- metrics/train_unary.summary.json / metrics/train_energy.summary.json
- metrics/infer_summary.json                     (optional)
- metrics/eval_summary.json                      (optional)
- metrics/tune_infer.summary.json                (optional)
- checkpoints/unary_logreg.json / checkpoints/energy_logreg.json
- artifacts/out_refined.json                     (optional)
- tuning/trials/*                                (optional)
- manifest.json

Example:
  python scripts/run_train.py \
    --trainset_config configs/trainset.yaml \
    --train_config configs/train_energy.yaml \
    --train_mode energy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _log(msg: str, pipeline_log_path: Optional[str] = None) -> None:
    print(msg)
    if pipeline_log_path:
        _append_text(pipeline_log_path, msg + "\n")


def _load_cfg_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml to sync threshold.") from e
        obj = yaml.safe_load(txt)
    else:
        obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError(f"Config root must be dict, got: {type(obj)}")
    return obj


def _dump_cfg_file(path: str, cfg: Dict[str, Any]) -> None:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "w", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("YAML config requires pyyaml to sync threshold.") from e
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        else:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")


def _read_best_threshold(summary_path: str) -> Optional[float]:
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        thr = obj.get("best_threshold", None)
        if thr is None:
            return None
        return float(thr)
    except Exception:
        return None


def _sync_best_threshold_to_infer_config(
    src_cfg_path: str,
    dst_cfg_path: str,
    best_thr: float,
) -> str:
    cfg = _load_cfg_file(src_cfg_path)
    thr = float(best_thr)
    cfg["ebm_auto_threshold_from_ckpt"] = True
    cfg["ebm_ckpt_threshold_field"] = "best_threshold"
    cfg["keep_thr"] = thr
    cfg["seed_keep_thr"] = thr
    fill_old = cfg.get("fill_keep_thr", thr)
    try:
        fill_val = float(fill_old)
    except Exception:
        fill_val = thr
    cfg["fill_keep_thr"] = min(fill_val, thr)
    cfg["ebm_stage_a_keep_thr"] = thr
    _ensure_dir(os.path.dirname(dst_cfg_path))
    _dump_cfg_file(dst_cfg_path, cfg)
    return dst_cfg_path


def _run_and_tee(
    cmd: List[str],
    log_path: str,
    cwd: str,
    pipeline_log_path: Optional[str] = None,
    stage_name: Optional[str] = None,
) -> int:
    _ensure_dir(os.path.dirname(log_path))
    master_f = open(pipeline_log_path, "a", encoding="utf-8") if pipeline_log_path else None
    with open(log_path, "w", encoding="utf-8") as f:
        header = "$ " + " ".join(cmd) + "\n\n"
        f.write(header)
        f.flush()
        if master_f is not None:
            tag = stage_name if stage_name else os.path.basename(log_path)
            master_f.write(f"\n[{tag}]\n{header}")
            master_f.flush()

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
            if master_f is not None:
                master_f.write(line)
        p.wait()
        exit_line = f"\n[exit_code] {p.returncode}\n"
        f.write(exit_line)
        if master_f is not None:
            master_f.write(exit_line)
            master_f.flush()
            master_f.close()
        return int(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run build_trainset + train_unary/train_energy and archive artifacts.")
    ap.add_argument("--trainset_config", type=str, required=True, help="Path to trainset config (yaml/json)")
    ap.add_argument("--train_config", type=str, default=None,
                    help="Path to train config (yaml/json). Use with --train_mode unary|energy.")
    ap.add_argument("--train_unary_config", type=str, default=None,
                    help="Deprecated alias of --train_config. Kept for compatibility.")
    ap.add_argument("--train_mode", type=str, default="energy", choices=["unary", "energy", "energy_structured"],
                    help="Training mode. 'energy_structured' uses scripts/train_energy_structured.py.")

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
    ap.add_argument(
        "--no_sync_train_threshold",
        action="store_true",
        help="Do not auto-sync train summary best_threshold into infer config before post-train inference.",
    )
    ap.add_argument(
        "--skip_save_out_json",
        action="store_true",
        help="When eval_after_train is enabled, delete large post-train out_refined.json after eval.",
    )
    ap.add_argument("--auto_tune_infer", action="store_true",
                    help="Auto-tune infer params after training and before post-train infer/eval.")
    ap.add_argument("--tune_target_recall", type=float, default=0.76,
                    help="Recall target used by auto-tuning objective.")
    ap.add_argument("--tune_max_trials", type=int, default=16,
                    help="Max number of auto-tuning trials.")
    ap.add_argument("--tune_seed", type=int, default=42,
                    help="Random seed for auto-tuning trial sampling.")
    ap.add_argument("--tune_search_profile", type=str, default="core", choices=["core", "full"],
                    help="Auto-tuning search profile. core is faster and recommended.")

    ap.add_argument("--skip_train", action="store_true", help="Only run dataset build stage")
    args = ap.parse_args()

    # Backward-compatible config resolution
    if args.train_config is None and args.train_unary_config is None:
        raise ValueError("One of --train_config / --train_unary_config is required.")
    if args.train_config is not None and args.train_unary_config is not None:
        if os.path.abspath(args.train_config) != os.path.abspath(args.train_unary_config):
            raise ValueError("Both --train_config and --train_unary_config provided but differ.")
    resolved_train_cfg = args.train_config if args.train_config is not None else args.train_unary_config
    assert resolved_train_cfg is not None

    if args.eval_after_train and args.skip_train:
        raise ValueError("--eval_after_train requires training. Do not use with --skip_train.")
    if args.eval_after_train and (not args.infer_config or not args.eval_config):
        raise ValueError("--eval_after_train requires --infer_config and --eval_config.")

    run_id = args.run_name if args.run_name else f"train_{_ts()}"
    exp_dir = os.path.join(args.experiments_root, run_id)
    pipeline_log = os.path.join(exp_dir, f"{run_id}.log")
    cfg_dir = os.path.join(exp_dir, "config")
    log_dir = os.path.join(exp_dir, "logs")
    metrics_dir = os.path.join(exp_dir, "metrics")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    artifacts_dir = os.path.join(exp_dir, "artifacts")

    for d in [exp_dir, cfg_dir, log_dir, metrics_dir, ckpt_dir, artifacts_dir]:
        _ensure_dir(d)

    with open(pipeline_log, "w", encoding="utf-8") as f:
        f.write(f"[train_pipeline] run_id={run_id}\n")
        f.write(f"[train_pipeline] started_at={datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"[train_pipeline] repo_root={REPO_ROOT}\n")
        f.write(f"[train_pipeline] experiments_root={os.path.abspath(args.experiments_root)}\n")
        f.write(f"[train_pipeline] args={json.dumps(vars(args), ensure_ascii=False)}\n")

    # Save config snapshots
    trainset_cfg_dst = os.path.join(cfg_dir, os.path.basename(args.trainset_config))
    train_cfg_dst = os.path.join(cfg_dir, os.path.basename(resolved_train_cfg))
    shutil.copy2(args.trainset_config, trainset_cfg_dst)
    shutil.copy2(resolved_train_cfg, train_cfg_dst)
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
    energy_ckpt = os.path.join(ckpt_dir, "energy_logreg.json")
    energy_summary = os.path.join(metrics_dir, "train_energy.summary.json")
    infer_summary = os.path.join(metrics_dir, "infer_summary.json")
    eval_summary = os.path.join(metrics_dir, "eval_summary.json")
    tune_summary = os.path.join(metrics_dir, "tune_infer.summary.json")
    out_refined_json = os.path.join(artifacts_dir, "out_refined.json")
    tuning_dir = os.path.join(exp_dir, "tuning")
    tuned_infer_cfg = os.path.join(cfg_dir, "infer.tuned.yaml")

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

    _log(f"[train_pipeline] run_id={run_id}", pipeline_log)
    _log(f"[train_pipeline] run_log -> {pipeline_log}", pipeline_log)
    _log("[train_pipeline] build_trainset stage...", pipeline_log)
    build_rc = _run_and_tee(
        build_cmd,
        os.path.join(log_dir, "build_trainset.log"),
        cwd=REPO_ROOT,
        pipeline_log_path=pipeline_log,
        stage_name="build_trainset",
    )

    train_rc: Optional[int] = None
    if args.train_mode == "unary":
        train_script = "train_unary.py"
        train_ckpt = unary_ckpt
        train_summary = unary_summary
        train_log_name = "train_unary.log"
        train_stage_name = "train_unary"
    elif args.train_mode == "energy":
        train_script = "train_energy.py"
        train_ckpt = energy_ckpt
        train_summary = energy_summary
        train_log_name = "train_energy.log"
        train_stage_name = "train_energy"
    else:
        train_script = "train_energy_structured.py"
        train_ckpt = energy_ckpt
        train_summary = energy_summary
        train_log_name = "train_energy.log"
        train_stage_name = "train_energy_structured"
    if build_rc == 0 and not args.skip_train:
        train_cmd: List[str] = [
            py,
            os.path.join(SCRIPTS_DIR, train_script),
            "--config",
            resolved_train_cfg,
            "--in_npz",
            trainset_npz,
            "--out_ckpt",
            train_ckpt,
            "--out_summary",
            train_summary,
        ]
        if args.max_train_rows is not None:
            train_cmd.extend(["--max_train_rows", str(args.max_train_rows)])

        _log(f"[train_pipeline] {train_stage_name} stage...", pipeline_log)
        train_rc = _run_and_tee(
            train_cmd,
            os.path.join(log_dir, train_log_name),
            cwd=REPO_ROOT,
            pipeline_log_path=pipeline_log,
            stage_name=train_stage_name,
        )

    infer_rc: Optional[int] = None
    eval_rc: Optional[int] = None
    tune_rc: Optional[int] = None
    infer_cfg_for_eval = args.infer_config
    infer_cfg_synced: Optional[str] = None
    best_thr_synced: Optional[float] = None
    out_refined_removed = False

    if args.eval_after_train and args.auto_tune_infer and build_rc == 0 and train_rc == 0:
        tune_cmd: List[str] = [
            py,
            os.path.join(SCRIPTS_DIR, "tune_infer_params.py"),
            "--infer_config",
            args.infer_config,
            "--eval_config",
            args.eval_config,
            "--out_dir",
            tuning_dir,
            "--best_infer_config_out",
            tuned_infer_cfg,
            "--summary_json",
            tune_summary,
            "--ckpt_path",
            train_ckpt,
            "--target_recall",
            str(args.tune_target_recall),
            "--max_trials",
            str(args.tune_max_trials),
            "--seed",
            str(args.tune_seed),
            "--search_profile",
            str(args.tune_search_profile),
        ]
        if args.eval_limit_scenes is not None:
            tune_cmd.extend(["--limit_scenes", str(args.eval_limit_scenes)])

        _log("[train_pipeline] tune_infer stage (auto)...", pipeline_log)
        tune_rc = _run_and_tee(
            tune_cmd,
            os.path.join(log_dir, "tune_infer.log"),
            cwd=REPO_ROOT,
            pipeline_log_path=pipeline_log,
            stage_name="tune_infer",
        )
        if tune_rc == 0:
            infer_cfg_for_eval = tuned_infer_cfg

    if args.eval_after_train and build_rc == 0 and train_rc == 0 and (tune_rc in (None, 0)):
        if not bool(args.no_sync_train_threshold):
            best_thr = _read_best_threshold(train_summary)
            if best_thr is not None and infer_cfg_for_eval is not None:
                infer_ext = os.path.splitext(infer_cfg_for_eval)[1] or ".yaml"
                infer_cfg_synced = os.path.join(cfg_dir, f"infer.synced_thr{infer_ext}")
                try:
                    infer_cfg_for_eval = _sync_best_threshold_to_infer_config(
                        src_cfg_path=infer_cfg_for_eval,
                        dst_cfg_path=infer_cfg_synced,
                        best_thr=float(best_thr),
                    )
                    best_thr_synced = float(best_thr)
                    _log(
                        f"[train_pipeline] synced best_threshold={best_thr_synced:.4f} -> {infer_cfg_for_eval}",
                        pipeline_log,
                    )
                except Exception as e:
                    _log(f"[train_pipeline] warning: failed to sync best threshold into infer config: {repr(e)}", pipeline_log)
        else:
            _log("[train_pipeline] skip threshold sync: using infer config thresholds as-is.", pipeline_log)

        infer_cmd: List[str] = [
            py,
            os.path.join(SCRIPTS_DIR, "run_infer.py"),
            "--config",
            infer_cfg_for_eval,
            "--out_json",
            out_refined_json,
            "--summary_json",
            infer_summary,
            "--ckpt_path",
            train_ckpt,
            "--ebm_unary_ckpt_path",
            train_ckpt,
        ]
        if args.eval_limit_scenes is not None:
            infer_cmd.extend(["--limit_scenes", str(args.eval_limit_scenes)])

        _log("[train_pipeline] infer stage (post-train)...", pipeline_log)
        infer_rc = _run_and_tee(
            infer_cmd,
            os.path.join(log_dir, "infer.log"),
            cwd=REPO_ROOT,
            pipeline_log_path=pipeline_log,
            stage_name="infer",
        )

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

            _log("[train_pipeline] eval stage (post-train)...", pipeline_log)
            eval_rc = _run_and_tee(
                eval_cmd,
                os.path.join(log_dir, "eval.log"),
                cwd=REPO_ROOT,
                pipeline_log_path=pipeline_log,
                stage_name="eval",
            )
            if bool(args.skip_save_out_json) and os.path.exists(out_refined_json):
                try:
                    os.remove(out_refined_json)
                    out_refined_removed = True
                    _log(f"[train_pipeline] removed out_refined_json -> {out_refined_json}", pipeline_log)
                except Exception as e:
                    _log(f"[train_pipeline] warning: failed to remove out_refined_json: {repr(e)}", pipeline_log)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "repo_root": REPO_ROOT,
        "experiments_root": os.path.abspath(args.experiments_root),
        "trainset_config": os.path.abspath(args.trainset_config),
        "train_config": os.path.abspath(resolved_train_cfg),
        "train_unary_config": os.path.abspath(resolved_train_cfg),  # backward-compatible alias
        "trainset_config_snapshot": trainset_cfg_dst,
        "train_config_snapshot": train_cfg_dst,
        "train_unary_config_snapshot": train_cfg_dst,  # backward-compatible alias
        "train_mode": str(args.train_mode),
        "eval_after_train": bool(args.eval_after_train),
        "infer_config": os.path.abspath(args.infer_config) if args.infer_config else None,
        "eval_config": os.path.abspath(args.eval_config) if args.eval_config else None,
        "infer_config_snapshot": infer_cfg_dst,
        "eval_config_snapshot": eval_cfg_dst,
        "overrides": {
            "limit_scenes": args.limit_scenes,
            "max_train_rows": args.max_train_rows,
            "eval_limit_scenes": args.eval_limit_scenes,
            "auto_tune_infer": bool(args.auto_tune_infer),
            "tune_target_recall": float(args.tune_target_recall),
            "tune_max_trials": int(args.tune_max_trials),
            "tune_seed": int(args.tune_seed),
            "tune_search_profile": str(args.tune_search_profile),
            "no_sync_train_threshold": bool(args.no_sync_train_threshold),
        },
        "build_exit_code": build_rc,
        "train_exit_code": train_rc,
        "tune_exit_code": tune_rc,
        "infer_exit_code": infer_rc,
        "eval_exit_code": eval_rc,
        "paths": {
            "build_log": os.path.join(log_dir, "build_trainset.log"),
            "train_log": os.path.join(log_dir, train_log_name),
            "tune_log": os.path.join(log_dir, "tune_infer.log"),
            "infer_log": os.path.join(log_dir, "infer.log"),
            "eval_log": os.path.join(log_dir, "eval.log"),
            "pipeline_log": pipeline_log,
            "trainset_npz": trainset_npz,
            "trainset_meta": trainset_meta,
            "train_ckpt": train_ckpt,
            "train_summary": train_summary,
            "unary_ckpt": unary_ckpt,
            "unary_summary": unary_summary,
            "energy_ckpt": energy_ckpt,
            "energy_summary": energy_summary,
            "tune_summary": tune_summary,
            "tuned_infer_config": tuned_infer_cfg if args.auto_tune_infer else None,
            "synced_infer_config": infer_cfg_synced,
            "synced_best_threshold": best_thr_synced,
            "infer_config_used_for_eval": os.path.abspath(infer_cfg_for_eval) if args.eval_after_train else None,
            "out_refined_json": (None if out_refined_removed else out_refined_json),
            "out_refined_json_removed": bool(out_refined_removed),
            "infer_summary": infer_summary,
            "eval_summary": eval_summary,
        },
    }

    manifest_path = os.path.join(exp_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log(f"[train_pipeline] manifest -> {manifest_path}", pipeline_log)
    _log(
        f"[train_pipeline] done. build_rc={build_rc} train_rc={train_rc} "
        f"tune_rc={tune_rc} infer_rc={infer_rc} eval_rc={eval_rc}",
        pipeline_log,
    )

    if build_rc != 0:
        raise SystemExit(build_rc)
    if (train_rc is not None) and (train_rc != 0):
        raise SystemExit(train_rc)
    if (tune_rc is not None) and (tune_rc != 0):
        raise SystemExit(tune_rc)
    if (infer_rc is not None) and (infer_rc != 0):
        raise SystemExit(infer_rc)
    if (eval_rc is not None) and (eval_rc != 0):
        raise SystemExit(eval_rc)


if __name__ == "__main__":
    main()
