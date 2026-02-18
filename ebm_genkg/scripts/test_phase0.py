#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase-0 smoke test for:
- ebm_refine/io.py
- ebm_refine/geom.py

What it checks:
1) JSON loading + sample discovery + grouping by scene.
2) Transform consistency:
   - T_g2l is inverse of T_l2g
   - T_k2ref consistency with global composition:
       p_ref == (T_g2l_ref @ T_l2g_k) p_k
   - T_k2ref(ref, ref) ≈ Identity
3) Writeback sanity:
   - write det_refined into output JSON for one sample
   - output JSON remains readable and contains the inserted field

Run:
export PYTHONPATH="$PWD:$PYTHONPATH"

  python scripts/test_phase0.py --json /home/code/3Ddetection/IS-Fusion/GenKG/code/model/dataset/sorted_by_scene_ISFUSIONandGTattr_val.json --out_json /home/code/3Ddetection/IS-Fusion/GenKG/ebm_genkg/tmp/out.json
"""

import argparse
import numpy as np

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from json_io import (
    load_root_and_samples,
    group_by_scene,
    sample_key_candidates,
    write_refined_json,
)
from geom import (
    T_l2g_from_sample,
    T_g2l_from_sample,
    T_k2ref,
    T_inv,
    transform_points,
    rotate_yaw,
    rotate_vel_xy,
)


def fro_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A.reshape(-1), ord=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Input JSON path (contains det+pose; gt optional)")
    ap.add_argument("--out_json", default="", help="Optional output JSON for writeback test")
    ap.add_argument("--require_gt", action="store_true", help="Require gt when discovering samples")
    ap.add_argument("--num_scenes", type=int, default=1, help="How many scenes to test (first N)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    print(f"[1] Loading samples from: {args.json}")
    lr = load_root_and_samples(args.json, require_gt=args.require_gt)
    samples = lr.samples
    print(f"    Found samples: {len(samples)}")

    scenes = group_by_scene(samples)
    print(f"    Found scenes: {len(scenes)}")

    if len(scenes) == 0:
        raise RuntimeError("No scenes found. Check JSON structure and iter_samples heuristic.")

    # Take first N scenes
    scene_tokens = list(scenes.keys())[: max(1, args.num_scenes)]
    print(f"[2] Testing transforms on {len(scene_tokens)} scene(s): {scene_tokens}")

    for sc_i, sc in enumerate(scene_tokens, 1):
        frames = scenes[sc]
        print(f"  Scene {sc_i}: {sc} | #frames={len(frames)}")

        if len(frames) < 1:
            continue

        # pick ref = first frame
        ref = frames[0]
        T_l2g_ref = T_l2g_from_sample(ref)
        T_g2l_ref = T_g2l_from_sample(ref)

        # Check inverse consistency
        I = np.eye(4)
        inv_err = fro_norm(T_g2l_ref @ T_l2g_ref - I)
        inv_err2 = fro_norm(T_inv(T_l2g_ref) - T_g2l_ref)
        print(f"    [geom] inv_err(T_g2l*T_l2g - I) = {inv_err:.3e}")
        print(f"    [geom] inv_err(T_inv(T_l2g) - T_g2l) = {inv_err2:.3e}")

        # Check T_k2ref(ref, ref) ~ I
        T_rr = T_k2ref(ref, ref)
        rr_err = fro_norm(T_rr - I)
        print(f"    [geom] err(T_k2ref(ref,ref) - I) = {rr_err:.3e}")

        # Check global composition consistency using another frame if possible
        if len(frames) >= 2:
            k = frames[1]
            T_l2g_k = T_l2g_from_sample(k)

            T_k2r = T_k2ref(ref, k)

            # random points in k-lidar
            pts_k = np.random.randn(10, 3).astype(np.float64)
            pts_ref_a = transform_points(T_k2r, pts_k)

            # via global: p_g = T_l2g_k p_k ; p_ref = T_g2l_ref p_g
            pts_g = transform_points(T_l2g_k, pts_k)
            pts_ref_b = transform_points(T_g2l_ref, pts_g)

            comp_err = float(np.max(np.linalg.norm(pts_ref_a - pts_ref_b, axis=1)))
            print(f"    [geom] max_err(compose vs direct T_k2ref) = {comp_err:.3e}")

            # yaw / vel rotation sanity (not a strict equality check, just ensures shapes/finite)
            yaw = np.random.uniform(-np.pi, np.pi, size=(10,))
            yaw_r = rotate_yaw(T_k2r, yaw)
            assert yaw_r.shape == yaw.shape
            assert np.isfinite(yaw_r).all()

            vxy = np.random.randn(10, 2).astype(np.float64)
            vxy_r = rotate_vel_xy(T_k2r, vxy)
            assert vxy_r.shape == vxy.shape
            assert np.isfinite(vxy_r).all()

            print("    [geom] yaw/vel rotation: OK (finite + correct shapes)")

        else:
            print("    [geom] only 1 frame in scene, skip cross-frame composition check")

    # -----------------------------
    # Writeback test (optional)
    # -----------------------------
    if args.out_json:
        print(f"[3] Writeback test -> {args.out_json}")

        # choose one sample
        s = samples[0]
        keys = sample_key_candidates(s)
        chosen_key = None
        for k in keys:
            if k is not None and len(str(k)) > 0:
                chosen_key = k
                break
        if chosen_key is None:
            raise RuntimeError("Cannot find a usable key for writeback (sample_token/sample_data_token/timestamp).")

        det = s.get("det") or {}
        boxes = det.get("boxes_3d") or []
        labels = det.get("labels_3d") or []
        scores = det.get("scores_3d") or []

        # Make a tiny payload: keep first 1-2 boxes (or empty if none)
        n_keep = min(2, len(boxes))
        payload = {
            "boxes_3d": boxes[:n_keep],
            "labels_3d": labels[:n_keep],
            "scores_3d": scores[:n_keep],
            # example extra fields:
            "attrs": ["unknown"] * n_keep,
            "track_ids": list(range(n_keep)),
        }

        repl_map = {chosen_key: payload}

        stats = write_refined_json(
            in_json_path=args.json,
            out_json_path=args.out_json,
            repl_map=repl_map,
            mode="add",              # write to det_refined, keep original det unchanged
            refined_key="det_refined",
            fields=None,             # copy all keys from payload
            indent=2,
        )

        print(f"    writeback stats: matched={stats.matched}, updated={stats.updated}, visited={stats.visited}")
        print("    Open the output JSON and verify one sample has `det_refined`.")

    print("\nAll Phase-0 tests completed.")


if __name__ == "__main__":
    main()
