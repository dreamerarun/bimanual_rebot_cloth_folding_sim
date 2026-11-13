"""run_episodes — orchestrate data collection episodes for the reBot fold task.

This module ties together RebotClothEnv + ControlPolicy + EpisodeRecorder and
implements Phase 4c (cloth randomization per episode). It is the main entry
point for collecting demonstrations.

Usage:
    from wam_fold.run_episodes import run_collection
    run_episodes = run_collection(episodes=10, out_dir="data/raw")
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import pickle
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from .config import SceneCfg, CollectCfg, FoldTaskCfg
from .rebot_env import RebotClothEnv, RebotRobotCfg
from .control_policy import (
    ControlPolicy, FoldState, EpisodeRecorder,
    compute_fold_error_cm, check_success, STATE_NAMES,
)


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------
def run_episode(
    env:        RebotClothEnv,
    policy:     ControlPolicy,
    episode_id: int,
    max_steps:  int   = 340,
    save_images: bool  = True,
    target_fold: float = 1.0,
    threshold_cm: float = 5.0,
) -> dict:
    """Run one episode and return the recorded data + metadata.

    Parameters
    ----------
    env : RebotClothEnv
        Initialised environment (already built).
    policy : ControlPolicy
        Freshly reset policy.
    episode_id : int
        Identifier for this episode.
    max_steps : int
        Maximum control steps before forced termination.
    save_images : bool
        Whether to capture camera frames per step.
    target_fold : float
        Target fold fraction for success metric.
    threshold_cm : float
        Fold error threshold (cm) for success.
    """
    # Reset
    env.reset(randomise=False)
    # let the cloth fall/settle before planning grasps from its mesh
    for _ in range(20):
        env.step(None)
    env._capture_cloth_reference()
    policy.reset(env)
    recorder = EpisodeRecorder(episode_id=episode_id, save_images=save_images)

    print(f"[episode {episode_id}] starting (max_steps={max_steps})")

    for step in range(max_steps):
        action = policy.get_action(env)
        env.step(action)

        # Compute fold error
        fold_err_cm = compute_fold_error_cm(env, target_fraction=target_fold)
        success = check_success(env, target_fraction=target_fold, threshold_cm=threshold_cm)

        # Record
        recorder.record(
            step=step, env=env, action=action,
            state_name=STATE_NAMES[policy.state],
            fold_error_cm=fold_err_cm,
            success=success,
        )

        if policy.is_done:
            # Allow a few settle steps after DONE
            for _ in range(30):
                env.step(None)
            break

    # Final measurement
    final_err = compute_fold_error_cm(env, target_fraction=target_fold)
    final_success = check_success(env, target_fraction=target_fold, threshold_cm=threshold_cm)

    data = recorder.finalize(
        success=final_success,
        fold_error_cm=final_err,
        config={
            "max_steps": max_steps,
            "target_fold": target_fold,
            "threshold_cm": threshold_cm,
            "control_hz": policy.control_hz,
        },
    )

    print(f"[episode {episode_id}] DONE: success={final_success}, "
          f"fold_error={final_err:.2f} cm, steps={len(data.steps)}")
    return data


# ---------------------------------------------------------------------------
# Multi-episode collection runner
# ---------------------------------------------------------------------------
def _load_meta(out_path: pathlib.Path) -> list[dict]:
    """Load existing metadata.json, tolerating missing/corrupt files."""
    meta_path = out_path / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return list(json.load(f))
        except Exception:
            return []
    return []


def _write_meta(out_path: pathlib.Path, meta: list[dict]):
    meta = sorted(meta, key=lambda m: m["episode_id"])
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_collection(
    episodes:      int   = 10,
    out_dir:        str   = "data/raw",
    max_steps:      int   = 340,
    save_images:    bool  = True,
    headless:       bool  = True,
    seed:           int   = 0,
    target_fold:    float = 1.0,
    threshold_cm:   float = 5.0,
    start_index:    int   = 0,
    keep_failed:    bool  = False,
    save_image_dirs: bool = False,
    cloth_jitter:   float = 0.0,
) -> list[dict]:
    """Run multiple episodes and save raw data to ``out_dir``.

    ``start_index`` names the first episode (useful when running one episode
    per fresh process to avoid cloth physics state carrying over).

    ``keep_failed`` controls whether unsuccessful episodes are written to
    disk; by default they are discarded so the dataset stays clean.

    ``save_image_dirs`` additionally dumps per-step ``.npy`` frames next to the
    pickle. The frames are already embedded in the pickle, so this is off by
    default (it roughly doubles disk usage).

    Returns a list of per-episode metadata dicts.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build env once, reuse across episodes
    cfg = SceneCfg()
    cfg.cloth.jitter_xy = float(cloth_jitter or 0.0)
    rcfg = RebotRobotCfg()
    env = RebotClothEnv(cfg=cfg, robot_cfg=rcfg, headless=headless,
                        cameras=save_images, seed=seed, cloth_readback=True)

    policy = ControlPolicy(control_hz=cfg.control_hz)

    all_meta = _load_meta(out_path)
    try:
        for ep in range(episodes):
            ep_id = start_index + ep
            data = run_episode(
                env=env, policy=policy, episode_id=ep_id,
                max_steps=max_steps, save_images=save_images,
                target_fold=target_fold, threshold_cm=threshold_cm,
            )

            if data.success or keep_failed:
                # Save raw episode data
                ep_path = out_path / f"episode_{ep_id:04d}.pkl"
                with open(ep_path, "wb") as f:
                    pickle.dump(data, f)

                if save_image_dirs and save_images and data.steps:
                    img_dir = out_path / f"episode_{ep_id:04d}_images"
                    img_dir.mkdir(exist_ok=True)
                    for rec in data.steps:
                        if rec.images:
                            for cam_name, img in rec.images.items():
                                img_file = img_dir / f"{cam_name}_{rec.step:04d}.npy"
                                np.save(img_file, img)

                meta = {
                    "episode_id":    ep_id,
                    "n_steps":       len(data.steps),
                    "success":       data.success,
                    "fold_error_cm":  data.fold_error_cm,
                    "duration_s":    round(data.duration_s, 2),
                    "filepath":       str(ep_path),
                }
                all_meta = [m for m in all_meta if m["episode_id"] != ep_id]
                all_meta.append(meta)
                _write_meta(out_path, all_meta)
                print(f"[collection] saved episode {ep_id} (success={data.success}) → {ep_path}")
            else:
                print(f"[collection] discarded failed episode {ep_id} "
                      f"(fold_error={data.fold_error_cm:.2f} cm)")

    finally:
        env.close()

    n_saved = len(all_meta)
    print(f"\n[collection] {n_saved} successful episodes in {out_path}")
    return all_meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="reBot cloth-fold data collection")
    ap.add_argument("--episodes",   type=int,   default=10)
    ap.add_argument("--out",        type=str,   default="data/raw")
    ap.add_argument("--max-steps",  type=int,   default=340)
    ap.add_argument("--no-images",  action="store_true")
    ap.add_argument("--gui",        action="store_true")
    ap.add_argument("--seed",       type=int,   default=0)
    ap.add_argument("--target-fold", type=float, default=1.0)
    ap.add_argument("--threshold",  type=float, default=5.0)
    ap.add_argument("--start-index", type=int,  default=0)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--image-dirs",  action="store_true")
    ap.add_argument("--cloth-jitter", type=float, default=0.0,
                    help="max |dx|,|dy| cloth translation at build (metres)")
    args = ap.parse_args()

    run_collection(
        episodes=args.episodes,
        out_dir=args.out,
        max_steps=args.max_steps,
        save_images=not args.no_images,
        headless=not args.gui,
        seed=args.seed,
        target_fold=args.target_fold,
        threshold_cm=args.threshold,
        start_index=args.start_index,
        keep_failed=args.keep_failed,
        save_image_dirs=args.image_dirs,
        cloth_jitter=args.cloth_jitter,
    )


if __name__ == "__main__":
    main()
