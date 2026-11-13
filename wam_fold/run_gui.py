#!/usr/bin/env python
"""Run one bimanual fold episode with GUI so you can watch the arms move.

Usage:
    env -u PYTHONPATH -u LD_LIBRARY_PATH \
      ~/miniconda3/envs/leisaac/bin/python -m wam_fold.run_gui

    # Or with custom parameters:
    env -u PYTHONPATH -u LD_LIBRARY_PATH \
      ~/miniconda3/envs/leisaac/bin/python -m wam_fold.run_gui --max-steps 200
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .config import SceneCfg, FoldTaskCfg
from .rebot_env import RebotClothEnv, RebotRobotCfg
from .control_policy import (
    ControlPolicy, FoldState, STATE_NAMES,
    compute_fold_error_cm, check_success,
)


def run_gui(max_steps: int = 420, randomise: bool = False,
            save_frames: bool = True, headless: bool = False,
            hold_seconds: float = 0.0):
    cfg  = SceneCfg()
    rcfg = RebotRobotCfg()

    # cloth_readback=True so we can measure fold success
    env = RebotClothEnv(
        cfg=cfg, robot_cfg=rcfg,
        headless=headless,
        cameras=True,
        cloth_readback=True,
    )

    policy = ControlPolicy(control_hz=cfg.control_hz)

    # State-change callback for console logging
    def on_state_change(old: FoldState, new: FoldState):
        print(f"  [FSM] {STATE_NAMES[old]} → {STATE_NAMES[new]}")

    policy.on_state_change = on_state_change

    print("=" * 60)
    print("  reBot Bimanual Cloth-Fold GUI Demo")
    print(f"  Max steps: {max_steps}  Control Hz: {cfg.control_hz}")
    print(f"  Physics Hz: {1.0/cfg.physics_dt:.0f}  Cloth: {cfg.cloth.size[0]}×{cfg.cloth.size[1]} m")
    print("=" * 60)

    # Reset and let the cloth settle before planning grasps from its mesh.
    env.reset(randomise=randomise)
    for _ in range(20):
        env.step(None)
    # re-capture the fold baseline now that the cloth has settled
    env._capture_cloth_reference()
    policy.reset(env)
    print(f"[gui] episode start — cloth randomised: {randomise}")
    print(f"[gui] grasp geometry: {policy._grasp_geom}")

    frame_count = 0
    t0 = time.time()

    for step in range(max_steps):
        action = policy.get_action(env)
        env.step(action)

        # Print progress every 20 steps
        if step % 20 == 0:
            # Get current EE positions for logging
            left_ee, _  = env.fk("left")
            right_ee, _ = env.fk("right")
            grip_l = env.get_gripper("left")
            grip_r = env.get_gripper("right")
            fold_err = compute_fold_error_cm(env)
            edge_gap = env.measure_edge_alignment()
            print(
                f"  step {step:3d} | state={STATE_NAMES[policy.state]:8s} "
                f"| L_ee=({left_ee[0]:+.3f},{left_ee[1]:+.3f},{left_ee[2]:+.3f}) "
                f"| R_ee=({right_ee[0]:+.3f},{right_ee[1]:+.3f},{right_ee[2]:+.3f}) "
                f"| grip=({grip_l:.1f},{grip_r:.1f}) "
                f"| fold_err={fold_err:.1f}cm | edge_gap={edge_gap*100:.1f}cm"
            )

        # Save a few camera frames for verification
        if save_frames and step % 40 == 0 and env.use_cameras:
            imgs = env.get_images()
            if imgs:
                import pathlib
                out = pathlib.Path("outputs")
                out.mkdir(exist_ok=True)
                for cam, img in imgs.items():
                    from PIL import Image
                    p = out / f"gui_{cam}_step{step:04d}.png"
                    Image.fromarray(np.ascontiguousarray(img)).save(str(p))
                frame_count += 1

        if policy.is_done:
            print(f"\n[gui] FSM reached DONE at step {step}")
            # Settle: let the cloth physics relax
            print("[gui] settling cloth physics (30 steps)...")
            for _ in range(30):
                env.step(None)
            break

    # Final measurement
    final_err = compute_fold_error_cm(env)
    final_ok  = check_success(env, threshold_cm=5.0)  # relaxed threshold for demo
    edge_gap  = env.measure_edge_alignment()
    elapsed   = time.time() - t0

    print()
    print("=" * 60)
    print(f"  Episode complete in {elapsed:.1f}s, {step+1} steps")
    print(f"  Final fold error: {final_err:.2f} cm")
    print(f"  Edge alignment gap: {edge_gap*100:.1f} cm (0 = near edge meets far edge)")
    if getattr(env, "cloth_readback", False):
        try:
            s = env.measure_edge_stats()
            if s:
                print(f"  Edge gap mean={s['gap_mean_cm']:.1f}cm "
                      f"std={s['gap_std_cm']:.1f}cm max={s['gap_max_cm']:.1f}cm")
                print("  edge profile (x: near->far) " +
                      " ".join(f"{x:+.2f}:{ny:.3f}->{fy:.3f}"
                               for x, ny, fy in s["profile"]))
                pts = env.get_cloth_points()
                near, far = env._cloth_edge_mask, env._cloth_far_mask
                for nm, m in (("near", near), ("far", far), ("mid", ~(near | far))):
                    if m.sum():
                        p = pts[m]
                        print(f"    {nm:4s} n={m.sum():3d} "
                              f"y[{p[:,1].min():+.3f},{p[:,1].mean():+.3f},{p[:,1].max():+.3f}] "
                              f"z[{p[:,2].min():+.3f},{p[:,2].mean():+.3f},{p[:,2].max():+.3f}]")
        except Exception:
            pass
    print(f"  Success (< 5 cm): {final_ok}")
    print(f"  Frames saved: {frame_count}")
    print("=" * 60)

    # Save final camera frames
    if save_frames and env.use_cameras:
        imgs = env.get_images()
        if imgs:
            import pathlib
            from PIL import Image
            out = pathlib.Path("outputs")
            out.mkdir(exist_ok=True)
            for cam, img in imgs.items():
                p = out / f"gui_{cam}_final.png"
                Image.fromarray(np.ascontiguousarray(img)).save(str(p))
                print(f"  saved {p}")

    # Keep the window on the folded result so it can be inspected
    if hold_seconds > 0:
        print(f"[gui] holding final result for {hold_seconds:.1f}s ...")
        for _ in range(int(hold_seconds * cfg.control_hz)):
            env.step(None, render=True)

    env.close()
    print("[gui] done — window closed")


def main():
    ap = argparse.ArgumentParser(description="reBot GUI fold demo")
    ap.add_argument("--max-steps", type=int, default=420)
    ap.add_argument("--randomise", action="store_true")
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    run_gui(
        max_steps=args.max_steps,
        randomise=args.randomise,
        save_frames=not args.no_frames,
        headless=args.headless,
        hold_seconds=0.0 if args.headless else 10.0,
    )


if __name__ == "__main__":
    main()
