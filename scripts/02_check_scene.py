"""Headless smoke test: build the scene, step it, dump a few frames.

    env -u PYTHONPATH -u LD_LIBRARY_PATH \
      ~/miniconda3/envs/leisaac/bin/python scripts/02_check_scene.py --headless
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from wam_fold.config import SceneCfg  # noqa: E402


def save_rgb(path, img):
    from PIL import Image

    Image.fromarray(img).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--out", default=str(PROJECT / "outputs" / "scene_check"))
    args = ap.parse_args()
    headless = not args.gui
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from wam_fold.env import WamClothEnv

    cfg = SceneCfg(headless=headless)
    t0 = time.time()
    env = WamClothEnv(cfg, headless=headless, cameras=True)
    print(f"[check] env built in {time.time()-t0:.1f}s")
    print("[check] dof names:", env.dof_names)
    print("[check] arm idx:", env.arm_idx, "finger idx:", env.finger_idx)

    pts0 = env.cloth.get_points()
    print(f"[check] cloth sim points: {pts0.shape}, z range {pts0[:,2].min():.4f}..{pts0[:,2].max():.4f}")

    obs = env.get_obs()
    for name, img in obs.get("images", {}).items():
        print(f"[check] camera {name}: {img.shape} mean={img.mean():.1f}")
        save_rgb(out / f"frame_0000_{name}.png", img)

    # hold the home pose for a moment so the cloth settles
    q = env.get_state()
    for i in range(args.steps):
        env.step(np.concatenate([q[:7], [0.0], q[8:15], [0.0]]), render=(i % 20 == 0))

    pts1 = env.cloth.get_points()
    print(f"[check] cloth after settle: z range {pts1[:,2].min():.4f}..{pts1[:,2].max():.4f}")
    print(f"[check] cloth moved by {np.abs(pts1-pts0).max():.4f} m (max)")

    obs = env.get_obs()
    for name, img in obs.get("images", {}).items():
        save_rgb(out / f"frame_{args.steps:04d}_{name}.png", img)

    print("[check] tcp left:", np.round(env.tcp_pose("left"), 3), "right:", np.round(env.tcp_pose("right"), 3))
    env.close()
    print(f"[check] images written to {out}")


if __name__ == "__main__":
    main()
