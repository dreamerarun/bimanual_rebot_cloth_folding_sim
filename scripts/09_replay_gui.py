"""Replay a saved action sequence in the Isaac Sim GUI (smooth, no inference).

The policy runs on CPU, so live closed-loop eval pauses ~1.6 s at every
re-plan.  A recorded action sequence can instead be replayed directly at the
control rate: physics is deterministic, so the replay reproduces the fold
while the GUI plays back in real time with no stalls.

    # 1. record actions (one episode per fresh process)
    python scripts/08_eval_smolvla.py --ckpt ... \
        --episode-index 3 --save-actions /tmp/opencode/ep3_actions.npz

    # 2. replay in the GUI (optionally writing a frame sequence / mp4)
    python scripts/09_replay_gui.py /tmp/opencode/ep3_actions.npz \
        --frames-dir /tmp/opencode/ep3_frames --video /tmp/opencode/ep3.mp4 \
        --hold-seconds 120
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from wam_fold.config import SceneCfg
from wam_fold.rebot_env import RebotClothEnv, RebotRobotCfg
from wam_fold.control_policy import compute_fold_error_cm, check_success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actions", help=".npz written by 08_eval_smolvla.py --save-actions")
    ap.add_argument("--fps", type=float, default=20.0, help="control rate (Hz)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (1.0 = real time)")
    ap.add_argument("--frames-dir", default=None,
                    help="write one record-camera JPEG per control step here")
    ap.add_argument("--video", default=None,
                    help="encode the saved frames to this .mp4 with ffmpeg")
    ap.add_argument("--hold-seconds", type=float, default=60.0,
                    help="keep the window open after the replay")
    ap.add_argument("--no-clean-ui", action="store_true")
    ap.add_argument("--no-obs-cameras", action="store_true",
                    help="skip the policy observation cameras (not needed for replay)")
    ap.add_argument("--window-size", default=None,
                    help="GUI window size, e.g. 960x540")
    args = ap.parse_args()

    data = np.load(args.actions)
    actions = np.asarray(data["actions"], dtype=np.float32)
    ref_err = float(data["fold_error_cm"][0]) if "fold_error_cm" in data else float("nan")
    ref_gap = float(data["edge_gap_cm"][0]) if "edge_gap_cm" in data else float("nan")
    print(f"[replay] {len(actions)} actions from {args.actions} "
          f"(recorded fold_error={ref_err:.2f} cm, edge_gap={ref_gap:.1f} cm)")

    frames_dir = pathlib.Path(args.frames_dir) if args.frames_dir else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Same construction as the eval script → same initial cloth state.
    window_size = None
    if args.window_size:
        w, h = args.window_size.lower().split("x")
        window_size = (int(w), int(h))
    cfg = SceneCfg()
    env = RebotClothEnv(cfg=cfg, robot_cfg=RebotRobotCfg(), headless=False,
                        cameras=not args.no_obs_cameras, seed=0, cloth_readback=True,
                        hide_ui=not args.no_clean_ui, window_size=window_size)
    env.reset(randomise=False)
    for _ in range(20):
        env.step(None)
    env._capture_cloth_reference()

    dt = 1.0 / (args.fps * args.speed)
    t0 = time.time()
    from PIL import Image
    n_saved = 0

    def grab_frame():
        """Return the current record-camera RGB frame, or None if empty."""
        if env.record_cam is None:
            return None
        for _ in range(5):
            rgba = env.record_cam.get_rgba()
            if rgba is not None:
                img = np.asarray(rgba)[..., :3].astype(np.uint8)
                if img.size and img.shape[0] > 0 and img.shape[1] > 0:
                    return img
            env.step(None, render=True)
            time.sleep(0.02)
        return None

    for step, act in enumerate(actions):
        tick = time.time()
        env.step(act, render=True)
        if frames_dir is not None:
            img = grab_frame()
            if img is not None:
                Image.fromarray(np.ascontiguousarray(img)).save(
                    frames_dir / f"{n_saved:04d}.jpg", quality=92)
                n_saved += 1
        if step % 40 == 0 or step == len(actions) - 1:
            err = compute_fold_error_cm(env, target_fraction=1.0)
            print(f"[replay] step {step:3d} fold_err={err:6.2f} cm")
        sleep = dt - (time.time() - tick)
        if sleep > 0:
            time.sleep(sleep)

    final_err = compute_fold_error_cm(env, target_fraction=1.0)
    final_success = check_success(env, target_fraction=1.0, threshold_cm=5.0)
    edge = {}
    try:
        edge = env.measure_edge_stats() or {}
    except Exception:
        pass
    print(f"[replay] DONE success={final_success} fold_error={final_err:.2f} cm "
          f"duration={time.time()-t0:.1f}s")
    if edge:
        print(f"[replay] edge gap mean={edge['gap_mean_cm']:.1f} cm "
              f"std={edge['gap_std_cm']:.1f} max={edge['gap_max_cm']:.1f} cm")
    if ref_err == ref_err:
        print(f"[replay] recorded fold_error={ref_err:.2f} cm → "
              f"replay delta={abs(final_err - ref_err):.2f} cm")

    if args.video and frames_dir is not None:
        pattern = str(frames_dir / "%04d.jpg")
        cmd = ["ffmpeg", "-y", "-framerate", f"{args.fps:g}", "-i", pattern,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               args.video]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[replay] wrote video → {args.video}")
        except Exception as exc:
            print(f"[replay] ffmpeg failed: {exc}")

    if args.hold_seconds > 0:
        print(f"[replay] holding the window open for {args.hold_seconds:.0f}s ...")
        end = time.time() + args.hold_seconds
        while time.time() < end:
            env.step(None, render=True)

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
