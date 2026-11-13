"""Closed-loop evaluation of a trained SmolVLA checkpoint on the reBot fold task.

Run ONE episode per fresh process (cloth physics state does not reset):

    python scripts/08_eval_smolvla.py \
        --ckpt /mnt/.../train/smolvla_run3/checkpoints/last/pretrained_model \
        --episode-index 0 --max-steps 340

Add ``--save-actions out.npz`` to record the exact action sequence; it can then
be replayed smoothly in the GUI with ``scripts/09_replay_gui.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# All policy weights live in the local HF cache.  Force offline mode before
# lerobot is imported: otherwise the loader re-validates against
# huggingface.co, and a slow/unreachable network hangs or crashes startup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from wam_fold.config import SceneCfg
from wam_fold.rebot_env import RebotClothEnv, RebotRobotCfg
from wam_fold.control_policy import compute_fold_error_cm, check_success

TASK = "fold the cloth in half"
CAMS = ("top", "front")


def to_policy_obs(env) -> dict:
    images = env.get_images() or {}
    obs = {
        "observation.state": torch.from_numpy(
            np.concatenate([
                np.asarray(env.get_arm_q("left"), dtype=np.float32),
                np.array([env.get_gripper("left")], dtype=np.float32),
                np.asarray(env.get_arm_q("right"), dtype=np.float32),
                np.array([env.get_gripper("right")], dtype=np.float32),
            ])
        ),
        "task": TASK,
    }
    for cam in CAMS:
        img = images.get(cam)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1).contiguous() / 255.0
        obs[f"observation.images.{cam}"] = t
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", default="/mnt/ntfsdata/wam-cloth-fold/data/lerobot_official/meta/stats.json")
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=340)
    ap.add_argument("--out", default="/mnt/ntfsdata/wam-cloth-fold/eval")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
    ap.add_argument("--clean-ui", action="store_true",
                    help="hide the Isaac Sim UI panels (viewport only, for recording)")
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="keep the window open this long after the episode")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override the policy's open-loop chunk execution length")
    ap.add_argument("--save-actions", default=None,
                    help="write the executed action sequence to this .npz file")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- env (starts Isaac Sim) ------------------------------------------
    cfg = SceneCfg()
    env = RebotClothEnv(cfg=cfg, robot_cfg=RebotRobotCfg(), headless=not args.gui,
                        cameras=True, seed=0, cloth_readback=True,
                        hide_ui=(True if args.clean_ui else None))
    env.reset(randomise=False)
    for _ in range(20):
        env.step(None)
    env._capture_cloth_reference()
    if args.gui:
        env.save_record_preview(out / f"eval_ep{args.episode_index}_record_start.png")

    # --- policy (load after the Sim app is up; run on CPU because Isaac Sim
    # already occupies nearly all of the 6 GB GPU) ----------------------
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    stats_path = pathlib.Path(args.stats) if args.stats else None
    if stats_path is None:
        raise SystemExit("--stats (LeRobot meta/stats.json) is required for CPU eval")
    with open(stats_path) as f:
        dataset_stats = json.load(f)

    pcfg = PreTrainedConfig.from_pretrained(args.ckpt)
    pcfg.device = "cpu"
    pcfg.pretrained_path = pathlib.Path(args.ckpt)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt, config=pcfg)
    if args.n_action_steps is not None:
        policy.config.n_action_steps = int(args.n_action_steps)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=None, dataset_stats=dataset_stats
    )
    print(f"[eval] loaded policy from {args.ckpt} (device=cpu)")

    t0 = time.time()
    fold_err = compute_fold_error_cm(env, target_fraction=1.0)
    success = False
    actions = []
    for step in range(args.max_steps):
        obs = to_policy_obs(env)
        batch = preprocessor(obs)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocessor(action)
        act = action.squeeze(0).to("cpu").numpy().astype(np.float32)
        actions.append(act.copy())
        env.step(act)

        fold_err = compute_fold_error_cm(env, target_fraction=1.0)
        success = check_success(env, target_fraction=1.0, threshold_cm=5.0)
        if step % 40 == 0 or step == args.max_steps - 1:
            try:
                z = float(env.fk("left")[0][2])
                g = float(env.get_gripper("left"))
            except Exception:
                z, g = float("nan"), float("nan")
            print(f"[eval] step {step:3d} fold_err={fold_err:6.2f} cm "
                  f"action[6]={act[6]:.2f} action[13]={act[13]:.2f} "
                  f"tcp_z={z:.3f} grip={g:.2f}")
        if args.save_frames and step in (0, 100, 200, args.max_steps - 1):
            imgs = env.get_images()
            for cam in CAMS:
                np.save(out / f"eval_ep{args.episode_index}_{cam}_{step:04d}.npy", imgs[cam])
                if args.gui:
                    from PIL import Image
                    Image.fromarray(np.ascontiguousarray(imgs[cam])).save(
                        out / f"eval_ep{args.episode_index}_{cam}_{step:04d}.png")

    final_err = compute_fold_error_cm(env, target_fraction=1.0)
    final_success = check_success(env, target_fraction=1.0, threshold_cm=5.0)
    edge = {}
    try:
        edge = env.measure_edge_stats() or {}
    except Exception:
        pass

    print(f"[eval] DONE episode={args.episode_index} success={final_success} "
          f"fold_error={final_err:.2f} cm duration={time.time()-t0:.1f}s")
    if edge:
        print(f"[eval] edge gap mean={edge['gap_mean_cm']:.1f} cm "
              f"std={edge['gap_std_cm']:.1f} max={edge['gap_max_cm']:.1f} cm")

    if args.save_actions:
        path = pathlib.Path(args.save_actions)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            actions=np.asarray(actions, dtype=np.float32),
            fold_error_cm=np.array([final_err], dtype=np.float32),
            success=np.array([bool(final_success)]),
            edge_gap_cm=np.array([edge.get("gap_mean_cm", float("nan"))], dtype=np.float32),
            episode_index=np.array([args.episode_index]),
        )
        print(f"[eval] saved {len(actions)} actions → {path}")

    imgs = env.get_images()
    for cam in CAMS:
        np.save(out / f"eval_ep{args.episode_index}_{cam}_final.npy", imgs[cam])
        if args.gui:
            from PIL import Image
            Image.fromarray(np.ascontiguousarray(imgs[cam])).save(
                out / f"eval_ep{args.episode_index}_{cam}_final.png")
    if args.gui:
        env.save_record_preview(out / f"eval_ep{args.episode_index}_record_final.png")

    if args.hold_seconds > 0:
        print(f"[eval] holding the window open for {args.hold_seconds:.0f}s ...")
        end = time.time() + args.hold_seconds
        while time.time() < end:
            env.step(None, render=True)

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
