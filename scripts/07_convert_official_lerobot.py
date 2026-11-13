"""Convert the raw reBot fold pickles into an official LeRobot v2 dataset.

Usage:
    python scripts/07_convert_official_lerobot.py \
        --raw /mnt/ntfsdata/wam-cloth-fold/data/raw \
        --out /mnt/ntfsdata/wam-cloth-fold/data/lerobot_official \
        --repo-id arun/rebot-cloth-fold
"""

from __future__ import annotations

import argparse
import numbers
import pathlib
import pickle

import numpy as np

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "fold the cloth in half"

STATE_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6",
    "left_gripper",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6",
    "right_gripper",
]
ACTION_NAMES = [n.replace("joint", "cmd_joint").replace("gripper", "cmd_gripper") for n in STATE_NAMES]

CAMERAS = ("top", "front")
IMG_SHAPE = (256, 256, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/mnt/ntfsdata/wam-cloth-fold/data/raw")
    ap.add_argument("--out", default="/mnt/ntfsdata/wam-cloth-fold/data/lerobot_official")
    ap.add_argument("--repo-id", default="arun/rebot-cloth-fold")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    raw = pathlib.Path(args.raw)
    files = sorted(raw.glob("episode_*.pkl"))
    if not files:
        raise SystemExit(f"no episode_*.pkl in {raw}")
    print(f"[convert] {len(files)} raw episodes")

    out = pathlib.Path(args.out)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": ACTION_NAMES},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": IMG_SHAPE,
            "names": ["channels", "height", "width"],
        }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=out,
        robot_type="rebot_bimanual",
        use_videos=False,
        image_writer_processes=2,
        image_writer_threads=4,
    )

    kept = 0
    for f in files:
        with open(f, "rb") as fh:
            ep = pickle.load(fh)
        if not getattr(ep, "success", False):
            print(f"[convert] skip {f.name} (unsuccessful)")
            continue
        for rec in ep.steps:
            state = np.concatenate([
                np.asarray(rec.joint_q["left"], dtype=np.float32),
                np.array([rec.gripper["left"]], dtype=np.float32),
                np.asarray(rec.joint_q["right"], dtype=np.float32),
                np.array([rec.gripper["right"]], dtype=np.float32),
            ]).astype(np.float32)
            action = np.asarray(rec.action, dtype=np.float32)
            frame = {"observation.state": state, "action": action, "task": TASK}
            for cam in CAMERAS:
                img = rec.images.get(cam) if rec.images else None
                if img is None:
                    img = np.zeros(IMG_SHAPE, dtype=np.uint8)
                frame[f"observation.images.{cam}"] = img.astype(np.uint8)
            ds.add_frame(frame)
        ds.save_episode()
        kept += 1
        print(f"[convert] episode {kept}: {f.name} ({len(ep.steps)} steps)")

    try:
        ds.stop_image_writer()
    except Exception:
        pass

    print(f"[convert] saved {kept} episodes / {ds.meta.total_frames} frames to {out}")
    print(f"[convert] features: {list(ds.features)}")


if __name__ == "__main__":
    main()
