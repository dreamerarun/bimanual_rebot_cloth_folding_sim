#!/usr/bin/env python
"""Fetch the Seeed reBot Arm T701 HT (B601) open-source assets into the project.

Downloads (git clone) if missing:
  - https://github.com/Seeed-Projects/reBot-Isaacsim.git   -> assets/rebot_t701/

Run (headless, from the project root):
    env -u PYTHONPATH -u LD_LIBRARY_PATH \
      ~/miniconda3/envs/leisaac/bin/python scripts/fetch_assets.py
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

PROJECT = pathlib.Path(__file__).resolve().parents[1]

CLONES = [
    (
        "https://github.com/Seeed-Projects/reBot-Isaacsim.git",
        PROJECT / "assets" / "rebot_t701",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="clone even if the dir exists")
    args = ap.parse_args()

    for url, dest in CLONES:
        dest = pathlib.Path(dest)
        if dest.is_dir() and not args.yes and any(dest.iterdir()):
            print(f"[fetch] already present, skipping: {dest}")
            continue
        if dest.is_dir() and not args.yes:
            shutil.rmtree(dest)
        print(f"[fetch] cloning {url} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    print("[fetch] done")


if __name__ == "__main__":
    main()
