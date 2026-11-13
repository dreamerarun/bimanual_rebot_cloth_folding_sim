#!/usr/bin/env python
"""Export dataset normalization statistics from a trained LeRobot checkpoint.

SmolVLA saves the normalization statistics it was trained with inside
``policy_preprocessor_step_5_normalizer_processor.safetensors`` (one tensor per
``<feature>.<stat>``).  ``scripts/08_eval_smolvla.py`` needs those statistics as
a ``meta/stats.json``-shaped file; this script recovers them from a checkpoint
so evaluation does not depend on keeping the training dataset around.

    python scripts/10_export_stats_from_ckpt.py \
        --ckpt /tmp/smolvla_run3 \
        --out outputs/smolvla_run3_stats.json

The committed ``outputs/smolvla_run3_stats.json`` was produced this way
(21,080 training frames / 62 episodes).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

from safetensors.torch import load_file

STATS_FILE = "policy_preprocessor_step_5_normalizer_processor.safetensors"


def export(ckpt: pathlib.Path) -> dict:
    src = ckpt / STATS_FILE
    if not src.is_file():
        raise SystemExit(f"not a SmolVLA checkpoint with preprocessor stats: {src}")
    tensors = load_file(str(src))
    stats: dict[str, dict[str, float | list[float]]] = collections.defaultdict(dict)
    for key, value in tensors.items():
        feature, field = key.rsplit(".", 1)
        t = value.detach().cpu().float()
        stats[feature][field] = float(t.item()) if t.numel() == 1 else t.reshape(-1).tolist()
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model directory")
    ap.add_argument("--out", required=True, help="destination stats.json")
    args = ap.parse_args()

    stats = export(pathlib.Path(args.ckpt))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=4) + "\n")
    frames = stats.get("task_index", {}).get("count")
    print(f"[stats] wrote {out} ({len(stats)} features, {frames} training frames)")


if __name__ == "__main__":
    main()
