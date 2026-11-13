#!/usr/bin/env bash
# Launch a closed-loop SmolVLA eval episode in the Isaac Sim GUI.
#
#   bash scripts/run_gui_eval.sh [episode-index] [hold-seconds] [ckpt]
#
# HF_HUB_OFFLINE=1 is required: the policy loader otherwise tries to
# re-validate files on huggingface.co and hangs/crashes when the network
# is slow or unavailable (all weights are already in ~/.cache/huggingface).
set -euo pipefail
cd "$(dirname "$0")/.."

EP="${1:-1}"
HOLD="${2:-180}"
CKPT="${3:-${CKPT:-/tmp/smolvla_run3}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/leisaac/bin/python -u scripts/08_eval_smolvla.py \
  --ckpt "$CKPT" \
  --episode-index "$EP" \
  --max-steps 340 \
  --clean-ui \
  --gui \
  --save-frames \
  --hold-seconds "$HOLD"
