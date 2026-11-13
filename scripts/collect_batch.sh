#!/usr/bin/env bash
# Collect one fresh-process episode at a time so cloth physics state never
# carries over. Failed episodes are discarded automatically by run_episodes.
set -u
cd /home/arun/rebot_cloth_fold || exit 1
OUT=${OUT:-/mnt/ntfsdata/wam-cloth-fold/data/raw}
START=${1:-5}
COUNT=${2:-10}
JITTER=${3:-0.0}
for ((i=START; i<START+COUNT; i++)); do
  echo "=== attempt $i $(date +%H:%M:%S) jitter=$JITTER ==="
  env -u PYTHONPATH -u LD_LIBRARY_PATH timeout 400 \
    ~/miniconda3/envs/leisaac/bin/python -u -m wam_fold.run_episodes \
    --episodes 1 --start-index "$i" --seed "$i" --cloth-jitter "$JITTER" --out "$OUT" 2>&1 \
    | grep -E "DONE:|collection|Cloth\] build|policy\] WARNING|readback|Traceback|Error:" | tail -12
done
echo "=== batch finished $(date +%H:%M:%S) ==="
