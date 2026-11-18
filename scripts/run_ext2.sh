#!/usr/bin/env bash
# Extension 2 round-1 pilot on one GPU, for the seeds given. Safe to run again after any stop: finished steps are skipped.
# Only one seed validates at a time across both GPUs (runs/.spark_validation.lock).
#   scripts/run_ext2.sh 0 8000 1 3
#   scripts/run_ext2.sh 1 8001 2
set -u
cd "$(dirname "$0")/.."
set -a; . configs/sandbox.env; . configs/runtime.env; set +a
GPU="$1"; PORT="$2"; shift 2
for SEED in "$@"; do
  echo "[$(date +%H:%M:%S) UTC] start ext2 seed $SEED on GPU $GPU"
  .venv/bin/python -m ts.ext2 --seed "$SEED" --gpu "$GPU" --port "$PORT" >> "logs/ext2_seed${SEED}.log" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S) UTC] end ext2 seed $SEED exit=$status"
  grep -a -E '^(LABELS|EXT2)' "logs/ext2_seed${SEED}.log" | cut -c1-600
  [ "$status" -eq 0 ] || exit "$status"
done
echo "[$(date +%H:%M:%S) UTC] ext2 on GPU $GPU finished"
