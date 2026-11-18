#!/usr/bin/env bash
# One arm of the core study on one GPU: 3 seeds, 3 rounds, 3,000 tables per round, with the settings in configs/base.yaml.
# Safe to run again after any stop: each finished step is skipped. Stops at the first failure.
#   scripts/run_study.sh table_specialist 0 8000
#   scripts/run_study.sh no_validation 1 8001
set -u
cd "$(dirname "$0")/.."
set -a; . configs/sandbox.env; . configs/runtime.env; set +a
ARM="$1"; GPU="$2"; PORT="$3"
for SEED in 1 2 3; do
  echo "[$(date +%H:%M:%S) UTC] start $ARM seed $SEED on GPU $GPU"
  .venv/bin/python -m ts.loop --arm "$ARM" --seed "$SEED" --serve-gpu "$GPU" --train-gpu "$GPU" --port "$PORT" \
    >> "logs/study_${ARM}_seed${SEED}.log" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S) UTC] end $ARM seed $SEED exit=$status"
  grep -a '^ROUND' "logs/study_${ARM}_seed${SEED}.log" | cut -c1-400
  [ "$status" -eq 0 ] || exit "$status"
done
echo "[$(date +%H:%M:%S) UTC] $ARM finished"
