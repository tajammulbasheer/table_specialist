#!/usr/bin/env bash
# Serve a model on GPU 0, evaluate it, then stop the server.
# Usage: scripts/run_baseline.sh <model-dir-under-models> <benchmark> <limit, or 0 for all> <run-id> [run_baseline args]
set -u
cd "$(dirname "$0")/.."
MODEL=$1; BENCHMARK=$2; LIMIT=$3; RUN_ID=$4
shift 4
PORT=8000
mkdir -p logs

set -a; source configs/runtime.env; set +a

# Bound to localhost only. Settings match scripts/smoke_vllm.py.
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve "models/$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization 0.92 --seed 0 > "logs/serve_$RUN_ID.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null' EXIT

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
  kill -0 "$SERVER" 2>/dev/null || { echo "Server exited during startup; see logs/serve_$RUN_ID.log"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "Server not ready after 10 minutes"; exit 1; }

LIMIT_ARG=()
[ "$LIMIT" -gt 0 ] && LIMIT_ARG=(--limit "$LIMIT")
.venv/bin/python -m ts.eval.run_baseline --benchmark "$BENCHMARK" "${LIMIT_ARG[@]}" --model "$MODEL" --run-id "$RUN_ID" "$@" \
  2> "logs/baseline_$RUN_ID.log"
