#!/usr/bin/env bash
# Phase 3 gate: run every model and API setting on WikiSQL and Spider, one at a time on GPU 0.
# All settings are reported, so the choice cannot be made just by which lands closest to the paper.
# Instruct without the system message was already run: baseline_wikisql_instruct_n1000, baseline_spider_instruct_test.
set -u
cd "$(dirname "$0")/.."

run() {
  echo "[$(date +%H:%M:%S)] start $4"
  scripts/run_baseline.sh "$@" > /dev/null
  local status=$?
  echo "[$(date +%H:%M:%S)] end $4 exit=$status"
}

run Llama-3.1-8B-Instruct WikiSQL 0 gate_wikisql_instruct_system --repo-system-message
run Llama-3.1-8B-Instruct spider 0 gate_spider_instruct_system --repo-system-message

# Wait for the base model, which downloads separately.
while pgrep -f "hf download meta-llama/Llama-3.1-8B " > /dev/null; do sleep 20; done
if [ "$(ls models/Llama-3.1-8B/*.safetensors 2>/dev/null | wc -l)" -ne 4 ]; then
  echo "Base model is incomplete"; exit 1
fi

run Llama-3.1-8B WikiSQL 0 gate_wikisql_base_raw --api completions --max-tokens 128
run Llama-3.1-8B spider 0 gate_spider_base_raw --api completions --max-tokens 128
echo "[$(date +%H:%M:%S)] all done"
