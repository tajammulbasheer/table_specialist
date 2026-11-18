#!/usr/bin/env bash
# Diagnostic: base on all four test sets with the repo's few-shot prompt from nl2sql_generation_v6.py.
# The paper's Appendix A shows this prompt but does not say whether evaluation used it, so this only tests whether
# the prompt explains the gap to the paper. It does not choose a setting by itself.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "scripts/[s]moke_vllm.py" > /dev/null; do sleep 15; done
export MAX_MODEL_LEN=8192  # the prompt's fixed text alone is about 2,150 tokens
for spec in "WikiSQL gate_wikisql_base_fewshot" "spider gate_spider_base_fewshot" "WikiQA gate_wikitq_base_fewshot" "bird gate_bird_base_fewshot"; do
  set -- $spec
  echo "[$(date +%H:%M:%S)] start $2"
  scripts/run_baseline.sh Llama-3.1-8B "$1" 0 "$2" --api completions --max-tokens 128 --prompt few-shot > /dev/null
  status=$?
  echo "[$(date +%H:%M:%S)] end $2 exit=$status"
done
echo "[$(date +%H:%M:%S)] all done"
