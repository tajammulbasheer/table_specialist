#!/usr/bin/env bash
# Faithful baseline: vanilla (not fine-tuned) Llama-3.1-8B base on all four test sets, one at a time on GPU 0.
# Verifies the weights first and refuses to run if any checksum differs.
set -u
cd "$(dirname "$0")/.."

# Wait for any baseline already using GPU 0.
while pgrep -f "final_bird_instruct" > /dev/null; do sleep 15; done

.venv/bin/python - <<'EOF' > logs/verify_llama_base.log 2>&1
import hashlib, sys
from huggingface_hub import HfApi
rev = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
info = HfApi().model_info("meta-llama/Llama-3.1-8B", revision=rev, files_metadata=True)
bad = 0
for s in info.siblings:
    if not s.rfilename.endswith(".safetensors"):
        continue
    h = hashlib.sha256()
    with open(f"models/Llama-3.1-8B/{s.rfilename}", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    ok = h.hexdigest() == s.lfs.sha256
    bad += not ok
    print(s.rfilename, h.hexdigest(), "OK" if ok else "MISMATCH")
sys.exit(1 if bad else 0)
EOF
status=$?
echo "[$(date +%H:%M:%S)] base checksums exit=$status"
[ "$status" -eq 0 ] || exit 1

for spec in "WikiSQL final_wikisql_base" "spider final_spider_base" "WikiQA final_wikitq_base" "bird final_bird_base"; do
  set -- $spec
  echo "[$(date +%H:%M:%S)] start $2"
  scripts/run_baseline.sh Llama-3.1-8B "$1" 0 "$2" --api completions --max-tokens 128 > /dev/null
  status=$?
  echo "[$(date +%H:%M:%S)] end $2 exit=$status"
done
echo "[$(date +%H:%M:%S)] all done"
