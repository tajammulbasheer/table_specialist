"""Plan A1 (docs/PLAN_NEXT.md): Llama-3.1-8B-Instruct vanilla on the 4 test sets, through the repo's chat call with its
system message, zero-shot and few-shot. A finished result file is not repeated."""
import subprocess
import sys

from ts import loop
from ts.repo import ROOT

GPU, PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 8000
lane = loop.Lane(GPU, GPU, PORT)
jobs = [(b, p) for b in ("WikiSQL", "spider", "WikiQA", "bird") for p in ("zero-shot", "few-shot")]
with loop.Server({}, ROOT / "logs/a1_instruct_server.log", lane, model_path="models/Llama-3.1-8B-Instruct",
                 model_name="Llama-3.1-8B-Instruct"):
    for benchmark, prompt in jobs:
        run_id = f"gate_instruct_{loop.BENCHMARKS[benchmark]}_{prompt.replace('-', '')}"
        if (ROOT / "results/baseline" / f"{run_id}.json").exists():
            continue
        subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "ts.eval.run_baseline", "--benchmark", benchmark,
                        "--model", "Llama-3.1-8B-Instruct", "--run-id", run_id, "--api", "chat",
                        "--repo-system-message", "--max-tokens", "256", "--prompt", prompt, "--base-url", lane.url],
                       cwd=ROOT, check=True)
