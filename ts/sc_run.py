"""Plan B1 driver: tune self-consistency on dev, then score the chosen setting on the test sets.

  python -m ts.sc_run dev --gpu 1 --port 8001
  python -m ts.sc_run test --temperature 0.7 --gpu 1 --port 8001
  TS_PROFILE=instruct python -m ts.sc_run test --temperature 1.0       # Instruct vanilla and Instruct TS round 3
  python -m ts.sc_run test --temperature 1.0 --benchmark bird --tag seq  # BIRD scored on one thread, new run IDs
"""
import argparse
import subprocess

from ts import loop
from ts.repo import ROOT

TS3 = "runs/train_tablespecialist_{profile}_seed{s}/round3/adapter_sql"
RTS3 = "runs/train_repairts_{profile}_seed{s}/round3/adapter_sql"


def models():
    if loop.PROFILE == "instruct":  # Plan step 4: B1 on the Instruct models from A2, then on Instruct repair-TS
        out = [("instruct_vanilla", None)]
        out += [(f"instruct_tablespecialist_seed{s}", ROOT / TS3.format(profile="instruct", s=s)) for s in (1, 2, 3)]
        return out + [(f"instruct_repairts_seed{s}", ROOT / RTS3.format(profile="instruct", s=s)) for s in (1, 2, 3)]
    out = [("vanilla", None)]
    for s in (1, 2, 3):
        out += [(f"tablespecialist_seed{s}", ROOT / TS3.format(profile="base", s=s)),
                (f"repairts_seed{s}", ROOT / RTS3.format(profile="base", s=s))]
    return out


def score(name, adapter, jobs, lane, temperature, samples, split, limit=None):
    served = loop.adapter_name("sql", adapter) if adapter else loop.MODEL_NAME
    missing = [j for j in jobs if not (ROOT / "results/baseline" / f"{j[2]}.json").exists()]
    if not missing:
        return
    from ts.repair_loop import gpu_slot
    with gpu_slot() as lane, loop.Server({served: str(adapter)} if adapter else {}, ROOT / f"logs/sc_server_{name}.log",
                                         lane):
        for benchmark, prompt, run_id in missing:
            subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "ts.eval.self_consistency", "--benchmark", benchmark,
                            "--model", served, "--run-id", run_id, "--api", loop.API,
                            *(["--repo-system-message"] if loop.SYSTEM else []),
                            "--max-tokens", str(loop.GENERATION_LIMITS["sql"]),
                            "--score-workers", "1" if benchmark == "bird" else "16",
                            "--prompt", prompt, "--samples", str(samples), "--temperature", str(temperature),
                            "--split", split, "--base-url", lane.url, *(["--limit", str(limit)] if limit else [])],
                           cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "test"])
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--smoke", type=int, help="score only this many cases, under smoke_ run IDs")
    ap.add_argument("--only", help="test stage: one model name, e.g. repairts_seed2")
    ap.add_argument("--benchmark", help="test stage: one test set, e.g. bird")
    ap.add_argument("--tag", default="", help="test stage: added to the run IDs, e.g. seq")
    args = ap.parse_args()
    lane = loop.Lane(args.gpu, args.gpu, args.port)
    if args.stage == "dev":
        adapter = ROOT / TS3.format(profile="base", s=1)
        for t in (0.5, 0.7, 1.0):
            tag = f"t{int(t * 10):02d}"
            prefix = "smoke" if args.smoke else "gate"
            jobs = [(b, "zero-shot", f"{prefix}_scdev_{tag}_{loop.BENCHMARKS[b]}") for b in ("WikiSQL", "spider", "WikiQA")]
            score("dev", adapter, jobs, lane, t, args.samples, "dev", args.smoke)
            if args.smoke:
                break
    else:
        tag = f"t{int(args.temperature * 10):02d}{args.tag}"
        prefix = "smoke" if args.smoke else "final"
        for name, adapter in models():
            if args.only and name != args.only:
                continue
            done = adapter is None or any((adapter.parent / f).exists() for f in ("summary.json", "result.json"))
            if not done:  # the round is still running; its adapter may be half-written
                print(f"skipping {name}: {adapter.parent} has not finished", flush=True)
                continue
            jobs = [(b, p, f"{prefix}_sc{tag}_{name}_{loop.BENCHMARKS[b]}_{p.replace('-', '')}")
                    for b in loop.BENCHMARKS for p in loop.EVAL_PROMPTS
                    if not args.benchmark or loop.BENCHMARKS[b] == args.benchmark]
            score(name, adapter, jobs, lane, args.temperature, args.samples, "test", args.smoke)


if __name__ == "__main__":
    main()
