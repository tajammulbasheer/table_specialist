"""Tune extension 2's DPO settings on the dev sets (ts/bench/dev.py), never on the test sets.

Each variant in configs/base.yaml extension2.tuning trains DPO from the tuning seed's bestof4 adapter on that seed's
pairs, and is scored zero-shot on the dev sets. Reference models are scored the same way: Table-Specialist round 1,
bestof4, and the first DPO settings. Each GPU runs the items it is given; finished items are skipped.

  python -m ts.ext2_tune --gpu 0 --port 8000 --items ref_ts ref_bestof4 ref_dpo v1_gentle
"""
import argparse
import json
import subprocess
from pathlib import Path

import yaml

from ts import ext2, loop
from ts.repo import ROOT
from ts.tracking import check_run_id

NEGATIVE_FILTERS = {None: lambda pair: True, "rows_differ": lambda pair: pair["rejected_status"] == "ok"}


def reference_adapters(seed):
    return {"ref_ts": ROOT / f"runs/train_tablespecialist_base_seed{seed}/round1/adapter_sql",
            "ref_bestof4": ROOT / f"runs/train_bestof4_base_seed{seed}/round1/adapter_sql",
            "ref_dpo": ROOT / f"runs/train_dpo_base_seed{seed}/round1/adapter_sql"}


def dev_run_id(item, seed, benchmark):
    name = item.replace("_", "")
    return check_run_id(f"train_dpotune{name}_base_seed{seed}_dev_{loop.BENCHMARKS[benchmark]}_zeroshot")


def evaluate_dev(item, adapter, seed, benchmarks, log_dir, lane):
    jobs = [(b, dev_run_id(item, seed, b)) for b in benchmarks]
    results = ROOT / "results/baseline"
    missing = [job for job in jobs if not (results / f"{job[1]}.json").exists()]
    if missing:
        name = loop.adapter_name("sql", adapter)
        with loop.Server({name: str(adapter)}, log_dir / "server_dev.log", lane):
            for benchmark, run_id in missing:
                with open(log_dir / "dev_eval.log", "a") as log:
                    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "ts.eval.run_baseline", "--benchmark",
                                    benchmark, "--model", name, "--run-id", run_id, "--api", "completions",
                                    "--max-tokens", str(loop.GENERATION_LIMITS["sql"]), "--prompt", "zero-shot",
                                    "--base-url", lane.url, "--split", "dev"],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    scores = {b: json.loads((results / f"{run_id}.json").read_text())["repo_faithful_acc"] for b, run_id in jobs}
    return {"scores": scores, "mean": round(sum(scores.values()) / len(scores), 4)}


def run_item(item, settings, lane):
    seed, benchmarks = settings["seed"], settings["dev_benchmarks"]
    work = ROOT / "runs" / f"train_dpotune{item.replace('_', '')}_base_seed{seed}" / "round1"
    work.mkdir(parents=True, exist_ok=True)
    if item in reference_adapters(seed):
        adapter, train = reference_adapters(seed)[item], None
    else:
        overrides = dict(settings["variants"][item])
        keep = NEGATIVE_FILTERS[overrides.pop("negatives", None)]
        pairs_path = work / "dpo_pairs.jsonl"
        if not pairs_path.exists():
            pairs = loop.read_jsonl(ROOT / f"runs/train_bestof4_base_seed{seed}/round1/dpo_pairs.jsonl")
            loop.write_jsonl(pairs_path, [p for p in pairs if keep(p)])
        adapter = work / "adapter_sql"
        train = ext2.train_dpo(pairs_path, reference_adapters(seed)["ref_bestof4"], adapter, seed, work / "train_dpo.log",
                               lane.serve_gpu, overrides=overrides)
    dev = evaluate_dev(item, adapter, seed, benchmarks, work, lane)
    record = {"item": item, "settings": settings["variants"].get(item), "dev": dev,
              "train": train and {k: train[k] for k in ("pairs", "global_steps", "train_loss", "last_reward_accuracy",
                                                         "last_reward_margin", "seconds")}}
    (work / "dev_summary.json").write_text(json.dumps(record, indent=1))
    print("TUNE " + json.dumps(record), flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--items", nargs="+", required=True, help="ref_ts, ref_bestof4, ref_dpo, or variant names")
    args = ap.parse_args()
    settings = yaml.safe_load((ROOT / "configs/base.yaml").read_text())["extension2"]["tuning"]
    unknown = [i for i in args.items if i not in settings["variants"] and i not in reference_adapters(settings["seed"])]
    if unknown:
        raise SystemExit(f"Unknown items: {unknown}")
    lane = loop.Lane(args.gpu, args.gpu, args.port)
    for item in args.items:
        run_item(item, settings, lane)


if __name__ == "__main__":
    main()
