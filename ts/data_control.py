"""Same-data control for repair-TS (for the final report): does repair-TS round 1 win only because it trains on more pairs?

For a seed s, the control trains a SQL adapter from vanilla, exactly as TS round 1 does, on
  TS round 1's kept pairs of seed s  +  N more kept pairs, N = the number of pairs repair-TS round 1 recovered for seed s.
The N extra pairs are taken at random (Random(s)) from TS round 1 of another seed (donor = s % 3 + 1), from tables that
seed s's round 1 never used. They were written by the same vanilla model and passed the same Algorithm 3 check, so the
control differs from TS only in the amount of data, and from repair-TS only in where the extra pairs come from.
Nothing is generated or validated again: only training (GPU) and the usual test scoring. Source runs are only read.

  TS_PROFILE=base python -m ts.data_control --seeds 1 2 3
"""
import argparse
import json
import random

from ts import loop
from ts.repair_loop import gpu_slot
from ts.repo import ROOT
from ts.tracking import log_run

PAIR_KEYS = ("table_id", "question", "sql", "scala")


def build(seed, profile):
    donor = seed % 3 + 1
    own = loop.read_jsonl(ROOT / f"runs/train_tablespecialist_{profile}_seed{seed}/round1/validated.jsonl")
    donor_rows = loop.read_jsonl(ROOT / f"runs/train_tablespecialist_{profile}_seed{donor}/round1/validated.jsonl")
    recovered = json.loads((ROOT / f"runs/train_repairts_{profile}_seed{seed}/round1/result.json").read_text())["recovered"]
    own_tables = {r["table_id"] for r in own}
    kept = [{k: r[k] for k in PAIR_KEYS} for r in own if r["keep"]]
    pool = [{k: r[k] for k in PAIR_KEYS} for r in donor_rows if r["keep"] and r["table_id"] not in own_tables]
    if len(pool) < recovered:
        raise SystemExit(f"seed {seed}: donor seed {donor} has {len(pool)} usable pairs, {recovered} needed")
    extra = random.Random(seed).sample(pool, recovered)
    return kept, extra, donor


def run(seed, eval_limit=None, sft_max_steps=None, tag=""):
    profile = loop.PROFILE
    run_id = f"train_tsdata{tag}_{profile}_seed{seed}"
    out = ROOT / "runs" / run_id / "round1"
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    kept, extra, donor = build(seed, profile)
    loop.write_jsonl(out / "extra_pairs.jsonl", extra)
    sql_examples, _ = loop.step_examples(out / "train_sql.jsonl", out / "train_scala.jsonl", kept + extra)
    with gpu_slot() as lane:
        train = loop.train_adapter(out / "train_sql.jsonl", out / "adapter_sql", seed, out / "train_sql.log", lane,
                                   max_steps=sft_max_steps)
    with gpu_slot() as lane:
        scores = loop.step_evaluate(run_id, 1, out / "adapter_sql", out, lane=lane, limit=eval_limit)
    result = {"seed": seed, "profile": profile, "donor_seed": donor, "kept_ts": len(kept), "extra": len(extra),
              "sft_examples": len(sql_examples), "train": train, "test_scores": scores}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    if not tag:
        log_run(f"{run_id}_round1", "extension3", {"variant": "tsdata_control", "seed": seed, "profile": profile},
                result)
    print("TSDATA " + json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tag", default="", help="smoke tests: a run-name suffix such as smoke")
    ap.add_argument("--eval-limit", type=int)
    ap.add_argument("--sft-max-steps", type=int)
    args = ap.parse_args()
    for seed in args.seeds:  # one seed at a time: at most one GPU is taken, the other stays free for repair-TS
        run(seed, args.eval_limit, args.sft_max_steps, args.tag)


if __name__ == "__main__":
    main()
