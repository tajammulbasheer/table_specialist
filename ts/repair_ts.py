"""Extension 3, full run: Table-Specialist round 1 plus repaired pairs (ts/repair.py, Instruct repairer).

For a seed: repair every pair TS round 1 rejected, then train a SQL adapter from vanilla on TS's kept pairs plus the
recovered pairs (a recovered pair's SQL is the SQL that passed Algorithm 3), and score it on the 4 test sets with
both prompts. Only the SQL adapter is trained, as for bestof4: round 1 evaluation uses the SQL adapter only.
TS's run directory is only read.

  python -m ts.repair_ts --seed 2 --gpu 1 --port 8001
"""
import argparse
import json
from contextlib import nullcontext

from ts import loop, repair
from ts.repo import ROOT
from ts.tracking import log_run


def training_pairs(validated, repaired):
    kept = [{k: r[k] for k in ("table_id", "question", "sql", "scala")} for r in validated if r["keep"]]
    recovered = [{"table_id": r["table_id"], "question": r["question"], "sql": r["repaired_sql"],
                  "scala": r["repaired_scala"]} for r in repaired if r["keep"]]
    return kept, recovered


def run(seed, lane, tag="", limit=None, sft_max_steps=None, eval_limit=None, slot=None, source_tag=""):
    """slot, if given, is ts.repair_loop.gpu_slot: each GPU step takes a free GPU (lane is then unused).
    TS_PROFILE picks the model: the source is that profile's TS round 1 (source_tag: a smoke TS run's suffix)."""
    source = ROOT / f"runs/train_tablespecialist_{loop.PROFILE}_seed{seed}{source_tag}/round1"
    run_id = f"train_repairts{tag}_{loop.PROFILE}_seed{seed}"
    out = ROOT / "runs" / run_id / "round1"
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    repair_summary = repair.run(source, out, lane, seed, limit=limit, repairer="instruct", gpu_slot=slot)
    kept, recovered = training_pairs(loop.read_jsonl(source / "validated.jsonl"),
                                     loop.read_jsonl(out / "repaired.jsonl"))
    sql_examples, _ = loop.step_examples(out / "train_sql.jsonl", out / "train_scala.jsonl", kept + recovered)
    with (slot() if slot else nullcontext(lane)) as held:
        train = loop.train_adapter(out / "train_sql.jsonl", out / "adapter_sql", seed, out / "train_sql.log",
                                   loop.Lane(held.serve_gpu, held.serve_gpu, held.port), max_steps=sft_max_steps)
    with (slot() if slot else nullcontext(lane)) as held:
        scores = loop.step_evaluate(run_id, 1, out / "adapter_sql", out, lane=held, limit=eval_limit)
    result = {"seed": seed, "repair": repair_summary, "kept_ts": len(kept), "recovered": len(recovered),
              "sft_examples": len(sql_examples), "train": train, "test_scores": scores}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    if not tag:
        log_run(f"{run_id}_round1", "extension3", {"variant": "repairts", "seed": seed}, result)
    print("REPAIRTS " + json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", default="", help="smoke tests: a run-name suffix such as smoke")
    ap.add_argument("--limit", type=int, help="smoke tests only: repair the first N rejected pairs")
    ap.add_argument("--sft-max-steps", type=int)
    ap.add_argument("--eval-limit", type=int)
    args = ap.parse_args()
    run(args.seed, loop.Lane(args.gpu, args.gpu, args.port), args.tag, args.limit, args.sft_max_steps, args.eval_limit)


if __name__ == "__main__":
    main()
