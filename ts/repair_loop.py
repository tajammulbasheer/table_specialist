"""Extension 3, rounds 2-3: the Table-Specialist loop (paper Algorithm 1) with repair of rejected pairs in every round.

Round 1 is ts/repair_ts.py's run (its directory gains only a Scala adapter, trained from vanilla on the Scala of the
same kept + recovered pairs). Each later round, as in ts/loop.py: the vanilla model writes questions on the round's
tables (the same tables as TS's round), the previous round's SQL and Scala adapters write code, Algorithm 3 keeps
agreeing pairs, the Instruct model repairs the rejected ones (ts/repair.py), two adapters train from vanilla on kept +
recovered pairs, and the SQL adapter is scored on the 4 test sets with both prompts.

Scheduling, for 3 seeds on 2 GPUs: each seed is its own process. A step that needs a GPU takes whichever GPU is free
(gpu_slot); Spark validation, the CPU bottleneck, takes the one validation slot (ext2.validation_slot). The SQL and
Scala adapters train at the same time when both GPUs are free, and evaluation starts as soon as the SQL adapter is done.

  python -m ts.repair_loop --seed 1
"""
import argparse
import fcntl
import json
import threading
import time
from contextlib import contextmanager, nullcontext

import yaml

from ts import ext2, loop, repair, repair_ts
from ts.llm.client import LLMClient
from ts.repair_ts import training_pairs
from ts.repo import ROOT
from ts.tracking import log_run

GPUS = (0, 1)
GPU_LOCK = ROOT / "runs" / ".gpu{}.lock"


@contextmanager
def gpu_slot(gpus=GPUS, wait_s=20):
    """Hold one free GPU for one step. The lock is released when the process exits, even if it crashes."""
    while True:
        for gpu in gpus:
            lock = open(str(GPU_LOCK).format(gpu), "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            print(f"[{time.strftime('%H:%M:%S')} UTC] GPU {gpu} taken", flush=True)
            try:
                yield loop.Lane(gpu, gpu, 8000 + gpu)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
                print(f"[{time.strftime('%H:%M:%S')} UTC] GPU {gpu} released", flush=True)
            return
        time.sleep(wait_s)


def in_parallel(*jobs):
    """Run jobs in threads; re-raise the first error."""
    results, errors = [None] * len(jobs), []

    def call(i, job):
        try:
            results[i] = job()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=call, args=(i, job)) for i, job in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def round1_adapters(run_dir, seed, sft_max_steps=None, slot=gpu_slot):
    """Round 1's SQL adapter exists; its Scala adapter is added beside it (new files only)."""
    r1 = run_dir / "round1"
    for name in ("result.json", "adapter_sql/train_summary.json", "train_scala.jsonl"):
        if not (r1 / name).exists():
            raise SystemExit(f"{r1 / name} is missing: run ts.repair_ts for this seed first")
    if not (r1 / "adapter_scala" / "train_summary.json").exists():
        with slot() as lane:
            loop.train_adapter(r1 / "train_scala.jsonl", r1 / "adapter_scala", seed, r1 / "train_scala.log", lane,
                               max_steps=sft_max_steps)
    return {"sql": str(r1 / "adapter_sql"), "scala": str(r1 / "adapter_scala")}


def run_round(run_id, seed, round_number, previous, config, per_round, limit=None, sft_max_steps=None,
              eval_limit=None, slot=gpu_slot):
    run_dir = ROOT / "runs" / run_id
    rd = run_dir / f"round{round_number}"
    adapters = {language: str(rd / f"adapter_{language}") for language in ("sql", "scala")}
    if (rd / "result.json").exists():
        return adapters, json.loads((rd / "result.json").read_text())
    rd.mkdir(parents=True, exist_ok=True)
    timings = {}

    start = time.perf_counter()
    if not all((rd / f).exists() for f in ("questions.jsonl", "sql.jsonl", "scala.jsonl")):
        table_ids = [line.split("\t")[0] for line in (ROOT / "results/ids/corpus.txt").read_text().splitlines()]
        tables = loop.round_tables(table_ids, seed, round_number, per_round)
        served = {language: loop.adapter_name(language, path) for language, path in previous.items()}
        with slot() as lane, loop.Server({served[l]: p for l, p in previous.items()}, rd / "server.log", lane):
            questions = loop.step_questions(rd / "questions.jsonl", tables, seed, round_number,
                                            LLMClient(loop.MODEL_NAME, base_url=lane.url),
                                            config["generation"]["question_temperature"])
            for language in ("sql", "scala"):
                loop.step_code(rd / f"{language}.jsonl", questions, language,
                               LLMClient(served[language], base_url=lane.url),
                               config["generation"]["code_temperature"], seed)
    questions = loop.read_jsonl(rd / "questions.jsonl")
    sql_rows, scala_rows = loop.read_jsonl(rd / "sql.jsonl"), loop.read_jsonl(rd / "scala.jsonl")
    timings["generation_s"] = round(time.perf_counter() - start)

    start = time.perf_counter()
    with ext2.validation_slot() if not (rd / "validated.jsonl").exists() else nullcontext():
        validated = loop.step_validate(rd / "validated.jsonl", loop.pair_code(sql_rows, scala_rows),
                                       "table_specialist", config["validation"])
    timings["validation_s"] = round(time.perf_counter() - start)

    start = time.perf_counter()
    repair_summary = repair.run(rd, rd / "repair", None, seed, limit=limit, repairer="instruct", gpu_slot=slot)
    timings["repair_s"] = round(time.perf_counter() - start)

    kept, recovered = training_pairs(validated, loop.read_jsonl(rd / "repair" / "repaired.jsonl"))
    sql_examples, scala_examples = loop.step_examples(rd / "train_sql.jsonl", rd / "train_scala.jsonl",
                                                      kept + recovered)

    def train(language):
        with slot() as lane:
            return loop.train_adapter(rd / f"train_{language}.jsonl", adapters[language], seed,
                                      rd / f"train_{language}.log", lane, max_steps=sft_max_steps)

    def sql_then_evaluate():
        summary = train("sql")
        with slot() as lane:
            scores = loop.step_evaluate(run_id, round_number, adapters["sql"], rd, lane=lane, limit=eval_limit)
        return summary, scores

    start = time.perf_counter()
    (sql_train, scores), scala_train = in_parallel(sql_then_evaluate, lambda: train("scala"))
    timings["train_and_evaluate_s"] = round(time.perf_counter() - start)

    result = {"seed": seed, "round": round_number, **loop.summarise(questions, sql_rows, scala_rows, validated),
              "repair": repair_summary, "kept_ts": len(kept), "recovered": len(recovered),
              "sft_examples": {"sql": len(sql_examples), "scala": len(scala_examples)},
              "train": {"sql": sql_train, "scala": scala_train}, "test_scores": scores, "timings": timings}
    (rd / "result.json").write_text(json.dumps(result, indent=1))
    return adapters, result


def run(seed, tag="", rounds=3, per_round=None, limit=None, sft_max_steps=None, eval_limit=None, source_tag=""):
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    per_round = per_round or config["study"]["tables_per_round"]
    run_id = f"train_repairts{tag}_{loop.PROFILE}_seed{seed}"
    if not (ROOT / "runs" / run_id / "round1" / "result.json").exists():  # base round 1 exists already
        repair_ts.run(seed, None, tag, limit, sft_max_steps, eval_limit, slot=gpu_slot, source_tag=source_tag)
    previous = round1_adapters(ROOT / "runs" / run_id, seed, sft_max_steps)
    for round_number in range(2, rounds + 1):
        finished = (ROOT / "runs" / run_id / f"round{round_number}" / "result.json").exists()
        previous, result = run_round(run_id, seed, round_number, previous, config, per_round, limit, sft_max_steps,
                                     eval_limit)
        if finished:
            continue
        if not tag:
            log_run(f"{run_id}_round{round_number}", "extension3",
                    {"variant": "repairts", "seed": seed, "round": round_number}, result)
        print("REPAIRTS_ROUND " + json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default="", help="smoke tests: a run-name suffix such as smoke")
    ap.add_argument("--tables-per-round", type=int, help="smoke tests only")
    ap.add_argument("--limit", type=int, help="smoke tests only: repair the first N rejected pairs")
    ap.add_argument("--sft-max-steps", type=int)
    ap.add_argument("--eval-limit", type=int)
    ap.add_argument("--source-tag", default="", help="smoke tests only: round 1 repairs this TS run's suffix")
    args = ap.parse_args()
    run(args.seed, args.tag, args.rounds, args.tables_per_round, args.limit, args.sft_max_steps, args.eval_limit,
        args.source_tag)


if __name__ == "__main__":
    main()
