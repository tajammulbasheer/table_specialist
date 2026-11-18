"""Phase 8 extension (not in the paper): repair-ED, the error-detection analogue of repair-TS, and its two controls.

TS for error detection (ts/ed_loop.py) discards every error that the validator does not find on all 5 shuffles. Here a
rejected error gets one more attempt from the same generator (vanilla, temperature 0.7), then Algorithm 2 again,
unchanged, with the same validator. Clean columns that fail are not retried (there is nothing to rewrite).

  repair    (train_edrepair)   the generator sees its own answer and what the validator did on the 5 shuffles
                               (how many found the error, and what it named instead), then writes a new error
  resample  (train_edresample) best-of-2 control: the same generation prompt sampled once more, no feedback. Against
                               repair it isolates the feedback from simply having a second attempt
  samedata  (train_eddata)     same-data control: TS round 1's kept columns + N more kept errors from another seed's
                               TS round 1 (tables this seed's round never used), N = repair round 1's recovered count,
                               matched by kind. Against repair it isolates where the extra examples come from
                               from how many there are. Training and scoring only

Each round is as in ts/ed_loop.py: the same 3,000 tables as TS's round, the same generated errors (vanilla with the
same prompts and seed; TS's errors.jsonl is read when it exists), validation by the previous round's adapter of this
run (round 1: vanilla, so TS round 1's validated.jsonl is read), one adapter from vanilla on kept + recovered columns,
AutoTest ST and RT. TS runs are only read. Stages resume from their files: validated.jsonl, retry.jsonl,
revalidated.jsonl, train_ed.jsonl, the adapter, the scores, summary.json.

  TS_PROFILE=instruct python -m ts.repair_ed --variant repair --seeds 1 2 3
  TS_PROFILE=instruct python -m ts.repair_ed --variant resample --seeds 1 2 3 --rounds 1
  TS_PROFILE=instruct python -m ts.repair_ed --variant samedata --seeds 1 2 3
  TS_PROFILE=instruct python -m ts.repair_ed --variant repair --seeds 1 --tag smoke --tables 40 --negative-tables 10 \\
      --sft-max-steps 5 --eval-limit 20 --rounds 2
"""
import argparse
import json
import os
import random
import time

import yaml

from ts import ed_loop, loop
from ts.eval import ed_eval
from ts.llm.client import LLMClient
from ts.repo import ROOT
from ts.tracking import check_run_id, log_run

RUN_PREFIX = {"repair": "edrepair", "resample": "edresample", "samedata": "eddata"}
INSTRUCTION = "Upon review, the error is not obvious enough and should be rewritten as follows:"  # ours, after the
# repo's correction phrase for code ("Upon review, code is incorrect and should be rewritten as follows:")
SHOWN_VALUES = 3       # ours: at most this many other values named by the validator are shown
SHOWN_LENGTH = 60      # ours: each shown value is cut to this many characters
RESAMPLE_SEED_OFFSET = 1000  # ours: a different request seed, so the second sample is not the cached first one


def run_id_for(variant, seed, tag=""):
    return check_run_id(f"train_{RUN_PREFIX[variant]}{tag}_{loop.PROFILE}_seed{seed}")


def ts_round_dir(seed, round_number, tag=""):
    return ROOT / "runs" / f"train_edts{tag}_{loop.PROFILE}_seed{seed}" / f"round{round_number}"


# ---------------------------------------------------------------- feedback

def _show(value):
    value = str(value)
    return json.dumps(value if len(value) <= SHOWN_LENGTH else value[:SHOWN_LENGTH] + "...", ensure_ascii=False)


def feedback(error, predictions):
    """What the validator did on the 5 shuffles, in words: how often it named exactly the error, and what else."""
    total = len(predictions)
    found = sum(p == [error] for p in predictions)
    nothing = sum(not p for p in predictions)
    others = []
    for p in predictions:
        for value in p:
            if value != error and value not in others:
                others.append(value)
    lines = [f"A checker was shown the column with your error added, in {total} different orders of the values. "
             f"It named exactly your error in {found} of the {total} orders."]
    if nothing:
        lines.append(f"In {nothing} of the orders it found no error.")
    if others:
        shown = ", ".join(_show(v) for v in others[:SHOWN_VALUES])
        more = f" and {len(others) - SHOWN_VALUES} more" if len(others) > SHOWN_VALUES else ""
        lines.append(f"It named other values as errors: {shown}{more}.")
    lines.append("Write a different error that is clearly wrong for this column, so that it is found in every order. "
                 "Answer in the same JSON format.")
    return " ".join(lines)


def repair_prompt(prompt, response, error, predictions):
    return f"{prompt}\n\nYour answer was:\n{response.strip()}\n\n{feedback(error, predictions)}\n\n{INSTRUCTION}"


def retry_requests(variant, generated, validated, seed):
    """One request per rejected error: (row, prompt, request seed). Clean columns are never retried."""
    by_key = {(r["table_id"], r["column"], r["kind"]): r for r in generated if r["error"]}
    out = []
    for v in validated:
        if v["keep"] or v["kind"] == "negative":
            continue
        row = by_key[(v["table_id"], v["column"], v["kind"])]
        if variant == "repair":
            out.append((row, repair_prompt(row["prompt"], row["response"], row["error"], v["predictions"]), seed))
        else:
            out.append((row, row["prompt"], seed + RESAMPLE_SEED_OFFSET))
    return out


def accept_retry(row, text):
    """The new error, or None when it is unusable: unparsable, empty, already in the column, or the same error."""
    error = ed_loop.parse_error(text, row["kind"])
    if error is None or error in row["values"] or error == row["error"]:
        return None
    return error


# ---------------------------------------------------------------- one round

def _generated(round_dir, ts_dir, tables, seed, round_number, negative_tables, slot, config):
    """The round's generated errors: TS's when it has them (same model, prompts, seed and tables), else generated."""
    ts_path, own_path = ts_dir / "errors.jsonl", round_dir / "errors.jsonl"
    if own_path.exists():
        return loop.read_jsonl(own_path)
    if ts_path.exists():
        rows = loop.read_jsonl(ts_path)
    else:
        typo, semantic, _ = ed_loop.candidates(tables, seed, round_number, negative_tables)
        rows = typo + semantic
        with slot() as lane, loop.Server({}, round_dir / "server_generate.log", lane):
            client = LLMClient(loop.MODEL_NAME, base_url=lane.url)
            answers = client.complete_many([r["prompt"] for r in rows], api=loop.API, system=loop.SYSTEM,
                                           temperature=config["generation"]["question_temperature"],
                                           max_tokens=ed_loop.GENERATION_TOKENS, seed=seed)
        for row, answer in zip(rows, answers):
            row["response"] = answer["text"]
            error = ed_loop.parse_error(answer["text"], row["kind"])
            row["error"] = error if error is not None and error not in row["values"] else None
    loop.write_jsonl(own_path, rows)
    return rows


def _validate(items, validator, log_path, slot, config):
    name, adapters = ed_loop._served(validator)
    with slot() as lane, loop.Server(adapters, log_path, lane):
        client = LLMClient(name, base_url=lane.url)
        flat = [p for item in items for p in item["prompts"]]
        answers = client.complete_many(flat, api=loop.API, system=loop.SYSTEM,
                                       temperature=config["generation"]["code_temperature"],
                                       max_tokens=ed_loop.VALIDATION_TOKENS, seed=0)
    out = []
    for i, item in enumerate(items):
        texts = [a["text"] for a in answers[i * ed_loop.SHUFFLES:(i + 1) * ed_loop.SHUFFLES]]
        v = ed_loop.verdict(item["expected"], texts)
        out.append({k: item[k] for k in ("table_id", "column", "kind", "expected", "orders")} | v | {"answers": texts})
    return out


def run_round(variant, run_id, seed, round_number, run_dir, previous, per_round, negative_tables, slot, sft_max_steps,
              eval_limit, ts_tag=""):
    round_dir = run_dir / f"round{round_number}"
    if (round_dir / "summary.json").exists():
        return json.loads((round_dir / "summary.json").read_text())
    round_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    timings = {}
    table_ids = [line.split("\t")[0] for line in (ROOT / "results/ids/corpus.txt").read_text().splitlines()]
    tables = loop.round_tables(table_ids, seed, round_number, per_round)
    ts_dir = ts_round_dir(seed, round_number, ts_tag)

    generated = _generated(round_dir, ts_dir, tables, seed, round_number, negative_tables, slot, config)
    val_path = round_dir / "validated.jsonl"
    if not val_path.exists():
        if previous is None and (ts_dir / "validated.jsonl").exists():  # round 1: the same vanilla validator as TS
            validated = loop.read_jsonl(ts_dir / "validated.jsonl")
        else:
            _, _, negative = ed_loop.candidates(tables, seed, round_number, negative_tables)
            start = time.perf_counter()
            validated = _validate(ed_loop.validation_items([r for r in generated if r["error"]], negative), previous,
                                  round_dir / "server_validate.log", slot, config)
            timings["validation_s"] = round(time.perf_counter() - start)
        loop.write_jsonl(val_path, validated)
    validated = loop.read_jsonl(val_path)

    retry_path, reval_path = round_dir / "retry.jsonl", round_dir / "revalidated.jsonl"
    if not retry_path.exists():
        requests = retry_requests(variant, generated, validated, seed)
        start = time.perf_counter()
        with slot() as lane, loop.Server({}, round_dir / "server_retry.log", lane):
            client = LLMClient(loop.MODEL_NAME, base_url=lane.url)
            answers = []
            for request_seed in sorted({s for _, _, s in requests}):
                group = [(i, p) for i, (_, p, s) in enumerate(requests) if s == request_seed]
                texts = client.complete_many([p for _, p in group], api=loop.API, system=loop.SYSTEM,
                                             temperature=config["generation"]["question_temperature"],
                                             max_tokens=ed_loop.GENERATION_TOKENS, seed=request_seed)
                answers += [(i, t) for (i, _), t in zip(group, texts)]
        answers = dict(answers)
        retries = []
        for i, (row, prompt, request_seed) in enumerate(requests):
            text = answers[i]["text"]
            retries.append({k: row[k] for k in ("table_id", "column", "kind", "values")}
                           | {"first_error": row["error"], "prompt": prompt, "request_seed": request_seed,
                              "response": text, "error": accept_retry(row, text)})
        loop.write_jsonl(retry_path, retries)
        timings["retry_s"] = round(time.perf_counter() - start)
    retries = loop.read_jsonl(retry_path)
    if not reval_path.exists():
        start = time.perf_counter()
        items = ed_loop.validation_items([r for r in retries if r["error"]], [])
        loop.write_jsonl(reval_path, _validate(items, previous, round_dir / "server_revalidate.log", slot, config)
                         if items else [])
        timings["revalidation_s"] = round(time.perf_counter() - start)
    revalidated = loop.read_jsonl(reval_path)
    kept = [v for v in validated if v["keep"]]
    recovered = [v for v in revalidated if v["keep"]]

    summary = train_and_score(run_id, seed, round_number, round_dir, kept + recovered, slot, sft_max_steps, eval_limit,
                              timings)
    by_kind = {}
    for v in validated:
        k = by_kind.setdefault(v["kind"], {"candidates": 0, "kept": 0})
        k["candidates"] += 1
        k["kept"] += v["keep"]
    retried = {}
    for kind in ("typo", "semantic"):
        retried[kind] = {"rejected": sum(1 for r in retries if r["kind"] == kind),
                         "usable": sum(1 for r in retries if r["kind"] == kind and r["error"]),
                         "recovered": sum(1 for r in recovered if r["kind"] == kind)}
    summary = {"run_id": run_id, "variant": variant, "seed": seed, "round": round_number, "tables": len(tables),
               "validator": str(previous) if previous else "vanilla", "validation": by_kind, "kept": len(kept),
               "retry": retried, "recovered": len(recovered), **summary}
    return finish(round_dir, summary, run_id, variant)


def train_and_score(run_id, seed, round_number, round_dir, rows, slot, sft_max_steps, eval_limit, timings):
    examples_path = round_dir / "train_ed.jsonl"
    if not examples_path.exists():
        loop.write_jsonl(examples_path, ed_loop.training_examples(rows))
    examples = loop.read_jsonl(examples_path)
    start = time.perf_counter()
    with slot() as lane:
        train = loop.train_adapter(examples_path, round_dir / "adapter_ed", seed, round_dir / "train_ed.log", lane,
                                   max_steps=sft_max_steps)
    timings["train_s"] = round(time.perf_counter() - start)
    model = f"{run_id.removeprefix('train_')}_round{round_number}"
    start = time.perf_counter()
    ed_eval.evaluate(model, round_dir / "adapter_ed", "train", eval_limit)
    timings["evaluate_s"] = round(time.perf_counter() - start)
    scores = {b: json.loads((ROOT / "results/baseline" / f"{ed_eval.run_id('train', model, b)}.json").read_text())
              for b in ed_eval.BENCHES}
    return {"examples": len(examples), "train": train,
            "test": {b: {"f1_repo": s["repo_faithful"]["f1"], "f1_autotest": s["autotest_protocol"]["f1"]}
                     for b, s in scores.items()},
            "timings": timings}


def finish(round_dir, summary, run_id, variant):
    (round_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if "smoke" not in run_id:
        log_run(f"{run_id}_round{summary['round']}", "error_detection_extension", {"variant": variant,
                "seed": summary["seed"], "round": summary["round"]},
                {"examples": summary["examples"], **{f"f1_{b}": s["f1_repo"] for b, s in summary["test"].items()}})
    print("ROUND " + json.dumps(summary), flush=True)
    return summary


# ---------------------------------------------------------------- same-data control

def samedata_rows(seed, own_tables, ts_tag="", repair_tag=""):
    """TS round 1's kept columns + extra kept errors from the donor seed's TS round 1, as many of each kind as repair
    round 1 recovered, from tables this seed's round 1 never used (own_tables: all of its 3,000)."""
    donor = seed % 3 + 1
    own = loop.read_jsonl(ts_round_dir(seed, 1, ts_tag) / "validated.jsonl")
    donor_rows = loop.read_jsonl(ts_round_dir(donor, 1, ts_tag) / "validated.jsonl")
    repair = json.loads((ROOT / "runs" / run_id_for("repair", seed, repair_tag) / "round1/summary.json").read_text())
    own_tables = set(own_tables)
    rng = random.Random(seed)
    extra = []
    for kind in ("typo", "semantic"):
        need = repair["retry"][kind]["recovered"]
        pool = [r for r in donor_rows if r["keep"] and r["kind"] == kind and r["table_id"] not in own_tables]
        if len(pool) < need:
            raise SystemExit(f"seed {seed}: donor seed {donor} has {len(pool)} kept {kind} errors, {need} needed")
        extra += rng.sample(pool, need)
    return [r for r in own if r["keep"]], extra, donor


def run_samedata(seed, tag, slot, sft_max_steps, eval_limit, per_round, ts_tag="", repair_tag=""):
    run_id = run_id_for("samedata", seed, tag)
    round_dir = ROOT / "runs" / run_id / "round1"
    if (round_dir / "summary.json").exists():
        return json.loads((round_dir / "summary.json").read_text())
    round_dir.mkdir(parents=True, exist_ok=True)
    table_ids = [line.split("\t")[0] for line in (ROOT / "results/ids/corpus.txt").read_text().splitlines()]
    kept, extra, donor = samedata_rows(seed, loop.round_tables(table_ids, seed, 1, per_round), ts_tag, repair_tag)
    if not (round_dir / "extra.jsonl").exists():
        loop.write_jsonl(round_dir / "extra.jsonl", extra)
    extra = loop.read_jsonl(round_dir / "extra.jsonl")
    summary = train_and_score(run_id, seed, 1, round_dir, kept + extra, slot, sft_max_steps, eval_limit, {})
    summary = {"run_id": run_id, "variant": "samedata", "seed": seed, "round": 1, "donor_seed": donor,
               "kept": len(kept), "extra": len(extra), **summary}
    return finish(round_dir, summary, run_id, "samedata")


# ---------------------------------------------------------------- driver

def run(variant, seed, tag="", rounds=3, per_round=None, negative_tables=ed_loop.NEGATIVE_TABLES, sft_max_steps=None,
        eval_limit=None, ts_tag=""):
    from functools import partial
    from ts.repair_loop import gpu_slot
    if os.environ.get("TS_ED_GPUS"):  # e.g. "1": keep off the GPU another experiment uses
        gpu_slot = partial(gpu_slot, gpus=tuple(int(g) for g in os.environ["TS_ED_GPUS"].split(",")))
    if loop.PROFILE != "instruct":
        raise SystemExit("Phase 8 is Instruct only: set TS_PROFILE=instruct")
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    per_round = per_round or config["study"]["tables_per_round"]
    if variant == "samedata":
        return run_samedata(seed, tag, gpu_slot, sft_max_steps, eval_limit, per_round, ts_tag, repair_tag=tag)
    run_id = run_id_for(variant, seed, tag)
    run_dir = ROOT / "runs" / run_id
    previous = None
    for r in range(1, rounds + 1):
        run_round(variant, run_id, seed, r, run_dir, previous, per_round, negative_tables, gpu_slot, sft_max_steps,
                  eval_limit, ts_tag)
        previous = run_dir / f"round{r}" / "adapter_ed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(RUN_PREFIX), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tag", default="", help="smoke tests: a run-name suffix such as smoke")
    ap.add_argument("--ts-tag", default="", help="the TS run to read (smoke tests: smoke2)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tables", type=int, help="tables per round (default: configs/base.yaml, 3,000)")
    ap.add_argument("--negative-tables", type=int, default=ed_loop.NEGATIVE_TABLES)
    ap.add_argument("--sft-max-steps", type=int)
    ap.add_argument("--eval-limit", type=int)
    args = ap.parse_args()
    for seed in args.seeds:  # one seed at a time in a process: at most one GPU is held
        run(args.variant, seed, args.tag, args.rounds, args.tables, args.negative_tables, args.sft_max_steps,
            args.eval_limit, args.ts_tag)


if __name__ == "__main__":
    main()
