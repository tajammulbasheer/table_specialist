"""Phase 9: Table-Specialist for NL-to-R (paper Section 6, Table 3; Algorithm 1 with Algorithm 3), Instruct only.

The dual task of NL-to-R is NL-to-SQL, as in the main study's dual pair (SQL, Scala) with R in place of Scala:
  questions  vanilla writes one question per table. These are TS's own questions of the same seed and round
             (runs/train_tablespecialist_instruct_seed<s>/round<k>/questions.jsonl, read only): the same model, prompt,
             request seed and tables, so they are identical by construction
  code       the previous round's SQL and R adapters of this run write SQL and R (round 1: vanilla; its SQL is TS round
             1's sql.jsonl, identical by construction), greedy, with the repo's few-shot prompts
             (Generation/NL2DSL/NL2SQL/nl2sql_generation_v6.py, Generation/NL2DSL/NL2R/nl2r_generation_v6.py)
  validate   SQL vs R execution agreement on 10 row samples (ts/validate/r_agree.py; ours, the repo has no NL-to-R
             validation script)
  repair     arm repairts only: each rejected pair gets one repair from vanilla, as Instruct repair-TS does for Scala
             (ts/repair.py rules: feedback from what each program did on the rejecting sample; retries; Algorithm 3
             again unchanged); recovered pairs join the kept ones
  train      two adapters from vanilla on the round's kept (+ recovered) pairs: SQL in the repo's NL-to-SQL format,
             R in the repo's NL-to-R format (Finetune/data_generator/nl_to_r)
Test scoring is separate (ts/eval/r_eval.py), as for NL-to-Scala.

Run IDs: train_rts_instruct_seed<s> (TS) and train_rrepairts_instruct_seed<s> (repair-TS). Repair-TS round 1 starts
from TS round 1's validated pairs (the same vanilla code), as Instruct repair-TS does. Every stage resumes from its file.
GPUs: TS_R_GPUS (default "0,1") limits the GPUs this run may take, so that it never competes with other experiments.

  TS_PROFILE=instruct TS_R_GPUS=0 python -m ts.r_loop --arm ts --seeds 1 2 3
  TS_PROFILE=instruct TS_R_GPUS=0 python -m ts.r_loop --arm ts --seeds 1 --tag smoke --tables 30 --sft-max-steps 5 --rounds 2
"""
import argparse
import contextlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial

import yaml

from ts import ext2, loop, repair
from ts.gen.code import code_prompts, extract_code
from ts.llm.client import LLMClient
from ts.repo import ROOT, repo_prompt_generator, repo_r_prompt_generator
from ts.tracking import check_run_id, log_run
from ts.train.sft import build_example
from ts.validate.exec_agree import row_samples, run_sql
from ts.validate.r_agree import run_r_logged, validate_pair_r

ARMS = {"ts": "rts", "repairts": "rrepairts"}
LANGUAGES = ("sql", "r")
R_MAX_TOKENS = 512           # ours: as Scala (an R script is about as long), chat replies add text
REPAIRABLE = ("sql_failed", "always_empty", "r_failed", "disagree")
VALIDATION_WORKERS = int(os.environ.get("TS_R_WORKERS", loop.VALIDATION_WORKERS))
_R_ERROR = re.compile(r"^(Error\b.*|Warning message.*|  .*)$")


def gpu_slot():
    from ts.repair_loop import gpu_slot as slot
    gpus = tuple(int(g) for g in os.environ.get("TS_R_GPUS", "0,1").split(","))
    return slot(gpus=gpus)


def run_id_for(arm, seed, tag=""):
    return check_run_id(f"train_{ARMS[arm]}{tag}_{loop.PROFILE}_seed{seed}")


def ts_round_dir(seed, round_number, tag=""):
    """The main study's TS round (SQL + Scala), read only."""
    return ROOT / "runs" / f"train_tablespecialist{tag}_{loop.PROFILE}_seed{seed}" / f"round{round_number}"


# ---------------------------------------------------------------- generation

def step_code(path, questions, language, client, temperature, seed):
    if path.exists():
        return loop.read_jsonl(path)
    usable = [q for q in questions if q["question"]]
    prompts = [code_prompts(loop.load_table(q["table_id"]), q["question"])[language] for q in usable]
    limit = R_MAX_TOKENS if language == "r" else loop.GENERATION_LIMITS[language]
    responses = client.complete_many(prompts, api=loop.API, system=loop.SYSTEM, temperature=temperature,
                                     max_tokens=limit, seed=seed)
    rows = [{"table_id": q["table_id"], "question": q["question"], "code": extract_code(a["text"], language),
             "response": a["text"], "finish_reason": a["finish_reason"]} for q, a in zip(usable, responses)]
    loop.write_jsonl(path, rows)
    return rows


def pair_code(sql_rows, r_rows):
    r = {row["table_id"]: row["code"] for row in r_rows}
    return [{"table_id": s["table_id"], "question": s["question"], "sql": s["code"], "r": r.get(s["table_id"])}
            for s in sql_rows]


def validate_one(pair, settings):
    if not pair["sql"] or not pair["r"]:
        return {**pair, "keep": False, "reason": "no_code", "samples_agreed": 0, "r_runs": 0}
    try:
        v = validate_pair_r(pair["sql"], pair["r"], loop.load_table_as_text(pair["table_id"]), settings["row_samples"],
                            settings["row_fraction"], settings["numeric_tolerance"])
    except Exception as error:  # recorded and counted as rejected, as in ts/loop.py
        return {**pair, "keep": False, "reason": f"error: {type(error).__name__}: {error}"[:200], "samples_agreed": 0,
                "r_runs": 0}
    return {**pair, "keep": v.keep, "reason": v.reason, "samples_agreed": v.samples_agreed, "r_runs": v.scala_runs}


# ---------------------------------------------------------------- repair (arm repairts)

def r_messages(log):
    """R's error messages, at most repair.SHOWN_ERRORS. A message is the Error line plus the lines after it up to the
    backtrace, so that rlang/dplyr errors keep their cause ("! object 'x' not found"), capped at 300 characters."""
    found = []
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Error"):
            message = line.strip()
            for more in lines[i + 1:]:
                if not more.strip() or more.startswith(("Backtrace:", "Execution halted", "Error", "In addition:")):
                    break
                message += " " + more.strip()
            message = message[:300]
            if message not in found:
                found.append(message)
    return found[:repair.SHOWN_ERRORS]


def feedback_for(pair, frame):
    """What each program did on the sample where the pair was rejected, as repair.feedback_for does for Scala."""
    samples = row_samples(frame.astype(str))
    reason = pair["reason"]
    if reason == "sql_failed":
        message = next((m for m in (repair.sql_error(pair["sql"], s) for s in samples) if m), "the query failed")
        return {"sql": f"Running this code on the table gives the error: {message}"}
    if reason == "always_empty":
        return {"sql": "Running this code on the table returns no rows."}
    sample = samples[min(pair["samples_agreed"], len(samples) - 1)]
    if reason == "r_failed":
        _, log = run_r_logged(pair["r"], sample)
        messages = r_messages(log)
        return {"r": "Running this code on the table gives the error: " + ("\n".join(messages) if messages
                                                                          else "the program failed")}
    if reason == "disagree":
        rows, _ = run_r_logged(pair["r"], sample)
        return {"sql": repair.show_rows(run_sql(pair["sql"], sample)), "r": repair.show_rows(rows)}
    return {}


def plan_one(pair):
    frame = loop.load_table(pair["table_id"])
    try:
        notes, problem = feedback_for(pair, loop.load_table_as_text(pair["table_id"])), None
    except Exception as error:
        notes, problem = {}, f"error: {type(error).__name__}: {error}"[:200]
    prompts = {language: repair.chat_repair_prompt(language, frame, pair["question"], pair[language], note)
               for language, note in notes.items()}
    return {**{k: pair[k] for k in ("table_id", "question", "sql", "r", "reason", "samples_agreed")},
            "feedback": notes, "chat_prompts": prompts, "feedback_error": problem}


def generate_repairs(plans, client, seed, settings, greedy_temperature):
    """Greedy first; a reply that is empty or copies the failed code is replaced by the first of up to
    settings["retries"] samples that differs (repair.generate_repairs, chat form)."""
    for language in LANGUAGES:
        todo = [p for p in plans if language in p["chat_prompts"]]
        limit = R_MAX_TOKENS if language == "r" else settings["chat_max_tokens"][language]
        for attempt in range(settings["retries"] + 1):
            todo = [p for p in todo if not repair.is_new(p.get(f"new_{language}"), p[language])]
            if not todo:
                break
            replies = client.complete_many([p["chat_prompts"][language] for p in todo], api="chat",
                                           temperature=greedy_temperature if attempt == 0 else settings["retry_temperature"],
                                           max_tokens=limit, seed=seed if attempt == 0 else seed * 100 + attempt)
            for plan, reply in zip(todo, replies):
                plan[f"new_{language}"] = extract_code(reply["text"], language)
                plan[f"new_{language}_response"] = reply["text"]
                plan[f"new_{language}_attempt"] = attempt
    return plans


def candidates(plan):
    new_sql, new_r = plan.get("new_sql"), plan.get("new_r")
    order = [("new_sql", new_sql, plan["r"]), ("new_r", plan["sql"], new_r), ("both", new_sql, new_r)]
    seen, out = set(), []
    for name, sql, r in order:
        if sql and r and (sql, r) != (plan["sql"], plan["r"]) and (sql, r) not in seen:
            seen.add((sql, r))
            out.append((name, sql, r))
    return out


def validate_repaired(plan, settings):
    tried, runs = [], 0
    try:
        frame = loop.load_table_as_text(plan["table_id"])
        for name, sql, r in candidates(plan):
            v = validate_pair_r(sql, r, frame, settings["row_samples"], settings["row_fraction"],
                                settings["numeric_tolerance"])
            runs += v.scala_runs
            tried.append({"combo": name, "reason": v.reason})
            if v.keep:
                return {**plan, "keep": True, "reason": "repaired", "combo": name, "repaired_sql": sql,
                        "repaired_r": r, "tried": tried, "r_runs": runs}
    except Exception as error:
        return {**plan, "keep": False, "reason": f"error: {type(error).__name__}: {error}"[:200], "tried": tried,
                "r_runs": runs}
    return {**plan, "keep": False, "reason": "not_repaired" if tried else "no_new_code", "tried": tried,
            "r_runs": runs}


def step_repair(round_dir, validated, seed, config, slot):
    plans_path, generated_path, repaired_path = (round_dir / "repair_plans.jsonl", round_dir / "repair_generated.jsonl",
                                                 round_dir / "repaired.jsonl")
    if not plans_path.exists():
        todo = [p for p in validated if not p["keep"] and p["reason"] in REPAIRABLE]
        with ext2.validation_slot(), ThreadPoolExecutor(VALIDATION_WORKERS) as pool:
            loop.write_jsonl(plans_path, list(pool.map(plan_one, todo)))
    if not generated_path.exists():
        plans = loop.read_jsonl(plans_path)
        with slot() as lane, loop.Server({}, round_dir / "server_repair.log", lane):
            generate_repairs(plans, LLMClient(loop.MODEL_NAME, base_url=lane.url), seed, config["repair"],
                             config["generation"]["code_temperature"])
        loop.write_jsonl(generated_path, plans)
    if repaired_path.exists():
        return loop.read_jsonl(repaired_path)
    items = []
    for p in loop.read_jsonl(generated_path):
        item = {k: v for k, v in p.items() if k != "chat_prompts"}
        item["reason_before"] = item.pop("reason")
        items.append(item)
    with ext2.validation_slot():
        return loop.step_validate(repaired_path, items, "table_specialist", config["validation"],
                                  workers=VALIDATION_WORKERS, validate_one=validate_repaired)


# ---------------------------------------------------------------- training examples

def step_examples(round_dir, pairs):
    """Paper Section 5: a validated pair gives a (question, SQL) and a (question, R) example, in the repo's formats."""
    paths = {language: round_dir / f"train_{language}.jsonl" for language in LANGUAGES}
    if all(p.exists() for p in paths.values()):
        return {language: loop.read_jsonl(p) for language, p in paths.items()}
    sql_generator, r_generator = repo_prompt_generator(), repo_r_prompt_generator()
    rows = {"sql": [], "r": []}
    for pair in pairs:
        table = loop.load_table(pair["table_id"])
        sql_example = build_example(sql_generator, table, pair["question"], pair["sql"])
        r_prompt = r_generator.generate_prompt((table, pair["question"]), None)  # the repo's NL-to-R input: (df, q)
        r_example = {"prompt": r_prompt, "completion": r_generator.generate_completion(pair["r"])} if r_prompt else None
        for language, example in (("sql", sql_example), ("r", r_example)):
            if example:
                rows[language].append(loop.as_chat(example) if loop.PROFILE == "instruct" else example)
    for language, path in paths.items():
        loop.write_jsonl(path, rows[language])
    return rows


# ---------------------------------------------------------------- one round

def run_round(arm, run_id, seed, round_number, run_dir, previous, per_round, slot, sft_max_steps, tag=""):
    round_dir = run_dir / f"round{round_number}"
    if (round_dir / "summary.json").exists():
        return json.loads((round_dir / "summary.json").read_text())
    round_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    timings = {}
    table_ids = [line.split("\t")[0] for line in (ROOT / "results/ids/corpus.txt").read_text().splitlines()]
    full = config["study"]["tables_per_round"]
    tables = loop.round_tables(table_ids, seed, round_number, full)[:per_round]  # smoke tests: TS's first tables
    source = ts_round_dir(seed, round_number)
    shared = (ROOT / "runs" / run_id_for("ts", seed, tag) / "round1") if arm == "repairts" and round_number == 1 else None

    # questions and code
    start = time.perf_counter()
    q_path = round_dir / "questions.jsonl"
    if not q_path.exists():
        by_table = {r["table_id"]: r for r in loop.read_jsonl((shared or source) / "questions.jsonl")}
        rows = [by_table[t] for t in tables if t in by_table]
        if per_round == full and len(by_table) != len(tables):
            raise SystemExit(f"{source}: its questions do not cover this round's {len(tables)} tables")
        loop.write_jsonl(q_path, rows)
    questions = loop.read_jsonl(q_path)
    wanted = {q["table_id"] for q in questions}
    for language in LANGUAGES:
        path = round_dir / f"{language}.jsonl"
        if path.exists():
            continue
        reuse = (shared / f"{language}.jsonl") if shared else (source / "sql.jsonl" if language == "sql"
                                                                and round_number == 1 else None)
        if reuse is not None:
            loop.write_jsonl(path, [r for r in loop.read_jsonl(reuse) if r["table_id"] in wanted])
            continue
        adapter = (previous or {}).get(language)
        served = loop.adapter_name(language, adapter) if adapter else loop.MODEL_NAME
        with slot() as lane, loop.Server({served: str(adapter)} if adapter else {},
                                         round_dir / f"server_{language}.log", lane):
            step_code(path, questions, language, LLMClient(served, base_url=lane.url),
                      config["generation"]["code_temperature"], seed)
    sql_rows, r_rows = loop.read_jsonl(round_dir / "sql.jsonl"), loop.read_jsonl(round_dir / "r.jsonl")
    timings["generation_s"] = round(time.perf_counter() - start)

    # validation (Algorithm 3, SQL vs R)
    start = time.perf_counter()
    val_path = round_dir / "validated.jsonl"
    if not val_path.exists() and shared and (shared / "validated.jsonl").exists():
        loop.write_jsonl(val_path, loop.read_jsonl(shared / "validated.jsonl"))
    if not val_path.exists():
        # TS_R_HOLD_GPU=1: keep this run's GPU during the CPU-only validation, so that another experiment's
        # gpu_slot (which takes any free GPU) cannot take it for hours in that gap.
        hold = slot() if os.environ.get("TS_R_HOLD_GPU") == "1" else contextlib.nullcontext()
        with hold, ext2.validation_slot():
            loop.step_validate(val_path, pair_code(sql_rows, r_rows), "table_specialist", config["validation"],
                               workers=VALIDATION_WORKERS, validate_one=validate_one)
    validated = loop.read_jsonl(val_path)
    timings["validation_s"] = round(time.perf_counter() - start)
    kept = [{k: v[k] for k in ("table_id", "question", "sql", "r")} for v in validated if v["keep"]]

    recovered = []
    if arm == "repairts":
        start = time.perf_counter()
        repaired = step_repair(round_dir, validated, seed, config, slot)
        recovered = [{"table_id": p["table_id"], "question": p["question"], "sql": p["repaired_sql"],
                      "r": p["repaired_r"]} for p in repaired if p["keep"]]
        timings["repair_s"] = round(time.perf_counter() - start)

    # training
    start = time.perf_counter()
    examples = step_examples(round_dir, kept + recovered)
    train = {}
    for language in LANGUAGES:
        with slot() as lane:
            train[language] = loop.train_adapter(round_dir / f"train_{language}.jsonl", round_dir / f"adapter_{language}",
                                                 seed, round_dir / f"train_{language}.log", lane, max_steps=sft_max_steps)
    timings["training_s"] = round(time.perf_counter() - start)

    reasons = {}
    for v in validated:
        reasons[v["reason"]] = reasons.get(v["reason"], 0) + 1
    summary = {"run_id": run_id, "arm": arm, "seed": seed, "round": round_number, "tables": len(tables),
               "questions": sum(bool(q["question"]) for q in questions),
               "sql_extracted": sum(bool(r["code"]) for r in sql_rows), "r_extracted": sum(bool(r["code"]) for r in r_rows),
               "kept": len(kept), "recovered": len(recovered), "reasons": reasons,
               "r_runs": sum(v.get("r_runs", 0) for v in validated),
               "examples": {language: len(rows) for language, rows in examples.items()},
               "train": train, "timings": timings,
               "generator": {language: str(p) for language, p in (previous or {}).items()} or "vanilla"}
    (round_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if "smoke" not in run_id:
        log_run(f"{run_id}_round{round_number}", "nl2r", {"arm": arm, "seed": seed, "round": round_number},
                {"kept": len(kept), "recovered": len(recovered), **{f"examples_{k}": v for k, v in
                                                                     summary["examples"].items()}})
    print("ROUND " + json.dumps({k: v for k, v in summary.items() if k != "train"}), flush=True)
    return summary


def run(arm, seed, tag="", rounds=3, per_round=None, sft_max_steps=None):
    if loop.PROFILE != "instruct":
        raise SystemExit("NL-to-R is Instruct only: set TS_PROFILE=instruct")
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    per_round = per_round or config["study"]["tables_per_round"]
    run_id = run_id_for(arm, seed, tag)
    run_dir = ROOT / "runs" / run_id
    previous = None
    for r in range(1, rounds + 1):
        run_round(arm, run_id, seed, r, run_dir, previous, per_round, gpu_slot, sft_max_steps, tag)
        previous = {language: run_dir / f"round{r}" / f"adapter_{language}" for language in LANGUAGES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tag", default="", help="smoke tests: a run-name suffix such as smoke")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tables", type=int, help="tables per round (default: configs/base.yaml, 3,000)")
    ap.add_argument("--sft-max-steps", type=int)
    args = ap.parse_args()
    for seed in args.seeds:
        run(args.arm, seed, args.tag, args.rounds, args.tables, args.sft_max_steps)


if __name__ == "__main__":
    main()
