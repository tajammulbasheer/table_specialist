"""Run the Table-Specialist loop (paper Algorithm 1) for one arm and seed.

Each round: the vanilla model writes questions, the previous round's SQL and Scala models write code, validation keeps
agreeing pairs (Table-Specialist arm only), two fresh LoRA adapters train from vanilla on this round's pairs, and the
new SQL adapter is scored on the test sets.

Every step writes its output file and is skipped if that file exists, so a stopped run resumes where it stopped.

  python -m ts.loop --arm table_specialist --seed 1 --rounds 3 --tables-per-round 3000
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from ts.gen.code import code_prompts, extract_code
from ts.gen.questions import build_requests, parse_question
from ts.llm.client import LLMClient
from ts.repo import REPO_SYSTEM_MESSAGE, ROOT, repo_prompt_generator, repo_scala_prompt_generator
from ts.tracking import check_run_id, log_run
from ts.train.sft import build_example
from ts.validate.exec_agree import validate_pair

ARMS = ("table_specialist", "no_validation")
MODEL_NAME = "Llama-3.1-8B"
MODEL_PATH = "models/Llama-3.1-8B"
# Plan A2 (docs/PLAN_NEXT.md): TS_PROFILE=instruct runs the same loop on Llama-3.1-8B-Instruct through the repo's chat
# call with its system message (Evaluete/queryWithDeployment.py), for generation, training and evaluation alike.
# Unset, everything below is exactly the base setting used for the core study.
PROFILE = os.environ.get("TS_PROFILE", "base")
if PROFILE not in ("base", "instruct"):
    raise ValueError(f"TS_PROFILE must be base or instruct, not {PROFILE}")
if PROFILE == "instruct":
    MODEL_NAME, MODEL_PATH = "Llama-3.1-8B-Instruct", "models/Llama-3.1-8B-Instruct"
API = "chat" if PROFILE == "instruct" else "completions"
SYSTEM = REPO_SYSTEM_MESSAGE if PROFILE == "instruct" else None
SERVER_CONTEXT = 8192  # the few-shot prompts' fixed text is about 2,150 tokens, plus a table of up to 2,000
GENERATION_LIMITS = ({"question": 256, "sql": 256, "scala": 512} if PROFILE == "instruct"  # chat answers carry prose
                     else {"question": 128, "sql": 128, "scala": 256})
VALIDATION_WORKERS = 36
BENCHMARKS = {"WikiSQL": "wikisql", "spider": "spider", "WikiQA": "wikitq", "bird": "bird"}  # repo name: our short name
EVAL_PROMPTS = ("zero-shot", "few-shot")


@dataclass(frozen=True)
class Lane:
    """The GPUs and port one run uses. Serving and training never overlap within a run, so one GPU can do both, and
    two runs on different GPUs do not meet."""
    serve_gpu: int = 0
    train_gpu: int = 1
    port: int = 8000

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"


def read_jsonl(path):
    # Only "\n" ends a row: str.splitlines also breaks on characters such as U+2028, which table text can contain.
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def write_jsonl(path, rows):
    """Write through a temporary name and force it to disk, so that neither a stopped run nor a power cut leaves a
    half-written step that looks finished."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w") as file:
        file.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        file.flush()
        os.fsync(file.fileno())
    temporary.rename(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def round_tables(table_ids, seed, round_number, per_round):
    """Each seed has its own table order, and each round takes the next slice, so tables never repeat within a seed."""
    order = list(table_ids)
    random.Random(seed).shuffle(order)
    end = per_round * round_number
    if end > len(order):
        raise ValueError(f"Round {round_number} needs {end} tables but the corpus has {len(order)}")
    return order[end - per_round:end]


def load_table(table_id, corpus=ROOT / "data/corpus/tables"):
    return pd.read_csv(corpus / f"{table_id}.csv", keep_default_na=False)


def load_table_as_text(table_id, corpus=ROOT / "data/corpus/tables"):
    """Validation samples strings, as the repo's prepare_exec.py does, so that values reach the CSV unchanged."""
    return pd.read_csv(corpus / f"{table_id}.csv", dtype=str, keep_default_na=False)


def adapter_name(language, path):
    """The served name, and so the response cache key, comes from the weights: the same name for different adapters
    would silently return another arm's or round's cached answers."""
    digest = hashlib.sha256((Path(path) / "adapter_model.safetensors").read_bytes()).hexdigest()
    return f"{language}-{digest[:16]}"


def step_questions(path, table_ids, seed, round_number, client, temperature):
    if path.exists():
        return read_jsonl(path)
    tables = [(t, load_table(t)) for t in table_ids]
    requests = build_requests(tables, seed * 1000 + round_number)
    responses = client.complete_many([r["prompt"] for r in requests], api=API, system=SYSTEM, temperature=temperature,
                                     max_tokens=GENERATION_LIMITS["question"], seed=seed)
    rows = [{"table_id": r["table_id"], "requirements": r["requirements"], "question": parse_question(a["text"]),
             "response": a["text"]} for r, a in zip(requests, responses)]
    write_jsonl(path, rows)
    return rows


def step_code(path, questions, language, client, temperature, seed):
    if path.exists():
        return read_jsonl(path)
    usable = [q for q in questions if q["question"]]
    prompts = [code_prompts(load_table(q["table_id"]), q["question"])[language] for q in usable]
    responses = client.complete_many(prompts, api=API, system=SYSTEM, temperature=temperature,
                                     max_tokens=GENERATION_LIMITS[language], seed=seed)
    rows = [{"table_id": q["table_id"], "question": q["question"], "code": extract_code(a["text"], language),
             "response": a["text"], "finish_reason": a["finish_reason"]} for q, a in zip(usable, responses)]
    write_jsonl(path, rows)
    return rows


def pair_code(sql_rows, scala_rows):
    scala = {r["table_id"]: r["code"] for r in scala_rows}
    return [{"table_id": r["table_id"], "question": r["question"], "sql": r["code"], "scala": scala.get(r["table_id"])}
            for r in sql_rows]


def _validate_one(pair, settings):
    if not pair["sql"] or not pair["scala"]:
        return {**pair, "keep": False, "reason": "no_code", "samples_agreed": 0, "scala_runs": 0}
    try:
        verdict = validate_pair(pair["sql"], pair["scala"], load_table_as_text(pair["table_id"]), settings["row_samples"],
                                settings["row_fraction"], settings["numeric_tolerance"])
    except Exception as error:  # one bad pair must not lose the whole round; it is recorded and counted as rejected
        return {**pair, "keep": False, "reason": f"error: {type(error).__name__}: {error}"[:200],
                "samples_agreed": 0, "scala_runs": 0}
    return {**pair, "keep": verdict.keep, "reason": verdict.reason, "samples_agreed": verdict.samples_agreed,
            "scala_runs": verdict.scala_runs}


def read_journal(path):
    """Rows saved so far by an interrupted validation. The last line may be cut off by a power cut, and pairs that hit
    an unexpected error are tried again."""
    rows = {}
    if path.exists():
        for line in path.read_text().split("\n"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row["reason"].startswith("error:"):
                rows[row["table_id"]] = row
    return rows


def step_validate(path, pairs, arm, settings, workers=VALIDATION_WORKERS, validate_one=_validate_one):
    if path.exists():
        return read_jsonl(path)
    if arm == "no_validation":
        rows = [{**p, "keep": bool(p["sql"] and p["scala"]), "reason": "not_validated", "samples_agreed": 0,
                 "scala_runs": 0} for p in pairs]
    else:
        # Each finished pair is saved at once, so that a power cut loses only the pairs running at that moment.
        journal_path = Path(path).with_suffix(".journal")
        done = read_journal(journal_path)
        lock = threading.Lock()
        with open(journal_path, "a") as journal:
            journal.write("\n")  # ends a line a power cut may have cut off

            def validate_and_save(pair):
                row = validate_one(pair, settings)
                with lock:
                    journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                return row

            remaining = [p for p in pairs if p["table_id"] not in done]
            with ThreadPoolExecutor(workers) as pool:
                for row in tqdm(pool.map(validate_and_save, remaining), total=len(remaining)):
                    done[row["table_id"]] = row
        rows = [done[p["table_id"]] for p in pairs]
    write_jsonl(path, rows)
    if arm != "no_validation":
        Path(path).with_suffix(".journal").unlink()
    return rows


def as_chat(example):
    """TRL's conversational prompt-completion form: training sees the same chat template and system message as serving."""
    return {"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": example["prompt"]}],
            "completion": [{"role": "assistant", "content": example["completion"]}]}


def step_examples(sql_path, scala_path, kept):
    """Paper Section 5: when a pair is validated, both (tG, SQL) and (tG, Scala) become training examples."""
    if sql_path.exists() and scala_path.exists():
        return read_jsonl(sql_path), read_jsonl(scala_path)
    sql_generator, scala_generator = repo_prompt_generator(), repo_scala_prompt_generator()
    sql_rows, scala_rows = [], []
    for pair in kept:
        table = load_table(pair["table_id"])
        # The repo's Scala task takes its label as (incorrect code, code); with no incorrect code it is a plain block.
        for generator, label, rows in ((sql_generator, pair["sql"], sql_rows),
                                       (scala_generator, (None, pair["scala"]), scala_rows)):
            example = build_example(generator, table, pair["question"], label)
            if example:
                rows.append(as_chat(example) if PROFILE == "instruct" else example)
    write_jsonl(sql_path, sql_rows)
    write_jsonl(scala_path, scala_rows)
    return sql_rows, scala_rows


class Server:
    """vLLM on GPU 0, optionally with the previous round's adapters. Bound to localhost only."""

    def __init__(self, adapters, log_path, lane=Lane(), model_path=MODEL_PATH, model_name=MODEL_NAME):
        self.adapters, self.log_path, self.lane = adapters, log_path, lane
        self.model_path, self.model_name = model_path, model_name  # another model only for Extension 3's repairer

    def __enter__(self):
        command = [str(ROOT / ".venv/bin/vllm"), "serve", self.model_path, "--served-model-name", self.model_name,
                   "--host", "127.0.0.1", "--port", str(self.lane.port), "--dtype", "bfloat16",
                   "--max-model-len", str(SERVER_CONTEXT), "--gpu-memory-utilization", "0.92", "--seed", "0"]
        if self.adapters:
            command += ["--enable-lora", "--max-lora-rank", "16", "--lora-modules",
                        *[f"{name}={path}" for name, path in self.adapters.items()]]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.lane.serve_gpu), "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        self.log = open(self.log_path, "w")
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(240):
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup; see {self.log_path}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.lane.port}/health", timeout=2)
                return self
            except OSError:
                time.sleep(5)
        raise RuntimeError(f"vLLM not ready after 20 minutes; see {self.log_path}")

    def __exit__(self, *_):
        self.process.terminate()
        self.process.wait(timeout=120)
        self.log.close()


def train_adapter(examples_path, output_dir, seed, log_path, lane=Lane(), max_steps=None):
    """Train in a subprocess, so that its memory is freed when it exits. max_steps is for smoke tests only."""
    if (Path(output_dir) / "train_summary.json").exists():
        return json.loads((Path(output_dir) / "train_summary.json").read_text())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(lane.train_gpu)}
    with open(log_path, "a") as log:  # a resumed run keeps the log of the interrupted one
        subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "ts.train.sft", "--model", MODEL_PATH,
                        "--examples", str(examples_path), "--output", str(output_dir), "--seed", str(seed),
                        *(["--max-steps", str(max_steps)] if max_steps else [])],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((Path(output_dir) / "train_summary.json").read_text())


def eval_run_id(run_id, round_number, benchmark, prompt):
    return f"{run_id}_round{round_number}_{BENCHMARKS[benchmark]}_{prompt.replace('-', '')}"


def step_evaluate(run_id, round_number, sql_adapter, round_dir, server=None, runner=subprocess.run, lane=Lane(),
                  limit=None):
    """Score the round's SQL adapter on each test set with both prompts: zero-shot is the repo's default, and few-shot
    is reported beside it (see docs/DECISIONS.md). A finished score is not repeated."""
    jobs = [(b, p, eval_run_id(run_id, round_number, b, p)) for b in BENCHMARKS for p in EVAL_PROMPTS]
    results = ROOT / "results/baseline"
    missing = [job for job in jobs if not (results / f"{job[2]}.json").exists()]
    if missing:
        name = adapter_name("sql", sql_adapter)
        with (server or Server)({name: str(sql_adapter)}, round_dir / "server_eval.log", lane):
            for benchmark, prompt, eval_id in missing:
                with open(round_dir / "eval.log", "a") as log:
                    runner([str(ROOT / ".venv/bin/python"), "-m", "ts.eval.run_baseline", "--benchmark", benchmark,
                            "--model", name, "--run-id", eval_id, "--api", API, *(["--repo-system-message"] if SYSTEM else []),
                            "--max-tokens", str(GENERATION_LIMITS["sql"]), "--prompt", prompt, "--base-url", lane.url,
                            *(["--limit", str(limit)] if limit else [])],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    scores = {}
    for benchmark, prompt, eval_id in jobs:
        result = json.loads((results / f"{eval_id}.json").read_text())
        scores[f"{BENCHMARKS[benchmark]}_{prompt}"] = {"repo": result["repo_faithful_acc"], "corrected": result["corrected_acc"]}
    return scores


def summarise(questions, sql_rows, scala_rows, validated):
    return {"tables": len(questions), "questions": sum(bool(q["question"]) for q in questions),
            "sql_extracted": sum(bool(r["code"]) for r in sql_rows),
            "scala_extracted": sum(bool(r["code"]) for r in scala_rows),
            "kept": sum(r["keep"] for r in validated),
            "reasons": {reason: sum(r["reason"] == reason for r in validated)
                        for reason in sorted({r["reason"] for r in validated})},
            "scala_runs": sum(r["scala_runs"] for r in validated)}


def _hold(slot, lane):
    """With a GPU pool (slot), take a free GPU for one step; otherwise use the run's fixed lane."""
    from contextlib import nullcontext
    return slot() if slot else nullcontext(lane)


def run_round(arm, seed, round_number, per_round, run_dir, config, previous, evaluate=True, lane=Lane(),
              eval_limit=None, slot=None):
    """slot: optional GPU pool (ts.repair_loop.gpu_slot) for several runs sharing both GPUs. With it, validation also
    takes the shared validation lock, so that only one run uses the 36 Spark workers at a time."""
    round_dir = run_dir / f"round{round_number}"
    round_dir.mkdir(parents=True, exist_ok=True)
    if (round_dir / "summary.json").exists():  # finished: a re-run must not replace its timings or its summary
        return ({language: str(round_dir / f"adapter_{language}") for language in ("sql", "scala")},
                json.loads((round_dir / "summary.json").read_text()))
    resumed = any(round_dir.iterdir())  # if so, timings cover only the work done after the restart
    table_ids = [line.split("\t")[0] for line in (ROOT / "results/ids/corpus.txt").read_text().splitlines()]
    tables = round_tables(table_ids, seed, round_number, per_round)
    timings = {}

    start = time.perf_counter()
    needs_generation = not all((round_dir / name).exists() for name in ("questions.jsonl", "sql.jsonl", "scala.jsonl"))
    served = {language: adapter_name(language, path) for language, path in (previous or {}).items()}
    questions, sql_rows, scala_rows = None, None, None
    if needs_generation:
        with _hold(slot, lane) as gen_lane, Server({served[language]: path for language, path in (previous or {}).items()},
                                                   round_dir / "server.log", gen_lane):
            lane_url = gen_lane.url
            vanilla = LLMClient(MODEL_NAME, base_url=lane_url)
            questions = step_questions(round_dir / "questions.jsonl", tables, seed, round_number, vanilla,
                                       config["generation"]["question_temperature"])
            for language in ("sql", "scala"):
                generator = LLMClient(served.get(language, MODEL_NAME), base_url=lane_url)
                rows = step_code(round_dir / f"{language}.jsonl", questions, language, generator,
                                 config["generation"]["code_temperature"], seed)
                if language == "sql":
                    sql_rows = rows
                else:
                    scala_rows = rows
    questions = questions or read_jsonl(round_dir / "questions.jsonl")
    sql_rows = sql_rows or read_jsonl(round_dir / "sql.jsonl")
    scala_rows = scala_rows or read_jsonl(round_dir / "scala.jsonl")
    timings["generation_s"] = round(time.perf_counter() - start)

    start = time.perf_counter()
    from contextlib import nullcontext
    from ts import ext2
    needs_lock = slot and arm != "no_validation" and not (round_dir / "validated.jsonl").exists()
    with ext2.validation_slot() if needs_lock else nullcontext():
        validated = step_validate(round_dir / "validated.jsonl", pair_code(sql_rows, scala_rows), arm,
                                  config["validation"])
    timings["validation_s"] = round(time.perf_counter() - start)

    start = time.perf_counter()
    kept = [r for r in validated if r["keep"]]
    step_examples(round_dir / "train_sql.jsonl", round_dir / "train_scala.jsonl", kept)
    new_adapters = {}
    summaries = {}
    for language in ("sql", "scala"):
        path = round_dir / f"adapter_{language}"
        with _hold(slot, lane) as train_lane:
            summaries[language] = train_adapter(round_dir / f"train_{language}.jsonl", path, seed,
                                                round_dir / f"train_{language}.log", train_lane)
        new_adapters[language] = str(path)
    timings["training_s"] = round(time.perf_counter() - start)

    start = time.perf_counter()
    scores = None
    if evaluate:
        with _hold(slot, lane) as eval_lane:
            scores = step_evaluate(run_dir.name, round_number, new_adapters["sql"], round_dir, lane=eval_lane,
                                   limit=eval_limit)
    timings["evaluation_s"] = round(time.perf_counter() - start)

    summary = {"arm": arm, "seed": seed, "round": round_number, **summarise(questions, sql_rows, scala_rows, validated),
               "train": summaries, "test_scores": scores, "timings": timings, "resumed": resumed}
    (round_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return new_adapters, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--tables-per-round", type=int, default=None)
    ap.add_argument("--tag", default="", help="Added to the run ID, for example test300")
    ap.add_argument("--no-eval", action="store_true", help="Skip scoring each round's adapter on the test sets")
    ap.add_argument("--eval-limit", type=int, default=None, help="Score only this many cases per test set; smoke tests only")
    ap.add_argument("--serve-gpu", type=int, default=0)
    ap.add_argument("--train-gpu", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-pool", action="store_true", help="take whichever GPU is free for each GPU step")
    args = ap.parse_args()
    lane = Lane(args.serve_gpu, args.train_gpu, args.port)

    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    rounds = args.rounds or config["study"]["rounds"]
    per_round = args.tables_per_round or config["study"]["tables_per_round"]
    arm_id = args.arm.replace("_", "")
    run_id = f"train_{arm_id}_{PROFILE}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    check_run_id(run_id)
    run_dir = ROOT / "runs" / run_id

    previous = None
    gpu_slot = None
    if args.gpu_pool:
        from ts.repair_loop import gpu_slot
    for round_number in range(1, rounds + 1):
        already_finished = (run_dir / f"round{round_number}" / "summary.json").exists()
        previous, summary = run_round(args.arm, args.seed, round_number, per_round, run_dir, config, previous,
                                      evaluate=not args.no_eval, lane=lane, eval_limit=args.eval_limit,
                                      slot=gpu_slot if args.gpu_pool else None)
        if already_finished:  # already logged, and already in MLflow
            continue
        print("ROUND " + json.dumps(summary), flush=True)
        log_run(f"{run_id}_round{round_number}", "fine_tuning",
                {"arm": args.arm, "seed": args.seed, "round": round_number, "tables": per_round}, summary)


if __name__ == "__main__":
    main()
