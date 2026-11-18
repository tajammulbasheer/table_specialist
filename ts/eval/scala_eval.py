"""Phase 8: NL-to-Spark-Scala test scoring, following the repo's Evaluete/evaluate/nss_evaluator.py.

Two phases, so that GPU and CPU work never wait on each other (docs/DECISIONS.md):
  generate  one vLLM server per model (GPU pool): Scala completions for every test set and prompt -> gen.jsonl
  score     under the Spark validation lock (36 workers): run each unique (code, table) once, then score every cell

Rules, as in nss_evaluator.py:
  - prompt: zero-shot from the repo's NSS generator (the format the adapters were trained on), few-shot from
    nl2sparkscala_generation_v6.py; cases over the generator's length limit are dropped, as the repo does;
  - code: the last ```scala block, or the whole completion if it starts with "val";
  - the code must read "data.csv" and write `.write.option("header", "true").csv("result")`, else it is not run;
  - Spark runs on the FULL table (the repo's csv_path); for BIRD the prompt shows 10 rows, so the table is exported
    from the case database once, into bench/.../bird_csv/;
  - gold: the gold SQL run on the case database (WikiTQ: the answer list), normalised like the repo's;
  - repo-faithful: the repo's exact list equality, row order included, after to_numeric and fillna("None"); empty
    results are equal to each other;
  - corrected: the same executed results, compared with ts.eval.compare.results_equal (order only with ORDER BY), and
    Spark's \\" quote escaping read correctly.
A verdict depends only on the code and the table, so identical (code, table) pairs run once and are cached in
runs/scala_exec_cache/, shared by all models. Generated code runs only in the repo's Docker sandbox.

  TS_PROFILE=instruct python -m ts.eval.scala_eval generate --final-only
  TS_PROFILE=instruct python -m ts.eval.scala_eval score --final-only --prompts zero-shot
  TS_PROFILE=instruct python -m ts.eval.scala_eval generate --smoke 10     # smoke_ run IDs, 10 cases per test set
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from ts import loop
from ts.eval.compare import EMPTY_RESULTS, gold_is_ordered, results_equal
from ts.eval.run_baseline import load_cases, run_sql, wilson_interval
from ts.gen.prompts import nl2scala_prompt
from ts.llm.client import LLMClient
from ts.repo import BENCH_ROOT, ROOT, load_sandbox, repo_scala_prompt_generator, to_numeric_or_keep
from ts.tracking import check_run_id, log_run

# runs/scala_exec_cache/ (first smoke test, 2026-11-14) holds only records of a missing sandbox setting; it is never read.
EXEC_CACHE = ROOT / "runs" / "nlscala_exec_cache"
GOLD_CACHE = ROOT / "runs" / "nlscala_gold"
BIRD_CSV = BENCH_ROOT / "bird_csv"
WRITE_SUFFIX = '.write.option("header", "true").csv("result")'
# Paper Table 3, Llama3.1-8B, NL-to-Scala, vanilla -> Table-Specialist. Read only from a summary of the paper's
# tables, not verbatim: unverified (docs/RESULTS.md, "Other tasks in the paper").
PAPER_UNVERIFIED = {"WikiSQL": (0.359, 0.728), "WikiQA": (0.043, 0.188), "spider": (0.249, 0.356), "bird": (0.100, 0.228)}
ROW_LIMIT_SAVED = 50  # rows of each result kept in preds.jsonl


# ---------------------------------------------------------------- models and run IDs

def models(final_only=False):
    """(name, adapter or None) for Instruct: vanilla, then TS and repair-TS, seeds 1-3, rounds 1-3."""
    if loop.PROFILE != "instruct":
        raise SystemExit("Phase 8 is Instruct only: set TS_PROFILE=instruct")
    out = [("instruct_vanilla", None)]
    for arm in ("tablespecialist", "repairts"):
        for seed in (1, 2, 3):
            for r in ((3,) if final_only else (1, 2, 3)):
                out.append((f"instruct_{arm}_seed{seed}_round{r}",
                            ROOT / f"runs/train_{arm}_instruct_seed{seed}/round{r}/adapter_scala"))
    return out


def adapter_ready(adapter):
    return adapter is None or (adapter / "train_summary.json").exists()


def run_id(prefix, model, benchmark, prompt, tag=""):
    return f"{prefix}_nlscala{tag}_{model}_{loop.BENCHMARKS[benchmark]}_{prompt.replace('-', '')}"


# ---------------------------------------------------------------- code extraction (the repo's _get_pred)

def extract_scala(completion):
    import re
    matches = re.findall(r"```scala(.*?)```", completion or "", re.DOTALL)
    if not matches:
        return completion if (completion or "").lower().strip().startswith("val") else None
    return matches[-1].strip()


def format_ok(code):
    return bool(code) and "data.csv" in code and WRITE_SUFFIX in code


# ---------------------------------------------------------------- tables and gold

def exec_table(benchmark, case):
    """The CSV Spark reads: the full table. Only BIRD's prompt table is a sample."""
    if benchmark != "bird":
        return BENCH_ROOT / benchmark / case["id"] / "data.csv"
    db = Path(case["info"]["sqlite_path"])
    path = BIRD_CSV / f"{db.stem}.csv"
    if not path.exists():
        BIRD_CSV.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            frame = pd.read_sql('SELECT * FROM "table"', conn)
        finally:
            conn.close()
        tmp = path.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.rename(path)
    return path


def gold_db(benchmark, case):
    if benchmark == "WikiSQL":
        return BENCH_ROOT / "WikiSQL_tables" / f"{case['info']['table_id']}.db"
    if benchmark == "spider":
        return BENCH_ROOT / "spider" / case["id"] / "data.sqlite"
    if benchmark == "bird":
        return Path(case["info"]["sqlite_path"])
    return None


def compute_gold(benchmark, case):
    label = case["info"]["label"]
    if benchmark == "WikiQA":  # an answer list, not SQL
        return [[to_numeric_or_keep(x)] for x in label.split("|")]
    return run_sql(gold_db(benchmark, case), label)  # read-only, the repo's normalisation; "SQLExecutionError" on failure


def gold_results(benchmark, cases):
    """Gold rows by case ID, computed once on one thread (BIRD's large results time out under many threads)."""
    path = GOLD_CACHE / f"{benchmark}.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    missing = [c for c in cases if c["id"] not in cached]
    for case in missing:
        cached[case["id"]] = compute_gold(benchmark, case)
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, default=_json_default))
        cached = json.loads(path.read_text())  # the same JSON types on every read
    return cached


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


# ---------------------------------------------------------------- Spark execution with a shared cache

def exec_key(code, table):
    return hashlib.sha256(f"{code}\0{Path(table).resolve()}".encode()).hexdigest()


def cache_path(key):
    return EXEC_CACHE / key[:2] / f"{key}.json"


def execute(code, table, runner=None):
    """Run one program in a fresh sandbox container on a copy of the table. Returns the cache record."""
    sandbox = load_sandbox()
    runner = runner or sandbox.run_generated_code
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nlscala_") as tmp:
        work = Path(tmp)
        shutil.copyfile(table, work / "data.csv")
        (work / "query.scala").write_text(code)
        try:
            result = runner(str(work), "scala", "query.scala", num_cpu=1)
            returncode = result.returncode
            stderr_tail = (getattr(result, "stderr", "") or "")[-1500:]  # for diagnosis only; never read by scoring
        except sandbox.SandboxError as error:
            return {"ok": False, "returncode": None, "n_csv": 0, "csv": None, "error": str(error)[:500],
                    "seconds": round(time.perf_counter() - start, 2)}
        out = work / "result_scala"
        parts = sorted(out.glob("*.csv")) if out.exists() else []
        csv_text = parts[0].read_text(errors="replace") if len(parts) == 1 else None
    return {"ok": returncode == 0 and csv_text is not None, "returncode": returncode, "n_csv": len(parts),
            "csv": csv_text, "error": None, "stderr_tail": stderr_tail, "seconds": round(time.perf_counter() - start, 2)}


STALL_SECONDS = 600  # a Scala run is limited to 120 s plus a few 15 s control steps; far longer means the machine stalled


def is_stalled(record):
    """A timeout recorded while the whole machine stalled (2026-11-14, 19:37-03:03 IST): not a verdict on the code."""
    return record.get("error") == "Sandbox operation timed out" and (record.get("seconds") or 0) > STALL_SECONDS


def rerun_path(path):
    return path.with_name(path.stem + ".rerun.json")


def cached_record(key):
    """The cached record, or None if there is none to trust. A stalled record is kept on disk; its re-run is stored
    beside it (<key>.rerun.json) and read in its place."""
    path = cache_path(key)
    for candidate in (rerun_path(path), path):
        if candidate.exists():
            record = json.loads(candidate.read_text())
            return None if is_stalled(record) else record
    return None


def cached_execute(code, table, runner=None):
    key = exec_key(code, table)
    record = cached_record(key)
    if record is not None:
        return record, True
    path = cache_path(key)
    if path.exists():  # stalled: never overwrite the original record
        path = rerun_path(path)
    record = execute(code, table, runner)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record))
    tmp.rename(path)
    return record, False


# ---------------------------------------------------------------- scoring

def parse_rows(csv_text, escape_aware):
    """The repo's read: header row, default NA handling, to_numeric, fillna("None"). None means unreadable."""
    if csv_text is None:
        return None
    escape = "\\" if escape_aware and '\\"' in csv_text else None
    try:
        frame = pd.read_csv(io.StringIO(csv_text), index_col=False, header=0, escapechar=escape)
    except pd.errors.EmptyDataError:
        return []
    except Exception:
        return None
    return json.loads(json.dumps(frame.map(to_numeric_or_keep).astype(object).fillna("None").values.tolist(),
                                 default=_json_default))


def repo_equal(pred, gold):
    return pred == gold or (pred in EMPTY_RESULTS and gold in EMPTY_RESULTS)


def score_case(benchmark, case, code, gold, record):
    """Both verdicts for one case. record is the execution record, or None if the code was not run."""
    gold_failed = gold == "SQLExecutionError"
    if code is None:
        error = "no_code"
    elif not format_ok(code):
        error = "format_error"
    elif record is None or not record["ok"]:
        error = "exec_failed"
    else:
        error = None
    repo_pred = parse_rows(record["csv"], False) if error is None else None
    corrected_pred = parse_rows(record["csv"], True) if error is None else None
    repo = error is None and not gold_failed and repo_pred is not None and repo_equal(repo_pred, gold)
    ordered = benchmark != "WikiQA" and gold_is_ordered(case["info"]["label"])
    corrected = (error is None and not gold_failed and corrected_pred is not None
                 and results_equal(corrected_pred, gold, ordered=ordered))
    if error is None and corrected_pred is None:
        error = "unreadable_result"
    return {"id": case["id"], "question": case["info"]["question"], "gold": case["info"]["label"], "code": code,
            "repo_correct": int(repo), "corrected_correct": int(corrected),
            "error": error or ("correct" if corrected else "wrong"), "gold_failed": gold_failed,
            "pred_rows": (corrected_pred or [])[:ROW_LIMIT_SAVED] if corrected_pred is not None else None,
            "gold_rows": gold[:ROW_LIMIT_SAVED] if isinstance(gold, list) else gold}


# ---------------------------------------------------------------- phases

def prompts_for(benchmark, prompt_style, limit):
    """(usable cases, prompts, dropped IDs). The same cases for both prompt styles, as for SQL."""
    cases = load_cases(benchmark, limit)
    generator = repo_scala_prompt_generator()
    usable, prompts, dropped = [], [], []
    for case in cases:
        prompt = generator.generate_prompt((case["table"], case["info"]["question"], None), None)
        if prompt is None:
            dropped.append(case["id"])
            continue
        usable.append(case)
        prompts.append(prompt if prompt_style == "zero-shot" else nl2scala_prompt(case["table"], case["info"]["question"]))
    return usable, prompts, dropped


def generate(selected, prompt_styles, prefix, limit, tag=""):
    from ts.repair_loop import gpu_slot
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    for name, adapter in selected:
        jobs = [(b, p, run_id(prefix, name, b, p, tag)) for b in loop.BENCHMARKS for p in prompt_styles]
        missing = [j for j in jobs if not (ROOT / "runs" / j[2] / "gen.jsonl").exists()]
        if not missing:
            continue
        for _, _, rid in missing:
            check_run_id(rid)
        served = loop.adapter_name("scala", adapter) if adapter else loop.MODEL_NAME
        with gpu_slot() as lane, loop.Server({served: str(adapter)} if adapter else {},
                                             ROOT / f"logs/nlscala_server_{name}.log", lane):
            client = LLMClient(served, base_url=lane.url)
            for benchmark, prompt_style, rid in missing:
                usable, prompts, dropped = prompts_for(benchmark, prompt_style, limit)
                start = time.perf_counter()
                responses = client.complete_many(prompts, api=loop.API, system=loop.SYSTEM,
                                                 temperature=config["generation"]["code_temperature"],
                                                 max_tokens=loop.GENERATION_LIMITS["scala"], seed=0)
                seconds = round(time.perf_counter() - start, 1)
                run_dir = ROOT / "runs" / rid
                run_dir.mkdir(parents=True, exist_ok=True)
                rows = [{"id": c["id"], "prompt": p, "response": r, "code": extract_scala(r["text"])}
                        for c, p, r in zip(usable, prompts, responses)]
                meta = {"run_id": rid, "model": served, "adapter": str(adapter) if adapter else None,
                        "benchmark": benchmark, "prompt": prompt_style, "limit": limit, "dropped_too_long": dropped,
                        "api": loop.API, "system_message": loop.SYSTEM, "max_tokens": loop.GENERATION_LIMITS["scala"],
                        "generation_seconds": seconds}
                (run_dir / "gen_meta.json").write_text(json.dumps(meta, indent=1))
                loop.write_jsonl(run_dir / "gen.jsonl", rows)  # written last: its presence marks the cell done
                print(f"GEN {rid}: {len(rows)} cases, {sum(r['code'] is not None for r in rows)} with code, {seconds} s",
                      flush=True)


def score(selected, prompt_styles, prefix, limit, workers, tag=""):
    from ts.ext2 import validation_slot
    sandbox = load_sandbox()
    sandbox.ensure_sandbox("scala")  # a missing image setting must stop the run, not be cached as failed programs
    cells = []
    for name, _ in selected:
        for benchmark in loop.BENCHMARKS:
            for prompt_style in prompt_styles:
                rid = run_id(prefix, name, benchmark, prompt_style, tag)
                done = (ROOT / "results/baseline" / f"{rid}.json").exists()
                if (ROOT / "runs" / rid / "gen.jsonl").exists() and not done:
                    cells.append((name, benchmark, prompt_style, rid))
    if not cells:
        print("nothing to score", flush=True)
        return
    cases = {b: {c["id"]: c for c in load_cases(b, limit)} for b in {c[1] for c in cells}}
    gold = {b: gold_results(b, list(cases[b].values())) for b in cases}
    todo = {}
    for _, benchmark, _, rid in cells:
        for row in loop.read_jsonl(ROOT / "runs" / rid / "gen.jsonl"):
            if format_ok(row["code"]):
                table = exec_table(benchmark, cases[benchmark][row["id"]])
                todo.setdefault(exec_key(row["code"], table), (row["code"], table))
    new = {k: v for k, v in todo.items() if cached_record(k) is None}
    print(f"SCORE {len(cells)} cells, {len(todo)} unique programs, {len(new)} to run", flush=True)
    with validation_slot():
        start = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            for i, _ in enumerate(pool.map(lambda kv: cached_execute(*kv), new.values()), 1):
                if i % 200 == 0:
                    rate = i / (time.perf_counter() - start)
                    print(f"[{time.strftime('%H:%M:%S')} UTC] {i}/{len(new)} programs, {rate:.2f}/s", flush=True)
        seconds = time.perf_counter() - start
    throughput = {"programs_run": len(new), "seconds": round(seconds, 1),
                  "programs_per_s": round(len(new) / seconds, 3) if new and seconds else None, "workers": workers}
    print("THROUGHPUT " + json.dumps(throughput), flush=True)
    for name, benchmark, prompt_style, rid in cells:
        write_cell(name, benchmark, prompt_style, rid, cases[benchmark], gold[benchmark], throughput)


def write_cell(name, benchmark, prompt_style, rid, cases, gold, throughput):
    run_dir = ROOT / "runs" / rid
    meta = json.loads((run_dir / "gen_meta.json").read_text())
    rows = loop.read_jsonl(run_dir / "gen.jsonl")
    records = []
    with (run_dir / "preds.jsonl").open("w") as out:
        for row in rows:
            case = cases[row["id"]]
            record = None
            if format_ok(row["code"]):
                record, _ = cached_execute(row["code"], exec_table(benchmark, case))
            result = score_case(benchmark, case, row["code"], gold[row["id"]], record)
            records.append(result)
            out.write(json.dumps({**result, "response": row["response"]}, ensure_ascii=False, default=str) + "\n")
    n = len(records)
    repo = sum(r["repo_correct"] for r in records)
    corrected = sum(r["corrected_correct"] for r in records)
    paper = PAPER_UNVERIFIED.get(benchmark)
    summary = {**{k: meta[k] for k in ("run_id", "model", "adapter", "benchmark", "prompt", "api", "system_message",
                                      "max_tokens", "dropped_too_long", "limit")},
               "task": "nl2scala", "split": "test", "n": n,
               "repo_faithful_acc": round(repo / n, 4), "repo_faithful_95ci": wilson_interval(repo, n),
               "corrected_acc": round(corrected / n, 4), "corrected_95ci": wilson_interval(corrected, n),
               "outcomes": dict(Counter(r["error"] for r in records)),
               "gold_failures": sum(r["gold_failed"] for r in records),
               "truncated_responses": sum(r["response"]["finish_reason"] == "length" for r in rows),
               "paper_unverified": {"vanilla": paper[0], "table_specialist": paper[1]} if paper else None,
               "generation_seconds": meta["generation_seconds"], "spark": throughput}
    out_file = ROOT / "results" / "baseline" / f"{rid}.json"
    out_file.write_text(json.dumps(summary, indent=1))
    log_run(rid, "nlscala", {"model": name, "benchmark": benchmark, "prompt": prompt_style, "limit": meta["limit"]},
            {k: summary[k] for k in ("n", "repo_faithful_acc", "corrected_acc")}, artifact=out_file)
    print("RESULT " + json.dumps({k: summary[k] for k in ("run_id", "n", "repo_faithful_acc", "corrected_acc",
                                                         "outcomes")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "score"])
    ap.add_argument("--final-only", action="store_true", help="vanilla and round 3 only")
    ap.add_argument("--only", nargs="+", help="model names, e.g. instruct_vanilla instruct_repairts_seed1_round3")
    ap.add_argument("--prompts", nargs="+", default=list(loop.EVAL_PROMPTS), choices=list(loop.EVAL_PROMPTS))
    ap.add_argument("--smoke", type=int, help="this many cases per test set, under smoke_ run IDs")
    ap.add_argument("--workers", type=int, default=loop.VALIDATION_WORKERS)
    ap.add_argument("--tag", default="", help="added after nlscala in the run IDs, e.g. r2")
    args = ap.parse_args()
    selected = [(n, a) for n, a in models(args.final_only) if not args.only or n in args.only]
    ready = [(n, a) for n, a in selected if adapter_ready(a)]
    for n, a in selected:
        if (n, a) not in ready:
            print(f"skipping {n}: {a} has not finished training", flush=True)
    prefix = "smoke" if args.smoke else "final"
    if args.stage == "generate":
        generate(ready, args.prompts, prefix, args.smoke, args.tag)
    else:
        score(ready, args.prompts, prefix, args.smoke, args.workers, args.tag)


if __name__ == "__main__":
    main()
