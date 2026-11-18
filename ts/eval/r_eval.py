"""Phase 9: NL-to-R test scoring, following the repo's Evaluete/evaluate/nr_evaluator.py.

As ts/eval/scala_eval.py (two phases: generate on the GPU pool, score under the CPU validation lock), with the
evaluator's R rules:
  - prompt: zero-shot from the repo's NR generator (Finetune/data_generator/nl_to_r, the format the adapters were
    trained on; its input is (table, question)); few-shot from Generation/NL2DSL/NL2R/nl2r_generation_v6.py; cases
    over the generator's length limit are dropped, as the repo does;
  - code: the FIRST ```r block (nr_evaluator._get_pred: matches[0]); no block means no code. There is no format check;
  - R runs in the repo's sandbox (runtime "r", query.R, 15 s) on the full table (the repo's csv_path; for BIRD the
    table exported from the case database, as for Scala);
  - gold: the gold SQL on the case database (WikiTQ: the answer list), shared with NL-to-Scala (runs/nlscala_gold);
  - repo-faithful: result.csv read as the repo reads it (header row, default NA handling, to_numeric,
    fillna("None")) and compared with the repo's exact list equality, row order included; empty results equal;
  - corrected: the same rows compared with ts.eval.compare.results_equal (order only with ORDER BY).
Identical (code, table) pairs run once, cached in runs/nlr_exec_cache/.

  TS_PROFILE=instruct TS_R_GPUS=0 python -m ts.eval.r_eval generate --only instruct_vanilla
  TS_PROFILE=instruct python -m ts.eval.r_eval score
  TS_PROFILE=instruct TS_R_GPUS=0 python -m ts.eval.r_eval generate --smoke 10 --only instruct_vanilla
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from ts import loop
from ts.eval.compare import gold_is_ordered, results_equal
from ts.eval.run_baseline import load_cases, wilson_interval
from ts.eval.scala_eval import ROW_LIMIT_SAVED, exec_table, gold_results, parse_rows, repo_equal
from ts.gen.prompts import nl2r_prompt
from ts.llm.client import LLMClient
from ts.repo import ROOT, load_sandbox, repo_r_prompt_generator
from ts.tracking import check_run_id, log_run

EXEC_CACHE = ROOT / "runs" / "nlr_exec_cache"
R_FILE = "query.R"
MAX_TOKENS = 512  # as ts.r_loop.R_MAX_TOKENS
# Paper Table 3, Llama3.1-8B, NL-to-R average vanilla -> Table-Specialist, read only from a summary of the paper's
# tables (docs/RESULTS.md, "Other tasks in the paper"): unverified, and no per-test-set numbers.
PAPER_UNVERIFIED_AVERAGE = (0.260, 0.333)
_BLOCK = re.compile(r"```r(.*?)```", re.DOTALL)  # nr_evaluator._get_pred


def models():
    if loop.PROFILE != "instruct":
        raise SystemExit("NL-to-R is Instruct only: set TS_PROFILE=instruct")
    out = [("instruct_vanilla", None)]
    for arm in ("rts", "rrepairts"):
        for seed in (1, 2, 3):
            for r in (1, 2, 3):
                out.append((f"instruct_{arm}_seed{seed}_round{r}", ROOT / f"runs/train_{arm}_instruct_seed{seed}/round{r}/adapter_r"))
    return out


def run_id(prefix, model, benchmark, prompt):
    return f"{prefix}_nlr_{model}_{loop.BENCHMARKS[benchmark]}_{prompt.replace('-', '')}"


def extract_r(completion):
    matches = _BLOCK.findall(completion or "")
    return matches[0].strip() if matches else None


# ---------------------------------------------------------------- execution with a shared cache

def exec_key(code, table):
    return hashlib.sha256(f"r\0{code}\0{Path(table).resolve()}".encode()).hexdigest()


def cache_path(key):
    return EXEC_CACHE / key[:2] / f"{key}.json"


def execute(code, table, runner=None):
    sandbox = load_sandbox()
    runner = runner or sandbox.run_generated_code
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nlr_") as tmp:
        work = Path(tmp)
        shutil.copyfile(table, work / "data.csv")
        (work / R_FILE).write_text(code)
        try:
            result = runner(str(work), "r", R_FILE)
        except sandbox.SandboxError as error:
            return {"ok": False, "returncode": None, "csv": None, "error": str(error)[:500],
                    "seconds": round(time.perf_counter() - start, 2)}
        out = work / "result.csv"
        csv_text = out.read_text(errors="replace") if out.exists() else None
        stderr_tail = (getattr(result, "stderr", "") or "")[-1500:]
    return {"ok": result.returncode == 0 and csv_text is not None, "returncode": result.returncode, "csv": csv_text,
            "error": None, "stderr_tail": stderr_tail, "seconds": round(time.perf_counter() - start, 2)}


def cached_execute(code, table, runner=None):
    path = cache_path(exec_key(code, table))
    if path.exists():
        return json.loads(path.read_text()), True
    record = execute(code, table, runner)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record))
    tmp.rename(path)
    return record, False


# ---------------------------------------------------------------- scoring

def score_case(benchmark, case, code, gold, record):
    gold_failed = gold == "SQLExecutionError"
    if code is None:
        error = "no_code"
    elif record is None or not record["ok"]:
        error = "exec_failed"
    else:
        error = None
    pred = parse_rows(record["csv"], False) if error is None else None
    repo = error is None and not gold_failed and pred is not None and repo_equal(pred, gold)
    ordered = benchmark != "WikiQA" and gold_is_ordered(case["info"]["label"])
    corrected = error is None and not gold_failed and pred is not None and results_equal(pred, gold, ordered=ordered)
    if error is None and pred is None:
        error = "unreadable_result"
    return {"id": case["id"], "question": case["info"]["question"], "gold": case["info"]["label"], "code": code,
            "repo_correct": int(repo), "corrected_correct": int(corrected),
            "error": error or ("correct" if corrected else "wrong"), "gold_failed": gold_failed,
            "pred_rows": pred[:ROW_LIMIT_SAVED] if pred is not None else None,
            "gold_rows": gold[:ROW_LIMIT_SAVED] if isinstance(gold, list) else gold}


def prompts_for(benchmark, prompt_style, limit):
    cases = load_cases(benchmark, limit)
    generator = repo_r_prompt_generator()
    usable, prompts, dropped = [], [], []
    for case in cases:
        prompt = generator.generate_prompt((case["table"], case["info"]["question"]), None)
        if prompt is None:
            dropped.append(case["id"])
            continue
        usable.append(case)
        prompts.append(prompt if prompt_style == "zero-shot" else nl2r_prompt(case["table"], case["info"]["question"]))
    return usable, prompts, dropped


def generate(selected, prompt_styles, prefix, limit):
    from ts.r_loop import gpu_slot
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    for name, adapter in selected:
        jobs = [(b, p, run_id(prefix, name, b, p)) for b in loop.BENCHMARKS for p in prompt_styles]
        missing = [j for j in jobs if not (ROOT / "runs" / j[2] / "gen.jsonl").exists()]
        if not missing:
            continue
        for _, _, rid in missing:
            check_run_id(rid)
        served = loop.adapter_name("r", adapter) if adapter else loop.MODEL_NAME
        with gpu_slot() as lane, loop.Server({served: str(adapter)} if adapter else {},
                                             ROOT / f"logs/nlr_server_{name}.log", lane):
            client = LLMClient(served, base_url=lane.url)
            for benchmark, prompt_style, rid in missing:
                usable, prompts, dropped = prompts_for(benchmark, prompt_style, limit)
                start = time.perf_counter()
                responses = client.complete_many(prompts, api=loop.API, system=loop.SYSTEM,
                                                 temperature=config["generation"]["code_temperature"],
                                                 max_tokens=MAX_TOKENS, seed=0)
                seconds = round(time.perf_counter() - start, 1)
                run_dir = ROOT / "runs" / rid
                run_dir.mkdir(parents=True, exist_ok=True)
                rows = [{"id": c["id"], "prompt": p, "response": r, "code": extract_r(r["text"])}
                        for c, p, r in zip(usable, prompts, responses)]
                meta = {"run_id": rid, "model": served, "adapter": str(adapter) if adapter else None,
                        "benchmark": benchmark, "prompt": prompt_style, "limit": limit, "dropped_too_long": dropped,
                        "api": loop.API, "system_message": loop.SYSTEM, "max_tokens": MAX_TOKENS,
                        "generation_seconds": seconds}
                (run_dir / "gen_meta.json").write_text(json.dumps(meta, indent=1))
                loop.write_jsonl(run_dir / "gen.jsonl", rows)  # written last: its presence marks the cell generated
                print(f"GEN {rid}: {len(rows)} cases, {sum(r['code'] is not None for r in rows)} with code, {seconds} s",
                      flush=True)


def score(selected, prompt_styles, prefix, limit, workers):
    from ts.ext2 import validation_slot
    load_sandbox().ensure_sandbox("r")  # a missing image setting must stop the run, not be cached as failures
    cells = []
    for name, _ in selected:
        for benchmark in loop.BENCHMARKS:
            for prompt_style in prompt_styles:
                rid = run_id(prefix, name, benchmark, prompt_style)
                if (ROOT / "runs" / rid / "gen.jsonl").exists() and not (ROOT / "results/baseline" / f"{rid}.json").exists():
                    cells.append((name, benchmark, prompt_style, rid))
    if not cells:
        print("nothing to score", flush=True)
        return
    cases = {b: {c["id"]: c for c in load_cases(b, limit)} for b in {c[1] for c in cells}}
    gold = {b: gold_results(b, list(cases[b].values())) for b in cases}
    todo = {}
    for _, benchmark, _, rid in cells:
        for row in loop.read_jsonl(ROOT / "runs" / rid / "gen.jsonl"):
            if row["code"]:
                table = exec_table(benchmark, cases[benchmark][row["id"]])
                todo.setdefault(exec_key(row["code"], table), (row["code"], table))
    new = {k: v for k, v in todo.items() if not cache_path(k).exists()}
    print(f"SCORE {len(cells)} cells, {len(todo)} unique programs, {len(new)} to run", flush=True)
    with validation_slot():
        start = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            for i, _ in enumerate(pool.map(lambda kv: cached_execute(*kv), new.values()), 1):
                if i % 500 == 0:
                    print(f"[{time.strftime('%H:%M:%S')} UTC] {i}/{len(new)} programs, "
                          f"{i / (time.perf_counter() - start):.2f}/s", flush=True)
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
            record = cached_execute(row["code"], exec_table(benchmark, case))[0] if row["code"] else None
            result = score_case(benchmark, case, row["code"], gold[row["id"]], record)
            records.append(result)
            out.write(json.dumps({**result, "response": row["response"]}, ensure_ascii=False, default=str) + "\n")
    n = len(records)
    repo = sum(r["repo_correct"] for r in records)
    corrected = sum(r["corrected_correct"] for r in records)
    summary = {**{k: meta[k] for k in ("run_id", "model", "adapter", "benchmark", "prompt", "api", "system_message",
                                      "max_tokens", "dropped_too_long", "limit")},
               "task": "nl2r", "split": "test", "n": n,
               "repo_faithful_acc": round(repo / n, 4), "repo_faithful_95ci": wilson_interval(repo, n),
               "corrected_acc": round(corrected / n, 4), "corrected_95ci": wilson_interval(corrected, n),
               "outcomes": dict(Counter(r["error"] for r in records)),
               "gold_failures": sum(r["gold_failed"] for r in records),
               "truncated_responses": sum(r["response"]["finish_reason"] == "length" for r in rows),
               "paper_unverified_average": {"vanilla": PAPER_UNVERIFIED_AVERAGE[0],
                                            "table_specialist": PAPER_UNVERIFIED_AVERAGE[1]},
               "generation_seconds": meta["generation_seconds"], "r_sandbox": throughput}
    out_file = ROOT / "results" / "baseline" / f"{rid}.json"
    out_file.write_text(json.dumps(summary, indent=1))
    if not rid.startswith("smoke"):
        log_run(rid, "nl2r", {"model": name, "benchmark": benchmark, "prompt": prompt_style, "limit": meta["limit"]},
                {k: summary[k] for k in ("n", "repo_faithful_acc", "corrected_acc")}, artifact=out_file)
    print("RESULT " + json.dumps({k: summary[k] for k in ("run_id", "n", "repo_faithful_acc", "corrected_acc",
                                                         "outcomes")}), flush=True)


def adapter_ready(adapter):
    return adapter is None or (adapter / "train_summary.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "score"])
    ap.add_argument("--only", nargs="+", help="model names, e.g. instruct_vanilla instruct_rts_seed1_round3")
    ap.add_argument("--prompts", nargs="+", default=list(loop.EVAL_PROMPTS), choices=list(loop.EVAL_PROMPTS))
    ap.add_argument("--smoke", type=int, help="this many cases per test set, under smoke_ run IDs")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TS_R_WORKERS", loop.VALIDATION_WORKERS)))
    args = ap.parse_args()
    selected = [(n, a) for n, a in models() if (not args.only or n in args.only)]
    ready = [(n, a) for n, a in selected if adapter_ready(a)]
    for n, a in selected:
        if (n, a) not in ready:
            print(f"skipping {n}: {a} has not finished training", flush=True)
    prefix = "smoke" if args.smoke else "final"
    if args.stage == "generate":
        generate(ready, args.prompts, prefix, args.smoke)
    else:
        score(ready, args.prompts, prefix, args.smoke, args.workers)


if __name__ == "__main__":
    main()
