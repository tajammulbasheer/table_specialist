"""Evaluate a model on a rebuilt NL-to-SQL test set and report repo-faithful and corrected scores.

Needs a running vLLM server (scripts/run_baseline.sh starts one). Outputs:
  runs/<run_id>/preds.jsonl   every prompt, response and both verdicts (git-ignored)
  results/baseline/<run_id>.json   the summary (tracked)
"""
import argparse
import contextlib
import io
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from ts.eval.compare import gold_is_ordered, results_equal
from ts.gen.prompts import nl2sql_prompt
from ts.llm.client import LLMClient
from ts.tracking import check_run_id, log_run
from ts.repo import (BENCH_ROOT, REPO_SYSTEM_MESSAGE, ROOT, connect_read_only, load_ns_module, repo_prompt_generator,
                     to_numeric_or_keep)

PAPER_VANILLA = {"WikiSQL": 0.525, "spider": 0.670, "WikiQA": 0.300, "bird": 0.388}  # paper Table 3, Llama3.1-8B Vanilla
GATE_MARGIN = 0.05


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return [round(centre - half, 3), round(centre + half, 3)]


SPLITS = {"test": (BENCH_ROOT, ROOT / "results" / "ids"),
          "dev": (BENCH_ROOT.parent / "dev", ROOT / "results" / "ids" / "dev")}  # dev: extension 2 tuning only


def load_cases(benchmark, limit, table_file="data.csv", split="test"):
    bench_root, ids_dir = SPLITS[split]
    ids_file = ids_dir / f"{benchmark.lower()}.txt"
    ids = ids_file.read_text().split()
    cases = []
    for case_id in ids[:limit] if limit else ids:
        case_dir = bench_root / benchmark / case_id
        info = json.loads((case_dir / "info.json").read_text())
        # Same read as the repo's BaseDataGenerator.load_df.
        table = pd.read_csv(case_dir / table_file, keep_default_na=False)
        cases.append({"id": case_id, "info": info, "table": table})
    return cases


def keep_first_code_block(text):
    """Cut a raw completion after its first SQL block. A base model keeps writing after it, and the repo's
    parser takes the last block, so without this it would score whatever the model rambled on to."""
    match = re.search(r"```sql.*?```", text, re.DOTALL)
    return match.group(0) if match else text


def run_sql(db_path, sql):
    """Execute SQL with the same result normalisation as the repo's patched evaluator."""
    conn = connect_read_only(db_path)
    try:
        return pd.read_sql(sql, conn).map(to_numeric_or_keep).astype(object).fillna("None").values.tolist()
    except Exception:
        return "SQLExecutionError"
    finally:
        conn.close()


def score_case(evaluator, case, completion, benchmark):
    """Return one record holding the repo's verdict and the corrected verdict on the same executed results."""
    gold = evaluator._get_gt(case["info"]["label"])
    predicted = evaluator._get_pred(completion)
    with contextlib.redirect_stdout(io.StringIO()):
        row = evaluator._evaluate_row(gold, predicted, case["id"], benchmark=benchmark)
    pred_result, repo_gold_result = row["y_pred_rst"], row["y_true_rst"]
    # The repo lowercases gold SQL, which breaks non-ASCII column names: SQLite then reads "álbum" as a string.
    # Our gold SQL already holds lowercased values, so the corrected score runs it exactly as written.
    # WikiTQ's gold is an answer list, not SQL, so there is no query to run.
    gold_result = repo_gold_result if benchmark == "WikiQA" else run_sql(row["db_path"], gold)
    executed = isinstance(pred_result, list) and isinstance(gold_result, list)
    corrected = executed and results_equal(pred_result, gold_result, ordered=gold_is_ordered(gold))
    return {"id": case["id"], "question": case["info"]["question"], "gold_sql": gold, "pred_sql": predicted,
            "repo_correct": int(row["correct"]), "corrected_correct": int(corrected), "error": row["error"],
            "gold_failed": gold_result == "SQLExecutionError",
            "repo_gold_differs": repo_gold_result != gold_result,
            "gold_result": gold_result, "pred_result": pred_result, "db_path": row["db_path"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="WikiSQL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", required=True, help="Name the vLLM server serves the model under")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--api", choices=["chat", "completions"], default="chat")
    ap.add_argument("--repo-system-message", action="store_true", help="Send the system message the repo sends")
    ap.add_argument("--prompt", choices=["zero-shot", "few-shot"], default="zero-shot",
                    help="zero-shot: main_build_data.py's test prompt; few-shot: nl2sql_generation_v6.py's prompt")
    ap.add_argument("--table-file", default="data.csv", help="Table shown in the prompt; SQL always runs on the database")
    ap.add_argument("--split", choices=sorted(SPLITS), default="test", help="dev is for extension 2 tuning only")
    args = ap.parse_args()
    check_run_id(args.run_id)  # fail before spending any GPU time on a badly named run
    system = REPO_SYSTEM_MESSAGE if args.repo_system_message else None

    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    cases = load_cases(args.benchmark, args.limit, args.table_file, args.split)
    generator = repo_prompt_generator()
    prompts, usable, too_long = [], [], []
    for case in cases:
        prompt = generator.generate_prompt((case["table"], case["info"]["question"], None), None)
        if prompt is None:  # over the repo's limit of 70% of 4,096 tokens; the repo drops these too
            too_long.append(case["id"])
        else:
            # The same cases stay in for every prompt style, so that scores remain comparable.
            prompts.append(nl2sql_prompt(case["table"], case["info"]["question"]) if args.prompt == "few-shot" else prompt)
            usable.append(case)

    client = LLMClient(args.model, base_url=args.base_url, concurrency=args.concurrency)
    start = time.perf_counter()
    responses = client.complete_many(prompts, api=args.api, system=system,
                                     temperature=config["generation"]["code_temperature"],
                                     max_tokens=args.max_tokens, seed=0)
    generation_seconds = time.perf_counter() - start

    evaluator, _ = load_ns_module(SPLITS[args.split][0])
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (run_dir / "preds.jsonl").open("w") as out:
        for case, prompt, response in zip(usable, prompts, responses):
            text = keep_first_code_block(response["text"]) if args.api == "completions" else response["text"]
            record = score_case(evaluator, case, text, args.benchmark)
            records.append(record)
            out.write(json.dumps({**record, "prompt": prompt, "response": response},
                                 ensure_ascii=False, default=str) + "\n")

    n = len(records)
    repo = sum(r["repo_correct"] for r in records)
    corrected = sum(r["corrected_correct"] for r in records)
    new_responses = [r for r in responses if not r["cached"]]
    paper = PAPER_VANILLA.get(args.benchmark)
    summary = {
        "run_id": args.run_id, "benchmark": args.benchmark, "split": args.split, "model": args.model, "n": n,
        "api": args.api, "system_message": system, "max_tokens": args.max_tokens, "prompt": args.prompt, "table_file": args.table_file,
        "dropped_too_long": too_long,
        "repo_faithful_acc": round(repo / n, 4), "repo_faithful_95ci": wilson_interval(repo, n),
        "corrected_acc": round(corrected / n, 4), "corrected_95ci": wilson_interval(corrected, n),
        "verdict_changes": {"repo_right_corrected_wrong": sum(r["repo_correct"] > r["corrected_correct"] for r in records),
                            "repo_wrong_corrected_right": sum(r["repo_correct"] < r["corrected_correct"] for r in records)},
        "errors": dict(Counter(r["error"] or "none" for r in records)),
        "gold_sql_failures": sum(r["gold_failed"] for r in records),
        "repo_gold_differs_from_corrected_gold": [r["id"] for r in records if r["repo_gold_differs"]],
        "truncated_responses": sum(r["finish_reason"] == "length" for r in responses),
        "paper_vanilla_acc": paper,
        "gate_passed_on_repo_faithful": None if paper is None else abs(repo / n - paper) <= GATE_MARGIN,
        "generation": {"seconds": round(generation_seconds, 1), "new_requests": len(new_responses),
                       "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in responses) / len(responses)),
                       "completion_tokens_mean": round(sum(r["completion_tokens"] for r in responses) / len(responses)),
                       "completion_tokens_per_s": round(sum(r["completion_tokens"] for r in new_responses) / generation_seconds)},
    }
    out_file = ROOT / "results" / "baseline" / f"{args.run_id}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=1))
    log_run(args.run_id, "baseline",
            {"benchmark": args.benchmark, "model": args.model, "api": args.api, "system_message": system,
             "max_tokens": args.max_tokens, "limit": args.limit, "prompt": args.prompt},
            summary, artifact=out_file)
    print("RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
