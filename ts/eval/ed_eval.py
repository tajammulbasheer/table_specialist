"""Phase 8: error detection on AutoTest (paper "Spreadsheet-Tables" = ST-bench, "Relational-Tables" = RT-bench).

Test cases: every benchmark column whose repo prompt fits the repo's length limit (EDPromptGenerator, 70% of 4,096
tokens). This gives exactly the paper's 1,126 (ST) and 1,081 (RT) columns; one ST row whose value list cannot be parsed
is left out. The column is its distinct values in the file's order, serialised as the repo's "ed" task does
(markdownvalueonly), under the repo's zero-shot prompt.

Scoring follows Evaluete/evaluate/ed_evaluator.py: the answer is {"obvious_error": ...} (first JSON object holding the
key; "NULL", null or no JSON means no prediction); per column, precision = |pred ∩ gold| / |pred| when anything is
predicted, recall = |pred ∩ gold| / |gold| when the column has errors; both are averaged over the columns where they are
defined, and F1 is taken from the two averages. Gold is AutoTest's `ground_truth`.
  repo-faithful: as above (a predicted debatable value counts against precision);
  autotest:      AutoTest's own protocol, where predictions of `ground_truth_debatable` values count neither way.

  TS_PROFILE=instruct python -m ts.eval.ed_eval                       # vanilla Instruct, both benchmarks
  TS_PROFILE=instruct python -m ts.eval.ed_eval --smoke 20
"""
import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from ts import loop
from ts.llm.client import LLMClient
from ts.repo import REPO, ROOT
from ts.tracking import check_run_id, log_run

RAW = ROOT / "data/raw/autotest/benchmarks"
BENCHES = {"st": "st_bench.csv", "rt": "rt_bench.csv"}
# Paper Table 3, Llama3.1-8B, vanilla -> Table-Specialist; read from a summary of the tables only: unverified.
PAPER_UNVERIFIED = {"st": (0.071, 0.136), "rt": (0.108, 0.161)}
MAX_TOKENS = 512  # the repo sets no limit; at 256, verbose Instruct answers were cut before their JSON (smoke test)


def ed_prompt_generator(random_state=1, max_token_length=4096):
    """main_build_data.py's settings for task "ed" (serializer markdownvalueonly), as repo_prompt_generator does."""
    finetune = str(REPO / "Finetune")
    if finetune not in sys.path:
        sys.path.insert(0, finetune)
    from data_generator.error_detection.ed_prompt_generator import EDPromptGenerator
    from data_generator.error_detection.ed_task_descriptor import EDTaskDescriptor
    descriptor = EDTaskDescriptor(serializer="markdownvalueonly", random_state=random_state,
                                  use_random_description=False, use_cot=False, use_format_suffix=True)
    return EDPromptGenerator(task_descriptor=descriptor, max_token_length=max_token_length, min_fewshot_samples=1,
                             use_random_template=False, random_state=random_state, use_system_message=False,
                             use_cot=False)


def _literal(value):
    return [] if not isinstance(value, str) else ast.literal_eval(value)


def load_cases(bench, limit=None):
    """(cases, dropped): each case has id, header, values, gold, debatable and prompt."""
    frame = pd.read_csv(RAW / BENCHES[bench], low_memory=False).dropna(subset=["index"])
    generator = ed_prompt_generator()
    cases, dropped = [], {"unparseable": [], "too_long": []}
    for _, row in frame.iterrows():
        cid = f"{bench}_{int(row['index']):04d}"
        try:
            values = [str(v) for v in _literal(row["dist_val"])]
            gold = [str(v) for v in _literal(row["ground_truth"])]
            debatable = [str(v) for v in _literal(row["ground_truth_debatable"])]
        except (ValueError, SyntaxError):
            dropped["unparseable"].append(cid)
            continue
        prompt = generator.generate_prompt(pd.DataFrame({str(row["header"]): values}), None)
        if prompt is None:
            dropped["too_long"].append(cid)
            continue
        cases.append({"id": cid, "header": str(row["header"]), "values": values, "gold": gold,
                      "debatable": debatable, "prompt": prompt})
    return (cases[:limit] if limit else cases), dropped


# ---------------------------------------------------------------- the repo's parsing and metric

def extract_json_answer(text, key):
    for match in re.findall(r"{[^}]*}", text or ""):
        for parse in (json.loads, ast.literal_eval):
            try:
                result = parse(match)
            except Exception:
                continue
            if isinstance(result, dict) and key in result:
                return result[key]
            break
    return "JSONParsingError"


def predicted_set(completion):
    pred = extract_json_answer(completion, "obvious_error")
    if extract_json_answer(completion, "corrected cell") == "NULL":
        pred = "NULL"
    if pred == "JSONParsingError" or pred is None:
        return set()
    if isinstance(pred, str):
        return set() if pred.lower() == "null" else {pred}
    if isinstance(pred, list):
        return {str(p) for p in pred}
    return {str(pred)}


def row_scores(gold, pred):
    gold = set(gold)
    if not gold:
        return {"recall": None, "precision": None if not pred else 0.0}
    if not pred:
        return {"recall": 0.0, "precision": None}
    tp = len(gold & pred)
    return {"precision": tp / len(pred), "recall": tp / len(gold)}


def metric(rows):
    precisions = [r["precision"] for r in rows if r["precision"] is not None]
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    p = sum(precisions) / len(precisions) if precisions else 0.0
    r = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = 2 * p * r / (p + r) if (p or r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "columns_predicted": len(precisions), "columns_with_errors": len(recalls)}


def score(cases, completions):
    records = []
    for case, completion in zip(cases, completions):
        pred = predicted_set(completion)
        repo = row_scores(case["gold"], pred)
        autotest = row_scores(case["gold"], pred - set(case["debatable"]))
        records.append({"id": case["id"], "gold": case["gold"], "debatable": case["debatable"], "pred": sorted(pred),
                        "repo": repo, "autotest": autotest})
    return records, {"repo": metric([r["repo"] for r in records]), "autotest": metric([r["autotest"] for r in records])}


# ---------------------------------------------------------------- run

def run_id(prefix, model, bench):
    return f"{prefix}_ed_{model}_{bench}_zeroshot"


def evaluate(model, adapter, prefix, limit):
    from ts.repair_loop import gpu_slot
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    jobs = [(b, run_id(prefix, model, b)) for b in BENCHES]
    jobs = [(b, rid) for b, rid in jobs if not (ROOT / "results/baseline" / f"{rid}.json").exists()]
    for _, rid in jobs:
        check_run_id(rid)
    if not jobs:
        return
    served = loop.adapter_name("ed", adapter) if adapter else loop.MODEL_NAME
    with gpu_slot() as lane, loop.Server({served: str(adapter)} if adapter else {}, ROOT / f"logs/ed_server_{model}.log",
                                         lane):
        client = LLMClient(served, base_url=lane.url)
        for bench, rid in jobs:
            cases, dropped = load_cases(bench, limit)
            start = time.perf_counter()
            responses = client.complete_many([c["prompt"] for c in cases], api=loop.API, system=loop.SYSTEM,
                                             temperature=config["generation"]["code_temperature"],
                                             max_tokens=MAX_TOKENS, seed=0)
            seconds = round(time.perf_counter() - start, 1)
            records, metrics = score(cases, [r["text"] for r in responses])
            run_dir = ROOT / "runs" / rid
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "preds.jsonl").open("w") as out:
                for case, record, response in zip(cases, records, responses):
                    out.write(json.dumps({**record, "prompt": case["prompt"], "response": response},
                                         ensure_ascii=False) + "\n")
            paper = PAPER_UNVERIFIED[bench]
            summary = {"run_id": rid, "task": "error_detection", "benchmark": bench, "model": served,
                       "adapter": str(adapter) if adapter else None, "n": len(cases), "limit": limit,
                       "dropped": {k: len(v) for k, v in dropped.items()}, "api": loop.API,
                       "system_message": loop.SYSTEM, "max_tokens": MAX_TOKENS,
                       "repo_faithful": metrics["repo"], "autotest_protocol": metrics["autotest"],
                       "truncated_responses": sum(r["finish_reason"] == "length" for r in responses),
                       "no_json": sum(extract_json_answer(r["text"], "obvious_error") == "JSONParsingError"
                                      for r in responses),
                       "paper_unverified": {"vanilla": paper[0], "table_specialist": paper[1]},
                       "generation_seconds": seconds}
            out_file = ROOT / "results/baseline" / f"{rid}.json"
            out_file.write_text(json.dumps(summary, indent=1))
            log_run(rid, "error_detection", {"model": model, "benchmark": bench, "limit": limit},
                    {"n": summary["n"], "f1_repo": metrics["repo"]["f1"], "f1_autotest": metrics["autotest"]["f1"]},
                    artifact=out_file)
            print("RESULT " + json.dumps({k: summary[k] for k in ("run_id", "n", "repo_faithful",
                                                                 "autotest_protocol", "no_json")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="instruct_vanilla", help="run-ID name of the model")
    ap.add_argument("--adapter", help="LoRA adapter directory; none means the vanilla model")
    ap.add_argument("--smoke", type=int, help="this many cases per benchmark, under smoke_ run IDs")
    args = ap.parse_args()
    if loop.PROFILE != "instruct":
        raise SystemExit("Phase 8 is Instruct only: set TS_PROFILE=instruct")
    evaluate(args.model, Path(args.adapter) if args.adapter else None, "smoke" if args.smoke else "final", args.smoke)


if __name__ == "__main__":
    main()
