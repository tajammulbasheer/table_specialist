"""Plan B1 + B2 (docs/PLAN_NEXT.md): execution-based self-consistency at test time.

For each test case: the greedy answer (temperature 0, exactly as in run_baseline) plus N sampled answers. Each SQL runs on
the case's database; answers that fail or return no rows are dropped (B2, execution-guided filtering), and the answer
returned is the first one of the largest group of identical results (Wang et al. 2022, self-consistency; OmniSQL uses the
same vote). The gold answer is never used for the choice. If every answer fails or is empty, the greedy answer is kept.
The chosen answer is then scored by run_baseline.score_case, the same scoring as every other result.

  python -m ts.eval.self_consistency --benchmark WikiSQL --model sql-... --run-id final_sc_... --api completions
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import yaml

from ts.eval.run_baseline import (GATE_MARGIN, PAPER_VANILLA, SPLITS, keep_first_code_block, load_cases, run_sql,
                                  score_case, wilson_interval)
from ts.gen.prompts import nl2sql_prompt
from ts.llm.client import LLMClient
from ts.repo import REPO_SYSTEM_MESSAGE, ROOT, load_ns_module, repo_prompt_generator
from ts.tracking import check_run_id, log_run


def result_key(rows):
    """Order-insensitive key for a result; None for answers that failed or returned nothing."""
    if not isinstance(rows, list) or not rows:
        return None

    def cell(v):
        return f"{round(v, 6)}" if isinstance(v, float) else str(v)

    return tuple(sorted(tuple(cell(v) for v in row) for row in rows))


def vote(keys):
    """Index of the answer to return: first member of the largest group of identical non-empty results; 0 if none."""
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return 0
    best = max(counts.values())
    return next(i for i, k in enumerate(keys) if k is not None and counts[k] == best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="WikiSQL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api", choices=["chat", "completions"], default="completions")
    ap.add_argument("--repo-system-message", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", choices=["zero-shot", "few-shot"], default="zero-shot")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--split", choices=sorted(SPLITS), default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--score-workers", type=int, default=16,
                    help="threads for scoring; use 1 for BIRD, whose 20k-50k-row results time out (5 s) under 16 threads")
    args = ap.parse_args()
    check_run_id(args.run_id)
    system = REPO_SYSTEM_MESSAGE if args.repo_system_message else None
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())

    cases = load_cases(args.benchmark, args.limit, "data.csv", args.split)
    generator = repo_prompt_generator()
    prompts, usable = [], []
    for case in cases:
        prompt = generator.generate_prompt((case["table"], case["info"]["question"], None), None)
        if prompt is not None:  # the same cases as run_baseline
            prompts.append(nl2sql_prompt(case["table"], case["info"]["question"]) if args.prompt == "few-shot" else prompt)
            usable.append(case)

    client = LLMClient(args.model, base_url=args.base_url, concurrency=32)
    start = time.perf_counter()
    runs = [client.complete_many(prompts, api=args.api, system=system,
                                 temperature=config["generation"]["code_temperature"], max_tokens=args.max_tokens, seed=0)]
    for i in range(1, args.samples + 1):
        runs.append(client.complete_many(prompts, api=args.api, system=system, temperature=args.temperature,
                                         max_tokens=args.max_tokens, seed=i))
    generation_seconds = time.perf_counter() - start

    evaluator, _ = load_ns_module(SPLITS[args.split][0])
    clean = (lambda t: keep_first_code_block(t)) if args.api == "completions" else (lambda t: t)

    def one(j):
        case = usable[j]
        texts = [clean(run[j]["text"]) for run in runs]
        greedy = score_case(evaluator, case, texts[0], args.benchmark)
        keys = [result_key(greedy["pred_result"])]
        for text in texts[1:]:  # sampled answers are only executed for the vote, never compared with the gold answer
            keys.append(result_key(run_sql(greedy["db_path"], evaluator._get_pred(text))) if greedy["db_path"] else None)
        chosen = vote(keys)
        record = greedy if chosen == 0 else score_case(evaluator, case, texts[chosen], args.benchmark)
        return {"id": case["id"], "repo_correct": record["repo_correct"], "corrected_correct": record["corrected_correct"],
                "greedy_repo_correct": greedy["repo_correct"], "chosen": chosen, "pred_sql": record["pred_sql"],
                "usable_answers": sum(k is not None for k in keys),
                "group_size": keys.count(keys[chosen]) if keys[chosen] is not None else 0}

    start_scoring = time.perf_counter()
    with ThreadPoolExecutor(args.score_workers) as pool:
        records = list(pool.map(one, range(len(usable))))
    scoring_seconds = time.perf_counter() - start_scoring

    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "preds.jsonl").open("w") as out:
        out.write("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records))
    n = len(records)
    repo, corrected, greedy = (sum(r[k] for r in records) for k in ("repo_correct", "corrected_correct", "greedy_repo_correct"))
    paper = PAPER_VANILLA.get(args.benchmark)
    summary = {
        "run_id": args.run_id, "benchmark": args.benchmark, "split": args.split, "model": args.model, "n": n,
        "api": args.api, "system_message": system, "max_tokens": args.max_tokens, "prompt": args.prompt,
        "self_consistency": {"samples": args.samples, "temperature": args.temperature, "greedy_included": True,
                             "filter": "drop failed and empty results", "tie": "first answer, greedy first"},
        "repo_faithful_acc": round(repo / n, 4), "repo_faithful_95ci": wilson_interval(repo, n),
        "corrected_acc": round(corrected / n, 4), "corrected_95ci": wilson_interval(corrected, n),
        "greedy_repo_faithful_acc": round(greedy / n, 4),
        "changed": {"greedy_wrong_vote_right": sum(r["repo_correct"] > r["greedy_repo_correct"] for r in records),
                    "greedy_right_vote_wrong": sum(r["repo_correct"] < r["greedy_repo_correct"] for r in records),
                    "vote_differs_from_greedy": sum(r["chosen"] != 0 for r in records)},
        "paper_vanilla_acc": paper,
        "gate_passed_on_repo_faithful": None if paper is None else abs(repo / n - paper) <= GATE_MARGIN,
        "score_workers": args.score_workers, "seconds": {"generation": round(generation_seconds), "scoring": round(scoring_seconds)},
    }
    out_file = ROOT / "results" / "baseline" / f"{args.run_id}.json"
    out_file.write_text(json.dumps(summary, indent=1))
    log_run(args.run_id, "extension3", {"method": "self_consistency", "benchmark": args.benchmark, "model": args.model,
                                        "prompt": args.prompt, "samples": args.samples, "temperature": args.temperature},
            summary, artifact=out_file)
    print("RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
