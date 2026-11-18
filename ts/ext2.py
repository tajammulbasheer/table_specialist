"""Extension 2: learn from rejected programs with DPO (round 1 pilot). Not part of the paper.

For each Table-Specialist round-1 question there are 4 SQL+Scala candidates: the greedy pair that Table-Specialist
already validated, and 3 pairs sampled from vanilla at temperature 0.7 (configs/base.yaml, extension2). Each question gets at most one chosen SQL program:
the first candidate, in order, whose SQL and Scala agree on all 10 row samples (paper Algorithm 3). Other candidates are
labelled by comparing their SQL results with the chosen SQL's on the same 10 samples:
  same_result  same rows on every sample: as correct as the chosen one by the validator's standard; not a negative
  rejected     different rows on some sample, an SQL error, or no rows at all
A rejected program is never judged by its own Scala, so a correct SQL program cannot become a negative because its Scala
failed. Two models are trained from vanilla, SQL only (the benchmarks score SQL):
  bestof4  SFT on one chosen program per question: isolates the effect of sampling more candidates
  dpo      bestof4, then DPO on (chosen, rejected) pairs: the effect of learning from the rejected programs
Both are compared with Table-Specialist round 1 (greedy candidate only) on the same questions and tables.

  python -m ts.ext2 --seed 1 --gpu 0 --port 8000
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

from ts import loop
from ts.llm.client import LLMClient
from ts.repo import ROOT, repo_prompt_generator
from ts.tracking import check_run_id, log_run
from ts.train.sft import build_example
from ts.validate.exec_agree import results_agree, row_samples, run_sql, validate_pair, without_empty_rows

VALIDATION_LOCK = ROOT / "runs" / ".spark_validation.lock"


def sample_seed(seed, index):
    return seed * 100 + index


@contextmanager
def validation_slot(path=VALIDATION_LOCK):
    """Spark validation uses 36 of the 40 CPUs, so only one seed validates at a time. A seed waiting here holds no
    GPU memory, and the other lane goes on training or evaluating."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as lock:
        print(f"[{time.strftime('%H:%M:%S')} UTC] waiting for the validation slot", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"[{time.strftime('%H:%M:%S')} UTC] validation slot taken", flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def gather_candidates(greedy_rows, sampled):
    """One item per question: the greedy pair (with its saved verdict) first, then the sampled pairs in order.
    `sampled` is a list of (sql_rows, scala_rows), one per sample index."""
    by_table = [({r["table_id"]: r for r in sql}, {r["table_id"]: r for r in scala}) for sql, scala in sampled]
    items = []
    for row in greedy_rows:
        candidates = [{"index": 0, "sql": row["sql"], "scala": row["scala"],
                       "verdict": {"keep": row["keep"], "reason": row["reason"], "scala_runs": 0}}]
        for index, (sql, scala) in enumerate(by_table, start=1):
            candidates.append({"index": index, "sql": (sql.get(row["table_id"]) or {}).get("code"),
                               "scala": (scala.get(row["table_id"]) or {}).get("code"), "verdict": None})
        items.append({"table_id": row["table_id"], "question": row["question"], "candidates": candidates})
    return items


def sql_status(sql, samples):
    if not sql:
        return "no_code", None
    results = []
    for sample in samples:
        rows = run_sql(sql, sample)
        if rows is None:
            return "sql_failed", None
        results.append(rows)
    if not any(without_empty_rows(rows) for rows in results):
        return "always_empty", results
    return "ok", results


def same_results(a, b, tolerance):
    return len(a) == len(b) and all(results_agree(x, y, tolerance) for x, y in zip(a, b))


def label_question(item, settings, frame, validate=validate_pair):
    """Find the chosen program and label every other candidate (see the module docstring)."""
    tolerance = settings["numeric_tolerance"]
    samples = row_samples(frame.astype(str), settings["row_samples"], settings["row_fraction"])
    candidates, chosen, scala_runs = [], None, 0
    for candidate in item["candidates"]:
        status, results = sql_status(candidate["sql"], samples)
        verdict = candidate["verdict"]
        if verdict is None and chosen is None and status == "ok" and candidate["scala"]:
            found = validate(candidate["sql"], candidate["scala"], frame, settings["row_samples"],
                             settings["row_fraction"], tolerance)
            verdict = {"keep": found.keep, "reason": found.reason, "scala_runs": found.scala_runs}
            scala_runs += found.scala_runs
        candidates.append({**candidate, "sql_status": status, "verdict": verdict, "_results": results})
        if chosen is None and verdict and verdict["keep"]:
            chosen = candidates[-1]
    for candidate in candidates:
        if chosen is None:
            candidate["label"] = "unlabelled"
        elif candidate is chosen:
            candidate["label"] = "chosen"
        elif candidate["sql_status"] == "no_code":
            candidate["label"] = "no_code"
        elif candidate["sql_status"] == "ok" and same_results(candidate["_results"], chosen["_results"], tolerance):
            candidate["label"] = "same_result"
        else:
            candidate["label"] = "rejected"
    for candidate in candidates:
        del candidate["_results"]
    reason = "no_chosen" if chosen is None else f"chosen_{chosen['index']}"
    return {**item, "candidates": candidates, "keep": chosen is not None, "reason": reason, "scala_runs": scala_runs}


def _label_one(item, settings):
    try:
        return label_question(item, settings, loop.load_table_as_text(item["table_id"]))
    except Exception as error:  # as in ts/loop.py: one bad question is recorded and does not stop the seed
        return {**item, "candidates": [{**c, "label": "unlabelled"} for c in item["candidates"]], "keep": False,
                "reason": f"error: {type(error).__name__}: {error}"[:200], "scala_runs": 0}


def preference_pairs(labelled):
    """(question, chosen SQL, rejected SQL) for every distinct rejected program of a question with a chosen one."""
    pairs = []
    for item in labelled:
        chosen = next((c for c in item["candidates"] if c["label"] == "chosen"), None)
        if chosen is None:
            continue
        seen = {chosen["sql"].strip()}
        for candidate in item["candidates"]:
            if candidate["label"] == "rejected" and candidate["sql"].strip() not in seen:
                seen.add(candidate["sql"].strip())
                pairs.append({"table_id": item["table_id"], "question": item["question"], "chosen": chosen["sql"],
                              "rejected": candidate["sql"], "rejected_index": candidate["index"],
                              "rejected_status": candidate["sql_status"]})
    return pairs


def step_training_files(round_dir, labelled):
    """SFT examples (one chosen program per question) and DPO examples, in the repo's prompt format."""
    sft_path, dpo_path = round_dir / "train_sql.jsonl", round_dir / "dpo_pairs.jsonl"
    if sft_path.exists() and dpo_path.exists():
        return loop.read_jsonl(sft_path), loop.read_jsonl(dpo_path)
    generator = repo_prompt_generator()
    sft, dpo = [], []
    for item in labelled:
        chosen = next((c for c in item["candidates"] if c["label"] == "chosen"), None)
        if chosen:
            example = build_example(generator, loop.load_table(item["table_id"]), item["question"], chosen["sql"])
            if example:
                sft.append(example)
    for pair in preference_pairs(labelled):
        table = loop.load_table(pair["table_id"])
        good = build_example(generator, table, pair["question"], pair["chosen"])
        bad = build_example(generator, table, pair["question"], pair["rejected"])
        if good and bad and good["completion"] != bad["completion"]:
            dpo.append({"prompt": good["prompt"], "chosen": good["completion"], "rejected": bad["completion"],
                        "table_id": pair["table_id"], "rejected_status": pair["rejected_status"]})
    loop.write_jsonl(sft_path, sft)
    loop.write_jsonl(dpo_path, dpo)
    return sft, dpo


def train_dpo(pairs_path, sft_adapter, output_dir, seed, log_path, gpu, max_steps=None, overrides=None):
    if (Path(output_dir) / "train_summary.json").exists():
        return json.loads((Path(output_dir) / "train_summary.json").read_text())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with open(log_path, "a") as log:
        subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "ts.train.dpo", "--model", loop.MODEL_PATH,
                        "--sft-adapter", str(sft_adapter), "--pairs", str(pairs_path), "--output", str(output_dir),
                        "--seed", str(seed), *(["--max-steps", str(max_steps)] if max_steps else []),
                        *(["--overrides", json.dumps(overrides)] if overrides else [])],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((Path(output_dir) / "train_summary.json").read_text())


def summarise_labels(labelled, sft, dpo):
    counts = {}
    for item in labelled:
        for candidate in item["candidates"]:
            key = f"candidate{candidate['index']}_{candidate['label']}"
            counts[key] = counts.get(key, 0) + 1
    return {"questions": len(labelled), "with_chosen": sum(i["keep"] for i in labelled),
            "chosen_greedy": sum(i["reason"] == "chosen_0" for i in labelled),
            "chosen_sampled": sum(i["keep"] and i["reason"] != "chosen_0" for i in labelled),
            "scala_runs": sum(i["scala_runs"] for i in labelled), "sft_examples": len(sft), "dpo_pairs": len(dpo),
            "dpo_pairs_by_rejected_status": {s: sum(p["rejected_status"] == s for p in dpo)
                                             for s in sorted({p["rejected_status"] for p in dpo})},
            "labels": dict(sorted(counts.items()))}


def run(seed, lane, tag="", tables=None, eval_limit=None, dpo_max_steps=None, sft_max_steps=None, generate_only=False):
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    suffix = f"_{tag}" if tag else ""
    sft_id, dpo_id = check_run_id(f"train_bestof4_base_seed{seed}{suffix}"), check_run_id(f"train_dpo_base_seed{seed}{suffix}")
    sft_dir, dpo_dir = ROOT / "runs" / sft_id / "round1", ROOT / "runs" / dpo_id / "round1"
    if (dpo_dir / "summary.json").exists():
        return json.loads((dpo_dir / "summary.json").read_text())
    sft_dir.mkdir(parents=True, exist_ok=True)
    dpo_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / "runs" / f"train_tablespecialist_base_seed{seed}" / "round1"
    questions = loop.read_jsonl(source / "questions.jsonl")
    greedy = loop.read_jsonl(source / "validated.jsonl")
    if tables:
        keep_ids = {q["table_id"] for q in questions[:tables]}
        questions = [q for q in questions if q["table_id"] in keep_ids]
        greedy = [g for g in greedy if g["table_id"] in keep_ids]
    timings = {}

    start = time.perf_counter()
    settings = config["extension2"]
    paths = [(sft_dir / f"sql_sample{j}.jsonl", sft_dir / f"scala_sample{j}.jsonl")
             for j in range(1, settings["sampled_candidates"] + 1)]
    if not all(p.exists() for pair in paths for p in pair):
        with loop.Server({}, sft_dir / "server.log", lane):
            vanilla = LLMClient(loop.MODEL_NAME, base_url=lane.url)
            for j, (sql_path, scala_path) in enumerate(paths, start=1):
                for language, path in (("sql", sql_path), ("scala", scala_path)):
                    loop.step_code(path, questions, language, vanilla, settings["sample_temperature"],
                                   sample_seed(seed, j))
    sampled = [(loop.read_jsonl(a), loop.read_jsonl(b)) for a, b in paths]
    timings["generation_s"] = round(time.perf_counter() - start)
    if generate_only:  # lets a GPU generate one seed while another seed holds the validation slot
        return {"seed": seed, "generated": [str(p) for pair in paths for p in pair]}

    start = time.perf_counter()
    labelled_path = sft_dir / "labelled.jsonl"
    if labelled_path.exists():
        labelled = loop.read_jsonl(labelled_path)
    else:
        with validation_slot():
            labelled = loop.step_validate(labelled_path, gather_candidates(greedy, sampled), "table_specialist",
                                          config["validation"], validate_one=_label_one)
    timings["labelling_s"] = round(time.perf_counter() - start)

    sft, dpo = step_training_files(sft_dir, labelled)
    labels = summarise_labels(labelled, sft, dpo)
    print("LABELS " + json.dumps(labels), flush=True)

    start = time.perf_counter()
    sft_summary = loop.train_adapter(sft_dir / "train_sql.jsonl", sft_dir / "adapter_sql", seed, sft_dir / "train_sql.log",
                                     loop.Lane(lane.serve_gpu, lane.serve_gpu, lane.port), max_steps=sft_max_steps)
    timings["sft_training_s"] = round(time.perf_counter() - start)
    start = time.perf_counter()
    sft_scores = loop.step_evaluate(sft_id, 1, sft_dir / "adapter_sql", sft_dir, lane=lane, limit=eval_limit)
    timings["sft_evaluation_s"] = round(time.perf_counter() - start)
    sft_record = {"variant": "bestof4", "seed": seed, **labels, "train": sft_summary, "test_scores": sft_scores}
    if not (sft_dir / "summary.json").exists():
        (sft_dir / "summary.json").write_text(json.dumps(sft_record, indent=1))
        log_run(f"{sft_id}_round1", "extension2", {"variant": "bestof4", "seed": seed}, sft_record)

    start = time.perf_counter()
    dpo_summary = train_dpo(sft_dir / "dpo_pairs.jsonl", sft_dir / "adapter_sql", dpo_dir / "adapter_sql", seed,
                            dpo_dir / "train_dpo.log", lane.serve_gpu, dpo_max_steps)
    timings["dpo_training_s"] = round(time.perf_counter() - start)
    start = time.perf_counter()
    dpo_scores = loop.step_evaluate(dpo_id, 1, dpo_dir / "adapter_sql", dpo_dir, lane=lane, limit=eval_limit)
    timings["dpo_evaluation_s"] = round(time.perf_counter() - start)

    summary = {"variant": "dpo", "seed": seed, **labels, "train": dpo_summary, "test_scores": dpo_scores,
               "bestof4_test_scores": sft_scores, "timings": timings}
    (dpo_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log_run(f"{dpo_id}_round1", "extension2", {"variant": "dpo", "seed": seed}, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True, help="Serving and training both use this GPU")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", default="", help="Added to the run IDs, for example smoke")
    ap.add_argument("--tables", type=int, default=None, help="Use only the first N questions; smoke tests only")
    ap.add_argument("--eval-limit", type=int, default=None, help="Smoke tests only")
    ap.add_argument("--sft-max-steps", type=int, default=None, help="Smoke tests only")
    ap.add_argument("--dpo-max-steps", type=int, default=None, help="Smoke tests only")
    ap.add_argument("--generate-only", action="store_true", help="Stop after generating the sampled candidates")
    args = ap.parse_args()
    summary = run(args.seed, loop.Lane(args.gpu, args.gpu, args.port), args.tag, args.tables, args.eval_limit,
                  args.dpo_max_steps, args.sft_max_steps, args.generate_only)
    print("EXT2 " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
