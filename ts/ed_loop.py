"""Phase 8: Table-Specialist for error detection (paper Section 4, Example 4.2; Algorithm 1 with Algorithm 2).

One round, for one seed (Instruct only, chat API with the repo system message, as everywhere in Phase 8):
  tables     the round's 3,000 corpus tables (ts.loop.round_tables: the seed's own order, the next slice). As in the repo,
             the typo, semantic-error and clean-column steps all draw on the same tables, each with its own random column.
  generate   vanilla writes one error per table with the repo's two prompts, read from its scripts:
               Generation/ErrorDetection/Table2Typo/table2typo_genneration_v1.py         {"typo cell": ...}
               Generation/ErrorDetection/Table2SemanticError/table2semerr_genneration_v6.py {"ERROR": ...}
             (the semantic prompt skips columns with a value over 50 characters, as the repo does). An error that is
             empty or already in the column is dropped.
  validate   paper Algorithm 2, as Validation/ErrorDetection/ed_validation_prompt_gen_v3.py builds it: the error is added
             to the column's distinct values, shuffled 5 times (random.seed(0..4)), and the validator must name exactly
             that value every time. Clean columns (ed_validation_for_negative_gen_v3.py: the first 1,000 tables, up to 2
             columns each) are kept if the validator names nothing on all 5 shuffles. The validator is the previous
             round's error-detection adapter (round 1: vanilla), greedy.
  train      one adapter from vanilla on this round's kept columns only (positives labelled with the error, clean
             columns with NULL), in the repo's test format (EDPromptGenerator zero-shot prompt, as ts.eval.ed_eval).
  evaluate   ts.eval.ed_eval on AutoTest ST and RT.
Run IDs: train_edts{tag}_instruct_seed{s}; per-round outputs in runs/<run_id>/round<k>/.

  TS_PROFILE=instruct python -m ts.ed_loop --seed 1
  TS_PROFILE=instruct python -m ts.ed_loop --seed 1 --tag smoke --tables 40 --negative-tables 10 --sft-max-steps 5 --eval-limit 20
"""
import argparse
import ast
import json
import random
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ts import loop
from ts.eval import ed_eval
from ts.llm.client import LLMClient
from ts.repo import REPO, ROOT
from ts.tracking import check_run_id, log_run

TYPO_SCRIPT = "Generation/ErrorDetection/Table2Typo/table2typo_genneration_v1.py"
SEMANTIC_SCRIPT = "Generation/ErrorDetection/Table2SemanticError/table2semerr_genneration_v6.py"
VALIDATION_SCRIPT = "Validation/ErrorDetection/ed_validation_prompt_gen_v3.py"
SHUFFLES = 5               # repo: range(5)
NEGATIVE_TABLES = 1000     # repo: file_list[:1000]
NEGATIVE_COLUMNS = 2       # repo: random.sample(columns, min(2, n))
SEMANTIC_MAX_LENGTH = 50   # repo: skip columns with a longer value
GENERATION_TOKENS = 128    # ours: the answer is one short JSON object
VALIDATION_TOKENS = ed_eval.MAX_TOKENS


# ---------------------------------------------------------------- the repo's prompts, read from its scripts

@lru_cache(maxsize=None)
def _fstring_template(script):
    """(text before, text after) the one {serialize_df(df_col)} field of the script's f-string prompt. The scripts run
    on import and write files, so their syntax tree is read instead."""
    tree = ast.parse((REPO / script).read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets)
                and isinstance(node.value, ast.JoinedStr)):
            parts = node.value.values
            fields = [i for i, p in enumerate(parts) if isinstance(p, ast.FormattedValue)]
            if len(fields) != 1 or "serialize_df" not in ast.unparse(parts[fields[0]]):
                continue
            head = "".join(p.value for p in parts[:fields[0]])
            tail = "".join(p.value for p in parts[fields[0] + 1:])
            return head, tail
    raise RuntimeError(f"Repo prompt not found in {script}")


def serialize_with_header(frame):
    """table2typo's serialize_df: header, separator row, then values."""
    text = [""] + [str(c) for c in frame.columns] + ["\n"] + ["---" for _ in frame.columns] + ["\n"]
    for row in frame.values:
        text.extend(str(x) for x in row)
        text.append("\n")
    return "|".join(text)


def serialize_values(frame):
    """The semantic-error and validation scripts' serialize_df: values only."""
    text = [""]
    for row in frame.values:
        text.extend(str(x) for x in row)
        text.append("\n")
    return "|".join(text)


def _fill(script, serialized):
    head, tail = _fstring_template(script)
    return (head + serialized + tail).replace(" ", " ")


def typo_prompt(column, values):
    return _fill(TYPO_SCRIPT, serialize_with_header(pd.DataFrame({column: values})))


def semantic_prompt(values):
    return _fill(SEMANTIC_SCRIPT, serialize_values(pd.DataFrame({"<Column>": values})))


def validation_prompt(column, values):
    return _fill(VALIDATION_SCRIPT, serialize_values(pd.DataFrame({column: values})))


def shuffles(values, count=SHUFFLES):
    """The repo's orders: a copy shuffled under random.seed(i) for i in 0..4 (module random, as in the script)."""
    state = random.getstate()
    out = []
    for i in range(count):
        copy = list(values)
        random.seed(i)
        random.shuffle(copy)
        out.append(copy)
    random.setstate(state)
    return out


# ---------------------------------------------------------------- candidates

def candidates(table_ids, seed, round_number, negative_tables=NEGATIVE_TABLES):
    """Typo, semantic and clean-column candidates for the round's tables, with this seed's and round's own RNG."""
    rng = random.Random(seed * 1000 + round_number)
    generator = ed_eval.ed_prompt_generator()

    def fits(column, values):
        """Only columns whose repo test prompt fits its length limit can become training examples (ours: also keeps
        every generation and validation prompt inside the server's context)."""
        return generator.generate_prompt(pd.DataFrame({column: values}), None) is not None

    typo, semantic, negative = [], [], []
    for index, table_id in enumerate(table_ids):
        frame = loop.load_table(table_id)
        if frame.empty or not len(frame.columns):
            continue
        column = rng.choice(list(frame.columns))
        values = [str(v) for v in frame[column].unique()]
        if fits(str(column), values):
            typo.append({"table_id": table_id, "column": str(column), "values": values, "kind": "typo",
                         "prompt": typo_prompt(str(column), values)})
        column = rng.choice(list(frame.columns))
        if frame[column].map(lambda x: len(str(x))).max() <= SEMANTIC_MAX_LENGTH:
            values = [str(v) for v in frame[column].replace("", np.nan).dropna().unique()]
            if values and fits(str(column), values):
                semantic.append({"table_id": table_id, "column": str(column), "values": values, "kind": "semantic",
                                 "prompt": semantic_prompt(values)})
        if index < negative_tables:
            for column in rng.sample(list(frame.columns), min(NEGATIVE_COLUMNS, len(frame.columns))):
                values = [str(v) for v in frame[column].unique()]
                if fits(str(column), values):
                    negative.append({"table_id": table_id, "column": str(column), "values": values,
                                     "kind": "negative"})
    return typo, semantic, negative


def parse_error(text, kind):
    value = ed_eval.extract_json_answer(text, "typo cell" if kind == "typo" else "ERROR")
    if value == "JSONParsingError" or value is None or isinstance(value, (list, dict)):
        return None
    value = str(value).strip()
    return value or None


# ---------------------------------------------------------------- validation (Algorithm 2)

def validation_items(positives, negatives):
    """One item per candidate with its 5 shuffled prompts and the expected answer (a value, or None for clean)."""
    items = []
    for c in positives:
        orders = shuffles(c["values"] + [c["error"]])
        items.append({**c, "expected": c["error"], "orders": orders,
                      "prompts": [validation_prompt(c["column"], o) for o in orders]})
    for c in negatives:
        orders = shuffles(c["values"])
        items.append({**c, "expected": None, "orders": orders,
                      "prompts": [validation_prompt(c["column"], o) for o in orders]})
    return items


def verdict(expected, answers):
    """Keep only if every shuffle gives exactly the expected value (positives) or nothing (clean columns)."""
    want = set() if expected is None else {expected}
    predictions = [ed_eval.predicted_set(a) for a in answers]
    agreed = sum(p == want for p in predictions)
    return {"keep": agreed == len(answers), "agreed": agreed, "predictions": [sorted(p) for p in predictions]}


# ---------------------------------------------------------------- training examples

def training_examples(kept):
    """Repo test format: zero-shot ED prompt of the column (first shuffle) and {"obvious_error": value or "NULL"}."""
    generator = ed_eval.ed_prompt_generator()
    rows = []
    for item in kept:
        frame = pd.DataFrame({item["column"]: item["orders"][0]})
        prompt = generator.generate_prompt(frame, None)
        if prompt is None:
            continue
        y = {"label": item["expected"], "cot": None, "cot_position": None, "use_cot": False}
        example = {"prompt": prompt, "completion": generator.generate_completion(y)}
        rows.append(loop.as_chat(example) if loop.PROFILE == "instruct" else example)
    return rows


# ---------------------------------------------------------------- one round

def _served(adapter):
    return (loop.adapter_name("ed", adapter), {loop.adapter_name("ed", adapter): str(adapter)}) if adapter \
        else (loop.MODEL_NAME, {})


def run_round(run_id, seed, round_number, run_dir, previous, per_round, negative_tables, slot, sft_max_steps,
              eval_limit):
    round_dir = run_dir / f"round{round_number}"
    if (round_dir / "summary.json").exists():
        return json.loads((round_dir / "summary.json").read_text())
    round_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    timings = {}
    table_ids = [line.split("\t")[0] for line in (ROOT / "results/ids/corpus.txt").read_text().splitlines()]
    tables = loop.round_tables(table_ids, seed, round_number, per_round)

    gen_path, val_path = round_dir / "errors.jsonl", round_dir / "validated.jsonl"
    if not val_path.exists():
        typo, semantic, negative = candidates(tables, seed, round_number, negative_tables)
        if not gen_path.exists():
            start = time.perf_counter()
            with slot() as lane, loop.Server({}, round_dir / "server_generate.log", lane):
                client = LLMClient(loop.MODEL_NAME, base_url=lane.url)
                rows = typo + semantic
                answers = client.complete_many([r["prompt"] for r in rows], api=loop.API, system=loop.SYSTEM,
                                               temperature=config["generation"]["question_temperature"],
                                               max_tokens=GENERATION_TOKENS, seed=seed)
            for row, answer in zip(rows, answers):
                row["response"] = answer["text"]
                error = parse_error(answer["text"], row["kind"])
                row["error"] = error if error is not None and error not in row["values"] else None
            loop.write_jsonl(gen_path, rows)
            timings["generation_s"] = round(time.perf_counter() - start)
        rows = loop.read_jsonl(gen_path)
        positives = [r for r in rows if r["error"]]
        items = validation_items(positives, negative)
        start = time.perf_counter()
        name, adapters = _served(previous)
        with slot() as lane, loop.Server(adapters, round_dir / "server_validate.log", lane):
            client = LLMClient(name, base_url=lane.url)
            flat = [p for item in items for p in item["prompts"]]
            answers = client.complete_many(flat, api=loop.API, system=loop.SYSTEM,
                                           temperature=config["generation"]["code_temperature"],
                                           max_tokens=VALIDATION_TOKENS, seed=0)
        validated = []
        for i, item in enumerate(items):
            texts = [a["text"] for a in answers[i * SHUFFLES:(i + 1) * SHUFFLES]]
            v = verdict(item["expected"], texts)
            validated.append({k: item[k] for k in ("table_id", "column", "kind", "expected", "orders")}
                             | v | {"answers": texts})
        loop.write_jsonl(val_path, validated)
        timings["validation_s"] = round(time.perf_counter() - start)
    validated = loop.read_jsonl(val_path)
    kept = [v for v in validated if v["keep"]]

    examples_path = round_dir / "train_ed.jsonl"
    if not examples_path.exists():
        loop.write_jsonl(examples_path, training_examples(kept))
    examples = loop.read_jsonl(examples_path)
    adapter = round_dir / "adapter_ed"
    start = time.perf_counter()
    with slot() as lane:
        train = loop.train_adapter(examples_path, adapter, seed, round_dir / "train_ed.log", lane, max_steps=sft_max_steps)
    timings["train_s"] = round(time.perf_counter() - start)
    model = f"{run_id.removeprefix('train_')}_round{round_number}"
    start = time.perf_counter()
    ed_eval.evaluate(model, adapter, "train", eval_limit)
    timings["evaluate_s"] = round(time.perf_counter() - start)
    scores = {b: json.loads((ROOT / "results/baseline" / f"{ed_eval.run_id('train', model, b)}.json").read_text())
              for b in ed_eval.BENCHES}

    by_kind = {}
    for v in validated:
        k = by_kind.setdefault(v["kind"], {"candidates": 0, "kept": 0})
        k["candidates"] += 1
        k["kept"] += v["keep"]
    generated = loop.read_jsonl(gen_path)
    summary = {"run_id": run_id, "seed": seed, "round": round_number, "tables": len(tables),
               "validator": str(previous) if previous else "vanilla",
               "errors_generated": {kind: sum(1 for r in generated if r["kind"] == kind) for kind in ("typo", "semantic")},
               "errors_usable": {kind: sum(1 for r in generated if r["kind"] == kind and r["error"])
                                 for kind in ("typo", "semantic")},
               "validation": by_kind, "examples": len(examples), "train": train,
               "test": {b: {"f1_repo": s["repo_faithful"]["f1"], "f1_autotest": s["autotest_protocol"]["f1"]}
                        for b, s in scores.items()},
               "timings": timings}
    (round_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if "smoke" not in run_id:
        log_run(f"{run_id}_round{round_number}", "error_detection_ts", {"seed": seed, "round": round_number},
                {"examples": len(examples), **{f"f1_{b}": s["repo_faithful"]["f1"] for b, s in scores.items()}})
    print("ROUND " + json.dumps(summary), flush=True)
    return summary


def run(seed, tag="", rounds=3, per_round=None, negative_tables=NEGATIVE_TABLES, sft_max_steps=None, eval_limit=None):
    from ts.repair_loop import gpu_slot
    if loop.PROFILE != "instruct":
        raise SystemExit("Phase 8 is Instruct only: set TS_PROFILE=instruct")
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    per_round = per_round or config["study"]["tables_per_round"]
    run_id = f"train_edts{tag}_{loop.PROFILE}_seed{seed}"
    check_run_id(run_id)
    run_dir = ROOT / "runs" / run_id
    previous = None
    for r in range(1, rounds + 1):
        run_round(run_id, seed, r, run_dir, previous, per_round, negative_tables, gpu_slot, sft_max_steps, eval_limit)
        previous = run_dir / f"round{r}" / "adapter_ed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="", help="smoke tests: a run-name suffix such as smoke")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tables", type=int, help="tables per round (default: configs/base.yaml, 3,000)")
    ap.add_argument("--negative-tables", type=int, default=NEGATIVE_TABLES)
    ap.add_argument("--sft-max-steps", type=int)
    ap.add_argument("--eval-limit", type=int)
    args = ap.parse_args()
    run(args.seed, args.tag, args.rounds, args.tables, args.negative_tables, args.sft_max_steps, args.eval_limit)


if __name__ == "__main__":
    main()
