"""Rebuild the WikiTableQuestions test set in the repo's folder format.

Writes data/bench/NL2SQL/benchmark/test/WikiQA/<case>/{data.csv,info.json}, one database per case in WikiQA_db/, and
results/ids/wikiqa.txt. "WikiQA" is the name the repo's evaluator uses for this benchmark.

The test split is pristine-unseen-tables (4,344 questions), per the dataset's README. The paper reports 1,000 cases
without saying how they were chosen, so we walk a seeded shuffle and keep the first 1,000 that fit the prompt limit.
Gold is an answer list, not SQL. Tables are loaded as the repo's Generation/NL2DSL/NL2SQL/execute.py loads them.
"""
import argparse
import csv
import json
import random
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from ts.bench.wikisql import unique_headers
from ts.repo import BENCH_ROOT, ROOT, repo_prompt_generator


def load_table(path):
    frame = pd.read_csv(path, keep_default_na=False, dtype=object, escapechar="\\", on_bad_lines="skip")
    frame.columns = unique_headers([str(c).replace("\\n", " ") for c in frame.columns])
    return frame


def build(raw, n, seed, out_root=BENCH_ROOT, ids_file=ROOT / "results" / "ids" / "wikiqa.txt",
          split="pristine-unseen-tables"):
    raw = Path(raw)
    cases_dir, db_dir = Path(out_root) / "WikiQA", Path(out_root) / "WikiQA_db"
    for path in (cases_dir, db_dir):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)

    with (raw / "data" / f"{split}.tsv").open(encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    order = list(range(len(questions)))
    random.Random(seed).shuffle(order)
    generator = repo_prompt_generator()
    ids, too_long, unreadable = [], 0, []
    for index in order:
        if len(ids) == n:
            break
        q = questions[index]
        case_id = q["id"]
        try:
            table = load_table(raw / q["context"])
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            unreadable.append({"case": case_id, "error": str(error)})
            continue
        case = cases_dir / case_id
        case.mkdir()
        table.to_csv(case / "data.csv", index=False)
        loaded = pd.read_csv(case / "data.csv", keep_default_na=False)
        if generator.generate_prompt((loaded, q["utterance"], None), None) is None:
            too_long += 1
            shutil.rmtree(case)
            continue
        conn = sqlite3.connect(db_dir / f"{case_id}.db")
        loaded.to_sql("table", conn, index=False, if_exists="replace")
        conn.close()
        (case / "info.json").write_text(json.dumps(
            {"question": q["utterance"], "label": q["targetValue"], "source_csv": q["context"]},
            ensure_ascii=False, indent=1))
        ids.append(case_id)

    ids_file = Path(ids_file)
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(ids) + "\n")
    manifest = {"benchmark": "WikiQA", "split": split, "source_questions": len(questions),
                "n": len(ids), "sampling_seed": seed, "skipped_too_long": too_long, "unreadable": unreadable}
    (cases_dir.parent / "WikiQA_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/wikitq"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(json.dumps(build(args.raw, args.n, args.seed), indent=1))


if __name__ == "__main__":
    main()
