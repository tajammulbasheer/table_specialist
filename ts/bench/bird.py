"""Rebuild the BIRD single-table test set in the repo's folder format.

Writes data/bench/NL2SQL/benchmark/test/bird/<case>/{data.csv,info.json}, one database per source table in
bird_tables/, and results/ids/bird.txt.

BIRD tables can hold 100,000+ rows, which no prompt fits. The repo's evaluator reads BIRD's database from
info["sqlite_path"] and its data.csv separately, unlike any other benchmark, so we follow that: the prompt shows the
first rows of the table while SQL runs on the whole table. The row count is our choice; the paper does not give it.
The repo's prompt uses only the question, so BIRD's evidence strings are not shown to the model.
"""
import argparse
import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from ts.bench.spider import copy_single_table, gold_tables, open_source, rewrite_gold
from ts.repo import BENCH_ROOT, ROOT, repo_prompt_generator

PROMPT_ROWS = (20, 10, 5, 3)  # the largest of these that fits the repo's prompt limit


def sample_rows_csv(headers, rows, question, generator):
    """Return the largest leading sample of rows that fits, or None if even the smallest does not."""
    for count in PROMPT_ROWS:
        frame = pd.DataFrame(rows[:count], columns=headers)
        # Read through CSV, as the repo's loader will, so the prompt checked here is the prompt sent later.
        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as tmp:
            frame.to_csv(tmp.name, index=False)
            loaded = pd.read_csv(tmp.name, keep_default_na=False)
        if generator.generate_prompt((loaded, question, None), None) is not None:
            return frame, count
    return None, None


def build(archive, out_root=BENCH_ROOT, ids_file=ROOT / "results" / "ids" / "bird.txt"):
    cases_dir, tables_dir = Path(out_root) / "bird", Path(out_root) / "bird_tables"
    for path in (cases_dir, tables_dir):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)

    generator = repo_prompt_generator()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as outer:
        tmp = Path(tmp)
        questions = json.loads(outer.read("dev_20240627/dev.json"))
        outer.extract("dev_20240627/dev_databases.zip", tmp)
        inner = zipfile.ZipFile(tmp / "dev_20240627" / "dev_databases.zip")
        sqlite_members = {Path(n).stem: n for n in inner.namelist()
                          if n.endswith(".sqlite") and "__MACOSX" not in n}

        ids, mismatches, order_only, failed, too_long, rows_shown = [], [], [], [], [], {}
        not_single = 0
        table_cache = {}
        for q in questions:
            tables = gold_tables(q["SQL"])
            if len(tables) != 1:
                not_single += 1
                continue
            table = tables.pop()
            case_id = f"bird_dev_{int(q['question_id']):05d}"
            db_member = sqlite_members[q["db_id"]]
            db = tmp / db_member
            if not db.exists():
                inner.extract(db_member, tmp)

            source = open_source(db)
            try:
                key = (q["db_id"], table)
                table_db = tables_dir / f"{q['db_id']}__{table}.db"
                if key not in table_cache:
                    table_cache[key] = copy_single_table(source, table, table_db)
                headers, rows = table_cache[key]
                expected = source.execute(q["SQL"]).fetchall()
            except (sqlite3.Error, StopIteration) as error:
                failed.append({"case": case_id, "error": str(error)})
                continue
            finally:
                source.close()

            frame, shown = sample_rows_csv(headers, rows, q["question"], generator)
            if frame is None:
                too_long.append(case_id)
                continue

            gold = rewrite_gold(q["SQL"])
            case = cases_dir / case_id
            case.mkdir()
            frame.to_csv(case / "data.csv", index=False)
            (case / "info.json").write_text(json.dumps(
                {"question": q["question"], "label": gold, "source_sql": q["SQL"], "db_id": q["db_id"],
                 "source_table": table, "sqlite_path": str(table_db.resolve()), "rows_in_prompt": shown,
                 "rows_in_table": len(rows), "evidence_not_shown": q["evidence"], "difficulty": q["difficulty"]},
                ensure_ascii=False, indent=1))
            ids.append(case_id)
            rows_shown[shown] = rows_shown.get(shown, 0) + 1

            conn = sqlite3.connect(table_db)
            conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
            got = conn.execute(gold).fetchall()
            conn.close()
            if got != expected:
                same_rows = sorted(map(repr, got)) == sorted(map(repr, expected))
                (order_only if same_rows else mismatches).append(case_id)

    ids_file = Path(ids_file)
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(ids) + "\n")
    manifest = {"benchmark": "bird", "split": "dev", "source_questions": len(questions), "n": len(ids),
                "not_single_table": not_single, "failed_to_run": failed, "too_long_even_at_3_rows": too_long,
                "rows_shown_in_prompt": rows_shown, "gold_mismatches_vs_source_db": mismatches,
                "gold_row_order_only_differences": order_only}
    (cases_dir.parent / "bird_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ROOT / "data/raw/bird/dev.zip"))
    args = ap.parse_args()
    print(json.dumps(build(args.archive), indent=1))


if __name__ == "__main__":
    main()
