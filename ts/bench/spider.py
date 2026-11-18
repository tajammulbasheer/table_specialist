"""Rebuild the Spider single-table test set in the repo's folder format.

Writes data/bench/NL2SQL/benchmark/test/spider/<case>/{data.csv,data.sqlite,info.json} and results/ids/spider.txt.

The paper's 1,198-case subset is undocumented. We use the test split (2,147 questions) and keep questions whose
gold SQL touches exactly one table, which gives 1,230 before the repo's prompt length limit. Dev gives only 575.
Each case gets its own database holding that one table, renamed "table", as the repo's prompt tells the model.
"""
import argparse
import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from sqlglot import exp, parse_one

from ts.bench.wikisql import quote_id
from ts.repo import BENCH_ROOT, ROOT


def gold_tables(sql):
    return {table.name.lower() for table in parse_one(sql, read="sqlite").find_all(exp.Table)}


def rewrite_gold(sql):
    """Point the query at the table called "table", keeping aliases and everything else."""
    tree = parse_one(sql, read="sqlite")
    for table in tree.find_all(exp.Table):
        table.set("this", exp.to_identifier("table", quoted=True))
    # A column qualified by the bare table name, as in singer.name, must follow the rename.
    original = gold_tables(sql).pop()
    for column in tree.find_all(exp.Column):
        if column.table.lower() == original:
            column.set("table", exp.to_identifier("table", quoted=True))
    return tree.sql(dialect="sqlite")


def open_source(path):
    conn = sqlite3.connect(path)
    conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")  # some Spider text is not valid UTF-8
    return conn


def copy_single_table(source, table, db_path):
    """Copy one table without its keys, which could point at tables this database does not hold."""
    name = next(row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                if row[0].lower() == table)
    info = source.execute(f"PRAGMA table_info({quote_id(name)})").fetchall()
    rows = source.execute(f"SELECT * FROM {quote_id(name)}").fetchall()
    headers = [column[1] for column in info]
    conn = sqlite3.connect(db_path)
    columns = ", ".join(f"{quote_id(column[1])} {column[2]}".strip() for column in info)
    conn.execute(f'CREATE TABLE "table" ({columns})')
    conn.executemany(f'INSERT INTO "table" VALUES ({", ".join("?" * len(headers))})', rows)
    conn.commit()
    conn.close()
    return headers, rows


SPIDER_SPLITS = {"test": ("spider_data/test.json", "spider_data/test_database"),
                 "dev": ("spider_data/dev.json", "spider_data/database")}


def build(archive, out_root=BENCH_ROOT, ids_file=ROOT / "results" / "ids" / "spider.txt", split="test"):
    cases_dir = Path(out_root) / "spider"
    shutil.rmtree(cases_dir, ignore_errors=True)
    cases_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as archive_file:
        tmp = Path(tmp)
        questions_file, databases = SPIDER_SPLITS[split]
        questions = json.loads(archive_file.read(questions_file))
        ids, mismatches, order_only, skipped = [], [], [], {"not_single_table": 0, "unreadable": []}
        for index, q in enumerate(questions):
            tables = gold_tables(q["query"])
            if len(tables) != 1:
                skipped["not_single_table"] += 1
                continue
            member = f"{databases}/{q['db_id']}/{q['db_id']}.sqlite"
            db = tmp / member
            if not db.exists():
                archive_file.extract(member, tmp)

            case_id = f"spider_{split}_{index:05d}"
            case = cases_dir / case_id
            case.mkdir()
            source = open_source(db)
            try:
                table = tables.pop()
                headers, rows = copy_single_table(source, table, case / "data.sqlite")
                expected = source.execute(q["query"]).fetchall()
            except (sqlite3.Error, StopIteration) as error:
                skipped["unreadable"].append({"case": case_id, "error": str(error)})
                shutil.rmtree(case)
                continue
            finally:
                source.close()

            gold = rewrite_gold(q["query"])
            pd.DataFrame(rows, columns=headers).to_csv(case / "data.csv", index=False)
            (case / "info.json").write_text(json.dumps(
                {"question": q["question"], "label": gold, "source_sql": q["query"], "db_id": q["db_id"],
                 "source_table": table, "spider_index": index}, ensure_ascii=False, indent=1))
            ids.append(case_id)

            conn = sqlite3.connect(case / "data.sqlite")
            conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
            got = conn.execute(gold).fetchall()
            conn.close()
            if got != expected:
                # Dropping keys can change SQLite's scan order. Gold and predictions both run on this database.
                same_rows = sorted(map(repr, got)) == sorted(map(repr, expected))
                (order_only if same_rows else mismatches).append(case_id)

    ids_file = Path(ids_file)
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(ids) + "\n")
    manifest = {"benchmark": "spider", "split": split, "source_questions": len(questions), "n": len(ids),
                "not_single_table": skipped["not_single_table"], "unreadable": skipped["unreadable"],
                "gold_mismatches_vs_source_db": mismatches, "gold_row_order_only_differences": order_only}
    (cases_dir.parent / "spider_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ROOT / "data/raw/spider/spider.zip"))
    args = ap.parse_args()
    print(json.dumps(build(args.archive), indent=1))


if __name__ == "__main__":
    main()
