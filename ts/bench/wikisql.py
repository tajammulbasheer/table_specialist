"""Rebuild the WikiSQL test set in the repo's folder format.

Writes data/bench/NL2SQL/benchmark/test/WikiSQL/<case>/{data.csv,info.json}, one database per table in
WikiSQL_tables/<table_id>.{db,csv}, and the case ID list in results/ids/wikisql.txt.

Cell values come from WikiSQL's official test.db, which stores them lowercased. The repo lowercases gold SQL,
which only works against lowercased data, so the prompt table and the database both use lowercased cells.
"""
import argparse
import json
import random
import re
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

from ts.repo import BENCH_ROOT, ROOT, repo_prompt_generator

AGG = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
OPS = ["=", ">", "<"]
NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def unique_headers(header):
    """Make column names usable in SQL and CSV: fill blanks and de-duplicate."""
    seen, out = {}, []
    for i, name in enumerate(header):
        name = name.strip() or f"col{i}"
        key = name.lower()
        seen[key] = seen.get(key, 0) + 1
        out.append(name if seen[key] == 1 else f"{name}_{seen[key]}")
    return out


def quote_id(name):
    return '"' + name.replace('"', '""') + '"'


def condition_value(value, column_type):
    """Coerce a condition value the way WikiSQL's lib/dbengine.py does."""
    if isinstance(value, str):
        value = value.lower()
    if column_type == "real" and not isinstance(value, (int, float)):
        try:
            value = float(value.replace(",", ""))
        except ValueError:
            value = float(NUMBER.findall(value)[0])
    return value


def literal(value):
    return "'" + value.replace("'", "''") + "'" if isinstance(value, str) else repr(value)


def gold_sql(sql, headers, types):
    column = quote_id(headers[sql["sel"]])
    select = f"{AGG[sql['agg']]}({column})" if sql["agg"] else column
    query = f'SELECT {select} FROM "table"'
    conds = [f"{quote_id(headers[c])} {OPS[op]} {literal(condition_value(v, types[c]))}"
             for c, op, v in sql["conds"]]
    return query + (" WHERE " + " AND ".join(conds) if conds else "")


def official_result(conn, table, sql, types):
    """Run the question the way WikiSQL's own engine does, to check our gold SQL against it."""
    column = f"col{sql['sel']}"
    select = f"{AGG[sql['agg']]}({column})" if sql["agg"] else column
    where, params = [], {}
    for c, op, v in sql["conds"]:
        where.append(f"col{c} {OPS[op]} :col{c}")
        params[f"col{c}"] = condition_value(v, types[c])
    query = f"SELECT {select} AS result FROM {table}" + (" WHERE " + " AND ".join(where) if where else "")
    return [row[0] for row in conn.execute(query, params)]


def write_table(official, table_id, headers, types, out):
    """Copy one table from the official database with real column names and table name "table"."""
    rows = official.execute(f"SELECT * FROM table_{table_id.replace('-', '_')}").fetchall()
    db = out / f"{table_id}.db"
    if not db.exists():
        conn = sqlite3.connect(db)
        columns = ", ".join(f"{quote_id(h)} {t}" for h, t in zip(headers, types))
        conn.execute(f'CREATE TABLE "table" ({columns})')
        conn.executemany(f'INSERT INTO "table" VALUES ({", ".join("?" * len(headers))})', rows)
        conn.commit()
        conn.close()
        pd.DataFrame(rows, columns=headers).to_csv(out / f"{table_id}.csv", index=False)
    return rows


def build(archive, n, seed, out_root=BENCH_ROOT, ids_file=ROOT / "results" / "ids" / "wikisql.txt", split="test"):
    cases_dir, tables_dir = Path(out_root) / "WikiSQL", Path(out_root) / "WikiSQL_tables"
    for path in (cases_dir, tables_dir):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive) as tar:
            for name in (f"data/{split}.jsonl", f"data/{split}.tables.jsonl", f"data/{split}.db"):
                tar.extract(name, tmp, filter="data")
        data = Path(tmp) / "data"
        questions = [json.loads(line) for line in (data / f"{split}.jsonl").open()]
        tables = {t["id"]: t for t in map(json.loads, (data / f"{split}.tables.jsonl").open())}
        official = sqlite3.connect(data / f"{split}.db")

        # The paper reports exactly 1,000 cases, so walk a seeded shuffle and keep the first n that fit the
        # prompt limit. Any prefix of the ID list is itself a random sample.
        order = list(range(len(questions)))
        random.Random(seed).shuffle(order)
        generator = repo_prompt_generator()
        ids, mismatches, empty_gold, renamed, too_long = [], [], 0, 0, 0
        for index in order:
            if len(ids) == n:
                break
            q = questions[index]
            table = tables[q["table_id"]]
            headers, types = unique_headers(table["header"]), table["types"]
            write_table(official, q["table_id"], headers, types, tables_dir)
            loaded = pd.read_csv(tables_dir / f"{q['table_id']}.csv", keep_default_na=False)
            if generator.generate_prompt((loaded, q["question"], None), None) is None:
                too_long += 1
                continue
            renamed += headers != table["header"]

            case_id = f"wikisql_{split}_{index:05d}"
            case = cases_dir / case_id
            case.mkdir()
            shutil.copy(tables_dir / f"{q['table_id']}.csv", case / "data.csv")
            # The repo keeps the prompt's data.csv apart from its database, so the prompt may have shown original
            # case while SQL ran on lowercase data. Keep this version too, so that both can be tested.
            pd.DataFrame(table["rows"], columns=headers).to_csv(case / "data_original_case.csv", index=False)
            sql = gold_sql(q["sql"], headers, types)
            (case / "info.json").write_text(json.dumps(
                {"question": q["question"], "label": sql, "table_id": q["table_id"], "wikisql_index": index},
                ensure_ascii=False, indent=1))
            ids.append(case_id)

            ours = sqlite3.connect(tables_dir / f"{q['table_id']}.db")
            got = [row[0] for row in ours.execute(sql.lower())]  # the repo lowercases gold SQL before running it
            ours.close()
            expected = official_result(official, f"table_{q['table_id'].replace('-', '_')}", q["sql"], types)
            empty_gold += not expected or expected == [None]
            if got != expected:
                mismatches.append(case_id)
        official.close()

    ids_file = Path(ids_file)
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(ids) + "\n")
    manifest = {"benchmark": "WikiSQL", "split": split, "source_questions": len(questions), "n": n, "sampling_seed": seed,
                "skipped_too_long": too_long, "tables": len(list(tables_dir.glob("*.db"))),
                "gold_mismatches_vs_official_engine": mismatches,
                "gold_empty_or_null": empty_gold, "cases_with_renamed_headers": renamed}
    (cases_dir.parent / "WikiSQL_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ROOT / "data/raw/wikisql/data.tar.bz2"))
    ap.add_argument("--n", type=int, default=1000)    # configs/base.yaml benchmarks.wikisql.n
    ap.add_argument("--seed", type=int, default=42)   # configs/base.yaml benchmarks.sampling_seed
    args = ap.parse_args()
    print(json.dumps(build(args.archive, args.n, args.seed), indent=1))


if __name__ == "__main__":
    main()
