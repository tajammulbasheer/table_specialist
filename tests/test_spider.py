"""Spider gold SQL rewriting and the patched repo evaluator on a Spider-shaped case."""
import json
import sqlite3

import pandas as pd

from ts.bench.spider import copy_single_table, gold_tables, rewrite_gold
from ts.eval.run_baseline import score_case
from ts.repo import load_ns_module


def source_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE Singer (Singer_ID int PRIMARY KEY, Name text, Age int)")
    conn.executemany("INSERT INTO Singer VALUES (?, ?, ?)", [(1, "Joe", 30), (2, "Ann", 40)])
    conn.commit()
    return conn


def test_gold_tables_counts_distinct_tables():
    assert gold_tables("SELECT count(*) FROM singer") == {"singer"}
    assert gold_tables("SELECT * FROM a AS t1 JOIN a AS t2 ON t1.x = t2.x") == {"a"}
    assert gold_tables("SELECT * FROM a JOIN b ON a.x = b.x") == {"a", "b"}
    assert gold_tables("SELECT x FROM a WHERE x IN (SELECT x FROM b)") == {"a", "b"}


def test_rewritten_gold_returns_the_same_rows(tmp_path):
    source = source_db(tmp_path / "source.sqlite")
    copy_single_table(source, "singer", tmp_path / "case.sqlite")
    case = sqlite3.connect(tmp_path / "case.sqlite")
    for sql in ["SELECT count(*) FROM singer",
                "SELECT Singer.Name FROM SINGER WHERE age > 35",
                "SELECT T1.name FROM singer AS T1 ORDER BY T1.age DESC",
                "SELECT name FROM singer WHERE age = (SELECT max(age) FROM singer)"]:
        assert case.execute(rewrite_gold(sql)).fetchall() == source.execute(sql).fetchall(), sql


def test_repo_evaluator_scores_a_spider_case(tmp_path):
    case_dir = tmp_path / "spider" / "case1"
    case_dir.mkdir(parents=True)
    headers, rows = copy_single_table(source_db(tmp_path / "source.sqlite"), "singer", case_dir / "data.sqlite")
    pd.DataFrame(rows, columns=headers).to_csv(case_dir / "data.csv", index=False)
    (case_dir / "info.json").write_text(json.dumps({}))
    evaluator, _ = load_ns_module(tmp_path)
    gold = rewrite_gold("SELECT name FROM singer WHERE age > 35")
    case = {"id": "case1", "info": {"question": "q", "label": gold}}
    right = score_case(evaluator, case, '```sql\nSELECT "Name" FROM table WHERE "Age" > 35\n```', "spider")
    wrong = score_case(evaluator, case, '```sql\nSELECT "Name" FROM table WHERE "Age" < 35\n```', "spider")
    assert (right["repo_correct"], right["corrected_correct"]) == (1, 1)
    assert (wrong["repo_correct"], wrong["corrected_correct"]) == (0, 0)
