"""WikiTQ answer-list scoring, and BIRD's prompt-sample-versus-whole-table behaviour."""
import json
import sqlite3

import pandas as pd

from ts.bench.bird import PROMPT_ROWS, sample_rows_csv
from ts.eval.run_baseline import score_case
from ts.repo import load_ns_module, repo_prompt_generator


def wikiqa_case(tmp_path, label):
    db_dir, case_dir = tmp_path / "WikiQA_db", tmp_path / "WikiQA" / "nu-1"
    db_dir.mkdir()
    case_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "nu-1.db")
    pd.DataFrame({"Country": ["Italy", "France"], "Wins": [3, 1]}).to_sql("table", conn, index=False)
    conn.close()
    return {"id": "nu-1", "info": {"question": "q", "label": label}}, load_ns_module(tmp_path)[0]


def test_wikitq_answer_list_matches_executed_sql(tmp_path):
    case, evaluator = wikiqa_case(tmp_path, "Italy")
    right = score_case(evaluator, case, '```sql\nSELECT "Country" FROM table WHERE "Wins" = 3\n```', "WikiQA")
    wrong = score_case(evaluator, case, '```sql\nSELECT "Country" FROM table WHERE "Wins" = 1\n```', "WikiQA")
    assert (right["repo_correct"], right["corrected_correct"]) == (1, 1)
    assert (wrong["repo_correct"], wrong["corrected_correct"]) == (0, 0)


def test_wikitq_numeric_answer_and_multiple_answers(tmp_path):
    case, evaluator = wikiqa_case(tmp_path, "3")
    record = score_case(evaluator, case, '```sql\nSELECT MAX("Wins") FROM table\n```', "WikiQA")
    assert (record["repo_correct"], record["corrected_correct"]) == (1, 1)
    case["info"]["label"] = "France|Italy"
    record = score_case(evaluator, case, '```sql\nSELECT "Country" FROM table\n```', "WikiQA")
    assert (record["repo_correct"], record["corrected_correct"]) == (0, 1)  # right rows, different order


def test_bird_prompt_shows_a_sample_but_sql_sees_the_whole_table(tmp_path):
    case_dir, tables = tmp_path / "bird" / "case1", tmp_path / "bird_tables"
    case_dir.mkdir(parents=True)
    tables.mkdir()
    db = tables / "d__t.db"
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE "table" ("Name" text, "Score" int)')
    rows = [(f"n{i}", i) for i in range(1000)]
    conn.executemany('INSERT INTO "table" VALUES (?, ?)', rows)
    conn.commit()
    conn.close()

    frame, shown = sample_rows_csv(["Name", "Score"], rows, "q", repo_prompt_generator())
    assert shown == PROMPT_ROWS[0] and len(frame) == PROMPT_ROWS[0]
    frame.to_csv(case_dir / "data.csv", index=False)
    (case_dir / "info.json").write_text(json.dumps({"sqlite_path": str(db)}))

    gold = 'SELECT MAX("Score") FROM "table"'
    case = {"id": "case1", "info": {"question": "q", "label": gold}}
    record = score_case(load_ns_module(tmp_path)[0], case, f"```sql\n{gold}\n```", "bird")
    assert record["gold_result"] == [[999]]
    assert (record["repo_correct"], record["corrected_correct"]) == (1, 1)


def test_bird_row_sample_shrinks_to_fit_and_gives_up_when_it_cannot():
    generator = repo_prompt_generator()
    big_cell = "word " * 600  # 20 rows far exceed the limit, 3 rows do not
    frame, shown = sample_rows_csv(["Text"], [(big_cell,)] * 30, "q", generator)
    assert shown < PROMPT_ROWS[0] and len(frame) == shown
    huge_cell = "word " * 5000  # one row alone exceeds the limit
    assert sample_rows_csv(["Text"], [(huge_cell,)] * 30, "q", generator) == (None, None)
