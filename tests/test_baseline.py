"""Gold SQL building, the safety of model-SQL execution, and the patched repo evaluator, end to end on one table."""
import json
import sqlite3

import pandas as pd
import pytest

from ts.bench.wikisql import gold_sql, unique_headers
from ts.eval.run_baseline import keep_first_code_block, score_case, wilson_interval
from ts.repo import REPO, REPO_SYSTEM_MESSAGE, connect_read_only, load_ns_module, load_sandbox, repo_prompt_generator


@pytest.fixture()
def bench(tmp_path):
    """One WikiSQL case laid out the way ts/bench/wikisql.py lays it out."""
    case_dir, tables = tmp_path / "WikiSQL" / "case1", tmp_path / "WikiSQL_tables"
    case_dir.mkdir(parents=True)
    tables.mkdir()
    conn = sqlite3.connect(tables / "t1.db")
    conn.execute('CREATE TABLE "table" ("Player" text, "Points" real)')
    conn.executemany('INSERT INTO "table" VALUES (?, ?)', [("ann", 10.0), ("bob", 20.0)])
    conn.commit()
    conn.close()
    pd.DataFrame({"Player": ["ann", "bob"], "Points": [10.0, 20.0]}).to_csv(tables / "t1.csv", index=False)
    (case_dir / "info.json").write_text(json.dumps({"table_id": "t1"}))
    evaluator, _ = load_ns_module(tmp_path)
    return tmp_path, evaluator


def case(gold, question="q"):
    return {"id": "case1", "info": {"question": question, "label": gold}}


def run(bench, gold, completion):
    return score_case(bench[1], case(gold), completion, "WikiSQL")


def test_correct_prediction_scores_on_both(bench):
    gold = 'SELECT "Points" FROM "table" WHERE "Player" = \'ann\''
    record = run(bench, gold, f"```sql\n{gold}\n```")
    assert (record["repo_correct"], record["corrected_correct"]) == (1, 1)
    assert record["gold_failed"] is False


def test_wrong_prediction_scores_zero(bench):
    gold = 'SELECT "Points" FROM "table" WHERE "Player" = \'ann\''
    record = run(bench, gold, '```sql\nSELECT "Points" FROM "table" WHERE "Player" = \'bob\'\n```')
    assert (record["repo_correct"], record["corrected_correct"]) == (0, 0)


def test_unparseable_response_scores_zero(bench):
    record = run(bench, 'SELECT "Points" FROM "table"', "I cannot answer that.")
    assert (record["repo_correct"], record["corrected_correct"]) == (0, 0)


def test_bad_sql_scores_zero_and_is_labelled(bench):
    record = run(bench, 'SELECT "Points" FROM "table"', "```sql\nSELECT nope FROM nowhere\n```")
    assert record["corrected_correct"] == 0
    assert record["error"] == "compile/execution error"


def test_row_order_changes_only_the_corrected_score(bench):
    gold = 'SELECT "Player" FROM "table"'
    record = run(bench, gold, '```sql\nSELECT "Player" FROM "table" ORDER BY "Player" DESC\n```')
    assert (record["repo_correct"], record["corrected_correct"]) == (0, 1)


def test_non_ascii_column_breaks_repo_gold_but_not_corrected_gold(tmp_path):
    case_dir, tables = tmp_path / "WikiSQL" / "case1", tmp_path / "WikiSQL_tables"
    case_dir.mkdir(parents=True)
    tables.mkdir()
    conn = sqlite3.connect(tables / "t1.db")
    conn.execute('CREATE TABLE "table" ("Álbum" text, "Year" real)')
    conn.execute('INSERT INTO "table" VALUES (?, ?)', ("xuxa dance", 1996.0))
    conn.commit()
    conn.close()
    pd.DataFrame({"Álbum": ["xuxa dance"], "Year": [1996.0]}).to_csv(tables / "t1.csv", index=False)
    (case_dir / "info.json").write_text(json.dumps({"table_id": "t1"}))
    evaluator, _ = load_ns_module(tmp_path)
    gold = 'SELECT "Álbum" FROM "table" WHERE "Year" = 1996'
    record = score_case(evaluator, case(gold), f"```sql\n{gold}\n```", "WikiSQL")
    assert record["gold_result"] == [["xuxa dance"]]
    assert record["repo_gold_differs"] is True
    assert (record["repo_correct"], record["corrected_correct"]) == (0, 1)


@pytest.mark.parametrize("statement", [
    'DELETE FROM "table"',
    'DROP TABLE "table"',
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "PRAGMA writable_schema = ON",
])
def test_model_sql_cannot_change_anything(bench, statement):
    path = bench[0] / "WikiSQL_tables" / "t1.db"
    conn = connect_read_only(path)
    with pytest.raises(sqlite3.Error):
        conn.execute(statement)
    conn.close()
    assert sqlite3.connect(path).execute('SELECT COUNT(*) FROM "table"').fetchone() == (2,)


def test_runaway_query_is_stopped(bench):
    conn = connect_read_only(bench[0] / "WikiSQL_tables" / "t1.db", timeout_seconds=0.2)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n").fetchall()


def test_gold_sql_shapes():
    headers, types = ["Player", "Points"], ["text", "real"]
    assert gold_sql({"sel": 0, "agg": 0, "conds": [[1, 1, "1,000"]]}, headers, types) == \
        'SELECT "Player" FROM "table" WHERE "Points" > 1000.0'
    assert gold_sql({"sel": 1, "agg": 3, "conds": [[0, 0, "O'Neil"]]}, headers, types) == \
        'SELECT COUNT("Points") FROM "table" WHERE "Player" = \'o\'\'neil\''
    assert gold_sql({"sel": 1, "agg": 1, "conds": []}, headers, types) == 'SELECT MAX("Points") FROM "table"'


def test_unique_headers():
    assert unique_headers(["A", "a", "", "B"]) == ["A", "a_2", "col2", "B"]


def test_prompt_comes_from_repo_classes():
    table = pd.DataFrame({"Player": ["ann"], "Points": [10.0]})
    prompt = repo_prompt_generator().generate_prompt((table, "Who scored 10?", None), None)
    assert prompt.startswith("# Task Description: Natural language to SQL")
    assert "|Player|Points|" in prompt and "**Question:**\nWho scored 10?" in prompt
    assert prompt.endswith("## Output:\n\n\n###\n\n")


def test_sandbox_patch_changes_only_the_open_file_limit():
    source = (REPO / "table_specialist_sandbox.py").read_text()
    assert source.count('"nofile=256:256"') == 1
    sandbox = load_sandbox()
    assert callable(sandbox.run_generated_code)
    for limit in ('"--network", "none"', '"--read-only"', '"--cap-drop", "ALL"', '"65534:65534"',
                  '"--pids-limit", "256"', '"--memory", "3g"'):
        assert limit in source


def test_system_message_matches_the_repo():
    source = (REPO / "Evaluete" / "queryWithDeployment.py").read_text()
    assert REPO_SYSTEM_MESSAGE in source


def test_raw_completion_keeps_only_the_first_sql_block():
    text = '```sql\nSELECT 1\n```\n\n###\n\n## Input:\nmore\n```sql\nSELECT 2\n```'
    assert keep_first_code_block(text) == "```sql\nSELECT 1\n```"
    assert keep_first_code_block("no code here") == "no code here"


def test_wilson_interval():
    assert wilson_interval(50, 100) == [0.404, 0.596]
