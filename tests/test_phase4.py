"""Question sampling and parsing, code extraction, and the SQL-versus-Scala agreement check."""
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ts.gen.code import code_prompts, extract_code
from ts.gen.questions import (_parts, build_requests, parse_question, question_prompt, sample_requirements)
from ts.repo import REPO
from ts.validate.exec_agree import (cells_agree, results_agree, row_samples, run_scala, run_sql, validate_pair)

TABLE = pd.DataFrame({"Team Name": [f"t{i}" for i in range(20)], "Wins": [str(i) for i in range(20)]})
SQL = 'SELECT "Team Name" FROM table WHERE CAST("Wins" AS INTEGER) >= 0'


# Questions

def test_question_prompt_pieces_come_from_the_repo_script():
    source = (REPO / "Generation/NL2DSL/Table2Question/table2question_generationv_v4.py").read_text()
    predicates, requirements, pieces = _parts()
    assert len(predicates) == 4 and len(requirements) == 6
    assert all(piece in source for piece in pieces[:1] + requirements[1:])
    prompt = question_prompt(TABLE, "(1) x")
    assert prompt.startswith("Task description: Given the input table below")
    assert "|Team Name|Wins|" in prompt and "(1) x" in prompt
    assert prompt.endswith('{"question": <QUESTION>}.\n\nOutput:\n')


def test_requirements_are_never_empty_and_are_numbered():
    rng = random.Random(0)
    for _ in range(200):
        text = sample_requirements(rng)
        assert text.startswith("(1) ")


def test_requirement_rates_match_the_repo():
    rng = random.Random(1)
    draws = [sample_requirements(rng) for _ in range(4000)]
    share = sum("filtering predicate" in d for d in draws) / len(draws)
    assert 0.68 < share < 0.75  # 70%, slightly raised because empty draws are retried


def test_requests_are_reproducible_and_seed_dependent():
    tables = [(f"t{i}", TABLE) for i in range(20)]
    assert build_requests(tables, 1) == build_requests(tables, 1)
    assert build_requests(tables, 1) != build_requests(tables, 2)


@pytest.mark.parametrize("response,expected", [
    ('{"question": "How many?"}', "How many?"),
    ("""Sure: {'question': 'Who won?'} done""", "Who won?"),
    ('{"question": "first"} {"question": "second"}', "second"),
    ("no json here", None),
    ('{"answer": "x"}', None),
    ('{"question": "  "}', None),
    ('{"question": 3}', None),
    ("", None),
    (None, None),
])
def test_parse_question(response, expected):
    assert parse_question(response) == expected


# Code

@pytest.mark.parametrize("response,language,expected", [
    ("```sql\nSELECT 1\n```", "sql", "SELECT 1"),
    ("```sql\nSELECT 1\n```\nmore\n```sql\nSELECT 2\n```", "sql", "SELECT 1"),
    ("```scala\nval a = 1\n```", "scala", "val a = 1"),
    ("```sql\nSELECT 1", "sql", None),  # cut off by the token limit
    ("```scala\nval a = 1\n```", "sql", None),
    ("```sql\n```", "sql", None),
    (None, "sql", None),
])
def test_extract_code(response, language, expected):
    assert extract_code(response, language) == expected


def test_code_prompts_contain_the_question():
    prompts = code_prompts(TABLE, "Who won?")
    assert "question: Who won?" in prompts["sql"] and "question: Who won?" in prompts["scala"]
    assert "SQL" in prompts["sql"][:60] and "Scala" in prompts["scala"][:60]


# Agreement

def test_samples_match_the_repo():
    samples = row_samples(TABLE)
    assert len(samples) == 10 and all(len(s) == 10 for s in samples)
    assert samples[3].equals(TABLE.sample(frac=0.5, random_state=3))
    assert not samples[0].equals(samples[1])


@pytest.mark.parametrize("a,b,expected", [
    ("x", "x", True), (" x ", "x", True), ("x", "X", False),
    (1, "1", True), (22.0, "22", True), (1.0000001, "1", True), (1.1, "1", False),
    (None, "", True), (float("nan"), "", True), ("abc", "1", False),
])
def test_cells_agree(a, b, expected):
    assert cells_agree(a, b, 1e-6) is expected


def test_results_ignore_order_but_count_rows():
    assert results_agree([["a"], ["b"]], [["b"], ["a"]], 1e-6)
    assert not results_agree([["a"], ["a"]], [["a"], ["b"]], 1e-6)
    assert not results_agree([["a"]], [["a"], ["a"]], 1e-6)
    assert not results_agree([["a", "b"]], [["a"]], 1e-6)
    assert results_agree([], [], 1e-6)


def test_sql_runs_on_the_sample_and_cannot_change_anything():
    sample = TABLE.head(3)
    assert run_sql('SELECT "Wins" FROM table', sample) == [[0], [1], [2]]
    assert run_sql('SELECT nope FROM table', sample) is None
    assert run_sql('DROP TABLE "table"', sample) is None


def test_sql_compares_numeric_columns_as_numbers_like_the_repo():
    sample = pd.DataFrame({"x": ["a", "b", "c"], "n": ["9", "10", "100"]})
    assert run_sql('SELECT MAX("n") FROM table', sample) == [[100]]
    assert run_sql('SELECT "x" FROM table WHERE "n" < 50', sample) == [["a"], ["b"]]


def test_null_only_rows_are_ignored_because_spark_cannot_write_them():
    assert results_agree([[None]], [], 1e-6)
    assert results_agree([[""], ["a"]], [["a"]], 1e-6)
    assert not results_agree([[None, "a"]], [], 1e-6)


def test_booleans_agree_with_their_stored_numbers():
    assert cells_agree(1, "True", 1e-6) and cells_agree(0, "false", 1e-6)
    assert not cells_agree(1, "False", 1e-6)


def test_aggregate_over_no_rows_counts_as_always_empty():
    sql = 'SELECT AVG("Wins") FROM table WHERE "Team Name" = \'none\''
    runner = fake_runner([])
    verdict = validate_pair(sql, "code", TABLE, scala_runner=runner)
    assert (verdict.keep, verdict.reason, runner.count) == (False, "always_empty", 0)


def spark_output_runner(text):
    def runner(work, runtime, code_file, num_cpu=1):
        out = Path(work) / "result_scala"
        out.mkdir()
        (out / "part-0.csv").write_text(text)
        return SimpleNamespace(returncode=0)
    return runner


def test_spark_quote_escaping_is_read_correctly_and_broken_output_is_a_failure():
    assert run_scala("code", TABLE, spark_output_runner('x\n"say \\"hi\\", ok"\n')) == [['say "hi", ok']]
    assert run_scala("code", TABLE, spark_output_runner('x\n"never closed\n')) is None


def test_a_table_sqlite_cannot_hold_rejects_instead_of_crashing():
    bad = pd.DataFrame({"Logon": ["a"] * 4, "logon": ["b"] * 4})
    assert run_sql('SELECT 1 FROM table', bad) is None
    verdict = validate_pair('SELECT 1 FROM table', "code", bad, count=2, scala_runner=fake_runner([]))
    assert (verdict.keep, verdict.reason) == (False, "sql_failed")


def fake_runner(rows_for_call):
    """Stand in for the sandbox: write the given rows for each call, or fail with None."""
    calls = iter(rows_for_call)

    def runner(work, runtime, code_file, num_cpu=1):
        assert (runtime, code_file) == ("scala", "query.scala")
        assert (Path(work) / "data.csv").exists()
        rows = next(calls)
        runner.count += 1
        if rows is None:
            return SimpleNamespace(returncode=1)
        out = Path(work) / "result_scala"
        out.mkdir()
        pd.DataFrame(rows, columns=["x"]).to_csv(out / "part-0.csv", index=False)
        return SimpleNamespace(returncode=0)

    runner.count = 0
    return runner


def agreeing_calls():
    return [s[["Team Name"]].values.tolist() for s in row_samples(TABLE)]


def test_keeps_a_pair_only_when_all_samples_agree():
    runner = fake_runner(agreeing_calls())
    verdict = validate_pair(SQL, "code", TABLE, scala_runner=runner)
    assert (verdict.keep, verdict.reason, verdict.samples_agreed, runner.count) == (True, "agree", 10, 10)


def test_stops_at_the_first_disagreement():
    calls = agreeing_calls()
    calls[2] = [["wrong"]] * 10
    runner = fake_runner(calls)
    verdict = validate_pair(SQL, "code", TABLE, scala_runner=runner)
    assert (verdict.keep, verdict.reason, verdict.samples_agreed, runner.count) == (False, "disagree", 2, 3)


def test_last_sample_disagreeing_still_rejects():
    calls = agreeing_calls()
    calls[9] = [["wrong"]] * 10
    verdict = validate_pair(SQL, "code", TABLE, scala_runner=fake_runner(calls))
    assert (verdict.keep, verdict.reason, verdict.samples_agreed) == (False, "disagree", 9)


def test_failed_scala_rejects():
    verdict = validate_pair(SQL, "code", TABLE, scala_runner=fake_runner([None]))
    assert (verdict.keep, verdict.reason) == (False, "scala_failed")


def test_failed_sql_rejects_without_running_scala():
    runner = fake_runner([])
    verdict = validate_pair("SELECT nope FROM table", "code", TABLE, scala_runner=runner)
    assert (verdict.keep, verdict.reason, runner.count) == (False, "sql_failed", 0)


def test_always_empty_rejects_without_running_scala():
    runner = fake_runner([])
    sql = 'SELECT "Wins" FROM table WHERE "Wins" = \'none\''
    verdict = validate_pair(sql, "code", TABLE, scala_runner=runner)
    assert (verdict.keep, verdict.reason, runner.count) == (False, "always_empty", 0)


@pytest.mark.skipif(not os.environ.get("TABLE_SPECIALIST_SPARK_IMAGE"), reason="Needs the Spark sandbox image")
def test_real_spark_agreement_and_disagreement():
    good = ('val df = spark.read.option("header", "true").csv("data.csv")\n'
            'val finalResult = df.select("Team Name")\n'
            'finalResult.coalesce(1).write.option("header", "true").csv("result")\n')
    bad = good.replace('df.select("Team Name")', 'df.select("Wins")')
    assert validate_pair(SQL, good, TABLE, count=2).keep
    verdict = validate_pair(SQL, bad, TABLE, count=2)
    assert (verdict.keep, verdict.reason, verdict.scala_runs) == (False, "disagree", 1)
