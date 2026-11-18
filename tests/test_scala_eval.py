"""Phase 8 NL-to-Scala scoring: extraction, format check, result parsing and both verdicts (no Spark needed)."""
import json
from types import SimpleNamespace

import pandas as pd

from ts.eval import scala_eval as se

GOOD = 'val df = spark.read.option("header", "true").csv("data.csv")\nval finalResult = df\nfinalResult.coalesce(1)' \
       '.write.option("header", "true").csv("result")'


def case(label="SELECT a FROM \"table\"", cid="c1"):
    return {"id": cid, "info": {"question": "q", "label": label}}


def test_extract_takes_last_block_like_the_repo():
    text = "```scala\nfirst\n```\nthen\n```scala\nsecond\n```"
    assert se.extract_scala(text) == "second"
    assert se.extract_scala("val x = 1") == "val x = 1"
    assert se.extract_scala("no code here") is None


def test_format_check_requires_input_and_output_paths():
    assert se.format_ok(GOOD)
    assert not se.format_ok(GOOD.replace("data.csv", "x.csv"))
    assert not se.format_ok(GOOD.replace('csv("result")', 'csv("out")'))
    assert not se.format_ok(None)


def test_parse_rows_matches_repo_normalisation():
    assert se.parse_rows("a,b\n1,x\n2.5,\n", False) == [[1, "x"], [2.5, "None"]]
    assert se.parse_rows("", False) == []
    assert se.parse_rows(None, False) is None


def test_escape_aware_read_only_in_corrected():
    text = 'a\n"say \\"hi\\""\n'
    assert se.parse_rows(text, True) == [['say "hi"']]


def record(csv):
    return {"ok": True, "csv": csv}


def test_repo_verdict_is_order_sensitive_corrected_is_not():
    gold = [[1], [2]]
    r = se.score_case("WikiSQL", case(), GOOD, gold, record("a\n2\n1\n"))
    assert r["repo_correct"] == 0 and r["corrected_correct"] == 1


def test_order_by_makes_corrected_order_sensitive():
    gold = [[1], [2]]
    r = se.score_case("WikiSQL", case('SELECT a FROM "table" ORDER BY a'), GOOD, gold, record("a\n2\n1\n"))
    assert r["corrected_correct"] == 0


def test_empty_results_equal_each_other():
    r = se.score_case("spider", case(), GOOD, [["None"]], record(""))
    assert r["repo_correct"] == 1 and r["corrected_correct"] == 1


def test_failures_are_wrong_with_reasons():
    assert se.score_case("spider", case(), None, [[1]], None)["error"] == "no_code"
    assert se.score_case("spider", case(), "val x = 1", [[1]], None)["error"] == "format_error"
    failed = se.score_case("spider", case(), GOOD, [[1]], {"ok": False, "csv": None})
    assert failed["error"] == "exec_failed" and failed["repo_correct"] == 0
    gold_failed = se.score_case("spider", case(), GOOD, "SQLExecutionError", record("a\n1\n"))
    assert gold_failed["repo_correct"] == 0 and gold_failed["gold_failed"]


def test_numbers_compare_across_int_and_float():
    r = se.score_case("WikiSQL", case(), GOOD, [[1]], record("a\n1.0\n"))
    assert r["repo_correct"] == 1


def test_cache_runs_each_program_once(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "EXEC_CACHE", tmp_path / "cache")
    table = tmp_path / "t.csv"
    pd.DataFrame({"a": [1]}).to_csv(table, index=False)
    calls = []

    def runner(work, runtime, code_file, num_cpu=1):
        calls.append(work)
        from pathlib import Path
        out = Path(work) / "result_scala"
        out.mkdir()
        (out / "part-0.csv").write_text("a\n1\n")
        return SimpleNamespace(returncode=0)

    first, hit1 = se.cached_execute(GOOD, table, runner)
    second, hit2 = se.cached_execute(GOOD, table, runner)
    assert first["ok"] and first["csv"] == "a\n1\n" and not hit1 and hit2 and len(calls) == 1
    assert json.loads(se.cache_path(se.exec_key(GOOD, table)).read_text())["csv"] == "a\n1\n"


def test_stalled_timeouts_are_rerun_beside_the_original(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "EXEC_CACHE", tmp_path / "cache")
    table = tmp_path / "t.csv"
    pd.DataFrame({"a": [1]}).to_csv(table, index=False)
    key = se.exec_key(GOOD, table)
    path = se.cache_path(key)
    path.parent.mkdir(parents=True)
    stalled = {"ok": False, "returncode": None, "n_csv": 0, "csv": None, "error": "Sandbox operation timed out",
               "seconds": 26944.0}
    path.write_text(json.dumps(stalled))
    assert se.cached_record(key) is None

    def runner(work, runtime, code_file, num_cpu=1):
        from pathlib import Path
        out = Path(work) / "result_scala"
        out.mkdir()
        (out / "part-0.csv").write_text("a\n1\n")
        return SimpleNamespace(returncode=0)

    record, hit = se.cached_execute(GOOD, table, runner)
    assert record["ok"] and not hit
    assert json.loads(path.read_text()) == stalled  # the original is kept
    assert se.cached_record(key)["ok"] and se.cached_execute(GOOD, table, runner)[1]
    genuine = {**stalled, "seconds": 130.0}  # an ordinary timeout is a verdict and stays cached
    path.write_text(json.dumps(genuine))
    se.rerun_path(path).unlink()
    assert se.cached_record(key) == genuine


def test_run_ids_are_valid_and_distinct_from_sql_runs():
    from ts.tracking import check_run_id
    rid = se.run_id("final", "instruct_repairts_seed1_round3", "WikiQA", "zero-shot")
    check_run_id(rid)
    assert rid == "final_nlscala_instruct_repairts_seed1_round3_wikitq_zeroshot"
