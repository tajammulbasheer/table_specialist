import pandas as pd

from ts import repair
from ts.validate.exec_agree import Verdict

FRAME = pd.DataFrame({"name": ["a", "b", "c", "d"], "score": ["1", "2", "3", "4"]})
LOG = """scala> Loading /tmp/job/query.scala...
Caused by: java.net.UnknownHostException: 6ad6b6e9a68b: Temporary failure in name resolution
/tmp/job/query.scala:23: error: value avg is not a member of org.apache.spark.sql.DataFrame
val finalResult = df.select("efficiency").avg()
/tmp/job/query.scala:23: error: not found: value finalResult
org.apache.spark.sql.AnalysisException: [UNRESOLVED_COLUMN] cannot resolve `x`
"""


def pair(reason, agreed=0):
    return {"table_id": "t", "question": "q?", "sql": 'SELECT "name" FROM "table"', "scala": "val df = 1",
            "reason": reason, "samples_agreed": agreed}


def test_scala_messages_keep_errors_and_drop_host_noise():
    assert repair.scala_messages(LOG) == ["error: value avg is not a member of org.apache.spark.sql.DataFrame",
                                          "error: not found: value finalResult",
                                          "org.apache.spark.sql.AnalysisException: [UNRESOLVED_COLUMN] cannot resolve `x`"]


def test_prompt_ends_in_open_block_and_reply_parses():
    prompt = repair.repair_prompt("sql", FRAME, "q?", "SELECT 1", "Running this code on the table returns no rows.")
    assert prompt.endswith(repair.CORRECTION + "\n```sql\n")
    assert "```sql\nSELECT 1\n```\nRunning this code" in prompt
    assert repair.parse_repair('SELECT "name" FROM "table"\n```\nmore text', "sql") == 'SELECT "name" FROM "table"'
    assert repair.parse_repair("SELECT cut off by the limit", "sql") is None


def test_feedback_never_shows_the_other_language():
    sql_rows = lambda sql, sample: [["a"], ["b"]]
    scala_run = lambda code, sample: ([["zzz-scala-only"]], "")
    notes = repair.feedback_for(pair("disagree", 3), FRAME, sql_runner=sql_rows, scala_runner=scala_run)
    assert "|a|" in notes["sql"] and "zzz-scala-only" not in notes["sql"]
    assert "zzz-scala-only" in notes["scala"] and "|a|" not in notes["scala"]
    assert set(repair.feedback_for(pair("always_empty"), FRAME)) == {"sql"}
    failed = repair.feedback_for(pair("scala_failed"), FRAME, scala_runner=lambda c, s: (None, LOG))
    assert set(failed) == {"scala"} and "value avg is not a member" in failed["scala"]


def test_sql_failed_feedback_has_the_sqlite_message():
    bad = {**pair("sql_failed"), "sql": 'SELECT "name" FROM "table" WHERE'}
    assert repair.feedback_for(bad, FRAME)["sql"] == "Running this code on the table gives the error: incomplete input"
    unknown = {**pair("sql_failed"), "sql": 'SELECT nosuch FROM "table"'}
    assert "no such column: nosuch" in repair.feedback_for(unknown, FRAME)["sql"]


def test_candidates_order_and_skip_unchanged_or_missing():
    plan = {"sql": "S", "scala": "C", "new_sql": "S2", "new_scala": "C2"}
    assert [c[0] for c in repair.candidates(plan)] == ["new_sql", "new_scala", "both"]
    assert repair.candidates({"sql": "S", "scala": "C", "new_sql": "S"}) == []
    assert [c[0] for c in repair.candidates({"sql": "S", "scala": "C", "new_scala": "C2"})] == ["new_scala"]


def test_validate_repaired_keeps_first_passing_combo(monkeypatch):
    monkeypatch.setattr(repair.loop, "load_table_as_text", lambda t: FRAME)
    settings = {"row_samples": 10, "row_fraction": 0.5, "numeric_tolerance": 1e-6}
    plan = {"table_id": "t", "sql": "S", "scala": "C", "new_sql": "S2", "new_scala": "C2"}
    verdicts = {("S2", "C"): Verdict(False, "disagree", 0, 1), ("S", "C2"): Verdict(True, "agree", 10, 10)}
    row = repair.validate_repaired(plan, settings, validate=lambda s, c, *a: verdicts[(s, c)])
    assert row["keep"] and row["combo"] == "new_scala" and row["scala_runs"] == 11
    assert [t["combo"] for t in row["tried"]] == ["new_sql", "new_scala"]
    nothing = repair.validate_repaired({"table_id": "t", "sql": "S", "scala": "C"}, settings)
    assert not nothing["keep"] and nothing["reason"] == "no_new_code"


def test_summary_counts():
    validated = [{"keep": True}] * 4 + [{"keep": False}] * 6
    repaired = [{"keep": True, "reason_before": "disagree", "combo": "both", "scala_runs": 10},
                {"keep": False, "reason_before": "disagree", "scala_runs": 3},
                {"keep": True, "reason_before": "scala_failed", "combo": "new_scala", "scala_runs": 10}]
    s = repair.summarise(validated, repaired, 10)
    assert s["kept_after_repair"] == 6 and s["increase"] == 0.5 and s["scala_runs"] == 23
    assert s["by_reason"]["disagree"] == {"tried": 2, "recovered": 1}
    assert s["recovered_by_combo"] == {"both": 1, "new_scala": 1}


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def complete_many(self, prompts, **params):
        self.calls.append((len(prompts), params["temperature"], params["seed"]))
        return [{"text": self.answers.pop(0)} for _ in prompts]


def test_copied_repair_is_retried_until_it_differs():
    plans = [{"sql": "SELECT 1", "scala": "C", "prompts": {"sql": "p"}}]
    client = FakeClient(["SELECT 1\n```", "SELECT 1\n```", "SELECT 2\n```"])
    repair.generate_repairs(plans, client, 1, 0.0, {"retries": 3, "retry_temperature": 0.7})
    assert plans[0]["new_sql"] == "SELECT 2" and plans[0]["new_sql_attempt"] == 2
    assert client.calls == [(1, 0.0, 1), (1, 0.7, 101), (1, 0.7, 102)]


def test_instruct_repairs_use_chat_and_first_block():
    plans = [{"sql": "SELECT 1", "scala": "C", "prompts": {"sql": "p"}, "chat_prompts": {"sql": "chat p"}}]
    client = FakeClient(["Here is the fix:\n```sql\nSELECT 2\n```\nDone."])
    settings = {"retries": 3, "retry_temperature": 0.7, "chat_max_tokens": {"sql": 256, "scala": 512}}
    repair.generate_repairs(plans, client, 1, 0.0, settings, chat=True)
    assert plans[0]["new_sql"] == "SELECT 2" and plans[0]["new_sql_attempt"] == 0
    prompt = repair.chat_repair_prompt("sql", FRAME, "q?", "SELECT 1", "Running this code on the table returns no rows.")
    assert "```sql\nSELECT 1\n```" in prompt and prompt.endswith("```sql<CODE>```.")


def test_training_pairs_use_the_sql_that_passed():
    from ts.repair_ts import training_pairs
    validated = [{"table_id": "a", "question": "qa", "sql": "SA", "scala": "CA", "keep": True},
                 {"table_id": "b", "question": "qb", "sql": "SB", "scala": "CB", "keep": False}]
    repaired = [{"table_id": "b", "question": "qb", "keep": True, "repaired_sql": "SB2", "repaired_scala": "CB"},
                {"table_id": "c", "question": "qc", "keep": False}]
    kept, recovered = training_pairs(validated, repaired)
    assert [p["sql"] for p in kept] == ["SA"] and [p["sql"] for p in recovered] == ["SB2"]
