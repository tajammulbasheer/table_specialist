"""Extension 2 labelling and pair building (ts/ext2.py). SQL runs for real in SQLite; Scala validation is faked."""
import pandas as pd
import pytest

from ts import ext2
from ts.train.dpo import accumulation_steps
from ts.validate.exec_agree import Verdict

FRAME = pd.DataFrame({"name": [f"n{i}" for i in range(12)], "score": [str(i) for i in range(12)]})
SETTINGS = {"row_samples": 4, "row_fraction": 0.5, "numeric_tolerance": 1e-6}
GOOD = 'SELECT name FROM table WHERE score > 5'
SAME = 'SELECT "name" FROM table WHERE score >= 6'  # different text, same rows
WRONG = 'SELECT name FROM table WHERE score > 2'
BROKEN = 'SELECT nonexistent FROM table'
EMPTY = 'SELECT name FROM table WHERE score > 100'


def item(*candidates):
    return {"table_id": "t", "question": "q", "candidates": [
        {"index": i, "sql": sql, "scala": "S", "verdict": verdict} for i, (sql, verdict) in enumerate(candidates)]}


def no_scala(*_):
    pytest.fail("Scala must not run once a chosen program exists")


def labels(result):
    return [c["label"] for c in result["candidates"]]


def test_greedy_kept_is_chosen_and_the_rest_are_judged_by_sql_results_alone():
    kept = {"keep": True, "reason": "agree", "scala_runs": 0}
    result = ext2.label_question(item((GOOD, kept), (SAME, None), (WRONG, None), (BROKEN, None), (EMPTY, None), (None, None)),
                                 SETTINGS, FRAME, validate=no_scala)
    assert labels(result) == ["chosen", "same_result", "rejected", "rejected", "rejected", "no_code"]
    assert (result["keep"], result["reason"], result["scala_runs"]) == (True, "chosen_0", 0)


def test_a_program_whose_scala_failed_is_not_a_negative_if_its_sql_matches_the_chosen_one():
    rejected = {"keep": False, "reason": "scala_failed", "scala_runs": 1}
    calls = []

    def fake_validate(sql, scala, frame, *_):
        calls.append(sql)
        return Verdict(sql == GOOD, "agree" if sql == GOOD else "disagree", 0, 3)

    result = ext2.label_question(item((SAME, rejected), (WRONG, None), (GOOD, None), (WRONG + " ", None)),
                                 SETTINGS, FRAME, validate=fake_validate)
    assert labels(result) == ["same_result", "rejected", "chosen", "rejected"]
    assert calls == [WRONG, GOOD]  # the greedy verdict is reused, and nothing after the chosen one runs Scala
    assert (result["reason"], result["scala_runs"]) == ("chosen_2", 6)


def test_programs_that_fail_in_sql_never_reach_scala():
    result = ext2.label_question(item((BROKEN, {"keep": False, "reason": "sql_failed", "scala_runs": 0}), (BROKEN, None),
                                      (EMPTY, None)), SETTINGS, FRAME, validate=no_scala)
    assert labels(result) == ["unlabelled"] * 3 and result["keep"] is False and result["reason"] == "no_chosen"


def test_pairs_use_each_distinct_rejected_program_once_and_never_the_chosen_text():
    labelled = [{"table_id": "t", "question": "q", "candidates": [
        {"index": 0, "sql": GOOD, "label": "chosen", "sql_status": "ok"},
        {"index": 1, "sql": WRONG, "label": "rejected", "sql_status": "ok"},
        {"index": 2, "sql": WRONG + "\n", "label": "rejected", "sql_status": "ok"},
        {"index": 3, "sql": BROKEN, "label": "rejected", "sql_status": "sql_failed"}]},
        {"table_id": "u", "question": "q", "candidates": [{"index": 0, "sql": WRONG, "label": "unlabelled"}]}]
    pairs = ext2.preference_pairs(labelled)
    assert [(p["chosen"], p["rejected"], p["rejected_index"]) for p in pairs] == [(GOOD, WRONG, 1), (GOOD, BROKEN, 3)]


def test_candidates_put_the_greedy_pair_first_with_its_saved_verdict():
    greedy = [{"table_id": "a", "question": "q", "sql": "G", "scala": "GS", "keep": False, "reason": "disagree"}]
    sampled = [([{"table_id": "a", "code": "S1"}], [{"table_id": "a", "code": "C1"}]), ([], [])]
    (only,) = ext2.gather_candidates(greedy, sampled)
    assert [(c["index"], c["sql"], c["scala"]) for c in only["candidates"]] == [(0, "G", "GS"), (1, "S1", "C1"), (2, None, None)]
    assert only["candidates"][0]["verdict"]["reason"] == "disagree" and only["candidates"][1]["verdict"] is None


def test_sample_seeds_differ_by_seed_and_index():
    assert len({ext2.sample_seed(s, j) for s in (1, 2, 3) for j in (1, 2, 3)}) == 9


def test_dpo_batch_is_sixteen_pairs_or_all_of_them():
    assert accumulation_steps(500, 16) == 16 and accumulation_steps(5, 16) == 5


def test_validation_slot_is_exclusive(tmp_path):
    import fcntl
    path = tmp_path / "lock"
    with ext2.validation_slot(path):
        with open(path, "a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(path, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_one_bad_question_is_recorded_and_does_not_stop_the_seed(monkeypatch):
    def explode(*_):
        raise OverflowError("Python int too large to convert to SQLite INTEGER")
    monkeypatch.setattr(ext2.loop, "load_table_as_text", explode)
    result = ext2._label_one(item((GOOD, None)), SETTINGS)
    assert result["keep"] is False and result["reason"].startswith("error: OverflowError")
    assert labels(result) == ["unlabelled"]


def test_a_table_sqlite_cannot_hold_reaches_the_error_record(monkeypatch):
    huge = pd.DataFrame({"name": ["a", "b"] * 6, "score": ["1" + "0" * 30, "2"] * 6})
    monkeypatch.setattr(ext2.loop, "load_table_as_text", lambda _: huge)
    result = ext2._label_one(item((GOOD, None)), SETTINGS)
    assert result["keep"] is False
