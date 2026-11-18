"""Repair-ED and its controls: feedback text, which errors are retried, and which retries are usable."""
from ts import repair_ed as r


def gen(error="Frnace", kind="typo", table="t1", column="c"):
    return {"table_id": table, "column": column, "kind": kind, "values": ["France", "Spain"], "error": error,
            "prompt": "PROMPT", "response": '{"typo cell": "%s"}' % error}


def val(keep, kind="typo", table="t1", column="c", predictions=None):
    return {"table_id": table, "column": column, "kind": kind, "keep": keep,
            "predictions": predictions or [["Frnace"]] * 5}


def test_feedback_counts_hits_misses_and_other_values():
    text = r.feedback("Frnace", [["Frnace"], ["Frnace"], [], ["Spain"], ["Spain", "x" * 100]])
    assert "named exactly your error in 2 of the 5 orders" in text
    assert "In 1 of the orders it found no error." in text
    assert '"Spain"' in text and "x" * r.SHOWN_LENGTH + '..."' in text and "x" * (r.SHOWN_LENGTH + 1) not in text
    assert "found no error" not in r.feedback("Frnace", [["Spain"]] * 5)


def test_feedback_limits_shown_values():
    text = r.feedback("e", [["a", "b"], ["c", "d"], [], [], []])
    assert text.count('"') == 2 * r.SHOWN_VALUES and "and 1 more" in text


def test_only_rejected_errors_are_retried():
    generated = [gen(table="t1"), gen(table="t2"), gen(error=None, table="t3")]
    validated = [val(True, table="t1"), val(False, table="t2", predictions=[[]] * 5),
                 val(False, kind="negative", table="t4")]
    repair = r.retry_requests("repair", generated, validated, seed=2)
    assert len(repair) == 1 and repair[0][0]["table_id"] == "t2" and repair[0][2] == 2
    prompt = repair[0][1]
    assert prompt.startswith("PROMPT\n\nYour answer was:\n{\"typo cell\": \"Frnace\"}") and prompt.endswith(r.INSTRUCTION)
    assert "found no error" in prompt
    resample = r.retry_requests("resample", generated, validated, seed=2)
    assert resample[0][1] == "PROMPT" and resample[0][2] == 2 + r.RESAMPLE_SEED_OFFSET


def test_retry_must_be_new_and_absent_from_the_column():
    row = gen()
    assert r.accept_retry(row, '{"typo cell": "Spian"}') == "Spian"
    assert r.accept_retry(row, '{"typo cell": "Frnace"}') is None   # the same error again
    assert r.accept_retry(row, '{"typo cell": "Spain"}') is None    # a real value
    assert r.accept_retry(row, "no json") is None
    assert r.accept_retry({**row, "kind": "semantic"}, '{"ERROR": "Tokyo"}') == "Tokyo"


def test_run_ids_are_valid():
    import os
    assert r.run_id_for("repair", 1).startswith("train_edrepair_")
    assert r.run_id_for("samedata", 3, "smoke").startswith("train_eddatasmoke_")
    assert r.run_id_for("resample", 2).endswith("_seed2")
