"""Round logic of ts/loop.py, using fakes for the model and sandbox. A real run on 300 tables is the end-to-end test."""
import json

import pandas as pd
import pytest

import ts.loop as loop
from ts.repo import repo_scala_prompt_generator


def test_tables_never_repeat_within_a_seed_and_differ_between_seeds():
    ids = [f"t{i}" for i in range(100)]
    rounds = [loop.round_tables(ids, 1, r, 30) for r in (1, 2, 3)]
    assert sum(len(set(r)) for r in rounds) == len({t for r in rounds for t in r}) == 90
    assert loop.round_tables(ids, 2, 1, 30) != rounds[0]
    assert loop.round_tables(ids, 1, 1, 30) == rounds[0]


def test_too_few_tables_is_an_error():
    with pytest.raises(ValueError):
        loop.round_tables(["a", "b"], 1, 2, 2)


def test_pairs_match_sql_and_scala_by_table():
    sql = [{"table_id": "a", "question": "q", "code": "S"}, {"table_id": "b", "question": "q2", "code": None}]
    scala = [{"table_id": "b", "code": "B"}, {"table_id": "a", "code": "A"}]
    assert loop.pair_code(sql, scala) == [{"table_id": "a", "question": "q", "sql": "S", "scala": "A"},
                                          {"table_id": "b", "question": "q2", "sql": None, "scala": "B"}]


PAIRS = [{"table_id": "a", "question": "q", "sql": "S", "scala": "C"},
         {"table_id": "b", "question": "q", "sql": None, "scala": "C"}]


def test_no_validation_keeps_every_pair_with_code_and_runs_no_scala(tmp_path):
    rows = loop.step_validate(tmp_path / "v.jsonl", PAIRS, "no_validation", {},
                              validate_one=lambda *_: pytest.fail("must not validate"))
    assert [r["keep"] for r in rows] == [True, False]
    assert {r["reason"] for r in rows} == {"not_validated"}


def test_table_specialist_uses_validation_and_preserves_order(tmp_path):
    def fake(pair, settings):
        return {**pair, "keep": pair["table_id"] == "b", "reason": "x", "samples_agreed": 0, "scala_runs": 1}
    rows = loop.step_validate(tmp_path / "v.jsonl", PAIRS, "table_specialist", {}, workers=2, validate_one=fake)
    assert [r["table_id"] for r in rows] == ["a", "b"] and [r["keep"] for r in rows] == [False, True]


def test_missing_code_is_rejected_before_any_execution():
    row = loop._validate_one(PAIRS[1], {})
    assert (row["keep"], row["reason"], row["scala_runs"]) == (False, "no_code", 0)


def test_one_bad_pair_is_recorded_and_does_not_stop_the_round(monkeypatch):
    def explode(*_):
        raise RuntimeError("boom")
    monkeypatch.setattr(loop, "load_table_as_text", explode)
    settings = {"row_samples": 1, "row_fraction": 0.5, "numeric_tolerance": 1e-6}
    row = loop._validate_one(PAIRS[0], settings)
    assert row["keep"] is False and row["reason"].startswith("error: RuntimeError: boom")


def test_finished_steps_are_not_repeated(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text(json.dumps({"table_id": "saved", "keep": True}) + "\n")
    rows = loop.step_validate(path, PAIRS, "table_specialist", {}, validate_one=lambda *_: pytest.fail("repeated"))
    assert rows[0]["table_id"] == "saved"


def test_interrupted_validation_resumes_without_repeating_saved_pairs(tmp_path):
    path = tmp_path / "v.jsonl"
    pairs = [{"table_id": t, "question": "q", "sql": "S", "scala": "C"} for t in "abcd"]
    saved = {"table_id": "a", "keep": True, "reason": "agree"}
    failed = {"table_id": "b", "keep": False, "reason": "error: docker stopped"}
    (tmp_path / "v.journal").write_text(json.dumps(saved) + "\n" + json.dumps(failed) + '\n{"table_id": "c", "ke')
    seen = []

    def fake(pair, settings):
        seen.append(pair["table_id"])
        return {**pair, "keep": False, "reason": "disagree"}

    rows = loop.step_validate(path, pairs, "table_specialist", {}, workers=1, validate_one=fake)
    assert seen == ["b", "c", "d"]  # "a" is saved, "b" hit an error, and "c" was cut off mid-line
    assert [r["table_id"] for r in rows] == ["a", "b", "c", "d"] and rows[0]["keep"] is True
    assert path.exists() and not (tmp_path / "v.journal").exists()


def test_a_crash_keeps_what_validation_finished(tmp_path):
    path = tmp_path / "v.jsonl"
    pairs = [{"table_id": t, "question": "q", "sql": "S", "scala": "C"} for t in "ab"]

    def fake(pair, settings):
        if pair["table_id"] == "b":
            raise KeyboardInterrupt
        return {**pair, "keep": True, "reason": "agree"}

    with pytest.raises(KeyboardInterrupt):
        loop.step_validate(path, pairs, "table_specialist", {}, workers=1, validate_one=fake)
    assert not path.exists()
    assert list(loop.read_journal(tmp_path / "v.journal")) == ["a"]


def test_writes_leave_no_partial_file(tmp_path):
    path = tmp_path / "a.jsonl"
    loop.write_jsonl(path, [{"x": 1}])
    assert loop.read_jsonl(path) == [{"x": 1}]
    assert not (tmp_path / "a.tmp").exists()


class FakeClient:
    def __init__(self, text):
        self.text, self.calls = text, []

    def complete_many(self, prompts, **params):
        self.calls.append((prompts, params))
        return [{"text": self.text, "finish_reason": "stop"} for _ in prompts]


@pytest.fixture()
def corpus(monkeypatch):
    frame = pd.DataFrame({"Team": ["a", "b"], "Wins": [1, 2]})
    monkeypatch.setattr(loop, "load_table", lambda table_id: frame)
    return frame


def test_questions_and_code_steps_use_raw_completions_and_skip_failed_questions(tmp_path, corpus):
    client = FakeClient('{"question": "Who won?"}')
    questions = loop.step_questions(tmp_path / "q.jsonl", ["a", "b"], 1, 1, client, 0.7)
    assert [q["question"] for q in questions] == ["Who won?", "Who won?"]
    assert client.calls[0][1]["api"] == "completions" and client.calls[0][1]["temperature"] == 0.7

    questions[1]["question"] = None
    coder = FakeClient("```sql\nSELECT 1\n```\nmore text")
    rows = loop.step_code(tmp_path / "s.jsonl", questions, "sql", coder, 0.0, 1)
    assert [(r["table_id"], r["code"]) for r in rows] == [("a", "SELECT 1")]
    assert coder.calls[0][1]["temperature"] == 0.0
    assert "question: Who won?" in coder.calls[0][0][0]


def test_validated_pair_gives_a_sql_and_a_scala_example(tmp_path, corpus):
    kept = [{"table_id": "a", "question": "Who?", "sql": 'SELECT "Team" FROM "table"', "scala": 'val x = 1'}]
    sql, scala = loop.step_examples(tmp_path / "s.jsonl", tmp_path / "c.jsonl", kept)
    assert "```sql" in sql[0]["completion"] and "```scala" in scala[0]["completion"]
    assert "SQL" in sql[0]["prompt"] and "Scala" in scala[0]["prompt"]


def test_adapter_name_changes_with_the_weights(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"one")
    first = loop.adapter_name("sql", tmp_path)
    assert first.startswith("sql-") and loop.adapter_name("sql", tmp_path) == first
    (tmp_path / "adapter_model.safetensors").write_bytes(b"two")
    assert loop.adapter_name("sql", tmp_path) != first


def test_evaluation_scores_every_set_and_prompt_once(tmp_path, monkeypatch):
    results = tmp_path / "results/baseline"
    results.mkdir(parents=True)
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    served, calls = [], []

    class FakeServer:
        def __init__(self, adapters, log, lane):
            served.append(adapters)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def runner(command, **_):
        calls.append(command)
        eval_id = command[command.index("--run-id") + 1]
        (results / f"{eval_id}.json").write_text(json.dumps({"repo_faithful_acc": 0.5, "corrected_acc": 0.6}))

    scores = loop.step_evaluate("run", 1, adapter, tmp_path, server=FakeServer, runner=runner,
                                lane=loop.Lane(1, 1, 8001), limit=5)
    assert "http://127.0.0.1:8001/v1" in calls[0] and calls[0][calls[0].index("--limit") + 1] == "5"
    assert len(calls) == 8 and len(scores) == 8 and scores["spider_few-shot"] == {"repo": 0.5, "corrected": 0.6}
    assert list(served[0].values()) == [str(adapter)] and all(list(served[0])[0] in c for c in calls)
    loop.step_evaluate("run", 1, adapter, tmp_path, server=FakeServer, runner=runner)
    assert len(calls) == 8 and len(served) == 1


def test_scala_generator_is_the_repo_one():
    prompt = repo_scala_prompt_generator().generate_prompt((pd.DataFrame({"a": [1]}), "q", None), None)
    assert prompt.endswith("\n\n###\n\n") and "Scala" in prompt


def test_rows_containing_unicode_line_separators_stay_whole(tmp_path):
    """Round 2 of the first no_validation run stopped here: table text held U+2028, which str.splitlines treats as a
    line end. Only the training files write it raw, but every reader has to keep such a row whole."""
    from ts.train.sft import load_examples
    rows = [{"table_id": "a", "reason": "agree", "text": f"x{c}y"} for c in "  \x85\x0b\x0c\x1c"]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert load_examples(path) == rows
    assert loop.read_jsonl(path) == rows
    journal = tmp_path / "v.journal"
    journal.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n")
    assert loop.read_journal(journal) == {"a": rows[0]}


def test_a_finished_round_is_left_untouched(tmp_path):
    finished = tmp_path / "round1"
    finished.mkdir()
    (finished / "summary.json").write_text('{"kept": 5, "timings": {"training_s": 99}}')
    before = (finished / "summary.json").read_bytes()
    adapters, summary = loop.run_round("no_validation", 1, 1, 3000, tmp_path, {}, None)
    assert (finished / "summary.json").read_bytes() == before
    assert summary["timings"]["training_s"] == 99
    assert adapters == {language: str(finished / f"adapter_{language}") for language in ("sql", "scala")}


def test_resumed_training_keeps_the_interrupted_log(tmp_path, monkeypatch):
    log, output = tmp_path / "train.log", tmp_path / "adapter"
    log.write_text("interrupted run\n")
    output.mkdir()

    def fake_run(_command, stdout, **_kwargs):
        stdout.write("resumed run\n")
        (output / "train_summary.json").write_text('{"global_steps": 9}')

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    assert loop.train_adapter(tmp_path / "e.jsonl", output, 1, log) == {"global_steps": 9}
    assert log.read_text() == "interrupted run\nresumed run\n"
