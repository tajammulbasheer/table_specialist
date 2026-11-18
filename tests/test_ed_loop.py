"""Phase 8 error-detection TS loop: repo prompts, Algorithm 2's verdict, candidates and training examples."""
import json

import pandas as pd

from ts import ed_loop as e


def test_prompts_come_from_the_repo_scripts():
    typo = e.typo_prompt("state", ["Ohio", "Utah"])
    assert typo.startswith("# Task Description: Given the table column below") and "|state|\n|---|\n|Ohio|" in typo
    assert '{"typo cell": "<CELL>"}' in typo
    sem = e.semantic_prompt(["Ohio", "Utah"])
    assert '{"ERROR": "new zealand"}' in sem and sem.rstrip().endswith("## Output:") and "|Ohio|\n|Utah|" in sem
    val = e.validation_prompt("state", ["Ohio"])
    assert '{"obvious_error": "NULL"}' in val and "|Ohio|" in val and "|state|" not in val


def test_shuffles_follow_module_random_seeds_and_restore_state():
    import random
    random.seed(123)
    before = random.random()
    random.seed(123)
    orders = e.shuffles(["a", "b", "c", "d"])
    assert random.random() == before  # caller's random state untouched
    assert len(orders) == 5 and all(sorted(o) == ["a", "b", "c", "d"] for o in orders)
    assert orders == e.shuffles(["a", "b", "c", "d"])


def test_verdict_requires_all_shuffles():
    hit = '{"obvious_error": "Frnace"}'
    assert e.verdict("Frnace", [hit] * 5)["keep"]
    assert not e.verdict("Frnace", [hit] * 4 + ['{"obvious_error": "NULL"}'])["keep"]
    assert not e.verdict("Frnace", [hit] * 4 + ['{"obvious_error": "Spain"}'])["keep"]
    assert e.verdict(None, ['{"obvious_error": "NULL"}'] * 5)["keep"]
    assert not e.verdict(None, ['{"obvious_error": "NULL"}'] * 4 + [hit])["keep"]


def test_parse_error_drops_missing_and_structured_answers():
    assert e.parse_error('{"typo cell": "Missisipi"}', "typo") == "Missisipi"
    assert e.parse_error('{"ERROR": "99999"}', "semantic") == "99999"
    assert e.parse_error('{"ERROR": ""}', "semantic") is None
    assert e.parse_error("nothing", "typo") is None
    assert e.parse_error('{"ERROR": ["a"]}', "semantic") is None


def test_training_examples_use_the_test_format(monkeypatch):
    monkeypatch.setattr(e.loop, "PROFILE", "instruct")
    monkeypatch.setattr(e.loop, "SYSTEM", "sys")
    kept = [{"column": "c", "expected": "Frnace", "orders": [["Spain", "Frnace"]]},
            {"column": "c", "expected": None, "orders": [["Spain", "France"]]}]
    rows = e.training_examples(kept)
    assert rows[0]["prompt"][0] == {"role": "system", "content": "sys"}
    assert "|Frnace|" in rows[0]["prompt"][1]["content"]
    assert json.loads(rows[0]["completion"][0]["content"].strip().removesuffix("END").strip()) == {"obvious_error": "Frnace"}
    assert json.loads(rows[1]["completion"][0]["content"].strip().removesuffix("END").strip()) == {"obvious_error": "NULL"}


def test_candidates_skip_long_semantic_values(monkeypatch):
    frame = pd.DataFrame({"a": ["x" * 60, "y"], "b": ["p", "q"]})
    monkeypatch.setattr(e.loop, "load_table", lambda t: frame)
    typo, semantic, negative = e.candidates(["t1", "t2", "t3"], seed=1, round_number=1, negative_tables=2)
    assert len(typo) == 3 and all(c["kind"] == "typo" for c in typo)
    assert all(c["column"] == "b" for c in semantic)
    assert len(negative) == 4  # first 2 tables, 2 columns each
