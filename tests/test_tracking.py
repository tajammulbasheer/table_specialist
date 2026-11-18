"""Run-ID scheme and local MLflow logging."""
import json

import pytest

from ts.tracking import _flat_numbers, check_run_id, log_run


@pytest.mark.parametrize("run_id", ["final_spider_base", "gate_spider_base_raw", "smoke_lora_base",
                                    "train_tablespecialist_base_seed1_round2", "baseline_wikisql_instruct_n1000"])
def test_valid_run_ids(run_id):
    assert check_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["spider_base", "final-spider", "Final_spider_base", "final", "final_a b"])
def test_invalid_run_ids(run_id):
    with pytest.raises(ValueError):
        check_run_id(run_id)


def test_only_numbers_become_metrics():
    flat = _flat_numbers({"acc": 0.5, "ok": True, "name": "x", "ci": [0.4, 0.6], "gen": {"seconds": 3}})
    assert flat == {"acc": 0.5, "gen.seconds": 3}


def test_log_run_writes_only_to_the_given_store(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"acc": 0.5}))
    store = tmp_path / "store"
    log_run("final_spider_base", "unit-test", {"model": "base"}, {"acc": 0.5}, artifact=summary, store=store)
    assert any(path.name == "summary.json" for path in store.rglob("*"))
