"""Extension 2 tuning: every configured item has a valid run ID, and the negative filter keeps only rows_differ pairs."""
import yaml

from ts import ext2_tune
from ts.repo import ROOT
from ts.eval.run_baseline import SPLITS


def test_every_item_has_a_valid_run_id_per_dev_set():
    settings = yaml.safe_load((ROOT / "configs/base.yaml").read_text())["extension2"]["tuning"]
    items = list(settings["variants"]) + list(ext2_tune.reference_adapters(1))
    ids = {ext2_tune.dev_run_id(i, 1, b) for i in items for b in settings["dev_benchmarks"]}
    assert len(ids) == len(items) * len(settings["dev_benchmarks"])
    assert all("_dev_" in i for i in ids)


def test_rows_differ_keeps_only_negatives_that_ran():
    pairs = [{"rejected_status": s} for s in ("ok", "always_empty", "sql_failed", "ok")]
    assert len([p for p in pairs if ext2_tune.NEGATIVE_FILTERS["rows_differ"](p)]) == 2
    assert len([p for p in pairs if ext2_tune.NEGATIVE_FILTERS[None](p)]) == 4


def test_dev_split_is_a_separate_folder_and_id_list():
    (test_root, test_ids), (dev_root, dev_ids) = SPLITS["test"], SPLITS["dev"]
    assert test_root != dev_root and test_ids != dev_ids and dev_root.name == "dev"
