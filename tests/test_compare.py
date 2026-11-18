"""The corrected comparator, and proof of the repo bug it fixes, using the repo's own function."""
import pytest

from ts.eval.compare import cells_equal, gold_is_ordered, results_equal
from ts.repo import load_ns_module


@pytest.fixture(scope="module")
def repo_equal():
    return load_ns_module()[1]


def test_repo_stops_at_first_close_cell_but_corrected_does_not(repo_equal):
    pred, gold = [[1.001, "wrong"]], [[1.0, "right"]]
    assert repo_equal(pred, gold) is True  # the bug: the second cell is never checked
    assert results_equal(pred, gold) is False


def test_repo_skips_later_rows_but_corrected_does_not(repo_equal):
    pred, gold = [[1.001], [999]], [[1.0], [2]]
    assert repo_equal(pred, gold) is True
    assert results_equal(pred, gold) is False


def test_gold_zero_no_longer_counts_as_mismatch(repo_equal):
    assert repo_equal([[0.001]], [[0]]) is False
    assert results_equal([[0.001]], [[0]]) is True
    assert results_equal([[0.5]], [[0]]) is False


def test_row_order_only_matters_with_order_by(repo_equal):
    pred, gold = [["a"], ["b"]], [["b"], ["a"]]
    assert repo_equal(pred, gold) is False
    assert results_equal(pred, gold, ordered=False) is True
    assert results_equal(pred, gold, ordered=True) is False


def test_duplicate_rows_are_counted():
    assert results_equal([["a"], ["a"], ["b"]], [["a"], ["b"], ["b"]]) is False
    assert results_equal([["a"], ["a"], ["b"]], [["b"], ["a"], ["a"]]) is True


def test_unordered_rows_use_numeric_tolerance():
    assert results_equal([["b", 2.0001], ["a", 1.0]], [["a", 1.0], ["b", 2.0]]) is True


def test_tolerance_needs_both_relative_and_absolute_closeness():
    assert cells_equal(1000.5, 1000.0) is False  # 0.05% off, but 0.5 absolute
    assert cells_equal(0.0051, 0.005) is False   # 0.0001 absolute, but 2% off
    assert cells_equal(1.0001, 1.0) is True


def test_non_numeric_and_empty_results():
    assert cells_equal("a", "b") is False
    assert cells_equal("None", 3) is False
    assert results_equal([], [["None"]]) is True
    assert results_equal([], [["x"]]) is False
    assert results_equal([["x"]], [["x"], ["y"]]) is False


def test_order_by_detection():
    assert gold_is_ordered('SELECT a FROM "table" ORDER  BY a') is True
    assert gold_is_ordered('SELECT "order by" FROM "table"') is True  # conservative: compares in order
    assert gold_is_ordered('SELECT a FROM "table"') is False


def test_large_unordered_results_compare_quickly():
    import time
    gold = [[i, f"name{i}"] for i in range(20000)]
    start = time.perf_counter()
    assert results_equal(gold[::-1], gold) is True
    assert results_equal(gold[::-1][:-1] + [[-1, "x"]], gold) is False
    assert time.perf_counter() - start < 5  # was quadratic: about 90 s for 20,000 rows
