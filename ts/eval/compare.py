"""Corrected result comparison. It fixes the three defects of the repo's are_lists_equal:

1. The repo returns True at the first cell that differs but is numerically close, skipping everything after it.
2. A differing cell whose gold value is 0 raises a division error, which the repo counts as a mismatch.
3. Rows are compared in order even when the query has no ORDER BY, where row order is undefined.

The numeric tolerance is unchanged, so any gap between the two scores comes only from these three fixes.
"""
import re
from collections import Counter

EMPTY_RESULTS = ([], [[]], [["None"]])  # treated as equal to each other, as in the repo's _evaluate_row
ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def cells_equal(pred, gold):
    if pred == gold:
        return True
    try:
        p, g = float(pred), float(gold)
    except (TypeError, ValueError):
        return False
    difference = abs(p - g)
    if g == 0:
        return difference < 0.01
    return difference / abs(g) < 0.01 and difference < 0.01


def rows_equal(pred, gold):
    return len(pred) == len(gold) and all(cells_equal(p, g) for p, g in zip(pred, gold))


def _unordered_equal(pred, gold):
    # Remove identical rows first, then pair the remainder with the numeric tolerance.
    pred_counts = Counter(map(repr, pred))
    gold_counts = Counter(map(repr, gold))
    # Each difference is computed once: inside the comprehensions it was rebuilt for every row, which is quadratic
    # (about 10 min for one 50,000-row BIRD result). The verdicts are unchanged.
    pred_extra = pred_counts - gold_counts
    gold_extra = gold_counts - pred_counts
    pred_left = [row for row in pred if pred_extra[repr(row)] > 0]
    gold_left = [row for row in gold if gold_extra[repr(row)] > 0]
    if len(pred_left) != len(gold_left):  # duplicates kept above are rare; fall back to pairing all rows
        pred_left, gold_left = list(pred), list(gold)
    for row in pred_left:
        match = next((i for i, candidate in enumerate(gold_left) if rows_equal(row, candidate)), None)
        if match is None:
            return False
        gold_left.pop(match)
    return not gold_left


def results_equal(pred, gold, ordered=False):
    if pred in EMPTY_RESULTS and gold in EMPTY_RESULTS:
        return True
    if len(pred) != len(gold):
        return False
    if ordered:
        return all(rows_equal(p, g) for p, g in zip(pred, gold))
    return _unordered_equal(pred, gold)


def gold_is_ordered(gold_sql):
    return bool(ORDER_BY.search(gold_sql))
