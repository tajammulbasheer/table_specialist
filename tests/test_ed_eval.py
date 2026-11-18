"""Phase 8 error detection: the repo's answer parsing and metric, and AutoTest's debatable-value rule."""
from ts.eval import ed_eval as ed


def test_parsing_follows_the_repo():
    assert ed.predicted_set('{"obvious_error": "febuary"}') == {"febuary"}
    assert ed.predicted_set('Answer: {"obvious_error": "NULL"}') == set()
    assert ed.predicted_set('{"obvious_error": null}') == set()
    assert ed.predicted_set("no json here") == set()
    assert ed.predicted_set('{"other": 1} {"obvious_error": ["a", "b"]}') == {"a", "b"}
    assert ed.predicted_set("{'obvious_error': 'x'}") == {"x"}  # literal_eval fallback


def test_row_scores_definedness():
    assert ed.row_scores([], set()) == {"recall": None, "precision": None}
    assert ed.row_scores([], {"x"}) == {"recall": None, "precision": 0.0}
    assert ed.row_scores(["x"], set()) == {"recall": 0.0, "precision": None}
    assert ed.row_scores(["x", "y"], {"x"}) == {"precision": 1.0, "recall": 0.5}


def test_metric_averages_over_defined_rows():
    rows = [{"precision": 1.0, "recall": 1.0}, {"precision": 0.0, "recall": None}, {"precision": None, "recall": 0.0}]
    m = ed.metric(rows)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["columns_predicted"] == 2 and m["columns_with_errors"] == 2


def test_autotest_protocol_ignores_debatable_predictions():
    case = {"id": "st_0001", "gold": [], "debatable": ["all"], "values": [], "prompt": ""}
    records, metrics = ed.score([case], ['{"obvious_error": "all"}'])
    assert records[0]["repo"]["precision"] == 0.0
    assert records[0]["autotest"]["precision"] is None


def test_case_counts_match_the_paper():
    st, dropped = ed.load_cases("st")
    assert len(st) == 1126 and len(dropped["unparseable"]) == 1
    rt, _ = ed.load_cases("rt")
    assert len(rt) == 1081


def test_run_id_is_valid():
    from ts.tracking import check_run_id
    check_run_id(ed.run_id("final", "instruct_vanilla", "st"))
