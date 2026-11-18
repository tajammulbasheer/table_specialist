"""Phase 9 NL-to-R: prompts, extraction, R result reading, SQL-vs-R agreement, repair helpers and scoring (no Docker)."""
from types import SimpleNamespace

import pandas as pd

from ts import r_loop
from ts.eval import r_eval
from ts.gen.code import code_prompts, extract_code
from ts.repo import repo_r_prompt_generator
from ts.validate import r_agree

FRAME = pd.DataFrame({"name": ["a", "b", "c", "d"], "pts": ["1", "2", "3", "4"]})


def test_prompts_come_from_the_repo():
    few = code_prompts(FRAME, "how many rows?")["r"]
    assert few.startswith("# Task description:") and "library(dplyr)" in few and "question: how many rows?" in few
    zero = repo_r_prompt_generator().generate_prompt((FRAME, "how many rows?"), None)
    assert "**Question:**\nhow many rows?" in zero and "```r<R CODE>```" in zero
    assert repo_r_prompt_generator().generate_completion("x <- 1") == " ```r\nx <- 1\n``` "


def test_extraction_takes_the_first_r_block_as_the_repo():
    text = "```r\nfirst\n```\n```r\nsecond\n```"
    assert extract_code(text, "r") == "first" and r_eval.extract_r(text) == "first"
    assert r_eval.extract_r("no code") is None and extract_code("```r\nunclosed", "r") is None


def test_r_na_is_an_empty_cell(tmp_path):
    path = tmp_path / "result.csv"
    path.write_text('"name","x"\n"a",NA\n"b","y"\n')
    assert r_agree.read_r_result(path) == [["a", ""], ["b", "y"]]
    path.write_text("")
    assert r_agree.read_r_result(path) == []


def fake_runner(rows_by_call):
    calls = []

    def runner(work, runtime, code_file, num_cpu=1):
        assert runtime == "r" and code_file == "query.R"
        calls.append(work)
        rows = rows_by_call(len(calls))
        if rows is None:
            return SimpleNamespace(returncode=1, stdout="", stderr="Error in filter(x): object 'y' not found\n")
        pd.DataFrame(rows).to_csv(f"{work}/result.csv", index=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return runner, calls


def test_agreement_needs_every_sample():
    frame = pd.DataFrame({"n": [str(i) for i in range(10)]})
    sql = "SELECT COUNT(*) FROM table"
    runner, calls = fake_runner(lambda i: {"c": [5]})  # every 50% sample of 10 rows has 5 rows
    v = r_agree.validate_pair_r(sql, "code", frame, r_runner=runner)
    assert v.keep and v.reason == "agree" and len(calls) == 10
    runner, calls = fake_runner(lambda i: {"c": [5 if i <= 3 else 6]})
    v = r_agree.validate_pair_r(sql, "code", frame, r_runner=runner)
    assert not v.keep and v.reason == "disagree" and v.samples_agreed == 3 and len(calls) == 4
    runner, _ = fake_runner(lambda i: None)
    assert r_agree.validate_pair_r(sql, "code", frame, r_runner=runner).reason == "r_failed"
    assert r_agree.validate_pair_r("SELECT nope FROM table", "code", frame, r_runner=runner).reason == "sql_failed"


def test_r_error_lines_are_kept():
    log = "Attaching package\nError in filter(data, x > 1) :\n  object 'x' not found\nExecution halted\n"
    assert r_loop.r_messages(log) == ["Error in filter(data, x > 1) : object 'x' not found"]



def test_rlang_error_keeps_cause():
    log = ("Error in `filter()`:\n\u2139 In argument: `Access - Control == \"x\"`.\nCaused by error:\n"
           "! object 'Access' not found\nBacktrace:\n     \u2586\n  1. \u251c\u2500data %>% ...\nExecution halted\n")
    assert r_loop.r_messages(log) == ["Error in `filter()`: \u2139 In argument: `Access - Control == \"x\"`. "
                                      "Caused by error: ! object 'Access' not found"]

def test_repair_candidates_order_and_dedup():
    plan = {"sql": "s", "r": "r", "new_sql": "s2", "new_r": "r2"}
    assert [c[0] for c in r_loop.candidates(plan)] == ["new_sql", "new_r", "both"]
    assert [c[0] for c in r_loop.candidates({**plan, "new_sql": None})] == ["new_r"]


def test_score_case_repo_and_corrected():
    case = {"id": "c", "info": {"question": "q", "label": 'SELECT a FROM "table"'}}
    record = {"ok": True, "csv": '"a"\n2\n1\n'}
    r = r_eval.score_case("WikiSQL", case, "code", [[1], [2]], record)
    assert r["repo_correct"] == 0 and r["corrected_correct"] == 1
    assert r_eval.score_case("WikiSQL", case, None, [[1]], None)["error"] == "no_code"
    assert r_eval.score_case("WikiSQL", case, "code", [[1]], {"ok": False, "csv": None})["error"] == "exec_failed"


def test_run_ids_are_valid():
    from ts.tracking import check_run_id
    assert check_run_id(r_eval.run_id("final", "instruct_rts_seed1_round3", "WikiQA", "zero-shot")) == \
        "final_nlr_instruct_rts_seed1_round3_wikitq_zeroshot"
    assert r_loop.run_id_for("repairts", 2, "smoke").startswith("train_rrepairtssmoke_")
