"""NL-to-R validation (ours; the repo has no NL-to-R validation script): paper Algorithm 3 with R in place of Scala.

A pair (SQL, R) is kept only if both run and agree on every one of the 10 row samples (50%, random states 0-9), exactly
as ts/validate/exec_agree.validate_pair does for SQL and Scala: the same samples, the same SQL runner and the same
comparison (order and column names ignored, numeric tolerance). The samples follow the repo's own NL2R
prepare_exec.py: read as strings, df.sample(frac=0.5, random_state=0..9).

R runs only in the repo's sandbox (runtime "r", Rscript --vanilla, 15 s), in a fresh container per run. R writes
missing values as an unquoted NA; it is read as an empty cell, as SQL NULL and Spark's null are (docs/DECISIONS.md).
"""
import tempfile
from pathlib import Path

import pandas as pd

from ts.repo import load_sandbox
from ts.validate.exec_agree import Verdict, results_agree, row_samples, run_sql, without_empty_rows

R_FILE = "query.R"  # the sandbox's default file name for runtime "r"


def read_r_result(path):
    """Rows as strings; NA (R's missing value) as an empty cell. None if unreadable."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["NA"]).fillna("").values.tolist()
    except pd.errors.EmptyDataError:
        return []
    except (pd.errors.ParserError, UnicodeDecodeError):
        return None


def run_r_logged(code, sample, runner=None):
    """(rows or None, log). rows is None when the program fails or writes no result.csv."""
    sandbox = load_sandbox()
    runner = runner or sandbox.run_generated_code
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sample.to_csv(work / "data.csv", index=False)
        (work / R_FILE).write_text(code)
        try:
            result = runner(str(work), "r", R_FILE)
        except sandbox.SandboxError as error:
            return None, str(error)
        log = (result.stdout or "") + "\n" + (result.stderr or "")
        out = work / "result.csv"
        if result.returncode != 0 or not out.exists():
            return None, log
        return read_r_result(out), log


def run_r(code, sample, runner=None):
    return run_r_logged(code, sample, runner)[0]


def validate_pair_r(sql, r_code, frame, count=10, fraction=0.5, tolerance=1e-6, r_runner=None):
    """Verdict reasons: sql_failed, always_empty, r_failed, disagree, agree. Verdict.scala_runs counts R runs."""
    samples = row_samples(frame.astype(str), count, fraction)
    sql_results = []
    for sample in samples:
        rows = run_sql(sql, sample)
        if rows is None:
            return Verdict(False, "sql_failed", 0, 0)
        sql_results.append(rows)
    if not any(without_empty_rows(rows) for rows in sql_results):
        return Verdict(False, "always_empty", 0, 0)
    for index, (sample, sql_rows) in enumerate(zip(samples, sql_results)):
        r_rows = run_r(r_code, sample, r_runner)
        if r_rows is None:
            return Verdict(False, "r_failed", index, index + 1)
        if not results_agree(sql_rows, r_rows, tolerance):
            return Verdict(False, "disagree", index, index + 1)
    return Verdict(True, "agree", count, count)
