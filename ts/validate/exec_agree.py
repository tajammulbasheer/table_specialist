"""Execution validation (paper Algorithm 3): keep a pair only if SQL and Scala agree on every row sample.

The repo's Validation/NL2DSL/NL2SQL_SparkScala/execute.py runs both languages but never compares them, so the comparison
is ours. Sampling follows the repo's prepare_exec.py: read as strings, then 10 samples at 50% with random states 0-9. SQL follows
its execute.py: each sample is written to CSV and read back with type inference before loading SQLite, so that numeric
columns compare as numbers. Spark reads the same CSV as strings.

Order of work saves time without changing any verdict: all SQL runs first because it is fast, and Scala stops at the
first disagreement because Algorithm 3 needs every sample to agree.
"""
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ts.repo import connect_read_only, load_sandbox

TABLE_NAME = re.compile(r"from\s+table\b", re.IGNORECASE)  # the repo's evaluator quotes this the same way


def row_samples(frame, count=10, fraction=0.5):
    return [frame.sample(frac=fraction, random_state=seed) for seed in range(count)]


def _cell_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _bool_as_number(text):
    # Type inference stores True as 1 in SQLite, while Spark keeps the text "True".
    return {"true": "1", "false": "0"}.get(text.lower(), text)


def cells_agree(a, b, tolerance):
    a, b = _bool_as_number(_cell_text(a)), _bool_as_number(_cell_text(b))
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= tolerance
    except ValueError:
        return False


def rows_agree(a, b, tolerance):
    return len(a) == len(b) and all(cells_agree(x, y, tolerance) for x, y in zip(a, b))


def without_empty_rows(rows):
    """Spark's CSV writer writes no line for a one-column row holding null, so those rows cannot be compared.
    An aggregate over no rows is the usual case: SQLite gives one NULL row and the Spark file has only its header."""
    return [row for row in rows if any(_cell_text(cell) for cell in row)]


def results_agree(a, b, tolerance):
    """Row order is ignored: Spark gives no order without an explicit sort, and SQLite's default order is not defined.
    Column names are ignored too, since the two languages name computed columns differently."""
    a, b = without_empty_rows(a), without_empty_rows(b)
    if len(a) != len(b):
        return False
    remaining = list(b)
    for row in a:
        match = next((i for i, other in enumerate(remaining) if rows_agree(row, other, tolerance)), None)
        if match is None:
            return False
        remaining.pop(match)
    return True


def run_sql(sql, sample):
    """Return rows, or None if the SQL fails. Model SQL only ever sees a read-only connection."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "data.db"
        sample.to_csv(Path(tmp) / "data.csv", index=False)
        loader = sqlite3.connect(db)
        try:
            pd.read_csv(Path(tmp) / "data.csv", keep_default_na=False).to_sql("table", loader, index=False)
        except (sqlite3.Error, ValueError):  # a table SQLite cannot hold cannot validate any SQL
            return None
        finally:
            loader.close()
        conn = connect_read_only(db)
        try:
            return [list(row) for row in conn.execute(TABLE_NAME.sub('FROM "table"', sql)).fetchall()]
        except (sqlite3.Error, ValueError):
            return None
        finally:
            conn.close()


def run_scala(code, sample, runner=None):
    """Return rows, or None if the program fails. Scala runs only in the repo's sandbox, in a fresh container."""
    sandbox = load_sandbox()
    runner = runner or sandbox.run_generated_code
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sample.to_csv(work / "data.csv", index=False)
        (work / "query.scala").write_text(code)
        try:
            result = runner(str(work), "scala", "query.scala", num_cpu=1)
        except sandbox.SandboxError:
            return None
        parts = list((work / "result_scala").glob("*.csv")) if (work / "result_scala").exists() else []
        if result.returncode != 0 or len(parts) != 1:
            return None
        # Spark writes a quote inside a value as \", which pandas reads only when told to.
        escape = "\\" if '\\"' in parts[0].read_text(errors="replace") else None
        try:
            return pd.read_csv(parts[0], dtype=str, keep_default_na=False, escapechar=escape).values.tolist()
        except pd.errors.EmptyDataError:
            return []
        except pd.errors.ParserError:
            return None


@dataclass
class Verdict:
    keep: bool
    reason: str
    samples_agreed: int
    scala_runs: int


def validate_pair(sql, scala, frame, count=10, fraction=0.5, tolerance=1e-6, scala_runner=None):
    samples = row_samples(frame.astype(str), count, fraction)
    sql_results = []
    for sample in samples:
        rows = run_sql(sql, sample)
        if rows is None:
            return Verdict(False, "sql_failed", 0, 0)
        sql_results.append(rows)
    # A pair that returns nothing on every sample has shown no agreement about any data.
    if not any(without_empty_rows(rows) for rows in sql_results):
        return Verdict(False, "always_empty", 0, 0)

    for index, (sample, sql_rows) in enumerate(zip(samples, sql_results)):
        scala_rows = run_scala(scala, sample, scala_runner)
        if scala_rows is None:
            return Verdict(False, "scala_failed", index, index + 1)
        if not results_agree(sql_rows, scala_rows, tolerance):
            return Verdict(False, "disagree", index, index + 1)
    return Verdict(True, "agree", count, count)
