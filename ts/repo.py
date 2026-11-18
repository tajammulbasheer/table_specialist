"""The one place that touches third_party/Table-Specialist. The clone itself is never edited.

The repo is a set of loose scripts: Finetune/ and Evaluete/ each have their own utils.py, its paths are
"<PATH-TO-TableSpecialist>" placeholders, and its evaluator uses pandas APIs removed in pandas 3. So:
- prompt classes are imported as they are, from a path added here;
- the evaluator is compiled from source with the explicit patches listed in EVALUATOR_PATCHES, each of
  which must match or loading fails, so an upstream change cannot silently alter the repo-faithful score.
"""
import ast
import functools
import json
import os
import re
import sqlite3
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Table-Specialist"
BENCH_ROOT = ROOT / "data" / "bench" / "NL2SQL" / "benchmark" / "test"
REPO_SERIALIZER = "markdownshortline"  # default of Finetune/main_build_data.py
SQL_TIMEOUT_SECONDS = 5
REPO_SYSTEM_MESSAGE = "You are an AI assistant. Provide an answer to some data/table related questions."  # queryWithDeployment.py


SANDBOX_NOFILE = 4096


def load_sandbox():
    """Load the repo's sandbox with one patch: its limit of 256 open files stops Spark's JVM from starting,
    because Spark has more jars than that ("tzdb.dat (Too many open files)"). Every other limit is unchanged."""
    if "table_specialist_sandbox" in sys.modules:
        return sys.modules["table_specialist_sandbox"]
    source = (REPO / "table_specialist_sandbox.py").read_text()
    old = '"nofile=256:256"'
    if old not in source:
        raise RuntimeError(f"Repo sandbox changed; patch no longer applies: {old}")
    module = types.ModuleType("table_specialist_sandbox")
    module.__file__ = str(REPO / "table_specialist_sandbox.py")
    sys.modules[module.__name__] = module  # dataclasses look their module up here while it is being defined
    exec(compile(source.replace(old, f'"nofile={SANDBOX_NOFILE}:{SANDBOX_NOFILE}"'), module.__file__, "exec"),
         module.__dict__)
    return module


def repo_r_prompt_generator(random_state=1, max_token_length=4096):
    """The same as repo_prompt_generator, for the NL-to-R task (Finetune/data_generator/nl_to_r)."""
    finetune = str(REPO / "Finetune")
    if finetune not in sys.path:
        sys.path.insert(0, finetune)
    from data_generator.nl_to_r.nr_prompt_generator import NRPromptGenerator
    from data_generator.nl_to_r.nr_task_descriptor import NRTaskDescriptor

    descriptor = NRTaskDescriptor(serializer=REPO_SERIALIZER, random_state=random_state,
                                  use_random_description=False, use_cot=False, use_format_suffix=True)
    return NRPromptGenerator(task_descriptor=descriptor, max_token_length=max_token_length, min_fewshot_samples=1,
                             use_random_template=False, random_state=random_state, use_system_message=False,
                             use_cot=False)


def repo_scala_prompt_generator(random_state=1, max_token_length=4096):
    """The same as repo_prompt_generator, for the NL-to-Spark-Scala task."""
    finetune = str(REPO / "Finetune")
    if finetune not in sys.path:
        sys.path.insert(0, finetune)
    from data_generator.nl_to_sparkscala.nss_prompt_generator import NSSPromptGenerator
    from data_generator.nl_to_sparkscala.nss_task_descriptor import NSSTaskDescriptor

    descriptor = NSSTaskDescriptor(serializer=REPO_SERIALIZER, random_state=random_state,
                                   use_random_description=False, use_cot=False, use_format_suffix=True)
    return NSSPromptGenerator(task_descriptor=descriptor, max_token_length=max_token_length, min_fewshot_samples=1,
                              use_random_template=False, random_state=random_state, use_system_message=False,
                              use_cot=False)


def repo_prompt_generator(random_state=1, max_token_length=4096):
    """Build the zero-shot NL-to-SQL prompt generator with main_build_data.py's defaults for task "ns"."""
    finetune = str(REPO / "Finetune")
    if finetune not in sys.path:
        sys.path.insert(0, finetune)
    from data_generator.nl_to_sql.ns_prompt_generator import NSPromptGenerator
    from data_generator.nl_to_sql.ns_task_descriptor import NSTaskDescriptor

    descriptor = NSTaskDescriptor(serializer=REPO_SERIALIZER, random_state=random_state,
                                  use_random_description=False, use_cot=False, use_format_suffix=True)
    return NSPromptGenerator(task_descriptor=descriptor, max_token_length=max_token_length, min_fewshot_samples=1,
                             use_random_template=False, random_state=random_state, use_system_message=False,
                             use_cot=False)


@functools.lru_cache(maxsize=1_000_000, typed=True)  # BIRD results can reach millions of mostly repeated cells
def to_numeric_or_keep(value):
    """Same result as the removed pd.to_numeric(value, errors="ignore")."""
    try:
        return pd.to_numeric(value)
    except (ValueError, TypeError):
        return value


_ALLOWED_SQL = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _authorize(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED_SQL else sqlite3.SQLITE_DENY


def connect_read_only(path, timeout_seconds=SQL_TIMEOUT_SECONDS):
    """Open a database for model-written SQL: read-only, SELECT-only (no ATTACH or PRAGMA), and time-limited."""
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    conn.set_authorizer(_authorize)
    deadline = time.monotonic() + timeout_seconds
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 100_000)
    return conn


class _SafeSqlite:
    connect = staticmethod(connect_read_only)


def _evaluator_patches(bench_root):
    return {
        # pandas 3 removed applymap and errors="ignore"; object dtype keeps fillna("None") working as before.
        ".applymap(lambda x: pd.to_numeric(x, errors='ignore')).fillna(\"None\")":
            ".map(to_numeric_or_keep).astype(object).fillna(\"None\")",
        # Fill in the repo's path placeholders.
        "<PATH-TO-TableSpecialist>/Finetune/data/data_v1.0/NL2SQL/benchmark/test/WikiSQL/":
            f"{bench_root}/WikiSQL/",
        "<PATH-TO-TableSpecialist>/resources/NL2SQL/WikiSQL_processed/test/":
            f"{bench_root}/WikiSQL_tables/",
        "<PATH-TO-TableSpecialist>/Finetune/data/data_v9.0/NL2SQL/benchmark/test/spider":
            f"{bench_root}/spider",
        "<PATH-TO-TableSpecialist>/Finetune/data/data_v10.0/NL2SQL/benchmark/test":
            f"{bench_root}",
        "<PATH-TO-TableSpecialist>/Finetune/data/data_v2.0/NL2Pandas/benchmark/test/WikiQA/":
            f"{bench_root}/WikiQA/",
        "<PATH-TO-TableSpecialist>/resources/TableQuestion/WikiTableQuestions/db/test/":
            f"{bench_root}/WikiQA_db/",
        "[[pd.to_numeric(x, errors='ignore')] for x in y_true_sql.split('|')]":
            "[[to_numeric_or_keep(x)] for x in y_true_sql.split('|')]",
    }


def load_ns_module(bench_root=BENCH_ROOT):
    """Compile ns_evaluator.py's definitions and return (NSEvaluator instance, repo's are_lists_equal)."""
    source = (REPO / "Evaluete" / "evaluate" / "ns_evaluator.py").read_text()
    for old, new in _evaluator_patches(bench_root).items():
        if old not in source:
            raise RuntimeError(f"Repo evaluator changed; patch no longer applies: {old}")
        source = source.replace(old, new)
    tree = ast.parse(source)
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    namespace = {"BaseEvaluator": object, "pd": pd, "np": np, "os": os, "re": re, "json": json,
                 "sqlite3": _SafeSqlite, "to_numeric_or_keep": to_numeric_or_keep}
    exec(compile(ast.Module(body=body, type_ignores=[]), "ns_evaluator.py", "exec"), namespace)
    return namespace["NSEvaluator"](), namespace["are_lists_equal"]
