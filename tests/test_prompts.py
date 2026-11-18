"""The few-shot prompts must come out exactly as the repo's scripts build them."""
import pandas as pd

from ts.gen.prompts import nl2scala_prompt, nl2sql_prompt, serialize_table
from ts.repo import REPO, repo_prompt_generator

TABLE = pd.DataFrame({"Team Name": ["Ann B", "Bob"], "Wins": [3, 1]})


def test_table_format_matches_the_repo_serializer():
    expected = repo_prompt_generator().task_descriptor.serializer.serialize_df(TABLE)
    assert serialize_table(TABLE) == expected


def test_sql_prompt_structure():
    prompt = nl2sql_prompt(TABLE, "Who won?")
    assert prompt.startswith("# Task description:\nNatural language to SQL")
    assert prompt.count('FROM "table"') == 4  # the four worked examples
    assert "|Team Name|Wins|\n|---|---|\n|Ann B|3|\n|Bob|1|\n" in prompt  # non-breaking space replaced
    assert prompt.endswith("\n\nquestion: Who won?\n\nOutput:\n")


def test_sql_prompt_text_is_in_the_repo_script():
    source = (REPO / "Generation/NL2DSL/NL2SQL/nl2sql_generation_v6.py").read_text()
    head = nl2sql_prompt(TABLE, "Q").split("|Team Name|")[0]
    assert head in source


def test_scala_prompt_structure():
    prompt = nl2scala_prompt(TABLE, "Who won?")
    assert prompt.startswith("# Task description:\nNatural language to Spark Scala")
    assert 'spark.read.option("header", "true").csv("data.csv")' in prompt
    assert prompt.rstrip().endswith("Output:")
    assert "question: Who won?" in prompt
