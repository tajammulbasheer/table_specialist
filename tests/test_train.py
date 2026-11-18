"""Training examples and batch size. A real short training run is exercised separately on GPU 1."""
import pandas as pd

from ts.repo import repo_prompt_generator
from ts.train.sft import accumulation_steps, build_example

TABLE = pd.DataFrame({"Team": ["a", "b"], "Wins": [1, 2]})


def test_example_uses_the_repo_prompt_and_completion_format():
    generator = repo_prompt_generator()
    example = build_example(generator, TABLE, "Who won?", 'SELECT "Team" FROM "table"')
    assert example["prompt"].startswith("# Task Description:")
    assert "**Question:**\nWho won?" in example["prompt"]
    assert example["prompt"].endswith("## Output:\n\n\n###\n\n")
    assert example["completion"] == ' ```sql\nSELECT "Team" FROM "table"\n``` '


def test_prompt_over_the_limit_gives_no_example():
    huge = pd.DataFrame({"Text": ["word " * 3000] * 2})
    assert build_example(repo_prompt_generator(), huge, "q", "SELECT 1") is None


def test_batch_is_one_percent_of_the_data():
    assert accumulation_steps(1000, 0.01) == 10
    assert accumulation_steps(1234, 0.01) == 12
    assert accumulation_steps(30, 0.01) == 1  # never below one example


def test_checkpoints_are_at_most_about_five_minutes_apart():
    from ts.train.sft import checkpoint_steps
    assert checkpoint_steps(2863, 29, 3) == 9  # no-validation, round 2: about 5 minutes
    assert checkpoint_steps(672, 7, 3) == 24  # Table-Specialist, round 3: 1/12 of the run, about 3 minutes
    assert checkpoint_steps(43, 1, 3) == 11
    assert checkpoint_steps(43, 1, 3, max_steps=2) == 1
