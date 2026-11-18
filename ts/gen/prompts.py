"""The repo's few-shot code-generation prompts, read from its scripts so that they stay byte for byte.

The repo's generation scripts run on import and hard-code paths, so the prompt text is taken from their syntax tree
instead of being copied here. Each script builds: <instructions and 4 examples> + table + "question: " + q + output.
"""
import ast
from functools import lru_cache

from ts.repo import REPO

NL2SQL_SCRIPT = "Generation/NL2DSL/NL2SQL/nl2sql_generation_v6.py"
NL2SCALA_SCRIPT = "Generation/NL2DSL/NL2SparkScala/nl2sparkscala_generation_v6.py"
NL2R_SCRIPT = "Generation/NL2DSL/NL2R/nl2r_generation_v6.py"


@lru_cache(maxsize=None)
def _template(script):
    """Return (text before the table, text after the question), in the order the script concatenates them."""
    source = (REPO / script).read_text()
    for node in ast.walk(ast.parse(source)):
        is_prompt = (isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
                     and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets))
        if not is_prompt:
            continue
        pieces = [n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        head = next((p for p in pieces if p.startswith("# Task description:")), None)
        middle = next((p for p in pieces if "question: " in p and not p.startswith("# Task")), None)
        tail = next((p for p in pieces if p.lstrip().startswith("Output:")), None)
        if head and middle and tail:
            return head, middle, tail
    raise RuntimeError(f"Repo prompt not found in {script}")


def serialize_table(frame):
    """The repo's serialize_df: pipe-delimited, with a header and a separator row."""
    text = [""] + [str(c) for c in frame.columns] + ["\n"] + ["---" for _ in frame.columns] + ["\n"]
    for row in frame.values:
        text.extend(str(x) for x in row)
        text.append("\n")
    return "|".join(text)


def few_shot_prompt(script, frame, question):
    head, middle, tail = _template(script)
    prompt = head + serialize_table(frame) + middle + question + tail
    return prompt.replace(" ", " ")  # the repo's scripts do this too


def nl2sql_prompt(frame, question):
    return few_shot_prompt(NL2SQL_SCRIPT, frame, question)


def nl2scala_prompt(frame, question):
    return few_shot_prompt(NL2SCALA_SCRIPT, frame, question)


def nl2r_prompt(frame, question):
    return few_shot_prompt(NL2R_SCRIPT, frame, question)
