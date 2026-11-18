"""Question generation with the repo's textbook sampler (Generation/NL2DSL/Table2Question/table2question_generationv_v4.py).

The prompt text and requirement lists are read from that script's syntax tree, so they stay byte for byte. The script
itself cannot be imported, because it runs on import.
"""
import ast
import json
import random
import re
from functools import lru_cache

from ts.gen.prompts import serialize_table
from ts.repo import REPO

SCRIPT = "Generation/NL2DSL/Table2Question/table2question_generationv_v4.py"
JSON_OBJECT = re.compile(r"{[^}]*}")


@lru_cache(maxsize=None)
def _parts():
    """Return (predicates, requirements, [text pieces of the prompt f-string])."""
    tree = ast.parse((REPO / SCRIPT).read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("predicates", "requirements"):
                found[name] = ast.literal_eval(node.value)
            elif name == "prompt" and isinstance(node.value, ast.JoinedStr):
                found["pieces"] = [v.value for v in node.value.values if isinstance(v, ast.Constant)]
    if set(found) != {"predicates", "requirements", "pieces"} or len(found["pieces"]) != 3:
        raise RuntimeError(f"Repo question prompt not found as expected in {SCRIPT}")
    return found["predicates"], found["requirements"], found["pieces"]


def sample_requirements(rng):
    """The repo's random_requirements: predicates at 70%, each other requirement at 33%; retried until not empty."""
    predicates, requirements, _ = _parts()
    while True:
        selected = []
        for index, requirement in enumerate(requirements):
            if index == 0:
                if rng.choices([True, False], weights=[0.7, 0.3], k=1)[0]:
                    selected.append(predicates[rng.randint(0, 3)])
            elif rng.choices([True, False], weights=[0.33, 0.67], k=1)[0]:
                selected.append(requirement)
        if selected:
            return "\n".join(f"({i + 1}) {text}" for i, text in enumerate(selected))


def question_prompt(frame, requirements_text):
    head, middle, tail = _parts()[2]
    return (head + serialize_table(frame) + middle + requirements_text + tail).replace(" ", " ")


def parse_question(response):
    """Same approach as the repo's extract_res.py: the last parseable {...} wins. Returns None if there is no question."""
    parsed = None
    for match in JSON_OBJECT.findall(response or ""):
        for loader in (ast.literal_eval, json.loads):
            try:
                candidate = loader(match)
            except (ValueError, SyntaxError, json.JSONDecodeError, MemoryError, RecursionError):
                continue
            if isinstance(candidate, dict):
                parsed = candidate
            break
    if not parsed or not isinstance(parsed.get("question"), str) or not parsed["question"].strip():
        return None
    return parsed["question"].strip()


def build_requests(tables, seed):
    """Return one request per (table id, frame): the requirements are drawn in table order from a seeded generator."""
    rng = random.Random(seed)
    requests = []
    for table_id, frame in tables:
        requirements_text = sample_requirements(rng)
        requests.append({"table_id": table_id, "requirements": requirements_text,
                         "prompt": question_prompt(frame, requirements_text)})
    return requests
