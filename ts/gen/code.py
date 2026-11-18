"""Extract SQL, Scala or R from a model's response, and build the generation prompts for a table and question."""
import re

from ts.gen.prompts import nl2r_prompt, nl2scala_prompt, nl2sql_prompt

_BLOCKS = {"sql": re.compile(r"```sql(.*?)```", re.DOTALL), "scala": re.compile(r"```scala(.*?)```", re.DOTALL),
           "r": re.compile(r"```r(.*?)```", re.DOTALL)}  # the repo's nr_evaluator._get_pred pattern


def extract_code(response, language):
    """Return the first complete code block, or None. A base model keeps writing after its answer, so the first
    block is its answer. A block cut off by the token limit has no closing fence, and so counts as no code."""
    match = _BLOCKS[language].search(response or "")
    if not match:
        return None
    return match.group(1).strip() or None


def code_prompts(frame, question):
    return {"sql": nl2sql_prompt(frame, question), "scala": nl2scala_prompt(frame, question),
            "r": nl2r_prompt(frame, question)}
