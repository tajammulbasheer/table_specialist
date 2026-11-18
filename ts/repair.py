"""Extension 3 (not in the paper): repair rejected pairs once, then validate them again with Algorithm 3 unchanged.

The paper discards every rejected pair. Here each rejected pair gets one repair attempt from the same generator,
guided only by what its own program did on the table: an error message, "returns no rows", or its own result.
A program never sees the other language's code or output, so agreement after repair is still two independent
programs agreeing. Which program is repaired depends on why the pair was rejected:

  sql_failed, always_empty   repair the SQL; the Scala is unchanged
  scala_failed               repair the Scala; the SQL is unchanged
  disagree                   repair both separately; try (new SQL, old Scala), (old SQL, new Scala), (new, new)
  no_code, error             not repaired

The repair prompt is the repo's generation prompt, then the failed code block, its feedback, and the repo's own
correction phrase from Finetune/data_generator/nl_to_sparkscala/nss_task_descriptor.py (get_output with an incorrect
label): "Upon review, code is incorrect and should be rewritten as follows:".

  python -m ts.repair --source runs/train_tablespecialist_base_seed1_test300/round1 --out runs/repair_pilot_seed1_test300/round1 --gpu 0 --port 8000
"""
import argparse
import json
import re
import sqlite3
import tempfile
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from ts import ext2, loop
from ts.gen.code import code_prompts, extract_code
from ts.llm.client import LLMClient
from ts.repo import ROOT, connect_read_only, load_sandbox
from ts.validate.exec_agree import TABLE_NAME, row_samples, run_sql, validate_pair, without_empty_rows

CORRECTION = "Upon review, code is incorrect and should be rewritten as follows:"
REPAIRABLE = ("sql_failed", "always_empty", "scala_failed", "disagree")
SHOWN_ROWS = 5
SHOWN_ERRORS = 3
_SCALA_ERROR = re.compile(r"^\S*query\.scala:\d+: (error: .*)$")
_EXCEPTION = re.compile(r"^((?:[a-z]+\.)+[A-Za-z]*(?:Exception|Error)\b.*)$")


def sql_error(sql, sample):
    """The SQLite message for SQL that fails on a sample, loaded exactly as exec_agree.run_sql loads it."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "data.db"
        sample.to_csv(Path(tmp) / "data.csv", index=False)
        loader = sqlite3.connect(db)
        try:
            pd.read_csv(Path(tmp) / "data.csv", keep_default_na=False).to_sql("table", loader, index=False)
        except (sqlite3.Error, ValueError) as error:
            return f"the table could not be loaded: {error}"
        finally:
            loader.close()
        conn = connect_read_only(db)
        try:
            conn.execute(TABLE_NAME.sub('FROM "table"', sql)).fetchall()
            return None
        except (sqlite3.Error, ValueError) as error:
            return str(error)
        finally:
            conn.close()


def scala_messages(log):
    """Compiler errors and exception lines from a Spark shell log, with the sandbox's file path removed."""
    found = []
    for line in log.splitlines():
        line = line.strip()
        match = _SCALA_ERROR.match(line) or _EXCEPTION.match(line)
        if match and "UnknownHostException" not in line and match.group(1) not in found:
            found.append(match.group(1))
    return found[:SHOWN_ERRORS]


def run_scala_logged(code, sample):
    """Run Scala in the repo's sandbox; return (rows or None, log). Mirrors exec_agree.run_scala."""
    sandbox = load_sandbox()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sample.to_csv(work / "data.csv", index=False)
        (work / "query.scala").write_text(code)
        try:
            result = sandbox.run_generated_code(str(work), "scala", "query.scala", num_cpu=1)
        except sandbox.SandboxError as error:
            return None, str(error)
        log = result.stdout + "\n" + result.stderr
        parts = list((work / "result_scala").glob("*.csv")) if (work / "result_scala").exists() else []
        if result.returncode != 0 or len(parts) != 1:
            return None, log
        escape = "\\" if '\\"' in parts[0].read_text(errors="replace") else None
        try:
            return pd.read_csv(parts[0], dtype=str, keep_default_na=False, escapechar=escape).values.tolist(), log
        except pd.errors.EmptyDataError:
            return [], log
        except pd.errors.ParserError:
            return None, log


def show_rows(rows):
    rows = without_empty_rows(rows or [])
    if not rows:
        return "Running this code on the table returns no rows."
    lines = ["|" + "|".join(str(c) for c in row) + "|" for row in rows[:SHOWN_ROWS]]
    more = f"\n({len(rows)} rows in total)" if len(rows) > SHOWN_ROWS else ""
    return "Running this code on the table returns:\n" + "\n".join(lines) + more


def feedback_for(pair, frame, sql_runner=run_sql, scala_runner=run_scala_logged):
    """What each program did, on the sample where the pair was rejected. Returns {language: feedback}."""
    samples = row_samples(frame.astype(str))
    reason = pair["reason"]
    if reason == "sql_failed":
        message = next((m for m in (sql_error(pair["sql"], s) for s in samples) if m), "the query failed")
        return {"sql": f"Running this code on the table gives the error: {message}"}
    if reason == "always_empty":
        return {"sql": "Running this code on the table returns no rows."}
    sample = samples[min(pair["samples_agreed"], len(samples) - 1)]
    if reason == "scala_failed":
        _, log = scala_runner(pair["scala"], sample)
        messages = scala_messages(log)
        return {"scala": "Running this code on the table gives the error: " + ("\n".join(messages) if messages
                                                                              else "the program failed")}
    if reason == "disagree":
        scala_rows, _ = scala_runner(pair["scala"], sample)
        return {"sql": show_rows(sql_runner(pair["sql"], sample)), "scala": show_rows(scala_rows)}
    return {}


def repair_prompt(language, frame, question, code, feedback):
    """The repo's prompt, the failed attempt as its answer, feedback, then the repo's correction phrase. The prompt
    ends inside an open code block, so the reply is the new code up to the closing fence."""
    return (code_prompts(frame, question)[language] + f"```{language}\n{code}\n```\n{feedback}\n{CORRECTION}\n"
            f"```{language}\n")


INSTRUCT_REPAIRER = {"path": "models/Llama-3.1-8B-Instruct", "name": "Llama-3.1-8B-Instruct"}


def chat_repair_prompt(language, frame, question, code, feedback):
    """For an instruction-tuned repairer: the same repo prompt, failed code and feedback, as one user message."""
    task = code_prompts(frame, question)[language]
    return (task + f"```{language}\n{code}\n```\n\n{feedback}\n\n"
            f"The code above is incorrect. Rewrite it so that it correctly answers the question on the input table. "
            f"No explanation, return the code only with the following markdown codeblock format : "
            f"```{language}<CODE>```.")


def parse_repair(text, language):
    return extract_code(f"```{language}\n{text}", language)


def plan_one(pair, frame_for=loop.load_table, feedback=feedback_for):
    frame = frame_for(pair["table_id"])
    try:
        notes = feedback(pair, loop.load_table_as_text(pair["table_id"]))
    except Exception as error:  # one bad table must not stop the pilot; it is recorded and not repaired
        notes, problem = {}, f"error: {type(error).__name__}: {error}"[:200]
    else:
        problem = None
    prompts = {language: repair_prompt(language, frame, pair["question"], pair[language], note)
               for language, note in notes.items()}
    return {**{k: pair[k] for k in ("table_id", "question", "sql", "scala", "reason", "samples_agreed")},
            "feedback": notes, "prompts": prompts, "feedback_error": problem}


def plan_repairs(rejected, workers=loop.VALIDATION_WORKERS, plan=plan_one):
    """One row per repairable rejected pair, with its feedback and the prompts to send. Feedback for Scala needs a
    sandbox run, so pairs are planned in parallel; the order of the input is kept."""
    todo = [p for p in rejected if p["reason"] in REPAIRABLE]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(plan, todo))


def candidates(plan):
    """Pairs to validate, in a fixed order; the first that passes is kept."""
    new_sql, new_scala = plan.get("new_sql"), plan.get("new_scala")
    order = [("new_sql", new_sql, plan["scala"]), ("new_scala", plan["sql"], new_scala), ("both", new_sql, new_scala)]
    seen, out = set(), []
    for name, sql, scala in order:
        if sql and scala and (sql, scala) != (plan["sql"], plan["scala"]) and (sql, scala) not in seen:
            seen.add((sql, scala))
            out.append((name, sql, scala))
    return out


def validate_repaired(plan, settings, validate=validate_pair):
    tried, runs = [], 0
    try:
        frame = loop.load_table_as_text(plan["table_id"])
        for name, sql, scala in candidates(plan):
            verdict = validate(sql, scala, frame, settings["row_samples"], settings["row_fraction"],
                               settings["numeric_tolerance"])
            runs += verdict.scala_runs
            tried.append({"combo": name, "reason": verdict.reason})
            if verdict.keep:
                return {**plan, "keep": True, "reason": "repaired", "combo": name, "repaired_sql": sql,
                        "repaired_scala": scala, "tried": tried, "scala_runs": runs}
    except Exception as error:
        return {**plan, "keep": False, "reason": f"error: {type(error).__name__}: {error}"[:200], "tried": tried,
                "scala_runs": runs}
    return {**plan, "keep": False, "reason": "not_repaired" if tried else "no_new_code", "tried": tried,
            "scala_runs": runs}


def summarise(validated, repaired, total):
    first = sum(r["keep"] for r in validated)
    recovered = [r for r in repaired if r["keep"]]
    by_reason = {}
    for r in repaired:
        entry = by_reason.setdefault(r["reason_before"], {"tried": 0, "recovered": 0})
        entry["tried"] += 1
        entry["recovered"] += r["keep"]
    combos = {}
    for r in recovered:
        combos[r["combo"]] = combos.get(r["combo"], 0) + 1
    return {"pairs": total, "kept_first_pass": first, "repair_attempts": len(repaired), "recovered": len(recovered),
            "kept_after_repair": first + len(recovered),
            "increase": round(len(recovered) / first, 3) if first else None,
            "by_reason": by_reason, "recovered_by_combo": combos,
            "scala_runs": sum(r.get("scala_runs", 0) for r in repaired)}


def is_new(code, old):
    return bool(code) and code.strip() != (old or "").strip()


def generate_repairs(plans, client, seed, greedy_temperature, settings, chat=False):
    """Greedy first. A base model often copies the failed code back, so a reply that is empty or identical is
    replaced by the first of up to settings["retries"] samples that differs. Records which attempt was used."""
    for language in ("sql", "scala"):
        todo = [p for p in plans if language in p["prompts"]]
        for attempt in range(settings["retries"] + 1):
            todo = [p for p in todo if not is_new(p.get(f"new_{language}"), p[language])]
            if not todo:
                break
            temperature = greedy_temperature if attempt == 0 else settings["retry_temperature"]
            key, api = ("chat_prompts", "chat") if chat else ("prompts", "completions")
            limit = settings["chat_max_tokens"][language] if chat else loop.GENERATION_LIMITS[language]
            replies = client.complete_many([p[key][language] for p in todo], api=api,
                                           temperature=temperature, max_tokens=limit,
                                           seed=seed if attempt == 0 else seed * 100 + attempt)
            for plan, reply in zip(todo, replies):
                plan[f"new_{language}"] = (extract_code(reply["text"], language) if chat
                                           else parse_repair(reply["text"], language))
                plan[f"new_{language}_response"] = reply["text"]
                plan[f"new_{language}_attempt"] = attempt
    return plans


def run(source, out, lane, seed, limit=None, repairer="base", plans_from=None, gpu_slot=None):
    """gpu_slot, if given, is called for a lane only while repairs are generated (Extension 3 rounds 2-3: GPU pool)."""
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    out.mkdir(parents=True, exist_ok=True)
    validated = loop.read_jsonl(source / "validated.jsonl")
    rejected = [r for r in validated if not r["keep"]][:limit]

    plans_path = out / "repair_plans.jsonl"
    if not plans_path.exists() and plans_from:  # the same diagnoses as an earlier pilot; that file is only read
        loop.write_jsonl(plans_path, loop.read_jsonl(plans_from / "repair_plans.jsonl")[:limit])
    if plans_path.exists():
        plans = loop.read_jsonl(plans_path)
    else:
        with ext2.validation_slot():
            plans = plan_repairs(rejected)
        loop.write_jsonl(plans_path, plans)

    generated_path = out / "repair_generated.jsonl"
    if generated_path.exists():
        plans = loop.read_jsonl(generated_path)
    else:
        chat = repairer == "instruct"
        if chat:
            for plan in plans:
                frame = loop.load_table(plan["table_id"])
                plan["chat_prompts"] = {language: chat_repair_prompt(language, frame, plan["question"], plan[language],
                                                                     note) for language, note in plan["feedback"].items()}
        model = INSTRUCT_REPAIRER if chat else {"path": loop.MODEL_PATH, "name": loop.MODEL_NAME}
        with (gpu_slot() if gpu_slot else nullcontext(lane)) as lane, \
                loop.Server({}, out / "server.log", lane, model_path=model["path"], model_name=model["name"]):
            generate_repairs(plans, LLMClient(model["name"], base_url=lane.url), seed,
                             config["generation"]["code_temperature"], config["repair"], chat=chat)
        loop.write_jsonl(generated_path, plans)

    items = [{**{k: v for k, v in p.items() if k not in ("prompts", "chat_prompts")}, "reason_before": p["reason"]} for p in plans]
    for item in items:
        item.pop("reason")
    with ext2.validation_slot() if not (out / "repaired.jsonl").exists() else nullcontext():
        repaired = loop.step_validate(out / "repaired.jsonl", items, "table_specialist", config["validation"],
                                      validate_one=validate_repaired)
    summary = summarise(validated, repaired, len(validated))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("REPAIR " + json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="a round directory with validated.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, help="smoke tests only: repair the first N rejected pairs")
    ap.add_argument("--repairer", choices=("base", "instruct"), default="base")
    ap.add_argument("--plans-from", type=Path, help="a pilot directory whose repair_plans.jsonl is reused (read only)")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    source = args.source if args.source.is_absolute() else ROOT / args.source
    if out.resolve() == source.resolve():
        raise SystemExit("--out must differ from --source; existing run directories are never overwritten")
    plans_from = args.plans_from and (args.plans_from if args.plans_from.is_absolute() else ROOT / args.plans_from)
    run(source, out, loop.Lane(args.gpu, args.gpu, args.port), args.seed, args.limit, args.repairer, plans_from)


if __name__ == "__main__":
    main()
