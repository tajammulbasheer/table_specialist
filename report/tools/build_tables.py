"""Generate every results table of the LaTeX report from the project's own output files.

Reads only results/baseline/*.json, results/analysis/*.json, results/corpus_manifest.json, results/ids/ and
runs/*/round*/{summary,result}.json; writes report/generated/*.tex. Nothing in a generated table is typed by hand:
a value that has no source file is printed as "--" (not run or not available), never filled in.

  python3 report/tools/build_tables.py            # from the project root
"""
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "baseline"
RUNS = ROOT / "runs"
OUT = ROOT / "report" / "generated"

BENCHES = [("wikisql", "WikiSQL"), ("wikitq", "WikiTQ"), ("spider", "Spider"), ("bird", "BIRD")]
PROMPTS = [("zeroshot", "z"), ("fewshot", "f")]
CELLS = [(b, p) for b, _ in BENCHES for p, _ in PROMPTS]
PAPER_SQL = {"wikisql": (0.525, 0.816), "wikitq": (0.300, 0.449), "spider": (0.670, 0.690), "bird": (0.388, 0.438)}
MISSING = []  # files looked for and not found, listed in generated/missing.tex


# ---------------------------------------------------------------- helpers

def tex(s):
    table = {"\\": "\\textbackslash{}", "_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$",
             "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}
    return "".join(table.get(ch, ch) for ch in str(s))


def f4(x, sign=False):
    if x is None:
        return "--"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def load(path):
    path = Path(path)
    if not path.exists():
        MISSING.append(str(path.relative_to(ROOT)))
        return None
    return json.loads(path.read_text())


def acc(prefix, bench, prompt, field="repo_faithful_acc"):
    d = load(RES / f"{prefix}_{bench}_{prompt}.json")
    return None if d is None else d.get(field)


def vanilla_base(bench, prompt, field="repo_faithful_acc"):
    if prompt == "zeroshot":
        d = load(RES / f"final_{bench}_base.json")
    else:
        d = load(RES / f"gate_{bench}_base_fewshot.json")
    return None if d is None else d.get(field)


def cells_of(prefix, field="repo_faithful_acc"):
    if prefix == "VANILLA_BASE":
        return [vanilla_base(b, p, field) for b, p in CELLS]
    if prefix == "VANILLA_INSTRUCT":
        return [acc("gate_instruct", b, p, field) for b, p in CELLS]
    return [acc(prefix, b, p, field) for b, p in CELLS]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def m8(cells):
    return statistics.mean(cells) if all(c is not None for c in cells) else None


def zs4(cells):
    zs = cells[0::2]
    return statistics.mean(zs) if all(c is not None for c in zs) else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else None


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)


def breakable_paths(text):
    """Turn \\texttt{...} in a note into \\fpath{...} (breakable), unescaping the characters LaTeX needed escaped."""
    out, i = [], 0
    while True:
        j = text.find("\\texttt{", i)
        if j < 0:
            out.append(text[i:]); break
        out.append(text[i:j]); k = j + len("\\texttt{"); depth = 1; m = k
        while depth:
            if text[m] == "\\":
                m += 2; continue
            depth += {"{": 1, "}": -1}.get(text[m], 0); m += 1
        inner = text[k:m - 1].replace("\\_", "_").replace("\\{", "{").replace("\\}", "}")
        out.append("\\fpath{" + inner + "}"); i = m
    return "".join(out)


def wrap_header(header):
    """Break multi-word header cells onto two lines (makecell), so that numeric tables stay within the text width."""
    out_rows = []
    for row in header.split("\\\\"):
        cells = []
        for cell in row.split(" & "):
            c = cell.strip()
            prefix = ""
            if c.startswith("\\rowcolor{headshade}"):
                prefix, c = "\\rowcolor{headshade}", c[len("\\rowcolor{headshade}"):]
            words = c.split(" ")
            if len(c) > 9 and len(words) > 1 and not c.startswith(("\\multicolumn", "\\makecell")) and "$" not in c:
                total, best, acc_len = len(c), 1, 0
                for i, w in enumerate(words[:-1], 1):
                    acc_len += len(w) + 1
                    if abs(acc_len - total / 2) < abs(sum(len(x) + 1 for x in words[:best]) - total / 2):
                        best = i
                c = "\\makecell{" + " ".join(words[:best]) + "\\\\" + " ".join(words[best:]) + "}"
            cells.append(prefix + c)
        out_rows.append(" & ".join(cells))
    return "\\\\".join(out_rows)


def longtable(spec, header, rows, caption, label, foot=None, size="\\small", landscape=False, colsep="4.2pt"):
    """A booktabs longtable with a repeated header and alternating row shading (\\rowcolors set in the style)."""
    import re as _re
    ncol = len(_re.sub(r"[pm]\{[^}]*\}", "p", spec.replace("@{}", "")))
    header = wrap_header(header)
    lines = [f"{{{size}", "\\rowcolors{3}{rowshade}{white}"] + ([f"\\setlength{{\\tabcolsep}}{{{colsep}}}"] if colsep else []) + [f"\\begin{{longtable}}{{{spec}}}",
             f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule", "\\rowcolor{headshade}" + header + "\\\\",
             "\\midrule", "\\endfirsthead",
             f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{\\tablename~\\thetable{{}} (continued)}}}}\\\\", "\\toprule",
             "\\rowcolor{headshade}" + header + "\\\\", "\\midrule", "\\endhead", "\\midrule",
             f"\\multicolumn{{{ncol}}}{{r}}{{\\textit{{continued on next page}}}}\\\\", "\\endfoot", "\\bottomrule"]
    lines += ["\\endlastfoot"] + [r if r.startswith("\\midrule") else r + "\\\\" for r in rows] \
        + ["\\end{longtable}", "}"]
    if foot:  # below the table, so that it never sets the table's width
        lines += ["\\vspace{-0.6\\baselineskip}", "{\\footnotesize\\color{slate}\\textit{Note.} " + breakable_paths(foot) + "\\par}",
                  "\\medskip"]
    body = "\n".join(lines) + "\n"
    if landscape:
        body = "\\begin{landscape}\n" + body + "\\end{landscape}\n"
    return body


CELL_HEADER = ("& \\multicolumn{2}{c}{WikiSQL} & \\multicolumn{2}{c}{WikiTQ} & \\multicolumn{2}{c}{Spider} "
               "& \\multicolumn{2}{c}{BIRD}")
CELL_SUB = "z & f & z & f & z & f & z & f"


def cell_row(lead, cells, extra=()):
    return " & ".join([lead] + [f4(c) for c in cells] + list(extra))


# ---------------------------------------------------------------- 1. NL-to-SQL vanilla, per test set, with intervals

def t_vanilla_sql():
    rows = []
    specs = [("Base", "completions", "zero-shot", lambda b: RES / (f"final_{b}_base.json")),
             ("Base", "completions", "few-shot", lambda b: RES / f"gate_{b}_base_fewshot.json"),
             ("Instruct", "chat + repo system message", "zero-shot", lambda b: RES / f"gate_instruct_{b}_zeroshot.json"),
             ("Instruct", "chat + repo system message", "few-shot", lambda b: RES / f"gate_instruct_{b}_fewshot.json")]
    for model, api, prompt, path in specs:
        for b, name in BENCHES:
            d = load(path(b))
            if d is None:
                continue
            ci = d.get("repo_faithful_95ci") or [None, None]
            paper = PAPER_SQL[b][0]
            diff = d["repo_faithful_acc"] - paper if prompt == "zero-shot" else None
            rows.append(" & ".join([model, prompt, name, str(d["n"]), f4(d["repo_faithful_acc"]),
                                    f"[{f4(ci[0])}, {f4(ci[1])}]", f4(d["corrected_acc"]),
                                    f"{paper:.3f}" if prompt == "zero-shot" else "", f4(diff, True)]))
    for prompt in ("zeroshot", "fewshot"):
        d = load(RES / f"gate_wikisql_base_origcase_{prompt}.json")
        if d:
            ci = d["repo_faithful_95ci"]
            rows.append(" & ".join(["Base (original-case tables)", prompt.replace("shot", "-shot"), "WikiSQL",
                                    str(d["n"]), f4(d["repo_faithful_acc"]), f"[{f4(ci[0])}, {f4(ci[1])}]",
                                    f4(d["corrected_acc"]), "", ""]))
    header = ("Model & Prompt & Test set & $n$ & Repo-faithful & 95\\% Wilson CI & Corrected & Paper vanilla "
              "& Ours $-$ paper")
    body = longtable("@{}lllrrcrrr@{}", header, rows,
                     "NL-to-SQL vanilla baselines on the four rebuilt test sets, both checkpoints and both prompts. "
                     "Execution accuracy (fraction of cases). The paper's vanilla figures (Table~3, Llama-3.1-8B) are "
                     "set beside our zero-shot rows only, because the paper does not state its test prompt.",
                     "tab:gen-vanilla-sql",
                     foot="Sources: \\texttt{results/baseline/final\\_*\\_base.json}, "
                          "\\texttt{gate\\_*\\_base\\_fewshot.json}, \\texttt{gate\\_instruct\\_*.json}, "
                          "\\texttt{gate\\_wikisql\\_base\\_origcase\\_*.json}. Base queried with raw completions "
                          "(128 tokens); Instruct through the repo's chat call (256 tokens).", size="\\footnotesize")
    write("vanilla_sql", body)


# ---------------------------------------------------------------- 2. per-seed, per-round NL-to-SQL scores

def kept_of(arm, profile, seed, r):
    d = load(RUNS / f"train_{arm}_{profile}_seed{seed}" / f"round{r}" / "summary.json")
    return None if d is None else d.get("kept")


def t_perseed_sql(profile, arms, name, caption):
    rows = []
    for arm, label in arms:
        for seed in (1, 2, 3):
            for r in (1, 2, 3):
                prefix = f"train_{arm}_{profile}_seed{seed}_round{r}"
                if not (RES / f"{prefix}_wikisql_zeroshot.json").exists():
                    continue  # never run (for example Instruct NV seeds 2-3)
                cells = cells_of(prefix)
                if all(c is None for c in cells):
                    MISSING.pop() if False else None
                    continue
                if arm == "repairts":  # training pairs = kept + recovered (result.json)
                    res = load(RUNS / f"train_repairts_{profile}_seed{seed}" / f"round{r}" / "result.json")
                    ex = res and res.get("sft_examples")
                    kept = ex.get("sql") if isinstance(ex, dict) else ex
                else:
                    kept = kept_of(arm, profile, seed, r)
                rows.append(cell_row(f"{label} & {seed} & {r}", cells,
                                     [f4(m8(cells)), f4(zs4(cells)), str(kept) if kept is not None else ""]))
        rows.append("\\midrule")
    rows = rows[:-1]
    header = "Arm & Seed & Rd " + CELL_HEADER + " & M8 & ZS4 & Kept"
    sub = "\\rowcolor{headshade} & & & " + CELL_SUB + " & & &"
    body = longtable("@{}lcc" + "r" * 8 + "rrr@{}", header + "\\\\" + sub, rows, caption, f"tab:gen-{name}",
                     foot="Repo-faithful execution accuracy; z = zero-shot, f = few-shot prompt. M8 = mean of the 8 "
                          "cells, ZS4 = mean of the 4 zero-shot cells. Kept = training pairs (validated pairs; every pair with code for NV; kept plus recovered for repair-TS). Source: \\texttt{results/baseline/train\\_<arm>\\_"
                          "<profile>\\_seed<s>\\_round<r>\\_*.json} and \\texttt{runs/.../summary.json}.")
    write(name, body.replace("\\\\\\midrule\\\\", "\\\\\\midrule"))


# ---------------------------------------------------------------- 3. round statistics over seeds

def round_stats_rows(specs):
    rows = []
    for label, prefix_fn, seeds, rounds in specs:
        for r in rounds:
            vals8, vals4 = [], []
            for s in seeds:
                pre = prefix_fn(s, r)
                if pre.startswith(("train_", "final_nlscala")) and not (RES / f"{pre}_wikisql_zeroshot.json").exists():
                    continue
                cells = cells_of(pre)
                vals8.append(m8(cells))
                vals4.append(zs4(cells))
            v8 = [v for v in vals8 if v is not None]
            v4 = [v for v in vals4 if v is not None]
            if not v8:
                continue
            rows.append(" & ".join([label, str(r), str(len(v8)), f4(mean(v8)), f4(sd(v8)), f4(min(v8)), f4(max(v8)),
                                    f4(mean(v4)), f4(sd(v4)), f4(min(v4)), f4(max(v4))]))
    return rows


def t_round_stats():
    def tr(arm, profile):
        return lambda s, r: f"train_{arm}_{profile}_seed{s}_round{r}"
    specs = [("Base vanilla", lambda s, r: "VANILLA_BASE", (1,), (0,)),
             ("Base TS", tr("tablespecialist", "base"), (1, 2, 3), (1, 2, 3)),
             ("Base NV", tr("novalidation", "base"), (1, 2, 3), (1, 2, 3)),
             ("Base repair-TS", tr("repairts", "base"), (1, 2, 3), (1, 2, 3)),
             ("Base bestof4", tr("bestof4", "base"), (1, 2, 3), (1,)),
             ("Base DPO", tr("dpo", "base"), (1, 2, 3), (1,)),
             ("Base same-data control", tr("tsdata", "base"), (1, 2, 3), (1,)),
             ("Instruct vanilla", lambda s, r: "VANILLA_INSTRUCT", (1,), (0,)),
             ("Instruct TS", tr("tablespecialist", "instruct"), (1, 2, 3), (1, 2, 3)),
             ("Instruct NV", tr("novalidation", "instruct"), (1,), (1, 2, 3)),
             ("Instruct repair-TS", tr("repairts", "instruct"), (1, 2, 3), (1, 2, 3)),
             ("Instruct same-data control", tr("tsdata", "instruct"), (1, 2, 3), (1,))]
    rows = round_stats_rows(specs)
    rows = [r.replace(" & 0 & ", " & -- & ", 1) for r in rows]
    header = ("Approach & Rd & Seeds & \\multicolumn{4}{c}{M8 over seeds} & \\multicolumn{4}{c}{ZS4 over seeds}\\\\"
              "\\rowcolor{headshade} & & & Mean & SD & Min & Max & Mean & SD & Min & Max")
    body = longtable("@{}lcc" + "r" * 8 + "@{}", header, rows,
                     "NL-to-SQL summary statistics over seeds for every approach and round. SD is the sample "
                     "standard deviation ($n-1$); it is omitted when only one seed exists. Vanilla is a single "
                     "deterministic run.", "tab:gen-round-stats",
                     foot="Computed by \\texttt{report/tools/build\\_tables.py} from the per-cell result files; "
                          "these are recalculated statistics, not values copied from the project log.")
    write("round_stats", body)


# ---------------------------------------------------------------- 4. validation outcomes per round

REASONS = ["agree", "always_empty", "disagree", "scala_failed", "sql_failed", "no_code"]


def t_validation():
    rows = []
    for profile in ("base", "instruct"):
        for arm, label in (("tablespecialist", "TS"), ("repairts", "repair-TS")):
            for seed in (1, 2, 3):
                for r in (1, 2, 3):
                    rd = RUNS / f"train_{arm}_{profile}_seed{seed}" / f"round{r}"
                    if arm == "repairts" and r == 1:
                        continue  # round 1 of repair-TS repairs TS round 1's validated pairs (same row as TS)
                    d = load(rd / ("summary.json" if arm == "tablespecialist" else "result.json"))
                    if d is None:
                        continue
                    reasons = d.get("reasons", {})
                    err = sum(v for k, v in reasons.items() if k.startswith("error"))
                    q = d.get("questions")
                    kept = d.get("kept")
                    rows.append(" & ".join([profile.capitalize(), label, str(seed), str(r), str(q),
                                            str(d.get("sql_extracted")), str(d.get("scala_extracted")),
                                            str(kept), f"{100 * kept / q:.1f}" if q else "--"]
                                           + [str(reasons.get(k, 0)) for k in REASONS[1:]] + [str(err),
                                                                                              str(d.get("scala_runs"))]))
    header = ("Model & Arm & S & R & Questions & SQL code & Scala code & Kept & Kept \\% & Always empty & Disagree "
              "& Scala failed & SQL failed & No code & Error & Scala runs")
    body = longtable("@{}llcc" + "r" * 12 + "@{}", header, rows,
                     "Algorithm~3 outcomes for every validated round (3,000 tables per round). Each question yields "
                     "one SQL--Scala pair; a pair is kept only if both programs run and agree on all 10 row samples. "
                     "Rejection reasons are mutually exclusive and are assigned in the order the validator checks "
                     "them.", "tab:gen-validation", landscape=True,
                     foot="``Error'' counts pairs stopped by an unexpected exception (all were SQLite integer "
                          "overflows). Repair-TS round~1 reuses TS round~1's validation, so it has no row of its own. "
                          "Sources: \\texttt{runs/train\\_<arm>\\_<profile>\\_seed<s>/round<r>/\\{summary,result\\}"
                          ".json}.", size="\\footnotesize")
    write("validation", body)


# ---------------------------------------------------------------- 5. timings and training statistics

def t_training():
    rows = []
    for profile in ("base", "instruct"):
        for arm, label in (("tablespecialist", "TS"), ("novalidation", "NV"), ("repairts", "repair-TS")):
            for seed in (1, 2, 3):
                for r in (1, 2, 3):
                    rd = RUNS / f"train_{arm}_{profile}_seed{seed}" / f"round{r}"
                    path = rd / ("result.json" if arm == "repairts" else "summary.json")
                    if not path.exists():
                        continue  # never run
                    d = load(path)
                    train = d.get("train")
                    if isinstance(train, dict) and "examples" in train:
                        train = {"sql": train}
                    for lang in ("sql", "scala"):
                        t = (train or {}).get(lang)
                        if not t:
                            continue
                        rows.append(" & ".join([profile.capitalize(), label, str(seed), str(r), lang.upper()
                                                if lang == "sql" else "Scala", str(t["examples"]),
                                                str(t["accumulation_steps"]), str(t["global_steps"]),
                                                f"{t['train_loss']:.4f}", f"{t['seconds'] / 60:.1f}",
                                                f"{t['peak_vram_reserved_gb']:.1f}",
                                                "yes" if t.get("resumed_from") else ""]))
    header = ("Model & Arm & S & R & Adapter & Examples & Accum. & Steps & Train loss & Minutes & Peak VRAM (GB) "
              "& Resumed")
    body = longtable("@{}llcclrrrrrrc@{}", header, rows,
                     "LoRA training runs of the NL-to-SQL study (one row per adapter). Batch = 1\\% of the round's "
                     "examples, built by gradient accumulation of single examples; 3 epochs.", "tab:gen-training",
                     foot="Resumed runs report timings and loss for the part after the restart only. Source: "
                          "\\texttt{train} field of each round's summary.", size="\\footnotesize")
    write("training", body)

    rows = []
    for profile in ("base", "instruct"):
        for arm, label in (("tablespecialist", "TS"), ("novalidation", "NV")):
            for seed in (1, 2, 3):
                for r in (1, 2, 3):
                    path = RUNS / f"train_{arm}_{profile}_seed{seed}" / f"round{r}" / "summary.json"
                    if not path.exists():
                        continue  # never run
                    d = load(path)
                    t = d.get("timings", {})
                    vals = [t.get(k) for k in ("generation_s", "validation_s", "training_s", "evaluation_s")]
                    total = sum(v for v in vals if v)
                    note = "resumed" if d.get("resumed") else ("0 = not recorded" if total == 0 else "")
                    rows.append(" & ".join([profile.capitalize(), label, str(seed), str(r)]
                                           + [f"{v / 3600:.2f}" if v else "0" for v in vals]
                                           + [f"{total / 3600:.2f}", note]))
    header = "Model & Arm & S & R & Generation & Validation & Training & Evaluation & Total & Note"
    body = longtable("@{}llcc" + "r" * 5 + "l@{}", header, rows,
                     "Wall-clock hours per stage of each NL-to-SQL round, as recorded in the round summaries.",
                     "tab:gen-timings",
                     foot="A resumed round records only the work after its restart, and a round re-run after a crash "
                          "on 2026-11-05 recorded zeros (a known defect, fixed afterwards; its original timings are in "
                          "the log). Training covers both adapters, and in GPU-pool runs includes waiting for a GPU.",
                     size="\\footnotesize")
    write("timings", body)


# ---------------------------------------------------------------- 6. repair statistics

def repair_summary(profile, seed, r):
    rd = RUNS / f"train_repairts_{profile}_seed{seed}" / f"round{r}"
    if r == 1:
        return load(rd / "summary.json")
    d = load(rd / "result.json")
    return None if d is None else d.get("repair")


def t_repair():
    rows = []
    for profile in ("base", "instruct"):
        for seed in (1, 2, 3):
            for r in (1, 2, 3):
                d = repair_summary(profile, seed, r)
                if d is None:
                    continue
                br = d.get("by_reason", {})
                combo = d.get("recovered_by_combo", {})

                def rr(k):
                    x = br.get(k)
                    return f"{x['recovered']}/{x['tried']}" if x else "--"
                rows.append(" & ".join([profile.capitalize(), str(seed), str(r), str(d["pairs"]),
                                        str(d["kept_first_pass"]), str(d["repair_attempts"]), str(d["recovered"]),
                                        f"{100 * d['increase']:.1f}", rr("disagree"), rr("scala_failed"),
                                        rr("always_empty"), rr("sql_failed"), str(combo.get("new_scala", 0)),
                                        str(combo.get("new_sql", 0)), str(combo.get("both", 0)),
                                        str(d.get("scala_runs"))]))
    header = ("Model & S & R & Pairs & Kept (1st pass) & Repaired & Recovered & Increase \\% & Disagree & Scala "
              "failed & Always empty & SQL failed & New Scala & New SQL & Both & Scala runs")
    body = longtable("@{}lcc" + "r" * 13 + "@{}", header, rows,
                     "Repair-TS (Extension~3): rejected pairs repaired once by Llama-3.1-8B-Instruct and validated "
                     "again with Algorithm~3 unchanged. Per-reason columns give recovered / attempted; the last "
                     "three give which side's new code passed.", "tab:gen-repair", landscape=True,
                     foot="``No code'' pairs are not repaired. Sources: \\texttt{runs/train\\_repairts\\_<profile>\\_"
                          "seed<s>/round1/summary.json} and \\texttt{round<r>/result.json} (field \\texttt{repair}).",
                     size="\\footnotesize")
    write("repair", body)


def t_repair_pilot():
    rows = []
    srcs = (("Base (Llama-3.1-8B)", "repair_pilot_seed1_test300"), ("Instruct", "repair_pilot_instruct_seed1_test300"))
    data = {k: load(RUNS / d / "round1" / "summary.json") for k, d in srcs}
    for reason, label in (("disagree", "SQL and Scala disagree"), ("scala_failed", "Scala failed"),
                          ("always_empty", "SQL always empty"), ("sql_failed", "SQL failed")):
        rows.append(" & ".join([label, str(data[srcs[0][0]]["by_reason"][reason]["tried"])]
                               + [str(data[k]["by_reason"][reason]["recovered"]) for k, _ in srcs]))
    rows.append(" & ".join(["\\textbf{Total}", str(data[srcs[0][0]]["repair_attempts"])]
                           + [f"\\textbf{{{data[k]['recovered']}}}" for k, _ in srcs]))
    first = data[srcs[0][0]]["kept_first_pass"]
    rows.append(" & ".join([f"Kept pairs (first pass {first})", ""]
                           + [f"{first + data[k]['recovered']} (+{100 * data[k]['recovered'] / first:.1f}\\%)"
                              for k, _ in srcs]))
    rows.append(" & ".join(["Recovered via new Scala / new SQL", ""]
                           + [f"{data[k]['recovered_by_combo'].get('new_scala', 0)} / "
                              f"{data[k]['recovered_by_combo'].get('new_sql', 0)}" for k, _ in srcs]))
    rows.append(" & ".join(["Scala runs", ""] + [str(data[k]["scala_runs"]) for k, _ in srcs]))
    write("repair_pilot", longtable("@{}lrrr@{}", "Rejected because & Attempts & Recovered: base repairer & "
                                    "Recovered: Instruct repairer", rows,
                                    "Repair pilot on seed~1's 300-table round~1 (291 questions, 41 pairs kept first "
                                    "pass): the same rejected pairs and diagnoses, repaired by base Llama-3.1-8B or by "
                                    "Llama-3.1-8B-Instruct.", "tab:gen-repair-pilot",
                                    foot="Go rule fixed beforehand: $\\geq$30\\% more kept pairs and $\\geq$80\\% "
                                         "of recovered pairs correct by hand. Source: \\texttt{runs/repair\\_pilot*"
                                         "/round1/summary.json}."))


# ---------------------------------------------------------------- 7. extension 2

def t_ext2():
    rows = []
    for seed in (1, 2, 3):
        d = load(RUNS / f"train_bestof4_base_seed{seed}" / "round1" / "summary.json")
        if d is None:
            continue
        st = d.get("dpo_pairs_by_rejected_status", {})
        rows.append(" & ".join([str(seed), str(d["questions"]), str(d["chosen_greedy"]), str(d["with_chosen"]),
                                str(d["chosen_sampled"]), f"{100 * d['chosen_sampled'] / d['chosen_greedy']:.0f}",
                                str(d["dpo_pairs"]), str(st.get("ok", 0)), str(st.get("always_empty", 0)),
                                str(st.get("sql_failed", 0)), str(d["scala_runs"])]))
    header = ("Seed & Questions & TS kept (greedy) & bestof4 chosen & From samples & Gain \\% & DPO pairs & "
              "Rejected: rows differ & always empty & SQL failed & Scala runs")
    write("ext2_labels", longtable("@{}c" + "r" * 10 + "@{}", header, rows,
                                   "Extension~2 labelling (base, round~1): 4 candidates per question (TS's greedy pair "
                                   "+ 3 vanilla samples at temperature 0.7); the first candidate that passes "
                                   "Algorithm~3 is chosen.", "tab:gen-ext2-labels", landscape=True,
                                   foot="Source: \\texttt{runs/train\\_bestof4\\_base\\_seed<s>/round1/summary.json}."))
    rows = []
    for seed in (1, 2, 3):
        for arm, label in (("tablespecialist", "TS r1"), ("novalidation", "NV r1"), ("bestof4", "bestof4"),
                           ("dpo", "DPO")):
            cells = cells_of(f"train_{arm}_base_seed{seed}_round1")
            rows.append(cell_row(f"{seed} & {label}", cells, [f4(m8(cells)), f4(zs4(cells))]))
        rows.append("\\midrule")
    header = "Seed & Model " + CELL_HEADER + " & M8 & ZS4\\\\\\rowcolor{headshade} & & " + CELL_SUB + " & &"
    body = longtable("@{}cl" + "r" * 10 + "@{}", header, rows[:-1],
                     "Extension~2 test scores per seed (base, round~1), beside TS and NV round~1 on the same "
                     "questions.", "tab:gen-ext2-scores", foot="Repo-faithful execution accuracy.")
    write("ext2_scores", body.replace("\\\\\\midrule\\\\", "\\\\\\midrule"))

    rows = []
    variants = [("refts", "TS round 1 (reference)"), ("refbestof4", "bestof4 (reference)"),
                ("refdpo", "first DPO: LR 2e-5, $\\beta$ 0.1, 2 epochs"), ("v1gentle", "v1: LR 5e-6, 1 epoch"),
                ("v2gentlebeta05", "v2: v1 + $\\beta$ 0.5"), ("v3gentlerowsdiffer", "v3: v1 + rows-differ only"),
                ("v4beta05", "v4: $\\beta$ 0.5"), ("v5verygentle", "v5: LR 1e-6, 2 epochs")]
    ref = None
    for key, label in variants:
        vals = []
        for b in ("wikisql", "spider", "wikitq"):
            d = load(RES / f"train_dpotune{key}_base_seed1_dev_{b}_zeroshot.json")
            vals.append(None if d is None else d["repo_faithful_acc"])
        mu = mean(vals) if all(v is not None for v in vals) else None
        if key == "refbestof4":
            ref = mu
        rows.append((label, vals, mu))
    out = [" & ".join([label] + [f4(v) for v in vals] + [f4(mu), f4(mu - ref if mu is not None and ref else None,
                                                                    True)]) for label, vals, mu in rows]
    header = "Model (seed 1) & WikiSQL dev & Spider dev & WikiTQ dev & Mean & $-$ bestof4"
    write("dpo_dev", longtable("@{}lrrrrr@{}", header, out,
                               "DPO tuning on dev sets only (zero-shot; WikiSQL 300, Spider single-table 411 within "
                               "the prompt limit, WikiTQ 300). Selection rule fixed beforehand: test a variant only if "
                               "its dev mean exceeds bestof4's.", "tab:gen-dpo-dev",
                               foot="No variant passed the rule. Source: \\texttt{results/baseline/train\\_dpotune*"
                                    "\\_dev\\_*.json}."))


# ---------------------------------------------------------------- 8. same-data control

def t_samedata():
    rows = []
    for profile in ("base", "instruct"):
        for seed in (1, 2, 3):
            d = load(RUNS / f"train_tsdata_{profile}_seed{seed}" / "round1" / "result.json")
            for arm, label in (("tablespecialist", "TS r1"), ("tsdata", "Control"), ("repairts", "Repair-TS r1")):
                cells = cells_of(f"train_{arm}_{profile}_seed{seed}_round1")
                pairs = ""
                if arm == "tsdata" and d:
                    pairs = f"{d['kept_ts']} + {d['extra']} (donor {d['donor_seed']})"
                rows.append(cell_row(f"{profile.capitalize()} & {seed} & {label}", cells,
                                     [f4(m8(cells)), f4(zs4(cells)), pairs]))
            rows.append("\\midrule")
    header = ("Model & S & Arm " + CELL_HEADER + " & M8 & ZS4 & Training pairs\\\\\\rowcolor{headshade} & & & "
              + CELL_SUB + " & & &")
    body = longtable("@{}lcl" + "r" * 10 + "l@{}", header, rows[:-1],
                     "Same-data control: TS round~1's kept pairs plus as many extra validated first-pass pairs (from "
                     "another seed's TS round~1, on tables this seed never used) as repair-TS round~1 recovered.",
                     "tab:gen-samedata", foot="Source: \\texttt{runs/train\\_tsdata\\_<profile>\\_seed<s>/round1/"
                                               "result.json} and the per-cell result files.", size="\\footnotesize", colsep="3.2pt")
    write("samedata", body.replace("\\\\\\midrule\\\\", "\\\\\\midrule"))


# ---------------------------------------------------------------- 9. self-consistency

def sc_cells(model, base, field):
    out = []
    for b, p in CELLS:
        if base and b == "bird":
            d = load(RES / f"final_sct10seq_{model}_{b}_{p}.json")
        else:
            d = load(RES / f"final_sct10_{model}_{b}_{p}.json")
        out.append(None if d is None else d.get(field))
    return out


def t_sc():
    rows = []
    for t in ("05", "07", "10"):
        vals, greedy = [], []
        for b in ("wikisql", "spider", "wikitq"):
            d = load(RES / f"gate_scdev_t{t}_{b}.json")
            vals.append(d and d["repo_faithful_acc"])
            greedy.append(d and d["greedy_repo_faithful_acc"])
        if t == "05":
            rows.append(" & ".join(["Greedy"] + [f4(v) for v in greedy] + [f4(mean(greedy)), ""]))
        rows.append(" & ".join([f"T {int(t) / 10:.1f}"] + [f4(v) for v in vals] + [f4(mean(vals)),
                                                                                    f4(mean(vals) - mean(greedy), True)]))
    write("sc_dev", longtable("@{}lrrrrr@{}", "Setting & WikiSQL dev & Spider dev & WikiTQ dev & Mean & $-$ greedy",
                              rows, "Self-consistency temperature tuning on dev sets (base TS round~3 seed~1 adapter, "
                                    "zero-shot, greedy + 8 samples). The highest dev mean was taken to the test sets.",
                              "tab:gen-sc-dev", foot="Source: \\texttt{results/baseline/gate\\_scdev\\_t*\\_*.json}."))
    rows = []
    models = [("vanilla", "Base vanilla", True, (None,)), ("tablespecialist", "Base TS r3", True, (1, 2, 3)),
              ("repairts", "Base repair-TS r3", True, (1, 2, 3)),
              ("instruct_vanilla", "Instruct vanilla", False, (None,)),
              ("instruct_tablespecialist", "Instruct TS r3", False, (1, 2, 3)),
              ("instruct_repairts", "Instruct repair-TS r3", False, (1, 2, 3))]
    for key, label, base, seeds in models:
        for s in seeds:
            model = key if s is None else f"{key}_seed{s}"
            g = sc_cells(model, base, "greedy_repo_faithful_acc")
            v = sc_cells(model, base, "repo_faithful_acc")
            rows.append(" & ".join([label, "--" if s is None else str(s), f4(m8(g)), f4(m8(v)),
                                    f4(m8(v) - m8(g) if m8(v) is not None and m8(g) is not None else None, True),
                                    f4(zs4(g)), f4(zs4(v))]))
    header = "Model & Seed & M8 greedy & M8 vote & Gain & ZS4 greedy & ZS4 vote"
    write("sc_test", longtable("@{}lcrrrrr@{}", header, rows,
                               "Execution-based self-consistency on the test sets (greedy + 8 samples at T = 1.0, "
                               "majority vote over execution results; failed or empty results dropped; gold never "
                               "used). ``Greedy'' is the greedy answer of the same run.", "tab:gen-sc-test",
                               foot="Base BIRD is scored on one thread (\\texttt{final\\_sct10seq\\_*}); see "
                                    "Section~\\ref{sec:fail-bird}. Source: \\texttt{results/baseline/final\\_sct10*}."))


# ---------------------------------------------------------------- 10. NL-to-Scala

def t_lang(prefix, arms, name, title, caption_extra, foot):
    rows = []
    models = [("vanilla", "Vanilla", (None,), (None,))] + [(arm, label, (1, 2, 3), (1, 2, 3)) for arm, label in arms]
    for arm, label, seeds, rounds in models:
        for s in seeds:
            for r in rounds:
                model = arm if s is None else f"{arm}_seed{s}_round{r}"
                cells = cells_of(f"{prefix}_{model}")
                corr = cells_of(f"{prefix}_{model}", "corrected_acc")
                rows.append(cell_row(f"{label} & {s or '--'} & {r or '--'}", cells,
                                     [f4(m8(cells)), f4(zs4(cells)), f4(zs4(corr))]))
        rows.append("\\midrule")
    header = ("Model & S & R " + CELL_HEADER + " & M8 & ZS4 & ZS4 corr.\\\\\\rowcolor{headshade} & & & " + CELL_SUB
              + " & & &")
    body = longtable("@{}lcc" + "r" * 11 + "@{}", header, rows[:-1],
                     f"{title}, all 19 Instruct models: execution accuracy per test set and prompt (repo-faithful, "
                     f"{caption_extra}), with the corrected zero-shot mean.", f"tab:gen-{name}", foot=foot,
                     size="\\footnotesize")
    write(name, body.replace("\\\\\\midrule\\\\", "\\\\\\midrule"))
    specs = [("Vanilla", lambda s, r: f"{prefix}_vanilla", (1,), (0,))] + \
            [(label, (lambda a: lambda s, r: f"{prefix}_{a}_seed{s}_round{r}")(arm), (1, 2, 3), (1, 2, 3))
             for arm, label in arms]
    rows = [r.replace(" & 0 & ", " & -- & ", 1) for r in round_stats_rows(specs)]
    header = ("Approach & Rd & Seeds & \\multicolumn{4}{c}{M8 over seeds} & \\multicolumn{4}{c}{ZS4 over seeds}\\\\"
              "\\rowcolor{headshade} & & & Mean & SD & Min & Max & Mean & SD & Min & Max")
    write(f"{name}_stats", longtable("@{}lcc" + "r" * 8 + "@{}", header, rows,
                                     f"{title} summary statistics over seeds (repo-faithful).", f"tab:gen-{name}-stats",
                                     foot="Recalculated from the per-cell files."))


def t_scala():
    t_lang("final_nlscala_instruct", (("tablespecialist", "TS"), ("repairts", "Repair-TS")), "scala", "NL-to-Scala",
           "following the repo's \\texttt{nss\\_evaluator}",
           "Scala adapters were trained inside the NL-to-SQL runs; no new training. "
           "Source: \\texttt{results/baseline/final\\_nlscala\\_instruct\\_*.json}.")


def t_nlr():
    t_lang("final_nlr_instruct", (("rts", "TS-R"), ("rrepairts", "Repair-TS-R")), "nlr", "NL-to-R",
           "following the repo's \\texttt{nr\\_evaluator}",
           "R adapters trained by \\texttt{ts/r\\_loop.py} (SQL--R validation). "
           "Source: \\texttt{results/baseline/final\\_nlr\\_instruct\\_*.json}.")


# ---------------------------------------------------------------- 11. error detection

def t_ed():
    rows = []
    for b, name in (("st", "ST"), ("rt", "RT")):
        d = load(RES / f"final_ed_instruct_vanilla_{b}_zeroshot.json")
        if d:
            rf, ap = d["repo_faithful"], d["autotest_protocol"]
            rows.append(" & ".join(["Vanilla", "--", "--", name, str(d["n"]), str(rf["columns_with_errors"]),
                                    str(rf["columns_predicted"]), f4(rf["precision"]), f4(rf["recall"]),
                                    f4(rf["f1"]), f4(ap["f1"])]))
    rows.append("\\midrule")
    for variant, label in (("edts", "TS"), ("edrepair", "Repair-ED"), ("edresample", "Best-of-2"),
                           ("eddata", "Same-data")):
        for seed in (1, 2, 3):
            for r in (1, 2, 3):
                for b, name in (("st", "ST"), ("rt", "RT")):
                    path = RES / f"train_ed_{variant}_instruct_seed{seed}_round{r}_{b}_zeroshot.json"
                    if not path.exists():
                        continue
                    d = json.loads(path.read_text())
                    rf, ap = d["repo_faithful"], d["autotest_protocol"]
                    rows.append(" & ".join([label, str(seed), str(r), name, str(d["n"]), str(rf["columns_with_errors"]),
                                            str(rf["columns_predicted"]), f4(rf["precision"]), f4(rf["recall"]),
                                            f4(rf["f1"]), f4(ap["f1"])]))
        rows.append("\\midrule")
    header = ("Model & S & R & Bench & Columns & With errors & Predicted & Precision & Recall & F1 (repo) "
              "& F1 (AutoTest)")
    body = longtable("@{}lccl" + "r" * 7 + "@{}", header, rows[:-1],
                     "Error detection on AutoTest (Llama-3.1-8B-Instruct, repo zero-shot prompt). Precision is "
                     "averaged over columns with a prediction, recall over columns with errors, and F1 is taken from "
                     "the two averages (the repo's \\texttt{ed\\_evaluator}); the AutoTest protocol ignores "
                     "predictions of debatable values.", "tab:gen-ed",
                     foot="The controls (best-of-2, same-data) were run for round~1 only. Source: \\texttt{results/baseline/train\\_ed\\_*.json}, "
                          "\\texttt{final\\_ed\\_instruct\\_vanilla\\_*.json}.", size="\\footnotesize")
    write("ed", body.replace("\\\\\\midrule\\\\", "\\\\\\midrule"))

    rows = []
    for variant, label in (("edts", "TS"), ("edrepair", "Repair-ED"), ("edresample", "Best-of-2"),
                           ("eddata", "Same-data")):
        for seed in (1, 2, 3):
            for r in (1, 2, 3):
                path = RUNS / f"train_{variant}_instruct_seed{seed}" / f"round{r}" / "summary.json"
                if not path.exists():
                    continue
                d = json.loads(path.read_text())
                v = d.get("validation", {})

                def kv(k):
                    x = v.get(k)
                    return f"{x['kept']}/{x['candidates']}" if x else "--"
                retry = d.get("retry", {})
                rec = (f"{retry['typo']['recovered']} + {retry['semantic']['recovered']}" if retry
                       else (str(d.get("extra")) if variant == "eddata" else "--"))
                rows.append(" & ".join([label, str(seed), str(r), kv("typo"), kv("semantic"), kv("negative"), rec,
                                        str(d.get("examples")), f"{d['train']['seconds'] / 3600:.2f}"]))
    header = ("Arm & S & R & Typo kept & Semantic kept & Clean kept & Recovered (typo + sem.) & Examples & "
              "Training h")
    write("ed_rounds", longtable("@{}lcc" + "r" * 6 + "@{}", header, rows,
                                 "Error-detection data per round: Algorithm~2 outcomes (kept / candidates; all 5 "
                                 "shuffles must agree), recovered errors for the repair and resample arms, and "
                                 "training examples.", "tab:gen-ed-rounds",
                                 foot="Clean columns come from the first 1,000 tables of the round, 2 per table. "
                                      "Source: \\texttt{runs/train\\_ed*\\_instruct\\_seed<s>/round<r>/summary.json}."))


def t_ed_summary():
    rows = []
    groups = [("Vanilla", "vanilla", (None,))] + [(label, v, (1, 2, 3)) for v, label in
                                                  (("edts", "TS"), ("edrepair", "Repair-ED"))] + \
             [(label, v, (1,)) for v, label in (("edresample", "Best-of-2"), ("eddata", "Same-data"))]
    for label, variant, rounds in groups:
        for r in rounds:
            vals = {}
            for b in ("st", "rt"):
                if variant == "vanilla":
                    d = load(RES / f"final_ed_instruct_vanilla_{b}_zeroshot.json")
                    vals[b] = [d["repo_faithful"]["f1"]] if d else []
                else:
                    vals[b] = [d["repo_faithful"]["f1"] for d in
                               (load(RES / f"train_ed_{variant}_instruct_seed{s}_round{r}_{b}_zeroshot.json")
                                for s in (1, 2, 3)) if d]
            def ms(xs):
                if not xs:
                    return "--", "--"
                return " / ".join(f4(x) for x in xs) if len(xs) > 1 else "--", \
                    f4(mean(xs)) + (f" $\\pm$ {sd(xs):.4f}" if len(xs) > 1 else "")
            st_seeds, st_m = ms(vals["st"])
            rt_seeds, rt_m = ms(vals["rt"])
            avg = f4(mean([(a + b) / 2 for a, b in zip(vals["st"], vals["rt"])])) if vals["st"] else "--"
            rows.append(" & ".join([label, str(r or "--"), st_seeds, st_m, rt_seeds, rt_m, avg]))
        rows.append("\\midrule")
    rows.insert(0, "Paper vanilla & -- & -- & 0.071 & -- & 0.108 & 0.090")
    rows.insert(1, "Paper TS & final & -- & 0.136 & -- & 0.161 & 0.148")
    rows.insert(2, "\\midrule")
    header = "Model & R & ST seeds 1 / 2 / 3 & ST mean $\\pm$ SD & RT seeds 1 / 2 / 3 & RT mean $\\pm$ SD & Avg"
    body = longtable("@{}lcrrrrr@{}", header, rows[:-1],
                     "Error detection, summary: repo-faithful F1 per seed, mean $\\pm$ sample SD over seeds, and the "
                     "mean of ST and RT.", "tab:gen-ed-summary",
                     foot="Paper figures: published Table~3, Llama-3.1-8B. Source: "
                          "\\texttt{results/baseline/train\\_ed\\_*.json}.", size="\\footnotesize")
    write("ed_summary", body.replace("\\\\\\midrule\\\\", "\\\\\\midrule"))


# ---------------------------------------------------------------- 12. significance

def t_significance():
    for fname, name, caption in (("significance.json", "significance", "NL-to-SQL"),
                                 ("significance_scala.json", "significance_scala", "NL-to-Scala"),
                                 ("significance_r.json", "significance_r", "NL-to-R")):
        data = load(ROOT / "results" / "analysis" / fname) or []
        rows = []
        for r in data:
            if r.get("missing"):
                rows.append(f"{tex(r['comparison'])} & \\multicolumn{{4}}{{l}}{{not available}}")
                continue
            m, z = r["mean_of_8"], r["zero_shot_mean_of_4"]

            def fmt(x):
                p = "$<$0.0001" if x["p"] == 0 else f"{x['p']:.4f}"
                return f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] & {p}"
            seeds = ", ".join(f"{v:+.4f}" for v in r["per_seed_mean_of_8"].values())
            rows.append(f"{tex(r['comparison'])} & {len(r['seeds'])} & {fmt(m)} & {fmt(z)} & {seeds}")
        header = "Comparison (A vs B) & Seeds & M8: $\\Delta$ [95\\% CI] & $p$ & ZS4: $\\Delta$ [95\\% CI] & $p$ & Per seed (M8)"
        write(name, longtable("@{}p{6.2cm}clrlrp{4.0cm}@{}", header, rows,
                              f"{caption}: paired bootstrap tests over test cases (10,000 draws, fixed seed). Per case, "
                              "correctness is averaged over the seeds both models have; the statistic is the mean over "
                              "cells of A $-$ B with equal cell weights. Per-seed differences show seed variation, "
                              "which the case bootstrap does not cover.", f"tab:gen-{name}", landscape=True,
                              foot=f"Source: \\texttt{{results/analysis/{tex(fname)}}}, written by "
                                   "\\texttt{ts/analysis/significance.py}.", size="\\footnotesize"))


def t_significance_ed():
    data = load(ROOT / "results" / "analysis" / "significance_ed.json") or []
    rows = []
    for r in data:
        if r.get("missing"):
            rows.append(f"{tex(r['comparison'])} & \\multicolumn{{6}}{{l}}{{not available}}")
            continue
        m, st_, rt = r["mean_of_2"], r["sets"]["st"], r["sets"]["rt"]
        p = lambda x: "$<$0.0001" if x["p"] == 0 else f"{x['p']:.4f}"
        seeds = ", ".join(f"{v:+.4f}" for v in r["per_seed_mean_of_2"].values())
        rows.append(f"{tex(r['comparison'])} & {m['diff']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}] & {p(m)} & "
                    f"{st_['diff']:+.4f} & {p(st_)} & {rt['diff']:+.4f} & {p(rt)} & {seeds}")
    header = "Comparison (A vs B) & Avg: $\\Delta$ F1 [95\\% CI] & $p$ & ST $\\Delta$ & $p$ & RT $\\Delta$ & $p$ & Per seed (avg)"
    write("significance_ed", longtable("@{}p{5.4cm}lrrrrrp{3.8cm}@{}", header, rows,
                                       "Error detection: paired bootstrap tests over test columns (10,000 draws, "
                                       "resampled within each test set, the same draw for both models and all seeds); "
                                       "F1 is recomputed on each draw, averaged over seeds, and the statistic is the "
                                       "mean over ST and RT of A $-$ B.", "tab:gen-significance-ed", landscape=True,
                                       foot="Source: \\texttt{results/analysis/significance\\_ed.json}, written by "
                                            "\\texttt{ts/analysis/ed\\_significance.py}.", size="\\footnotesize"))


# ---------------------------------------------------------------- 13. corpus and test sets

def t_data():
    m = load(ROOT / "results" / "corpus_manifest.json")
    rows = []
    if m:
        lim = m["limits"]
        rows += [f"Source & GitTables 1M licensed subset, version {m['source']['version']}, DOI "
                 f"\\texttt{{{tex(m['source']['doi'])}}}, licence {tex(m['source']['license']['id'])}",
                 f"Files / bytes & {m['source']['files']} zips / {m['source']['bytes']:,}",
                 f"Tables examined & {m['tables_examined']:,}",
                 f"Tables kept & {m['tables_kept']:,} (target {m['target']:,}; sampling seed {m['seed']})",
                 f"Columns & {lim['min_columns']}--{lim['max_columns']}",
                 f"Rows & {lim['min_rows']}--{lim['max_rows']}",
                 f"Serialised size & at most {lim['max_serialised_tokens']:,} tokens",
                 f"De-duplication & header Jaccard $\\geq$ {lim['dedupe_header_jaccard']}, at most "
                 f"{lim['max_per_header_group']} tables per header group",
                 f"Contamination check & {m['benchmark_header_sets_checked']:,} test-set header sets",
                 f"Topics used / largest topic share & {m['topics_used']} / {100 * m['largest_topic_share']:.1f}\\%"]
        for k, v in sorted(m["rejected"].items(), key=lambda kv: -kv[1]):
            rows.append(f"Rejected: {tex(k.replace('_', ' '))} & {v:,}")
    write("corpus", longtable("@{}lp{9.5cm}@{}", "Property & Value", rows,
                              "Training-table corpus built from GitTables (\\texttt{ts/corpus/build.py}).",
                              "tab:gen-corpus", foot="Source: \\texttt{results/corpus\\_manifest.json}. Rejection "
                                                     "counts are first-failing filter, so they do not sum to the "
                                                     "examined total minus the kept total exactly when a table is "
                                                     "counted under its first failing rule."))
    rows = []
    for fname, name in (("wikisql.txt", "WikiSQL"), ("wikiqa.txt", "WikiTQ"), ("spider.txt", "Spider"),
                        ("bird.txt", "BIRD"), ("corpus.txt", "Training corpus")):
        p = ROOT / "results" / "ids" / fname
        n = len([x for x in p.read_text().splitlines() if x.strip()]) if p.exists() else None
        rows.append(f"{name} & \\texttt{{results/ids/{tex(fname)}}} & {n if n is not None else '--'}")
    write("idlists", longtable("@{}llr@{}", "Set & ID list & IDs", rows,
                               "Saved case and table ID lists, so the exact sets can be rebuilt.", "tab:gen-idlists",
                               foot="Spider's and WikiTQ's lists may include cases later dropped by the prompt-length "
                                    "limit; the evaluated $n$ is in Table~\\ref{tab:gen-vanilla-sql}."))


def fs4(cells):
    fs = cells[1::2]
    return statistics.mean(fs) if all(c is not None for c in fs) else None


def t_sensitivity():
    models = [("NL-to-SQL", "Base vanilla", ["VANILLA_BASE"]),
              ("NL-to-SQL", "Base TS r3", [f"train_tablespecialist_base_seed{s}_round3" for s in (1, 2, 3)]),
              ("NL-to-SQL", "Base NV r3", [f"train_novalidation_base_seed{s}_round3" for s in (1, 2, 3)]),
              ("NL-to-SQL", "Base repair-TS r3", [f"train_repairts_base_seed{s}_round3" for s in (1, 2, 3)]),
              ("NL-to-SQL", "Instruct vanilla", ["VANILLA_INSTRUCT"]),
              ("NL-to-SQL", "Instruct TS r3", [f"train_tablespecialist_instruct_seed{s}_round3" for s in (1, 2, 3)]),
              ("NL-to-SQL", "Instruct repair-TS r3", [f"train_repairts_instruct_seed{s}_round3" for s in (1, 2, 3)]),
              ("NL-to-Scala", "Vanilla", ["final_nlscala_instruct_vanilla"]),
              ("NL-to-Scala", "TS r3", [f"final_nlscala_instruct_tablespecialist_seed{s}_round3" for s in (1, 2, 3)]),
              ("NL-to-Scala", "Repair-TS r3", [f"final_nlscala_instruct_repairts_seed{s}_round3" for s in (1, 2, 3)])]
    rows_p, rows_s = [], []
    for task, label, prefixes in models:
        rep = [cells_of(x) for x in prefixes]
        cor = [cells_of(x, "corrected_acc") for x in prefixes]
        z = mean([zs4(c) for c in rep]); f = mean([fs4(c) for c in rep])
        zc = mean([zs4(c) for c in cor]); fc = mean([fs4(c) for c in cor])
        rows_p.append(" & ".join([task, label, str(len(prefixes)), f4(z), f4(f), f4(f - z, True)]))
        sp_r = mean([c[4] for c in rep]); sp_c = mean([c[4] for c in cor])
        rows_s.append(" & ".join([task, label, f4(z), f4(zc), f4(zc - z, True), f4(sp_r), f4(sp_c),
                                  f4(sp_c - sp_r, True)]))
    write("prompt_sensitivity", longtable("@{}llcrrr@{}", "Task & Model & Seeds & Zero-shot mean of 4 & Few-shot mean "
                                          "of 4 & Few $-$ zero", rows_p,
                                          "Prompt sensitivity: the same models scored with the repository's zero-shot "
                                          "test prompt and its few-shot generation prompt (repo-faithful).",
                                          "tab:gen-prompt", foot="Means over seeds; recalculated from the per-cell "
                                                                 "files."))
    write("scoring_sensitivity", longtable("@{}llrrrrrr@{}", "Task & Model & ZS4 repo & ZS4 corr. & $\\Delta$ & "
                                           "Spider z repo & Spider z corr. & $\\Delta$", rows_s,
                                           "Scoring sensitivity: repo-faithful against corrected scores (zero-shot). "
                                           "Spider, whose gold queries often lack ORDER BY, is shown separately "
                                           "because it carries most of the difference.", "tab:gen-scoring",
                                           foot="Corrected = order-insensitive unless the gold SQL has ORDER BY, no "
                                                "early return, no division by zero, and (Scala) escape-aware CSV "
                                                "reading."))


def t_registry():
    import re
    groups = {}
    for f in sorted(RES.glob("*.json")):
        stem = re.sub(r"_(wikisql|wikitq|spider|bird)_(zeroshot|fewshot)$", "", f.stem)
        stem = re.sub(r"_(st|rt)_zeroshot$", "", stem)
        stem = re.sub(r"_dev_(wikisql|wikitq|spider)_zeroshot$", "_dev", stem)
        fam = re.sub(r"_seed\d+", "_seed<s>", stem)
        fam = re.sub(r"_round\d+", "_round<r>", fam)
        fam = re.sub(r"_t(05|07|10)_", "_t<T>_", fam)
        groups.setdefault(fam, {"cells": 0, "runs": set()})
        groups[fam]["cells"] += 1
        groups[fam]["runs"].add(stem)
    rows = [f"\\texttt{{{tex(k)}}} & {len(v['runs'])} & {v['cells']}" for k, v in sorted(groups.items())]
    write("registry", longtable("@{}p{11.2cm}rr@{}", "Run family (result-file prefix) & Runs & Result files", rows,
                                "Every result file in \\texttt{results/baseline/}, grouped by run family. A ``run'' "
                                "is one model (one seed and round); each writes one file per test set and prompt.",
                                "tab:gen-registry", foot="Generated by listing the directory; smoke and gate runs are "
                                                         "included for completeness.", size="\\footnotesize"))


def t_code():
    rows = []
    files = sorted([*(ROOT / "ts").rglob("*.py"), *(ROOT / "scripts").glob("*"), *(ROOT / "sandbox").glob("*"),
                    *(ROOT / "report" / "tools").glob("*.py")])
    total = 0
    for f in files:
        if "__pycache__" in f.parts or not f.is_file():
            continue
        text = f.read_text(errors="replace")
        n = len(text.splitlines())
        total += n
        first = next((ln.strip().strip('"').strip("#").strip() for ln in text.splitlines()
                      if ln.strip() and not ln.startswith("#!")), "")
        rows.append(f"\\texttt{{{tex(f.relative_to(ROOT))}}} & {n} & {tex(first[:110])}")
    rows.append(f"\\textbf{{Total}} & \\textbf{{{total}}} &")
    write("code", longtable("@{}p{4.6cm}rp{9.2cm}@{}", "File & Lines & First line of its docstring or comment", rows,
                            "Code inventory: every source file we wrote, with its size and the first line of its "
                            "description.", "tab:gen-code", size="\\footnotesize"))


# ---------------------------------------------------------------- 14. plot coordinates (pgfplots) and cross-check

def series(prefix_fn, seeds, rounds, metric):
    pts = []
    for r in rounds:
        vals = []
        for s in seeds:
            pre = prefix_fn(s, r)
            if not (RES / f"{pre}_wikisql_zeroshot.json").exists():
                continue
            v = metric(cells_of(pre))
            if v is not None:
                vals.append(v)
        if vals:
            pts.append(f"({r},{mean(vals):.4f}) +- (0,{(sd(vals) or 0):.4f})")
    return " ".join(pts)


def t_plots():
    tr = lambda arm, prof: (lambda s, r: f"train_{arm}_{prof}_seed{s}_round{r}")
    sc = lambda arm: (lambda s, r: f"final_nlscala_instruct_{arm}_seed{s}_round{r}")
    defs = {
        "plotBaseTS": series(tr("tablespecialist", "base"), (1, 2, 3), (1, 2, 3), m8),
        "plotBaseNV": series(tr("novalidation", "base"), (1, 2, 3), (1, 2, 3), m8),
        "plotBaseRTS": series(tr("repairts", "base"), (1, 2, 3), (1, 2, 3), m8),
        "plotInstTS": series(tr("tablespecialist", "instruct"), (1, 2, 3), (1, 2, 3), m8),
        "plotInstNV": series(tr("novalidation", "instruct"), (1,), (1, 2, 3), m8),
        "plotInstRTS": series(tr("repairts", "instruct"), (1, 2, 3), (1, 2, 3), m8),
        "plotScalaTS": series(sc("tablespecialist"), (1, 2, 3), (1, 2, 3), zs4),
        "plotScalaRTS": series(sc("repairts"), (1, 2, 3), (1, 2, 3), zs4),
    }
    for seed in (1, 2, 3):
        for b in ("st", "rt"):
            for variant, key in (("edts", "TS"), ("edrepair", "RED")):
                pts = []
                for r in (1, 2, 3):
                    d = load_opt(RES / f"train_ed_{variant}_instruct_seed{seed}_round{r}_{b}_zeroshot.json")
                    if d:
                        pts.append(f"({r},{d['repo_faithful']['f1']:.4f})")
                defs[f"plotED{key}{b.upper()}s{'abc'[seed - 1]}"] = " ".join(pts)
    van_b = cells_of("VANILLA_BASE"); van_i = cells_of("VANILLA_INSTRUCT")
    defs["valBaseVanilla"] = f"{m8(van_b):.4f}"
    defs["valInstVanilla"] = f"{m8(van_i):.4f}"
    defs["valScalaVanilla"] = f"{zs4(cells_of('final_nlscala_instruct_vanilla')):.4f}"
    # Instruct TS r3 zero-shot per test set (bar chart against the paper)
    for b, _ in BENCHES:
        vals = [acc(f"train_tablespecialist_instruct_seed{s}_round3", b, "zeroshot") for s in (1, 2, 3)]
        defs[f"valInstTSz{b}"] = f"{mean(vals):.4f}"
        defs[f"valInstTSzsd{b}"] = f"{sd(vals):.4f}"
        defs[f"valInstVanz{b}"] = f"{acc('gate_instruct', b, 'zeroshot'):.4f}"
    write("plotdata", "".join(f"\\def\\{k}{{{v}}}\n" for k, v in defs.items()))


def load_opt(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def t_crosscheck():
    """Headline values as stated in docs/RESULTS.md (typed here from that file), against values recomputed from the
    per-cell files. A mismatch is printed in the report, never corrected silently."""
    tr = lambda arm, prof, s, r: f"train_{arm}_{prof}_seed{s}_round{r}"
    def m(arm, prof, r, metric=m8, seeds=(1, 2, 3)):
        return mean([metric(cells_of(tr(arm, prof, s, r))) for s in seeds])
    checks = [
        ("Base vanilla, M8", 0.4788, m8(cells_of("VANILLA_BASE"))),
        ("Base TS r1 / r2 / r3, M8 (r3)", 0.5435, m("tablespecialist", "base", 3)),
        ("Base TS r1, M8", 0.5266, m("tablespecialist", "base", 1)),
        ("Base NV r3, M8", 0.4943, m("novalidation", "base", 3)),
        ("Base NV r1, M8", 0.5401, m("novalidation", "base", 1)),
        ("Base repair-TS r3, M8", 0.5856, m("repairts", "base", 3)),
        ("Base repair-TS r3, ZS4", 0.5904, m("repairts", "base", 3, zs4)),
        ("Base TS r3, ZS4", 0.5460, m("tablespecialist", "base", 3, zs4)),
        ("Base bestof4 r1, M8", 0.5330, m("bestof4", "base", 1)),
        ("Base DPO r1, M8", 0.5057, m("dpo", "base", 1)),
        ("Base same-data control r1, M8", 0.5385, m("tsdata", "base", 1)),
        ("Instruct vanilla, ZS4", 0.5001, zs4(cells_of("VANILLA_INSTRUCT"))),
        ("Instruct TS r3, ZS4", 0.6113, m("tablespecialist", "instruct", 3, zs4)),
        ("Instruct TS r3, M8", 0.6069, m("tablespecialist", "instruct", 3)),
        ("Instruct NV r3 (seed 1), ZS4", 0.5852, m("novalidation", "instruct", 3, zs4, (1,))),
        ("Instruct repair-TS r3, ZS4", 0.6122, m("repairts", "instruct", 3, zs4)),
        ("Instruct same-data control r1, M8", 0.6231, m("tsdata", "instruct", 1)),
        ("Scala TS r3, ZS4", 0.4666, mean([zs4(cells_of(f"final_nlscala_instruct_tablespecialist_seed{s}_round3"))
                                            for s in (1, 2, 3)])),
        ("Scala repair-TS r3, ZS4", 0.4817, mean([zs4(cells_of(f"final_nlscala_instruct_repairts_seed{s}_round3"))
                                                   for s in (1, 2, 3)])),
        ("Scala vanilla, ZS4", 0.1936, zs4(cells_of("final_nlscala_instruct_vanilla"))),
        ("Base repair-TS r3 + vote, M8", 0.6015, mean([m8(sc_cells(f"repairts_seed{s}", True, "repo_faithful_acc"))
                                                        for s in (1, 2, 3)])),
        ("Instruct repair-TS r3 + vote, ZS4", 0.6300, mean([zs4(sc_cells(f"instruct_repairts_seed{s}", False,
                                                                           "repo_faithful_acc")) for s in (1, 2, 3)])),
        ("ED TS r3 ST F1, mean of 3 seeds", 0.2785, mean([load_opt(RES / f"train_ed_edts_instruct_seed{s}_round3_st"
                                                                   "_zeroshot.json")["repo_faithful"]["f1"]
                                                          for s in (1, 2, 3)])),
        ("ED TS r3 RT F1, mean of 3 seeds", 0.2979, mean([load_opt(RES / f"train_ed_edts_instruct_seed{s}_round3_rt"
                                                                   "_zeroshot.json")["repo_faithful"]["f1"]
                                                          for s in (1, 2, 3)])),
    ]
    rows, bad = [], 0
    for label, stated, computed in checks:
        ok = computed is not None and abs(round(computed, 4) - stated) <= 0.00011
        bad += not ok
        rows.append(f"{tex(label)} & {stated:.4f} & {f4(computed)} & "
                    + ("\\textcolor{good}{match}" if ok else "\\textcolor{bad}{\\textbf{mismatch}}"))
    write("crosscheck", longtable("@{}lrrl@{}", "Headline value & Stated in project log & Recomputed & Check", rows,
                                  "Cross-check of headline values: the value stated in \\texttt{docs/RESULTS.md} "
                                  "against the value recomputed from the per-cell result files by the table generator. "
                                  "A tolerance of 0.0001 allows for rounding of means.", "tab:gen-crosscheck"))
    print(f"cross-check: {len(checks) - bad} of {len(checks)} match")


def main():
    t_vanilla_sql()
    t_perseed_sql("base", [("tablespecialist", "TS"), ("novalidation", "NV")], "perseed_core_base",
                  "Core study on Llama-3.1-8B base: every seed and round of Table-Specialist (TS) and no-validation "
                  "(NV).")
    t_perseed_sql("instruct", [("tablespecialist", "TS"), ("novalidation", "NV")], "perseed_core_instruct",
                  "Core study on Llama-3.1-8B-Instruct: TS on seeds 1--3 and NV on seed~1, every round.")
    t_perseed_sql("base", [("repairts", "Repair-TS")], "perseed_repair_base",
                  "Repair-TS on base, every seed and round (repairs written by Llama-3.1-8B-Instruct).")
    t_perseed_sql("instruct", [("repairts", "Repair-TS")], "perseed_repair_instruct",
                  "Repair-TS on Instruct, every seed and round (the Instruct model is generator and repairer).")
    t_round_stats()
    t_validation()
    t_training()
    t_repair()
    t_repair_pilot()
    t_ext2()
    t_samedata()
    t_sc()
    t_scala()
    t_nlr()
    t_ed()
    t_ed_summary()
    t_significance()
    t_significance_ed()
    t_data()
    t_sensitivity()
    t_registry()
    t_code()
    t_plots()
    t_crosscheck()
    stamp = datetime.now(timezone.utc)
    ist = stamp.timestamp() + 5.5 * 3600
    write("meta", "\\newcommand{\\tablesgenerated}{" + datetime.fromtimestamp(ist, timezone.utc).strftime(
        "%Y-%m-%d %H:%M") + " IST}\n")
    missing = sorted(set(MISSING))
    write("missing", "\\begin{itemize}\\small\n" + "".join(f"\\item \\texttt{{{tex(m)}}}\n" for m in missing)
          + ("\\item (none)\n" if not missing else "") + "\\end{itemize}\n")
    print(f"wrote {len(list(OUT.glob('*.tex')))} files; {len(missing)} expected files not found")
    for m_ in missing:
        print("  missing:", m_)


if __name__ == "__main__":
    main()
