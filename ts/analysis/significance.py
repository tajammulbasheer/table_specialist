"""Paired significance tests for the main comparisons in docs/RESULTS.md (for the final report).

Each model is scored on the same test cases, so the comparisons are paired by case. For a comparison A against B:
  - per test case, A's and B's correctness are averaged over the seeds both have (repo-faithful verdicts, as reported);
  - the statistic is the mean over the 8 cells (4 test sets x 2 prompts) of A's accuracy minus B's, each cell weighted
    equally, as in the "mean of 8" headline; the zero-shot mean of 4 is reported too;
  - a 95% interval and a two-sided p-value come from a paired bootstrap over test cases (cases resampled within each
    cell, 10,000 times, fixed seed);
  - the per-seed differences are listed as well, because the case bootstrap does not cover seed-to-seed variation.
Only saved predictions are read; nothing is run on a GPU.

  python -m ts.analysis.significance                # NL-to-SQL (results/analysis/significance.json)
  python -m ts.analysis.significance --task scala   # NL-to-Scala, Phase 8 (results/analysis/significance_scala.json)
"""
import json
from pathlib import Path

import numpy as np

from ts.repo import ROOT

BENCHES = ("wikisql", "spider", "wikitq", "bird")
PROMPTS = ("zeroshot", "fewshot")
CELLS = [(b, p) for b in BENCHES for p in PROMPTS]
DRAWS = 10_000


def read(path, field):
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    return {r["id"]: float(r[field]) for r in rows}


def trained(arm, profile, round_number):
    def load(seed, bench, prompt):
        return read(ROOT / f"runs/train_{arm}_{profile}_seed{seed}_round{round_number}_{bench}_{prompt}/preds.jsonl",
                    "repo_correct")
    return load


def voted(name, field, base_bird_seq):
    """Self-consistency runs: field repo_correct is the vote, greedy_repo_correct the greedy answer of the same run.
    Base BIRD comes from the one-thread re-scoring (final_sct10seq_*)."""
    def load(seed, bench, prompt):
        model = name.format(s=seed)
        tag = "sct10seq" if bench == "bird" and base_bird_seq else "sct10"
        return read(ROOT / f"runs/final_{tag}_{model}_{bench}_{prompt}/preds.jsonl", field)
    return load


def scala(arm, round_number):
    """Phase 8 NL-to-Scala cells (Instruct; repo verdicts). arm "vanilla" has no seed or round."""
    def load(seed, bench, prompt):
        model = "vanilla" if arm == "vanilla" else f"{arm}_seed{seed}_round{round_number}"
        return read(ROOT / f"runs/final_nlscala_instruct_{model}_{bench}_{prompt}/preds.jsonl", "repo_correct")
    return load


def scala_comparisons():
    van = single(scala("vanilla", None))
    out = [("Scala: TS r3 vs vanilla", scala("tablespecialist", 3), van, (1, 2, 3)),
           ("Scala: repair-TS r3 vs vanilla", scala("repairts", 3), van, (1, 2, 3)),
           ("Scala: TS r3 vs TS r1", scala("tablespecialist", 3), scala("tablespecialist", 1), (1, 2, 3))]
    out += [(f"Scala: repair-TS r{r} vs TS r{r}", scala("repairts", r), scala("tablespecialist", r), (1, 2, 3))
            for r in (1, 2, 3)]
    return out


def rlang(arm, round_number):
    """Phase 9 NL-to-R cells (Instruct; repo verdicts). arm "vanilla" has no seed or round."""
    def load(seed, bench, prompt):
        model = "vanilla" if arm == "vanilla" else f"{arm}_seed{seed}_round{round_number}"
        return read(ROOT / f"runs/final_nlr_instruct_{model}_{bench}_{prompt}/preds.jsonl", "repo_correct")
    return load


def r_comparisons():
    van = single(rlang("vanilla", None))
    out = [(f"R: TS r{r} vs vanilla", rlang("rts", r), van, (1, 2, 3)) for r in (1, 2, 3)]
    out += [("R: TS r3 vs TS r1", rlang("rts", 3), rlang("rts", 1), (1, 2, 3))]
    out += [(f"R: repair-TS r{r} vs TS r{r}", rlang("rrepairts", r), rlang("rts", r), (1, 2, 3)) for r in (1, 2, 3)]
    return out


def single(loader):
    """A model with one score (vanilla): the same predictions stand for every seed."""
    return lambda seed, bench, prompt: loader(1, bench, prompt)


def collect(loader, seeds):
    """{seed: {cell: {id: correct}}} for the seeds whose 8 cells all exist."""
    out = {}
    for seed in seeds:
        cells = {cell: loader(seed, *cell) for cell in CELLS}
        if all(v is not None for v in cells.values()):
            out[seed] = cells
    return out


def compare(name, a, b, seeds=(1, 2, 3), rng_seed=0):
    A, B = collect(a, seeds), collect(b, seeds)
    common = sorted(set(A) & set(B))
    if not common:
        return {"comparison": name, "missing": True}
    rng = np.random.default_rng(rng_seed)
    cell_diffs, boot = {}, np.zeros((DRAWS, len(CELLS)))
    for j, cell in enumerate(CELLS):
        ids = sorted(set.intersection(*[set(A[s][cell]) & set(B[s][cell]) for s in common]))
        d = np.array([np.mean([A[s][cell][i] for s in common]) - np.mean([B[s][cell][i] for s in common])
                      for i in ids])
        cell_diffs[f"{cell[0]}_{cell[1]}"] = round(float(d.mean()), 4)
        boot[:, j] = d[rng.integers(0, len(d), size=(DRAWS, len(d)))].mean(axis=1)
    zero = [j for j, c in enumerate(CELLS) if c[1] == "zeroshot"]

    def summary(values, draws):
        p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
        return {"diff": round(float(values), 4), "ci95": [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])],
                "p": round(float(min(p, 1.0)), 4)}

    per_seed = {}
    for s in common:
        da = np.mean([np.mean(list(A[s][c].values())) for c in CELLS])
        db = np.mean([np.mean(list(B[s][c].values())) for c in CELLS])
        per_seed[s] = round(float(da - db), 4)
    return {"comparison": name, "seeds": common,
            "mean_of_8": summary(np.mean(list(cell_diffs.values())), boot.mean(axis=1)),
            "zero_shot_mean_of_4": summary(np.mean([cell_diffs[f"{c[0]}_{c[1]}"] for c in CELLS if c[1] == "zeroshot"]),
                                           boot[:, zero].mean(axis=1)),
            "per_seed_mean_of_8": per_seed, "cells": cell_diffs}


def comparisons():
    b, i = "base", "instruct"
    van_b = single(voted("vanilla", "greedy_repo_correct", True))
    van_i = single(voted("instruct_vanilla", "greedy_repo_correct", False))
    return [
        ("Base: TS r3 vs vanilla", trained("tablespecialist", b, 3), van_b, (1, 2, 3)),
        ("Base: TS r3 vs NV r3", trained("tablespecialist", b, 3), trained("novalidation", b, 3), (1, 2, 3)),
        ("Base: repair-TS r3 vs TS r3", trained("repairts", b, 3), trained("tablespecialist", b, 3), (1, 2, 3)),
        ("Base: repair-TS r1 vs TS r1", trained("repairts", b, 1), trained("tablespecialist", b, 1), (1, 2, 3)),
        ("Base: repair-TS r1 vs bestof4 r1", trained("repairts", b, 1), trained("bestof4", b, 1), (1, 2, 3)),
        ("Base: same-data control r1 vs TS r1", trained("tsdata", b, 1), trained("tablespecialist", b, 1), (1, 2, 3)),
        ("Base: repair-TS r1 vs same-data control r1", trained("repairts", b, 1), trained("tsdata", b, 1), (1, 2, 3)),
        ("Base: DPO r1 vs TS r1", trained("dpo", b, 1), trained("tablespecialist", b, 1), (1, 2, 3)),
        ("Instruct: TS r3 vs vanilla", trained("tablespecialist", i, 3), van_i, (1, 2, 3)),
        ("Instruct: TS r3 vs NV r3 (seed 1)", trained("tablespecialist", i, 3), trained("novalidation", i, 3), (1,)),
        ("Instruct: TS r3 vs TS r1", trained("tablespecialist", i, 3), trained("tablespecialist", i, 1), (1, 2, 3)),
        ("Instruct: repair-TS r1 vs TS r1", trained("repairts", i, 1), trained("tablespecialist", i, 1), (1, 2, 3)),
        ("Instruct: repair-TS r2 vs TS r2", trained("repairts", i, 2), trained("tablespecialist", i, 2), (1, 2, 3)),
        ("Instruct: repair-TS r3 vs TS r3", trained("repairts", i, 3), trained("tablespecialist", i, 3), (1, 2, 3)),
        ("Instruct: same-data control r1 vs TS r1", trained("tsdata", i, 1), trained("tablespecialist", i, 1),
         (1, 2, 3)),
        ("Instruct: repair-TS r1 vs same-data control r1", trained("repairts", i, 1), trained("tsdata", i, 1),
         (1, 2, 3)),
        ("Vote vs greedy: base vanilla", single(voted("vanilla", "repo_correct", True)), van_b, (1, 2, 3)),
        ("Vote vs greedy: base TS r3", voted("tablespecialist_seed{s}", "repo_correct", True),
         voted("tablespecialist_seed{s}", "greedy_repo_correct", True), (1, 2, 3)),
        ("Vote vs greedy: base repair-TS r3", voted("repairts_seed{s}", "repo_correct", True),
         voted("repairts_seed{s}", "greedy_repo_correct", True), (1, 2, 3)),
        ("Vote vs greedy: Instruct vanilla", single(voted("instruct_vanilla", "repo_correct", False)), van_i,
         (1, 2, 3)),
        ("Vote vs greedy: Instruct TS r3", voted("instruct_tablespecialist_seed{s}", "repo_correct", False),
         voted("instruct_tablespecialist_seed{s}", "greedy_repo_correct", False), (1, 2, 3)),
        ("Vote vs greedy: Instruct repair-TS r3", voted("instruct_repairts_seed{s}", "repo_correct", False),
         voted("instruct_repairts_seed{s}", "greedy_repo_correct", False), (1, 2, 3)),
        ("Instruct: repair-TS r3 + vote vs TS r3 + vote", voted("instruct_repairts_seed{s}", "repo_correct", False),
         voted("instruct_tablespecialist_seed{s}", "repo_correct", False), (1, 2, 3)),
    ]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("sql", "scala", "r"), default="sql")
    task = ap.parse_args().task
    chosen = {"sql": comparisons, "scala": scala_comparisons, "r": r_comparisons}[task]()
    results = [compare(name, a, b, seeds) for name, a, b, seeds in chosen]
    out = ROOT / "results" / "analysis" / {"sql": "significance.json", "scala": "significance_scala.json", "r": "significance_r.json"}[task]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("| Comparison | Seeds | Mean of 8: diff [95% CI], p | Zero-shot mean of 4: diff [95% CI], p | Per seed (mean of 8) |")
    print("|---|---|---|---|---|")
    for r in results:
        if r.get("missing"):
            print(f"| {r['comparison']} | not available yet | | | |")
            continue
        m, z = r["mean_of_8"], r["zero_shot_mean_of_4"]
        fmt = lambda x: f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}], p {x['p']:.4f}"
        seeds = ", ".join(f"{v:+.4f}" for v in r["per_seed_mean_of_8"].values())
        print(f"| {r['comparison']} | {len(r['seeds'])} | {fmt(m)} | {fmt(z)} | {seeds} |")
    print(f"written {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
