"""NL-to-Scala results by round: mean of the 4 test sets per model, then mean and SD over seeds 1-3 (repo score,
corrected score in brackets). Reads results/baseline/final_nlscala_instruct_*.json only.

  python -m ts.analysis.scala_table
"""
import json
import statistics

from ts.repo import ROOT

SETS = ("wikisql", "wikitq", "spider", "bird")
PROMPTS = ("zeroshot", "fewshot")


def cell(model, test_set, prompt):
    path = ROOT / "results/baseline" / f"final_nlscala_instruct_{model}_{test_set}_{prompt}.json"
    record = json.loads(path.read_text())
    return record["repo_faithful_acc"], record["corrected_acc"]


def mean4(model, prompt, which):
    return statistics.mean(cell(model, s, prompt)[which] for s in SETS)


def rows(prompt):
    out = [("Vanilla", [mean4("vanilla", prompt, 0)], [mean4("vanilla", prompt, 1)], None)]
    for method, label in (("tablespecialist", "TS"), ("repairts", "Repair-TS")):
        for r in (1, 2, 3):
            models = [f"{method}_seed{s}_round{r}" for s in (1, 2, 3)]
            per_set = {s: statistics.mean(cell(m, s, prompt)[0] for m in models) for s in SETS}
            out.append((f"{label} r{r}", [mean4(m, prompt, 0) for m in models],
                        [mean4(m, prompt, 1) for m in models], per_set))
    return out


def fmt(values):
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def main():
    for prompt in PROMPTS:
        print(f"\n{prompt}\n| Model | WikiSQL | WikiTQ | Spider | BIRD | Mean of 4 (repo) | Per-seed | Corrected |")
        print("|---|---|---|---|---|---|---|---|")
        for label, repo, corrected, per_set in rows(prompt):
            sets = " | ".join(f"{per_set[s]:.4f}" for s in SETS) if per_set else \
                " | ".join(f"{cell('vanilla', s, prompt)[0]:.4f}" for s in SETS)
            seeds = ", ".join(f"{v:.4f}" for v in repo) if len(repo) > 1 else ""
            print(f"| {label} | {sets} | {fmt(repo)} | {seeds} | {fmt(corrected)} |")


if __name__ == "__main__":
    main()
