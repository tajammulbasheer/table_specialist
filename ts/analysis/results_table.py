"""Every approach, every test set, both prompts, from the saved per-cell result files (for the final report).

Trained models: results/baseline/train_{arm}_{profile}_seed{s}_round{r}_{bench}_{prompt}.json, mean ± SD over the seeds
that have all 8 cells (n is shown). Vanilla and the vote: the self-consistency runs (greedy = vanilla, exactly), base
BIRD from the one-thread re-scoring. Repo-faithful accuracy, as everywhere else.

  python -m ts.analysis.results_table > results/analysis/results_table.md
"""
import json
import statistics as st

from ts.repo import ROOT

BENCHES = (("wikisql", "WikiSQL"), ("wikitq", "WikiTQ"), ("bird", "BIRD"), ("spider", "Spider"))
PROMPTS = ("zeroshot", "fewshot")
PAPER = {"Paper: vanilla": {"wikisql": 0.525, "wikitq": 0.300, "bird": 0.388, "spider": 0.670},
         "Paper: Table-Specialist": {"wikisql": 0.816, "wikitq": 0.449, "bird": 0.438, "spider": 0.690}}


def acc(path, field="repo_faithful_acc"):
    return json.loads(path.read_text())[field] if path.exists() else None


def trained(arm, profile, r):
    return lambda s, b, p: acc(ROOT / f"results/baseline/train_{arm}_{profile}_seed{s}_round{r}_{b}_{p}.json")


def voted(name, field, base):
    def load(s, b, p):
        tag = "sct10seq" if b == "bird" and base else "sct10"
        return acc(ROOT / f"results/baseline/final_{tag}_{name.format(s=s)}_{b}_{p}.json", field)
    return load


def rows():
    B, I = "base", "instruct"
    g, v = "greedy_repo_faithful_acc", "repo_faithful_acc"
    one = (1,)
    out = [("Base Llama-3.1-8B", None, None)]
    out += [("Vanilla", voted("vanilla", g, True), one), ("Vanilla + vote", voted("vanilla", v, True), one)]
    out += [(f"No-validation r{r}", trained("novalidation", B, r), (1, 2, 3)) for r in (1, 2, 3)]
    out += [(f"Table-Specialist r{r}", trained("tablespecialist", B, r), (1, 2, 3)) for r in (1, 2, 3)]
    out += [("bestof4 r1", trained("bestof4", B, 1), (1, 2, 3)), ("DPO r1", trained("dpo", B, 1), (1, 2, 3)),
            ("Same-data control r1", trained("tsdata", B, 1), (1, 2, 3))]
    out += [(f"Repair-TS r{r}", trained("repairts", B, r), (1, 2, 3)) for r in (1, 2, 3)]
    out += [("TS r3 + vote", voted("tablespecialist_seed{s}", v, True), (1, 2, 3)),
            ("Repair-TS r3 + vote", voted("repairts_seed{s}", v, True), (1, 2, 3))]
    out += [("Llama-3.1-8B-Instruct", None, None)]
    out += [("Vanilla", voted("instruct_vanilla", g, False), one),
            ("Vanilla + vote", voted("instruct_vanilla", v, False), one)]
    out += [(f"No-validation r{r}", trained("novalidation", I, r), (1, 2, 3)) for r in (1, 2, 3)]
    out += [(f"Table-Specialist r{r}", trained("tablespecialist", I, r), (1, 2, 3)) for r in (1, 2, 3)]
    out += [("Same-data control r1", trained("tsdata", I, 1), (1, 2, 3))]
    out += [(f"Repair-TS r{r}", trained("repairts", I, r), (1, 2, 3)) for r in (1, 2, 3)]
    out += [("TS r3 + vote", voted("instruct_tablespecialist_seed{s}", v, False), (1, 2, 3)),
            ("Repair-TS r3 + vote", voted("instruct_repairts_seed{s}", v, False), (1, 2, 3))]
    return out


def fmt(values):
    if not values:
        return "–"
    return f"{st.mean(values):.4f}" + (f" ± {st.stdev(values):.4f}" if len(values) > 1 else "")


def paper_mean(paper):
    """Exact decimal mean, rounded half up (0.59825 -> 0.5983; a float mean prints 0.5982)."""
    from decimal import ROUND_HALF_UP, Decimal
    total = sum(Decimal(str(v)) for v in paper.values()) / len(paper)
    return str(total.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def main():
    header = "| Approach | n | " + " | ".join(n for _, n in BENCHES) + " | Mean of 4 |"
    line = "|---|---|" + "---|" * (len(BENCHES) + 1)
    tables = {p: [header, line] for p in PROMPTS}
    overall = ["| Approach | n | Zero-shot mean of 4 | Few-shot mean of 4 | Mean of 8 |", "|---|---|---|---|---|"]
    for name, paper in PAPER.items():
        tables["zeroshot"].append(f"| **{name}** | | " + " | ".join(f"{paper[b]:.3f}" for b, _ in BENCHES)
                                  + f" | {paper_mean(paper)} |")
    for name, loader, seeds in rows():
        if loader is None:
            for p in PROMPTS:
                tables[p].append(f"| ***{name}*** | | | | | | |")
            overall.append(f"| ***{name}*** | | | | |")
            continue
        cells = {s: {(b, p): loader(s, b, p) for b, _ in BENCHES for p in PROMPTS} for s in seeds}
        done = [s for s in seeds if all(x is not None for x in cells[s].values())]
        if not done:
            for p in PROMPTS:
                tables[p].append(f"| {name} | 0 | " + " | ".join("–" for _ in BENCHES) + " | not finished |")
            overall.append(f"| {name} | 0 | not finished | | |")
            continue
        n = len(done) if len(seeds) > 1 else "–"
        for p in PROMPTS:
            per = [fmt([cells[s][(b, p)] for s in done]) for b, _ in BENCHES]
            mean4 = fmt([st.mean(cells[s][(b, p)] for b, _ in BENCHES) for s in done])
            tables[p].append(f"| {name} | {n} | " + " | ".join(per) + f" | {mean4} |")
        z = fmt([st.mean(cells[s][(b, "zeroshot")] for b, _ in BENCHES) for s in done])
        f = fmt([st.mean(cells[s][(b, "fewshot")] for b, _ in BENCHES) for s in done])
        a = fmt([st.mean(cells[s].values()) for s in done])
        overall.append(f"| {name} | {n} | {z} | {f} | {a} |")
    print("**Zero-shot (the repo's default test prompt; compare with the paper)**\n")
    print("\n".join(tables["zeroshot"]))
    print("\n**Few-shot (ours only)**\n")
    print("\n".join(tables["fewshot"]))
    print("\n**Summary**\n")
    print("\n".join(overall))


if __name__ == "__main__":
    main()
