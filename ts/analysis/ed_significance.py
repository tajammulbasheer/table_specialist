"""Paired bootstrap tests for error detection (Phase 8): TS, repair-ED and its two controls.

The metric is not per-case accuracy but column-averaged precision and recall combined into F1 (ts.eval.ed_eval.metric,
repo-faithful verdicts), so the bootstrap resamples test columns (within each test set, the same draw for both models
and every seed) and recomputes F1 on each draw:
  - a model's F1 on a draw is the mean over its seeds of each seed's F1 (seeds run on the same columns);
  - the statistic is the mean over the two test sets (ST, RT) of A's F1 minus B's; each test set is reported too;
  - 10,000 draws, fixed seed; two-sided p-value; per-seed differences listed, since the draw does not cover seeds.
Only saved predictions are read.

  python -m ts.analysis.ed_significance   # results/analysis/significance_ed.json
"""
import json

import numpy as np

from ts.repo import ROOT

SETS = ("st", "rt")
DRAWS = 10_000


def load(model, seed, round_number, bench):
    if model == "vanilla":
        path = ROOT / f"runs/final_ed_instruct_vanilla_{bench}_zeroshot/preds.jsonl"
    else:
        path = ROOT / f"runs/train_ed_{model}_instruct_seed{seed}_round{round_number}_{bench}_zeroshot/preds.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    return {r["id"]: (r["repo"]["precision"], r["repo"]["recall"]) for r in rows}


def arrays(rows, ids):
    """precision and recall as float arrays with NaN where undefined, in ids order."""
    p = np.array([np.nan if rows[i][0] is None else rows[i][0] for i in ids])
    r = np.array([np.nan if rows[i][1] is None else rows[i][1] for i in ids])
    return p, r


def f1_draws(p, r, index):
    """F1 for each bootstrap draw (rows of index), as ed_eval.metric: means over defined values, 0 if none."""
    def mean(x):
        taken = x[index]
        defined = ~np.isnan(taken)
        n = defined.sum(axis=1)
        return np.where(n > 0, np.nansum(taken, axis=1) / np.maximum(n, 1), 0.0)
    pm, rm = mean(p), mean(r)
    return np.where(pm + rm > 0, 2 * pm * rm / np.where(pm + rm > 0, pm + rm, 1), 0.0)


def f1_point(p, r):
    return float(f1_draws(p, r, np.arange(len(p))[None, :])[0])


def compare(name, a, b, seeds=(1, 2, 3), rng_seed=0):
    """a, b: (model, round). vanilla stands for every seed."""
    rng = np.random.default_rng(rng_seed)
    out = {"comparison": name, "seeds": list(seeds), "sets": {}}
    total_draws, total_point = np.zeros(DRAWS), 0.0
    per_seed = {s: 0.0 for s in seeds}
    for bench in SETS:
        A = {s: load(a[0], s, a[1], bench) for s in seeds}
        B = {s: load(b[0], s, b[1], bench) for s in seeds}
        if any(v is None for v in list(A.values()) + list(B.values())):
            return {"comparison": name, "missing": True}
        ids = sorted(set.intersection(*[set(v) for v in list(A.values()) + list(B.values())]))
        index = rng.integers(0, len(ids), size=(DRAWS, len(ids)))
        da = np.mean([f1_draws(*arrays(A[s], ids), index) for s in seeds], axis=0)
        db = np.mean([f1_draws(*arrays(B[s], ids), index) for s in seeds], axis=0)
        pa = {s: f1_point(*arrays(A[s], ids)) for s in seeds}
        pb = {s: f1_point(*arrays(B[s], ids)) for s in seeds}
        point = np.mean(list(pa.values())) - np.mean(list(pb.values()))
        for s in seeds:
            per_seed[s] += (pa[s] - pb[s]) / len(SETS)
        out["sets"][bench] = summary(point, da - db)
        out["sets"][bench].update({"a_f1": round(float(np.mean(list(pa.values()))), 4),
                                   "b_f1": round(float(np.mean(list(pb.values()))), 4), "columns": len(ids)})
        total_draws += (da - db) / len(SETS)
        total_point += point / len(SETS)
    out["mean_of_2"] = summary(total_point, total_draws)
    out["per_seed_mean_of_2"] = {s: round(float(v), 4) for s, v in per_seed.items()}
    return out


def summary(point, draws):
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"diff": round(float(point), 4), "ci95": [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])],
            "p": round(float(min(p, 1.0)), 4)}


def comparisons():
    out = [("ED: TS r3 vs vanilla", ("edts", 3), ("vanilla", None)),
           ("ED: repair-ED r3 vs vanilla", ("edrepair", 3), ("vanilla", None)),
           ("ED: TS r3 vs TS r1", ("edts", 3), ("edts", 1))]
    out += [(f"ED: repair-ED r{r} vs TS r{r}", ("edrepair", r), ("edts", r)) for r in (1, 2, 3)]
    out += [("ED: best-of-2 r1 vs TS r1", ("edresample", 1), ("edts", 1)),
            ("ED: same-data r1 vs TS r1", ("eddata", 1), ("edts", 1)),
            ("ED: repair-ED r1 vs best-of-2 r1", ("edrepair", 1), ("edresample", 1)),
            ("ED: repair-ED r1 vs same-data r1", ("edrepair", 1), ("eddata", 1))]
    return out


def main():
    results = [compare(*c) for c in comparisons()]
    path = ROOT / "results" / "analysis" / "significance_ed.json"
    path.write_text(json.dumps(results, indent=1))
    for r in results:
        if r.get("missing"):
            print(f"{r['comparison']}: missing")
            continue
        m, st, rt = r["mean_of_2"], r["sets"]["st"], r["sets"]["rt"]
        print(f"{r['comparison']:36s} mean {m['diff']:+.4f} {m['ci95']} p={m['p']:.4f} | ST {st['diff']:+.4f} "
              f"p={st['p']:.4f} | RT {rt['diff']:+.4f} p={rt['p']:.4f} | seeds {r['per_seed_mean_of_2']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
