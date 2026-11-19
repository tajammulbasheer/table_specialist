"""Write the pgfplots sources for the README charts (assets/figures_src/*.tex) from the project's result files.

Reads results/baseline/*.json and results/analysis/*.json through report/tools/build_tables.py, plus
runs/train_tablespecialist_*/round*/summary.json for the validation outcomes. Paper figures are constants
from the published paper (Table 3, Llama-3.1-8B). Build the PNGs with scripts/build_readme_figures.sh.

  .venv/bin/python scripts/make_readme_figures.py        # from the project root
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report" / "tools"))
import build_tables as bt  # noqa: E402

OUT = ROOT / "assets" / "figures_src"
SEEDS = (1, 2, 3)
ROUNDS = (1, 2, 3)
SETS = [("wikisql", "WikiSQL"), ("wikitq", "WikiTQ"), ("spider", "Spider"), ("bird", "BIRD")]

# Published paper, Table 3, Llama-3.1-8B: (vanilla, Table-Specialist) per test set, zero-shot.
PAPER = {
    "sql": {"wikisql": (0.525, 0.816), "wikitq": (0.300, 0.449), "spider": (0.670, 0.690), "bird": (0.388, 0.438)},
    "scala": {"wikisql": (0.359, 0.728), "wikitq": (0.043, 0.188), "spider": (0.249, 0.356), "bird": (0.100, 0.228)},
    "r": {"wikisql": (0.331, 0.409), "wikitq": (0.138, 0.257), "spider": (0.503, 0.536), "bird": (0.225, 0.261)},
    "ed": {"st": (0.071, 0.136), "rt": (0.108, 0.161)},
}

PRE = r"""\documentclass[border=6pt]{standalone}
\usepackage{fontspec}
\usepackage[sfdefault,lining]{FiraSans}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{tsnavy}{HTML}{1A2E4F}\definecolor{tsred}{HTML}{A6252D}\definecolor{tsink}{HTML}{1F2430}
\definecolor{tsgrey}{HTML}{5F6B7C}\definecolor{tsmuted}{HTML}{A7B2C2}\definecolor{tsline}{HTML}{D6DCE5}
\definecolor{cRepair}{HTML}{1B7F9E}\definecolor{cControl}{HTML}{C98A2B}
\definecolor{cVanOurs}{HTML}{F2C3C6}\definecolor{cVanPaper}{HTML}{8AA4CC}
\pgfplotsset{rd/.style={axis lines*=left, ymajorgrids, grid style={tsline,line width=0.4pt},
  axis line style={tsgrey,line width=0.5pt}, major tick style={draw=none},
  tick label style={font=\small,text=tsgrey,/pgf/number format/assume math mode=true}, label style={font=\small,text=tsink},
  title style={font=\bfseries\color{tsnavy}},
  legend style={draw=none,fill=none,font=\small,text=tsink,/tikz/every even column/.append style={column sep=10pt}},
  legend cell align=left, every axis plot/.append style={line width=0.5pt}}}
\begin{document}
\pagecolor{white}
\begin{tikzpicture}
"""
POST = "\\end{tikzpicture}\n\\end{document}\n"


def ms(values):
    vals = [v for v in values if v is not None]
    return bt.mean(vals), (bt.sd(vals) or 0.0)


def over_seeds(prefix_fn, metric, seeds=SEEDS):
    return ms([metric(bt.cells_of(prefix_fn(s))) for s in seeds
               if (bt.RES / f"{prefix_fn(s)}_wikisql_zeroshot.json").exists()])


def per_set_zs(prefix_fn, bench, seeds=SEEDS):
    return ms([bt.acc(prefix_fn(s), bench, "zeroshot") for s in seeds])


def ed_f1(variant, r, b):
    if variant == "vanilla":
        d = bt.load(bt.RES / f"final_ed_instruct_vanilla_{b}_zeroshot.json")
        return [d["repo_faithful"]["f1"]]
    return [d["repo_faithful"]["f1"] for d in
            (bt.load(bt.RES / f"train_ed_{variant}_instruct_seed{s}_round{r}_{b}_zeroshot.json") for s in SEEDS) if d]


def coords(pairs, err=False):
    if err:
        return " ".join(f"({x},{y:.4f}) +- (0,{e:.4f})" for x, y, e in pairs)
    return " ".join(f"({x},{y:.4f})" for x, y in pairs)


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(PRE + body + POST)
    print("wrote", (OUT / f"{name}.tex").relative_to(ROOT))


# ---------------------------------------------------------------- 1. NL-to-SQL Instruct vs the paper, per test set

def fig_sql_vs_paper():
    ts = lambda s: f"train_tablespecialist_instruct_seed{s}_round3"
    van, pv, our, pt = [], [], [], []
    for b, lab in SETS:
        van.append((lab, bt.acc("gate_instruct", b, "zeroshot")))
        pv.append((lab, PAPER["sql"][b][0]))
        m, e = per_set_zs(ts, b)
        our.append((lab, m, e))
        pt.append((lab, PAPER["sql"][b][1]))
    m, e = over_seeds(ts, bt.zs4)
    van.append(("Mean of 4", bt.zs4(bt.cells_of("VANILLA_INSTRUCT"))))
    pv.append(("Mean of 4", bt.mean([v[0] for v in PAPER["sql"].values()])))
    our.append(("Mean of 4", m, e))
    pt.append(("Mean of 4", bt.mean([v[1] for v in PAPER["sql"].values()])))
    body = r"""\begin{axis}[rd, ybar, bar width=9pt, width=15cm, height=7cm, ymin=0, ymax=1,
  symbolic x coords={WikiSQL,WikiTQ,Spider,BIRD,Mean of 4}, xtick=data, enlarge x limits=0.12,
  ylabel={Execution accuracy (zero-shot)}, ytick={0,0.2,...,1.0}, y tick label style={/pgf/number format/.cd,fixed,fixed zerofill,precision=1},
  error bars/error bar style={tsink,line width=0.6pt}, error bars/error mark options={rotate=90,mark size=2pt,tsink},
  legend style={at={(0.5,1.03)},anchor=south,legend columns=4}]
"""
    body += f"\\addplot[area legend,fill=cVanOurs,draw=tsred!45] coordinates {{{coords(van)}}};\n"
    body += f"\\addplot[area legend,fill=cVanPaper,draw=cVanPaper] coordinates {{{coords(pv)}}};\n"
    body += ("\\addplot[area legend,fill=tsred,draw=tsred,error bars/.cd,y dir=both,y explicit] coordinates "
             f"{{{coords(our, err=True)}}};\n")
    body += f"\\addplot[area legend,fill=tsnavy,draw=tsnavy] coordinates {{{coords(pt)}}};\n"
    body += "\\legend{Our vanilla,Paper vanilla,Our Table-Specialist (round 3),Paper Table-Specialist}\n\\end{axis}\n"
    write("sql_vs_paper", body)


# ---------------------------------------------------------------- 2. all four tasks, vanilla / TS / repair vs the paper

def fig_all_tasks():
    sc = lambda arm: (lambda s: f"final_nlscala_instruct_{arm}_seed{s}_round3")
    nr = lambda arm: (lambda s: f"final_nlr_instruct_{arm}_seed{s}_round3")
    tr = lambda arm: (lambda s: f"train_{arm}_instruct_seed{s}_round3")
    rows = {
        "NL-to-SQL": (bt.zs4(bt.cells_of("VANILLA_INSTRUCT")), PAPER["sql"],
                      over_seeds(tr("tablespecialist"), bt.zs4), over_seeds(tr("repairts"), bt.zs4)),
        "NL-to-Scala": (bt.zs4(bt.cells_of("final_nlscala_instruct_vanilla")), PAPER["scala"],
                        over_seeds(sc("tablespecialist"), bt.zs4), over_seeds(sc("repairts"), bt.zs4)),
        "NL-to-R": (bt.zs4(bt.cells_of("final_nlr_instruct_vanilla")), PAPER["r"],
                    over_seeds(nr("rts"), bt.zs4), over_seeds(nr("rrepairts"), bt.zs4)),
    }
    ed = {}
    for v in ("vanilla", "edts", "edrepair"):
        st, rt = ed_f1(v, 3, "st"), ed_f1(v, 3, "rt")
        ed[v] = ms([(a + b) / 2 for a, b in zip(st, rt)])
    rows["Error detection"] = (ed["vanilla"][0], PAPER["ed"], ed["edts"], ed["edrepair"])
    series = {k: [] for k in ("van", "pv", "pt", "ts", "rep")}
    for task, (van, paper, ts, rep) in rows.items():
        series["van"].append((task, van))
        series["pv"].append((task, bt.mean([v[0] for v in paper.values()])))
        series["pt"].append((task, bt.mean([v[1] for v in paper.values()])))
        series["ts"].append((task, ts[0], ts[1]))
        series["rep"].append((task, rep[0], rep[1]))
    body = r"""\begin{axis}[rd, ybar, bar width=10pt, width=15cm, height=7cm, ymin=0, ymax=0.8,
  symbolic x coords={NL-to-SQL,NL-to-Scala,NL-to-R,Error detection}, xtick=data, enlarge x limits=0.15,
  ylabel={Zero-shot mean of 4 test sets (F1 for errors)}, ytick={0,0.2,...,0.8}, y tick label style={/pgf/number format/.cd,fixed,fixed zerofill,precision=1},
  error bars/error bar style={tsink,line width=0.6pt}, error bars/error mark options={rotate=90,mark size=2pt,tsink},
  legend style={at={(0.5,1.03)},anchor=south,legend columns=5}]
"""
    eb = ",error bars/.cd,y dir=both,y explicit"
    body += f"\\addplot[area legend,fill=cVanOurs,draw=tsred!45] coordinates {{{coords(series['van'])}}};\n"
    body += f"\\addplot[area legend,fill=cVanPaper,draw=cVanPaper] coordinates {{{coords(series['pv'])}}};\n"
    body += f"\\addplot[area legend,fill=tsnavy,draw=tsnavy] coordinates {{{coords(series['pt'])}}};\n"
    body += f"\\addplot[area legend,fill=tsred,draw=tsred{eb}] coordinates {{{coords(series['ts'], err=True)}}};\n"
    body += f"\\addplot[area legend,fill=cRepair,draw=cRepair{eb}] coordinates {{{coords(series['rep'], err=True)}}};\n"
    body += ("\\legend{Our vanilla,Paper vanilla,Paper TS,Our TS,Our Repair-TS}\n\\end{axis}\n")
    write("all_tasks", body)
    return rows


# ---------------------------------------------------------------- 3. NL-to-SQL over rounds, base and Instruct

def fig_rounds():
    def line(arm, prof, seeds):
        pts = []
        for r in ROUNDS:
            m, e = over_seeds(lambda s: f"train_{arm}_{prof}_seed{s}_round{r}", bt.m8, seeds)
            if m is not None:
                pts.append((r, m, e))
        return coords(pts, err=True)
    body = r"""\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.6cm}, rd, width=7.6cm, height=6.4cm,
  xmin=0.75, xmax=3.25, xtick={1,2,3}, y tick label style={/pgf/number format/.cd,fixed,fixed zerofill,precision=2}, xticklabels={Round 1,Round 2,Round 3},
  error bars/error bar style={line width=0.6pt}, error bars/error mark options={rotate=90,mark size=2pt},
  every axis plot/.append style={line width=1.4pt, mark size=2.6pt}]
"""
    for prof, title, nv_seeds, lo, hi in (("base", "Llama-3.1-8B (base)", SEEDS, 0.44, 0.62),
                                          ("instruct", "Llama-3.1-8B-Instruct", (1,), 0.54, 0.65)):
        van = bt.m8(bt.cells_of("VANILLA_BASE" if prof == "base" else "VANILLA_INSTRUCT"))
        extra = (",legend to name=roundslegend, legend columns=4,"
                 "legend style={draw=none,font=\\small,/tikz/every even column/.append style={column sep=12pt}}"
                 if prof == "base" else "")
        body += (f"\\nextgroupplot[title={{{title}}}, ymin={lo}, ymax={hi}, "
                 f"ylabel={{{'Mean of 8 (4 sets $\\times$ 2 prompts)' if prof == 'base' else ''}}}{extra}]\n")
        body += f"\\addplot[tsmuted,dashed,mark=none,line width=1.1pt,domain=0.75:3.25] {{{van:.4f}}};\n"
        nv = "no validation" + (", 1 seed" if prof == "instruct" else "")
        body += ("\\addplot[tsgrey,mark=square*,mark options={fill=white},error bars/.cd,y dir=both,y explicit] "
                 f"coordinates {{{line('novalidation', prof, nv_seeds)}}};\n")
        body += ("\\addplot[tsred,mark=*,error bars/.cd,y dir=both,y explicit] coordinates "
                 f"{{{line('tablespecialist', prof, SEEDS)}}};\n")
        body += ("\\addplot[cRepair,mark=diamond*,mark size=3.4pt,error bars/.cd,y dir=both,y explicit] coordinates "
                 f"{{{line('repairts', prof, SEEDS)}}};\n")
        if prof == "base":
            body += "\\legend{Vanilla,No validation,Table-Specialist,Repair-TS (ours)}\n"
        else:
            body += "\\node[font=\\scriptsize,text=tsgrey,anchor=south west] at (rel axis cs:0.02,0.02) {No validation: 1 seed};\n"
    body += "\\end{groupplot}\n\\node[anchor=south] at ($(group c1r1.north)!0.5!(group c2r1.north)+(0,0.75cm)$) {\\pgfplotslegendfromname{roundslegend}};\n"
    write("rounds", body)


# ---------------------------------------------------------------- 4. repair minus TS at round 3, with 95% CI

def fig_repair_gain():
    sig = {}
    for f in ("significance.json", "significance_scala.json", "significance_r.json"):
        for r in json.loads((ROOT / "results" / "analysis" / f).read_text()):
            sig[r["comparison"]] = r["mean_of_8"]
    ed = {r["comparison"]: r["mean_of_2"] for r in
          json.loads((ROOT / "results" / "analysis" / "significance_ed.json").read_text())}
    rows = [("Base NL-to-SQL", sig["Base: repair-TS r3 vs TS r3"]),
            ("Instruct NL-to-Scala", sig["Scala: repair-TS r3 vs TS r3"]),
            ("Instruct NL-to-R", sig["R: repair-TS r3 vs TS r3"]),
            ("Instruct NL-to-SQL", sig["Instruct: repair-TS r3 vs TS r3"])]
    edrow = ed["ED: repair-ED r3 vs TS r3"]
    labels = [r[0] for r in rows][::-1]

    def pts(items):
        out = []
        for i, (lab, d) in enumerate(items):
            col = "cRepair" if d["ci95"][0] > 0 else "tsmuted"
            diff = f"{d['diff']:+.3f}".replace("-", "−")
            pval = "< 0.0001" if d["p"] == 0 else f"= {d['p']:.2f}"
            out.append(f"\\addplot[{col},line width=2.2pt,mark=none] coordinates {{({d['ci95'][0]},{i}) "
                       f"({d['ci95'][1]},{i})}};\n"
                       f"\\addplot[only marks,mark=*,mark size=3.6pt,{col},mark options={{draw=white,line width=0.8pt}}] "
                       f"coordinates {{({d['diff']},{i})}};\n"
                       f"\\node[font=\\scriptsize,text=tsink,anchor=west] at (axis cs:{d['ci95'][1]},{i}) "
                       f"{{\\,{diff}, p {pval}}};\n")
        return "".join(out)
    body = r"""\begin{axis}[rd, name=main, ymajorgrids=false, xmajorgrids, width=11cm, height=6cm,
  xmin=-0.012, xmax=0.068, ymin=-0.6, ymax=3.6, ytick={0,1,2,3}, yticklabels={""" + ",".join(labels) + r"""},
  xlabel={Repair-TS minus TS, round 3 (mean of 8; dot = mean, bar = 95\% CI)},
  xtick={-0.01,0,0.01,...,0.06}, scaled x ticks=false, x tick label style={/pgf/number format/.cd,fixed,precision=2},
  title={Execution accuracy}]
\addplot[tsgrey,line width=0.8pt,mark=none] coordinates {(0,-0.6) (0,3.6)};
""" + pts(rows[::-1]) + r"""\end{axis}
\begin{axis}[rd, at={(main.south east)}, anchor=south west, xshift=1.4cm, ymajorgrids=false, xmajorgrids,
  width=6.2cm, height=3.2cm, xmin=-0.04, xmax=0.13, ymin=-0.6, ymax=0.6, ytick={0}, yticklabels={Error detection},
  xlabel={Repair-ED minus TS, F1}, xtick={-0.04,0,0.04,0.08}, scaled x ticks=false,
  x tick label style={/pgf/number format/.cd,fixed,precision=2}, title={F1 (different scale)}]
\addplot[tsgrey,line width=0.8pt,mark=none] coordinates {(0,-0.6) (0,0.6)};
""" + pts([("Error detection", edrow)]) + "\\end{axis}\n"
    write("repair_gain", body)


# ---------------------------------------------------------------- 5. what the validator keeps

def fig_validation():
    cats = [("kept", "agree"), ("disagree", "disagree"), ("scala_failed", "scala_failed"),
            ("always_empty", "always_empty")]
    rows = []
    for prof, plab in (("instruct", "Instruct"), ("base", "Base")):
        for r in (3, 2, 1):
            tot = {k: 0 for k, _ in cats}
            tot["other"] = 0
            n = 0
            for s in SEEDS:
                p = bt.RUNS / f"train_tablespecialist_{prof}_seed{s}" / f"round{r}" / "summary.json"
                d = json.loads(p.read_text())
                for k, v in d["reasons"].items():
                    key = {"agree": "kept", "disagree": "disagree", "scala_failed": "scala_failed",
                           "always_empty": "always_empty"}.get(k, "other")
                    tot[key] += v
                    n += v
            rows.append((f"{{{plab}, round {r}}}", {k: 100 * v / n for k, v in tot.items()}, n))
    ylabels = [r[0] for r in rows]
    body = r"""\begin{axis}[rd, xbar stacked, bar width=13pt, width=14cm, height=7.4cm, ymajorgrids=false, xmajorgrids,
  xmin=0, xmax=100, ytick={0,...,5}, yticklabels={""" + ",".join(ylabels) + r"""}, ymin=-0.6, ymax=5.6,
  xlabel={Share of generated SQL/Scala pairs (\%), seeds 1--3 pooled}, xtick={0,20,...,100},
  legend style={at={(0.5,1.03)},anchor=south,legend columns=5}, every axis plot/.append style={draw=white,line width=0.8pt}]
"""
    colours = {"kept": "tsred", "disagree": "tsnavy", "scala_failed": "cVanPaper", "always_empty": "cControl",
               "other": "tsline"}
    for key in ("kept", "disagree", "scala_failed", "always_empty", "other"):
        pts = " ".join(f"({row[1][key]:.2f},{i})" for i, row in enumerate(rows))
        body += f"\\addplot[area legend,fill={colours[key]}] coordinates {{{pts}}};\n"
    for i, row in enumerate(rows):
        body += (f"\\node[font=\\scriptsize\\bfseries,text=white,anchor=west] at (axis cs:0.6,{i}) "
                 f"{{{row[1]['kept']:.0f}\\%}};\n")
    body += ("\\legend{Kept (SQL and Scala agree),Results disagree,Scala failed,Always empty,"
             "{Other (SQL failed, no code)}}\n\\end{axis}\n")
    write("validation", body)
    return rows


# ---------------------------------------------------------------- 6. round-1 alternatives on base

def fig_round1():
    arms = [("tablespecialist", "Table-Specialist", "tsred"), ("dpo", "DPO on rejected pairs", "tsmuted"),
            ("bestof4", "Best-of-4 sampling", "cVanPaper"), ("tsdata", "Same-data control", "cControl"),
            ("repairts", "Repair-TS (ours)", "cRepair")]
    van = bt.m8(bt.cells_of("VANILLA_BASE"))
    body = r"""\begin{axis}[rd, ymajorgrids=false, xmajorgrids, width=12cm, height=6.4cm,
  xmin=0.465, xmax=0.575, ymin=-0.6, ymax=4.75, ytick={0,...,4}, yticklabels={""" + \
        ",".join(a[1] for a in arms[::-1]) + r"""},
  xlabel={Base Llama, round 1: mean of 8 (dot = mean of 3 seeds, bar = $\pm$1 SD, rings = single seeds)},
  scaled x ticks=false, x tick label style={/pgf/number format/.cd,fixed,fixed zerofill,precision=2},
  xtick={0.48,0.50,...,0.56}]
"""
    body += (f"\\draw[tsgrey,dashed,line width=1pt] (axis cs:{van:.4f},-0.6) -- (axis cs:{van:.4f},4.45);\n"
             f"\\node[font=\\scriptsize,text=tsgrey,anchor=south] at (axis cs:{van:.4f},4.42) {{vanilla {van:.3f}}};\n")
    for i, (arm, _, col) in enumerate(arms[::-1]):
        pre = lambda s: f"train_{arm}_base_seed{s}_round1"
        m, e = over_seeds(pre, bt.m8)
        seeds = " ".join(f"({bt.m8(bt.cells_of(pre(s))):.4f},{i})" for s in SEEDS)
        body += (f"\\draw[{col},line width=2.6pt] (axis cs:{m - e:.4f},{i}) -- (axis cs:{m + e:.4f},{i});\n"
                 f"\\addplot[only marks,mark=o,mark size=2.4pt,{col},line width=0.8pt] coordinates {{{seeds}}};\n"
                 f"\\addplot[only marks,mark=*,mark size=4.2pt,{col},mark options={{draw=white,line width=1pt}}] "
                 f"coordinates {{({m:.4f},{i})}};\n"
                 f"\\node[font=\\scriptsize\\bfseries,text=tsink,anchor=south] at (axis cs:{m:.4f},{i}) "
                 f"[yshift=5pt] {{{m:.3f}}};\n")
    body += "\\end{axis}\n"
    write("round1_alternatives", body)


if __name__ == "__main__":
    fig_sql_vs_paper()
    rows = fig_all_tasks()
    fig_rounds()
    fig_repair_gain()
    v = fig_validation()
    fig_round1()
    for task, (van, paper, ts, rep) in rows.items():
        print(f"{task:16s} vanilla {van:.4f}  TS {ts[0]:.4f}±{ts[1]:.4f}  repair {rep[0]:.4f}±{rep[1]:.4f}")
    for lab, d, n in v:
        print(f"{lab:20s} n={n:5d} kept {d['kept']:.1f}%")
    if bt.MISSING:
        print("MISSING:", bt.MISSING)
