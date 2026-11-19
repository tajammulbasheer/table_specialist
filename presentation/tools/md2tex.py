"""One-off converter: docs/PRESENTATION_TRANSCRIPT.md -> presentation/transcript.tex (LuaLaTeX).
Handles exactly the Markdown subset used in that file."""
import re, sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().split("\n")

def esc(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s.replace(r"\textbackslash\{\}", r"\textbackslash{}")

def quotes(s):
    s = re.sub(r'(^|[\s(\[{/–—-])"', lambda m: m.group(1) + "\u201c", s)
    s = s.replace('"', "\u201d")
    s = re.sub(r"(^|[\s(\[])'", lambda m: m.group(1) + "\u2018", s)
    return s.replace("'", "\u2019")

def inline(s):
    codes = []
    def keep(m):
        codes.append(m.group(1)); return f"\x00{len(codes)-1}\x00"
    s = re.sub(r"`([^`]+)`", keep, s)
    s = quotes(esc(s))
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: r"\texttt{" + esc(codes[int(m.group(1))]) + "}", s)
    return s

out = []
i = 0
mode = None            # 'must' | 'dont' while collecting a boxed block
box = []

def close_box():
    global mode, box
    if mode:
        env = "mustsay" if mode == "must" else "dontsay"
        out.append(r"\begin{%s}" % env)
        out.extend(box)
        out.append(r"\end{%s}" % env)
    mode, box = None, []

def emit(x):
    (box if mode else out).append(x)

def flush_list(items, kind):
    if not items:
        return
    env = {"ul": "itemize", "ol": "enumerate", "chk": "itemize"}[kind]
    opt = "[label=$\\square$]" if kind == "chk" else ""
    emit(r"\begin{%s}%s" % (env, opt))
    for it in items:
        emit(r"\item " + it)
    emit(r"\end{%s}" % env)

def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], [r for r in cells[2:]]
    n = len(head)
    spec = "@{}" + "X" * n + "@{}"
    emit(r"\begin{center}\small\begin{tabularx}{\linewidth}{%s}\toprule" % spec)
    emit(" & ".join(r"\textbf{%s}" % inline(h) for h in head) + r"\\\midrule")
    for r in body:
        emit(" & ".join(inline(c) for c in r) + r"\\")
    emit(r"\bottomrule\end{tabularx}\end{center}")

LABEL = re.compile(r"^\*\*(Time|Script|Must say|Point at|Don't say|Next|If short of time|Hand-over \([^)]*\)):\*\*\s*(.*)$")

while i < len(lines):
    ln = lines[i]
    s = ln.strip()
    # headings
    if ln.startswith("# "):
        close_box()
        out.append(r"\tstitle{%s}" % inline(ln[2:])); i += 1; continue
    if ln.startswith("## "):
        close_box()
        h = ln[3:].strip()
        m = re.match(r"Slide (\d+): (.*)", h)
        if m:
            out.append(r"\slidehead{%s}{%s}" % (m.group(1), inline(m.group(2))))
        elif h.startswith("Title slide") or h.startswith("Thank-you slide"):
            out.append(r"\slidehead{}{%s}" % inline(re.sub(r"\s*\(no number\)", "", h)))
        elif h.startswith("Section"):
            out.append(r"\partsection{%s}" % inline(h))
        else:
            out.append(r"\section*{%s}" % inline(h))
        i += 1; continue
    if ln.startswith("### "):
        close_box()
        out.append(r"\subsection*{%s}" % inline(ln[4:])); i += 1; continue
    if s == "---":
        close_box(); i += 1; continue
    if s.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append(lines[i]); i += 1
        table(rows); continue
    m = LABEL.match(s)
    if m:
        close_box()
        lab, rest = m.group(1), m.group(2)
        if lab == "Time":
            out.append(r"\timeline{%s}" % inline(rest))
        elif lab == "Script":
            out.append(r"\scriptlabel")
        elif lab == "Must say":
            mode = "must"
            if rest: box.append(inline(rest))
        elif lab == "Don't say":
            mode = "dont"
            if rest: box.append(inline(rest))
        else:
            out.append(r"\labelled{%s}{%s}" % (inline(lab), inline(rest)))
        i += 1; continue
    # Q&A: numbered bold question followed by indented answer lines
    m = re.match(r"^(\d+[a-z]?)\. \*\*(.+?)\*\*\s*$", s)
    if m and i + 1 < len(lines) and lines[i + 1].startswith("   ") and lines[i + 1].strip():
        close_box()
        ans = []
        i += 1
        while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
            ans.append(lines[i].strip()); i += 1
        out.append(r"\qa{%s}{%s}{%s}" % (m.group(1), inline(m.group(2)), inline(" ".join(ans))))
        continue
    # bold-only line = group heading in Q&A
    m = re.match(r"^\*\*([^*]+)\*\*$", s)
    if m:
        close_box()
        out.append(r"\qagroup{%s}" % inline(m.group(1))); i += 1; continue
    # lists
    if re.match(r"^- \[ \] ", s):
        items = []
        while i < len(lines) and re.match(r"^- \[ \] ", lines[i].strip()):
            items.append(inline(lines[i].strip()[6:])); i += 1
        flush_list(items, "chk"); continue
    if re.match(r"^- ", s) and not ln.startswith("   "):
        items = []
        while i < len(lines) and re.match(r"^- ", lines[i].strip()):
            items.append(inline(lines[i].strip()[2:])); i += 1
        flush_list(items, "ul"); continue
    if re.match(r"^\d+\. ", s):
        items = []
        while i < len(lines) and (re.match(r"^\d+\. ", lines[i].strip()) or
                                  (lines[i].startswith("   ") and lines[i].strip())):
            t = lines[i].strip()
            if re.match(r"^\d+\. ", t):
                items.append(inline(re.sub(r"^\d+\. ", "", t)))
            else:
                items[-1] += " " + inline(t)
            i += 1
        flush_list(items, "ol"); continue
    if not s:
        emit(""); i += 1; continue
    # paragraph: join consecutive text lines
    para = [s]; i += 1
    while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |\d+\. |\*\*[^*]+:\*\*|---)", lines[i].strip()):
        para.append(lines[i].strip()); i += 1
    txt = inline(" ".join(para))
    if txt.startswith("Divider:"):
        emit(r"\divnote{%s}" % txt)
    else:
        emit(txt)
    emit("")
close_box()

pre = r"""% Speaker transcript for the end-semester presentation. Build: ./build_transcript.sh (LuaLaTeX, TeX Live 2025).
% GENERATED from docs/PRESENTATION_TRANSCRIPT.md by tools/md2tex.py. Do not edit; edit the Markdown and rerun ./build_transcript.sh.
\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2.1cm,top=2.2cm,bottom=2.4cm]{geometry}
\usepackage{fontspec}
\usepackage[sfdefault,lining]{FiraSans}
\usepackage[scale=0.86]{FiraMono}
\usepackage{xcolor,booktabs,tabularx,enumitem,amssymb,fancyhdr,titlesec}
\usepackage[most]{tcolorbox}
\usepackage[hidelinks]{hyperref}
\usepackage{needspace}
\definecolor{tsnavy}{HTML}{1A2E4F}\definecolor{tsred}{HTML}{A6252D}\definecolor{tsgrey}{HTML}{5F6B7C}
\definecolor{tslight}{HTML}{ECEFF4}\definecolor{tspink}{HTML}{F6E6E7}\definecolor{tspanel}{HTML}{F4F6F9}
\definecolor{tsink}{HTML}{1F2430}
\color{tsink}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.55em}
\setlist{itemsep=0.15em,topsep=0.3em,leftmargin=1.4em}
\setlist[itemize]{label={\color{tsred}\rule[0.15em]{0.4em}{0.4em}}}
\renewcommand{\arraystretch}{1.25}
\titleformat{\section}{\Large\bfseries\color{tsnavy}}{}{0pt}{}[{\color{tsred}\rule{1.4cm}{1.6pt}}]
\titleformat{\subsection}{\large\bfseries\color{tsnavy}}{}{0pt}{}
\titlespacing*{\section}{0pt}{1.4em}{0.6em}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\color{tsgrey}Table-Specialist \textperiodcentered{} Speaker transcript}
\fancyhead[R]{\small\color{tsgrey}End-semester presentation}
\fancyfoot[C]{\small\color{tsgrey}\thepage}
\renewcommand{\headrulewidth}{0pt}
\newcommand{\tstitle}[1]{{\color{tsnavy}\fontsize{21}{25}\selectfont\bfseries #1\par}\vspace{0.2em}{\color{tsred}\rule{2.4cm}{2.5pt}}\par\vspace{0.6em}}
\newcommand{\partsection}[1]{\clearpage\begin{tcolorbox}[enhanced,colback=tsnavy,colframe=tsnavy,arc=3pt,left=10pt,top=8pt,bottom=8pt]
  {\color{white}\Large\bfseries #1}\end{tcolorbox}\vspace{0.4em}}
\newcommand{\slidehead}[2]{\par\vspace{1.0em}\needspace{8\baselineskip}%
  \begin{tcolorbox}[enhanced,colback=tslight,colframe=tslight,arc=3pt,left=6pt,right=8pt,top=5pt,bottom=5pt,
    borderline west={3pt}{0pt}{tsred}]
  \if\relax\detokenize{#1}\relax\else{\color{tsred}\large\bfseries Slide #1}\quad\fi{\color{tsnavy}\large\bfseries #2}\end{tcolorbox}\vspace{-0.3em}}
\newcommand{\timeline}[1]{{\small\color{tsgrey}\textbf{Time}\enspace #1}\par}
\newcommand{\scriptlabel}{{\small\color{tsnavy}\bfseries SCRIPT}\par\vspace{-0.3em}}
\newcommand{\labelled}[2]{\textbf{\color{tsnavy}#1:}\enspace #2\par}
\newcommand{\divnote}[1]{{\small\color{tsgrey}\itshape #1}\par}
\newtcolorbox{mustsay}{enhanced,colback=tspink,colframe=tspink,arc=3pt,left=8pt,right=8pt,top=4pt,bottom=4pt,
  borderline west={3pt}{0pt}{tsred},before upper={{\small\color{tsred}\bfseries MUST SAY}\par\vspace{0.1em}}}
\newtcolorbox{dontsay}{enhanced,colback=tspanel,colframe=tspanel,arc=3pt,left=8pt,right=8pt,top=4pt,bottom=4pt,
  borderline west={3pt}{0pt}{tsgrey},before upper={{\small\color{tsgrey}\bfseries DON'T SAY}\par\vspace{0.1em}}}
\newcommand{\qagroup}[1]{\par\vspace{0.6em}{\large\bfseries\color{tsnavy}#1}\par}
\newcommand{\qa}[3]{\par\begin{tcolorbox}[enhanced,breakable,colback=white,colframe=tslight,boxrule=0.6pt,arc=3pt,left=8pt,right=8pt,top=4pt,bottom=5pt]
  {\color{tsred}\bfseries Q#1.}\enspace\textbf{#2}\par\vspace{0.25em}#3\end{tcolorbox}}

\begin{document}
"""
open(dst, "w", encoding="utf-8").write(pre + "\n".join(out) + "\n\\end{document}\n")
print("wrote", dst, len(out), "lines")
