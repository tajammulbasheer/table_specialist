# Table-Specialist reproduction: LaTeX report

The technical and research report of this project: what the original paper proposes, what we reproduced, every
approach we tried (including the failures), and what the results show.

## Build

```bash
report/build.sh          # from the project root, or ./build.sh from report/
```

The script:

1. regenerates every results table and plot coordinate from the project's own output files
   (`tools/build_tables.py` reads `results/` and `runs/` only; writes `generated/*.tex`), and
2. compiles `main.tex` with `latexmk` inside a pinned TeX Live 2025 Docker image, with no network:
   `texlive/texlive:TL2025-historic@sha256:dbd65d77...`. Nothing is installed on the host.

The output is `main.pdf`; the full log is `main.log`, and `build.log` holds the latexmk output.

The build reports three counts. "errors" must be 0. "Undefined references or citations" should be 0. "Ignored longtable
notes" are expected: with the LaTeX 2025-11 kernel, every page break inside a `longtable` logs "ignored error:
Infinite glue shrinkage found in box being split". TeX ignores it, and the same message appears in a minimal document
containing nothing but a longtable. The PDF is unaffected.

To compile with a local TeX installation instead, run `python3 tools/build_tables.py` and then
`latexmk -pdf main.tex` in this folder. It needs TeX Live 2024 or later with biber.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Document skeleton: front matter, chapters, bibliography, appendices |
| `style/tsreport.sty` | Fonts, colour palette, headings, headers and footers, provenance boxes, algorithms, references |
| `chapters/` | 00 title page and abstract; 01-15 chapters |
| `appendices/` | A configuration, B supplementary tables, C significance, D registry, E incidents, F code, G cross-check, H reproduction |
| `figures/` | TikZ diagrams (method loop, our pipeline, repair-TS flow, architecture, timeline) |
| `generated/` | Tables and plot data written by `tools/build_tables.py`; never edit by hand |
| `tools/build_tables.py` | The table generator, including the cross-check of headline values (Appendix G) |
| `references.bib` | Bibliography; arXiv metadata checked against the abstract pages on 2026-11-16 |

## Updating

When experiments finish (repair-ED, its controls, NL-to-R), rerun `build.sh`. The generated tables pick up the new
result files automatically. Text that describes running experiments is dated, and must be updated by hand.
