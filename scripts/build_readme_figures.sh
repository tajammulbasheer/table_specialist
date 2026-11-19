#!/usr/bin/env bash
# Build the README charts: assets/figures_src/*.tex -> assets/*.png (LuaLaTeX + Ghostscript, TeX Live 2025 image, no network).
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/make_readme_figures.py
docker run --rm --network none -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/assets:/w" -w /w/figures_src \
  texlive/texlive:TL2025-historic sh -c '
    set -e
    for f in *.tex; do
      n=${f%.tex}
      lualatex -interaction=nonstopmode -halt-on-error "$f" > "$n.build.log" 2>&1 || { tail -30 "$n.build.log"; exit 1; }
      gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r220 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
         -sOutputFile="../$n.png" "$n.pdf"
      rm -f "$n.aux" "$n.log"
    done'
ls -la assets/*.png
