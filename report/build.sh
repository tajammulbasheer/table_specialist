#!/usr/bin/env bash
# Build the report PDF. Tables are regenerated from the project's result files first, then LaTeX runs inside a
# pinned TeX Live 2025 image with no network. The project root is mounted read-only except for report/.
#
# Note: with the LaTeX 2025-11 kernel, every page break inside a longtable logs "ignored error: Infinite glue
# shrinkage found in box being split". TeX ignores it and the output is unaffected (it also happens in a minimal
# document with only longtable). latexmk counts it as an error, so this script checks the log itself: it fails on any
# real error ("! ..." line) and reports the number of ignored longtable notes.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
IMAGE="texlive/texlive:TL2025-historic@sha256:dbd65d778eb83de9b45fb0a8deb3afe44cbbaba2272e4e1e644e94dda52c7109"
python3 tools/build_tables.py || exit 1
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT:/project:ro" -v "$ROOT/report:/project/report" -w /project/report "$IMAGE" \
  latexmk -pdf -f -interaction=nonstopmode -file-line-error main.tex > build.log 2>&1
real=$(grep -c '^! \|^\./.*:[0-9]*: [^i]' main.log)
ignored=$(grep -c 'ignored error: Infinite glue shrinkage' main.log)
undefined=$(grep -c 'undefined on input line' main.log)
echo "errors: $real; ignored longtable notes: $ignored; undefined references or citations: $undefined"
if [ "$real" -gt 0 ] || [ ! -s main.pdf ]; then
  grep -A3 '^! \|^\./.*:[0-9]*: [^i]' main.log | head -40
  exit 1
fi
echo "built $(pwd)/main.pdf"
