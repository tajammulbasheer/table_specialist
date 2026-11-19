#!/usr/bin/env bash
# Build the speaker transcript PDF from docs/PRESENTATION_TRANSCRIPT.md (the single source).
# Step 1 converts the Markdown to transcript.tex; step 2 compiles it with LuaLaTeX in the pinned TeX Live image.
set -uo pipefail
cd "$(dirname "$0")"
IMAGE="texlive/texlive:TL2025-historic@sha256:dbd65d778eb83de9b45fb0a8deb3afe44cbbaba2272e4e1e644e94dda52c7109"
../.venv/bin/python tools/md2tex.py ../docs/PRESENTATION_TRANSCRIPT.md transcript.tex || exit 1
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$(pwd):/deck" -w /deck "$IMAGE" \
  latexmk -lualatex -f -interaction=nonstopmode -file-line-error transcript.tex > build_transcript.log 2>&1
errors=$(grep -c '^! \|^\./.*:[0-9]*: ' transcript.log)
echo "errors: $errors; overfull boxes: $(grep -c '^Overfull' transcript.log)"
if [ "$errors" -gt 0 ] || [ ! -s transcript.pdf ]; then grep -A4 '^! \|^\./.*:[0-9]*: ' transcript.log | head -40; exit 1; fi
echo "built $(pwd)/transcript.pdf ($(grep -o 'Output written.*' transcript.log))"
