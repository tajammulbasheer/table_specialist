#!/usr/bin/env bash
# Build the end-semester slides (endsem.pdf) in the same pinned TeX Live 2025 image as the report, with no network.
set -uo pipefail
cd "$(dirname "$0")"
IMAGE="texlive/texlive:TL2025-historic@sha256:dbd65d778eb83de9b45fb0a8deb3afe44cbbaba2272e4e1e644e94dda52c7109"
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$(pwd):/deck" -w /deck "$IMAGE" \
  latexmk -pdf -f -interaction=nonstopmode -file-line-error endsem.tex > build.log 2>&1
errors=$(grep -c '^! \|^\./.*:[0-9]*: ' endsem.log)
echo "errors: $errors; overfull boxes: $(grep -c '^Overfull' endsem.log)"
if [ "$errors" -gt 0 ] || [ ! -s endsem.pdf ]; then grep -A4 '^! \|^\./.*:[0-9]*: ' endsem.log | head -60; exit 1; fi
echo "built $(pwd)/endsem.pdf ($(grep -o 'Output written.*' endsem.log))"
