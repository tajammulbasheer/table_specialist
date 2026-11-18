#!/usr/bin/env bash
# Download raw benchmarks and the training-table corpus. Safe to re-run: finished files are skipped.
# Only fetches data. Nothing downloaded here is executed.
set -u
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"/{wikisql,bird,gittables}

step() { echo "[$(date +%H:%M:%S)] $*"; }

clone_pinned() { # url dest
  if [ -d "$2/.git" ]; then step "have $2"; else git clone -q --depth 1 "$1" "$2" && step "cloned $2"; fi
  git -C "$2" rev-parse HEAD > "$2.commit" 2>/dev/null
}

step "WikiSQL"
wget -q -c -O "$RAW/wikisql/data.tar.bz2" https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2 \
  && step "WikiSQL ok" || step "WikiSQL FAILED"

step "WikiTableQuestions"
clone_pinned https://github.com/ppasupat/WikiTableQuestions "$RAW/wikitq"

step "AutoTest (Error Detection, stretch goal)"
clone_pinned https://github.com/qixuchen/AutoTest "$RAW/autotest"

step "BIRD dev"
wget -q -c -O "$RAW/bird/dev.zip" https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip \
  && step "BIRD ok" || step "BIRD FAILED"

step "GitTables 1M licensed subset (96 topic zips)"
curl -sL https://zenodo.org/api/records/6517052 > "$RAW/gittables/record.json"
jq -r '.files[] | [.key, .links.self, .checksum] | @tsv' "$RAW/gittables/record.json" > "$RAW/gittables/files.tsv"
while IFS=$'\t' read -r key url sum; do
  printf '%s\0%s\0' "$RAW/gittables/$key" "$url"
done < "$RAW/gittables/files.tsv" \
  | xargs -0 -n 2 -P 3 sh -c 'wget -q -c --tries=8 --waitretry=30 -O "$0" "$1" || echo "FAILED $0"'
step "GitTables files present: $(find "$RAW/gittables" -name '*.zip' -size +0 | wc -l) of $(wc -l < "$RAW/gittables/files.tsv")"

step "done"
