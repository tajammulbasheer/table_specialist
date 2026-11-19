#!/usr/bin/env bash
# Back up the whole project to the Windows E: drive (separate physical disk from the WSL disk).
#   1. exact archive of everything (tar, uncompressed; keeps symlinks, permissions, ':' file names) + SHA-256 + file list
#   2. browsable folder copy for Windows (everything except .venv, which only works inside WSL)
#   3. verification: archive compared byte for byte with the project; folder copy compared by rsync dry run
# Nothing in the project is changed. Run in its own tmux session; the log is appended with tee by the caller.
set -euo pipefail
SRC_PARENT=/home/ts
NAME=table-specialist-repro
STAMP=2026-11-18
DEST=/mnt/e/table-specialist-repro-backup
ARCHIVE="$DEST/${NAME}_${STAMP}.tar"
FOLDER="$DEST/$NAME"
mkdir -p "$DEST"
[ -e "$ARCHIVE" ] && { echo "refusing to overwrite $ARCHIVE"; exit 1; }
echo "$(date -u '+%F %T UTC') start; source size: $(du -sh --apparent-size "$SRC_PARENT/$NAME" | cut -f1)"

echo "$(date -u '+%F %T UTC') [1/5] archive -> $ARCHIVE"
tar -cf "$ARCHIVE" -C "$SRC_PARENT" "$NAME"
ls -l "$ARCHIVE"

echo "$(date -u '+%F %T UTC') [2/5] checksum and file list"
(cd "$DEST" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
tar -tvf "$ARCHIVE" > "$DEST/${NAME}_${STAMP}_filelist.txt"
echo "files and dirs in archive: $(wc -l < "$DEST/${NAME}_${STAMP}_filelist.txt")"

echo "$(date -u '+%F %T UTC') [3/5] verify archive against the project (tar --compare)"
tar -df "$ARCHIVE" -C "$SRC_PARENT" && echo "archive matches the project byte for byte"

echo "$(date -u '+%F %T UTC') [4/5] browsable folder copy -> $FOLDER (without .venv)"
mkdir -p "$FOLDER"
rsync -a --exclude '/.venv/' "$SRC_PARENT/$NAME/" "$FOLDER/"

echo "$(date -u '+%F %T UTC') [5/5] verify folder copy (rsync dry run; any listed file differs)"
DIFF=$(rsync -a -n -i --exclude '/.venv/' "$SRC_PARENT/$NAME/" "$FOLDER/" | grep -v '^\.d' || true)
if [ -z "$DIFF" ]; then echo "folder copy matches the project (size and time)"; else echo "DIFFERENCES:"; printf "%s\n" "$DIFF" | head -50 || true; echo "($(printf "%s\n" "$DIFF" | wc -l) differing entries in total)"; fi
du -sh --apparent-size "$FOLDER"
echo "$(date -u '+%F %T UTC') done"
