#!/usr/bin/env bash
# Wait until no process matches a pattern, then exit. Run in the background so that its exit wakes the session.
# Usage: scripts/watch_until_done.sh <label> <process pattern> [seconds between checks, default 30]
# Any process whose command line names this script is ignored, so the watcher and the shell that started it,
# which both contain the pattern, cannot keep it waiting forever.
set -u
LABEL=$1; PATTERN=$2; INTERVAL=${3:-30}
START=$(date +%T)
while pgrep -af -- "$PATTERN" | grep -v "watch_until_done" | grep -q .; do sleep "$INTERVAL"; done
echo "$LABEL finished; watched from $START to $(date +%T)"
