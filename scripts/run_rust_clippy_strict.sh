#!/usr/bin/env bash
set -euo pipefail

LIST_FILE="${1:-${CLIPPY_STRICT_LIST_FILE:-configs/clippy_strict_packages.txt}}"

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

if [[ ! -f "$LIST_FILE" ]]; then
  echo "Strict clippy package list not found: $LIST_FILE (skipping)"
  exit 0
fi

did_run=0
while IFS= read -r raw || [[ -n "$raw" ]]; do
  line="$(trim "${raw%%#*}")"
  if [[ -z "$line" ]]; then
    continue
  fi
  did_run=1
  echo "==> strict clippy: $line"
  cargo clippy -p "$line" --all-targets --no-deps -- -D warnings
done < "$LIST_FILE"

if [[ "$did_run" -eq 0 ]]; then
  echo "No packages configured in $LIST_FILE; skipping strict clippy."
fi

