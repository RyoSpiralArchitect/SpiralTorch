#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

check_graph() {
  local label="$1"
  shift

  local graph
  graph="$(cargo tree --locked "$@" -e normal --prefix none)"
  if grep -Eq '^(wgpu|wgpu-core|wgpu-hal) v' <<<"$graph"; then
    echo "CPU feature isolation failed for $label:" >&2
    grep -E '^(wgpu|wgpu-core|wgpu-hal) v' <<<"$graph" >&2
    return 1
  fi
  echo "CPU feature isolation ok: $label"
}

check_graph "st-core" -p st-core --no-default-features --features cpu
check_graph "st-nn" -p st-nn --no-default-features
check_graph \
  "spiraltorch-py" \
  -p spiraltorch-py \
  --no-default-features \
  --features python-default,cpu
check_graph "spiraltorch-sys" -p spiraltorch-sys --no-default-features
