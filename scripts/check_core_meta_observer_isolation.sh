#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v rg >/dev/null 2>&1; then
  echo "ripgrep is required to check st-core metadata observer isolation" >&2
  exit 1
fi

violations="$(
  rg -n '\bset_tensor_op_meta_observer\b' crates/st-core/src \
    --glob '*.rs' \
    --glob '!**/plugin/mod.rs' || true
)"

if [[ -n "$violations" ]]; then
  echo "st-core metadata tests must use the thread-local observer:" >&2
  echo "$violations" >&2
  exit 1
fi

echo "st-core metadata observer isolation ok"
