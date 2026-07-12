#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if command -v rg >/dev/null 2>&1; then
  violations="$(
    rg -n '\bset_tensor_op_meta_observer\b' crates/st-core/src \
      --glob '*.rs' \
      --glob '!**/plugin/mod.rs' || true
  )"
else
  violations="$(
    find crates/st-core/src -type f -name '*.rs' ! -path '*/plugin/mod.rs' \
      -exec grep -nH 'set_tensor_op_meta_observer' {} + || true
  )"
fi

if [[ -n "$violations" ]]; then
  echo "st-core metadata tests must use the thread-local observer:" >&2
  echo "$violations" >&2
  exit 1
fi

echo "st-core metadata observer isolation ok"
