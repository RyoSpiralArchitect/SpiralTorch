#!/usr/bin/env bash
set -euo pipefail
VER="${1:-v1.7.0}"
BR="${2:-feat/v1.7.0}"
git checkout -b "$BR" || true
git add -A
git commit -m "$VER: WGPU 1CE TopK, MidK compaction, Hypergrad, Self-Rewrite, Unison chooser" || true
git tag -f "$VER"
git push origin "$BR"
git push -f origin "$VER"
echo "Draft Release body (paste into GitHub):"
cat <<'EOF'
## SpiralTorch v1.7.0
- WGPU: Single‑CE TopK (Subgroups). Candidate→Final in one pass.
- MidK: where_nd + scan compaction kernels (WGSL/HIP), 1CE/2CE switch.
- Ameba Autograd: hypergrad (unrolled/implicit) utilities.
- Self‑Rewrite: SpiralK soft(...) auto‑append under CI/win‑rate guards.
- Unison chooser: unified heuristics across backends; WGPU absorbs HIP/CUDA.
EOF
