#!/usr/bin/env bash
set -euo pipefail
VER="${1:-v1.7.2}"
BR="${2:-feat/v1.7.2}"
git checkout -b "$BR" || true
git add -A
git commit -m "$VER: WGPU Subgroups 1CE generalized; MidK 2CE; Hypergrad CG; Self-Rewrite×Unison; CI wheels" || true
git tag -f "$VER"
git push origin "$BR"
git push -f origin "$VER"
echo "Draft Release body:"
cat <<'EOF'
## SpiralTorch v1.7.2
### WGPU
- Subgroups single‑CE TopK generalized: auto pool control (keep_m) -> 1CE viable at longer cols.
- MidK 2CE (scan+apply) orchestrated; 1CE for small rows.
### Hypergrad
- implicit(..., solve="cg") added. Neumann remains default; CG via SpiralK `sv:1` or env.
### Self‑Rewrite × Unison
- Optional Redis bucket median -> low‑weight soft(...) injection; local win‑rate promotes weights.
### CI
- Sample universal2/musllinux wheel matrix with optional HIP/CUDA/WGPU features.
EOF
