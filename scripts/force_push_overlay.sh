#!/usr/bin/env bash
set -euo pipefail

TAG="overlay-v1.22.0"
BRANCH_CUR=$(git rev-parse --abbrev-ref HEAD)

echo "[i] Add/Update overlay files..."
git add -A
git commit -m "SpiralTorch overlay v1.22.0: SoftMode/Beam/Learn + Fractional Tensor" || true
git tag -f "${TAG}"

echo "[i] Pushing branch '${BRANCH_CUR}' and tag '${TAG}' (force) ..."
git push -f origin "${BRANCH_CUR}"
git push -f origin "${TAG}"

echo "[✓] Done."
