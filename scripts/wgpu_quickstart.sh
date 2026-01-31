#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TEXT_PATH="${1:-"$ROOT/models/samples/spiral_demo_en.txt"}"
RUN_DIR="${2:-"$ROOT/models/runs/wgpu_quickstart_$(date +%Y%m%d_%H%M%S)"}"

EPOCHS="${EPOCHS:-2}"
BATCHES_PER_EPOCH="${BATCHES_PER_EPOCH:-8}"
BATCH="${BATCH:-8}"
DESIRE_PRIME="${DESIRE_PRIME:-16}"

echo "[SpiralTorch] building Python extension with WGPU runtime..."
(cd "$ROOT" && cargo build -p spiraltorch-py --features wgpu-rt)

echo "[SpiralTorch] running coherence-wave learning stack demo..."
PYTHONNOUSERSITE=1 python3 -S -s "$ROOT/models/python/llm_char_coherence_wave.py" "$TEXT_PATH" \
  --backend wgpu \
  --epochs "$EPOCHS" \
  --batches "$BATCHES_PER_EPOCH" \
  --batch "$BATCH" \
  --desire \
  --desire-prime "$DESIRE_PRIME" \
  --events "$RUN_DIR/events.jsonl" \
  --atlas \
  --run-dir "$RUN_DIR"

echo "[SpiralTorch] done:"
echo "  run_dir: $RUN_DIR"
echo "  events:  $RUN_DIR/events.jsonl"
echo "  atlas:   $RUN_DIR/atlas_summary.json"
echo "  weights: $RUN_DIR/weights.json"
