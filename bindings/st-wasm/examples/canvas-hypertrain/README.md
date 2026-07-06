# SpiralTorch Canvas hypertrain (WASM)

This example demonstrates running a tiny “learning loop” in the browser using
`spiraltorch-wasm`'s `FractalCanvas` bindings:

- `FractalCanvas.framePacket()` returns pixels + relation + trail + Desire metrics in one call.
- CPU loop: update relation via `hypergradWaveCurrent` / `realgradWaveCurrent`.
- Optional supervised mode: capture a target relation and train by minimising mean-squared error.
- WebGPU: optional 3D trail renderer + FFT row probe + “hypergrad operator” compute step.

## Quickstart

From the repo root:

```bash
bash scripts/wasm_demo.sh canvas-hypertrain
```

## Prerequisites

Build the WASM package (and sync it into each example’s `pkg/` directory):

```bash
./scripts/build_wasm_web.sh --dev
```

Install frontend deps:

```bash
cd bindings/st-wasm/examples/canvas-hypertrain
npm ci
```

## Running

```bash
npm run dev
```

## Target MSE quick demo

1. Click “Capture target” to store the current relation as the supervised target.
2. Click “Seed relation” (optional) to create a new starting point.
3. Set “loss = target mse” and click “Step” (or enable “run continuously”).

## Run artifact export

Use “Copy report JSON” or “Download report” to capture the live browser experiment as
`spiraltorch.wasm.canvas_hypertrain_report.v1`. The report includes:

- Current `FractalCanvas.framePacket()` summaries for relation, vector field, trail, gradients,
  Desire controls, and WebGPU readiness.
- The bounded training metrics tail (loss, hyper/real RMS, LR scale, balance/stability/saturation).
- Optional target-relation stats when supervised target-MSE mode is active.

This makes the WASM demo usable as an auditable frontend probe for downstream Python notebooks,
API-LLM runtime comparison, and future fine-tuning replay flows.
