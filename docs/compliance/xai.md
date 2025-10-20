# Explainability (XAI) Compliance

SpiralTorch surfaces vision-focused attribution pipelines that satisfy internal
compliance controls around reproducibility, determinism, and reporting. The
components introduced in `st-vision` and `st-core` enable Grad-CAM and
Integrated Gradients analyses to be executed in isolation or batched through the
CLI.

## Algorithms

* **Grad-CAM** (`st_vision::xai::GradCam`) accepts channel-major activation and
gradient tensors sourced from registered forward hooks. Channel weights are
computed using global-average pooling of the gradients before projecting back
into the requested spatial dimensions. Heatmaps are rectified (configurable),
and callers can opt-in to raw (unnormalised) outputs or min-max scaling with an
epsilon guard to preserve determinism.
* **Integrated Gradients** (`st_vision::xai::IntegratedGradients`) interpolates
between a deterministic baseline and the analysed sample. A provided
`st_nn::module::Module` instance is evaluated along the integration path and the
target index is seeded for the backward pass on each step. Accumulated
gradients are averaged and reprojected onto the original input space.

Both algorithms emit `AttributionOutput` structures that capture the tensor map
and rich metadata for downstream telemetry.

## Model Hooks

`st_vision::models::ForwardAttributionHooks` registers layer-specific Grad-CAM
configurations and records activation/gradient pairs. When both tensors are
available the hook produces an `AttributionOutput`, ensuring intermediate state
is drained to avoid stale data. Hooks can also flush every registered layer via
`compute_all_grad_cam`, keeping attribution metadata synchronised per layer.
Integrated Gradients support is exposed through `run_integrated_gradients`,
combining attribution metadata with optional human-readable labels.

## Telemetry Reports

`st_core::telemetry::xai_report` serialises explainability artefacts. Reports
contain:

* Algorithm metadata (`algorithm`, `layer`, `target`, `steps`).
* A flexible `extras` map for algorithm-specific fields (spatial geometry,
  epsilon guards, model descriptors, etc.).
* The flattened heatmap plus explicit `(rows, cols)` shape for reliable
  reconstruction.

`AttributionReport` instances round-trip via `serde_json`, allowing downstream
services to archive or transport explainability packets without bespoke schema
code.

## CLI Batch Execution

The `st-xai-cli` binary under `tools/xai-cli` runs Grad-CAM or Integrated
Gradients over JSON fixtures:

```bash
cargo run -p st-xai-cli -- \
  --algorithm grad-cam \
  --activations activations.json \
  --gradients gradients.json \
  --height 14 --width 14 \
  --layer stem.conv1 \
  --output heatmap.json
```

Grad-CAM outputs are min-max normalised by default; append `--raw-heatmap` to
preserve the raw weighted activations. Integrated Gradients accepts optional
linear weights for lightweight models and a `--target-label` annotation for
reporting:

```bash
cargo run -p st-xai-cli -- \
  --algorithm integrated-gradients \
  --input sample.json \
  --baseline baseline.json \
  --steps 32 --target 0 \
  --target-label car \
  --output attribution.json
```

Outputs conform to the telemetry schema described above, ensuring compliance
pipelines receive structured and reproducible artefacts.
