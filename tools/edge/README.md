# Edge deployment toolkit

This directory contains a lightweight deployment harness for exporting
compressed SpiralTorch models to on-device runtimes such as TensorFlow Lite and
ONNX Runtime.  The scripts rely on the `spiraltorch.spiral.export` module to run
quantisation-aware training (QAT) calibration, structured pruning, and
benchmarking prior to serialising the model artefacts.

## Contents

- `deploy_tflite.py` – end-to-end pipeline targeting TensorFlow Lite clients.
- `deploy_onnx.py` – analogous pipeline for ONNX Runtime targets.
- `runtime/` – thin adapters that emulate device-side execution for local
  smoke testing.  They consume the JSON artefacts produced by the deployment
  scripts and compute synthetic latency/accuracy estimates.
- `benchmarks/` – machine-readable benchmark snapshots recorded during export.

All scripts are pure Python and avoid heavy framework dependencies.  They are
intended to run inside CI as well as on developer laptops.
