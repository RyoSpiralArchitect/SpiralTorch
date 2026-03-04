# ONNX / Interop Status

SpiralTorch currently ships **export scaffolding** that emits **JSON artefacts** describing
compressed weights and deployment metadata. Full ONNX ingest/export is **not implemented
yet** (tracked in `docs/backend_matrix.md`).

## What exists today

- Python export pipeline: `bindings/st-py/spiral/export.py` (`ExportPipeline.export(...)`)
- CLI: `spiral-export` (emits JSON artefacts; ONNX/TFLite binaries are planned)
- Compression helpers exposed to Python: `spiraltorch.export` (quantization + structured pruning)

These are intentionally backend-agnostic and can be used to produce reproducible build
artefacts while the ONNX operator surface is being defined.

## ONNX MVP (proposed)

CPU-first target that can round-trip a minimal, well-scoped module set:

- Tensor basics: constants, reshape, transpose, concat/slice
- Linear stack: `MatMul`, `Add`, `Gemm`, `Relu`, `Softmax`
- Vision starter: `Conv`, `MaxPool`, `BatchNorm` (optional for MVP)

## Next steps

1. Define an explicit supported-operator list and document it here.
2. Extend `spiral-export` to emit the current JSON artefacts plus an ONNX skeleton graph.
3. Add parity tests against `onnxruntime` for the MVP operator set (CPU).
4. Expand backend coverage (WGPU/CUDA/HIP) after CPU parity stabilizes.
