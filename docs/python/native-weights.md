# Native Safetensors Weight Ingress

Source-only extension after the fixed 0.4.32 release candidate. It is not part
of that candidate's qualified wheels or a claim of PyPI publication.

`SafeTensorArchive` loads a local safetensors file through Rust, without the
Python safetensors package, Torch, NumPy, Transformers, pickle or remote code.
`NdTensor` preserves its shape and explicitly bridges to existing ST tensors.
This is weight ingress, **not automatic HF model loading or full-model FT**.

## Use a Local Checkpoint

```python
import spiraltorch as st

weights = st.SafeTensorArchive("model.safetensors")
print(weights.names())
print(weights.info("projection.weight"))  # dtype, shape, elements, byte_len

weight = weights.tensor("projection.weight")  # F32/F16/BF16 -> finite host f32
print(weight.shape)
# This example expects [input_features, output_features] in the checkpoint.
# Torch Linear usually stores the reverse order: explicitly use
# weight.permute([1, 0]) when the model's contract requires it.
prediction = inputs.matmul(weight.to_tensor(), backend="cpu")
```

Single-file and immutable-byte entry points are supported. For an already
obtained payload use `SafeTensorArchive.from_bytes(payload)`; writable buffers
are intentionally not accepted. No `from_pretrained`, shard-index resolution,
architecture inference, tied-parameter inference, dtype guessing or automatic
transpose occurs. Caller-provided model contracts still own those decisions.

The shared Rust entry points are
`st_nn::weights::{SafeTensorArchive, WeightLoadLimits}`. Parsing uses the
[official safetensors Rust library](https://docs.rs/safetensors/0.8.0/safetensors/).
The byte constructor is also compilable into WASM; this change does not expose
a new browser JavaScript binding or establish browser execution.

## Shape and Ownership

`tensor(name)` returns an immutable host snapshot backed by the existing
`st_tensor::NdTensor`, not a foreign-framework tensor. Scalars, empty dimensions
and higher ranks are retained. `to_tensor()` requires rank two; use an explicit
`reshape([rows, cols])` if the model contract permits it. `permute(axes)` is a
view, and incompatible noncontiguous reshapes fail rather than changing meaning.
`NdTensor.from_tensor(tensor)` snapshots an existing native 2-D tensor.

`values()` returns a list; `values_buffer()` returns a fresh bytearray of
native-endian f32 values, suitable for
`memoryview(result).cast("f")`. Neither can modify the snapshot. The archive
owns its original bytes and remains valid after the original file changes or
closes. File snapshots are not authenticated: pin a trusted content digest
separately when identifying a model. A concurrent same-size writer is not a
coherent checkpoint transaction; use completed immutable checkpoints.

This initial route reads an owned bounded file and materializes selected
weights. It is not mmap, zero-copy, GPU-resident, lazy shard loading, or a
throughput improvement claim. f32 conversion of F16/BF16 preserves their
numeric values but changes storage size and dtype.

## Admission Limits

Both constructors accept keyword overrides:

| Limit | Default | Applied |
| --- | --- | --- |
| `max_file_bytes` | 1 GiB | Before payload allocation/copy |
| `max_header_bytes` | 8 MiB | Before header allocation/parsing |
| `max_tensors` | 100,000 | During header admission |
| `max_rank` | 32 | During header admission; configurable up to 128 |
| `max_tensor_elements` | 67,108,864 | Before each selected f32 materialization |

Limits are per archive or materialization, not a process RSS guarantee. The
input bytes, decoded tensor and returned Python copies can coexist. Large
models require deliberate memory budgets; do not raise limits blindly.

Duplicate tensor names (including escaped aliases), duplicate fields/metadata
keys, unknown tensor fields, non-string metadata, malformed offsets,
overlaps/holes/trailing bytes, shape/byte overflow and truncated/growing file
reads fail closed. Valid unsupported dtypes remain inspectable in `info()` but
are not silently cast. NaN/Inf are rejected at **materialization**, not merely
at archive inspection. This stricter admission intentionally rejects unknown
header fields and null metadata even if other format readers tolerate them.

## Connect to Geometry Learning

```sh
python -I -B bindings/st-py/examples/native_weights_geometry.py
python -I -B bindings/st-py/tests/test_native_weights.py
```

The example feeds a synthetic checkpoint through native matrix multiplication
and a Rust-owned WaveGate learning loop, then checks exact adapter continuation
and unchanged frozen projection weights. It demonstrates composability, not a
pretrained LM, whole-model native backpropagation or language-quality gain.
The test entry forbids importing optional ML frameworks and the Python
safetensors package. Rust tests additionally use the official format writer.

Continue with [native geometry learning](native-geometry-runtime.md). Optional
HF/Torch integration remains available separately and is not silently replaced.

## Local Qualification

On 2026-10-18, the reviewed macOS arm64 development wheel passed 14 Rust weight
tests (including the bounded FIFO regression), 19 native geometry Rust
regressions, 8 no-framework Python weight tests, 14 native geometry Python
regressions, 109 optional writer/Torch/geometry/artifact interoperability tests,
32 release/compatibility contracts and three core smoke programs. The isolated
environment contained none of Torch, NumPy, Transformers, Accelerate or Python
safetensors. The optional comparison used a separate existing Torch environment.
The WASM core compiled; no browser execution or non-macOS runtime is claimed.

The source and wheel-packaged synthetic example produced identical output:
40 updates lowered activation MSE from 0.02330000326037407 to
0.007973438128829002, frozen base weights remained unchanged, and one restored
continuation step matched exactly. These are mechanics checks, not a language
model experiment or a geometric advantage claim.

Independent source review identified the Unix FIFO open-before-type-check
hang. Nonblocking descriptor-based admission and the bounded child-process
regression resolve it; source re-review found no remaining actionable finding.
Runtime checks were performed by the implementing agent, not the reviewer.

The local test wheel SHA-256 is
`79655307e5fa21b04748a85365a8e6f37688dd9d6970ee5766d8fa04f3d78a03`.
It uses development metadata 0.4.32 but is **not** the fixed candidate or an
official release wheel, and was not uploaded to PyPI. The first release-smoke
attempt accidentally used the stale global package and failed; its log is
retained, and the isolated reviewed-wheel rerun passed without weakening tests.
