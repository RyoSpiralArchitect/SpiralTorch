# WaveGate Bulk Transport And Matched Learning Replay

CPU float32 interfaces, two Torch threads, release native build, seed 239.
Each row uses three separate processes per binary, two warmups and 12 rotated
route-order rounds per process. Values below are medians of process medians.
All routes request the same forward and input/gate/bias gradients; radius mode
also requests the log-radius derivative (`R=4`). No gradient is omitted for speed.

| Shape | Map | Old public ms | New public ms | New list ms | Torch reference ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 2x16x64 | legacy | 0.3109 | 0.0877 | 0.3018 | 0.2120 |
| 2x16x64 | radius | 0.3176 | 0.0936 | 0.3085 | 0.2257 |
| 2x128x768 | legacy | 24.8690 | 2.9895 | 24.8713 | 2.2840 |
| 2x128x768 | radius | 23.6927 | 2.9645 | 24.8425 | 2.2880 |

The large public calls are about 8x faster than their previous list-based
counterparts in this bounded sample, **still about 1.3x slower than the matched
Torch reference**. The small cases favor the native route. This is not an LLM,
GPU, full-training, general-library speed or quality result. Timings include
autograd and host transport. Old/new binary runs were sequential, not interleaved;
the three routes within each process use rotated permutations. No timing threshold
is asserted in CI. The independent Torch reference uses f64 projection
intermediates and f32 interfaces; its accumulation/rounding may differ within
`rtol=5e-4, atol=3e-5`.

`benchmark-runs.json.gz` contains all per-round measurements and correctness
receipts, without dropping slower runs or conditions.

Output and every requested gradient hash are bit-identical across old/new
binaries and list/public routes in **all 24 process reports**. The Rust radius
kernel also reuses row scratch storage, but the list-only timings do not establish
a separate scratch-reuse speedup. The measured improvement is dominated by
avoiding Python scalar boxing. Host copies and CPU computation remain explicit.

## Learning And WASM

- A saved step-2 GPT-2 checkpoint with WaveGate, Topos, elliptic and fractional
  adapters was continued for **one auxiliary update**, without retraining the
  completed study or scoring a held-out set. Loss, all 12 gradient tensors,
  12,294 adapter parameters, Adam and RNG match the old saved step 3 bit-for-bit.
  The frozen base is unchanged. WaveGate bulk forward/backward calls were observed.
- Scalar WASM executed 16 shape/radius cases using the preserved October 2
  NN-enabled module and the current module. All output/gradient hashes match.
  Both completed the same 400-update scalar-radius SGD test and exact next-update
  continuation. This is Node-hosted WASM, not browser/WebGPU or WASM speed evidence.
- Python geometry/fractional/WaveGate regression suite: 844 passed. Benchmark and
  replay-checker tests: 99 passed. Rust WaveGate tests: 21 passed. Native release
  and WASM `nn` release builds completed. Public-receipt tests are separate and
  are not additional execution evidence.

Strict `st-nn` Clippy (`-D warnings`) stopped on 23 diagnostics, all in files
unchanged from the parent. Ordinary advisory Clippy completed; this record does
not call the strict run clean. Logs and both setup failures are preserved:
pytest initially collected a parent-directory package before any test body, and
the first HF replay was rejected before model loading because its generated
runtime manifest had an extra field. Neither guard was relaxed; collection was
scoped and a new correctly shaped manifest was generated.

## Reproduction And Provenance

The native/Python change is `ea0b954dd8f25e8d39846a42dc29737bb91cb685`, on top of
`3c6296d1e831fb4ab30553acaeb87a55e394ad37`. Runtime manifests identify all package
files; the new package differs only in the native binary, type stub, two bridges,
and the added shared transport module. Benchmark/client hashes bind the exact
executed scripts. Numeric records, validation and hashes are public. Original
native binaries, checkpoints, corpus and logs stay in local validation storage.
No cleanup was performed, and previous artifacts were not rewritten.

```sh
# Run each mode/shape three times with distinct output paths under OLD and NEW.
PYTHONPATH="$PACKAGE" python -P -B tools/benchmark_wave_gate_learning.py \
  --native-profile release --shape 2 128 768 --rounds 12 --output "$RESULT"
PYTHONPATH="$PACKAGE" python -P -B tools/benchmark_wave_gate_learning.py \
  --native-profile release --shape 2 128 768 --rounds 12 \
  --log-radius 1.38629436112 --output "$RADIUS_RESULT"

cargo build --locked --release -p spiraltorch-wasm \
  --target wasm32-unknown-unknown --features nn
wasm-bindgen --target web --out-dir "$NEW_WASM" \
  target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm
node tools/probe_wave_gate_wasm.mjs "$NEW_WASM" "$WASM_RESULT"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$NEW_PACKAGE" \
  python -P -B tools/replay_geometry_stack_update.py \
  --previous "$SAVED_STACK_RUN" --client-root "$FROZEN_STACK_CLIENT" \
  --model-dir "$LOCAL_MODEL" --corpus "$LOCAL_CORPUS" \
  --package-root "$NEW_PACKAGE" --runtime-manifest "$NEW_MANIFEST" \
  --output "$NEW_REPLAY"
```

Use the two-field `source_revision`/`files` runtime manifest, the original public
stack example's hash-bound local inputs, and fresh output paths. Keep the same
Torch/Transformers versions for the exact learning replay. The manifest hashes
are identities, not a claim that the private native binaries are publicly shipped.
