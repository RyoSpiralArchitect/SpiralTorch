# Restartable Vision Input

`st-vision` owns the input restart contract. Rust's `DataLoaderCheckpoint`
stores the sample order, next position, shuffle mode/RNG, transform
configuration/RNG, batch size and a caller-supplied dataset SHA256. A standalone
`TransformPipelineCheckpoint` stores just transform configuration/RNG.
Enable `input-checkpoint` for a CPU-only Rust client; `nn` includes it.
Python exposes the same contract even without the `nn` or `wgpu` features.

## Ownership And Boundaries

- Restore into an explicitly supplied dataset and the same batch/transform
  configuration. The dataset identifier must cover the caller's immutable
  samples, labels and ordering. Length alone is not identity. Rust compares
  the supplied SHA256 but does not scan the dataset or authenticate a false ID.
- Validation, RNG reconstruction and allocations precede mutation. Wrong
  identity, dimensions/configuration, duplicate/out-of-range indices, invalid
  cursor and unknown schemas leave the destination unchanged.
- GPU devices, dispatchers and caches are local resources, not serialized
  handles. Restore preserves the destination's explicitly configured dispatcher.
- Resident submission consumes the batch and augmentation draws on successful
  submission, even if the GPU later rejects the update. Restore preserves this
  existing behavior; it does not introduce automatic retry or cursor rollback.
- Save model and input states at the same **settled** boundary, after observing
  update acceptance/rejection. In-flight GPU submissions are not captured.
  These are separate payloads, not an atomic model/input/trainer bundle.
  Schedule state, epoch counters and optimizer policy remain caller-owned.

Input randomness now explicitly uses ChaCha12 instead of relying on the future
choice of `StdRng`. Tests preserve the current rand 0.8 seeded draws and 64-bit
native shuffle order. Shuffle range sampling is pinned to `u64` on every
target: historical wasm32 seed-only shuffle sequences can change. There was
no prior portable input checkpoint schema to migrate.
RNG stream/68-bit word position use decimal strings, and f32 transform settings
use bit patterns, avoiding JavaScript number precision loss. JSON payloads are
limited to 32 MiB, sample orders to four million entries, and transforms to
1,024 operations. Unsupported versions fail rather than guessing a migration.

## Python And Browser APIs

```python
# `loader` and `model` have reached the same settled step boundary.
input_json = loader.checkpoint_json(dataset_sha256)
model_json = model.checkpoint_snapshot().read_json()

# Construct `resumed_loader` with the same dataset/batch/transforms first.
resumed_loader.restore_checkpoint_json(dataset_sha256, input_json)
resumed_model = st.ResidentConvNeXtClassifier.from_checkpoint_json(device, model_json)

# Standalone transforms also support replay, without a DataLoader.
transform_json = pipeline.checkpoint_json()
same_config_pipeline.restore_checkpoint_json(transform_json)
```

The browser `VisionTransformPipeline` has `checkpointJson()` and
`restoreCheckpointJson(payload)`. Construct it with `createCpu` or `createGpu`
and add matching transforms before restoring. No JS DataLoader is introduced:
browser batch ordering and its persistence remain caller-owned. There is no
Python/JS implementation of the RNG or restore decision.

## Verified Slice

The [bounded verification record](../benchmarks/results/2026-10-01-vision-input-checkpoint/README.md)
contains the public cross-runtime fixture and source/log hashes.

Native tests compare 100 consecutive classifier update attempts with a restart
after attempt 37. Attempts 37 and 72 deliberately contain invalid CE targets.
The checkpoint therefore resumes directly after a rejected update. All later
batch identities and transformed image bits match, and the final model JSON
(every weight plus attempted-update revision) and input state are identical.
This uses a tiny synthetic two-stage ConvNeXt and fixed-rate plain SGD, not
real-dataset quality, variable schedules or general optimizer-state restart.

CPU tests cover shuffle, ColorJitter, partial tail batches, multiple epoch
resets, invalid restore atomicity and RNG block-boundary/large-counter cases.
Fresh-wheel Python tests exercise public CPU/GPU input continuation. A native
Python transform checkpoint also replays 20 exact transforms and final RNG
state in real wasm32 Rust under Node. That is **WASM CPU portability**, not a
real-browser WebGPU learning-resume result. The WASM binding builds and both
generated/shipped TypeScript declarations are checked.

```bash
cargo test -p st-vision --no-default-features --features input-checkpoint --lib
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test -p st-vision \
  --features wgpu --lib input_checkpoint::tests -- --test-threads=1

SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
SPIRALTORCH_INPUT_CHECKPOINT_HANDOFF="$NEW_HANDOFF_JSON" \
python -I bindings/st-py/tests/test_vision_input_checkpoint.py

cargo build -p spiraltorch-wasm --release --features webgpu \
  --target wasm32-unknown-unknown
wasm-bindgen --target nodejs --out-dir "$NEW_WASM_DIR" --out-name spiraltorch_wasm \
  target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm
node bindings/st-wasm/tests/vision_input_checkpoint.cjs \
  "$NEW_WASM_DIR/spiraltorch_wasm.js" "$NEW_HANDOFF_JSON"
node bindings/st-wasm/tests/resident_types.cjs "$NEW_WASM_DIR/spiraltorch_wasm.js"
```

Use a freshly built wheel for the Python checks and new artifact paths. The
handoff contains synthetic image values only. Next is an integrity-bound
model/input/trainer restart boundary, followed by shared resident optimizer
control; the full roadmap restart gate remains open.
