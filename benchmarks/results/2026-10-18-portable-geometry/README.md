# Portable Geometry State: Local Qualification

Source candidate on `312557b339f6f1db3281e63979a8cf46dc1062a8`, with the
implementation/test hashes in [validation.json](validation.json). Not a PyPI
release, model-quality study, performance benchmark or full trainer checkpoint.

## Observed

- Five artifact Rust tests pass: all built-in recipes, exact-byte/base-identity loading,
  invalid routing/shape/domain rejection, and a pinned cross-client fixture.
- All 23 st-frac learning tests pass, including the new native recipe preflight.
- Installed-wheel Python: 58 tests pass (27 artifact tests plus 31 existing
  geometry-stack regressions). All 13 supported configurations preserve state,
  trainability, output and VJP; reconstruction leaves CPU RNG unchanged.
- Tiny randomly initialized GPT-2 and Llama, four geometric MLP insertions each:
  two AdamW updates, detached export/reconstruction, exact inference, then exact
  third-update logits/loss/gradients/weights/AdamW state with an independently
  saved optimizer companion. All 12 geometric parameter tensors have finite
  nonzero gradients at the checked update; base weights remain unchanged.
- A fresh isolated Python process reconstructs the fractional-angle artifact.
- Actual `wasm32` NN release, executed in Node, validates all 13 fixture entries,
  rejects the tested corruptions and produces the same pinned artifact ID as
  native Rust/Python. No browser HF execution is claimed.
- Minimal WASM build still checks without `nn`; standard st-nn clippy succeeds
  with 24 existing warnings. Strict warning-free lint is not claimed.

The fixture is explicitly synthetic, not learned/private model output. Real
corpora, weights, optimizer states and build binaries are not published here.
Local raw logs/build artifacts are retained in the validation directory.

## Corrections During Qualification

The first shared-fixture reconstruction failed because named tensors retained
different list ordering. Rust now sorts named tensors, preserving placement
order and exact original bytes separately. A subsequent failure exposed an L2
history gain stored as f64 even though its native/Python recipe is f32; the typed
contract now uses f32. Neither failure was papered over by weakening equality:
the final check pins the native/WASM ID and requires reconstruction to reproduce
that exact canonical artifact.

Independent read-only review then found two additional issues: tensor
parameter/buffer role changes were not represented, and fractional recipes
could have input-independent scale/coefficient overflow. Explicit role fields
now preserve/reject the former; a native st-frac preflight using the existing
coefficient builders handles the latter. Kernel-tap budgets bound preflight
allocation. Regression tests retain valid normalized and empty-history cases.
Adding role fields intentionally changed the synthetic fixture's pinned ID;
the final shared expected file binds the updated schema across all clients.

CI subsequently caught module ordering in the WASM registration file. The
follow-up changes only rustfmt ordering; the qualification record retains the
original binary-source hash as well as the corrected source hash. The recorded
binaries precede that formatting-only edit.

## Reproduce

Build/install the source wheel in an isolated environment with Torch and
Transformers 4.57.6, then run from the repository root:

```sh
cargo +1.99.0 test -p st-nn --no-default-features --lib geometry_artifact -- --test-threads=1
cargo +1.99.0 test -p st-frac --lib learning
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python -I -B -m pytest --import-mode=importlib --confcutdir=bindings/st-py/tests \
  --rootdir=bindings/st-py/tests -q bindings/st-py/tests/test_geometry_artifact.py \
  bindings/st-py/tests/test_geometry_adapters.py
cargo +1.99.0 build --locked -p spiraltorch-wasm --target wasm32-unknown-unknown --release --no-default-features --features nn
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm --target nodejs --out-dir /tmp/geometry-wasm
node bindings/st-wasm/tests/geometry_artifact.cjs /tmp/geometry-wasm/spiraltorch_wasm.js
```

Use the wasm-bindgen CLI matching Cargo.lock (0.2.129 for this run). No model or
dataset download is required. See the [API/boundaries](../../../docs/python/geometry-artifacts.md).
The geometry artifact omits base weights, optimizer/cursor/RNG and runtime code
identity; the surrounding HF experiment must retain and verify those companions.
