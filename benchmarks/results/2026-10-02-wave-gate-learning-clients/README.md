# WaveGate Learning Clients: Gradients, Updates And Controls

Date: 2026-10-02. Implementation: `03e0e820cf4520c5b9f529b1c77c5a1bf6ddc3c0`.
Final integrated source: `ecf0e8ac345d50e358cf41caa9cc06c69f5109ac`.
This is bounded learning-path evidence, not a speed or language-quality claim.

## What Is Connected

Rust owns an immutable forward snapshot of input, gate, bias and geometric
recipe. Torch consumes its policy-free VJP, including sum-reduced shared
parameter derivatives. WASM consumes the same snapshot and pullback.
The new residual adapter starts at identity and can learn both gate and bias
on its first update. It supports arbitrary leading axes, empty batches,
noncontiguous tensors and explicit CPU transport of Torch accelerator tensors.
It does not add resident GPU execution, AMP, trainable curvature or a text encoder.

## Correctness And Continuation

- Final native `st-nn` library: **755 passed**, no failures or ignored tests.
- Python geometric integration: **29 passed, zero skipped**, including both CPU
  and MPS transport against independent Torch f64 forward/VJP references.
- Public imports/type surface: **22 passed**.
- Native build, WASM `nn` build, pinned formatting and targeted Ruff pass.
  This does not supersede the parent's documented existing strict Clippy failures.
- Torch tests exercise identity initialization, nonzero gradients, causal row
  independence, recipe replacement after forward, in-place mutation rejection,
  empty/invalid inputs, Adam learning and exact adapter/optimizer next-update
  continuation after `weights_only=True` checkpoint loading.
- A random tiny HF causal LM test verifies that both parameters receive finite
  nonzero gradients, loss decreases, logits change and base weights remain fixed.
  The separate cached-pretrained run below is not that random-model test.
- Actual browser WASM forward and all three VJPs match the native fixture with
  **maximum absolute difference 0**. This is one fixture, not universal equality.
  Snapshot independence, invalid values/shapes, empty rows and reuse after freeing
  the kernel are also checked in the browser.
- Browser plain-SGD gate/bias learning on a two-row teacher-generated target
  reduces MSE from **0.06280464 to 0.0001286676** after 200 updates.
  Restoring the parameter JSON reproduces the next update exactly.
  That browser check uses stateless SGD, not an Adam checkpoint claim.

The first Python check had three failures: public export registration was
missing and the independent test oracle omitted the porous map's 0.25 factor
(CPU and MPS cases). Both were corrected; the failed log hash is retained.
The corrected checks were rerun after integrating main. The final matched-model
losses, gradient norms and logit changes are identical to the pre-integration
record. The final browser loss history also agrees exactly.

## Cached Pretrained GPT-2 Control

Cached snapshot: `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
No downloads. Frozen float32 CPU GPT-2; adapter explicitly after
`transformer.h.0.mlp`, width 768; 1536 trainable parameters in each arm.
Two authored training sentences and two development sentences, padding masked
with label -100; full batch; Adam lr 0.001; strength 0.1; eight updates.
All arms start with exactly the baseline logits. All base-weight hashes remain
unchanged. The off arm performs no backward or effective updates.

The tangent control is the first-order map at zero gate/bias:
`x + strength * (x * gate + bias) / sqrt(-curvature)`.
It has the same parameters, loss, data and update count as WaveGate, but not the
same compute/transfer cost. This is **not** a PyTorch speed comparison.

| Arm | Train loss, initial -> final | Development loss, initial -> final | Max logit change |
| --- | --- | --- | --- |
| Off | 6.256789 -> 6.256789 | 5.348199 -> 5.348199 | 0 |
| Tangent | 6.256789 -> 6.251640 | 5.348199 -> 5.346810 | 0.109573 |
| WaveGate | 6.256789 -> 6.251854 | 5.348199 -> 5.346854 | 0.103836 |

Both active arms learn. **The tangent control is slightly better on this tiny
corpus.** This is one deterministic trajectory, seed label 17; zero initialization
and full-batch updates make additional seed labels alone non-independent.
No held-out quality, multi-seed advantage or long-FT stability is established.
All loss/gradient traces, authored texts, token IDs and checkpoint hashes are in
`pretrained.json`. Three adapter/optimizer checkpoints and the native extension
are retained locally, not committed.

## Reproduction

```bash
cargo test --locked -p st-nn --lib
maturin develop --manifest-path bindings/st-py/Cargo.toml
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
python -m pytest bindings/st-py/tests/test_wave_gate_learning.py bindings/st-py/tests/test_wave_gate_vjp.py bindings/st-py/tests/test_geometry_autograd.py bindings/st-py/tests/test_elliptic_learning.py -v -rs
python tests/test_runtime_imports.py -v
python bindings/st-py/examples/hf_elliptic_learning.py --geometry wave_gate --seeds 17 --model-dir "$LOCAL_GPT2" --block transformer.h.0.mlp --features 768 --steps 8 --checkpoint-dir "$NEW_CHECKPOINT_DIR" > pretrained.json
cargo build --locked -p spiraltorch-wasm --features nn --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir "$SERVE_DIR/module" target/wasm32-unknown-unknown/debug/spiraltorch_wasm.wasm
```

Serve the generated module (including its snippets) at `/module/`, the model
report at `/pretrained.json`, and
`bindings/st-wasm/tests/wave_gate_learning.html` at `/` using loopback HTTP.
Its visible result must have status `passed`; compilation alone is not browser
evidence. The test page frees its snapshots and gradient handles.

Rust 1.98.0; formatter nightly-2026-04-15; Python 3.12.6; Torch 2.12.1;
Transformers 4.57.6; wasm-bindgen 0.2.104; macOS aarch64; debug builds.
`validation.json` records source, binary and retained-log hashes.
Public files contain no model weights, raw local logs or local paths.
`SHA256SUMS` binds this complete public record.
