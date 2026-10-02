# WaveGate On A Novel: Saturation Is Not The Active Limiter

Date: 2026-10-02. The study code and four-arm protocol were committed before
training at `10764de454315e03fd69ba9e9c620b64d4688c7d`.
All **12 conditions completed**; no arm, seed or checkpoint was selected away.
This is a paired, bounded learning/conditioning pilot, not a speed benchmark
or evidence of general language-quality advantage.

## Fixed Design

Use the existing local English `Pride and Prejudice` text, raw SHA256
`df06eafcd4c1793dc0bf12211401f4cfb0477ef6d68fe278642fd434a6f56f9c`,
709597 bytes. Remove the trailing Gutenberg notice at the unique explicit
end marker, then split at the last paragraph boundary before 90% of the
remaining characters. No dataset or model was downloaded.

- Cached GPT-2 snapshot `607a30d783dfa663caf39e06633721c8d4cfcd7e`; frozen float32
  CPU base; explicit adapter after `transformer.h.0.mlp`; width 768.
- Token blocks are disjoint 128-token chunks within each text partition:
  1204 training blocks and 136 development blocks. Tail fragments are dropped.
- Each active arm sees 32 batches of two blocks: **8128 causal prediction
  targets**, not an entire-book training pass. Three seeded shuffles
  (41, 43, 47) are exactly paired across arms. They change sampled batches,
  unlike the preceding zero-initialized full-batch probe's seed labels.
- Development evaluation uses 16 evenly spaced blocks from the separate final
  text partition. An eight-block training-pool probe is held constant for
  descriptive before/after comparisons. Neither is a newly untouched test set;
  the book may also have appeared in GPT-2 pretraining.
- Adam lr 0.001; residual strength 0.1; 1536 gate/bias parameters; identity
  initialization; no dropout; CPU, two Torch threads.
- Four arms: off, matched tangent-linear, WaveGate (saturation 1), and WaveGate
  with saturation 10000. Curvature -1 and porosity 0.05 are fixed.
- Endpoints are step 32. Step 8/16/24 evaluations are descriptive, not a
  checkpoint-selection rule. Off performs no updates.

`plan.json` is the pre-execution plan, intentionally still marked running with
an empty run list. **`results.json` is the completed result.** It contains all
minibatch indices, text/token hashes, losses, gradient norms, local conditioning
and checkpoint hashes. No source book text or model weights are committed.

## Results

Baseline development loss is **3.9822350144**. All active arms improve that
small, fixed probe; the tangent control is slightly better for all three seeds.

| Seed | Tangent Development CE | WaveGate Development CE | WaveGate Minus Tangent |
| --- | --- | --- | --- |
| 41 | 3.9778005481 | 3.9785041213 | +0.0007035732 |
| 43 | 3.9778837264 | 3.9786026776 | +0.0007189512 |
| 47 | 3.9780568480 | 3.9786638021 | +0.0006069541 |

The mean paired difference is **+0.00067649285**, not a statistical-significance
claim. The tangent also has lower final training-probe CE for all seeds.
See `summary.json`; it derives from the hashed complete result.

The two WaveGate arms have **identical numerical training records and development
loss histories** for every seed. All logged training forwards have zero gate
and affine saturation counts. Simply relaxing the saturation threshold therefore
does not address this observed gap.

On each seed's last training forward (before update 32), Rust reports:

| Seed | Mean Dimensionless Norm | Mean Relative Radial Gain | Mean Relative Tangential Gain |
| --- | --- | --- | --- |
| 41 | 0.59542465 | 0.71495983 | 0.89633327 |
| 43 | 0.63011816 | 0.68844072 | 0.88562926 |
| 47 | 0.59996212 | 0.71151284 | 0.89496088 |

These are the local projection Jacobian gains relative to its zero-norm slope,
not ratios of complete model gradients. They locate a real nonlinear attenuation
while ruling out active elementwise saturation on the logged training inputs.
They do **not** establish that increasing a projection radius will improve CE,
nor that the current geometry is generally inferior.

## Actual Learning And Restart

All nine active conditions have finite, nonzero gate and bias gradients, finite
updates, unchanged frozen-base hashes, and unchanged base gradient buffers.
Every active run loads its saved adapter/Adam checkpoint from disk with
`weights_only=True`, takes the same next batch through uninterrupted and restored
states, and reproduces both parameter and optimizer states exactly.
The 288 primary updates are separate from **18 extra validation-only updates**.
Published endpoint losses precede these continuation checks.

The Rust conditioning API is available directly and through Python/WASM.
Torch's `forward_with_conditioning` returns scalars from its actual forward
snapshot without a second forward or global last-call state. Ordinary calls
do not compute diagnostics. Empty aggregates are null; disabled adapters report
no geometry execution. Tests show that enabling observation changes neither
outputs nor VJPs.

Validation:
- Entire native `st-nn` library: **756 passed**, zero ignored.
- Python geometry/data-split tests: **32 passed**, zero skipped; CPU/MPS transport
  against independent Torch references remains covered.
- Public import/type tests: **22 passed**.
- Native extension and WASM `nn` builds pass; pinned format and targeted Ruff pass.
- Actual browser: native forward, VJP and conditioning fixture differences are
  all **zero**. Browser learning and stateless-SGD continuation still pass.

## Review Fix And Boundaries

During the frozen run, PR #2179 review found that the optional `nn` feature did
not gate the new native binding module. The minimal build failure was reproduced,
then fixed in parent commit `db20806c1399123f656e40ce22e4fbbd64b487aa`.
The module, registration and native exports are now conditional. Both the
no-default-features and extension-module,text checks pass.
The study used its already-loaded, NN-enabled binary from the frozen source;
this feature-only repair did not alter running mathematics or restart training.
The local logs retain the before/after build evidence.

CPU copies, list transport and on-demand diagnostic passes have overhead, which
is not timed here. No AMP, higher-order gradient, resident GPU geometry,
long-horizon stability or generalization claim is added. The parent's recorded
strict Clippy findings are not claimed fixed by targeted Ruff/format checks.

## Reproduction

```bash
maturin develop --manifest-path bindings/st-py/Cargo.toml
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
python bindings/st-py/examples/hf_wave_gate_conditioning.py --config bindings/st-py/examples/hf_wave_gate_pride_conditioning.json --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" --output-dir "$NEW_RESULT_DIRECTORY"
cargo test --locked -p st-nn --lib
cargo check --locked -p spiraltorch-py --no-default-features
cargo check --locked -p spiraltorch-py --no-default-features --features extension-module,text
python -m pytest bindings/st-py/tests/test_wave_gate_learning.py bindings/st-py/tests/test_wave_gate_vjp.py bindings/st-py/tests/test_geometry_autograd.py bindings/st-py/tests/test_elliptic_learning.py bindings/st-py/tests/test_wave_gate_conditioning_example.py -v -rs
```

For the browser, build `spiraltorch-wasm --features nn` for wasm32, run matching
wasm-bindgen with target web, serve the generated module/snippets at `/module/`,
`results.json` at `/pretrained.json` and
`bindings/st-wasm/tests/wave_gate_learning.html` at `/` using loopback HTTP.
Require the page's visible status to be passed.

Recorded environment: Rust 1.98.0; formatter nightly-2026-04-15; Python 3.12.6;
Torch 2.12.1; Transformers 4.57.6; wasm-bindgen 0.2.104; macOS aarch64;
debug native/WASM builds. `validation.json` binds binary/source/log hashes.
Raw logs and all 12 small adapter/optimizer checkpoints remain local.
