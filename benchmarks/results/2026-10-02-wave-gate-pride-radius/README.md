# A Trainable Radius Reaches The Language-Model Loss

Date: 2026-10-02. Implementation and six-arm protocol were frozen before training
at `7564a2aeba2a84f437816d18543dc59f52a9f784`.
All **18 conditions completed**, with no condition or seed omitted.
This is a learning-mechanism pilot, not a speed benchmark or a language-quality win.

## Mechanism

Rust owns `R = exp(log_radius)` and the map
`R * tanh(norm(z)/(sqrt(-curvature)*R)) * z/norm(z)` after the existing porous
gate/affine saturation. Its origin Jacobian is `I/sqrt(-curvature)` for every
radius, so the new parameter does not change initial gain. The sum-reduced VJP
includes input, gate, bias and log-radius derivatives. A small-argument series
prevents cancellation in the last derivative.

Native Python and WASM consume the same owned Rust snapshots. The Torch adapter
can keep radius fixed as a buffer or learn it as one extra scalar. Default
construction retains the legacy API and v1 checkpoint layout; explicit radius
adapters use v2 state and require matching fixed/learnable modes when loading.
The scalar's derivative is zero at identity initialization, then becomes nonzero
after gate/bias start moving. This is expected, not a disconnected parameter.

## Fixed Comparison

The cached GPT-2 model, corpus SHA256, footer removal, paragraph split, 128-token
packing, 16 development probe blocks and three shuffled schedules are the same
as the [preceding pilot](../2026-10-02-wave-gate-pride-conditioning/README.md).
No model or corpus download was performed.

- Frozen float32 CPU GPT-2, `transformer.h.0.mlp`, width 768, no dropout/cache.
- Seeds 41/43/47; 32 steps, batch 2, Adam lr 0.001, residual strength 0.1.
- Off; tangent; fixed radius 1/4; learnable radius initialized at 1/4.
- All geometric arms use the new radius path and identical saturation/porosity.
  Each fixed/learnable pair starts with exactly the same map and initial gain.
- Fixed and tangent arms have 1536 parameters; learned-radius arms have 1537.
  These latter arms are therefore **not parameter-count-matched** to tangent.
- Each active arm sees 8128 causal targets, not the entire book. Endpoint is
  fixed at step 32; intermediate evaluations do not select a checkpoint.

The wider f64 parameter-gradient accumulation in the new radius path can differ
slightly from the legacy path's f32 reduction. Comparisons below rerun all radius
arms through the same implementation, rather than treating historical numbers
as bit-identical controls.

## Results

Baseline development CE: **3.9822350144**. Lower is better.

| Seed | Tangent | Fixed R=1 | Learned From R=1 | Fixed R=4 | Learned From R=4 |
| --- | --- | --- | --- | --- | --- |
| 41 | 3.9778005481 | 3.9785044193 | 3.9784788191 | 3.9778843522 | 3.9778806865 |
| 43 | 3.9778837264 | 3.9786022604 | 3.9785754383 | 3.9779589474 | 3.9779545367 |
| 47 | 3.9780568480 | 3.9786642790 | 3.9786385000 | 3.9781171083 | 3.9781140983 |

Learning radius lowers the final probe CE for every seed relative to its fixed
initial-radius control. The mean differences are only **-0.0000260671** from R=1
and **-0.00000369549** from R=4. These are small deterministic observations, not
statistical significance or practically meaningful quality gains.

The tangent control remains best for all seeds. Learned-from-R=4 still trails it
by mean CE **+0.0000693997**. Increasing radius weakens projection nonlinearity
and approaches the tangent map; closing that gap is not evidence that geometry
outperforms a linear adapter. The scalar moves upward in every learning arm;
last recorded pre-update log radii are approximately 0.0293-0.0300 from zero,
and 1.4143-1.4155 from log(4). Those are **before update 32**, not the final saved
parameter values.

This development probe was already used by the preceding pilot; it is not a new
untouched test set. The book may have been in GPT-2 pretraining. Longer learning,
unseen corpora and nonlinear matched controls remain necessary before any
generalization or geometric-advantage claim.

## Learning, Restart And Validation

All 15 active conditions have finite, nonzero gate/bias gradients and updates.
All six learned-radius conditions have a zero first radius gradient followed
by finite nonzero radius gradients. All 18 retain identical frozen-base hashes
and no base gradients. All 15 active runs reload their saved adapter/Adam state
from disk and reproduce the exact next parameter update and optimizer state.
There are **480 primary updates and 30 extra validation-only updates**; endpoint
losses precede those continuation checks. All 18 checkpoint hashes were verified.

- Entire native st-nn library: **759 passed**, zero failed/ignored.
- Python geometry/protocol tests: **42 passed**, zero skipped. Independent
  float64 Torch autograd checks all four VJPs on CPU and via MPS transport.
- Public import/type tests: **22 passed**.
- Native extension, WASM nn and minimal extension-module/text builds pass.
- Actual browser forward, four VJPs and conditioning match the native fixture
  with maximum error **zero**. Radius-only teacher fitting lowers MSE from
  0.0234444542 to 8.88e-16; serialized-state SGD continuation matches exactly.
  This teacher check is not an LLM quality measurement.
- Pinned formatter and targeted Ruff pass. Non-strict st-nn Clippy exits zero
  with 23 existing warnings, none in the new radius/learning module; it is not
  claimed strict-clean.

`results.json` contains every condition, schedule, loss, gradient, diagnostic and
checkpoint hash. `plan.json` is the pre-execution running/empty-run plan, not a
completion receipt. `summary.json` binds its derivation to the results hash;
`validation.json` binds source, native/WASM binaries and local raw-log hashes.
Original logs and small optimizer checkpoints stay local. No book or base-model
weights are committed.

## Reproduction

Build the native extension with its nn feature, then run:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
python bindings/st-py/examples/hf_wave_gate_conditioning.py --config bindings/st-py/examples/hf_wave_gate_pride_radius.json --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" --output-dir "$NEW_RESULT_DIRECTORY"
cargo test --locked -p st-nn --lib
python -m pytest --import-mode=importlib bindings/st-py/tests/test_wave_gate_radius.py bindings/st-py/tests/test_wave_gate_learning.py bindings/st-py/tests/test_wave_gate_conditioning_example.py -v -rs
```

For the browser, build `spiraltorch-wasm --features nn` for wasm32 and process
it with matching wasm-bindgen, target web. Serve generated module/snippets at
`/module/`, this study's `results.json` at `/pretrained.json`, and
`bindings/st-wasm/tests/wave_gate_radius.html` at `/` on loopback HTTP. Require
the page's visible passed result.

Measured with Rust 1.98.0, formatter nightly-2026-04-15, Python 3.12.6,
Torch 2.12.1, Transformers 4.57.6, wasm-bindgen 0.2.104, macOS aarch64,
debug native/WASM builds. Explicit host copies remain: this is not resident
WGPU geometry, AMP, higher-order differentiation or a legacy ModuleTrainer
radius parameter. No cross-algorithm speed comparison is made.
