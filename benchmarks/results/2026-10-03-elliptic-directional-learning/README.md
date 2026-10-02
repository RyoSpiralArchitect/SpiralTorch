# Elliptic Directional Learning and Chart Probe

Source: `d7abe0beeca216b5d2b09e496703656877cfc4f4`.
See the [operator/client contract](../../../docs/elliptic_directional_learning.md).
This record contains first-order derivative verification, synthetic fitting and
an exploratory diagnosis of completed training. It does not establish better
language-model quality, a causal explanation of the CE gap, or a speed advantage.

## Training-Only Observations

The completed [anchored study](../2026-10-03-elliptic-anchored-study/README.md)
supplies the frozen model and checkpoints. Six trained projections plus three
paired initial projections see the same 16 uniformly spaced **training** blocks
(2048 token rows). Both nonlinear and affine maps are evaluated at each identical
orientation. No model training or endpoint evaluation is performed. This is not
an independent replication of the training study.

Bare elliptic chart statistics at the trained elliptic projection:

| Seed | Median coordinate norm | Median sigma-min | Median sigma-max | Median condition | Condition p90 | Covariance participation ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 16.1407 | 0.009968 | 0.954009 | 85.5078 | 343.6670 | 1.2848 |
| 43 | 15.1414 | 0.015981 | 0.448040 | 26.6834 | 113.5498 | 1.1552 |
| 47 | 16.8505 | 0.013094 | 0.435796 | 31.6936 | 151.9796 | 1.2145 |

At initialization, median coordinate norm is 0.69-0.77 and median condition
1.27-1.51. The ordinary affine chart's condition is constant at 1.0934. Full
distributions and all trained-tangent counterfactuals are in `chart-probe.json.gz`.
These are separate distribution medians: median condition need not equal the
ratio of the two singular-value medians.

The observation is increased directional anisotropy, **not** uniform gradient
collapse. The probe excludes the learned anchor gate and readout; the complete
model may amplify or rotate those directions. It does not prove that a
preconditioner, alternate chart or additional gate would close the quality gap.
JVP basis values exactly equal VJP basis values throughout. Every metric also
exactly reproduces the preceding frozen VJP-only local diagnostic.

## Learning and Correctness

- Rust: 16 tests, zero skipped; finite differences, adjoint identity, pullback
  nonnegative quadratic form, signed/shared gates, zero/empty/overflow cases.
  Clippy with `-D warnings` and workspace formatting pass.
- Python: 189 related tests, zero skipped. Forward-mode directions agree with
  native snapshots, ordinary reverse-mode is retained, and updated residual
  adapters propagate cross-entropy directions consistently.
- Python's synthetic matrix-free damped Gauss-Newton example accepts five
  updates: half-squared feature loss 2.3235356 -> 1.1818324e-13.
- Actual Node-hosted WASM passes directional/adjoint checks and 17 curvature-based
  gradient updates: loss 2.3235355 -> 4.8437143e-11. Strict TypeScript passes.

The two synthetic clients use different solvers; their update counts are **not**
a backend performance comparison. No new browser, WGPU, CUDA or pretrained-FT
optimization result is claimed. Production adapter backward rules are unchanged.

## Reproduce

Build/install the bindings from the source above. Paths below are local inputs;
no corpus text, hidden activations, model weights, checkpoints or raw logs are
published. `validation.json` binds source/runtime/binary hashes and local logs.
`SHA256SUMS` covers this record's files.

```sh
cargo +1.98.0 test -p st-core --no-default-features \
  --test elliptic_learning --test elliptic_anchored --test elliptic_gated_causal
cargo +1.98.0 clippy -p st-core --no-default-features \
  --test elliptic_learning --test elliptic_anchored -- -D warnings
cargo +nightly-2026-04-15 fmt --all -- --check
cargo +1.98.0 build -p spiraltorch-py
python -m pytest --import-mode=importlib -q \
  bindings/st-py/tests/test_elliptic_*.py \
  bindings/st-py/tests/test_geometry_*.py \
  bindings/st-py/tests/test_wave_gate_*.py
python bindings/st-py/examples/elliptic_pullback_fit.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_elliptic_chart_probe.py \
  --study "$COMPLETED_ANCHORED_STUDY" --model-dir "$LOCAL_GPT2" \
  --corpus "$LOCAL_PRIDE" --transfer-corpus "$LOCAL_ALICE" \
  --output "$NEW_PROBE_JSON"

cargo +1.98.0 build -p spiraltorch-wasm --target wasm32-unknown-unknown
wasm-bindgen --target nodejs --out-dir "$WASM_MODULE_DIR" \
  target/wasm32-unknown-unknown/debug/spiraltorch_wasm.wasm
node bindings/st-wasm/tests/elliptic_directional.mjs \
  "$WASM_MODULE_DIR/spiraltorch_wasm.js"
tsc --noEmit --strict --target ES2020 --moduleResolution node \
  bindings/st-wasm/types/spiraltorch-wasm.d.ts \
  bindings/st-wasm/tests/elliptic_directional_types.ts
```

Use wasm-bindgen 0.2.104, matching the lockfile. The probe validates the completed
study's model/data/checkpoint identities and refuses to replace an existing
output. To recreate the underlying private checkpoint set, use the earlier
study's published plan and reproduction instructions. The Python and WASM JVP
learning tests are included in CI rather than relying only on this local record.
