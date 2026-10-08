# Elliptic/Lie Learning Bridge: Pretrained Wiring Evidence

Date: 2026-10-02. Frozen source: `83745e04a9fbb0344f1bb7f867279698db68f95c`.
This is a small **learning-path and numerical-correctness experiment**, not a
speed comparison or evidence that elliptic geometry improves language quality.

## Implementation And Validation

Rust now owns a validated batch of the existing nine elliptic features and its
immutable first-order VJP snapshot. Python autograd and WASM consume that same
implementation. The residual adapter trains `F -> 2 -> (1,u,v) -> 9 -> F`, with
an identity-initialized output projection and a local hemisphere chart.

Three numerical regressions are reproduced in `before.json`: small polar angles
and rotor features collapsed to zero, an azimuth denominator floor suppressed
the near-pole gradient, and large finite directions overflowed normalization.
The repaired map uses atan2/hypot and wider norm intermediates; VJP contraction
accumulates in f64 before returning f32. Undefined or unrepresentable chart
derivatives and nonfinite outputs fail rather than silently returning zeros.

- Rust: three new learning tests and 16 existing microlocal-related tests pass.
- Python: 18 actual-extension tests pass, zero skipped, covering elliptic and
  Topos finite differences, saved-input versions, immutable recipes, empty and
  noncontiguous batches, context-local telemetry, exact next-update resume,
  pointwise causality, tiny HF losses and MPS transport.
- Public import/type surface: 22 unittest checks pass. An earlier wrong-path
  invocation collected nothing; a subsequent pytest invocation picked up an
  unrelated parent package and failed collection. Direct unittest execution
  isolates this checkout. Failed invocation logs remain in the local record.
- Browser: native/WASM forward and VJP maximum errors are both **zero** for the
  ordinary, near-pole and large-finite fixtures. A 100-update local-coordinate
  fit reduces MSE from `0.0488547796` to `0.000000692863`. The target comes from
  the same operator: this establishes trainability, not generalization.
- Nine saved adapter/optimizer checkpoints reload with `weights_only=True`;
  checkpoint hashes are published, while the small binaries stay local.

Native/WASM builds, pinned workspace formatting, core strict Clippy and the
changed Python implementation/tests/example Ruff checks pass. Whole WASM strict
Clippy was not rerun here: the preceding Topos milestone recorded 19 findings
in unchanged files. This is not a claim of a clean workspace-wide lint suite.

## Pretrained GPT-2 Controls

The local cached GPT-2 snapshot is `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
All base weights are frozen and hash-identical before and after every arm.
The adapter is placed after `transformer.h.0.mlp`, width 768. The model runs
in eval mode, CPU f32; dropout is off. There are two authored training sentences
and two distinct development sentences, all included with token IDs in the report.
Padding labels are masked; HF owns the ordinary shifted causal-LM loss.

Seeds are 17, 29 and 43. Each active arm takes six Adam updates at 0.001,
strength 0.1, with 8,450 trainable parameters. The off arm bypasses geometry and
does not update. The tangent control is the first-order map at `(1,0,0)`, derived
from the Rust Jacobian, with the same initial affine weights and parameter count.
The controls are update/parameter matched, **not compute-cost matched**.

Initial train CE is `6.2567892075`; development CE is `5.3481988907` for all arms.

| Seed | Tangent train CE | Elliptic train CE | Tangent dev CE | Elliptic dev CE |
| --- | ---: | ---: | ---: | ---: |
| 17 | 6.235655 | 6.243685 | 5.342649 | 5.344567 |
| 29 | 6.232206 | 6.242518 | 5.340158 | 5.343521 |
| 43 | 6.234463 | 6.241402 | 5.342145 | 5.343454 |

Both active arms send nonzero gradients through both projections and change
logits. The off arm stays identical. **The tangent control is better on train
and development CE in every seed.** No optimum, statistical significance, LoRA
advantage or production FT benefit is established. Repeating the full run on the
frozen final implementation reproduces every numerical observation exactly;
the added checkpoint records are excluded from that comparison. Browser results
also repeat exactly.

## Artifacts And Reproduction

`pretrained.json` contains all nine conditions and every loss/gradient observation.
`before.json` is the older, frozen native extension's three-row numerical probe.
`browser.json` contains the complete browser trajectory. `checkpoint-load.json`
records safe checkpoint loading. `provenance.json` binds source, model, binary
and local-log hashes; `SHA256SUMS` covers all public files. No model weights,
credentials, full checkpoints or machine-local paths are published.

From an isolated source checkout with Torch and Transformers installed:

```bash
maturin develop --manifest-path bindings/st-py/Cargo.toml --no-default-features --features python-default,cpu
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest bindings/st-py/tests/test_elliptic_learning.py bindings/st-py/tests/test_geometry_autograd.py bindings/st-py/tests/test_elliptic_stub_runtime.py -v -rs
python tests/test_runtime_imports.py -v
python bindings/st-py/examples/hf_elliptic_learning.py --model-dir "$LOCAL_GPT2" --block transformer.h.0.mlp --features 768 --checkpoint-dir "$NEW_CHECKPOINT_DIR" > pretrained.json
cargo test --locked -p st-core --test elliptic_learning
cargo test --locked -p st-core --lib microlocal
```

The example downloads nothing and refuses to overwrite existing checkpoints.
The recorded local native build used default features, but the geometry itself
does not need a GPU feature. Recorded versions: Rust 1.98.0, Python 3.12.6,
Torch 2.12.1, Transformers 4.57.6, wasm-bindgen 0.2.104; both builds are debug.

For the browser, build `spiraltorch-wasm` for `wasm32-unknown-unknown`, then use
matching `wasm-bindgen --target web`. Serve its generated module and snippets at
`/module/spiraltorch_wasm.js`, the native report at `/pretrained.json`, and
`bindings/st-wasm/tests/elliptic_learning.html` at `/index.html`. Read the pass/fail
report in `#result`. This is scalar WASM, not WebGPU.

GPU Torch inputs explicitly round-trip through the CPU Rust map. Resident WGPU,
CUDA, AMP, higher-order derivatives, compilation and distributed/sharded model
support are not claimed. Standard Torch-equivalent operator performance belongs
to a separate benchmark. See the [API guide](../../../docs/elliptic_learning_bridge.md).
