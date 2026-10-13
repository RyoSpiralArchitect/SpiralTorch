# Phase Trajectory And Residual Association

This is a diagnosis of the remaining geometric QKV learning discrepancy, not
a kernel optimization or a replacement qualification result. The original v1
qualification remains **4/6 for native and browser**, with both historical
failure classes retained. No runtime arithmetic or numerical threshold changed.

## What Was Separated

At update 12 of the two-block projective case, the current GPU phase VJP agrees
much more closely with CPU f32 evaluated at the **same parameter state** than
with CPU f32 on its separately evolved reference trajectory:

| Phase relative L2 at update 12 | Native | Browser |
| --- | ---: | ---: |
| GPU versus frozen CPU trajectory | 0.003332629851 | 0.003369978672 |
| GPU versus CPU f32 at the same runtime state | 0.000023886593 | 0.000071682919 |
| Runtime-state versus frozen-state effect, both evaluated in CPU f64 | 0.003208363301 | 0.003266393880 |

All 37 parameter tensors were substituted individually in both directions at
the pre-update-12 state. The largest individual sensitivities were the first
block's output bias and feed-down bias, head weight, and the second block's
feed-down and output biases. These effects are nonlinear and are **not additive
causal percentages**. All substitutions and all 16 same-state steps are retained
in [diagnosis.json](diagnosis.json), not only the largest five.

The reference and composed Rust block also had different addition association:

```text
Frozen Torch: (x + attention @ W) + bias
Rust modules: x + (attention @ W + bias)
```

The second expression completes the Linear projection, including its bias,
before adding the residual. Floating-point results need not agree between
these two graphs. An optional reference-only flag now evaluates the module
grouping; its default retains the frozen legacy graph. The Rust API docs and
shared native/WASM tests make the existing module boundary explicit.

## Alternate-Graph Diagnostic

Before the alternate evaluation, CPU f32 exactly replayed all six frozen
initial outputs/parameter VJPs and all 16 parameter/geometry-gradient updates.
Then only the CPU residual association changed. Initial parameters, data,
learning rate 0.125, and thresholds remained fixed. The measured QKV native and
browser reports are reused unchanged from commit `d7922413`; **no new QKV GPU
trajectory is implied by these CPU-only replays**.

| Phase relative L2 at update 12 | Frozen association | Module-grouped association |
| --- | ---: | ---: |
| Native | 0.003332629851 | 0.001835436664 |
| Browser | 0.003369978672 | 0.001871310235 |

Under the alternate graph, all 1,440 geometry-gradient checks across the six
cases, 16 updates and two runtimes meet the existing relative threshold 0.002
and non-inert reference/actual conditions. All 12 initial-output checks, 384
initial parameter-VJP checks and 192 complete updated-parameter families meet
the unchanged absolute tolerance `3e-6 + 5e-5 * abs(reference)`.
[association.json](association.json) retains every case, step and gradient
scalar, along with the original failures. These are **alternate comparisons**,
not a relabeling of v1 as 6/6. The legacy mixed-runtime query-control failure is
not addressed or re-evaluated by this association probe.

## Boundary Tests And Limits

Fresh native Metal and browser WASM/WebGPU runs each passed 60 forward/VJP
cases, 32 plain plus 32 Topos updates, 12 guards and four new residual-boundary
controls. The new controls use signed powers `2^24` and `2^48`: the completed
projection yields residual output `+1` or `-1`, whereas separately reassociated
resident additions yield zero. Both input and projection-bias VJPs are 2; those
derivatives verify intact routes, not association discrimination by themselves.

The ten CPU tests pass both normally and under optimized Python (`-O`). They
cover the analytic boundary and rejection of missing cases, truncated updates,
missing tensors, wrong names/shapes/revisions, non-finite data and changed
legacy replay values. Input validation and replay checks use explicit errors,
not assertions that disappear under optimization.

The f64 diagnostics retain some legacy f32 constants and are not exact-real
oracles. Same-state update residuals mix gradient and update-rounding effects;
they do not prove an SGD defect. This six-case, short-run result is not a claim
about large geometry mixtures, corpus quality, LLM fine-tuning, speed or CUDA.
The next qualification must freeze a separately versioned, matched graph rather
than waive the original failure. Geometry-combination learning can then be
expanded without confusing reference association with geometry efficacy.

- [Initial diagnosis protocol](protocol.json) and [alternate-graph protocol](association-protocol.json).
- [Fresh native/browser scalar controls](residual-controls.json).
- [Validation and hashes](validation.json), [independent review](review.md), and [checksums](SHA256SUMS).
- [Previous unchanged QKV qualification](../2026-10-13-wave-batch-adjoint-v1/README.md).

Raw weights, vectors, checkpoints, binaries and screenshots remain local.
No PR, merge, CI run or release is part of this increment.

## Reproduce

Use a fresh private `RAW`. `FIXTURE` is the frozen projective-v1 `reference.json`
with SHA-256 `07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358`.
`CURRENT` holds the prior wave-batch milestone's `native-projective.json` and
`browser-lm.json`; their hashes are in both diagnostic files. The
[previous recipe](../2026-10-13-wave-batch-adjoint-v1/README.md#reproduce) describes
recreating these inputs. Use CPU Torch 2.12.1, Python 3.12 and one Torch thread.
The scripts pin the fixture, verify the browser fixture identity, validate all
runtime case/tensor/step identities, record all input hashes and refuse to
overwrite output files. Other runtime receipts are not silently identified as
the retained experiment: compare their hashes with the published inputs.

```sh
python3 -I -S -B -c 'import sys,sysconfig,runpy;sys.path.append(sysconfig.get_path("purelib"));sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/diagnose_byte_phase_trajectory.py "$FIXTURE" "$CURRENT" "$RAW/diagnosis.json"
python3 -O -I -S -B -c 'import sys,sysconfig,runpy;sys.path.append(sysconfig.get_path("purelib"));sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/diagnose_byte_residual_association.py "$FIXTURE" "$CURRENT" "$RAW/association.json"
python3 -I -S -B -c 'import sys,sysconfig,unittest;sys.path.append(sysconfig.get_path("purelib"));sys.path.insert(0,"tools");import torch;torch.set_num_threads(1);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName("test_residual_association_reference"));raise SystemExit(not r.wasSuccessful())'
```

Repeat the test command with `-O` to check optimized execution. The imports
above intentionally avoid user site customization and MPS monkey patches.

```sh
export CARGO_INCREMENTAL=0
export SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1
cargo test --locked -p st-nn --no-default-features --features wgpu \
  --test resident_residual_attention -- --nocapture --test-threads=1 > "$RAW/native-residual.log" 2>&1
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown --example resident_residual_attention_browser
wasm-bindgen "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/examples/resident_residual_attention_browser.wasm" \
  --target web --out-dir "$RAW/web"
```

Use Rust 1.97.0, wasm-bindgen 0.2.129 and the repository lockfile; set
`CARGO_TARGET_DIR` explicitly or substitute the normal `target` path. Run builds
and GPU tests serially. On loopback only, serve
`crates/st-nn/tests/residual_attention_browser.html` and map its two generated
JS/WASM assets to `target/residual-attention-web/`. Do not expose the entire
private artifact directory. Run the page once, inspect its result and retain
the download. A successful native integration reports one passing test with
the embedded scalar JSON; the browser explicitly reports all four boundary
controls along with the existing checks.
