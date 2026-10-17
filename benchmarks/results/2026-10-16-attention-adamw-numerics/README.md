# Shared AdamW Numerical Repair

This repairs a Rust/WGSL optimizer calculation exposed by the earlier
[language capacity study](../2026-10-16-language-capacity-budget/README.md).
It is not a language-quality, geometry-efficacy, or speed claim. No tolerance,
Attention gradient, parameter mask, optimizer control, or old result changed.

## What Changed

Bias correction now computes `1-beta^n` directly, using complement composition
`a+(1-a)*b` and the exact `u32` accepted-update clock. The previous implementation
squared float32 powers and subtracted the rounded result from one. A new Rust
regression failed before the patch at `beta=0.99, n=2` and passes afterward.
The 41,940 sampled beta/clock comparisons use an independent float64
`-expm1(n*log(beta))` oracle and a four-ULP bound; this is a sampled check, not
a proof of that bound over every legal configuration.

The shared change affects all resident AdamW parameters, including geometric
coordinates and routers. Moments, decay, rejection, freezing, and history
ownership are unchanged. This remains raw-coordinate AdamW, not Riemannian Adam.

## Same-Gradient Results

Three fixed synthetic gradient streams run for 1,024 updates each. The fixture
uses float32 controls and independent CPU PyTorch 2.12.1 AdamW with
`foreach=False, fused=False`. Values and both moments are checked at nine fixed
cursors against the original `atol=3e-6, rtol=5e-5` gate. Rust CPU and WGPU receive
identical gradients; Attention is absent.

| Condition | Final GPU/Torch max weight error before | After |
| --- | ---: | ---: |
| Default betas, constant gradients | 7.63685e-6 | 5.96046e-8 |
| Default betas, periodic gradients | 3.57628e-7 | 5.96046e-8 |
| Near-one betas, periodic gradients | 2.38419e-7 | 1.86265e-9 |

The baseline failed one weight component at the final constant-gradient cursor.
The corrected native and browser runs pass all matched float32 checks, with all
observations retained in [before](native-before.json), [after](native-after.json)
and [browser](browser.json). Native execution identified Apple M4 / Metal.
The browser record establishes actual `BrowserWebGpu` execution, **not hardware
identification**: WGPU 0.20 does not expose the selected adapter's hardware
identity here, so `hardware_execution` is explicitly `unverified`.

Eight additional restored-state probes cover two betas at accepted clocks 32,
1,024, `2^24`, and `u32::MAX`. Both clients are within 0-1 ULP of their independent
float64 oracle, below the fixed eight-ULP test bound. These are successful
synthetic updates, not billions of executed training steps. The baseline report
predates these additional probes; it only contains the three common streams.

**Remaining float64 drift:** The constant-gradient final weight trajectory still
differs from the scalar float64 oracle by up to 7.37184e-5, including one component
outside the float32 comparison envelope, both before and after. Float64 is an
explicit diagnostic, not part of the reported matched-float32 pass. This repair
does not eliminate accumulated float32 rounding or prove full-model agreement.

## Bounded Language Follow-Up

After the optimizer isolation, the [follow-up plan](PLAN.md) selected the first,
smallest prior condition, width16/seed11, before running the changed model.
The exact same request and independent Torch report were pinned to their earlier
committed hashes. All three models completed 1,024 updates; their new reports
and saved checkpoint remain private. [All numerical diagnostics](language.json)
are public, including the still-failed strict gate.

| Case | Failed final weight components before | After | Max error before | After |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 3 | 2 | 5.18091e-6 | 5.84920e-6 |
| Learned Poincare | 5 | 3 | 5.19402e-6 | 4.43087e-6 |
| Frozen Poincare | 4 | 3 | 6.02287e-6 | 4.52929e-6 |

All remaining failed components are in `block.0.qkv.bias`'s Key slice. Training
and development-loss errors are at most 9.53674e-7; neither moments nor other
weight components breach their checked envelopes in this seed. The ordinary
case's maximum weight error **increased**, despite fewer failing components.
This is not uniform full-model improvement, and all six earlier studies stay
unqualified. No held-out quality, free generation or model promotion was added.

Unlike the tightly matched optimizer-only comparison, this follow-up uses a
native debug CLI and uninterrupted run, versus the previous installed-wheel,
segmented run. Their build identities are recorded. These differences prevent
uniquely attributing the component-count change to the correction alone. The
useful result is that the strict failure persists and remains localized to
Key-bias, motivating an Attention invariance/VJP experiment, not a relaxed gate
or simply ignoring that parameter.

`language_probe.py` runs the pinned private request once into a new directory,
checks input/build drift and uses the existing comparator and localization
helpers. Six negative/retention tests pass. Process success means that the
measurement completed; `strict_gate.passed` remains the numerical decision.
After this run, review added explicit 0700/0600 output permissions and a final
source/input check after comparison. These are future-run safeguards, not checks
retroactively attributed to this run. The exact executed script remains private
under its [started-record hash](language-started.json). Its archived bytes, the
binary and the unchanged comparison-helper hashes were rechecked after completion.
The retained experiment directory has restricted permissions.

## Verification And Reproduction

The shared Rust suite passes 127 tests; the resident-parameter suite passes eight;
the native byte decoder/study suite passes 120. These counts overlap and are not
independent experimental samples. Native and WASM strict Clippy pass. The browser
also runs the existing 12-update transactional controls, including same-build
bitwise resume, preserved snapshots, frozen history and atomic rejection.

```bash
cargo +1.99.0 test --locked -p st-kernel-contracts
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --lib resident_training::parameters -- --test-threads=1
cargo +1.99.0 run --locked -p st-backend-wgpu --example resident_adamw_numerics
```

Regenerate the fixture with `tools/generate_resident_adamw_numerics_fixture.py`;
it reproduced byte-for-byte locally. Build `resident_parameters_browser` for
`wasm32-unknown-unknown`, generate web files with wasm-bindgen 0.2.129, and serve
only those assets alongside `adamw_numerics_browser.html` over loopback. A WASM
build is not a browser execution result.

Older checkpoints remain loadable, but exact continuation evidence is tied to
the same build/device. The optimizer policy name is not a kernel fingerprint.
No new Python wheel was built or uploaded in this experiment. The production
update is in the common Rust backend; no independent Python rule was added.

[Verification](verification.json), [source identities](source-hashes.json) and
[private-file hashes](private-hashes.json) bind the retained outputs. Hashes
identify bytes; they are not signatures or independent execution attestation.
The baseline kernels and original failed-test log are retained privately. No
raw corpus or trained language weights/moments/checkpoints are published;
validation binaries also remain private. Public optimizer fixtures are synthetic.
