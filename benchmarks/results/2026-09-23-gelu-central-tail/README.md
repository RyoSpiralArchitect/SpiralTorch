# GELU Liveness With Central-Domain Training Compatibility

Measured source: `4c872c2c6ddace6a38ad39d1cdda247e9ac25289`.
Original optimization base: `62dced674158234726cfb2521c09d378e8ec3b6f`.
This is a separate complete measurement of the final central/tail correction,
not a relabeling or pooling of either earlier candidate's intervals.

## Correction And Regression Coverage

Ordinary Tensor GELU backward computes and reads only its live gradient;
the full residual/bias helper retains its outputs with batched observation.
The Rust backend owns their embedded shaders, layouts and checked plans.
Native, WASM, Tensor and resident VJP/training share the derivative source.

Strict GPU CI first exposed cancellation near saturated tanh tails, then a
full-domain exp rewrite passed single-operation checks but regressed an existing
classification trajectory. Both investigations and original logs remain in
[the first failure record](../2026-09-23-gelu-backward/CI-FOLLOWUP.md) and
[the classification failure record](../2026-09-23-gelu-stable-tail/CI-FOLLOWUP.md).
The latter reproduced locally at seed 17, ModuleCompatible policy, step 25.
Changing only the derivative back to its old expression passed that integration
test. The full-domain rewrite is therefore retained as a rejected candidate,
despite its passing local scalar accuracy and timing checks.

The final derivative retains the established tanh evaluation order at abs(x)<=3,
uses the equivalent exp identity outside that central domain, and keeps the
exact abs(x)>=10 saturation policy. CPU code, finite guards, layouts, residual
accumulation and fallbacks are unchanged. No original fixture, learning rate,
step count or numerical tolerance was weakened. An additional test covers four
adjacent f32 values on each side of both +/-3 boundaries, plus the boundaries.
A ReLU branch crossing could amplify small rounding changes during independent
trajectories, but that internal mechanism was not directly measured here.

## Results

The frozen six shapes, two live-output contracts, bursts 1/4 and balanced
nine-block order were repeated in three serial runtime-order-rotated rounds:
**5,184 final intervals**, separate from 5,184 screening intervals.

| Live outputs | Native legacy/candidate | Browser legacy/candidate |
| --- | ---: | ---: |
| Gradient only | 2.836x | 2.762x |
| Gradient, residual and bias | 1.959x | 2.132x |

These are descriptive geometric means of twelve median paired ratios, not
universal speed guarantees. All combined cells exceed 1 in this run. For one
output, removing unused reads gives 2.194x native / 2.564x browser; removing
unused fused computation gives 1.262x / 1.005x. The latter browser factor is
greater than 1 in only three cells and reaches 0.800x in one. For three outputs,
the formula-only factor is 0.994x / 0.910x, while batching reads gives
1.966x / 2.316x. Slower factors are preserved, not discarded as outliers.

At 128x1025, one output, burst=1, median intervals are 0.235 ms native plain,
0.400 ms browser plain, 0.205 ms Torch CPU and 0.791 ms Torch MPS. Torch CPU is
still faster in some cells. Torch 2.12.1 runs the actual matched ATen tanh-GELU
backward, preallocated packed outputs and one owning CPU copy, with 4/1 threads,
no MPS fallback and no torch.compile.

The prepared interval includes encoding/bindings, per-operation submissions
and residual reset, and terminal CPU ownership. It excludes allocations,
fixed uploads, compilation, checks and JSON. Bursts repeat one operation and
observe its last result, not a training trajectory. WGPU controls share the
same snapshot decoder rather than byte-identical historical read_buffer calls.
This is not a whole-Tensor/model multiplier, a paired estimate across the three
studies, or universal superiority to PyTorch. The host is shared; browser clock
granularity limits small differences. No CUDA/Furnace or model-quality claim.

## Evidence And Replay

All **29 clean-source stages** pass: the original 23-stage benchmark/validation
protocol in validation.json, plus six training stages in training-validation.json.
The latter independently checks the same clean source identity before collecting
receipts. It includes all six native resident integration tests, 772 NN library
and twelve layout tests, seven self-supervised tests, and the real browser graph
training fixture. The browser covers six 64-step classification cases, twelve
64-step custom-objective cases, resident losses, VJP ownership, fusion, resume,
microbatch accumulation, clipping and momentum. These are regression fixtures,
not evidence of improved language-model quality.

The base stages also pass 192 backend library + one WGSL + two example tests,
16 autograd + four strict GPU GELU tests, twelve strict NN layout tests, 1,001
core tests, native/WASM lint and all nine timing runs. Maximum absolute error
is 2.9010104843e-6 and maximum scaled error 0.365060258 < 1 under the unchanged
abs=2e-6, rel=1e-5 gate; only bias absolute allowance scales with row count.
The historical backend's four nonfinite outputs after syntax-only repair remain
an explicit diagnostic, not part of the passing numerical gate.

exploration.json retains all twenty local screening/control stages, including
the failed full-domain expression and old-expression/hybrid controls. Raw data
lists **279 files / 2,215,886,705 bytes**. Arrays, executables and generated
WASM/JS remain local; all conditions, timing intervals, receipts, failure logs,
source/asset hashes and the training-summary collector are public.
Native identifies Apple M4/Metal. Browser Rust identifies BrowserWebGpu; its
separate Apple adapter probe is not attestation of the Rust device identity.
Low-level buffer/uniform ownership and batched-readback peak-memory caveats
remain as documented in [the protocol](../../gelu-backward/README.md).

```sh
python3 -I -B benchmarks/gelu-backward/evidence.py verify benchmarks/results/2026-09-23-gelu-central-tail
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn --no-default-features --features wgpu --test resident_graph_training -- --test-threads=1
```

Use --raw-root and --source-root with the verifier to rehash/recompute evidence
and check measured source. This does not execute GPU kernels. The extra collector
expects the preserved raw tree and a fresh published archive; training requires
the recorded native test or browser fixture commands. Final CI/review acceptance
is recorded separately on [PR #2118](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2118).
