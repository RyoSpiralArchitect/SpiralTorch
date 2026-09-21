# Bounded Checked GELU

This follows PR #2106. The only production change is in `Tensor::try_gelu`,
also used by host `st_nn::Gelu::forward`. Baseline:
`5c57b2326525b0b8529724e5e0c9ba4542c8d3bb`; candidate:
`6bf69c8656e31f63d3021b44044f67a0b2e4dc2b`.
The baseline already contains the expanded tests and benchmark grid.

## Change And Safety

During input validation, classify whether every value has `abs(x) <= 1e12f32`.
Within this conservative bound, even the rounded cube is around 1e36, far below
the float32 limit; all remaining intermediates are finite. Run the same scalar
tanh-GELU arithmetic without repeating finite checks for each intermediate.
This is a proof bound, not an activation cutoff, saturation rule or tuning knob.

Any finite outlier selects the original checked evaluation for the whole batch.
Continue scanning after an outlier so NaN/Inf input errors still take precedence
over earlier intermediate overflow. Preserve signed zeros, evaluation order,
row-major logical output, input aliases and allocation ownership. No unsafe code,
fast-math, route override or public API is added. Backward is unchanged.

The 12,288-value deterministic exponent fixture includes subnormals, signed zeros,
and both sides of the bound. Entirely bounded and mixed/outlier batches are tested
separately in RowMajor, ColMajor and Chimera layouts. Native tests compare exact
float32 bits with a frozen scalar evaluation, not just an approximate reference.

## Measurements

All runs are on the recorded Apple M4 host, with four Rayon threads. Host
exclusivity and thermal state are unknown. Precondition both workers, then run
baseline/candidate and candidate/baseline. Do not trim intervals or exclude cases.
Ratios below are geometric means of per-condition median-time ratios,
baseline/candidate, so greater than one favors the candidate.

| Checked GELU forward | All 9 shapes | At most 64 elements | More than 64 elements |
| --- | ---: | ---: | ---: |
| Native CPU Module | 1.171x | 1.118x | 1.216x |
| Node CPU WASM shim | 1.059x | 1.021x | 1.090x |

All ten non-tiny comparisons favor the candidate on each target. Tiny WASM is
mixed: 5/8 favorable, ratios 0.975-1.074. Native backward, an unchanged control,
is 1.010x overall and 1.003x non-tiny. Allocation requests do not change.

The unchanged 90-condition NN grid retains packing, transpose, warmed Linear,
warmed Linear-GELU-Linear and parameter-cache invalidation across Auto/Faer/CpuSimd.
Warmed MLP ratios are 1.012x/1.033x/1.028x respectively, with unfavorable cases;
invalidated Auto MLP is 0.982x. Unchanged Auto Linear is 0.971x and packing 0.941x,
including substantial outliers. These noisy whole-model comparisons do not
establish a training-throughput gain. Cache invalidation is not an optimizer step.

Torch 2.12.1 CPU uses the same float32 fixture, tanh approximation, four intra-op
and one inter-op thread. For non-tiny shapes, Torch/candidate is **0.647x forward**
and **0.535x VJP**: Torch remains faster overall. Tiny aggregates favor Rust, but
Torch has Python entry overhead and its VJP traverses a retained autograd graph;
Rust measures a supplied-seed Module derivative. Guards also differ. These are
not fastest-PyTorch, equivalent-entry-cost, browser, GPU or FT claims.

The complete cohorts contain 504 measured conditions and 252 preconditioning
conditions: native GELU 72+36, WASM GELU 36+18, NN 360+180, Torch 36+18.
GELU uses three warmups and 15 intervals; native/Torch use eight calls per
interval, WASM uses 512 for tiny inputs and eight otherwise. NN uses three
warmups, nine intervals and two calls. Output allocation/free is timed; setup
and output export are not. Independent float64 checks pass throughout. Old/new
GELU output-bit hashes agree within each target, including WASM wide fixtures.

## WASM Boundary

`reproduction/wasm-shim` is a measurement-only wasm-bindgen adapter calling the
real Rust `Tensor::try_gelu`. JS invocation and owning-result free are timed.
It is **not** a new deployed API. Public WASM autograd forward still uses its
existing unchecked activation path; resident Sequential still uses WebGPU.
Neither path inherits this checked-forward speed claim. No browser was timed.
The shim lockfile uses the same dependency versions as the repository lockfile.

## Validation And Reproduction

All 13 local stages passed at the candidate commit: formatting; 486 tensor tests;
715 NN unit tests plus five GELU/one Linear integration tests; scoped strict native
and WASM Clippy; WASM build and binding generation; four public WASM GELU and 20
matmul cases; Python extension build plus two GELU/six autograd tests; and six
opted-in WGPU integration tests including nine strict-GPU layout pairs.
See the exact commands and source hashes in `verification/*/receipt.json`.
Hosted CI and review status live on PR #2107, not in the timing claim.

`manifest.json` binds the exact published bytes. Run:

```sh
python3 -B -I verify.py .
python3 -B -I test_verify.py .
```

This checks integrity, source/worker identities, full cohorts, output-hash
agreement and derived ratios. It does **not** rerun kernels or establish hardware
performance from hashes. Raw arrays and compiled workers remain at the local root
recorded in `provenance.json`; their hashes are public, not their payloads.

For numerical/performance replay, use separate clean checkouts of the two commits.
Build `cpu_gelu` and `cpu_nn_layout` with Cargo 1.98.0, `--locked --release`,
four build jobs, and freeze each worker before building the other version.
Copy the shim to a temporary directory and point its three repository paths at
the selected checkout, preserving the recorded lockfile. Build its wasm32 release
with `--locked`, then generate Node bindings with wasm-bindgen 0.2.104. Freeze the
JS/WASM pair under `baseline-build/wasm` or `candidate-build/wasm`.

Arrange both worker directories below a new log root alongside the reproduction
drivers. Run `measure.py LOG native`, `measure.py LOG wasm`, then
`nn_measure.py LOG/nn LOG/baseline-build/cpu_nn_layout LOG/candidate-build/cpu_nn_layout`.
Run `torch_gelu.py` with Torch 2.12.1 separately; do not overlap builds/timing.
The recorded `check.py` preserves original local paths; for another machine,
point its Python extension loader to the included `python-client.py`.
Each driver fails rather than overwrite an existing cohort. A new machine/compiler
is a new numerical/performance observation, not an identical timing reproduction.
