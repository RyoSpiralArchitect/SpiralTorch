# Checked Host GELU: Layout Correctness And Copy Removal

This follows the merged resident/cross-crate stack (#2103, #2104, #2105).
It fixes an existing logical-layout bug while reducing CPU output preparation,
without changing the scalar approximation, finite checks or Auto routing.

## Source And Correctness

- Baseline worker: `9efb5bf58bdb14472a6fb709d16f70e1d9868ad9`. Runtime code is
  unchanged from the preceding `e2cab541` checkpoint; only tests/harness were added.
- Measured candidate: `1308592db0dce79f4ff5c078a10b240475f8f06e`.
- Final validation: `8a75683b83238468c0f66ee80f095b355f6feae6`. The only subsequent
  change corrects the Python test's expected exception to the existing ValueError.
  The failed first attempt is preserved, not counted as successful validation.
- Review follow-up: `4cbf27b421bb5d0b47cfbb556cff9e2354a31985` adds the two client
  tests to CI's explicit invocation lists. The recorded local checks already ran
  them; this two-line CI change does not alter the measured/validated runtime.

The pre-fix forward and backward layout tests both fail. Previously, column-major
and Chimera storage was consumed as row-major, including mixed input/seed layouts.
Now checked NN forward delegates to `Tensor::try_gelu`; host backward normalizes
logical pairs before CPU or WGPU dispatch. The permissive Module fallback also
uses logical row-major values. CPU results are written directly into aligned
owning storage instead of first building and copying a temporary Vec.

Input errors retain precedence over intermediate overflow. Error labels, signed
zero, empty shapes, aliases and the saturated derivative policy remain covered.
The unchecked inplace/autograd forward and resident forward policies are unchanged;
this is not a claim that every historical GELU entry point has identical guards.

## Measurements

Standalone shapes are `(rows, cols)` = `(1,64)`, `(8,3072)`, `(32,3072)`,
`(64,1024)`, `(17,195)`, `(65,97)`, for both forward and supplied-seed VJP.
Three warmups, 15 untrimmed intervals of eight calls, output allocation/free
included; input creation, output extraction and validation excluded.

Whole-grid preconditioning for both workers precedes AB/BA order: 48 measured
conditions plus 24 preconditioning conditions. All outputs pass independent f64
checks (`2e-6 * (1 + abs(reference))`), and every old/new output hash agrees across
both rounds and preconditioning. Hashes cover little-endian f32 bits.

Ratios are baseline median time / candidate median time. Geometric means summarize
12 shape-round pairs per direction, not confidence intervals:

| Boundary | Geometric Mean | Range | Favorable |
| --- | ---: | ---: | ---: |
| Module GELU forward | 1.062x | 1.032-1.140 | 12/12 |
| Module GELU supplied-seed VJP | 1.010x | 0.997-1.036 | 10/12 |

Allocation requests fall **4 -> 3** in both directions, saving one full output
buffer. At `32x3072`, allocated bytes fall **786,512 -> 393,296**. These are summed
allocation requests, not peak memory or RSS. The benchmark uses row-major inputs;
normalizing other layouts can still allocate, and is not included in these gains.

The unchanged real Linear/MLP harness also ran its complete 90-condition grid in
both directions of AB/BA: **360 measured + 180 preconditioning conditions**. It
covers six workloads, three CPU backend choices and cached/invalidation forwards,
plus packing/transpose controls. Every independent f64 check passes. Warmed MLP
ratios are Auto **1.026x**, Faer **0.990x**, CPU-SIMD **1.015x**. Unchanged controls
also vary (e.g. cached Auto Linear 0.984x). This remains a **mixed end-to-end result**,
not universal warmed-NN acceleration. Invalidation is not a real optimizer step.

PyTorch **2.12.1** CPU eager GELU uses the same f32 fixtures and tanh approximation,
four intra-op threads and one inter-op thread. Both rounds and one preconditioning
round cover all 12 conditions; all independent f64 checks pass. Torch/candidate
ratios geometrically average **0.783x forward**, **0.953x VJP**; only **4/12** and
**2/12** shape-round pairs favor SpiralTorch. Small-case Python entry overhead
strongly affects these aggregates. Torch VJP includes autograd traversal on a
retained forward graph; Rust invokes the explicit Module derivative directly.
Torch does not provide these intermediate-finiteness/error-label guarantees.
No fastest-PyTorch, parity or equivalent-runtime claim is made.

All intervals and regressions are retained. Host exclusivity and thermal state
are unknown. No training, FT quality, browser-speed or CUDA result is claimed.

## Validation And Evidence

All 13 final recorded stages pass: formatting; 486 Tensor CPU tests; all 715 NN
library tests plus three GELU and one Linear layout tests; scoped strict native
and WASM Clippy; WASM build/bindgen; four WASM GELU and 20 matmul forward/VJP cases;
fresh native Python build with two GELU and six packed-autograd tests; and four
WGPU-feature layout tests with the mandatory live strict WGPU case. That case
exercises all nine input/seed layout combinations without CPU fallback.

WASM GELU uses the existing AutogradTensor forward and the shared Tensor backward;
it does not expose or benchmark the new checked host forward as a browser API.
The Python test loads the recorded newly built extension, not an installed wheel.
Native strict Clippy is scoped to st-tensor/st-bench with `--no-deps`, not every
workspace crate. Verification of this archive checks recorded evidence, not a
fresh GPU execution or numerical replay.

`provenance.json` binds source revisions and local native/Python/WASM products.
Raw GELU arrays and executables stay local under its named root; their hashes and
all-condition compact results are published. Earlier published archives are not
changed. The initially filtered unit-only test run and the Python expectation
failure are retained alongside the complete corrected validation.

## Replay

Verify a clean Git export of this directory, without additional files:

```bash
python3 -B -I verify.py
python3 -B -I test_verify.py
```

An unlisted Finder file is correctly rejected by the strict byte manifest. Do not
delete unrelated local data or weaken the verifier; use a clean exported copy.

For numerical/performance replay, obtain the two source commits above in separate
checkouts, build `st-bench` examples `cpu_gelu` and `cpu_nn_layout` with Rust 1.98.0
in release mode, and freeze both products before running either comparison. Keep
four Rayon threads, the same features, and avoid simultaneous builds/workloads.
Use `reproduction/measure.py` with sibling `baseline-build/cpu_gelu` and
`candidate-build/cpu_gelu` paths. Use `reproduction/nn_measure.py OUTPUT BASELINE
CANDIDATE` for the full NN grid and `reproduction/torch_gelu.py` for Torch.
These scripts retain all intervals and do not select conditions after the result.

`reproduction/check.py` and per-stage receipts record the validation commands.
Adapt machine-local paths (Cargo target, Python extension loader and wasm-bindgen)
to the new environment; the archived loader is `reproduction/python-client.py`.
Keep the failed pre-fix layout test as a negative control. Recheck source hashes
and the new product identities; an old archive hash does not establish new run
correctness. `execute.py` records the original orchestration, including its local
reference to the preceding NN harness driver; the driver is also archived here.
