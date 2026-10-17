# Language VJP Isolation and Attention Range Repair

This is a language-first runtime reliability experiment, not a new model-quality
claim. Rust owns the numerical path; native WGPU and browser WebGPU exercise
the same shaders. Python extracts and compares independent local references.
No Key-gradient masking, matmul default change, or tolerance relaxation is used.

## Fixed Operands, Separate References

The [plan](PLAN.md) pins the previous width16/seed11 saved ordinary, learned
Poincare and frozen Poincare models. Four fixed training batches yield 68 linear
projection cases and 12 Attention cases. Every local comparison uses identical
float32 operands and upstream gradients; Torch float64 promotes those same
operands rather than recomputing a different model trajectory.

The linear probe exercises the actual shared graph autograd under all three
existing accumulation policies. Compensated matmul reduces forward/input/weight
L2 error in all 68 cases relative to sequential. That is numerical evidence,
not evidence to change the default policy or claim a speedup. Exact dense-bias
reduction from the preceding change is common to all three policies.

The original Attention VJP is already closer to the local float64 reference
than Torch float32 in these operands. Its nonzero Key-bias residual is similar
to rounding each float64 gradient coordinate to float32 before summing. It must
not be "repaired" by forcing a null-space gradient to zero.

Attention forward, however, had room for improvement: local L2 error decreases
in all 12 cases. Median error against float64 changes from approximately
1.23e-6 to 3.27e-7 (Torch float32: 6.53e-7). This is a local forward comparison,
not a claim of better language generation or full-training equivalence.

## Reproduced Numerical Defects

- A probability such as `exp(-110)` underflows before multiplying a finite
  `1e38` value, losing a representable output near `1.69e-10`.
- Unshifted softmax centering loses the dominant-key derivative for scores
  `[0,-85]`, values `[1,0]`, upstream `2^120`.
- Value-gradient summation loses the residual in `[2^121,1,-2^121]`.
- During repair, prematurely collapsing the shifted value dot introduced a
  regression for values `[2^121,-2^121,1]`: the last score derivative should
  be `2/9`, not zero. The failing intermediate is retained, not discarded.
- Independent review found that backward discarded tiny denominator terms
  which forward retained. A native scratch-level regression verifies the
  actual denominator tail against its shader-generated dyadic weights, without
  subtracting near-one host floats. Both paths now compensate the denominator.

Forward now scans scores at the final maximum, uses a shared extended-exponent
exponential, and divides compensated value sums once at the end. Backward
shifts by a maximum-score key's value and retains both compensation components
through weighted centering. Neither path retains a quadratic probability tape.
Scratch for backward is fourteen words per query; per-lane gradient accumulators
are private rather than workgroup arrays. Selected QK reduction trees, causal
masks, strided views, inherited guards and immutable output ownership remain.

The shared analytic fixtures use relative-only error checks so a returned zero
cannot pass for a tiny nonzero expected value. Native tests also check constant
values at both finite float32 extremes. The browser report retains complete
failed analytic results as well as successful ones.

This extended arithmetic is not IEEE float64 or arbitrary precision. In
particular, the existing `wide_float` normal-output midpoint rounding limitation
remains a separate helper issue. Lower shared-memory use is not a speed claim:
two score scans and compensated private arrays add work and may cause spills.

The denominator witness uses identical `2^127` keys, one scalar query,
upstream `2^120` and scores `[0,-85,...]`. With 8/16/32 tiny weights, the observed
query residual changes from approximately `-214/-856/-3425` to
`-3.81e-5/1.68e-4/1.80e-3`. The denominator-tail regression passes exactly;
the remaining total-gradient cancellation residual is recorded, not declared
unavoidable or forced to zero. The separate midpoint witness returns `1`
instead of `1+2^-23` both before and after this denominator repair.

## Reproduction

Keep private model tensors and full gradients outside the repository. The
following commands assume the pinned input files named in [the plan](PLAN.md):

```sh
python3 -I -B benchmarks/results/2026-10-18-language-vjp-isolation/replay.py extract \
  --repo "$PWD" --request "$REQUEST" --saved "$SAVED_MODEL_REPORT" --output "$FIXTURE"
cargo +1.99.0 build --locked -p st-backend-wgpu --example language_local_vjp
target/debug/examples/language_local_vjp < "$FIXTURE" > "$NATIVE_REPORT"
python3 -I -B benchmarks/results/2026-10-18-language-vjp-isolation/replay.py compare \
  --fixture "$FIXTURE" --native "$NATIVE_REPORT" --output "$COMPARISON"
python3 -I -S benchmarks/results/2026-10-18-language-vjp-isolation/test_replay.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --test resident_attention_vjp -- --test-threads=1 --nocapture
```

`learning.py` repeats the same three 1,024-update studies, pins the original
Torch float32 report, archives the built executable and retains the original
strict comparison. It includes the preceding dense-bias reduction repair, so
it is not an Attention-only causal ablation. No additional seed is selected
after seeing the result. Compiler/profile/source hashes and complete private
optimizer checkpoints bind the run; they are not independent attestation.

## Recorded Outcomes

The final source completed all three 1,024-update runs. Its full report is
byte-identical to the retained pre-denominator-repair intermediate, so the
denominator repair did not change this ordinary-range training trajectory.
The strict Torch float32 gate **still fails**; the failure count changes from
nine components in the previous study to ten, all in the Key slice of QKV bias.

| Case | Failed weight components before | Final | Final maximum weight error |
| --- | ---: | ---: | ---: |
| Ordinary | 2 | 3 | 4.06365e-6 |
| Learned Poincare | 4 | 4 | 4.89521e-6 |
| Frozen Poincare | 3 | 3 | 4.58086e-6 |

Training/development loss errors remain at most 9.536743e-7. This does not
override the failed weight gate or qualify earlier studies. No learning-quality
or Z-space advantage is established here. Both full runs and their original
strict failures are published, not only a preferred outcome.

The final 1,024-update checkpoint was also loaded by a fresh native process at
the completed cursor. Both its re-emitted checkpoint and report are
byte-identical. This validates completed-checkpoint restoration, **not** a new
interrupted-trajectory continuation experiment.

Validation on the final kernel:

- 28 focused native Attention tests with the GPU opt-in enabled; one timestamp
  benchmark intentionally ignored.
- Shared native/browser checks: 36 frozen Torch fixture comparisons, 13 shift
  invariant cases, 46 analytic range cases and 16 resident SGD steps.
- 347 backend and 81 byte-study tests with runtime opt-ins disabled; five Python
  comparison tests. Skipped GPU bodies are not additional GPU evidence.
- Browser backend is `BrowserWebGpu`; physical hardware execution remains
  unverified. Native selected an Apple M4 Metal adapter.

Metrics, both learning comparisons, synthetic review probes, sanitized build
receipts and private-artifact hashes are in [results](results/manifest.json).
The full private input, gradient and checkpoint artifacts are not published.
The [review dispositions](REVIEW.md) retain the separate midpoint-rounding
limitation. External CI remains unexecuted due to the account billing lock;
these local results are not a substitute for that gate.

After the recorded final full run, the only changed file from its source
receipt was `attention/vjp/tests.rs`: a finite, positive-weight assertion was
added to reject a vacuous `0 == 0` denominator test. The native suite was rerun.
No production source changed after that learning run.

```sh
python3 -I -B benchmarks/results/2026-10-18-language-vjp-isolation/learning.py \
  --repo "$PWD" --prior "$SHIFT_INVARIANCE_ROOT" \
  --reference "$CAPACITY_BUDGET_ROOT" --output "$FRESH_PRIVATE_OUTPUT"
```
