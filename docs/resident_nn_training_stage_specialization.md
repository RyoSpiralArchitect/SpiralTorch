# Resident Training Stage Specialization

The resident training plan already knows whether a matrix dispatch is plain
forward, GELU forward, a weight gradient, or an input gradient. Rust now turns
those attributes into shader constants instead of re-reading uniform flags
inside each output operation. This joins high-level plan information to the
existing low-level kernel; Python and JavaScript do not implement another policy.

## Implementation Boundary

`TrainingMatmulKind` owns the fusion/transpose flags and validation mask.
`MatrixPipeline` keeps that kind with the compiled pipeline so dispatch uniforms
cannot independently select a conflicting flag set. Training's fixed output
scale is also specialized. The original public dynamic-flags shader generator
and canonical WGSL templates remain unchanged, including the generic host-tensor
path. This is not the earlier rejected cooperative-transpose-loading candidate.

Forward pipelines are created lazily for the activation kinds present in the
plan, at most two, and reused across matching layers. Shapes and stage indices
remain uniform data. Dispatch count/order, native Metal pass coalescing, browser
pass cadence, loss readbacks, accumulation policies, finite guards and the
transactional all-layer SGD decision are unchanged. No new binding API is needed.

## Measured Results

The [source-bound experiment record](../benchmarks/results/2026-09-09-resident-training-stage-specialization/README.md)
compares baseline `00a955a7` against candidate `e15f463d`. The baseline production
code matches main `f9f5784d`; its extra opt-in diagnostic test is not executed by
the benchmark. Both clients ran the same complete nine-recipe matrix twice,
declared before timing: three seeds, shapes `[2,16,32]`, `[4,16,64]`, `[4,32,128]`,
depths 2/8/16, two warmups and eight retained intervals per cadence. Each interval
resets the same weights and executes eight real nonzero-rate SGD steps.

Native order rotates baseline/candidate/eager PyTorch MPS. Browser order
alternates baseline/candidate and is numerically checked against captured native
PyTorch states. Only owned GPU jobs were serialized, without compilation during
timing. macOS contention/power state was uncontrolled. Only Apple M4 Metal and
Chrome WebGPU were measured, not CUDA, Furnace or other native GPUs.

Each range below spans three seeds' ratios of median elapsed time, old/candidate.
Above 1 favors the candidate. These are diagnostic observations, not confidence
intervals, precise routing thresholds or universal speed guarantees.

| Run | Depth | Native Immediate | Native Deferred | Browser Immediate | Browser Deferred |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 2 | 0.98-2.31 | 1.03-1.05 | 0.93-1.71 | 1.01-1.14 |
| First | 8 | 1.00-1.29 | 1.00-1.45 | 1.04-1.07 | 1.00-1.03 |
| First | 16 | 0.96-1.07 | 1.05-1.08 | 1.04-1.09 | 0.76-1.27 |
| Repeat | 2 | 0.98-1.02 | 0.36-2.47 | 0.95-1.09 | 1.00-1.08 |
| Repeat | 8 | 1.36-2.00 | 1.10-1.80 | 0.99-1.00 | 1.01-1.02 |
| Repeat | 16 | 1.07-1.10 | 1.07-1.11 | 1.00-1.04 | 1.02-1.06 |

The most consistent result is native depth-16 deferred throughput, improving
in all six seed/run pairs. Small native depth-2 deferred intervals remain very
noisy, including regressions; do not sell the largest outlier as a speedup.
Browser depth-16 deferred also has a first-run regression, not repeated in the
second matrix. The largest native workload **still loses to eager PyTorch MPS**.
Rust's per-stage finite/transactional checks and individual deferred snapshots
differ from the PyTorch guard/readback implementation, as in the original bench.

## Preparation And Correctness

Setup is outside the timed SGD interval and includes pipeline creation, uploads
and the initial zero-rate VJP. The first captured native candidate setup was
381 ms versus 12 ms for baseline; in the repeat the corresponding captures were
9.3/9.8 ms. This is not an isolated cold-compiler comparison. Native setup was
retained only for the first captured interval per cadence, so do not invent a
steady-state native setup distribution. The browser records every interval's
setup, with mixed results retained in `summary.json`. Compile once and reuse the
workspace for long-running training; one-shot/cold-start latency is not a win
established by this change. The pipeline cache is per workspace, not cross-device.

Both complete matrices pass full captured-state comparison against independent
PyTorch references (maximum absolute error about `1.49e-8`). All 18 native
recipe/run pairs have matching baseline/candidate state fingerprints. Native
and browser each also pass 18 VJP fixtures, three 128-step fits, non-finite
rejection, GELU saturation and transactional rollback/recovery; separate CPU/MPS
PyTorch replay passes. Fingerprints are consistency receipts, not independent
physical-device attestation. Browser adapter metadata comes from a separate probe.

Local final checks pass: 106 backend tests, 740 selected NN tests, native/wasm32
strict backend Clippy, whole-workspace formatting, benchmark/client validators,
and production Python/WASM binding compilation checks. No new wheel is published
by this source change. General N-D autograd, legacy `pure::Tensor` residency and
automatic lowering of arbitrary `ModuleTrainer` graphs remain separate work.
