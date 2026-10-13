# Geometry Guard Batching: Unadopted Candidate

Declared before candidate execution or timing. Baseline source is
`386966e747013b7c40c45c83f76746e94793ec02`, with checkpoint evidence at
`edf5e2008e810fca3e498d50bf80e11be714da36`.

## Scope

Encode the existing input validity capture, unchanged geometry passes and
output validity capture in one command encoder. This removes two explicit
queue submissions in each pair-bias, causal-wave and headwise-mixture forward
and backward method. Querywise routing reuses the headwise implementation.
It does not fuse arithmetic, remove guards, change shader reductions, or
promise a particular total submission count for a complete LM step.

The original contextual v1 and corpus v8 executables, WASM modules, fixtures
and reports remain immutable. Do not overlap builds or GPU measurements.
Do not alter the active corpus study's request, served page or executable.

## Correctness Before Timing

- Execute enabled GPU tests for validity, pair metrics, causal wave, shared
  and querywise mixture, then full-model geometry and contextual routing.
- Add a single-encoder regression for validity captures on either side of a
  late producer failure, including immutable earlier aliases and original data.
- Reuse the complete contextual v1 Torch fixture without regenerating it:
  `164062e811489f97779d5b23f17389ff3a7dbd20ac536ae32bf33b5fbf19ee4a`.
  Compare candidate native/browser raw reports bit-exactly with each runtime's
  retained baseline, including outputs, VJPs, guards, updates and resume.
- The original browser router-delta qualification failure remains a failure.
  A scheduling optimization cannot be used to waive that failed criterion.

## Performance Boundary

No speed claim or adoption from fewer submissions or a single timed run.
Before performance measurement, freeze the timing harness, repetitions,
native/browser scopes and promotion rule in a separate protocol revision.
Keep this candidate unadopted until that comparison and independent review.
Raw tensors, checkpoints and executables remain local; publish scalar checks,
hashes and reproduction steps only.
