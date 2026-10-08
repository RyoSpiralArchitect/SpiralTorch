# Native Shared-Gate NN: Matched CPU Comparison

This study measures the actual `st-nn::ToposResonator` shared-gate module,
not just its core helper. The candidate keeps F gate values and directly
accumulates the compact F-value gate VJP, while preserving the prior audit
of the logical per-element gradients. The baseline expands those values
and reduces an N-value gate VJP through a separate tensor operation.

## Scope And Results

Apple M4, macOS 26.4.1, Rust 1.97.0 release, CPU Torch 2.12.1 eager with
one thread. Both implementations compute the same finite Picard recurrence
and both VJPs, including gate-gradient accumulation into a preallocated
buffer. Only Rust includes its semantic audits. Setup, reset, validation,
file transport and optimizer updates are outside the timed interval.
The forward-only route still records a training tape; it is not inference.

There are 9 conditions, 4 ABBA processes per condition, and 20 alternating
samples per route per process. Every native process is followed by a fresh
Torch process: 36 native and 36 Torch reports, **2,880 timed samples** total.
The cases run forward/reverse/forward/reverse across the four phases.
The plan was frozen before measurement. The aggregation rule below was
chosen after measurement but before inspecting the results, not preregistered
in the plan: take each process's median, then the median of the two process
medians per arm. No cases or individual timings were discarded.

All values below are milliseconds. `F+B` means forward, both VJPs and gate
accumulation, not an entire optimizer/trainer step. Torch is from the two
candidate-phase processes; the bundle also retains both baseline-phase runs.

| Rows x Features | Iterations | Native F Before | Native F After | Native F+B Before | Native F+B After | Torch F+B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 3 | 1 | 0.000739 | 0.000438 | 0.001521 | 0.000782 | 0.058906 |
| 8 x 3 | 5 | 0.000636 | 0.000583 | 0.001167 | 0.000906 | 0.238052 |
| 8 x 3 | 16 | 0.000938 | 0.000875 | 0.001417 | 0.001209 | 0.725510 |
| 64 x 128 | 1 | 0.054208 | 0.048865 | 0.083917 | 0.069458 | 0.091365 |
| 64 x 128 | 5 | 0.098396 | 0.094458 | 0.127968 | 0.114010 | 0.384052 |
| 64 x 128 | 16 | 0.218521 | 0.213719 | 0.247375 | 0.234636 | 1.189646 |
| 256 x 768 | 1 | 1.317490 | 1.143979 | 2.011406 | 1.631958 | 1.396490 |
| 256 x 768 | 5 | 2.329667 | 2.146698 | 3.029667 | 2.617042 | 6.172969 |
| 256 x 768 | 16 | 5.053406 | 4.984135 | 5.781219 | 5.475896 | 22.583458 |

The large native F+B cases are about 18.9%, 13.6% and 5.3% shorter in this
study. **Torch remains faster at 256 x 768 / 1 iteration.** Tiny cases are
dominated by overhead and timing noise: the two old 8 x 3 / 1 iteration F+B
medians were 0.002083 and 0.000958 ms, versus 0.000771 and 0.000792 ms after.
The apparent 49% tiny-case reduction is not a robust general speed claim.
There was no simultaneous build/training, but normal desktop noise was not
controlled. Two processes per arm are not a confidence-interval study.
No `torch.compile`, accelerator, peak-memory, end-to-end FT or model-quality
claim follows from these numbers.

For every condition, all six native vector hashes and both audits were
bit-identical across the four runs. Every Torch call was checked against
native output and both gradients with rtol `5e-4`, atol `3e-5`. The largest
gate-gradient error was `2.6226043701171875e-6`; the largest normalized error
was below `0.024` (the acceptance boundary is `1`).

## Sources And Records

- Baseline: `1ec25672cd2658d091aba31f26891eaa8dd1a8e3`.
- Candidate: `af7ef9aaa9adc466052611e8bdc950dbbf51dcfd`.
- Identical harness on both source revisions: committed afterward as
  `fe3e1fb624d957fd8a0e5497cb5e4f5b4ee76772`. At measurement time the new probe
  was an untracked example overlay, with no production source overlay.

`measurements.json.gz` retains the exact plan text, both build identities,
all unmodified native/Torch JSON reports, per-process medians, the full
summary and two excluded pilot reports. It also freezes the four harness
source files, so later edits cannot change the measured implementation.
`verification.json` records runtime/configuration and hashes/sizes of the
original local files, including binaries, vectors and logs. `SHA256SUMS`
binds the public files. Full float32 vector binaries and logs remain local;
they are not included here. The standalone checker reconstructs all summary
cells and rejects identity/parity/tolerance inconsistencies. This verifies
the saved record, not a new independent execution or proof of execution.
Independent read-only review of the four harness files found no actionable
P1/P2 issues and ran 15 isolated tests; it did not rerun the full matrix.

The separate [200-update learning/client record](../2026-10-07-topos-nn-shared-capture/README.md)
covers native SGD trajectories, route switching, Python and WASM correctness.
It is not pooled into these timings.

## Reproduce

Use fresh output paths; no models, datasets or network APIs are required.
For each pinned source revision, overlay the identical frozen example from
`fe3e1fb6` (and its two Python reference files) without changing production
files. Build the example below, then preserve the binary separately before
building the other revision. The hashes identify this machine's artifacts;
different toolchains need not produce identical binary bytes.

```bash
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  cargo build --locked --offline --release -p st-nn --example topos_shared_module_probe
```

Follow the frozen plan's ABBA/case order, with a fresh native process then
Torch process for each condition. Replace the example shape/iteration and
binary path as specified by the plan. Both scripts refuse to replace output.

```bash
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /path/to/preserved-probe 256 768 5 0.25 20 /tmp/topos-nn-new.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python tools/benchmark_topos_shared_module_reference.py /tmp/topos-nn-new.json /tmp/topos-nn-new-torch.json
python -I -S -B tools/test_topos_shared_nn_benchmark_results.py
```
