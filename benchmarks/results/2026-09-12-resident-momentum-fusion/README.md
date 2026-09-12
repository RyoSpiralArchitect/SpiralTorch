# Resident Momentum Preparation Fusion

Effective-gradient normalization, optional clip scaling, Topos EMA and weight
candidate preparation now share one GPU kernel per parameter. This removes an
intermediate GPU-buffer write/read and one dispatch per parameter, while keeping
global norm reduction, validation and whole-state commit separate. It removes
no CPU readback: the previous learner was already GPU-resident.

For seven parameters, the update phase excluding VJP composition changes from
22 to 15 dispatches, or from 30 to 23 with clipping. Bindings and history storage
are reused. Python and WASM consume the same Rust implementation with unchanged
public APIs. This structural reduction is verified; universal speedup is not.

## Sources And Correctness

- Benchmark baseline: `4581ec99ae9968d69a18e166c4cec6b1b106a9b6`, tree
  `d793dcfff18171bfd38a16c68f53a2e78a2b7003`.
- Fused runtime: `25873daf5c7647ebb6dcc262ee01f5da3e272ed2`, tree
  `7f66f6eec59d97553cfa3f8072df56eabe3628d6`.
- All 69 source-bound regression stages passed in 634.00 seconds. Backend 148,
  WGPU-enabled NN 760 plus one doc test, integration 6, Python, browser WebGPU,
  generated/shipped TypeScript and CPU-only checks passed. One existing
  tensor-WGPU fractional-GL test remained ignored, not counted as a pass.
- Independent PyTorch 2.12.1 CPU/MPS regression replay passed 432 cases /
  197,976 tensor or scalar comparisons. The 24 EMA cases accounted for 38,472
  comparisons, maximum absolute error `9.5367431640625e-7`. The full set's maximum
  was `1.811981201171875e-5`. Four pre-existing tiny-tail reference gaps remain
  separate, not counted as matches.
- The new native test covers large finite gradients, tiny normal-factor clip
  scales, both gain policies and both enable orders. Existing rejection,
  zero-rate preservation, reset/re-enable and owning-history tests still pass.
- The published Python example ran unchanged against the frozen native binding:
  32 updates, weight-only Module handoff with output difference 0, and retained
  history snapshots readable after reset, disable and learner destruction.
- Benchmark admission has 17 dependency-light tests. Both source versions use
  byte-identical benchmark/oracle/validator files; their hashes are retained.
- Independent integrity verification checked 276 archived source/log/data
  records and 51 frozen products, retaining 1,330,608,098 raw bytes in
  76,569,996 compressed bytes. This verifies the archive, not a second GPU run.

## Paired Measurements

The protocol is in [the guide](../../../docs/resident_momentum_fusion.md).
Each run includes all three modes: plain SGD control, EMA with damping `0.5`,
and the same EMA with global clip limit `1/1024`. Three seeds and three mixed
graphs give 27 distinct mode/shape/seed conditions. Each cadence has two
discarded warmups and eight retained blocks of eight updates, with rotated
native baseline/candidate/Torch and alternating browser baseline/candidate.
The timing matrix uses the Exact update policy; ModuleCompatible numerical
coverage belongs to the separate regression suite.

Run A was followed by a complete repeat after mixed small-case timings. Run B
reverses only the order of the three modes; all recipes and within-case rotations
remain unchanged. No selected cell was retried or discarded. Both runs and
their raw captures/timings remain separate. The two runs total 4,320 retained
timed intervals across all lanes; they are short reset intervals, not one long
continuous training run. Browser revalidation checked 2,160 interval receipts
including warmups. Terminal weights, both VJPs, predictions, loss and history
matched the independent Torch captures, maximum absolute error
`3.259629011154175e-8` across the benchmark comparisons.

The table gives ranges across the nine recipe-specific ratios of eight-sample
medians. Values above 1 favor the fused candidate. These are not confidence
intervals, pooled speedups or universal backend guarantees.

| Mode / Cadence | Native A | Native B | Browser A | Browser B |
| --- | ---: | ---: | ---: | ---: |
| Plain / immediate | 0.956-1.045 | 1.000-1.021 | 0.951-1.018 | 0.983-1.012 |
| Plain / deferred | 0.983-1.059 | 0.995-1.022 | 0.958-1.000 | 0.944-1.040 |
| EMA / immediate | 0.948-1.079 | 0.949-1.081 | 1.000-1.064 | 0.922-1.117 |
| EMA / deferred | 1.004-1.269 | 0.994-1.038 | 0.960-1.021 | 0.940-1.027 |
| Clip + EMA / immediate | 0.843-1.053 | 0.913-1.037 | 0.712-1.077 | 0.984-1.021 |
| Clip + EMA / deferred | 1.002-1.058 | 1.007-1.176 | 0.960-1.096 | 0.973-1.028 |

## Interpretation And Limits

Dispatch reduction does not translate into a uniform end-to-end win. Many
deferred results improve by a few percent, but small-case scatter overlaps the
unchanged SGD control. Some immediate observations are slower. The first run's
large EMA gain did not repeat at that condition; a different clipped case had
a large ratio in the repeat. Neither maximum is a representative speedup.
In the first browser run, seed 43 / depth 2 / clipped EMA immediate had a
candidate median of 12.5 ms versus baseline 8.9 ms; this slower result is retained.
The measurements do not isolate a cause for these variations.

Eager Torch MPS remained faster in every measured native comparison: its elapsed
time divided by the candidate's was 0.469-0.991. Rust performs stage guards and
transactional commits and reads owning acceptance receipts; Torch only
synchronizes completion. These costs differ even though the numerical task is
matched. There is no claim about fastest PyTorch or browser-versus-MPS timing.

The fused path is retained for its simpler update dataflow and eliminated
intermediate traffic, with the measured timing tradeoffs explicit. No automatic
shape/backend speed threshold is inferred. Larger throughput gains require
attribution of the full forward/VJP/capture/update path, not more claims from
the dispatch count alone. Native adapter: Apple M4 / Metal. Browser backend:
BrowserWebGpu, physical GPU UNKNOWN. Owned GPU work was serial; host exclusivity
UNKNOWN. MPS fallback was disabled. CUDA/Furnace, FT quality, generalization,
full optimizer checkpoint/resume, release, push and merge are outside this record.

See [full results](summary.json), [archive manifest](manifest.json) and
[independent archive integrity verification](verification.json). Source, raw
captures, stderr, progress, validators and preliminary checks are compressed
without dropping numeric data. Binaries stay outside Git with hashes recorded.
The evidence/documentation commit adds no runtime changes beyond the frozen
fused source above.
