# Direct Owning VJP Outputs

This record replaces the final checked copy of every graph VJP output with
direct checked producers and one whole-result guard capture. Correctness and
bounded ownership passed native, Python and browser checks. **Neither repeated
run establishes a uniform end-to-end speedup.** Eager Torch MPS was faster in
every native case/cadence comparison in both runs. Slower candidate cases are
preserved, not retried selectively or removed.

## Change and Boundaries

- Candidate: `23542255c9a988d476a4f60a6a9ba416973897d5`, tree
  `37fc6575357e6932876156382d0582c1b0aa381b`.
- Baseline: `5fddb415bfbb0f64366db6d45b7675db17a16f61`, the previous bounded
  checked-copy VJP pool. The intervening evidence-only commit is not recursively
  archived as runtime source. Benchmark and reference helper sources match.
- Dense weight/bias gradients and the first stage's input gradient write into
  owning buffers. Pointwise terminal reductions bind those destinations while
  keeping contribution/partial scratch in the serial workspace. Every parameter
  has exactly one producer under `GraphDefinition`'s ownership validation.
- Producer finite checks and forward/seed failures feed one final guard-only
  dispatch. The extra full-gradient value copy/check dispatch per tensor is gone.
  The kernel math, accumulation policy and dense backward pass grouping are not
  changed. This does not remove all GPU copies, allocations or synchronization.
- At most four whole versions and 32 MiB of output data are retained per
  workspace. One held member/view pins the whole version. Shared ownership checks
  also reject weak owners; busy/oversized versions spill without aliasing or
  waiting. This budget excludes tape, binding resources and externally held
  versions. Explicit history snapshots retain the checked-copy path.
- No implicit change to `pure::Tensor`, generic autograd, `ModuleTrainer`,
  published wheels, optimizer resume, or unsupported graph operations.

## Verification

The source-bound 69-stage regression completed successfully, including CPU/WGPU
tensor and NN suites, Python GPU/CPU surfaces, WASM clients, browser fixtures and
the independent Torch CPU/MPS replay. Backend tests: 152 passed. NN WGPU library
tests: 760 passed, plus the parameter-storage integration test.

The direct-output tests poison the old terminal scratch with NaNs and verify that
correct owning gradients are produced without modifying that scratch. Dense and
pointwise paths cover nonbroadcast, one/two-pass reductions, tail shapes, old
views, pending reads, invalid zero-multiplied consumers, failed-guard recovery,
six held versions, pool spill, workspace drop and simulated zero retention.
The shared native/browser fixture additionally preserves nine VJPs and seven
held versions across reuse/drop. Byte-budget arithmetic and weak ownership have
separate tests; an oversized GPU allocation is not required by those tests.

Torch 2.12.1 replay: **432 matched cases / 197,976 tensor or scalar comparisons**,
maximum absolute error `1.811981201171875e-5`. Four existing tiny-tail reference
gaps remain explicitly separate, not passed matched cases. The tensor-WGPU
suite retains one ignored fractional-GL test. Complete logs and fixtures are
compressed under `raw/`; `summary.json` records counts and product hashes.

## Repeated Measurements

Three graph shapes/depths, three seeds and three optimizer modes: 27 recipes per
run, two complete runs. Each cadence retains eight intervals after two warmups;
each interval performs eight updates. Native lanes rotate baseline/candidate/
Torch, browser lanes alternate baseline/candidate. The second run reverses only
optimizer-mode order. All 4,320 retained intervals remain in the evidence.

Immediate/deferred refers to **update acceptance receipt** observation, not
gradient readback. Parameters and gradients remain resident during timing;
terminal full-state reads, setup/reset and initial zero-rate validation are
outside the measured interval. Every interval is admitted and reset states are
checked. No host-phase instrumentation is mixed into these throughput samples.

Ratios below are **baseline time / candidate time**; greater than 1 favors direct
outputs. Ranges span nine recipe-specific medians, **not confidence intervals**.

| Mode / Cadence | Native A | Native B | Browser A | Browser B |
| --- | --- | --- | --- | --- |
| Plain / immediate | 0.936-1.058 | 0.907-1.095 | 0.975-1.093 | 0.970-1.130 |
| Plain / deferred | 0.948-1.042 | 0.968-1.113 | 0.985-1.098 | 0.992-1.283 |
| EMA / immediate | 0.920-1.063 | 0.944-1.079 | 0.948-1.068 | 1.000-1.216 |
| EMA / deferred | 0.959-1.061 | 0.967-1.171 | 0.978-1.090 | 0.743-1.149 |
| Clip + EMA / immediate | 0.958-1.052 | 0.931-1.048 | 0.902-1.137 | 0.901-1.084 |
| Clip + EMA / deferred | 0.942-1.052 | 0.913-1.117 | 0.960-1.075 | 0.853-1.154 |

The notable slower browser B cases both have shape `[2,16,32]`, depth 2, seed 43,
deferred receipts. EMA's baseline/candidate medians are `11.15 / 15.00 ms`
(retained ranges `10.90-15.10 / 10.20-15.60 ms`); clipped EMA's are
`13.05 / 15.30 ms` (`10.50-15.40 / 11.00-15.70 ms`). These regressions are not
dismissed as noise. The record does not identify their cause, a GPU-copy
bottleneck, a routing threshold, or a universal performance benefit.

Native uses Apple M4 / Metal / IntegratedGpu. Browser reports BrowserWebGpu /
Other without a physical device name; its physical GPU remains **UNKNOWN**.
Owned GPU jobs are serial; host exclusivity is **UNKNOWN**. Torch MPS runs with
fallback disabled, but the eager comparator lacks equivalent finite-stage
guards and rollback. This is not a fastest-Torch, CUDA, large-model or FT-quality
claim. Larger workloads and execution batching need separate measurements.

`manifest.json` binds compressed records to originals or immutable Git objects.
`verification.json` independently rechecks decompression, source/product hashes,
comparison receipts and all retained browser intervals. Binary products are
hash-bound, not committed. No push, merge or release is performed here.
