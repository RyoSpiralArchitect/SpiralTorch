# Native Topos traversal follow-up: mixed, not adopted

This follow-up asks whether the native forward-only observations from the
[rejected WASM-first candidate](../2026-10-08-topos-core-forward-traversal/README.md)
survive the complete shared-NN forward/backward path. It is motivated by that
diagnostic, **not an independent replication**. The earlier WASM screen remains
failed and production source remains restored. No optimization is adopted here.

Baseline source: `71b7a05dff1d18db3d1b20dbf88f8c7389ab5703`.
Candidate source: `dcad72c1949e3d82d898b987f597a48c2ca3c517`.
Both exact native executables are reused from the prior study, with their hashes
rechecked before execution. There are **no fresh builds** in this follow-up.
Original build records and the explicit reuse metadata are both retained.

## Complete matched comparison

Before timing, the new plan fixed all nine conditions, ABBA process order,
forward/reverse/forward/reverse case order, two warmups and 20 measured rounds
per route. Two two-round pilots passed and are retained but excluded. There are
36 native reports and 36 independent Torch reports, totalling 2,880 timings.
Aggregates are the median of two process medians per arm; no case is removed.

The unchanged native NN and single-thread eager Torch harnesses compute the
same finite Picard recurrence, input VJP, summed shared-gate VJP and gradient
accumulation into a preallocated zero buffer. Only Rust includes semantic audits.
Setup, gradient reset, file transport, destruction and comparisons are outside
timing. This is not an optimizer step, compiled Torch, accelerator, full decoder
or model-quality comparison. Hardware: Apple M4, macOS 26.4.1, Rust 1.97.0
release executables; Torch 2.12.1, CPU float32, one thread, isolated Python
without site-startup patches. Offline environment and harness identities are
recorded. No model or corpus was acquired, trained or re-evaluated.

| Rows x features | K | Native before, ms | Native candidate, ms | Change | Torch candidate-phase, ms | Native/Torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 3 | 1 | 0.000719 | 0.001177 | +63.8% | 0.086448 | 0.014 |
| 8 x 3 | 5 | 0.000854 | 0.000833 | -2.5% | 0.216229 | 0.004 |
| 8 x 3 | 16 | 0.001073 | 0.002073 | +93.2% | 0.691771 | 0.003 |
| 64 x 128 | 1 | 0.063740 | 0.082750 | +29.8% | 0.087240 | 0.949 |
| 64 x 128 | 5 | 0.133510 | 0.103688 | -22.3% | 0.461688 | 0.225 |
| 64 x 128 | 16 | 0.248948 | 0.220000 | -11.6% | 1.208719 | 0.182 |
| 256 x 768 | 1 | 1.880489 | 1.411177 | -25.0% | 1.321823 | 1.068 |
| 256 x 768 | 5 | 2.425073 | 2.377281 | -2.0% | 5.660438 | 0.420 |
| 256 x 768 | 16 | 5.163688 | 4.871708 | -5.7% | 18.913927 | 0.258 |

These are **forward plus backward** values, not backward-only values. All
forward-only timings and both phases of the unchanged Torch control are also
retained. Ratios below one mean the native candidate is shorter than this eager
CPU reference in the named scope, not that SpiralTorch is generally faster.
Small sub-microsecond results are particularly sensitive to process behavior.

All six native vector hashes and both audits agree exactly across all arms.
Every Torch output/input-gradient/gate-gradient check passes the unchanged
`rtol=5e-4, atol=3e-5`. Per-field errors and tolerance ratios are retained, as
are hashes and byte lengths of the private full f32 vectors. This is numerical
agreement for this operation, not evidence of better learning or language.

## Screen and process sensitivity

The follow-up's prespecified screen asks for at least 5% shorter complete NN
medians in two large conditions, with no greater than 5% regression in the third,
plus exact native parity and unchanged Torch checks. It **formally passes**.
That criterion only supports investigating a separately scoped candidate; it is
not a significance test or permission to adopt the reverted source.

The apparent 25% K=1 gain is sensitive to baseline process variation:

| Large case | Baseline process medians, ms | Candidate process medians, ms | All process-pair ratio range |
| --- | --- | --- | --- |
| K=1 | 2.328791, 1.432188 | 1.403667, 1.418687 | 0.603 to 0.991 |
| K=5 | 2.420750, 2.429396 | 2.398041, 2.356521 | 0.970 to 0.991 |
| K=16 | 5.127000, 5.200376 | 4.878270, 4.865146 | 0.936 to 0.951 |

`process_sensitivity.json` is explicitly **post-hoc offline analysis**, not new
measurements. It enumerates all four cross-arm process-median ratios for every
condition. These bounds are not confidence intervals and do not replace the
original screen or select the later baseline as the truth. No large condition
is at least 5% shorter in every process pairing, despite the aggregate screen
passing. Together with small/medium regressions, this rules out a blanket speed
claim. A new native-specific source change would need fresh validation and an
independent comparison; the failed WASM result cannot be relabeled as a pass.

## Reproduction record

The compact bundle contains unchanged plan/report bytes, all pilots, summaries,
decision, harness sources, isolated runner source, original build metadata and
reuse records. `verification.json` inventories 201 originals captured before
the later sensitivity analysis and validation logs. Full binaries, vectors and
logs remain local. Previously frozen records are untouched.

```sh
python -I -S -B benchmarks/results/2026-10-08-topos-native-traversal-followup/verify.py
```

This reuses the existing complete NN-matrix validator with an explicitly pinned
plan schema and source revisions, then checks reuse lineage, the screen and
all process-pair bounds. Negative controls reject dropped records, changed
provenance, false decisions and changed sensitivity. It does not rerun timing.
Existing callers retain their original schema requirement; no CI job is added.

For a fresh measurement, use the pinned source/harness and new output names.
Run `topos_shared_module_probe ROWS FEATURES K 0.25 20 NEW_JSON`, then the
recorded isolated `benchmark_topos_shared_module_reference.py` runner against
that receipt and its sibling f32le file. Execute the complete saved ABBA plan,
including pilots, thread settings and both routes. Never overwrite originals.
