# Fractional Learning: Typed Float32 Transport

This removes Python scalar boxing from the existing finite causal GL learning
path, not its mathematical definition. The paired Rust kernel from parent
`987a05bae5d8b47f95ee3c9e56eec3ed72253577` is unchanged. Both timed Rust routes
use the **same native binary, input, upstream direction and selective order VJP**.
The list baseline is explicit, rather than assuming the current public API
still takes the old route.

## Matched Operator Timing

Median of three process medians, milliseconds, lower is better:

| Map / BxTxF | Rust list | Rust buffer | Torch convolution |
| --- | ---: | ---: | ---: |
| Full / 2x128x768 | 16.409 | 2.463 | 3.237 |
| History / 2x128x768 | 17.023 | 2.547 | 3.302 |
| Full / 1x256x768 | 17.009 | 2.642 | 2.789 |
| History / 1x256x768 | 17.161 | 2.623 | 2.891 |

The observed list/buffer ratio is **6.44-6.68**. The buffer route takes
0.76-0.95 times the matched Torch reference time in these four groups.
This is a bounded CPU operator comparison, **not a general PyTorch win,
model-training speedup, quality improvement or GPU result**. Torch computes
the same finite polynomial with different accumulation; this is not a contest
between GL geometry and a different model or learned operation.

All 12 preplanned processes finished successfully. Two shapes, two maps and
three fresh-process repetitions each run all three routes, with two warmups
per route and all six route permutations twice. Forward, autograd and native
host transport are timed together. Correctness checks and SHA256 computation
are outside each interval. List/buffer outputs and order gradients are bitwise
identical; the Torch comparison passes the predeclared rtol=atol=3e-5 checks.
Every interval and every condition is retained, not only the best runs.

Conditions: Apple M4, 10 cores, 24 GiB; Rust 1.98.0 release, Python 3.12,
Torch 2.12.1, CPU float32 interfaces, K=32, alpha=0.9, h=1, seed=239.
Torch uses two intra-op threads; Rust loops remain serial. No own builds,
tests or model probes ran alongside timing. Pre-process CPU idle samples
ranged from 47.74% to 69.62%; other OS/application activity remained.
No isolated host, thermal control, confidence interval or portable performance
guarantee is claimed. The plan retains its original pre-measurement status;
`processes.json` and `summary.json` record completion separately.

## Ownership And Learning Regression

The native methods accept C-contiguous, native-endian float32 exporters,
including read-only and unaligned buffers. They validate type, layout and
budgets, then copy into Rust-owned storage before releasing the GIL.
Outputs are fresh writable bytearrays, never aliases of saved snapshots.
The Torch client retains a memoryview export to prevent resizing the backing
CPU storage. This is **bulk copying, not zero-copy or resident GPU execution**.

No new dependency, unsafe block, mathematical backend, ABI floor or checkpoint
schema is introduced. The optional Torch bridge probes NumPy interoperability
once; missing interoperability retains the existing sequence route. Errors
in real operators never trigger a silent fallback/retry.

- 399 native Python regression tests passed with no skips, including all
  differential variants, noncontiguous views, ownership/lifetime cases,
  buffer rejection and a fresh-process NumPy-free fallback.
- Separate native binaries replayed two identical scheduled updates on copies
  of six completed GPT-2 adapters. All losses, gradients, resulting parameters
  and Adam tensors were bitwise identical across the 24 auxiliary updates.
  Instrumentation confirmed 24 list input/direction conversions in the parent
  process and 24 buffer conversions in the candidate process, with none on
  the other route. Frozen base and original study files were unchanged.
- Default release build, CPU/text-only check and workspace formatting passed.
  Binding clippy passed with the two existing argument-count/type-complexity
  lint categories allowed locally. Unqualified strict binding clippy was not
  green: two new chunk-iteration style findings were fixed, while nine existing
  findings in unrelated binding APIs remain. No project lint policy was relaxed.
- Rust core and WASM sources did not change. The parent's WASM validation is
  prior evidence, not a newly executed browser/WASM test for this Python change.

The pretrained probe is a regression check, not new primary training, independent
seeds, held-out scoring or a quality claim. The receipt's legacy
`paired_native_sha256` field identifies this buffer candidate; its explicit
`comparison` and `transport_calls` fields distinguish it from the earlier
paired-kernel experiment.

## Reproduce And Verify

Build the current binding in release, keep its package isolated, and pin imports
to that package. Use the frozen benchmark beside this README for all conditions:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PACKAGE_ROOT" \
  python -P -B benchmark_fractional_learning.py --compare-transport \
  --shape 2 128 768 --kernel-len 32 --alpha 0.9 --step 1 --seed 239 \
  --threads 2 --warmup 2 --rounds 12 --map history --native-profile release \
  --output "$NEW_TIMING_JSON"
```

Repeat each full/history shape from `timing-plan.json` three times in fresh
processes. Use `--validate-only` first in a separate output file. The buffer
comparison fails rather than measuring a silently selected list fallback.
The declared release profile is not inferred from the binary.

`tools/test_benchmark_fractional_learning.py` reconstructs all process/group
statistics and checks identities, complete condition coverage, receipts and
published hashes without running hardware speed tests. It does not rerun the
private pretrained comparison. The frozen probe/comparator require the
original local completed studies, model and corpus; use a fresh private
directory, not this publication directory. The comparator expects
`pretrained-baseline.{json,pt}` and `pretrained-paired.{json,pt}` beside it.

Only sources, numeric results, verification records and hashes are published.
Weights, corpus text, tensor payloads, native packages and raw host/build logs
remain local. `SHA256SUMS` covers the published artifacts; the package manifest
binds the 70 frozen runtime files used for both timing and candidate validation.
