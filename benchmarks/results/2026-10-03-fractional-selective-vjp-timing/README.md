# Matched Fractional Operator Timing Under Ambient Load

This is a bounded local CPU comparison, **not an idle-host performance
guarantee or a model benchmark**. All three routes compute the same finite
Grunwald-Letnikov map and requested scalar-order VJP. Forward, reverse AD,
allocations and Python/native host transport are included; reference checking
is outside each timed interval. The input does not require gradients.

## Recorded Results

Median of three independent-process medians, milliseconds, lower is better:

| Map / BxTxF | Rust joint backward | Rust selective backward | Torch conv1d reference |
| --- | ---: | ---: | ---: |
| Full / 2x128x768 | 34.074 | 21.139 | 3.194 |
| History / 2x128x768 | 33.911 | 20.996 | 3.414 |
| Full / 1x256x768 | 33.173 | 20.405 | 2.766 |
| History / 1x256x768 | 33.116 | 20.899 | 2.834 |

Skipping the unrequested input adjoint reduces the observed operator time in
every process. The group joint/selective ratios are 1.58-1.63. The optimized
Rust route is still **6.15-7.38 times slower** than this Torch reference by the
same group calculation. These are observed ratios, not a portable speedup or
slowdown promise. Forward still computes and retains the order differential.

All twelve selected processes completed, no failed or omitted conditions.
Each has two warmups per route followed by twelve rounds, cycling all six
route orders twice. The JSON records preserve every measured interval, all
execution orders, correctness errors and hashes. `summary.json` preserves
per-process medians and their ranges, not only a pooled favorable number.

Rust joint/selective outputs and alpha gradients agree exactly on every check.
The Torch reference uses the same polynomial and causal map with float32
interfaces, but different accumulation: Rust uses wider accumulators. The
predefined rtol=atol=3e-5 check passed; maximum output error was 4.77e-7 and
maximum absolute alpha-gradient difference was 3.25e-4 (about 1.09e-5 relative
in that case). At step=1, full and history order derivatives agree because the
omitted zero-lag tap has zero order derivative.

## Conditions And Limits

- Apple M4, ten physical/logical cores, 24 GiB memory; native release build,
  Rust 1.98.0, Python 3.12 and Torch 2.12.1. Native, bridge and benchmark hashes
  are bound in every record and the frozen plan.
- K=32, alpha=0.9, step=1, random seed 239 and CPU float32. Torch is configured
  for two intra-op threads; the Rust convolution loops are serial. This is a
  comparison of these concrete routes, not equal implementation work/threading.
- The primary learning job and this task's builds/tests finished before timing.
  Other OS/application activity remained: pre-run CPU samples ranged from
  56.92% to 76.43% idle. Measurements are **under ambient load**, with no claim
  that the host was isolated, pinned, thermally controlled or noise-free.
- Raw host/process logs remain local. Published host receipts contain only
  numeric CPU samples and log hashes, not application names or local paths.
- There is no GPU/WASM timing, adapter/model throughput, generation quality or
  cross-library overall ranking. The Torch polynomial is a benchmark reference,
  not a second production owner of fractional semantics.

The next profiling targets are the list-based Python/native transport and the
two forward convolutions that generate output and cached order derivative.
This comparison does not yet quantify their individual contributions. Improve
them separately while retaining the Rust mathematical contract and matched
learning updates; do not substitute different geometry to win a speed test.

## Reproduce

Build/install the release native package, verify its source/hash identity, and
run `tools/benchmark_fractional_learning.py`. `plan.json` fixes all conditions
before measurement. For each full/history map and each listed shape, run three
fresh processes, never overwriting an earlier result:

```sh
OMP_NUM_THREADS=2 python tools/benchmark_fractional_learning.py \
  --shape 2 128 768 --kernel-len 32 --alpha 0.9 --step 1 --seed 239 \
  --threads 2 --warmup 2 --rounds 12 --map history --native-profile release \
  --output "$NEW_TIMING_JSON"
```

Prefer an idle host for a confirmation run; retain this ambient-load result
rather than replacing it. `--validate-only` performs correctness checks without
timing. CI validates the mathematical reference and rebuilds published timing
statistics without asserting hardware speed or running performance tests.
