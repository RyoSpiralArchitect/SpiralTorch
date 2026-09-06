# Active-Lane Rank Reductions

The production change is a small shared-WGSL helper and two reduction
initializers. The first reduction stride now reflects the number of contributing
tile lanes, rather than always starting at 128. The 256-lane workgroup, planner
tile choice, two shader dispatches, buffers, ordering and ownership are unchanged.
No global lock, CPU fallback, dependency override or Python ranking policy is
added. The separate [native lifetime risk](../2026-09-06-wgpu-lifecycle/README.md)
is not fixed or covered by these single-runtime comparisons.

## Algorithm And Frozen Controls

Lane `i` owns tiles `i + n * 256`. When there are fewer than 256 tiles,
higher unused lanes contain only zero counts or invalid candidates. The largest
power of two strictly below the tile count, capped at 128, is therefore the
first necessary stride. Skipped upper levels only combine an existing value
with the identity; all surviving comparisons, sums and barriers remain intact.
One tile needs no pairwise reduction. At 129 or more tiles, all eight levels
remain. This applies to k-way TopK/BottomK selection and the fragmented MidK
prefix-count probes. The <=32-tile parallel MidK path is unchanged.

Baseline: `c4532d3e30b39b1fbb28cdde9b535fac797101f6`.
Candidate: `189a2f0e35d7347bc1763fc4a38d84db3ba39352`.
Their only source difference is the shared shader. The Rust example, native
request generator, browser runner/fixture and new GPU regression test are
byte-identical. An earlier helper return form failed Naga validation; that
failed log is retained, and that revision was not used for GPU comparisons.

The native `active-lanes` suite covers 1/2/3/5/17/33/65/129/257 tiles of
32 columns, a last partial tile, k=1/7/min(65,cols), all three rank kinds,
two rows and seeds 17/29/43: 243 cases per process. Seed 43 is quantized for
ties without mixed signed zeros. Additional native regression coverage spans
23 tile counts across every reduction boundary, five row patterns and three
k values. It includes signed zero, finite extremes, all-non-finite rows, and
a sole finite candidate in the last partial tile.

## Furnace Versus PyTorch CUDA

All four sequential A/B/B/A runs pass the exact value/index oracle before and
after timing, source/build binding, execution-image stability, and foreign
compute-PID pre/post gates: 972 cases in total. Both backends report RTX 5090;
WGPU uses Vulkan/NVIDIA 595.84 and PyTorch is `2.13.0+cu132`.
The common request SHA-256 is
`4202077b2056fa9cad6bf71f08bdc2d4a487c4d5f5c9f02b9902cefd99326009`.
Both executables were built/copied before timing; no build ran during the
native sequence. The existing CPU llama-server was left running. This is not
an exclusive machine/GPU reservation or continuous activity monitoring.

Each native sample submits sixteen resident rank pairs and a completion fence,
then divides by sixteen. CUDA enqueues sixteen preallocated calls and
synchronizes. There are two warmups and twelve samples, with no upload/full
readback in these intervals. CUDA uses stable sort for MidK and tied rows,
and topk only for tie-free TopK/BottomK; exact CUDA indices are checked on both
sides of timing. Full stable sort is a comparator, not a claim of optimal
canonical CUDA selection. Frameworks run in separate blocks.

Representative numbers below are microseconds, medians of three seed medians.
CUDA is from B2. The complete, unfiltered tables and raw samples remain in
[summary.json](summary.json) and the eight compressed reports.

| Kind | Tiles | Columns | k | A1 | B1 | B2 | A2 | CUDA B2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TopK | 1 | 31 | 7 | 34.30 | 28.30 | 28.20 | 34.44 | 6.13 |
| TopK | 3 | 95 | 65 | 118.59 | 78.50 | 83.59 | 119.67 | 8.25 |
| BottomK | 3 | 95 | 65 | 120.21 | 78.53 | 78.46 | 119.27 | 8.22 |
| TopK | 5 | 159 | 65 | 120.38 | 93.43 | 93.91 | 120.27 | 8.25 |
| TopK | 17 | 543 | 65 | 137.76 | 120.26 | 120.32 | 138.42 | 9.40 |
| MidK | 33 | 1055 | 7 | 90.26 | 78.69 | 78.35 | 90.73 | 14.61 |
| TopK | 129 | 4127 | 7 | 41.16 | 41.16 | 41.09 | 41.33 | 19.17 |
| TopK | 257 | 8223 | 65 | 185.32 | 185.00 | 186.76 | 184.46 | 28.88 |
| TopK | 5 | 159 | 1 | 22.93 | 22.90 | 28.17 | 22.87 | 4.42 |

The three-tile k=65 cases shorten by roughly 30-35%; several other shapes are
flat. The last k=1 control is slower in B2 but not B1; it is retained, not
excluded as inconvenient noise. Some unchanged MidK controls also drift.
**PyTorch is still faster.** No all-shape non-regression, confidence interval,
GPU-event, training-convergence or full-model speedup is established.

## Browser And Python

Four isolated Chrome `152.0.7977.77` A/B/B/A processes each pass 89 cases and
1,920 assertions, with no page errors. The 81 timed cases use mixed finite/
non-finite rows, unlike the native all-finite inputs; eight other cases validate
wide output, padding and snapshot ownership. Browser identity is
`BrowserWebGpu`, not a physical adapter name. Each interval contains eight
resident repetitions plus a four-byte asynchronous completion fence.

| Browser Case | A1 us | B1 us | B2 us | A2 us |
| --- | ---: | ---: | ---: | ---: |
| TopK, 3 tiles, k=65 | 187.50 | 125.00 | 125.00 | 150.00 |
| BottomK, 3 tiles, k=65 | 150.00 | 125.00 | 125.00 | 150.00 |
| TopK, 5 tiles, k=65 | 175.00 | 137.50 | 137.50 | 162.50 |
| MidK, 33 tiles, k=7 | 175.00 | 137.50 | 137.50 | 137.50 |
| MidK, 65 tiles, k=65 | 306.25 | 331.25 | 300.00 | 300.00 |
| MidK, 257 tiles, k=65 | 387.50 | 400.00 | 406.25 | 431.25 |

Small fan-in cases shorten, but coarse clocks/event-loop variation and mixed
controls prevent a universal browser gain factor. Native speedups must not be
projected onto these browser workloads.

Metal and Furnace each pass 71 backend tests with runtime tests enabled.
A rebuilt private Python wheel passes 61 resident rank/matmul API tests; its
loaded native extension hash matches the wheel payload. Native and wasm32
strict backend clippy, pinned rustfmt, nine benchmark harness tests and six
negative evidence checks pass. No public package version/release changed.

## Reproduction

Use clean builds of the frozen revisions; keep their executables distinct.
Do not start another native GPU process until the preceding handle is terminal.

```bash
cargo build --release --locked -p st-core --no-default-features --features wgpu-rt --example resident_rank_bench
python3 tools/bench_resident_rank_vs_torch.py --executable /path/to/frozen/resident_rank_bench --suite active-lanes --resident-only --output /path/to/new-report.json
```

The existing isolated browser runner accepts `rank-active-lanes` as its final
fixture argument. Use WebGPU-enabled WASM assets and preserve the emitted asset
and page hashes. To recheck the archived request, native fp32 values/indices,
boundaries and unfiltered tables without a GPU:

```bash
python3 -I -S benchmarks/results/2026-09-06-rank-active-lanes/summarize.py
```

[validation.json](validation.json) binds build/test logs, source files, executable
and private-wheel identities. Browser asset hashing is not native source/build
attestation. Failed initial shader validation remains separate from admitted
GPU results. The Rust-owned planner and Black Cat candidate identities retain
their existing meaning; no new adaptive-policy win is claimed by this probe.
