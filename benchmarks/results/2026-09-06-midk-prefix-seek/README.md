# Exact MidK Prefix Seek

[admission.json](admission.json) binds all raw artifacts by SHA-256 and separates
the admitted native sequence from noisy browser observations, diagnostics and
excluded older runs. A raw `passed` field alone is not comparison admission.

## Change And Proof Boundary

The shared Rust/WGSL backend no longer serially discards a large prefix of a
fragmented MidK row. For more than 32 sorted tiles and at least 64 discarded
finite candidates, one row workgroup seeks the first retained total-float key
and source index. Each probe sums per-tile lower bounds. The largest key/index
with at most `start` predecessors identifies the candidate at rank `start`;
setting every tile cursor to its lower bound skips exactly that prefix. The
existing merge then emits the retained band. This uses at most
`32 + ceil(log2(cols))` cooperative probes, with scratch reuse separated by
workgroup barriers. No extra buffer, dispatch, planner override or language-side
ranking implementation is added. Short prefixes and the <=32-tile candidate
path are unchanged.

Final native baseline: `82bdd5e262431cd3382e9db93bbbce14933fe1e4`, available on
`spiralreality/midk-seek-canonical-baseline` (the frozen harness commit `40fb3b86`
plus the canonical-control fix, without the kernel change).
Candidate: `817af4651505e2b0faa6f59377f39635bcec4f22`.
The benchmark example, Python runner and its unit tests are byte-identical
between these revisions. Their production difference is the shared WGSL shader;
the remaining differences are regression tests and documentation.

## Admitted Native Comparison

Furnace reports RTX 5090 for both WGPU Vulkan and PyTorch `2.13.0+cu132` CUDA.
Both executables were built and preserved before the final A/B/B/A sequence;
no compilation was launched during it. Four runs of 63 cases all pass exact
native values/indices, exact CUDA indices before/after timing, clean source/build
binding and stable-image gates. Pre/post foreign-compute-PID checks pass, but
they are not an exclusive GPU reservation or continuous activity monitor.

The request hash is
`c5dc53615cd5e831b5ee7f254a3f8e4a7545c2a7ce91036498c9929beac6f9c3`.
Each case has two rows and k=7. Seeds 17/29 use finite fp32 random values; seed
43 is quantized to create ties without mixed signed zeros. Native ordering is
checked against the CPU stable reference. Timed CUDA uses stable sort for MidK
and tied TopK/BottomK, and `topk` only for tie-free controls (35 stable-sort and
28 topk cases per run). The selected operation and exact-index check are recorded
per case. Full stable sort is a comparator, not a claim that it is the fastest
possible canonical CUDA selection algorithm.

Each native interval dispatches sixteen resident rank pairs in one submit and
waits for a completion fence, then divides by sixteen. CUDA enqueues sixteen
calls with preallocated outputs and synchronizes. Two warmups precede twelve
samples; fixed-input correctness is checked before and after all intervals.
No timed upload/full readback, or intervening native map/upload, is included.
Frameworks run in separate blocks. These are host-API/fence timings, not GPU-event,
end-to-end inference, projection-chain or full-model throughput measurements.

Numbers below are microseconds: medians of the three seed medians. CUDA is the
second candidate run; all raw per-seed samples and both baseline repeats remain
available. No confidence interval or all-shape performance guarantee is claimed.

| Kind | Columns | Tile | A1 | B1 | B2 | A2 | CUDA B2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MidK | 257 | 1 | 331.91 | 78.20 | 78.22 | 328.89 | 13.60 |
| MidK | 1024 | 32 | 57.50 | 57.47 | 57.47 | 57.41 | 14.29 |
| MidK | 1025 | 8 | 1091.41 | 78.48 | 78.44 | 1097.52 | 14.59 |
| MidK | 1025 | 32 | 983.89 | 89.88 | 90.61 | 978.85 | 14.60 |
| MidK | 1025 | 256 | 41.29 | 41.09 | 41.13 | 41.26 | 14.59 |
| MidK | 4097 | 128 | 3626.32 | 104.65 | 104.64 | 3637.07 | 54.79 |
| MidK | 8193 | 256 | 7198.23 | 119.76 | 118.59 | 7195.16 | 55.16 |
| TopK | 1025 | 32 | 41.32 | 41.10 | 41.16 | 41.12 | 11.99 |
| BottomK | 1025 | 32 | 41.07 | 41.09 | 41.11 | 41.03 | 11.61 |

The 33-tile cases improve roughly 11x (1025 columns), 35x (4097) and 60x (8193).
TopK/BottomK controls and existing small-tile-count MidK are essentially flat.
CUDA is still about 2.2x faster for MidK/8193 and 6.2x for MidK/1025 with tile=32.
The unchanged five-tile path remains faster than the new 33-tile path at 1025
columns. These data do not justify silently changing SpiralK/Black Cat tile
choices, or claiming WGPU has caught up to PyTorch generally.

## Browser Evidence Is Noisy

Four real Chrome `152.0.7977.77` runs use the same fixture in A/B/B/A asset order.
All pass 71 cases / 1,524 assertions: 63 timed cases and eight validation-only
cases. They include ties, signed zeros, finite extremes, non-finite padding,
start=63/64, repeated snapshots and workspace destruction. The browser input
mix includes a mostly non-finite row, unlike the native all-finite workload.
The fixture hash is
`fc29259f637246ecf73997d17857d0452bcc7cfe5c0bfaa98e1d7a790b41a884`.
Baseline assets were rebuilt from PR #2075 code (`1fd71570`); the WASM hash is
`69572264c5c8d7c8649195a9896a7d1a6f124e8034c8ac1cd76166dd15e854a9`.
Candidate assets have hash
`01da8c20e33c6a1f1583b20f58114c5fdb3058b9d67686db009a93791655949c`.
Browser identity is `BrowserWebGpu` with no physical adapter name. These asset
hashes are not native build/source attestation.

Browser numbers are median microseconds across ten intervals, each containing
eight dispatch repetitions plus a four-byte asynchronous completion fence.
The coarse clock, event loop and shared desktop introduce substantial variation.

| Kind | Columns | Tile | A1 | B1 | B2 | A2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MidK | 257 | 1 | 1668.75 | 1037.50 | 600.00 | 506.25 |
| MidK | 1024 | 32 | 156.25 | 875.00 | 143.75 | 156.25 |
| MidK | 1025 | 8 | 2462.50 | 875.00 | 1068.75 | 1493.75 |
| MidK | 1025 | 32 | 2281.25 | 881.25 | 675.00 | 1793.75 |
| MidK | 1025 | 256 | 225.00 | 1056.25 | 856.25 | 118.75 |
| MidK | 4097 | 128 | 6931.25 | 981.25 | 987.50 | 4762.50 |
| MidK | 8193 | 256 | 19375.00 | 1037.50 | 1418.75 | 8718.75 |
| TopK | 1025 | 32 | 100.00 | 1137.50 | 943.75 | 1000.00 |
| BottomK | 1025 | 32 | 500.00 | 125.00 | 306.25 | 62.50 |

Large fragmented cases shorten, but unchanged controls also vary dramatically,
and the tiny fragmented case is not consistently faster than both baselines.
Retain these negative/mixed outcomes: no universal browser speedup or reliable
browser gain factor is established. The native gains must not be projected onto
WASM end-to-end workloads. The composed browser fixture separately passes
27 cases / 4,887 assertions with the same candidate assets.

## Validation And Superseded Evidence

- Apple M4 and Furnace: 70 backend unit tests plus shader validation pass with
  runtime tests enabled. The new test crosses five fragmented geometries, five
  k values, and four input patterns (100 combinations, three rows each). It does
  not exhaust all fp32 inputs or performance shapes.
- Rebuilt private Python wheel: 61 resident matmul/rank API tests pass. Public
  package version/release is unchanged. Generated TypeScript declarations,
  native/wasm strict clippy, pinned fmt and eight harness unit tests pass.
- Earlier weaker-control A/B/B/A reports and one corrected but build-overlapped
  baseline are retained under [superseded](superseded/README.md), excluded from
  the final comparison rather than relabelled or pooled with it.
- The original 54-case rotated suite also passes with corrected CUDA index
  checks. It is a compatibility smoke run, not pooled into the 63-case comparison.
- Additional Furnace parallel-completion diagnostics did not finish, including
  on the unchanged kernel. Serial runs pass on both revisions. The
  [diagnostic record](diagnostics/README.md) preserves the failures and unresolved
  concurrency risk; this is not an all-parallel-runtime correctness guarantee.

See the [resident guide](../../../docs/performance/resident_rank.md) for the
reproduction command, API boundaries and algorithm. This improves the shared
backend, not model quality or training convergence.
