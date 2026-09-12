# Terminal Guard In The Graph Compute Pass

Runtime source: `6703df35e8389c3ebb9ddf5eb099f440995a930d`.
Baseline: `04fec016ad1ee8871d2a786974e94caeb556b84b`, the preceding verified
content-stamp runtime. This is a local, source-bound candidate, not a release
or a claim that every browser route is faster. No CUDA/Furnace or push is included.

## Change And Contract

Direct resident Module/graph forwards previously opened one WGPU compute pass
for NN stages and another for the terminal output-guard capture. The capture is
now the final dispatch in the graph pass. There are still the same shaders,
dispatches in the same order, one queue submission, the shared validation clear,
and the upstream-input flag copy. No finite check or error flag is removed.

This is pass grouping, not operator fusion. View packing may use its own pass.
Explicit dispatch/snapshot/output-capture APIs retain their separate behavior.
Original Rust Modules, Python and WASM all inherit the shared backend change;
there is no client-side policy mirror or silent Tensor/ModuleTrainer migration.
The untargeted explicit graph dispatch route and eager Torch remain controls.

New tests put an overflow only in the final element of a `[3,7,65]` input:
1,365 elements spanning multiple workgroups. They cover dense/pointwise producers,
late stage indices, immediate ReLU masking, queued snapshots, retained outputs,
16 subsequent valid forwards and recovery after switching API routes. The
terminal guard must observe all earlier workgroups, not just an early producer.
Stage indices and old output errors must survive later valid execution.

## Verification

All 50 full verification steps and 12 predeclared replication steps passed.
Frozen binaries/WASM and the source tree were hash-checked before and after runs.

- WGPU backend: 131 tests; NN WGPU-enabled: 758 unit tests plus one integration
  regression; NN CPU: 706 plus the same regression. GPU work was serial with
  the real-runtime opt-in enabled, not a claim that every unit test uses the GPU.
- Tensor CPU: 449; Tensor WGPU-enabled: 497 plus one unchanged ignored
  fractional-backend test. Contracts: 23; resident integrations: six.
- Python: 59 tests, including nine original-Module tests; six selected CPU-only
  tests plus the CPU capability and parameter-handoff probes also passed.
- Browser original Module: 132 assertions, including `terminal_guard_ordering`.
  Explicit graph: 536 assertions. Pointwise, learner, autograd, fusion, bidirectional
  handoff, declarations and CPU-only WASM compilation passed.
- Torch CPU/MPS training/VJP replay: 120 cases, 17,136 comparisons, maximum
  absolute error `5.7220458984375e-6`. Browser original-Module replay: 936
  comparisons, maximum `1.1920928955078125e-7`.
- Existing descriptor allocation checks passed on CPU and WGPU-enabled builds:
  four tests each, with the prior 69-to-one flat assembly result retained.

## Matched Timing

Unchanged fixture: seeds 17/29/43, three shapes, Scaler/Linear/GELU/ReLU blocks,
three warmups and nine retained rotated blocks per case. Burst means eight
independent forwards of the same resident input and one final completed host
read; D2H reads every forward. Neither is recurrent decoding. Upload and cold
compilation are outside these two timed routes.

Four complete matrices remain: initial baseline/candidate, then native orders
candidate/baseline, baseline/candidate, candidate/baseline. Browser versions run
in one page with rotated routes and separate frozen WASM/device instances.
Native Metal/Apple M4 is recorded; physical browser GPU and exclusive host access
are UNKNOWN. No trial was retried, filtered, or selected adaptively.

Median of 12 paired seed-run ratios per shape, candidate/baseline; lower is faster:

| Shape; blocks | Python burst | Python D2H | Browser burst | Browser D2H |
| --- | ---: | ---: | ---: | ---: |
| `[2,3,7]`; 2 | 0.799 | 0.913 | 1.000 | 1.250 |
| `[2,8,64]`; 8 | 0.950 | 0.977 | 1.000 | 1.000 |
| `[4,8,128]`; 16 | 0.985 | 0.989 | 1.000 | 1.000 |

Native middle/large bursts improve in all 12 pairs each (0.929-0.974 and
0.976-0.997). Small native bursts span 0.683-1.228, including two regressions.
The small explicit-graph control spans 0.943-1.017, while its Torch control spans
0.849-1.473. No control normalization is used to erase drift or outliers.
Across all sizes, 2/36 native burst and 12/36 browser burst ratios are strictly
above one; D2H has 7/36 native and 12/36 browser such ratios. Some browser ratios
are only clock-rounding differences, but none are silently removed.

Native burst/eager Torch MPS median ratios are 0.574/0.511/0.750. These describe
this eager reference and fixture, not fastest/compiled Torch, peak GPU throughput,
generic training superiority, or a causal breakdown of GPU phase time.

### Browser Single-Call Uncertainty

The small browser's primary paired-median ratio regresses to 1.250, despite the
initial matrix suggesting improvement. This remains in the primary table.
[browser_clock_diagnostic.json](browser_clock_diagnostic.json) separately examines
all 108 untrimmed samples per version from the same four matrices. Total elapsed
candidate/baseline ratios are 0.819, 0.963, 0.986 and 0.986 by matrix; pooled total
elapsed is 0.935. Most readings occupy coarse 0.2/0.3 ms bins.

This is a post-hoc diagnostic, not a replacement endpoint. Paired-median ratios,
pooled medians and total elapsed are different statistics. The disagreement does
not establish either a uniform 25% slowdown or a browser speedup. Browser burst
medians remain unchanged. Before selecting browser-specific scheduling policy,
the remaining measurement should use longer timed intervals with a completed
read on every call, while retaining this original matrix.

## Evidence

[summary.json](summary.json) retains all matrices, ranges, regressions, source
identities and product hashes. [manifest.json](manifest.json) binds 105 compressed
records: 73,539,130 raw bytes and 1,709,276 compressed bytes. The primary archive
contains the exact verification, replication and archive scripts, commands,
receipts, timing samples and correctness outputs. All members round-trip checked.

The supplementary [analysis script](analyze_browser_resolution.py) reads only
manifest-verified primary archive members. Its output includes its own hash,
the primary-summary hash and all four input hashes; it does not modify or replace
the primary evidence. Frozen runtime products remain locally under
`Library/Logs/SpiralTorch/module-inline-guard-20260912/verified-a`, not in Git.
