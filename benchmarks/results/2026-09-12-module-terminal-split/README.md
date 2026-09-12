# Terminal NN Capture: Keep The Original Split Scheduling

**Decision: keep the explicit terminal API, with the original forward-then-copy
ordering.** In the same-API browser comparison, this version is faster than the
rejected single-submission prototype in all 36 retained pairs. It does **not**
establish a speedup over the original API: those shape medians remain slightly
slower. Both findings are preserved.

Original runtime: `6703df35e8389c3ebb9ddf5eb099f440995a930d`.
Rejected single-submission prototype: `7626150d31fef4da440fedefff66c066514df48a`.
Split terminal runtime: `06ca8046234aeabb37b131e1deb8ec2fd66bf199`.
This record follows the [negative single-submission experiment](../2026-09-12-module-terminal-capture/README.md).

## Implementation And Boundaries

Rust `Module::forward_resident_snapshot`, Python `model.forward_snapshot(input)`
and WASM `model.forwardSnapshot(input)` keep the same owning snapshot contract.
The implementation now calls the existing forward, then the existing snapshot:
GPU computation is submitted before allocating/preparing the terminal copy.
Gelu/Relu and explicit graph terminal calls follow the same ordering. Strided
Tensor snapshots also restore the original pack-then-copy submissions.

Six ordinary-path source functions are byte-identical to the original runtime:
Tensor apply/apply_into/snapshot/contiguous_into, graph forward_tensor and the NN
cache forward. The hash-bound body comparison is retained. This establishes
source-body identity, **not** generated-code identity or identical performance.
The new API is not a staging-cache or shader-math optimization.

Ordinary NN forwarding still returns an owning GPU Tensor; only the explicit
terminal API requests a capture. Parameters, cache selection and guards remain
Rust-owned and shared across the clients. Retained results survive later
forwards, weight changes, cache clear and module destruction. Empty Sequential
keeps its zero-dispatch counters. Unsupported routes reject, without CPU fallback.
Generic autograd, ModuleTrainer and ordinary pure::Tensor are not migrated.
See the [API guide](../../../docs/module_resident_forward.md).

## Fixed Browser Results

Three separate protocols used four matrices, nine shape/seed cases, two warmup
and nine retained intervals per route. Each interval contained 256 independent
forwards, **each with its own completed host read and release**. Setup, upload,
cold compilation and numerical checking were excluded. All typed outputs were
checked after each interval; all interval timings and final per-route arrays
remain in the archive. No retries, trimming or adaptive stopping were used.
Context/case order alternated and the four routes rotated.

Primary endpoint: median of 12 case-median candidate/reference ratios per shape.
Lower is faster. The protocols are not pooled into a single performance claim.

| Shape | Ordinary split / original | Terminal split / original ordinary | Same API: split / single submission |
| --- | ---: | ---: | ---: |
| `[2,3,7]` | 1.0031 | 1.0040 | 0.9431 |
| `[2,8,64]` | 1.0063 | 1.0024 | 0.9565 |
| `[4,8,128]` | 1.0023 | 1.0041 | 0.9819 |

Same-API ranges are 0.9240-0.9611 / 0.9474-0.9754 / 0.9713-0.9910,
with zero slower pairs in each shape. Pooled elapsed ratios are
0.9502 / 0.9598 / 0.9892. Its independent explicit-graph control medians are
1.0039 / 1.0029 / 0.9997; these are not used to normalize away differences.
The result supports choosing split scheduling over this prototype, not an
isolated GPU queue-cost or overlap mechanism claim.

Original ordinary controls have 7/12, 12/12 and 10/12 slower pairs; their pooled
ratios are 1.0135 / 1.0043 / 1.0125. Terminal-vs-original has 9/12 slower pairs
in every shape, with pooled ratios 0.9988 / 1.0031 / 1.0143. These residuals
remain visible. All ranges, individual rows and independent controls are in
[summary.json](summary.json) and the raw reports.

Chrome `152.0.7977.84` reported `BrowserWebGpu` / `Other`, a blank physical
adapter name and `crossOriginIsolated=false`. Browser physical GPU and host
exclusivity are **UNKNOWN**. Minimum measured intervals were 59.2 / 58.7 / 59.3 ms;
the smallest observed positive clock delta was about 0.1 ms, not an uncertainty
bound. Across the three protocols, **1,216,512 completed reads** and
**2,093,211,648 values** were checked, with maximum error `2.384185791015625e-7`.

## Native Controls And Verification

Four fixed native pairs alternated process order, using the terminal candidate
against original ordinary forwarding, plus eager PyTorch CPU/MPS controls.
The burst has eight independent forwards and only one final read; it is not the
browser completed-read workload. Native device: Apple M4/Metal, MPS CPU fallback
disabled. Native results do not establish a universal gain:

| Shape | Terminal completed-read ratio | Range | Terminal burst ratio | Range |
| --- | ---: | ---: | ---: | ---: |
| `[2,3,7]` | 0.9952 | 0.6073-1.0387 | 0.9707 | 0.6022-1.0521 |
| `[2,8,64]` | 0.9944 | 0.9860-1.0074 | 0.9980 | 0.9846-1.0111 |
| `[4,8,128]` | 0.9969 | 0.9759-1.0132 | 1.0009 | 0.9942-1.0076 |

The small native case has substantial variation in both runtime and independent
controls. No fastest-PyTorch claim follows; every trial is retained.

All 56 verification stages passed in 559.6 s and all 18 measurement stages in
642.1 s. Backend: 134 tests. Tensor: 449 CPU and 497 WGPU tests, one unchanged
ignored test. NN: 707 CPU and 760 WGPU tests including storage tracking.
Python GPU suites: 61 tests. Browser Module: 224 assertions. CPU-only clients,
handoff, generated/shipped types, learner/autograd/fusion and ten interval
admission tests passed. PyTorch 2.12.1 CPU/MPS training/VJP checks passed
120 cases and 17,136 comparisons, maximum error `5.7220458984375e-6`.

The browser N-D fixture retained 24 snapshots, 48 intervening discard/read cycles,
two delayed guard failures and negative-zero bits. Four ordinary and four
terminal pending Rust reads were cancelled without poisoning surviving captures;
native blocking reads do not claim cancellation coverage.

[manifest.json](manifest.json) binds 276 compressed records: 127,954,868 raw bytes
and 3,009,052 compressed bytes, round-trip checked. All 61 frozen products from
the three source versions were rehashed. Drivers, source blobs, streamed cases,
receipts and raw controls are preserved; binaries remain in their original local
verification directories. The old prototype's literal `--launch` directory is
unchanged. No CUDA/Furnace run, release, push or merge is part of this record.
