# Single-Pass Resident Rank and GPU Stage Diagnostics

Measured candidate: `2c72e26b5af58da8138c603e8ceb42303bd11135`.
Baseline: `ed21f2000d49b3add759b8a4b53fb3b46b4e60ee`, whose executable
source equals main `1934d7091fb244fe5566dfc20712c28f384c0cd1`.

## What Changed

Ordinary resident rank now encodes all sort/merge dispatches in one compute
pass and one submission. WebGPU defines each compute dispatch as a separate
[usage scope](https://gpuweb.github.io/gpuweb/#synchronization); wgpu tracks
storage dependencies between dispatches, including scratch reuse. Kernel
algorithms, tile selection, source-index ordering and adaptation rewards are
unchanged. Rust owns this path for native, Python and WASM clients.

An opt-in timestamp runtime exposes `dispatch_profiled` in Rust and `profile`
in Python/WASM, with one Rust report schema. It does not replace the default
runtime or feed instrumented timings into Black Cat policy feedback. Portable
stage timestamps use **separate passes**, so they are diagnostics, not direct
measurements of the normal single-pass path. More than 256 repetitions use
multiple submissions; native Metal paces those chunks with bounded waits.
Ordinary dispatch remains enqueue-only, without this pacing or query allocation.

The new instrumentation exposed two defects before admission: Metal exhausted
its command-buffer pool while encoding the unchunked 1024-repetition prototype,
and repeated browser query allocation exhausted counter-sample buffers while
waiting for JavaScript GC. The browser initially returned cleared timestamp
buffers after uncaptured errors. That entire run is invalid evidence.
The final code explicitly destroys owned browser query resources after
submission, captures validation/allocation errors, and checks them before
accepting timestamps. The narrow pinned-wgpu extension does not change normal
query-handle Drop or native backend behavior. Both negative logs are retained.

## Native Comparison

Furnace ran WGPU/Vulkan and PyTorch **2.13.0+cu132 CUDA** on the same named
RTX 5090, with no overlapping GPU processes. Each native process completed
before its CUDA reference started. Four runs used A-B-B-A source order:
baseline, candidate, candidate, baseline. These are process-block controls,
not cross-framework interleaved samples or GPU-event comparisons.

Each run has 36 cases: three seeds, three rank kinds, two row widths, two
policies, and four fixed tile controls per case. Controls are correctness-gated
but not policy feedback. There are **144 cases and 9,216 real Rust feedback
observations** across all four runs. Timings retain 12 batches of 16 resident
operations plus completion, after two warmup batches; allocation/upload,
validation readback and policy work are outside these timing intervals.

| Metric | First Pair | Reverse-Order Pair |
| --- | ---: | ---: |
| Fixed candidate/baseline mean-latency ratio, median over 144 controls | 0.819 | 0.814 |
| Ratio range across those controls | 0.572-0.984 | 0.573-0.992 |
| Hindsight-best fixed WGPU / CUDA median-latency ratio | 1.450-5.168 | 1.446-5.176 |

Thus the fixed controls improve by about **18-19% at the median**, but this
is still not a general PyTorch win. Repeated seeds and the two policies are
not new independent seeds. The hindsight-best fixed tile is not a guaranteed
online-policy outcome, and this study measures rank, not model quality.

## Browser Controls

Two same-browser matched runs passed 44 MidK cases each. Frozen baseline and
candidate modules use separate devices, alternate arm order, warm four paired
batches, and retain 16 pairs of 64 operations plus a four-byte completion
fence. Every batch checks exact values and canonical source indices, including
ties, signed zero and NaNs, outside timing. All served assets are hashed.

| Candidate/Baseline Mean-Latency Ratios | Median | Range |
| --- | ---: | ---: |
| Matched run 1 | 0.981 | 0.813-1.058 |
| Matched run 2 | 0.981 | 0.840-1.083 |
| Baseline vs itself | 1.003 | 0.827-1.129 |

The A/A spread is substantial. The roughly 2% median shift is **not a general
browser speedup claim**, and slower cells remain visible. These are repeats of
one deterministic fixture, not independent seeds. Chrome 152.0.7977.77 exposes
an anonymous BrowserWebGpu adapter; no physical GPU identity or native/browser
timing equivalence is claimed.

## Stage Diagnosis and Validation

The separate native stage probe passes 72 cases. It alternates uninstrumented
dispatch/completion and instrumented query readback, never subtracting GPU
clocks from host clocks. For rows=2, cols=8193, k=65, the following are medians
over the three seeds of mean per-operation **instrumented** stage spans:

| Kind | Tile | Sort (us) | Merge (us) |
| --- | ---: | ---: | ---: |
| MidK | 128 | 12.8 | 121.5 |
| MidK | 256 | 18.6 | 115.1 |
| MidK | 512 | 33.5 | 43.6 |
| TopK | 256 | 18.6 | 107.0 |
| BottomK | 256 | 18.6 | 107.0 |

This makes merge a useful next optimization hypothesis, not proof that these
numbers describe the uninstrumented path. Any kernel change still needs the
normal matched benchmark. Larger tiles trade sorting cost against merge cost;
the diagnostic does not silently rewrite SpiralK geometry or policy choices.

Final checks passed:

- macOS and Furnace: 82 live-enabled backend library tests plus standalone WGSL
  parsing. Maximum rank-only repetition and query budgets both execute.
- Packaged Python: 103 rank/resident/SpiralK/adaptation/profile tests.
- Browser profiling: 12 paired cases, pending-profile ownership across reuse
  and drop, unchanged default runtimes, and 1024 profile/ordinary repetitions.
- Native/WASM backend strict Clippy, Linux core all-targets strict Clippy,
  formatting, WASM builds, and the CPU-only Python feature check.

Raw reports validate 1,009 timestamp profiles, preserving integer ticks as
strings and checking periods, stage order, integer deltas, sums and zero counts.
Quantized zero intervals are allowed only after successful GPU validation.
The separate matmul/copy/rank chain's maximum repetition boundary is **not**
covered by this rank-only test or claimed as fixed here.

## Artifacts

[summary.json](summary.json) contains all cell ratios and raw-file hashes.
[raw-logs.tar.xz](raw-logs.tar.xz) retains reports, failures, source bundles,
build/test logs, product identities and the `run_profile_probe.py`, `analyze.py`
and `commands.md` replay recipes. Native reports embed clean source/build
binding and executable hashes. Python/WASM attribution uses build logs and
product hashes, not an embedded Git attestation. Large products are excluded.
No GPU jobs overlapped; the existing Furnace CPU workload was not modified.

After extraction:

```sh
python -I analyze.py --repo /path/to/SpiralTorch --output recomputed.json
```

Extraction and recomputation reproduced the committed summary byte-for-byte.

Archive SHA-256: `22fb757b9de8d622892297d0318c643bf1d9f5b79d069541f53792931e1e9b7e`.
Summary SHA-256: `535ad51f37ed527bad902481c58bb8cc9255cb2c3aa6a9d15d5957d1f00a6b45`.

## Post-Review Revalidation

The study above remains frozen at `2c72e26b`. Review found two additional
diagnostic-path defects, reproduced before fixing them in
`b7999e8f6ac1082a55a5f9f560859e145969da3e`:

- Profiled output now stays stale until that particular profile successfully
  validates and reads its timestamps. Uploads, newer profiles and ordinary
  dispatch detach its publication token, so an older completion cannot publish
  another operation's output. Failed or dropped profiles do not publish.
- Native diagnostic encoding sharing the same actual device now rejects an
  overlapping diagnostic error scope before query allocation or submission.
  Pending readbacks do not hold the guard. Ordinary execution and independent
  devices remain unrestricted; no execution/readback lock or retry was added.

The browser regression injects one actual invalid WebGPU query count and checks
rejection, stale output and recovery through ordinary dispatch. It restores the
isolated test prototype immediately. Generated TypeScript constructor and
profile signatures are also checked; this fixes the independent CI contract
failure rather than disabling the check.

Fresh products passed 86 live backend tests plus WGSL parsing on each of macOS
and Furnace, 103 packaged Python tests, 72 native profile cases and 12 browser
profile cases including the rank-only 1024-repetition boundary. The replay
validates another 1,009 timestamp reports and the real browser rejection.

One native before/after pair passes 72 cases and 4,608 Rust policy observations.
For the 144 fixed controls, the mean-latency candidate/baseline ratios have
median **0.998** and range **0.967-1.023**. One 44-case browser matched run has
median **1.002** and range **0.908-1.157**. These are regression controls, not
additional speedup evidence or a claim that every cell is unchanged. The
original PyTorch comparison and its limitations remain in force.

[review-summary.json](review-summary.json) and
[review-logs.tar.xz](review-logs.tar.xz) preserve the separate raw reports,
red/green tests, build logs, product hashes, source bundle and replay scripts.
The installed Python extension is byte-checked against the new wheel; browser
assets are checked against the captured products. Native reports retain their
clean source/build binding. After extraction, run:

```sh
python -I analyze_review.py --repo /path/to/SpiralTorch --output recomputed.json
```

Extraction and recomputation reproduced the review summary byte-for-byte.
The original archive and summary were not replaced.

Review archive SHA-256: `a098243828e9c3b59c2853f0ca460214f209c0dd03586ae0ae8f76c353b5c711`.
Review summary SHA-256: `1d0b40ce49e712b0de82c31befbc2caa74d515bc37ff4ad0f44273c7063061d0`.

## Internal-Error Review

A second review found that diagnostic scopes omitted `wgpu::Error::Internal`.
`d8749294c6266f8b43668fc2fbe566146460ac3d` adds the third scope and checks all
three results before publication. The pinned browser converter also maps the
`GPUError` base type to a typed internal failure rather than trapping when its
generated web-sys snapshot lacks `GPUInternalError`.

The regression performs real scope push/pop calls and substitutes a synthetic
`GPUInternalError` result for the Internal scope. This is **not an induced driver
fault**. The frozen old module lacks that scope and publishes output; the fixed
module rejects with the injected message, keeps output stale and recovers through
ordinary dispatch. The separate real invalid-query Validation test still passes.

Fresh products pass 87 live backend tests plus WGSL parsing on each native host,
103 packaged Python tests, generated TypeScript contracts, 72 native profile
cases and 12 browser profile cases. Another 1,009 timestamp reports revalidate.
Native/WASM backend strict Clippy also passes. Ordinary rank dispatch and shader
code are unchanged from `b7999e8f`; this adds no normal-path speedup claim.

[internal-error-summary.json](internal-error-summary.json) and
[internal-error-logs.tar.xz](internal-error-logs.tar.xz) retain red/green reports,
builds, tests, product hashes and `analyze_internal.py`. After extraction, run
that analyzer with `--repo /path/to/SpiralTorch --output recomputed.json`.
Extraction and recomputation reproduced the summary byte-for-byte. Earlier
archives remain unchanged.

Internal-error archive SHA-256: `f66a5712dd6b10862307ef406f8a8a262e2bda05aa721aa61ebcc291895269ec`.
Internal-error summary SHA-256: `fd174c52c69643785733be9f96e1973eb3ddd2f6e709228ce256986bc059b6bc`.
