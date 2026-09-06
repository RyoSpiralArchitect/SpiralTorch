# Resident Readback Reuse

## Confirmed Comparison Without Probes

The primary evidence is now four runs **without `--readback-probe`**, in
baseline/candidate/candidate/baseline order. All 72 cases pass exact values and
canonical indices, clean source/build binding, stable image checks and the
pre/post foreign-compute-PID gates. Every normal report has
`readback_probe_requested=false`; no diagnostic process or per-case probe runs.
This is still not an exclusive GPU reservation or an interleaved-framework test.

Baseline: `7eca66104b2efe231bbedda5c9d895b70f81c9d5`.
Candidate: `d5e45127b478a57cfcc591858f89ab7c9baa2adb`.
The pool is unchanged from `0fb6955b`; the later commit separates diagnostics.
All request hashes match `f987634dda04fa6dc7ab232441c62cdb184d02771f18afaf184b015188b600e7`.
Hardware, integer fixtures, configuration and timing boundaries are described
below and unchanged in these reruns.

Units: microseconds. Median of three seed medians, each from twelve samples
after two warmups. The composed call includes final owned rank readback.

| Kind | Columns | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TopK | 257 | 763.20 | 78.77 | 79.28 | 770.50 |
| TopK | 1025 | 760.77 | 85.98 | 85.35 | 766.50 |
| MidK | 257 | 764.48 | 70.68 | 70.45 | 757.64 |
| MidK | 1025 | 758.95 | 91.82 | 91.54 | 767.33 |
| BottomK | 257 | 761.33 | 78.30 | 78.62 | 767.33 |
| BottomK | 1025 | 763.78 | 85.89 | 85.41 | 766.13 |

Warm latency remains **88-91% lower** after removing the diagnostic confound.
The no-map resident interval is a different result: candidate 2 is approximately
50.07-75.62 us/chain versus CUDA 16.23-16.76 us. CUDA remains about 3-4.5x faster.
MidK/1025 resident time is 75.67 / 75.62 us in the candidates versus baseline
70.38 / 70.43 us, a reproducible roughly **7.5% regression**. Other groups improve
slightly. Do not present the end-to-end gain as a general kernel throughput gain.
Cold starts, arbitrary floating inputs and full-model performance remain unmeasured.

One additional corrected `--readback-probe` run validates the new orchestration:
18 normal comparisons complete before a separate process emits 18 `probe_only`
results with `samples_ms=null`. They live in `readback_diagnostics`, not in normal
case timings. The normal native results contain no probes. This report is kept
separately and is not one of the four primary no-probe runs. Five dependency-free
harness tests include all-18-CUDA-cases-before-probe ordering and rejection of
changed indices, request identities, modes and timing-bearing probe results.

## Earlier Probe-Enabled Runs

**Exploratory only, not the final comparison:** review found that these runs
inserted diagnostics between requests and before CUDA timing. Allocator and
GPU load history could therefore contaminate later comparison samples. Raw
reports are retained unchanged. The runner now completes every comparison
before a separate probe-only process; the corrected reruns are above.

Retaining one idle staging buffer per resident rank/matmul workspace removes a
large warm readback allocation cost on Furnace. It does not fuse shaders, remove
the final GPU copy/map, or change ranking semantics. Python and browser clients
use the same Rust lease; outstanding snapshots remain independently owned.

Four clean-source, 18-case runs were performed in baseline/candidate/candidate/
baseline order. Baseline is `7eca66104b2efe231bbedda5c9d895b70f81c9d5`;
candidate is `0fb6955b2e33b7d88d1e21866d4e25e1f1799a54`. All pass exact values
and canonical indices, source/build binding, image stability, and pre/post
foreign-compute-PID gates. This is not an exclusive GPU reservation. Furnace
reports RTX 5090 for both WGPU Vulkan and PyTorch `2.13.0+cu132` CUDA.

All four request hashes are
`f987634dda04fa6dc7ab232441c62cdb184d02771f18afaf184b015188b600e7`:
three seeds (17, 29, 43), two projection shapes `(2,32,257)` / `(3,128,1025)`,
three rank kinds, k=7. Bounded integer inputs keep fp32 projection and tie checks
exact. CUDA TF32 is disabled. Native uses scalar/sequential matmul and rank
tile_cols=256. This is not an exhaustive kernel configuration search.

### Warm End-To-End Calls

Units: microseconds. Each cell is the median of three seed medians, with twelve
samples per seed after two warmups. `single_submit_bridge` includes one complete
matmul/copy/rank chain and final owned rank readback; inputs are already uploaded.
The six native modes rotate within timing blocks. Frameworks run in separate
blocks, not interleaved.

| Kind | Columns | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TopK | 257 | 762.25 | 78.91 | 78.19 | 767.89 |
| TopK | 1025 | 755.56 | 85.26 | 85.45 | 756.47 |
| MidK | 257 | 759.26 | 71.00 | 70.38 | 756.95 |
| MidK | 1025 | 757.69 | 91.87 | 91.34 | 757.73 |
| BottomK | 257 | 762.92 | 77.88 | 78.04 | 762.77 |
| BottomK | 1025 | 763.15 | 85.81 | 85.66 | 756.86 |

The warm group medians decrease about **88-91%**, or approximately 8.2-10.8x
lower latency. This is not a PyTorch speedup, cold-start result, confidence
interval, arbitrary-float correctness claim, or full-model performance claim.
Initial allocation remains; one idle buffer per live workspace consumes memory.

### Resident Throughput Is Still Behind CUDA

Units: microseconds per chain. Sixteen composed calls plus a completion fence
are timed and divided by sixteen; host maps are outside these intervals.
The CUDA reference uses preallocated matmul/stable-sort outputs and the same
repetition count, validating canonical tie indices as well as values.

| Kind | Columns | Baseline 1 | Candidate 1 | Candidate 2 | CUDA 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TopK | 257 | 61.37 | 57.95 | 58.05 | 16.38 |
| TopK | 1025 | 69.25 | 66.88 | 66.93 | 16.71 |
| MidK | 257 | 52.84 | 50.08 | 50.00 | 16.33 |
| MidK | 1025 | 70.43 | 76.78 | 76.67 | 16.70 |
| BottomK | 257 | 61.16 | 57.88 | 58.01 | 16.27 |
| BottomK | 1025 | 68.87 | 66.88 | 66.77 | 16.69 |

Keep the MidK/1025 regression visible: about 9% slower in both candidate runs.
The other groups improve slightly, but pooling is not a general compute-kernel
optimization. **CUDA remains about 3-4.6x faster** on this resident boundary.
Extra-fence stage timings below are diagnostics, not GPU-event measurements.

## Allocation Diagnosis And Caveat

The initial synthetic byte-copy probe (`2a26ef45`) looked cheap even with fresh
staging allocations. It held another MAP_READ buffer alive in every mode.
That control was therefore not representative of an isolated rank snapshot.
Its original raw report is preserved without rewriting its historical labels.
The current probe and guide explicitly state this confound.

The baseline fenced-stage probe locates roughly 0.72 ms in snapshot creation/
submission, not mapping (roughly 16 us). A private native test then measures
allocation, encoding, submission and map separately, with and without an
additional live MAP_READ buffer. The included Furnace test log records all
fourteen samples per condition: unanchored allocation about 683-812 us, anchored
allocation about 0.31-0.36 us. These conditions are fixed-order, not a randomized
allocator experiment. Their contrast supports allocation-lifetime sensitivity;
the exact driver/allocator mechanism has not been instrumented.

With the pool, the fenced snapshot submission stage is roughly 5.4-5.5 us and
the read stage roughly 12 us. Extra fences perturb this path; do not add these
diagnostic stages to claim an uninstrumented critical-path decomposition.
The four-run table above is historical probe-enabled evidence, not an isolated
comparison. Its direction must be checked against runs without those probes.

## Ownership And Validation

- One idle buffer maximum per workspace; outstanding snapshots do not alias.
  Native uses a mutex, browser uses thread-local ownership. The return link is
  weak, so a surviving snapshot does not keep a destroyed workspace cache alive.
- A successful read returns the staging buffer only after unmap. Unread snapshot
  drop can return it with queue ordering preserved. Failed/cancelled mapping
  detaches the cache and discards the buffer. Blocking and async cleanup have
  distinct unmap ownership, preventing the double-unmap caught during development.
- Native backend: 67 unit tests plus one shader validation test pass on Apple M4
  and RTX 5090. The focused lease test checks pending-map cancellation, bounded
  cache retention and reading after workspace drop. A follow-up assertion also
  explicitly checks that successful reads return a buffer to the cache.
- Real Chrome WebGPU: composed chain 27 cases / 4,887 assertions; rank 42 / 934;
  matmul 15 / 86. Four differently valued pending chain snapshots survive source
  and rank destruction. No page errors. This tests overlapping promises, not a
  JavaScript promise-cancellation API.
- Rebuilt private Python wheel: 88 passed, one optional skip, including retained
  rank and matmul results across buffer reuse. TypeScript contract, native
  backend/example and wasm st-core all-target strict clippy, and pinned fmt pass.

Browser reports identify `BrowserWebGpu`, not a physical GPU. They hash generated
assets and the fixture; they are not native source/build attestations. Generated
assets are local test outputs, not checked-in binaries. Only a private wheel was
installed; no package version or public release was changed.

See the [resident guide](../../../docs/performance/resident_rank.md) for API and
benchmark commands. Add `--readback-probe` to the Python runner for the extra
diagnostics, or run the opt-in Rust test with `--nocapture` for allocation stages.

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| readback-normal-baseline-7eca6610.json | 282298d4a17574b6fb68c3f98450481793c38bc1d49e68dd7e7d2ea1aac8f66f |
| readback-normal-baseline-repeat-7eca6610.json | 7783aebdfb0638eacc4f04bb04a778457a860dd40b42c7a700f39505e0be6d99 |
| readback-normal-candidate-d5e45127.json | f1dfe697fd1a569069ff7635d87e778aa043275b4a9147a7dcf57958a33485ce |
| readback-normal-candidate-repeat-d5e45127.json | 1125b6be96e5b1ff66ffe986b088dd393022eeaabc006e470a3c205f5c1b47b1 |
| readback-isolated-probe-d5e45127.json | 9b8ddd6bc5cf39a3656052a68d517d8a5b8d37bb28decd5746f7d18e9682bbd7 |
| readback-probe-2a26ef45.json | acba9b701f5377c8136e69a67efc11ccf41173fc9917e4a3153222d2d9e7d40c |
| readback-stages-7eca6610.json | bfcd65ff431f986f7c617569a39df3e7b8c36597ffccf46789f86740f1d90d10 |
| readback-stages-repeat-7eca6610.json | 3d34e35661f7549eab4f4b1131e2fca826fbe0941fca93943190f3bfdf163f39 |
| readback-pool-0fb6955b.json | de7bbdae947496832089d31c66eccf1c92a61d8ff4628030df92dfc96b3af9df |
| readback-pool-repeat-0fb6955b.json | 5dcd92027bd920d0f204858eaf690d2c9afb1f9d663099ff1defa60b8c409588 |
| spiraltorch-readback-pool-browser-final.json | c889a81f6e2ec7ebab6e53421843553c24bea000cef8570f1fbca5daf20e9af4 |
| spiraltorch-readback-pool-browser-rank.json | cfc2030705325f5e8002f7db0904f20a0509b113c32b71536e8e8822803d223d |
| spiraltorch-readback-pool-browser-matmul.json | 2f7faf5b28f07e8a949c772a9586da12a2b22d8f98d83b99211ccfad3d2cc664 |
| spiraltorch-readback-pool-furnace-tests.log | 38d5c8e9e8013d7fef5262ee2d4d61ce786ca28ef44abdef6e3e6c8e0d27f992 |
