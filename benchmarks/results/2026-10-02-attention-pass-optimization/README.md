# Attention Pass Optimization

This study separates an isolated GPU diagnostic from full-chain inference
latency. It retains a numerically correct but slower normalization experiment,
rather than presenting every algebraic simplification as an optimization.
The full-chain comparison completed all 64 engine-runs and numerical gates.
The selected candidate improves the measured larger resident chains versus
the prior ST implementation, but is not a PyTorch win or a no-regression claim.

## Full-Chain Results And Adoption

Eight rotated rounds compare before, physical-row-order and eight-key kernels
with both Scalar and Register16 projections, plus PyTorch CPU/MPS. Three shapes
each have plain, zero-geometry and Z-RBF controls, causal and unmasked. Every
engine/case/boundary has 72 samples after 50 warmup blocks per round; resident
samples contain four forwards. All samples are retained in `comparison.json`.

Selected causal Z-RBF resident medians, milliseconds per forward:

| Input [B,T,I] | Scalar before / selected | Register16 before / selected | Torch CPU | Torch MPS |
| --- | --- | --- | --- | --- |
| [1,32,64] | 0.272 / 0.274 | 0.275 / 0.277 | 0.029 | 0.172 |
| [2,128,128] | 0.920 / 0.812 | 0.662 / 0.576 | 0.270 | 0.199 |
| [1,256,256] | 3.036 / 3.031 | 2.195 / 1.921 | 0.842 | 0.396 |

All 24 larger-shape/projection resident pooled medians improve versus before
and versus the row-order-only candidate. Before/selected ratios are 1.002-1.144
for larger Scalar resident cases and 1.092-1.184 for Register16. A ratio very
near one is not a resolved improvement beyond noise. Larger host-to-host
medians include two small Scalar reversals; Register16 ratios are 1.051-1.253.

Across all shapes and boundaries, 11 of 72 pooled medians are slower. The
largest reversal is short/unmasked/zero-geometry Scalar host-to-host, 0.496 to
0.558 ms, even though this shape still selects single-key reduction. Of 576
within-round paired medians, 443 favor the candidate; 328 of 384 larger-shape
pairs do. These are correlated descriptive observations, not independent
trials or a significance test. No thermal/compiler/driver cause is established.

Retain eight-key reduction within the existing `keys >= 128, head_dim <= 32`
range, together with direct-output row ordering. Do not widen that range or
promote Register16 or merged-head output to the ordinary NN default. The shape
rule is informed by this small synthetic study, not every shape/device in the
range. The ordinary head-major route is numerically tested and internally
timestamped; its complete NN chain is not timed in this study.

The before release binary calls direct merged-head output through the former
`forward` implementation; subsequent binaries call the explicit
`forward_merged_heads` API. All three measure the same direct-output topology,
not an accidental default-versus-opt-in comparison. Source revisions and binary
hashes distinguish this from the diagnostic baseline revision.

Resident timings include host encoding/allocation/submission and completion,
but exclude output reads/checks. Host-to-host includes fresh input/bias copies
and owning output readback, with weights resident. Setup, geometry construction
and mask preparation are excluded. ST retains its finite guards; Torch does
not execute equivalent guards. Torch 2.12.1 uses eager SDPA default dispatch,
one CPU thread and no MPS fallback/fast math or optional global patches.
The same fixed geometric bias is supplied to both engines: this tests equivalent
operations, not the learning value of one geometry against another.

## Candidates And Diagnostic

The baseline uses four simultaneous 16-lane key dot products for at least 128
keys and head dimensions up to 32. Short sequences and wider heads use a
single 64-lane dot product. All candidates keep portable core WGSL, structural
causal masking, additive Z-space biases and inherited finite-value guards.

1. **Physical output order:** direct merged-head output dispatches workgroups
   in `[B,Q,H]` order, avoiding a final row-address remap. Ordinary output keeps
   `[B,H,Q]`. This is not a change to the public tensor layout.
2. **Rejected tile normalization:** computes a tile maximum and normalizes its
   weights together. Numerical gates pass, but the isolated larger-input pass
   medians are 11-15% slower than the row-order candidate. The implementation
   is reverted; its commit, executable hashes and every diagnostic condition
   remain available. Its release executable was built but not benchmarked.
3. **Eight-key ordered normalization:** computes eight dot products with eight
   lanes each, retaining sequential online softmax and value accumulation.
   It reduces workgroup synchronization frequency, not numerical safeguards.
   Dot-product reduction order changes; agreement is tolerance-based. The
   short/wide-head selection boundary is unchanged. Workgroup storage grows
   from 2344 to 2376 bytes, with matching preflight validation.

The native-only ignored probe uses existing GPU timestamp infrastructure through
a private cursor in the real attention/guard encoding path. Ordinary API calls
use a disabled cursor: no timestamp queries or extra passes are allocated.
There is no new public profiling API or host-clock substitute for GPU timestamps.

For each of 18 conditions and two projection implementations, resident graph
primitives produce QKV once from the same frozen chain fixture. After 50 warmup
pairs, 25 head-major/direct pairs alternate order. Only the attention and
inherited-guard passes are timed. Every retained output subsequently passes
through the output Linear and is compared with the Torch oracle. Each report
contains all 1800 records. QKV/output projections, head packing, CPU work,
uploads, query resolution, readbacks and checks are outside these timestamps.
The diagnostic host executable is a debug test build, not the release benchmark.

These are sequential exploratory diagnostics without clock isolation. Short
inputs vary sharply even on the unchanged head-major control. The eight-key
candidate also has a slower batched/Scalar diagnostic condition despite faster
wide-input observations. This motivates the rotated full-chain comparison;
neither a hardware/driver cause nor a universal speedup follows from the probe.

## Verification

- 74 native resident-tensor tests passed, including 17 attention tests. New
  coverage includes flat and sharply biased distributions, multiple batches
  and heads, all eight-key tail residues, cached causal offsets and overflow
  in the last key. Both output orders preserve rejection and masking.
- 15 NN attention unit tests and four chain integration test functions passed.
  The integration suite checks 72 outputs across three projection variants and
  both default and direct-output routes.
- Browser WASM passed 80 standalone outputs, three independent compilation
  constant checks, 180 chain outputs and 15 geometry checks. Maximum chain
  error was `1.825392246246338e-7`; standalone error was `8.940696716308594e-8`.
- Strict backend Clippy and pinned Rust formatting passed. Four lightweight
  benchmark-driver validation tests passed.

Browser evidence proves numerical checks, not browser throughput or physical
adapter identity. All comparisons use `3e-6 + 3e-5 * abs(expected)` tolerance.
No backward, KV-cache ownership, complete decoder, learning quality or CUDA
claim is made. Pairwise geometry still occupies quadratic storage and is
prepared outside the inference timers. Python/JavaScript do not reinterpret it.

## Reproduction

Source revisions and binary/fixture hashes are recorded in `provenance.json`.
The full-chain fixture is produced by the existing generator's benchmark suite;
see [the shared commands and boundaries](../../../docs/resident_zspace_attention.md#bounded-performance-comparison).
Build release `resident_attention_chain_bench` executables separately for each
revision, then pass their paths to `tools/bench_attention_chain_vs_torch.py`.
Keep Scalar and Register16 as separate labelled engines, plus Torch CPU/MPS.
Use eight rotated rounds, nine samples, 50 warmups and a resident burst of four.
Disable the documented global patches, MPS fallback and fast math before startup.

To reproduce the diagnostic, build the native `st-backend-wgpu` library tests
at each recorded source revision. Set `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1`,
`SPIRALTORCH_ATTENTION_PROFILE_FIXTURE` to the generated benchmark fixture and
`SPIRALTORCH_ATTENTION_PROFILE_OUTPUT` to a fresh path. Run only
`resident_tensor::attention::profile_probe::profile_fixture_attention_passes`
with `--ignored --exact --test-threads=1 --nocapture`. A timestamp-capable real
GPU is mandatory; failure must not silently become a CPU or wall-clock result.

Full condition results, rejected observations, rendered WASM reports and hashes
are public. Original executables, generated WASM/JS and full validation logs stay
local. Rebuilt binaries can have different hashes on a different toolchain or
host. Earlier frozen studies are not rewritten. Verify published artifacts with
`shasum -a 256 -c SHA256SUMS` from this directory.
