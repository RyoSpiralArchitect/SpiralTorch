# Attention Key Tiling: Full-Chain Comparison

**Bounded improvement versus the previous ST kernel, not a PyTorch win.**
The final shape-selected kernel reduced measured 128/256-token full-chain
resident latency by factors of 1.21 to 1.39 on this Apple M4. PyTorch MPS remains
substantially faster. Short-sequence timings are sensitive to warmup; no general
no-regression, learning-quality, CUDA or browser-throughput claim is made.

## What Changed

The previous portable kernel reduced one key dot product across 64 lanes at a
time. The candidate computes four independent dot products with 16 lanes each,
reducing workgroup barriers per key while retaining ordered online softmax and
value accumulation. Dot-product rounding order changes, not the attention
function. Structural masks, Z-RBF bias and inherited non-finite guards remain.

Unconditional four-key tiling regressed the short case, so it was **not adopted
as a blanket policy**. The final Rust selector uses it for at least 128 keys and
head dimension at most 32. Other shapes keep the single-key reduction. This is
a bounded heuristic informed by three measured shapes, not evidence that every
shape/device within that range wins. Both specializations share core WGSL and
are cached lazily; no Python/JS geometry semantics were added.

## Fixed Workload And Observations

All runs use the complete frozen QKV -> attention -> output-projection chain,
float32, with identical analytic inputs and four projection parameters. Shapes
`[B,T,I]` are `[1,32,64]`, `[2,128,128]`, `[1,256,256]`, with 4/4/8 heads and head
dimensions 16/32/32. Each has plain, zero-bias and Z-RBF conditions, causal and
unmasked: 18 numerical cases. The same bias is supplied to the Torch function;
biased ST is not checked against a different, plain-attention oracle.

The main comparisons rotate four engines through four order positions, with
three warmup blocks and seven retained blocks per engine/case/boundary per
round. Each resident block has four forwards plus GPU completion; each
host-to-host block uploads fresh input/bias and reads an owning host output.
Weights stay resident. Setup/compilation, fixed geometry/mask preparation and
numerical checks are outside the timers. Every burst output is checked. These
are host-observed timings including encoding/allocation, not kernel timestamps.

Illustrative **causal + Z-RBF** medians, milliseconds per forward from the final
four-round run; all other conditions and every sample are in the JSON files:

| Shape | ST before, resident | ST selected, resident | Torch MPS, resident | ST before, H2H | ST selected, H2H |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[1,32,64]` | 0.458 | 0.414 | 0.170 | 0.663 | 0.637 |
| `[2,128,128]` | 1.082 | 0.884 | 0.199 | 1.488 | 1.312 |
| `[1,256,256]` | 3.521 | 2.812 | 0.364 | 4.369 | 3.775 |

Across all 12 larger-shape conditions, pooled-median speed ratios are 1.21-1.39
resident and 1.13-1.38 host-to-host. All 96 larger-shape within-round paired
medians favor the selected kernel; these correlated observations are descriptive,
not 96 independent trials or a significance test. CPU results are also retained;
the one-thread CPU route is a distinct comparison, not a GPU fallback.

## Rejected Alternative And Sensitivity

`unconditional-comparison.json` keeps the first four-round experiment. Its
short-shape resident ratios were 0.82-0.92 (slower), despite larger-shape gains.
`adaptive-comparison.json` keeps the follow-up rather than overwriting it. Even
with single-key short execution restored, short/unmasked/plain remained slower
in that run: 0.416 -> 0.494 ms resident, 0.615 -> 0.841 ms H2H. Two of four
candidate rounds were elevated; the other short conditions were much closer.

`warmup-diagnostic.json` is a **post-hoc sensitivity check**, not a replacement
endpoint: the same short shape, three rotated engines, 50 warmups and 21 samples
per round, three rounds. That condition measured 0.455 -> 0.448 ms resident and
0.736 -> 0.640 ms H2H. The slower short observation did not reproduce, but the
diagnostic does not identify a clock/driver cause or prove non-regression.
The driver now defaults to 50 warmups for future work; the earlier results and
their actual three-warmup recipe remain frozen. Do not advertise a short-shape
speedup from these noisy observations.

## Correctness And Scope

All timed outputs pass `abs(actual - expected) <= 3e-6 + 3e-5 * abs(expected)`
against a frozen PyTorch 2.12.1 CPU math reference. Timed Torch uses eager default
SDPA dispatch, one CPU thread, MPS fallback/fast-math disabled, and disabled
optional global Spiralton patches. ST retains runtime non-finite guard kernels;
Torch is not given equivalent guards, so costs are not identical.

Native validation passed 68 resident-tensor tests, 14 NN attention unit tests,
the original 12-case chain fixture, and strict backend Clippy. Browser WASM
passed all 18 larger cases and three geometry checks, including both pipeline
specializations; its largest output error was 1.7881393432617188e-7. Browser
metadata does not identify the physical GPU. The browser observation is
correctness-only. NN benchmark Clippy passed with pre-existing library warnings.

This is a single-machine exploratory optimization study with deterministic
synthetic values, no clock isolation and a shape policy chosen from its results.
It does not establish out-of-sample performance, full-decoder speed, training
benefit or equivalence on other hardware. Resident QKV packing, head merge,
parameter projection and multiple queue submissions remain optimization targets.

## Reproduction And Files

- Baseline implementation/harness: `107d733197f4b0b55742efcf57e003239330ba6d`.
- Rejected unconditional tile: `b88ece705f4e287c5939f586d72a88d746ff4541`.
- Selected implementation: `a147cf56e94f8f087d0266333941f320f67d6150`.
- Build each native example with Rust 1.98.0, `--locked --release --features wgpu`,
  preserving separate executables. [Commands and boundaries](../../../docs/resident_zspace_attention.md#bounded-performance-comparison)
  describe the fixture generator and driver. Use `--warmup 3` to reproduce the
  historical main recipe rather than the new default, `--rounds 4 --samples 7`.
- For the diagnostic, keep only the first generated scenario and run
  `--devices cpu --rounds 3 --samples 21 --warmup 50`; both native binaries remain.
- `scout.json` preserves the initial one-round exploratory run. The two main
  comparison files and the diagnostic contain complete conditions and samples.
  `browser.json` is the actual rendered WASM report.
- `provenance.json` binds source revisions, fixture and executable hashes. Raw
  generated fixtures, executables and validation logs remain local. No model
  weights or full raw logs are published. Rebuilt executables need not hash
  identically on a different build environment.
- Run `shasum -a 256 -c SHA256SUMS` from this directory to verify published files.
  Earlier correctness records and their source-revision hashes are unchanged.
