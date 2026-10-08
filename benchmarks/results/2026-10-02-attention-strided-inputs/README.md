# Direct Strided Attention Inputs

The portable attention kernel now reads Q/K/V and optional Z-space score biases
from their existing immutable N-D views. It no longer materializes those views
into contiguous GPU buffers before attention. Head merging after attention
still copies values on GPU; projection outputs, multiple submissions and
non-finite guard work are also still present. This is not a zero-copy decoder.

## Implementation And Semantics

Rust builds five checked, fixed-rank view descriptors from the existing
`NdLayout`. Offsets and element strides, including zero broadcast strides, are
carried in a 208-byte uniform block. WGSL uses those addresses for Q/K/V and
both score-bias inputs. The attention equation, key-tiling policy, causal
visibility and ordered online normalization are unchanged. Both clients use
the same Rust/WGSL implementation, not a Python or JavaScript reinterpretation.

The original source buffers remain borrowed until submission; owned output
versions and queue ordering are retained. Inherited failure guards are copied
from the original inputs, so a crop, masked key or empty query cannot conceal
an upstream failure. Original storage bounds, address width and device limits
are checked before dispatch. There is no CPU fallback or host activation read.

For the benchmark layouts, the removed QKV materializations account for
`3 * B * T * I * 4` bytes of temporary f32 value buffers: 24 KiB, 384 KiB and
768 KiB for the three shapes. A broadcast pair bias in the batched case also
previously materialized 512 KiB. These are sizes inferred from the removed
allocations, not measured peak-memory savings; guard/metadata copies remain.

## Matched Workload

Both frozen release executables run the full QKV -> attention -> head merge ->
output-projection chain. Each runs Scalar and Register2x2/16x16 projections.
Torch CPU and MPS complete the six-engine rotation. The like-for-like comparison
is before/after **within the same projection preset**, not old Scalar against
new Register16. Parameters, accumulation and bias are unchanged.

Shapes `[B,T,I]` are `[1,32,64]`, `[2,128,128]` and `[1,256,256]`, with 4/4/8
heads. All plain/zero-bias/Z-RBF and causal/unmasked combinations are included:
18 cases per engine, six rotated rounds, 50 warmup blocks and nine retained
blocks per boundary per round. Each resident block has four forwards plus GPU
completion; host-to-host blocks copy fresh input/bias and return an owning host
output. Weights remain resident. Compilation, fixed geometry/mask preparation
and output checks are excluded from timing; every burst output is checked.

These are host-observed full-chain measurements, not kernel timestamps or
isolated allocation timings. Timing includes the extra strided address math,
as well as any saved encoding, allocation and packing work.

## Main Results

**Useful reduction in short-input latency; no universal speedup.** The main
six-round run has short-shape resident speed ratios of 1.25-1.46 with Scalar and
1.16-1.24 with Register16. Register16 resident pooled medians improve in all 18
conditions, with ratios 1.01-1.24, but some wide host-to-host observations regress.
The default Scalar path also has two slower wide resident observations. All
conditions remain in the published comparison, including these exceptions.

Illustrative **causal + Z-RBF** pooled medians, milliseconds per forward, from
the main run (54 retained samples per engine/case/boundary):

| Shape | Scalar before / after, resident | Register16 before / after, resident | Torch CPU resident | Torch MPS resident | Register16 before / after, H2H |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[1,32,64]` | 0.422 / 0.332 | 0.419 / 0.337 | 0.029 | 0.167 | 0.648 / 0.580 |
| `[2,128,128]` | 0.911 / 0.837 | 0.717 / 0.638 | 0.278 | 0.195 | 1.111 / 1.037 |
| `[1,256,256]` | 2.797 / 2.933 | 2.101 / 1.964 | 0.909 | 0.366 | 2.916 / 2.821 |

For wide Scalar causal zero-bias and Z-RBF, pooled resident ratios are 0.943 and
0.954 (slower). For wide Register16 unmasked zero-bias and Z-RBF, host-to-host
ratios are 0.931 and 0.965. Within-round observations vary: both wide Scalar
causal cases improve in four rounds and regress in two. Those paired observations
do not erase the pooled result or identify a thermal/driver cause.

### Post-Hoc Wide-Shape Sensitivity

`wide-diagnostic.json` repeats all six wide conditions with the same binaries,
but five engines (four ST routes plus CPU), five rounds, 100 warmups, 21 samples
and order seed 73. This changes several factors, so it neither isolates warmup
as a cause nor replaces the main endpoint. Regenerate its fixture by retaining
only the `wide_prefill` scenario from the benchmark fixture; run the same command
with `--devices cpu --rounds 5 --warmup 100 --samples 21 --order-seed 73`.
The recorded subset uses this JSON serialization (otherwise equivalent serializers
may produce a different file hash):

```bash
node -e 'const fs=require("node:fs"); const f=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); f.scenarios=f.scenarios.filter(s=>s.name==="wide_prefill"); fs.writeFileSync(process.argv[2],JSON.stringify(f,null,2)+"\n",{flag:"wx"});' "$FIXTURE" "$WIDE_FIXTURE"
```

The main run's slower Scalar causal cases did not reproduce: zero-bias was
3.207 -> 3.127 ms and Z-RBF 3.176 -> 3.125 ms resident. The slower Register16
unmasked H2H observations also did not reproduce, but new small resident
reversals remain: Scalar/unmasked/zero-bias ratio 0.974 and
Register16/causal/zero-bias ratio 0.997. Register16 causal Z-RBF was only
2.334 -> 2.323 ms resident in this follow-up, not the main run's larger gain.

Thus wide-input latency is near-flat to modestly improved in this diagnostic,
with residual reversals and appreciably different absolute timing in both
binaries. There is no general no-regression or stable wide-shape speedup claim.
The implementation is retained for direct-view execution, eliminated temporary
materializations, demonstrated short-input gains and preserved correctness;
there is no shape heuristic added just to hide the slower observations.

## Correctness

- 71 native resident-tensor tests passed, including 14 attention tests. Coverage
  includes independent batch/head broadcasts, nonunit column strides, offsets,
  both key-tile regimes, tails, causal offsets, ownership, masked/cropped failed
  inputs and failure propagation through empty queries.
- Rust descriptor size/member offsets match the WGSL type layout. Required
  storage bounds and u32 address rejection are tested without GPU allocation.
- 15 NN attention tests and four full-chain integration test functions passed.
  The latter checks 36 outputs across Scalar, Register8 and Register16 against
  the frozen PyTorch reference, including unequal input/output widths.
- Browser WebGPU passed 40 standalone cases (20 reference cases in canonical
  and reversed/padded strided layouts) and 90 full-chain cases, plus 15 geometry
  checks. Maximum errors were 8.940696716308594e-8 standalone and
  1.7881393432617188e-7 full-chain. Browser validation is correctness-only; the
  browser report does not identify the physical GPU.
- Pinned formatting and strict backend library/test Clippy passed. Vendored WGPU
  emits pre-existing warnings; this is not a warning-free-workspace claim.

All compared outputs must be finite and within `3e-6 + 3e-5 * abs(expected)`
of the PyTorch 2.12.1 CPU math oracle. Timed PyTorch uses its default eager SDPA,
one CPU thread, disabled MPS fallback/fast-math and disabled optional global
Spiralton patches. ST retains runtime guard kernels; Torch does not run an
equivalent guard implementation, so costs are not identical.

## Reproduction And Limits

Before: `34cd278d46d2ac014df2730a2b7b0526ec52e802` (the prior projection study's
frozen executable). After: `e70f7cc80d39e9faec7648782ebba9d0dffb9613`.
The benchmark/fixture code is unchanged between these revisions. Build the
release example with Rust 1.98.0 and preserve separate executables. Use the
environment settings in the [benchmark instructions](../../../docs/resident_zspace_attention.md#bounded-performance-comparison),
then register each executable twice:

```bash
python3 -I tools/bench_attention_chain_vs_torch.py --fixture "$FIXTURE" \
  --native "before_scalar=$BEFORE" --native-projection before_scalar=scalar \
  --native "after_scalar=$AFTER" --native-projection after_scalar=scalar \
  --native "before_register16=$BEFORE" --native-projection before_register16=register16 \
  --native "after_register16=$AFTER" --native-projection after_register16=register16 \
  --devices cpu mps --rounds 6 --samples 9 --warmup 50 --burst 4 --output "$NEW_RESULT"
```

`comparison.json` and `wide-diagnostic.json` preserve all timings/conditions;
`browser.json` contains the
actual rendered WASM reports. `provenance.json` binds source revisions, source
files, fixtures, executables and WASM bundles. Executables, generated fixtures
and full validation logs remain local. No weights or large binaries are added
to Git. Run `shasum -a 256 -c SHA256SUMS` here to verify published records.

Single Apple M4, synthetic deterministic inputs, correlated rounds and no clock
isolation: this is not evidence for all shapes/devices, native-vs-browser speed,
CUDA, training benefit or complete LLM throughput. Backward, uncertainty output,
KV-cache ownership and a complete decoder remain separate work. Earlier frozen
studies, including their negative/noisy observations, are not rewritten.
