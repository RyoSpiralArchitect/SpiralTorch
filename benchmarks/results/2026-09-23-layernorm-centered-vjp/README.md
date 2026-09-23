# Centered LayerNorm VJP, 2026-09-23

Measured source: `4239ea1da448c5c97d9319598d73f955c12a42cc`, clean and unchanged
through all 13 accepted stages. This is a correctness-first candidate, **not a
performance admission or a default-route change**. The earlier normalized-tape
[candidate](../2026-09-23-resident-layernorm/README.md) remains historical.

## Correctness and Review

- 202 native WGPU backend tests, 31 contract tests, and the existing 11
  LayerNorm autograd plus 4 numerical tests passed (248 total).
- Native/WASM strict clippy, formatting and three Python evidence tests passed.
- Real Chrome 153 / Rust BrowserWebGpu passed 7 boundary cases with all 8 masks
  (including none), 5 scale-nullspace cases, 2 tiny-epsilon cases, 20 signed
  seed-scale/permutation variants, and 4 guard checks. Gradient presence is
  checked before values, so an unwanted VJP cannot pass the fixture.
- The unchanged 400-step browser learning fixture went from `1.98799634` to
  `7.81713472e-10`, without intermediate host readbacks. Native retains both
  the convergent control and the original slow-convergence negative control.
- Review P1's second input VJP is now `0.0002187904`. The original centered-f64
  check (`0.00022702335`) passes its unchanged `2e-5 * (1 + abs(expected))`
  bound. An independent 100-digit Decimal calculation on exact f32 inputs
  gives `0.000218790399518444...`; scaled/permuted variants use that oracle.

The fix retains **centered values**, raw variance and inverse standard deviation,
then cancels variance/covariance products before division. Epsilon is added
after cancellation. Three significand components fit the existing 16-byte
per-element tape; row statistics use 32 bytes. This is not IEEE f64 emulation
or a proof over all inputs.

Failed scale-nullspace, tiny-epsilon and dynamic-range probes are retained,
including the unsuccessful division-only refinement. Source patches and full
logs remain local and hash-addressed by `raw-manifest.json` (111 files).
The original local PyTorch MPS masked-backward failure remains documented in
the historical candidate; this comparison requests all three VJPs and does not
silently apply the different-work forced-input workaround.

## Descriptive Timings

Six fixed shapes, five routes, three warmups, eighteen intervals per route:
540 intervals, all numerical gates passed. Each interval includes input
materialization, forward, all VJPs, scale 0.5 and four owning host outputs.
Values are medians in milliseconds on the shared M4, not isolated multi-round
evidence. Rust direct VJP and Python/ATen autograd have different host overheads.

| Shape | Rust CPU | Existing hybrid WGPU | Resident WGPU | PyTorch CPU | PyTorch MPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2x3 | 0.003 | 5.799 | 5.512 | 0.089 | 2.560 |
| 8x257 | 0.022 | 5.687 | 7.990 | 0.083 | 2.081 |
| 32x256 | 0.070 | 6.334 | 9.510 | 0.090 | 1.703 |
| 64x768 | 0.398 | 8.055 | 31.673 | 0.156 | 1.888 |
| 128x1025 | 1.043 | 11.666 | 70.746 | 0.320 | 2.200 |
| 256x256 | 0.529 | 8.456 | 46.852 | 0.186 | 2.553 |

**The precision fix is expensive.** Both CPU routes and MPS win every shape;
resident is slower than the old hybrid in five of six shapes. This is a
retained negative performance result, not a claimed optimization. Next work
must reduce arithmetic cost without relaxing the new numerical gates.

## Replay and Verification

See the [protocol](../../layer-norm-resident/README.md). `validation.json`
records exact stage commands; `measured-source.json` records source hashes.
Browser module/page hashes are included; raw binaries are not published.

```sh
python3 -I -B benchmarks/layer-norm-resident/test_evidence.py
python3 -I -B benchmarks/layer-norm-resident/evidence.py verify-public \
  benchmarks/results/2026-09-23-layernorm-centered-vjp
python3 -I -B benchmarks/layer-norm-resident/decimal_probe.py
```

With the private raw directory available, use `verify RAW PUBLIC` to check all
111 raw hashes too. The public payload manifest excludes this Git-versioned
README so explanatory updates cannot rewrite measured payloads.
