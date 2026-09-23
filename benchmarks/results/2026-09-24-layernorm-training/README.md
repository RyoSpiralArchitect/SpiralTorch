# Fixed-source resident LayerNorm training results

Measured from clean source commit
`b9b23501123abfa7179bc969cdd0291c3cddb5d0` on Apple M4 integrated GPU
(Metal), macOS 26.4.1 (25E253), PyTorch 2.12.1 eager CPU/MPS with MPS fallback
disabled and four intra-op CPU threads. Rust used the release profile and four
Cargo build jobs. This was a shared-machine exploratory run, not an isolated
hardware admission test. Full raw reports remain in the local archive; the
three `comparison-run*.json` files here retain all six shapes, all route and
stage medians, validation status, final losses, and raw-report SHA-256 hashes.

Each cell below is the median of three independent run medians (nine measured
32-step intervals per route per run), in milliseconds. `H2H` includes input
upload; `resident` starts with resident input/target/parameters. `affine` omits
dx and is not work-matched to the all-gradient columns.

| Shape | Rust CPU all | WGPU H2H all | WGPU resident all | WGPU resident affine | PyTorch CPU all | PyTorch MPS all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x3 | 0.033 | 26.393 | 26.660 | 24.593 | 2.121 | 6.754 |
| 8x257 | 0.636 | 42.447 | 42.843 | 25.528 | 2.179 | 6.802 |
| 32x256 | 2.307 | 51.002 | 50.545 | 30.017 | 2.313 | 7.178 |
| 64x768 | 13.379 | 207.714 | 207.382 | 116.363 | 4.230 | 7.910 |
| 128x1025 | 35.950 | 530.122 | 529.850 | 273.780 | 5.321 | 7.265 |
| 256x256 | 17.791 | 267.729 | 267.173 | 130.854 | 4.636 | 7.270 |

Separate stage probes, also median of three run medians in milliseconds, each
include a terminal snapshot. They must not be summed into a 32-step estimate.

| Shape | Readback only | Forward | MSE | Backward all | Backward dx | Backward affine | Update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x3 | 0.160 | 0.606 | 0.370 | 0.591 | 0.514 | 0.352 | 0.455 |
| 8x257 | 0.169 | 0.603 | 0.409 | 1.060 | 0.683 | 0.658 | 0.482 |
| 32x256 | 0.176 | 0.700 | 0.379 | 1.220 | 0.792 | 0.793 | 0.477 |
| 64x768 | 0.223 | 2.164 | 0.438 | 5.083 | 2.881 | 2.506 | 0.519 |
| 128x1025 | 0.305 | 5.196 | 0.542 | 11.726 | 7.956 | 4.636 | 0.620 |
| 256x256 | 0.238 | 3.302 | 0.481 | 4.463 | 3.277 | 1.907 | 0.530 |

All 3 comparisons passed: 648 Rust training intervals, 1,134 stage intervals,
and 324 PyTorch intervals, with no missing shape or route. Initial losses
were 1.09-1.24 and 32nd-step pre-update losses were 0.068-0.091 on every
route. The largest scaled numerical error across native and cross-runtime
final values was 0.000272 of the allowed `5e-4 * (1 + abs(reference))` bound.
No GPU fallback or CPU adapter was accepted. The raw input/target f32 digests
matched exactly.

The negative performance result is the point: simply retaining tensors on the
GPU barely changes the all-gradient time here. On the 128x1025 case, WGPU
resident all is about 530 ms for 32 steps versus about 530 ms H2H, while the
forward and input VJP probes cost about 5.2 and 8.0 ms individually. Skipping
unused dx roughly halves the WGPU run but does not make it work-matched to
PyTorch's all-gradient path or competitive with it. The next optimization
target is resident LayerNorm forward/VJP work and its dispatch structure,
not a default-route switch or a claim that removing transfer solves training.

Replay commands and scope are in `benchmarks/layer-norm-training/README.md`.
Each comparison JSON includes the exact raw-report SHA-256 hashes. Public
comparison-file SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `comparison-run1.json` | `75ff93ab9bb1d062afe0542a9871a5770918d0aefa0d9b8ccfd4febf023d6a96` |
| `comparison-run2.json` | `eceb2512afaca57afef7acbebc41ec8d9dc1d043faedf763cf986c89f8ded62e` |
| `comparison-run3.json` | `3dfce2d811cef90a760b30edd2f387d5c54de0af725a0b6f2915728c9d05a9c8` |
