# Golden Portable Reduction

The signed-vector mean formerly embedded in GoldenRetriever is now a shared
`st-tensor` operator, callable from Rust, Python, and real browser/Node WASM.
Input-order f64 accumulation and Golden's f32 guard addition are preserved.
The operation remains CPU/WASM CPU; it does not execute Golden's rank plan.
See the [API and benchmark guide](../../../docs/performance/tensor_mean.md).

## Results

Ratios below are candidate time / reference time; lower is better. Ranges are
the minima/maxima of nine shape/count group medians across three seeds. Each
seed ratio uses the mean of 16 retained samples after three warmups. Native
legacy/candidate arms alternate; Mac Python Torch/SpiralTorch arms alternate.

| Measurement | Ratio range |
| --- | ---: |
| Apple M4 native, row-major, vs legacy body (run + confirmed repeat) | 0.428-0.656 |
| Apple M4 native, mixed column-major, vs legacy body | 0.188-0.412 |
| Furnace native, row-major, vs legacy body (two runs) | 0.800-0.916 |
| Furnace native, mixed column-major, vs legacy body | 0.354-0.803 |
| Portable browser WASM CPU / ordered JavaScript reference | 0.529-0.700 |
| Mac Python SpiralTorch / ordered PyTorch CPU | 0.187-0.648 |
| Furnace native CPU / ordered PyTorch CPU | 0.218-1.048 |

Furnace is an Intel Core Ultra 9 285K with RTX 5090. Native CPU and Torch there
are separate same-host runs, not interleaved arms. Torch CPU uses one thread;
versions are 2.12.1 on Mac and 2.13.0+cu132 on Furnace. Torch inputs are validated
outside timing; SpiralTorch validates them inside each call. Outputs are new
allocations. The three shapes are 1x1025, 32x2048, 128x2048, with 4/16/64 partials
and seeds 17/29/43. All ordinary fixtures and adversarial tests check output bits.

CUDA is useful, not beaten generally: native CPU / PyTorch CUDA ranges from
0.049-0.057 for the smallest shape, 2.253-3.323 for 32x2048, and 5.639-35.261 for
128x2048. These are synchronized host-clock measurements with transfers excluded,
not device-event timings or a SpiralTorch CUDA implementation. CUDA's 27 cases
passed the ordered-f64 numerical reference; 55 GPU availability point checks saw
no foreign compute PID. They are not continuous contention monitoring.

## Iterations And Gates

- The first implementation scanned finite inputs separately, slowing row-major
  cases. Fusing validation and reducing a block validity mask removed that cost.
- Direct strided reads regressed large mixed-layout cases on Furnace. 16x64
  output tiles removed that regression without replacing the row-major fast path.
- Accumulation scratch is bounded to 8 KiB, plus output and input handles.
  Row/column-major inputs no longer require full layout-conversion buffers.
- 433 tensor unit tests, two tensor integration tests, 17 Golden tests, two
  native WASM transport tests, 19 Python tests, Node WASM/type tests, and 27
  browser cases passed. Tensor strict Clippy and workspace rustfmt passed.
- Strict `-D warnings` on whole st-nn/WASM crates still reports 23/19 diagnostics.
  The baseline produces the identical messages and locations; these were not
  suppressed or folded into this numerical change. CI reuses existing jobs.
- Invalid Chimera test geometry, a browser fixture syntax error, and a Python
  shape-method test typo were corrected. Their failure logs remain. The first
  native exploratory run overlapped a Torch import; use the confirmed runs.

The compiled library candidate is `2928e357463ca3909dc87c08d52c4f5d464cf487`.
Later changes are fixtures, CI, metadata, and documentation. Immutable native
images were hash-checked before/after runs; the installed extension matched the
captured wheel. Browser results bind frozen generated asset hashes. Regenerating
web glue changed helper numbering/output hashes, so this is not a byte-reproducible
build claim; Node is a separately generated product. No PyPI release was made.

## Replay

`raw-logs.tar.xz` preserves 91 log/fixture/source-bundle/metadata files, including
failed candidates. Executables, wheels, environments and generated modules are
not embedded; their hashes and build provenance are retained. Summary replay
requires only Python's standard library, not Torch, a GPU, or a browser.

```sh
mkdir NEW_REPLAY
tar -xJf raw-logs.tar.xz -C NEW_REPLAY
python3 -I NEW_REPLAY/analyze.py --output NEW_REPLAY/recomputed.json
cmp summary.json NEW_REPLAY/recomputed.json
```

SHA-256: summary `1ba04536ba6b12def322443acf0208b4ce08203ba684bf63906bef45c7dce30f`;
archive `ef9af741624951b40685dd2dc9641a70573d14687a7f44c2527394df296cda40`.
These component measurements do not establish model quality, training speed,
or general framework superiority. The largest workload still strongly favors
PyTorch CUDA over this CPU reducer.
