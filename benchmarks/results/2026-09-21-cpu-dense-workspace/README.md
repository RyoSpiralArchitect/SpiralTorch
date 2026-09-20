# CPU Dense Workspace Follow-up

This follows the [cross-crate first pass](../2026-09-21-rust-source-crosscut/README.md).
It is a CPU forward comparison on Apple M4 / macOS 26.4.1, Rust 1.98.0 release,
with a separate PyTorch 2.12.1 CPU comparator. Host exclusivity and thermal state
are unknown. No training, CUDA, resident GPU speed, or browser speed claim is made.

## Implementation And Source Identity

- Small unpacked products use fixed 8/4-column accumulators without heap packing.
  Each output retains the original sequential float32 K-reduction order.
- Serial and single-row-tile blocked products reuse one bounded A-panel buffer
  across RHS blocks, rather than allocating another panel for every block.
- Tensor-returning InfoNCE builds float32 labels directly; it no longer constructs
  a temporary usize label vector. The shared objective and public result types remain.
- Public CPU entry points check dimension multiplication and lengths before writing.
  Empty prepacking returns immediately even with a very large zero-sized shape.
  The changed blocked path invalidates old kernel-tuning records with revision 3.

The selected measured/verified candidate is
`c6c82992` (full identity in `provenance.json`). Integration commit
`04ed74b623efb6d8ea24fddbf1ffea10ed7c5ae4` has the **identical complete Git tree**,
`d3502abb890da0f753dabefe485a1f4900af1bbd`, after undoing an unsuccessful follow-up.
Subsequent evidence/documentation commits do not change that production source.

The dense baseline is `7ead4cb1ce0271b169198895364721cec650c39b`.
The crosscut baseline executable was frozen at
`bf8cc8eabdf011203d065215d163ac618928de74`; its production sources are identical
to the dense baseline. Both benchmark source hashes and all local binary hashes
are recorded. These compare against the first-pass PR version, not directly
against its earlier pre-fix scalar Tensor implementation.

## Measurements

`dense/` retains 15 shapes x packed/unpacked x serial/configured-four-Rayon-thread
modes x two AB/BA rounds: **240 bitwise-valid condition-runs**, including baseline
and candidate. Three warmups and nine timing intervals, each with 16 calls, are
retained without trimming. Input/output allocation and prepacking are outside this
low-level timer. A separate warmed call counts Rust allocation requests and bytes,
not peak memory. Small shapes can still execute serially in the four-thread mode.

Both revisions run without HOME or an autotune-store path, fixing the default
kernel instead of incorporating historical tuning decisions. Serial mode also
locks reduction scheduling. Neither global environment nor tuning policy is changed.

Geometric means of baseline/candidate median-time ratios follow. Greater than one
favors the selected candidate. All shape/round values remain in `dense/comparison.json`.

| Mode | Unpacked | Prepacked |
| --- | ---: | ---: |
| Serial, all 15 shapes | 1.667 | 0.933 |
| Four-thread configuration, all 15 shapes | 1.729 | 0.916 |
| Serial, 13 nonempty shapes only | 1.945 | 0.988 |
| Four-thread configuration, 13 nonempty shapes only | 1.962 | 0.985 |

The nonempty rows are a supplementary breakdown, not a replacement for the full
grid: zero-sized cases make validation overhead visible. **Prepacked execution is
not an across-the-board speedup.** Large products are mostly unchanged in time,
and individual regressions remain. The main reproducible blocked-path change is
allocation reduction: at 96x128x96 in serial mode, unpacked calls fall **9 -> 2**
and requested bytes **38,912 -> 10,240**; prepacked calls fall **8 -> 1** and bytes
**32,768 -> 4,096**. Multi-tile parallel execution still has per-task temporaries.

`crosscut/` reruns the unchanged 36-condition first-pass harness with the same
fixed serial-kernel setup, two AB/BA rounds and nine retained intervals. All **72
candidate condition-runs** pass; all baseline conditions now pass too. InfoNCE
uses an independent float64 objective at atol=rtol=1e-4; fractal weaving is checked
bitwise and is an unchanged control, not a new optimization in this follow-up.

| InfoNCE API | Baseline / Candidate |
| --- | ---: |
| Vector | 1.208 |
| Tensor, row-major | 1.201 |
| Tensor, column-major | 1.152 |
| Tensor-to-vector, row-major | 1.181 |
| Tensor-to-vector, column-major | 1.156 |

Normalized 96x128 Tensor-returning inputs drop from **23 to 15** allocation requests,
and **164,336 to 134,896** requested bytes. The PyTorch comparator passes all six
matched numerical conditions with one intra/inter-op thread and startup customization
disabled. It is still about **3.59x faster in the batch-96 Tensor group**. This is
matched eager CPU forward math, not a fastest-PyTorch or learning-quality claim.

`preflight/v1/` retains the first numerically correct but slower small-product loop.
`preflight/v3/` retains a rejected split-guard implementation: nonempty prepacked
ratios were about 0.913/0.930 versus the selected version's 0.988/0.985. Its source,
complete measurements, and successful numerical verification remain available.
These are implementation experiments, not discarded repetitions of the same binary.

## Validation And Replay

The selected source passed 14 recorded stages: Tensor CPU tests (478 passes),
serial CPU kernel tests (10 passes, including 297 small-width combinations),
self-supervision CPU tests (32 passes, three existing ignored cases), scoped strict
native/WASM Clippy, actual WASM build and numerical smoke, **14 WASM autograd
forward/backward cases**, three real Python-extension tests, seven WGPU-feature
tests with a required live dense GPU probe, and the benchmark build.

Another 26 CPU-only integration tests cover dispatch/autograd without faer. The
prepacked dispatch test is feature-gated out of the faer-enabled suite; the
separate run proves it actually executed. WASM is CPU-only here, with no faer/GPU
Tensor feature. High-level Auto routing is unchanged: testing a small WASM shape
does not claim it selected the new explicit CPU direct path.

Verify archived bytes and recorded gates, without claiming numerical replay:

```bash
python3 -B -I benchmarks/results/2026-09-21-cpu-dense-workspace/verify.py
python3 -B -I benchmarks/results/2026-09-21-cpu-dense-workspace/test_verify.py
```

For numerical/performance replay, build the same examples at the pinned baseline
and selected candidate in separate checkouts. Keep the binaries locally, then run
the archived drivers into fresh output directories:

```bash
cargo +1.98.0 build --locked --release -p st-bench \
  --example cpu_dense_workspace --example source_crosscut
python3 -B -I measure_dense.py measure /absolute/path/to/new-dense-output \
  --baseline /absolute/path/to/baseline-dense --candidate /absolute/path/to/candidate-dense
python3 -B -I measure_crosscut.py measure --directory /absolute/path/to/new-crosscut-output \
  --torch-python /absolute/path/to/python --torch-site /absolute/path/to/site-packages
```

The crosscut directory must already contain the two executables as `baseline-worker`
and `candidate-worker`. Exact validation commands are in `verification/receipt.json`;
the archived validation driver records local tool paths, which must be adapted on
another machine. `python-client.py` explicitly loads a chosen native extension,
not the installed wheel. Native binaries/extensions stay local; this archive
publishes results, validation logs, hashes and replay instructions.

Remaining work: avoid repeated A-panel packing itself, reduce multi-tile parallel
temporaries, and assess high-level Auto routing separately. Do not infer that fewer
allocations close the large-GEMM gap with PyTorch.
