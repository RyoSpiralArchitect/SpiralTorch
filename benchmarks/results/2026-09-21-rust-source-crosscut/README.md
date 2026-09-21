# Rust Source Crosscut: CPU Forward Comparison

## Sources And Scope

- Inventory base: `7f64e24a8bcccc1464faea85c1eae878e51041ca` (35 packages, 693 src files).
- Baseline with identical benchmark harness and failing regression tests:
  `3f1581c81d9be0f4403850e91440c333f0404239`.
- Candidate: `bf8cc8eabdf011203d065215d163ac618928de74`.
- Apple M4, 24 GiB, macOS 26.4.1, Rust 1.98.0 release; PyTorch 2.12.1 CPU.
- Rust serial reduction / one Rayon thread; no autotune store path, so both
  revisions use the same default CPU kernel. PyTorch intra/inter-op threads are
  both one, verified after configuration. Its startup bypasses sitecustomize.
- Host exclusivity and thermal state are unknown. This is a small CPU forward
  microbenchmark, not a resident GPU, autograd, FT-quality, or fastest-PyTorch claim.

See [the audit](../../../docs/development/rust_source_crosscut.md) for implementation
boundaries, confirmed defects, and the next candidates. The inventory JSON is
triage, not evidence that all source files received a full manual audit.

## Fixed Conditions

36 conditions, two AB/BA rounds, three warmups, nine retained intervals per
condition per process, no trimming or selection of the better round. Each
InfoNCE interval contains 32 calls; each fractal interval contains four calls.
All 1,296 Rust intervals remain in the four result files.

- InfoNCE: `(batch, features)` = `(8,31)`, `(32,64)`, `(96,128)`; normalization off/on;
  vector API, Tensor API (row/column-major), Tensor-to-vector adapter (row/column-major).
  LCG fixtures use seeds 17/29, temperature is float32 0.3.
- Fractal weave: lengths 16/4096/65536; `(octaves, iterations)` = `(1,1)` / `(4,16)`.
- Inputs are prepared outside timing; the complete public forward API, output
  allocation and destruction are inside. A disabled counting-allocator branch
  is present in both native timed binaries; a separate warmed call counts Rust
  allocation requests and requested bytes, not peak live or GPU memory.
- The PyTorch eager comparator uses the same float64 norm accumulation / float32
  dot and post-normalization objective, prebuilt float32 inputs, scalar loss,
  logits and labels. It is not a search over faster alternative PyTorch formulations.

## Results

All **72 candidate condition-runs** pass numerical checks. InfoNCE is compared
with an independent float64 objective at `atol=rtol=1e-4`; maximum logit absolute
error is `2.1060543073048166e-5`. All 12 fractal condition-runs match the separate
base-plus-branch calculation bit for bit. PyTorch passes the same six objective
conditions. Raw worker outputs are validations/timings, not full Rust tensor dumps;
numerical replay requires rerunning the frozen recipes.

The baseline fails 12 column-major Tensor condition-runs. Their timings are kept
but **excluded from speedup claims** because the old outputs are wrong.

The following are geometric means of baseline/candidate interval-median ratios
across both rounds. Greater than one favors the candidate; these are speed ratios,
not elapsed-time reduction percentages.

| Path | Ratio | Full Range | Favorable / Comparisons |
| --- | ---: | ---: | ---: |
| InfoNCE vector | 1.089 | 0.958-2.787 | 6/12 |
| InfoNCE Tensor, row-major | 1.233 | 0.661-2.414 | 8/12 |
| Tensor-to-vector, row-major | 1.163 | 0.991-1.505 | 11/12 |
| Tensor-to-vector, column-major | 1.371 | 1.122-2.157 | 12/12 |
| Fractal weave | 1.070 | 0.943-1.412 | 9/12 |

There is **no universal speedup**. Small-batch Tensor-returning calls regress in
some conditions. For normalized 96x128 row-major inputs, the adapter reduces
allocation requests **209 -> 16** and bytes **278,528 -> 126,544**, but the
Tensor-returning API increases requests **10 -> 23** and bytes **75,424 -> 164,336**
because it now uses the shared prepared/blocked path rather than its own scalar
loop. This tradeoff is a follow-up target, not hidden behind the aggregate speed.

For a 65,536-sample fractal grid, requests drop **4 -> 3** and bytes
**1,835,008 -> 1,310,720**, saving exactly **512 KiB of allocation requests per call**.
Both generator configurations show the same saving; this is not a peak-memory claim.

The matched PyTorch comparison favors SpiralTorch on small fixtures but not on
the largest fixture. For the row-major Tensor API, Torch/native time ratios
geometrically average 14.78 (batch 8), 1.71 (batch 32), and **0.275 (batch 96)**:
PyTorch is about 3.63x faster in the largest group. Per-condition results and
all intervals are in `torch.json` and `comparison.json`.

## Verification And Retained Failures

`verification/receipt.json` records 13 successful stages: 243 CPU tests (three
pre-existing ignored cases), scoped strict native Clippy, seven WGPU-feature tests
including a required live dense GPU probe, frac WASM strict Clippy, logic and
binding WASM checks, real WASM build/numerical smoke, three real Python-extension
tests (no skips), and the final benchmark build. These are distinct from the CPU
performance measurements; WASM smoke is not a browser speed measurement.

`preflight/` preserves the five reproduced baseline failures, the failed
dependency-inclusive Clippy attempt, and the old test lints corrected without
allowances. The successful native lint boundary is `--all-targets --no-deps` for
the changed packages, not a claim that every dependency passes strict Clippy.

`preflight/measurement-a/` retains the complete initial Rust runs and the PyTorch
setup failure. Those runs used unfixed Rayon/autotune settings and are not admitted
to the comparison. The full unchanged grid was rerun after correcting isolation;
neither global Python settings nor production Rust code was changed for that retry.

`recovery.json` separately records restoration of two required older archive parts,
96 MiB total, matching their Git blob identities. Byte restoration is not numerical
replay. No caches were restored.

## Verify Or Rerun

Byte/schema verification needs only Python's standard library:

```bash
python3 -B -I benchmarks/results/2026-09-21-rust-source-crosscut/verify.py
python3 -B -I benchmarks/results/2026-09-21-rust-source-crosscut/test_verify.py
```

For runtime replay, use separate checkouts at the two commits above and build the
same harness in each. Keep the two executables as `baseline-worker` and
`candidate-worker` in a fresh output directory (executables stay local).

```bash
cargo +1.98.0 build --locked --release -p st-bench --example source_crosscut
```

Then run the archived driver with an explicit directory containing Torch. `-I -S`
loads that package path without executing global startup customization:

```bash
python3 -B -I benchmarks/results/2026-09-21-rust-source-crosscut/measure.py measure \
  --directory /absolute/path/to/fresh-output \
  --torch-python /absolute/path/to/python \
  --torch-site /absolute/path/to/site-packages
```

The driver refuses to overwrite result files. `provenance.json` pins source/harness
identity; `measurement-receipt.json` pins binary/result hashes and actual settings.
The local executable/extension paths are retention pointers, not portable download
URLs. `manifest.json` binds the published files; its verifier checks bytes, condition
coverage, and recorded gates, **not GPU or Rust numerical replay**.
