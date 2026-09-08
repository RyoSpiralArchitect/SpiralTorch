# Resident Training Throughput, 2026-09-08

Bounded native Metal / browser WebGPU measurements of the shared Rust
Linear/GELU, mean-MSE, plain-SGD workload against the prior implementation and
eager PyTorch MPS. Read the
[method, limitations and observed ratios](../../../docs/resident_nn_training_benchmarks.md)
before interpreting the numbers.

- `summary.json` is the adopted-source read-only revalidation, including every
  recipe, median/min/max and captured-state errors. It makes no quality claim.
- `raw.tar.xz.part-*` preserves all completed reports, the original browser timeout,
  both unconditional browser runs, the adopted runs, progress/case streams,
  build logs, reference replay and ten final check logs. No executable, wheel,
  WASM binary, external model, credentials or training corpus is included. The
  lossless XZ archive is split into ordered parts below GitHub's per-file limit.
- `manifest.json` binds the archive's individual inputs and the separately
  retained native/WASM products by SHA-256. Embedded clean source identities
  distinguish baseline `b2a2eb0b`, experimental `e8db3b84`, and adopted `5062aaa7`.
- `SHA256SUMS` verifies the public summary, manifest and compressed raw archive.
  Inside the archive, `RAW_SHA256SUMS` verifies the byte-exact raw members.

The adopted path coalesces native Metal passes only. Browser and unmeasured
backends retain their earlier dispatch cadence. In particular, the noisy
unchanged-path browser comparison is **not** a claimed browser acceleration.
The largest native shape still loses to PyTorch MPS. Other Metal GPUs and CUDA
were not measured; browser metadata is a separate adapter observation, not an
exact physical-device receipt. No new PyPI release was made.

```bash
shasum -a 256 -c SHA256SUMS
mkdir /tmp/new-training-evidence
cat raw.tar.xz.part-* | tar -xJf - -C /tmp/new-training-evidence
cd /tmp/new-training-evidence
shasum -a 256 -c RAW_SHA256SUMS
```

The compact validator can also be rerun without PyTorch/GPU on the extracted
`adopted-*` reports using the source commits listed in the method document.
The original absolute local paths in receipts are provenance, not required
destinations for replay. The separate correctness replay requires PyTorch.
