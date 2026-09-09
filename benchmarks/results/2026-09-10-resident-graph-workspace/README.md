# Graph-Owned Pointwise Workspace

## Change And Sources

The graph compiler now prepares pointwise forward/VJP bind groups, contribution
buffers and unbroadcast partials once per graph, instead of allocating them on
every step. Shaders, reduction order, finite checks and transactional SGD are
unchanged. Standalone pointwise calls keep independent mutable workspaces and
immutable results; no global pool or per-plan shared scratch was introduced.

- Baseline: `b13d5bdf556c1e1c436692b53e165bd947103e8e` (benchmark-only extension of main).
- Candidate: `3b95f84012f3d7e25d89eba98936338458d0a071`.
- Both clean sources were built with Rust 1.98.0, release, `st-nn`, no default
  features, `wgpu`. Native products embed their source identity. Browser fixture
  manifests and served JS/WASM hashes are retained. Benchmark code is identical
  across baseline/candidate. Publication is a later, evidence-only commit.

## Measurements

Nine fixed cases: three seeds (17, 29, 43), shapes/depths `[2,16,32]/2`,
`[2,129,32]/4`, `[4,32,64]/8`. Mixed Linear/tanh-GELU/gain/ReLU graphs use exact
mean-MSE gradients and plain SGD. Every timed interval performs eight real updates
and reads all eight losses. Immediate reads after each step; deferred captures
owning snapshots and reads them after enqueue. Setup, uploads, resets, zero-rate
probes and final full-state serialization are excluded. No GPU-only timing claim.

Lane order rotates; two warmup blocks precede eight retained blocks per cadence.
All 540 Native/Torch and 360 browser intervals are retained, without selective
retries. The table is the geometric mean of nine **per-case median ratios**,
baseline time / candidate time; greater than one favors the candidate.

| Client | Loss Cadence | Ratio | Case Range | Faster Cases |
| --- | --- | ---: | ---: | ---: |
| Native Metal | Immediate | 1.110 | 0.976-1.308 | 8/9 |
| Native Metal | Deferred | 1.044 | 0.950-1.273 | 6/9 |
| Browser WebGPU | Immediate | 1.235 | 1.150-1.482 | 9/9 |
| Browser WebGPU | Deferred | 1.187 | 0.994-1.806 | 8/9 |

Native runs used Apple M4 Metal and eager PyTorch 2.12.1/MPS, fallback explicitly
disabled. **PyTorch remains faster**: Torch/candidate geometric time ratios were
0.674 (immediate) and 0.678 (deferred), equivalent to about 1.48x/1.47x faster.
Torch lacks matching per-stage finite/atomic-rollback checks, and its deferred
losses use one stacked copy instead of Rust's individual snapshot reads. This is
not a comparison against the fastest compiled Torch implementation.

Chrome 152.0.7977.83 executed the Rust WASM fixtures through BrowserWebGpu. A
separate navigator probe reported Apple/metal-3/non-fallback; the Rust workspaces'
physical adapter identity is unavailable. Do not treat the probe as exact device
attestation or compare browser wall time directly to native PyTorch. Ambient GPU
contention is unknown on macOS. These are bounded local observations, not a
significance claim; small regressions and all outliers are retained.

## Correctness And Checks

- All benchmark captures match the recorded independent Torch oracle; maximum
  absolute error across native/browser comparisons: `1.4901161193847656e-8`.
- Separate training fixture: 24 Native/browser vs CPU/MPS replays, 5,184 checks,
  maximum absolute error `1.1920928955078125e-7`.
- Per client, three shapes interleave two graphs for 12 steps each, change batches,
  and read 72 retained snapshots only after graph drop. The fixture also reuses
  workspaces after numerical rejection without partial parameter commits.
- Standalone N-D/VJP regressions remain valid: 26 matched checks, plus two retained
  pre-existing Torch MPS empty-tensor reference gaps. No silent CPU fallback.
- Backend 110 tests, NN 733 library tests, four real-GPU integration tests and nine
  Python benchmark admission tests passed. Backend strict Clippy passed on native
  and wasm32; NN Clippy passed with existing warnings. Initial Clippy rejection of
  a large enum is preserved; its boxed representation was fixed before measurement.

`summary.json` contains derived ratios; `raw/bench-validation.json.xz` contains
every reaggregated case. Raw reports, build/test logs, commands, hashes and the
optimization diff are losslessly XZ-archived under `raw/`. `manifest.json` records
compressed and uncompressed hashes; `SHA256SUMS` verifies all generated evidence.

The implementation still uses explicit graph training, not automatic resident
`pure::Tensor` storage or production Python/JS graph wrappers. Scratch remains
allocated until graph drop. Command encoders, queue staging and requested
snapshots can still allocate. CUDA was not measured in this slice.

## Reproduce

Build `resident_training_bench` and `resident_training_bench_browser` from each
clean source and preserve separate products (including wasm-bindgen output).
Then run the current source's harnesses, using new output paths:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/bench_resident_training_vs_torch.py \
  --graph --device mps --baseline /absolute/baseline --baseline-source b13d5bdf \
  --candidate /absolute/candidate --candidate-source 3b95f840 --output native.json
node tools/bench_resident_training_browser.cjs /absolute/baseline-module \
  /absolute/candidate-module /absolute/chrome /absolute/browser.json graph
python -I tools/validate_resident_training_bench.py --native native.json \
  --browser browser.json --browser-progress browser.json.progress.jsonl \
  --baseline-source b13d5bdf --candidate-source 3b95f840 \
  --browser-harness-source 3b95f840 --output validation.json
```
