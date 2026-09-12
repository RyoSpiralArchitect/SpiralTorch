# Direct VJP: Repeated Larger-Shape Stratum

The repeated wider comparison passes the saved-state numerical checks, but
**does not establish a uniform speedup**. Eager Torch MPS is faster in every
native case/cadence. This is additional evidence, not a replacement for the
mixed and slower cases in `../2026-09-12-resident-direct-vjp`.

## Results

- Baseline: `5fddb415bfbb0f64366db6d45b7675db17a16f61` (checked-copy pool).
- Candidate: `23542255c9a988d476a4f60a6a9ba416973897d5` (direct owning VJP outputs).
- Harness: `d0a64825fcee6dcfc956b76478e5b92615974cf0`; the seven benchmark/reference helpers are byte-identical in all three sources.
- Shapes/depths: `[4,64,64]/8`, `[2,128,128]/8`, `[2,64,256]/4`; seeds 17, 29, 43.
- Plain, Topos EMA and clipped Topos EMA; two full runs, reversed optimizer order only. No selective retry.
- 54 optimizer recipes, 4,320 retained intervals, eight updates per interval, two warmups and eight retained intervals per lane/cadence. All six validation reports are included.
- Maximum saved-state absolute difference against the reference: `7.916241884231567e-8`.
- Torch/candidate time ratios range from `0.15785555585650898` to `0.7791463065120506`; smaller favors Torch. The comparator does not implement equivalent finite-stage guards and transactional rollback.

`summary.json` contains every case and all range/median/geomean aggregations.
Its `baseline_over_candidate` ratios above 1 favor direct outputs. Case ranges
are not confidence intervals. Immediate/deferred means acceptance-receipt
observation, not gradient readback. Full-state reads and initialization are
outside measured intervals.

Native: Apple M4, Metal; Torch 2.12.1 MPS, CPU fallback disabled. Browser:
Chrome 152.0.7977.84, verified WebGPU execution; the Rust workspace's physical
GPU identity remains UNKNOWN. A separate navigator apple/metal-3 observation
does not resolve it. Host exclusivity is UNKNOWN; owned GPU runs were serial.
No phase attribution, fastest-Torch, CUDA, FT-quality or optimizer-resume claim.

## Publication Boundary

At the owner's request, this new stratum publishes **results, validation
records, source/product hashes and the rerun recipe**, not the roughly 10 GiB
of raw tensor payloads. `raw-inventory.json` binds every original measurement
file to its local path, byte count and SHA-256. All originals remain local.
Hashes alone do not permit a reader to independently replay absent payloads.
Earlier already-committed evidence archives remain unchanged.

`verify.py` checks published bytes, source bindings, all 54 saved conditions and
their aggregation. With the original raw directory it additionally checks raw
bytes; neither mode reruns GPU computations or numerical reference validation.

```sh
python3 verify.py
python3 verify.py --raw-root /path/to/measurement-a
```

## Reproduction

Use clean worktrees at the baseline, candidate and harness commits above, not
an installed wheel or a mutable shared target binary. `harness/measure.py`
preserves the exact original driver and its original machine paths;
`run/receipt.json` preserves all 18 actual command argument arrays. Change local
paths in a new copy, not in the frozen record. New builds/runs get new hashes
and separate output directories; they do not reproduce the original bytes.

For each worker, use Rust 1.98.0 and wasm-bindgen 0.2.104, with a separate target
directory. Build the native and browser examples from that worker's clean tree:

```sh
cargo +1.98.0 build --locked --release -p st-nn --no-default-features --features wgpu --example resident_training_bench
cargo +1.98.0 build --locked --release -p st-nn --no-default-features --features wgpu --example resident_training_bench_browser --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_training_bench_browser.wasm --target web --out-dir /path/to/frozen-worker/wasm --out-name spiraltorch_wasm
```

Retain the native binary and browser package before building the next worker;
check the native `--build-info` commit/tree and browser package hashes. From the
harness worktree, run `tools/bench_resident_training_vs_torch.py` with
`--device mps --graph --learner --matrix wide`, explicit worker paths and source
commits, with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Add `--learner-optimizer topos_ema`
or `clipped_topos_ema` for those modes. Then run the browser collector and
`tools/validate_resident_training_bench.py` with the corresponding native,
browser and progress files. Use the exact positional/optional arguments in
`run/receipt.json`; repeat every mode in the recorded reversed order.

## Publication Review

`review/review-c` records 32 successful source-bound steps at `8d2bd26a`:
769 native NN GPU tests, strict checks, three real GPU guard tests, and 82
Python tests per build (GPU: no skips; CPU-only: 22 expected GPU skips).
Both frozen CPU/GPU builds additionally pass the 20 runtime-import tests after
the type-stub-only fix. Earlier failed review attempts and the failing CI log
are retained. The bare local Python attempts lack the compiled Rust route and
are not accepted validation; the `stub-after-frozen-*` logs use explicit frozen
extensions. A type-only follow-up does not retroactively change benchmark
worker identity. GitHub CI and review status are reported on the code PR.
