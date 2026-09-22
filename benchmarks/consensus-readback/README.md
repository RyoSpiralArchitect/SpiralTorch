# Consensus and Ordered Readback

Measure two changes independently, on native WGPU and browser WebGPU:

| Route | Entropy/hard-mass reduction | Owning CPU readback |
| --- | --- | --- |
| separate_separate | Original two trees | One buffer at a time |
| paired_separate | Same trees, shared barriers | One buffer at a time |
| separate_batch | Original two trees | Packed snapshot |
| paired_batch | Same trees, shared barriers | Packed snapshot |

Six shapes, two output contracts, bursts 1/4, three warm-ups and nine rotated
blocks per cell, repeated in three serial runtime rounds. This is 6,480 measured
intervals across eight WGPU routes and eager Torch CPU/MPS. Report all cells,
including regressions. Pair mode never dispatches consensus, so its reduction
factor is a placebo. Use both single-factor ratios before attributing a change.

## What Is Timed

Count=2 returns probabilities and **all tied maxima**. Count=4 additionally
returns raw GPU spiral weights and per-row entropy, hard mass, enrichment and
coherence. The same Rust code composes softmax and consensus into one command
buffer. Readback includes copy submission, map completion and an owning CPU
result. Both WGPU readback controls share Rust lease ownership and decoding;
the native separate control is not the old read_buffer function byte for byte.
Legacy native read_buffer correctness is checked separately.

Inputs/outputs are prepared before timing. Encoding, binding creation, one
submission per operation and one terminal observation per burst are included.
Validation, oracle, JSON and Python list conversion are outside timing.
Torch uses reusable output/scratch tensors, eager CPU/MPS without fallback,
four intra-op threads and one inter-op thread. It performs the same scalar
operations, not the same fused kernel. This is not a torch.compile comparison.

The original shader is reconstructed from the canonical shader by restoring
only the separate reductions and redundant entry barrier. Admission checks its
whitespace-normalized SHA-256 against main commit 826887e6. The independent
scalar f64 oracle uses the ABI's rounded f32 parameters, and Torch CPU f64
cross-checks it. Frozen tolerance: abs=2e-6, rel=5e-6. Existing strict Tensor
softmax tests retain their tighter 2e-7/2e-6 gate. WGPU route outputs must also
match bitwise, including twelve extreme-value/Chimera domain cases.

## Actual Wiring And Limits

ReadbackBatch is a Rust runtime API shared by native blocking and browser async
clients. It snapshots ordered prefixes, preserves empty entries, validates all
requests before copies and spills at per-buffer device limits. It retains
existing bounded native map waits and browser cancellation through owning leases.
Packing may hold more staging memory at once than sequential readback; the
per-buffer limit is not a total memory budget. No implicit sanitization occurs.

The ordinary Tensor softmax/hardmax path now batches its two outputs. Its spiral
API still computes consensus and the telemetry blend on CPU. The separate dense
GPU consensus helper now batches four outputs. Tests cover both, and assert
that the latter really dispatches consensus instead of silently using its CPU
fallback. The new backend consensus pipeline exposes the same ABI/bindings
without a filesystem dependency; Tensor retains its lazy compilation/cache.

This does not add resident graph softmax, migrate autograd, or establish model
quality. Desktop timings share an M4 with other workloads, and browser clocks
are coarse. Publish descriptive intervals, never a universal speed guarantee.

## Replay

Use Cargo 1.98.0 and the pinned formatter nightly-2026-04-15. Set
CARGO_BUILD_JOBS=4 and RAYON_NUM_THREADS=4 for serial local validation. Run:

```sh
python3 -I -B benchmarks/consensus-readback/test_consensus_protocol.py
python3 -I -B benchmarks/consensus-readback/test_evidence.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --lib --test readback_batch --example consensus_readback_bench -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 cargo test --locked --release -p st-tensor --features wgpu --test wgpu_softmax_extremes -- --test-threads=1
cargo run --locked --release -p st-backend-wgpu --example consensus_readback_bench
cargo build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown --example consensus_readback_bench_browser
```

Generate web assets with wasm-bindgen 0.2.104 using --out-name spiraltorch_wasm.
Use tools/test_resident_browser.cjs with the consensus-readback-bench fixture
and an isolated test Chrome, then run torch_bench.py against the native JSON.
Rotate runtime order native/browser/Torch, browser/Torch/native,
Torch/native/browser. Keep complete arrays/binaries locally; the archive
publishes all intervals, source identity, validation attempts and raw hashes.
Archive verification checks fixity and recomputes summaries, not GPU execution.

The browser streams each complete case to a bounded JSONL sink, outside timing;
the owned test driver reconstructs the report and records the stream hash.
This avoids keeping/rendering the complete large result in the page. The initial
all-at-once export crashed after reaching the last condition, despite the
ownership and domain checks passing. Those attempts remain separate; the complete
screening uses the explicit screen2 prefix, not overwritten files.
