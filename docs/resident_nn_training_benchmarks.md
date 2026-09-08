# Resident NN Training: Measured Pass Encoding

This is a bounded comparison of **actual forward, backward, and plain SGD** on
the existing `Sequential` / `InferencePlan` resident path. It is not a peak GPU,
fastest-PyTorch, LLM quality, or general N-D autograd benchmark. See
[the public Rust, Python and browser training API](resident_nn_training.md).

## What Changed

On native Metal, a step now encodes all dispatches inside one compute pass.
For the measured depth-2/8/16 graphs, this changes 17/59/115 compute passes into
one. Dispatch count and order, shaders, uploads, loss snapshots, finite checks,
and transactional all-layer commit stay unchanged. This is pass coalescing,
**not shader fusion** and not removal of the requested loss readbacks.

The [WebGPU synchronization model](https://gpuweb.github.io/gpuweb/#synchronization)
defines compute usage scopes per dispatch. Keeping those dispatch boundaries
allows resource transitions within a pass; the native and browser numerical
tests still verify the implementation rather than relying on that rule alone.

An initially unconditional candidate was measured on both clients. Browser
results depended on readback cadence and varied substantially across repeats.
Therefore **BrowserWebGpu and unmeasured native backends retain the prior
one-pass-per-dispatch path**. The selection lives in Rust, not in a second
Python/JavaScript optimizer. Only Apple M4 native Metal was performance-tested;
other Metal devices, Vulkan, DX12, GL, CUDA and Furnace were not measured here.

## Frozen Workload

- Three seeds, 17/29/43; shapes `[2,16,32]`, `[4,16,64]`, `[4,32,128]`; respective
  depths 2/8/16. Each layer is square Linear; all but the last have tanh-GELU.
- f32, deterministic Rust-generated weights and batch, mean-MSE, learning rate
  0.01, eight nonzero-rate SGD updates per interval. The batch and weights stay
  device-resident, and input gradients are computed on both implementations.
- Register-2x2 GEMM, sequential accumulation. No changing kernels, prompts,
  shapes or sample counts after observing a result.
- Every interval starts from the same frozen plan and batch. Compilation,
  uploads, reset, initial/final zero-rate probes, full-state serialization and
  numerical validation are outside timing. Setup can still affect caches/heat.
- Immediate cadence reads every step's loss and finite flags. Deferred cadence
  captures all eight owning snapshots before reading them. Both time the full
  interval through its requested readbacks, not GPU timestamps alone.
- Each cadence has two discarded warmup blocks and eight retained blocks.
  Native A/B/PyTorch order rotates; browser A/B order alternates. All raw timings
  are retained, not only wins. PyTorch uses eager SGD, `foreach=False`, one CPU
  thread, no TF32 and no MPS CPU fallback.
- Rust validates intermediate finite values and commits transactionally;
  eager PyTorch has no equivalent per-stage guard cost in this test. Deferred
  PyTorch stacks losses for one host copy; Rust reads its individual snapshots.
  Thus numerical trajectories match, but safety/readback implementations differ.

## Observed Results

Recorded on Apple M4, PyTorch 2.12.1 MPS, Chrome 152.0.7977.77 with an isolated
headless profile. macOS GPU contention was **unknown**. Only owned GPU jobs were
serialized; other applications and OS scheduling were not controlled. Browser
adapter metadata comes from a separate probe, not exact Rust-device attestation.

The table reports ranges across the three seeds of ratios of **eight-sample
medians**, not confidence intervals. Values above 1 favor the adopted candidate.
These are diagnostic observations, not universal speed guarantees.

| Depth | Readback | Native old/new | MPS/new | Browser old/new |
| --- | --- | ---: | ---: | ---: |
| 2 | immediate | 2.82-2.96 | 2.81-3.82 | 0.97-1.15 |
| 2 | deferred | 1.66-3.43 | 2.70-3.32 | 0.36-1.12 |
| 8 | immediate | 2.42-3.19 | 1.33-1.78 | 0.67-1.26 |
| 8 | deferred | 2.64-3.02 | 1.48-1.85 | 0.94-1.27 |
| 16 | immediate | 1.41-1.57 | 0.66-0.67 | 1.03-1.05 |
| 16 | deferred | 1.47-1.54 | 0.49-0.56 | 0.99-1.01 |

The largest native workload **still loses to eager PyTorch MPS**. The adopted
browser implementation keeps the old pass cadence; even that control shows
large timing scatter on small workloads. Do not interpret the table as a
browser speedup, proof of performance parity, or a precise threshold for
automatic shape-based routing.

The earlier unconditional candidate improved native medians by 1.29-2.42x,
but browser deferred depth-2 ratios were 0.83-0.87 in one run and 0.77-1.03 in
the full-matrix repeat. Both runs remain in the evidence. This motivated keeping
the prior browser path, not a claim that every coalesced browser run is slower.

The first browser runner timed out after 600 seconds without intermediate
captures. That failure is retained and contributes no performance result. The
revised runner persists each completed case and small progress receipts outside
timing, instead of putting the entire large capture in a final DOM element.
The precise point reached by the failed first run is unknown.

## Correctness And Provenance

All nine adopted benchmark recipes passed. Native and browser captured final
weights, predictions, input gradients and parameter gradients matched the
independent native PyTorch captures with maximum absolute error about `1.49e-8`
(tolerance `1e-5 + 1e-4 * abs(reference)`). All retained loss trajectories were
checked during execution. Every reset native/browser trajectory's full-state
fingerprint was checked live, including intervals without a full emitted state.

The read-only validator rechecks captured arrays, all timings and medians,
rotation/warmup membership, source manifests, and all 360 browser interval
progress receipts. Fingerprints are consistency receipts, not independent
post-hoc state rehashes. The separate correctness fixture passes 18 VJP cases,
three 128-update fits, nonfinite rollback/recovery and export. CPU/MPS PyTorch
replays have maximum absolute error about `1.19e-7`. These are synthetic learning
tests, not evidence of downstream model-quality improvement.

Source commits are immutable and both native/WASM build manifests are clean:

| Role | Commit |
| --- | --- |
| Prior implementation with shared benchmark | `b2a2eb0bb863a13933d3cc159fb282acec9309fb` |
| Unconditional coalescing experiment | `e8db3b842cbbe96447b83456ff144d071b2eec62` |
| Browser progress/case retention | `00652cbd` |
| Adopted Metal-only policy and validator | `5062aaa79c54f3ea6cc6568483af6134291a50d2` |

See [retained evidence](../benchmarks/results/2026-09-08-resident-nn-training-throughput/README.md)
for raw reports, failures, build/validation logs, checksums and compact summaries.
No newly built Python wheel was performance-tested or published in this change;
Python/WASM production bindings were compile-checked against the adopted Rust.

## Reproduce

Build baseline and candidate from separate clean checkouts of the selected
commits and retain separate products. Use `CARGO_TARGET_DIR` appropriate for
your disk; do not overwrite the baseline executable while building a candidate.

```bash
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_training_bench
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_training_bench_browser --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir /tmp/new-training-module --out-name spiraltorch_wasm \
  target/wasm32-unknown-unknown/release/examples/resident_training_bench_browser.wasm

python -I tools/bench_resident_training_vs_torch.py \
  --baseline /path/to/baseline-binary --baseline-source BASELINE_COMMIT \
  --candidate /path/to/candidate-binary --candidate-source CANDIDATE_COMMIT \
  --device mps --output /tmp/new-native-training-bench.json

node tools/bench_resident_training_browser.cjs \
  /path/to/baseline-module /path/to/candidate-module /path/to/chromium \
  /tmp/new-browser-training-bench.json

python -I tools/validate_resident_training_bench.py \
  --native /tmp/new-native-training-bench.json \
  --browser /tmp/new-browser-training-bench.json \
  --browser-progress /tmp/new-browser-training-bench.json.progress.jsonl \
  --baseline-source BASELINE_COMMIT --candidate-source CANDIDATE_COMMIT \
  --browser-harness-source HARNESS_COMMIT --output /tmp/new-training-validation.json
```

The local run used Rust 1.98.0 and wasm-bindgen 0.2.104 matching Cargo.lock.
Playwright must be available to Node. Run owned GPU jobs sequentially, reserve
fresh output names and retain failures. Native reports are intentionally large
because they keep independent full-state references. The ordered
`raw.tar.xz.part-*` archive compresses them without dropping numeric captures.
Cheap admission/negative tests and the
browser example's compile check also run in CI.
