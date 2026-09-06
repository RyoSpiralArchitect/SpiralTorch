# Backend Comparisons Against PyTorch

Benchmarks are a correctness and optimization tool, not a claim of model-quality
superiority. The harnesses below are source-checkout tools. The fixes and
`GoldenRetriever::run_epoch_owned` described here are not in the public 0.4.27
wheel; build the current source to exercise them.

## Matched Native Tensor Calls

```bash
python tools/bench_backend_vs_torch.py \
  --st-backend wgpu --torch-device cuda \
  --sizes '64,64;256,256;512,512;1024,64x64,32' \
  --seeds 17 29 43 --warmup 5 --iters 20 \
  --output wgpu-vs-cuda.json
```

Use `--st-backend faer --torch-device cpu` for a CPU comparison, or
`--torch-device mps` on supported Macs. `bench_st_vs_torch.py` delegates to this
harness; old implicit `auto`/fallback behavior is intentionally removed and
`--output` is now required. A new output path is reserved without overwriting an
earlier result. `--native-prefix /path/to/venv` can enforce import provenance.
Faer only implements matmul here: gather/scatter use the explicit CPU indexing
path. Every case records both `requested_st_backend` and `effective_st_backend`,
including failed cases; this dispatch is not a GPU-error fallback.
The Python matmul label `cpu` is also a Faer alias and is reported as
`effective_st_backend: "faer"`, not as a distinct CPU implementation.

- Both runtimes receive identical float32 values, with fixture hashes and seeds.
- Matmul, gather, and scatter must match a float64 CPU reference before timing.
- GPU timings include input upload and output readback. Host tensor construction
  and list conversion are excluded for both runtimes. Index conversion inside
  the public API is included; ST uses integer IDs and Torch uses int64 tensors.
- Torch device-resident timing is a separate diagnostic, never the denominator
  of a host-to-host speed claim. TF32 and ST int8 opt-in are disabled.
- Warmed implementations run in randomized paired order. Raw samples, first
  call, median, p95, native-module hash, and requested backends are retained.
- Accelerator errors fail the case; they do not silently become CPU timings.
  The autotune cache is private to that output path.

Float32 WGPU matmul no longer implicitly quantizes large RHS matrices. Existing
lossy int8 machinery remains available through the process-level
`SPIRALTORCH_WGPU_ALLOW_INT8=1` opt-in, read on first use. Set it before execution;
changing it after initialization has no effect. This is an approximate mode,
not a float32-parity optimization, and its autotune cache identity is separate.
The precision bit comes from the original RHS dimensions, before shape
bucketing; synthetic autotune samples use that same precision.
The matched float32 harness explicitly disables this option.

## Strict Rank Execution And Control

```bash
cargo build --release --locked -p st-core \
  --features cuda,wgpu-rt,kdsl --example backend_rank_bench
python tools/bench_rank_vs_torch.py \
  --executable target/release/examples/backend_rank_bench \
  --output rank-comparison.json
```

The Rust executable consumes JSON lines, executes real CUDA/WGPU TopK, BottomK,
and MidK with fallback forbidden, and checks canonical values and tie indices.
The driver also compares those outputs to PyTorch using identical fixtures.
These timings are **unpaired Rust/Python wrapper diagnostics**, not a direct
kernel speed ratio. CUDA Tensor.matmul is not exposed by this experiment.

SpiralK scripts produce the actual executed RankPlans. On supported WGPU cases,
Black Cat's seeded MultiBandit chooses between direct and two-stage scripts. On
CUDA it chooses real workgroups (`32`, `128`, or `256`) consumed by heap,
exact-rescan, and MidK launches. Dynamic shared memory and heap capacity are
derived from the selected workgroup and checked against both the plan's device
caps and the verified no-opt-in envelope: at most 256 threads and 48 KiB of
dynamic shared memory.

UCB selects every arm once before comparing posterior scores. Selection attempts
are distinct from observations: failed correctness or explicit abandonment
advances initial exploration without training the posterior. The witness records
both counts under Black Cat bandit witness contract v3. Correctness failure
additionally quarantines the arm; abandonment does not. Candidate scripts are
deduplicated by their effective kernel execution signature, not by unused plan
fields. WGPU two-stage tiles are normalized to the row width, while CUDA `k=1`
ignores the planner workgroup because the executor always launches one fixed
warp. Reward remains `1 / (1 + elapsed_ms)` and is credited only after successful
execution and correctness. This verifies live feedback, not convergence or a
cross-runtime speed advantage.

The v2 harness measures every admitted arm equally in a rotating round-robin
control stream before the adaptive stream. Control timings never update the
posterior, and candidate input order rotates across seeds. The retained artifact
therefore separates arm performance from controller allocation while preserving
both in one receipt. Process-level failures also produce an atomic v2 receipt
with a named stage. Successful runs record the clean Git commit/tree and tracked
diff identity, execute a private hardlink snapshot of the native binary, and
verify source, path, inode, size, timestamp, and SHA-256 stability after the run.
Before measuring, the harness also asks that exact execution image for its
Rust-embedded build manifest. Its package, Git commit, Git tree, and clean-build
state must match the source checkout or the run fails at
`build_identity_preflight`; two independently stable but unrelated artifacts are
not accepted as valid provenance.

NVRTC rank kernels are header-independent in runtime compilation. The original
heap remains in use where it is exact; wide rows with k > 8 use an exact rescan
when eight retained candidates per thread can lose concentrated winners.
This prioritizes correctness in that regime, not faster large-k CUDA execution.
The `k=1` bitonic CUDA path still launches one fixed warp, so workgroup-only
scripts collapse to one effective candidate instead of forming a false tuning
domain.

See the [RTX 5090 rank-adaptation receipt](../benchmarks/results/2026-09-04-rank-adaptation-furnace/README.md)
for the retained 36-case WGPU/CUDA run and its unpaired PyTorch diagnostics.

## Actual WASM CPU Calls

Build `spiraltorch-wasm` for `wasm32-unknown-unknown` and generate a Node module
using a matching wasm-bindgen CLI, then run:

```bash
python tools/bench_wasm_vs_torch.py \
  --module /absolute/path/to/spiraltorch_wasm.js \
  --output wasm-cpu-vs-torch.json
```

This executes actual AutogradTensor WASM methods, including repeated-ID scatter,
with fixed tensors and 27 correctness-gated cases. It records the `.wasm` hash.
It measures single-threaded WASM CPU calls versus PyTorch CPU, **not browser
WebGPU**, and does not include JS transport in tensor-call timing. The new
native host-poll backoff does not block or replace the browser event loop.

## Golden Training Continuation

Rust `GoldenRetriever::run_epoch_owned(...)` returns a `GoldenTrainingOutput`
containing the existing report plus `GoldenTrainedWorker { worker, module, loss }`
records. Keep these models and loss objects for subsequent epochs; the retriever
retains the matching trainer state. The legacy `run_epoch` still returns only a
report and drops the models on return. This new API is Rust-only for now.

With dropout enabled, returned worker IDs remain stable and only successful
workers are returned. Do not treat a partial result as a dense replacement for
all configured workers. Use `run_epoch_workers` to continue with survivors:

```rust,ignore
let next = output.trained.into_iter().map(|worker| {
    let schedule = schedules[worker.worker].clone();
    worker.with_epoch(make_loader(), schedule)
}).collect();
let output = retriever.run_epoch_workers(next)?;
```

This worker-ID-keyed input can be reordered. Sparse input requires
`allow_dropout` and at least `min_successful_workers` participants; duplicate
or unknown IDs are rejected before any epoch work starts. Omitted workers are
not trained or counted as fresh dropouts. Failed module/loss state is not
recoverable, and errors do not imply rollback of completed trainer updates.
Do not attach a new model to a failed worker's partially advanced trainer.

Regression coverage compares all parameters across three epochs of two-worker
parallel and sequential training. Barycenter synchronization also handles rank
hints above tensor width without panic, normalizes storage layout, accumulates
in float64, and rejects nonfinite inputs/results.
Additional coverage drops worker 0, reverses surviving IDs 1 and 2, and checks
three epochs against sequential training with distinct per-worker learning rates.

`sync_z_barycenter_with_receipt(...)` preserves the portable WGPU-first default
and returns the exact planning snapshot, effective rank, guard, and semantic
owner. Golden uses `golden_barycenter_adaptation_session(...)`, which deduplicates
SpiralK candidates by the FFT tile actually consumed by the guard rather than by
the unrelated rank-k kernel path:

```rust,ignore
let mut adaptation = golden_barycenter_adaptation_session(
    &base_plan,
    &scripts,
    SoftBanditMode::UCB,
    seed,
)?;
let selection = adaptation.try_choose()?;
let (barycenter, receipt) = retriever.sync_z_barycenter_with_plan(
    &partials,
    rank,
    selection.plan(),
)?;
let correctness_passed = validate_golden_barycenter_output(
    &partials,
    rank,
    selection.plan(),
    &barycenter,
    GOLDEN_BARYCENTER_DEFAULT_ATOL,
    GOLDEN_BARYCENTER_DEFAULT_RTOL,
)?;
adaptation.try_observe(
    selection.receipt().selection_id,
    elapsed_ms,
    correctness_passed,
)?;
```

The receipt states `plan_executed: false`: Golden consumes the plan only to
parameterize its curvature guard. It does not claim a rank kernel launch.
Golden integration coverage varies only a consumed FFT tile, rejects duplicate
guards even when rank-k knobs differ, compares every candidate against an
independently recomputed partial/guard formula, and credits measured wall time
rather than a hard-coded reward.

See the [measured pilot and retained failures](../benchmarks/results/2026-09-04-furnace-wgpu-first/README.md).
