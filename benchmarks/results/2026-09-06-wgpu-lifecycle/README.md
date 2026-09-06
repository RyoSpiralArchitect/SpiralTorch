# Concurrent WGPU Lifetime Probe

This is a correctness diagnostic, **not a production fix or a throughput
benchmark**. The mainline change is a standalone Rust example plus a JSON
dev-dependency; runtime selection, kernels, Python and WASM execution are
unchanged. The failed lazy-GLES/shared-instance candidates remain unmerged.

## Frozen Comparison

Each process runs four workers for four rounds. Every worker creates its own
headless runtime, rejects a CPU adapter, uploads four u32 values, submits a
copy, reads back and checks exact equality. Startup and readback stay parallel.
The modes differ only in final lifetime handling:

- `drop`: the worker drops its runtime before returning.
- `destroy`: the worker explicitly destroys and polls the device before Drop.
- `join_then_drop`: workers return retained runtimes. The parent joins every
  worker before dropping any returned runtime, then retires them serially.

The last mode changes both destruction overlap and the retiring thread. It
does not distinguish those causes, guarantee safety for escaped device/queue
clones, or exercise compute kernels and arbitrary asynchronous callbacks.
Error paths can still drop resources on a worker; this is not a general-purpose
retirement API. An external process timeout is mandatory because Rust polling
deadlines cannot interrupt a foreign driver call or Drop.

| Code / Platform | Mode | Completed Processes | Outcome |
| --- | --- | ---: | --- |
| Main runtime, RTX 5090 / Vulkan | `join_then_drop` | 6 / 6 | 96 exact readbacks and completed runtime drops |
| Main runtime, RTX 5090 / Vulkan | `drop` | 0 / 2 | both exceed 30 seconds, exit 124 |
| Lazy GL + shared primary instance, same GPU | `join_then_drop` | 6 / 6 | 96 exact readbacks and completed runtime drops |
| Lazy GL + shared primary instance, same GPU | `drop` | 0 / 2 | both exceed 30 seconds, exit 124 |
| Main runtime, Apple M4 / Metal | all three modes | 3 / 3 | one process per mode, 48 exact readbacks |

All processes ran sequentially, with no build overlapping a measured process.
The candidate block ran first, then the main-runtime block; each block ran six
join-retirement processes before two Drop controls. This is not randomized or
an exclusive GPU reservation. The same executable hash was checked before and
after each block. The final JSON, all 16 unique worker/round receipts, four
joined rounds and 16 completed drops were checked for every successful process.
Timeouts have no final success report. Exit codes are runner observations, not
values reconstructed from incomplete log text.

Main runtime is `f51f4b0fd33fff0e891e94be0337f1ac40e50f5e`; the harness-only
revision is `d05983059b11273cf1ea97737fb1d4b6ecd369f5`. The unmerged candidate is
`952e9b8c56f41d121e7cf7f8ea760f574f878ef7`. Both use byte-identical example blob
`e0bc682db7f2ea85917d340c380f8d67a0b75b4a`. See [evidence.json](evidence.json)
for executable identities, per-process outcomes and archived artifact hashes.

On the harness-only revision, 70 backend tests with real Metal enabled and the
shader validation test pass. Native and wasm32 all-target strict clippy, pinned
rustfmt, and seven invalid-argument cases also pass. The 63 archived artifacts
were independently re-read and checked against their size and SHA-256 entries.

## Earlier Negative Controls

The archive also retains the earlier probes rather than pooling them into the
frozen comparison. Furnace was running NVIDIA 595.84, Vulkan loader 1.4.341,
Linux 7.0.0-29, glibc 2.43 and Rust 1.98.1; pinned wgpu/hal/core are
0.20.1/0.21.1/0.21.1. Existing CPU workloads were left running and untouched.

- An unchanged-main unit suite passes 70 tests under GDB once and times out on
  the next run. The minimal original Drop harness also times out.
- Baseline stacks include `vkDestroyDevice`, GLES/EGL teardown and driver
  thread-exit paths. They localize stalled lifetimes, not a proven root cause.
- Lazy GLES probing alone passes two lifecycle runs, then times out. Its GDB
  run reports an inferior **SIGSEGV**, despite the GDB wrapper exiting zero.
  That record is a failure, never a passing test.
- Retaining the primary instance still times out. Explicit device destruction
  and polling also time out; all four workers report `device_destroyed` before
  final runtime destruction stalls.
- Process-scoped implicit-layer exclusion and an NVIDIA-only ICD selection
  each still time out. No system driver or persistent configuration changed.
- `lifecycle-candidate-01.log` is **invalid setup evidence**: a disk-full link
  failed, the attempted binary copy failed, and the incomplete/stale image was
  executed accidentally. It is excluded from all candidate outcomes. A clean,
  successful rebuild and hash verification precede the `verified-*` runs.
  Only this task's failed copies and backend build cache were cleaned; no
  weights, corpora, other jobs or source changes were removed.

The unmerged runtime patch is archived for inspection, not recommended as a
fix. Its historical comments/commit subjects describe hypotheses which the
negative controls did not establish. Metal/browser passes on that candidate
do not erase its Linux failure. Python was not rebuilt for these candidates.

## Reproduce

Build successfully first, then run the resulting executable with an outer
bound. Do not benchmark a stale binary after a failed build. On Linux:

```bash
cargo build --locked -p st-backend-wgpu --example headless_lifecycle
sha256sum target/debug/examples/headless_lifecycle
timeout --signal=TERM --kill-after=10s 30s target/debug/examples/headless_lifecycle 4 4 drop
timeout --signal=TERM --kill-after=10s 30s target/debug/examples/headless_lifecycle 4 4 join_then_drop
```

Use the appropriate path if `CARGO_TARGET_DIR` is set. Run each command only
after the preceding process is terminal; record exit 124 as incomplete, not
success. The CLI limits workers to 1..=8 and rounds to 1..=16 and rejects
unknown modes before requesting a GPU. The wasm32 example intentionally has
no native-thread probe; real browser coverage remains in the asynchronous
[resident rank fixtures](../2026-09-06-midk-prefix-seek/README.md).

## Next Boundary

The minimal distinction survives removal of the runtime candidates. Investigate
final ownership, worker/thread exit and independent device retirement before
changing production selection. A safe API must account for externally cloned
devices/queues and error paths, not simply lock all dispatches, leak devices,
drop GPU support or hide failures behind a CPU fallback. No production
concurrency guarantee or universal driver-bug attribution follows from these
finite diagnostic runs.
