# Retirement Order: Failed Guards And Stronger Controls

This follows the [lifetime probe](../2026-09-06-wgpu-lifecycle/README.md), merged
in PR #2077 as `1ef53ed0`. **No runtime or dependency workaround in this record
is admitted for production.** The repository change only adds diagnostic
example modes. The original independent-runtime completion risk remains open.

## Worker Controls

The example revision is `386d7cf2208f85e49a229002b119e6d44d83b8f8`; production
runtime code is unchanged. Creation and exact four-u32 copy/readback are not
serialized by the harness. Three additional modes separate final lifetime
handling from worker exit:

- `worker_serial_drop`: a round-local mutex covers only the worker's runtime
  Drop, not creation, upload, readback, or the entire worker body.
- `worker_drop_park`: Drop remains concurrent; an exit guard keeps each worker
  alive until all workers reach their exit guards.
- `worker_serial_drop_park`: combine both controls. The exit guard also runs
  on ordinary Rust errors/unwind, but cannot rescue a blocked driver call.

All runs below use Furnace's RTX 5090, Vulkan, NVIDIA 595.84, Linux 7.0.0-29,
glibc 2.43, and Rust 1.98.1. Each process has an external timeout and is fully
reaped before another GPU process starts. Successful repeated rows are not
statistical guarantees; the conditions also change scheduling and timing.

| Unchanged Runtime | Workers x Rounds | Outcome |
| --- | --- | --- |
| `drop` | 4 x 4 | 0/1 completes; exit 124 at 30 seconds |
| `worker_drop_park` | 4 x 4 | 0/1 completes; exit 124 at 30 seconds |
| `worker_serial_drop` | 4 x 4 | 6/6 complete; 96 exact readbacks |
| `worker_serial_drop` | 8 x 8 | corrected framed pair: 2/2 complete; 128 exact readbacks; 60-second bound |
| `worker_serial_drop_park` | 4 x 4 | 3/3 complete; 48 exact readbacks |

The first three serial processes precede the park controls and ordinary Drop
control. The remaining three serial processes and two larger serial processes
run after the first HAL experiments. Their original SSH-combined stream can
splice stderr into long JSON stdout, so those two original stress logs are
retained as completion-only diagnostics, not admitted structured receipts.
A later pair with the same executable captures stdout/stderr separately on
Furnace before transport; both framed reports validate all 64 unique receipts.
Do not strip diagnostic text out of malformed JSON and relabel it a clean run.
Do not pool this chronology into an
interleaved throughput comparison. Metal separately completes one process for
each of all six modes (96 exact readbacks). Native example strict clippy,
wasm32 all-target strict clippy, and pinned rustfmt pass. No Python rebuild or
new browser throughput comparison was performed for these diagnostics.

## HAL Experiments: Not Fixes

These are temporary, external Cargo path overrides of `wgpu-hal 0.21.1`,
not edits to the shared Cargo registry or the repository's dependency policy.
The upstream crate archive SHA-256 is
`172e490a87295564f3fcc0f165798d87386f6231b04d4548bca458cbbfd63222`.
Each attempted change is retained as a small patch in the raw archive.

| Candidate | Non-Debugger Outcome | Additional Evidence |
| --- | --- | --- |
| NVIDIA Vulkan `destroy_device` mutex only, pinned graph | 0/1 completes | GDB interrupt shows simultaneous Vulkan and EGL destruction |
| Shared reentrant gate for NVIDIA Vulkan destruction and native Linux EGL teardown | 5/6 complete | GDB 8x8 exits normally; GDB 8x16 exceeds its 40-second bound |
| Lazy GLES/shared primary instance (`952e9b8c`) plus the pinned Vulkan-only guard | 1/2 complete | another 30-second timeout; no production admission |

The first unpinned path-override run is **confounded**: Cargo resolves
`libloading` to 0.7.4 instead of the mainline graph's 0.8.9. It times out but
is excluded from single-change comparisons. The later snapshots pin 0.8.9;
the only lockfile difference is the path source replacing the registry source
and checksum for wgpu-hal. `external-hal-lockfile.patch` records that change.
The combined candidate uses the earlier, byte-identified three-mode example;
it is not presented as byte-identical to the newer six-mode harness.

The pinned Vulkan-only debugger stack contains one worker in
`vkDestroyDevice`, one waiting for the new Rust mutex, and two in
`eglDestroyContext`. With the cross-backend gate, the later stalled stack
instead has an application worker and a driver-owned worker running NVIDIA
cleanup from glibc's `__nptl_deallocate_tsd`, outside the Rust Drop guards;
another worker waits inside `vkDestroyDevice`. Thus guarding selected Drop
bodies does not close the observed thread-local cleanup boundary. These are
localization clues, not proof of the driver's internal root cause.

The repository's vendored wgpu wrapper was also compared with upstream
wgpu 0.20.1: its source delta is confined to four lines in `webgpu.rs`, not
the native destructor implementation. That check alone does not prove which
layer is faulty. Vulkan's [device-destruction requirements](https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyDevice.html)
still require completed work, freed child objects, and synchronization for
the affected device/queues; a cross-device mutex is not a substitute for those
requirements.

## Evidence And Boundary

[evidence.json](evidence.json) binds the archived logs and patches, executable
hashes, sources/overrides, process exits, and successful receipt checks. The
debugger's wrapper status is distinguished from the inferior's outcome.
Failed and confounded runs remain visible. Existing CPU jobs were left running;
this was not an exclusive machine reservation. No driver, kernel, system
configuration, package release, notification workflow, or production WASM path
was changed.

The corrected transport uses Python's standard-library `subprocess.run` with
`capture_output=True` on Furnace, an outer GNU `timeout` for the native child,
and one JSON envelope containing the child's exit code, separate stdout/stderr,
and pre/post executable hashes. Only the envelope crosses SSH stdout. The
underlying Rust example and runtime are unchanged by this collection repair.

Do not turn these findings into a global dispatch lock, leaked devices,
unqualified driver-bug attribution, or silently shared independent runtimes.
Raw device/queue clones and driver-owned thread-local destructors make a
wrapper-level lifetime guarantee insufficient. The successful ownership/exit
controls motivate a more explicit retirement design, but they are not that
design or a replacement for the requested independent-runtime semantics.
