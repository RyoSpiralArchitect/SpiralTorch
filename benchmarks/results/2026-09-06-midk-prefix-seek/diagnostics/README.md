# Parallel Runtime Completion Diagnostic

The final canonical performance comparison finished before these diagnostics.
No samples below are throughput evidence or substitutes for a passing test run.

On Furnace (RTX 5090, Vulkan, NVIDIA driver 595.84), a later default-parallel
backend test run stopped producing progress. The process remained live; two
threads named `midk_bottomk::t` and `runtime::tests:` each used about one CPU,
other test threads waited, and the GPU reported 0% utilization / 422 MiB. Kernel
stack inspection was denied. After more than seven minutes, only this owned
test process was sent SIGTERM. Cargo exited 101, explicitly reporting signal 15.
`furnace-parallel-interrupted.log` is not a passed verification.

Controlled runs used `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1` and an outer
`timeout --signal=TERM --kill-after=10s 120s`, with no other GPU test/benchmark
launched concurrently:

| Revision | libtest options | Outcome |
| --- | --- | --- |
| Baseline 82bdd5e2 | default parallel, `--nocapture` | exit 124; incomplete after 120 seconds |
| Baseline 82bdd5e2 | `--test-threads=1 --nocapture` | exit 0; 69 unit tests + shader validation pass |
| Candidate 817af465 | `--test-threads=1 --nocapture` | exit 0; 70 unit tests + shader validation pass |
| Candidate 817af465 | default parallel, captured output | exit 124; incomplete after 120 seconds |

The unchanged kernel also fails bounded parallel completion, while serial
correctness completes on both revisions. An earlier candidate full suite passed
too, as retained in `../furnace-initial-suite.log`. This indicates a reproducible
parallel runtime/test-completion risk, not a demonstrated MidK arithmetic error.
Its root cause and production concurrency impact are **unresolved**. It must not
be relabelled as a fixed driver bug, an all-parallel pass, or proof that a global
slow fallback is required. No production serialization/fallback policy was added.

## Startup Isolation Probe

A separate diagnostic worktree (`085376cc`, patch retained here) changes only
the four instance-construction sites in `st-backend-wgpu` to request Vulkan
instead of `Backends::all()`. It is Linux-only diagnostic code, not part of the
PR's production changes and not a portable default policy.

With default-parallel `--nocapture` and the same 120-second outer bound, both
Vulkan-only runs complete: 70 unit tests plus shader validation pass. The first
unit run takes 2.10 seconds. This isolates an initialization-backend condition
that changes completion on this machine; it does not identify a proven driver
defect or establish safety of excluding other backends on every platform.

The separate 30-second `strace` run exits 124. Its compressed trace includes
waits under `vkCreateInstance`, `vkCreateDevice`, `vkDestroyDevice`, and NVIDIA
library paths. Tracing changes scheduling; these frames are localization clues,
not a root-cause proof. The uncompressed trace SHA-256 is
`476ad24845f1ad3963d3e6cc2f6f7065db9906e55675271f6d0172667b2b9579`.

Next work should consolidate Rust instance/bootstrap selection across runtime,
core and tensor callers with explicit backend/fallback semantics, then repeat
the parallel control. Do not merge the hardcoded Vulkan-only probe as that fix.
