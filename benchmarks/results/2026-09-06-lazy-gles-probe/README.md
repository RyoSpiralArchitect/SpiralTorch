# Unmerged Bootstrap Candidate: Not Admitted

Do not merge this branch as a concurrent-runtime fix. Its historical `fix`
commit subjects express hypotheses, not a verified outcome.

Lazy GLES probing still times out and its debugger run has an inferior
SIGSEGV (the GDB wrapper exits zero). Sharing the primary instance, explicitly
destroying/polling the device, excluding implicit Vulkan layers, and selecting
only the NVIDIA ICD also fail bounded completion on Furnace.

The example's `join_then_drop` control completes six four-worker/four-round
processes. Crucially, it also completes six processes after removing all of
this branch's runtime changes. Worker-local Drop times out in both two-process
controls. Final retirement timing/thread ownership, not these bootstrap
changes, is the next boundary to investigate. This is not a production fix.

The harness-only branch `spiralreality/wgpu-lifecycle-regression` preserves the
unaltered main runtime (`f51f4b0f`), all three example modes, and hash-bound
positive/negative evidence under
`benchmarks/results/2026-09-06-wgpu-lifecycle/`. Its frozen example revision is
`d0598305`; the identical example on this branch is `952e9b8c`.

The disk-full `lifecycle-candidate-01.log` run is invalid setup evidence and
excluded from candidate outcomes. Later `verified-*` runs follow a successful
rebuild and executable hash verification. Browser/Metal passes cannot override
the observed Linux failure. No Python release or system driver change is part
of these probes.
