# Full-model regression after pair-seed caching

The [Poincare review repair](../2026-10-09-poincare-pair-seed-cache/README.md)
has been integrated into the [complete byte-model geometry owner](../2026-10-09-byte-causal-geometry-owner/README.md).
This record qualifies that combined source, not just the standalone metric.

Native Metal and actual browser WebGPU each pass all four fixed models with
18/31/23/37 parameter tensors and 16 CE/SGD updates per model. Every parameter
VJP and update remains within the unchanged independent PyTorch criteria.
Geometry gradients remain nonzero and within their relative L2 gate of 0.002.
Geometry-off, detached-coordinate, zero-Q/K, prefix/suffix/document isolation,
retained-tape and all-or-none update controls remain active.

Both new decoded reports are semantically identical to their previous reports,
including every error value and every update trajectory. See the unchanged
[native values](../2026-10-09-byte-causal-geometry-owner/native.json) and
[browser values](../2026-10-09-byte-causal-geometry-owner/browser.json).
This equality concerns the **verification reports**, not an additional claim
of bitwise equality for every unreported intermediate GPU tensor.

[validation.json](validation.json) identifies both merged revisions, the actual
tested source, fresh raw report/log/binary hashes, all four per-client summaries
and reproduction commands. Native unit tests (8), complete-model integration
(1), residual-attention integration (1) and CI-pinned nightly formatting pass.
Only the full-model WGPU path was rerun here; this does not claim a fresh
CPU-only test run or a clean strict workspace lint run.

Raw reports and build artifacts remain local. The earlier evidence files are
unchanged; their values are referenced only because this rerun explicitly
compared and verified them. Reproduce using the commands in
[the byte-model API document](../../../docs/resident_byte_decoder.md).

This is synthetic end-to-end training correctness after a structural kernel
optimization. It is not a corpus-quality improvement, a throughput benchmark,
a long-sequence qualification or a completed-model checkpoint/resume test.
