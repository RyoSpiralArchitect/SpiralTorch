# Frozen corpus replay after pair-seed caching

This rerun combines the [paired corpus pilot](../2026-10-09-byte-corpus-matched-learning/README.md)
with the [metric head-reduction repair](../2026-10-09-poincare-pair-seed-cache/README.md)
and its [full-model integration checks](../2026-10-09-byte-causal-geometry-owner-pair-cache/README.md).
No data, initial weight, batch ordering, optimizer rate or acceptance criterion
was regenerated or selected after seeing results.

Native Metal and actual browser WebGPU each complete all six arms (three seeds,
ordinary/geometry) and 128 updates per arm. Both pass the independent frozen
CPU-f32 PyTorch comparison, including final geometry change-vector checks.
Each environment's full raw JSON is **byte-for-byte equal** to its original
pilot output, including every loss and final parameter value. This is equality
of serialized reports on this hardware/recipe, not a cross-device or general
determinism guarantee.

The unchanged results remain available as the original
[native comparison](../2026-10-09-byte-corpus-matched-learning/native-comparison.json),
[browser comparison](../2026-10-09-byte-corpus-matched-learning/browser-comparison.json),
[native scalar trajectories](../2026-10-09-byte-corpus-matched-learning/native.json)
and [browser scalar trajectories](../2026-10-09-byte-corpus-matched-learning/browser.json).
Two small geometry improvements and one small regression are still the result;
unequal parameter counts and compute still prevent a superiority claim.

[validation.json](validation.json) records fresh source, binary, raw report and
log hashes, the exact frozen input/reference hashes, all paired differences,
comparison summaries and commands. Full raw weights remain local. The original
evidence is unchanged and is referenced because this rerun verified equality,
not because it has been retrospectively relabelled as the new implementation.

Rust runner tests (8), Python controls (8) and CI-pinned nightly formatting pass.
Use the [same reproduction recipe](../../../docs/byte_corpus_learning.md).
No timing was measured; this does not establish speedup, large-sequence
readiness, better language generation or checkpoint/resume support.
