# Request-bound byte corpus resume

Tested implementation: `9abc41362bf2d73b92b1dc8f067276d54792e5d0`.
The public Rust `ByteCorpusStudy` now owns the existing paired corpus runner,
including data order, acceptance gates, evaluation schedule and resume state.
Native and actual browser WASM clients invoke the same Rust learner.

Both clients complete all six frozen arms (three seeds, ordinary/geometry),
pause at update 37, then resume to 128 in a fresh process/page. The resumed
report matches a fresh uninterrupted report **exactly within each runtime**:
all losses, evaluation records and final weights, including JSON value types
and signed-zero-sensitive float bits. Saved tensors also match the paused
readback in shape and float32 bits. Evaluation stays at 0/64/128, with no
extra evaluation at the pause. This is not native/browser bitwise equivalence.

- [Native verification](native.json) and [browser verification](browser.json)
  identify all six cases and the raw input/report/checkpoint hashes.
- [Native Torch comparison](native-torch-comparison.json) and
  [browser Torch comparison](browser-torch-comparison.json) pass the unchanged
  `3e-6 + 5e-5 * abs(reference)` and geometry change-vector relative-L2 gates.
- Reloading the completed native checkpoint performs no extra updates and
  preserves both the complete report and checkpoint bytes.
- A separate finite but extreme SGD-rate control (`3e38`) fails with a guarded
  rejection. No new checkpoint is published; the previous zero-update checkpoint
  remains byte-identical. This tests failed-segment persistence, not power-loss
  durability or rollback of a caller-owned live model.
- [Signed-zero transport](signed-zero.json) verifies six actual browser cases:
  `0x80000000` survives input, GPU snapshot, checkpoint and report download at
  update zero. Browser JSON reformatting is display-only; downloads retain
  Rust's original JSON strings.

Independent read-only local review found and closed four issues: checkpoint
size growth after compact input, unchecked stored weights in the verifier,
Python equality accepting boolean/signed-zero changes, and browser report
downloads erasing signed zero. The final focused review had no findings.
Rust study tests (12), CLI/I/O tests (5), model checkpoint regression tests (6),
Python resume-verifier tests (6), and the existing Torch comparator tests (8)
pass. CPU-only `st-nn` library checking, WASM compilation, browser-script syntax
checks and CI-pinned nightly formatting pass. No strict workspace lint claim.

See [validation.json](validation.json) for source/artifact hashes and exact
verification scope. Original raw weights, checkpoints, reports, build outputs
and logs remain local. The inventory also retains intermediate pre-review runs;
only the explicitly selected final artifacts support the final conclusions.
Earlier evidence has not been rewritten. Browser download formatting changed
to preserve Rust numeric text, so no byte-equality claim is made against the
old JavaScript-reformatted browser report.

Reproduce using the [study and resume recipe](../../../docs/byte_corpus_learning.md)
and `tools/verify_byte_corpus_resume.py`, followed by the independent Torch
comparison. Use a new output directory. To reproduce the failure control,
change only `rate` to `3e38` in a separate request, save at zero, then attempt
three updates from that checkpoint; the output path must remain absent. For
the signed-zero control, set `token_embedding.values[0]` to `-0.0` in every
case of a separate request and download at `?stop=0`.

This is the same tiny document-held-out SGD study as the
[original pilot](../2026-10-09-byte-corpus-matched-learning/README.md), not new
quality or speed evidence. Geometry still adds parameters; no winner is claimed.
The checkpoint binds exact request bytes, including the complete preselected
batch schedule and fixed rate. Conservative size admission can reject a
resumable study before GPU execution even when the input itself fits 64 MiB.
General streaming training, Adam/schedules, native-to-browser checkpoint
handoff, long-sequence qualification and pretrained LLM FT are not tested here.
