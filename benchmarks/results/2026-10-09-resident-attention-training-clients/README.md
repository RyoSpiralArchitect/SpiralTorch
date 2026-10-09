# Public attention training clients

This is **correctness and client-lifetime evidence**, not a speed, decoder,
fine-tuning or language-quality result. It extends the
[Rust-core evidence](../2026-10-09-resident-attention-projection-training/README.md)
through newly built Python and complete WASM packages, without changing the frozen
CPU-f32 PyTorch oracle or its `3e-6 + 5e-5 * abs(reference)` comparison gate.

## Results

- Fresh default-feature Python wheel, isolated CPython 3.12 environment on Apple
  M4/Metal: 30 forward/input/parameter/geometry-gradient conditions and 16
  MSE/SGD steps passed. Five unittest methods passed; the CPU-only method skipped.
- Full public WASM client, in-app browser WebGPU: all 30 conditions, 16 steps,
  and 12 composition/ownership/rejection guards passed. The browser reports
  `BrowserWebGpu`/`Other`, without a hardware name; no stronger identity is inferred.
- Maximum browser parameter-gradient error was `2.9802322387695312e-8`;
  final-parameter error was `1.862645149230957e-9`. First and last pre-update
  synthetic MSE were `0.16081055998802185` and `0.12581828236579895`.
- Separate CPU-only Python wheel: all three surface/feature-gate methods passed,
  with three GPU methods explicitly skipped. CPU-only WASM Node execution also
  passed composition and explicit GPU-rejection checks.
- Generated and shipped TypeScript contracts, existing attention/NN/vision type
  regressions, existing Python attention GPU tests, the executable Python docs
  example, and the Rust composition regression passed.
- Read-only independent review found no actionable introduced issue. It did not
  independently repeat the runtime checks.

The checks cover plan/owner immutability, result handles retained after owner
destruction, freeing a WASM plan during asynchronous compilation, repeated
cotangents, foreign/stale tokens, invalid rates, all-or-none rejection and recovery.
Runtime geometry biases stay caller-owned and are not optimized silently.

## Provenance and reproduction

`browser.json` is the exact downloaded browser result. `validation.json` records
fixture/source/artifact hashes, tool versions and separate test scopes. Complete
build/test logs, both local wheels, and generated WASM packages are retained
locally, not published as release artifacts. These local wheels retain version
0.4.29 but contain an **unreleased source delta**; they are not the PyPI 0.4.29 wheel.

The first Python probe attempt used an unavailable tensor method; the first
browser attempt expected the wrong text for a shape error. Both stopped rather
than passing falsely. Only the probes were corrected; the implementation and
numerical gate were unchanged. The failed Python log and corrected run are
retained separately; the published browser JSON is the completed run.

See [the client guide](../../../docs/resident_attention_training_clients.md) for
build commands and examples. GPU runs require explicit opt-in, and browser
execution must show completion rather than merely a successful WASM build.
Local Rust builds used 1.97.0; workspace formatting used nightly-2026-04-15.
No strict whole-workspace Clippy success is claimed by this record.

## Broader resident regression follow-up

The same default-feature wheel also passed all 57 methods from
`test_nn_resident*.py`, including 100 existing Topos graph updates against the
independent CPU PyTorch reference. This extends the regression scope, not the
attention quality/performance claim. `resident-regression.json` records the
tested source and log hashes separately from the frozen first validation.
The initial attempt lacked the optional Torch dependency; the completed run used
the already installed Torch with `-I -S -B`, startup patches disabled, and the
fresh wheel's site-packages first. No new dependency installation was needed.
