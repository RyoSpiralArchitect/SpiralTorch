# Python Geometric Byte Decoder, 0.4.30 Candidate

The Python client now uses the same Rust-owned complete byte model as the
native/WASM qualification. This record covers an **installed local macOS arm64
candidate wheel**, not a published release. No production geometry equations,
thresholds, optimizer semantics or kernel choices were changed in this increment.

## Result

- All six matched-graph cases pass against the frozen v2 CPU reference and
  retained native/WASM reports: 96 updates and 720 geometry-step VJPs.
- All six split-at-seven continuations match uninterrupted updates 8-16 exactly,
  including losses, revisions, full parameter VJPs, embedding VJPs, mixture
  weights, parameters and final checkpoints.
- Four metrics, heterogeneous heads, shared mixtures and contextual mixtures
  compose with QKV/Topos through nine new native `spiraltorch.nn` classes.
  The contextual router consumes geometry coordinates, not projected Q.
- The packaged example runs 16 repeated-window updates: CE
  `5.574340 -> 5.243394`, with exact prediction after checkpoint restoration.
  This is a mechanics example, not language-quality evidence.
- The seven test suites list 42 tests: 39 passed and three CPU-only runtime
  gates skipped on the WGPU wheel. The separate CPU-only build check passed.
  Existing attention (30 forward/VJP cases) and residual attention (60 cases,
  two 32-update loops) regressions also passed.

The GPU was Apple M4 / Metal. Geometry relative error reached
`0.0018354366643319622` under the unchanged `0.002` gate. This qualification
does not select a winning geometry or imply portability to untested hardware.

[Parity results](parity.json) include every condition and learning step,
adapter identity, candidate/native hashes, and exact continuation status.
[Validation](validation.json) retains log/source hashes, test counts, skipped
coverage, the packaged example's scalar losses and all failed attempts.
[Source checksums](source-files.sha256) are relative to the repository root;
[artifact checksums](SHA256SUMS) are relative to this directory.

## Review And Failed Attempts

The initial example had two real coordinates but two decay/phase values;
the portable smoke also used an odd projection width. Rust correctly rejected
both. Correcting those examples did not change native equations.

The first parity run passed only 3/6 because its reader treated a flat list of
external-bias VJPs as per-block dictionaries. [All six original outcomes remain
retained](first-parity-attempt.json). Independent source review also required
per-step/resumed derivatives, a genuinely heterogeneous two-head smoke, and
candidate-wheel identity. All four findings were addressed and re-reviewed;
see [review scope](review.md). The corrected final verifier passed 6/6.

The local checkout had intentional storage-cleanup deletions. Only six tiny
historical package manifests were restored unchanged for license preflight.
Those deletions and unrelated files are not part of this increment. Raw tensors,
checkpoints, binary wheels and full local logs stay private; hashes and scalar
results are public. No credential was read to prepare or validate this candidate.

## Reproduce

Use the exact v2 bundle from the
[matched-graph qualification](../2026-10-13-matched-geometry-learning-v2/README.md#reproduce):
`reference.json`, `fixture-identity.json`, `native.json` and `browser.json`.
The Python verifier independently validates both retained runtime reports before
using them. A newly generated reference must match the frozen hash; this is not
permission to replace an inconvenient oracle.

Build a wheel from the reviewed source, install it in an isolated Python 3.12
environment, and keep GPU tests sequential. `WHEEL` is the exact wheel installed
in that environment, `FIXTURE_DIR` is the private validated bundle, and `OUT`
must not already exist.

```sh
maturin build -m bindings/st-py/Cargo.toml --release --locked --features logic,kdsl
python -I -m pip install --no-deps --force-reinstall "$WHEEL"
python -I -B tools/smoke_learning_stack.py
python -I -B tests/test_release_learning_smoke.py
python -I -B tests/test_python_byte_decoder_validation.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_byte_decoder.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_attention_training.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_residual_attention_training.py
python -I -B -m examples.resident_byte_geometry_lm --updates 16
python -I -B -c 'import sys,runpy;sys.path.insert(0,"tools");sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/validate_python_byte_decoder.py "$FIXTURE_DIR" "$OUT" --wheel "$WHEEL"
cargo check --locked -p spiraltorch-py --no-default-features --features python-default,cpu
```

The recorded candidate wheel hash identifies this local measurement, not a
reproducible-binary promise: build metadata and toolchains can change wheel
bytes. The verifier rejects an in-tree extension, version mismatch or installed
native/facade bytes different from the supplied wheel.

## Release Boundary

Source metadata is 0.4.30; PyPI was still 0.4.29 during readiness inspection.
This increment does not create a tag, sign release assets or upload to PyPI.
Integrate/review the preceding Rust stack first, then this client/version change.
Official Linux, macOS universal2 and Windows wheels plus minimum-Python smoke,
signed assets and verified release finalization remain required. This local
candidate is not a substitute for that release process.
