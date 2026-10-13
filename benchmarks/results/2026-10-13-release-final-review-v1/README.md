# Python 0.4.30 Integration Review

This record qualifies local macOS 14+ arm64 wheels, not a published release.
Core source: `1463627f443f59fed42d46a763ae93d583ab23fa`.
Client/build source: `cc2f349b420dd53c77aef41581a8edb200b55d25`.
The later evidence commit does not change either wheel's code.

## Review And Repairs

Three read-only independent reviews covered the Python API, model/checkpoint
ownership, and corpus/calibration contracts. This was ordinary local review,
not disk-blinded review or an exhaustive mathematical audit. The parent
inspected shared backend and release integration separately.

Two reproduced P2 findings were closed and re-inspected by their reviewers:

- The contextual query graph accepted a full stage/parameter budget before
  appending its shared-logit stage. Portable construction and checkpoint import
  now validate the same augmented graph that execution compiles. Tests cover
  4095/4096 stages and 8191/8192 slots, including accepted boundaries.
- A positive `candidate_rms` mutation could pass preparation verification.
  Every block/head's actual float32 softplus-gain transformation is now bound
  to the reference/candidate RMS ratio. All four retained v4/v5 native/browser
  preparation packets still pass; their original bytes are not rewritten.

The query tape's large enum member was also boxed, and introduced byte-study
lint warnings were removed. No model equation, default, or comparison gate
was changed by this follow-up.

## Results

| Check | Result |
| --- | --- |
| CPU-only Rust byte decoder | 26 pass |
| Rust byte model/study, real WGPU tests enabled | 79 pass |
| Configured Python corpus/verifier suites | 81 pass, 6 optional Torch skips |
| Installed default WGPU wheel API | 8 pass, 1 CPU-artifact-only skip |
| Installed CPU-only wheel API | 6 pass, 3 GPU opt-in skips |
| Installed learning-stack / identity test suites | 7 / 6 pass |
| Final Python vs frozen native/browser/CPU fixture | 6 cases, 96 updates, 720 geometric step VJPs |
| Exact Python checkpoint continuation | 54 resumed updates |
| Wheel-packaged example | 16 updates, exact restored prediction |

Both local wheels are `spiraltorch-0.4.30-cp38-abi3-macosx_14_0_arm64.whl`
in separate directories, not interchangeable files. The imported native
extension and facade are hash-bound to the corresponding wheel, excluding
checkout shadowing. The CPU wheel constructs all eight routes and explicitly
rejects resident WGPU compilation; it does not claim CPU execution of this LM.

Default WGPU SHA-256:
`4fe761aec5a057062f60de05a259c028af0a2a4f380319faaffb8aa415502181`.
CPU-only SHA-256:
`24cbcdc254fd04c1a692e42c93af89cb9a5c940f3b68098f8550f950dfe86c57`.

Native/WASM Clippy complete with zero warnings in the byte-model/study scope.
Whole-crate strict Clippy is **not green**: 40 unique native warnings remain
in unchanged mainline files. Earlier strict failures and incorrect isolated
test-environment invocations remain in the raw receipt hashes; they are not
relabeled as passing tests. No lint allowance or CI weakening was introduced.

The relative geometry-gradient gate remains `0.002`. Fresh Python results
use the retained [Wave-qualified native/browser references](../2026-10-13-wave-expansion-v1/README.md),
not fresh browser execution after the portable validation/refactor. These
synthetic mechanics and repeated-window losses do not establish language
quality, a best geometry, pretrained HF support, or a speedup.

## Reproduce

Use the pinned client source and separate virtual environments/output folders
for the two feature configurations. The [release runbook](../../../docs/ops/release.md)
defines the build platform and deployment target. Build default WGPU with
`maturin build --release --locked --features logic,kdsl`; build CPU-only with
`--no-default-features --features python-default,logic,kdsl` in addition.
Use `--manifest-path bindings/st-py/Cargo.toml` when building from repo root.

After installing each corresponding wheel, run:

```bash
python -I -B tools/smoke_learning_stack.py
python -I -B bindings/st-py/tests/test_nn_byte_decoder.py -v
```

Set `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1` for the WGPU API run only.
Run the packaged example with
`python -I -B -m examples.resident_byte_geometry_lm --mode query_mixture --updates 16`.
The Wave record explains how to regenerate the frozen reference inputs.
Against that fixture directory, invoke `tools/validate_python_byte_decoder.py`
with its required `--wheel` argument, an unused output path, and only `tools`
added to the isolated interpreter's module path for sibling validator imports.

`validation.json` records raw hashes and source identities;
`python-parity.json` keeps every scalar comparison. Original tensors, binaries
and logs stay local. `SHA256SUMS` checks every public result in this directory.
Official Linux/Windows/universal2 wheels, exact-head CI, tag and PyPI publication
remain separate release gates, not claims made by this record.
