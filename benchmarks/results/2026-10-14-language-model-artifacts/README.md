# Source-Bound Language Model Artifacts

Source-only qualification after 0.4.31: an executable deployment unit, not a
model registry, promotion decision, quality experiment or new PyPI release.
The private candidate wheels retain 0.4.31 metadata but are **not** the published
0.4.31 binaries and must not be uploaded over that immutable release.

## What Is Bound

Rust's `ByteCorpusModelArtifact` binds one selected byte-LM model, including
learned geometry and mixture/query routers, to its exact study request, full
study checkpoint and explicit `fixed_window_frozen.v1` operating policy.
Import requires a separately trusted exact-byte SHA256 pin. Source validation
checks all cases, not just the selected model. A constrained inference owner
has no SGD, mutable parameter or external-bias API, and retained forwards carry
their original artifact ID. Python and the standalone WASM client delegate
these semantics to Rust; existing HF adapter IDs are unchanged.

The inference owner reuses existing forward kernels and retained tapes. This
does not add a KV cache, streaming decoder, generation sampler, CPU inference
fallback, automatic activation/rollback or inference-speed improvement.
The packaged `st-wasm` module does not yet expose these standalone example APIs.

## Measured Results

The retained v8 archives each contain 75 cases at update 128. Before evaluation,
this qualification selected the first ordinary arm and the first learned/frozen
query-router pair from each native/browser archive: **six model replays**, not
all 75 models' inference. It applied **zero new training updates**.

Every selected model's restricted-owner and fresh-process logits, loss and
model JSON exactly matched the direct checkpoint owner on the same local
Apple M4/Metal device. Each forward contained 16,384 logits. Altered imports
were rejected without changing the existing owner; all original source bytes
remained unchanged. Full per-case identities and outcomes are in
[`qualification.json`](qualification.json).

The actual browser imported the completed 75-case study and ran a fresh
WebGPU inference owner for `mixture-11-0-train-query-train`. Its reported backend
was `BrowserWebGpu`, device type `Other`, loss `3.648268699645996`, and 16,384
finite logits. The downloaded complete artifact's SHA256 matched Python:
`628a805ca6b6379c170481566872335d087ea38770631f431697c157a311172f`.
See [`browser-observation.json`](browser-observation.json). Browser logits were
not independently hashed; this is not a cross-device bitwise-logits claim.

One historical browser frozen-query loss differs from current native replay by
about 3e-7 (one f32 ULP after decoding the historical f32). The per-case evidence
retains that difference. Same-device/fresh-process equality and historical
cross-backend comparison are separate checks; no tolerance was widened.

Final validation counts and binary/log identities are in
[`validation.json`](validation.json). Independent read-only review found an
empty-name import/export mismatch, then confirmed its fix. Installed-wheel
tests separately caught missing native `nn.__all__` exports; those were fixed
and retested. A second review caught default output permissions in the probe;
its output directory/model now explicitly use owner-only `0700`/`0600` modes.
Initial failed/interrupted attempt logs remain in the private
archive and are not represented as successful checks.

The first PR CI also caught a stale release-smoke assertion expecting eight
native study exports instead of ten. The failure was reproduced locally, the
explicit expectation updated, and negative tests extended to both new classes
on the native and facade surfaces. The installed CPU-only and WGPU release-smoke
suites then each passed all nine tests, including dependency-free execution.
Only tests and evidence changed in this follow-up; native binary identities above
remain valid. Independent static review found no remaining issue in the fix.

## Reproduction And Limits

Use the [Python build guide](../../../docs/python/build-and-validation.md) and
[artifact API guide](../../../docs/python/byte-model-artifacts.md). Install
separate CPU-only/WGPU candidates; do not use `PYTHONPATH` to shadow the native
extension. The public synthetic tests exercise training and retained inference:

```sh
cargo test --locked -p st-nn --lib resident::byte_study --no-default-features
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-nn --lib resident::byte_study::artifact --no-default-features --features wgpu -- --test-threads=1
python -I -B bindings/st-py/tests/test_nn_byte_study.py -v
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_byte_study.py -v
```

For historical reproduction, retain the original v8 `input/request.json`,
`native-checkpoint-128.json` and `browser-checkpoint-128.json` with the hashes
in `qualification.json`. The following read-only-source probe uses the installed
candidate and writes only to a new, private output directory:

```sh
python -I -B benchmarks/results/2026-10-14-language-model-artifacts/verify_artifacts.py REPO_ROOT PRIVATE_V8_ARCHIVE NEW_PRIVATE_OUTPUT
```

Build the standalone WASM example using the
[browser recipe](../../../docs/byte_corpus_learning.md#browser-execution), serve
only the qualification assets on loopback, and open
`byte_learning_resume_browser.html?resume=1&stop=128&case=mixture-11-0-train-query-train`
with the completed browser checkpoint. Inspect `data-status=completed`, backend,
artifact ID and finite-output checks, then compare the downloaded artifact hash.

Raw corpus, model weights and build outputs remain private. Public hashes are
integrity references, not authentication or proof of historical training. The
original private records are necessary for exact historical reproduction; the
synthetic tests can be run without them. Identity/restore correctness does not
show better language quality or deployment readiness. The next MLOps layer is
held-out language plus geometry-health gates and tested activation/rollback,
not a dashboard that labels unchecked candidates ready.
