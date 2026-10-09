# Resident attention VJP: numerical and update evidence

This is a correctness slice, not a speed or language-quality result. Native
Metal and browser WebGPU run the same Rust kernel. The independent oracle is
CPU float32 PyTorch 2.12.1: matmul, additive biases, structural causal mask,
softmax, autograd and SGD. No models, external corpora or held-out text are used.

Public-client follow-ups below also pass: the full WASM package in the browser
and the default-feature Python wheel on macOS WGPU. Earlier pending statements
are historical checkpoints, not the current client qualification status.

## Results

- Both backends pass all 18 conditions in canonical and strided layouts: 36
  forward/gradient comparisons per backend. The largest absolute gradient error
  in these ordinary-scale fixtures is `3.5762786865234375e-7`.
- Both execute all 16 synthetic MSE/SGD steps while Q/K/V and both biases remain
  on the GPU. Validation readbacks happen after the update sequence. The largest
  final parameter difference from PyTorch is `2.9802322387695312e-8`.
- MSE before the first update is about `0.15879469`; before update 16 it is
  `0.14808737`. This small synthetic decrease is not an FT or model-quality gain.
- Each backend also passes 12 cropped/empty inherited-failure cases, two
  wide-range Rust-reference cases and three cancellation/reduction-order cases.
  Wide-range checks use relative as well as absolute tolerance; their large
  absolute errors must not be compared with ordinary-scale fixture errors.
- Native attention regression: 25 passed, one existing benchmark ignored.
  All eight new VJP tests execute with the real-GPU opt-in enabled.

The common elementwise admission bound is `atol=3e-6 + rtol=5e-5 * abs(reference)`.
It was fixed before the comparisons. Full condition results are in
[native.json](native.json) and [browser.json](browser.json); source/oracle/build
hashes and verification scope are in [validation.json](validation.json).

## Independent Review

A read-only independent review found that backward initially recomputed every
dot with 64 lanes, even where forward selects eight. For finite cancellation
inputs `[2^24, 1, -2^24]`, those orders can produce scores 1 and 0, a real change
in the softmax distribution rather than a harmless ULP difference.

Backward now obtains the forward `key_tile(spec)` choice and uses the same
per-lane accumulation and reduction order. Regression checks use 127, 128 and
131 keys and compare an analytic control, actual forward probability and `dV`.
The fix passed native and browser execution and the independent static re-review;
no actionable P1/P2 remained in that scoped review.

## Boundaries And Reproduction

See [the API/runbook](../../../docs/resident_attention_learning.md). The generator
refuses fixture overwrites; a separate regeneration with Python startup hooks
disabled was byte-identical. Preserve the committed fixture and its SHA-256.

At the initial checkpoint, Python and full-WASM binding compilation passed, but
their new public client runtime tests had not been executed locally: less than
1 GiB of disk remained, so full native relinking and full-WASM linking were
deferred. The standalone
browser result is real WebGPU kernel execution, not evidence that the full WASM
package's public JS handles were exercised. Their regression tests are included.

The local Clippy toolchain does not recognize an existing
`clippy::chunks_exact_to_as_chunks` allowance in unrelated established kernels.
The focused check passed with `-D warnings -A unknown-lints`; no repository-wide
lint policy was weakened. CI's own toolchain remains an independent gate.

No claim is made about CUDA/Furnace, full decoder training, projection ownership,
KV-cache management, throughput, geometry quality, or PyTorch superiority. These
remain separately measured follow-ups. Version 0.4.29 is unchanged; this API has
not been republished to PyPI.

## CI Follow-Up

CI on `b68ebb97` rejected a constant-size `chunks_exact(5)` in the new VJP test.
It now uses `as_chunks::<5>()` and asserts that no remainder was discarded.
The real-Metal attention suite again passed 25 tests with one existing benchmark
ignored. Local Clippy now also checks **all targets**, including unit tests, with
the same toolchain-only `-A unknown-lints` limitation. CI policy is unchanged.

The original `validation.json` Rust/Clippy versions match the outer routing
directory's toolchain, not the actual checkout's. Those two provenance fields
are withdrawn. [ci-followup.json](ci-followup.json) records the checkout-selected
Rust/Clippy 1.97.0 and fresh validation; it does not retroactively regenerate the
browser result. Original numerical receipts and hashes remain unchanged.

The shipped TypeScript surface now includes the attention methods and both new
handle types. A CI check compares their contract with freshly generated
wasm-bindgen declarations. Locally, the shipped declarations passed strict
TypeScript checking and four negative contract mutations were rejected. Generated
declarations were not tested locally. Full public
Python/WASM GPU execution remains deferred pending disk-space approval.

## Full WASM Client Follow-Up

The public WASM gap is now closed for all 18 fixture cases at PR head `d7e1ffe2`.
CI run `37855393243` built the full `webgpu` package, passed its generated and
shipped declaration checks, and uploaded artifact `11583937677`. Its ZIP digest
was verified against the CI upload receipt before extraction. The package was
then executed in the browser through `resident_attention_clients.html`, not the
standalone backend example. No local rebuild or cache deletion was necessary.

[public-wasm-ci.json](public-wasm-ci.json) contains every condition and error;
[public-wasm-ci.manifest.json](public-wasm-ci.manifest.json) pins the CI head,
artifact, module/WASM/source/fixture hashes and exact validation scope. Maximum
absolute gradient error is `3.5762786865234375e-7`. Retained gradient tensors were
read after freeing the parent gradient container, inputs and bias container.
The page also rejected missing and cross-origin modules rather than reporting a
successful check. Browser console inspection found no relevant warnings/errors.

The artifact has one-day CI retention; its original ZIP and raw result are also
kept locally. Rebuild the qualified source with the pinned lockfile/CLI to repeat
the check after expiry. The public-client test covers canonical forward/VJP and
handle ownership, not another SGD/strided/wide-range or performance experiment.
Public Python GPU execution is still pending in macOS CI at this checkpoint.

## Full Python Client Follow-Up

The macOS WGPU job in the same run completed successfully. It built and installed
the full default-feature wheel, then ran all three public attention tests with
no skips. The GPU fixture test checks all 18 conditions, rejects a CPU adapter,
and reads retained outputs/gradients after dropping their parent handles. The
other tests cover strict causal offsets, upstream shape and NaN scale
rejection, root/submodule aliases and the opaque gradient constructor.

[public-python-ci.json](public-python-ci.json) records the qualified head, source
hashes, visible terminal test log and all ten successful CI check results.
The diagnostic log was truncated, but the three individual results and final
`Ran 3 tests` / `OK` summary were all visible. Exact adapter hardware and wheel
hashes were not recorded; the fixture test asserts a non-CPU adapter.

This closes both public-client runtime gates for `d7e1ffe2`. Evidence-only
follow-up commits still require their own current-head CI before merge. The
installed CI wheel uses version 0.4.29 but is a source build, not a replacement
of the already-published PyPI wheel. No cache or historical evidence was deleted.
