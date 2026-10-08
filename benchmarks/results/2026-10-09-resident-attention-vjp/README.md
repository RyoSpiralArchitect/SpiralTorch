# Resident attention VJP: numerical and update evidence

This is a correctness slice, not a speed or language-quality result. Native
Metal and browser WebGPU run the same Rust kernel. The independent oracle is
CPU float32 PyTorch 2.12.1: matmul, additive biases, structural causal mask,
softmax, autograd and SGD. No models, external corpora or held-out text are used.

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

Python and full-WASM binding compilation passed, but their new public client
runtime tests have not been executed locally: less than 1 GiB of disk remains,
so full native relinking and full-WASM linking are deferred. The standalone
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
