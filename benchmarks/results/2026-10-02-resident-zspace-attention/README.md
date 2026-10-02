# Resident Z-space attention: forward numerical evidence

**Scope: correctness, not a speed or model-quality benchmark.**

The resident WGPU kernel and browser WASM client evaluate the same operation as
PyTorch's float32 CPU SDPA math backend, with identical Q/K/V, scale, optional
key-wise Z-bias, pairwise bias, and explicit causal query offset.

## Observed results

| Path | Cases | Maximum absolute error against PyTorch |
| --- | ---: | ---: |
| Native WGPU / Metal, Apple M4 | 20 | 8.940696716308594e-8 |
| Browser WebGPU / WASM | 20 | 8.940696716308594e-8 |

The browser adapter reports `BrowserWebGpu` and device type `Other`; it does not
expose its physical GPU name. Do not infer hardware identity from matching errors.
Every element is checked with `abs(actual - expected) <= 3e-6 + 3e-5 * abs(expected)`.

Each of five scenarios has four controls: plain, key bias, pairwise bias, both.

| Scenario | B | H | Q | K | D | Causal query offset |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Rectangular | 2 | 2 | 3 | 5 | 7 | none |
| Prefill | 2 | 2 | 5 | 5 | 7 | 0 |
| Cached decode | 2 | 2 | 1 | 5 | 7 | 4 |
| Cached chunk | 2 | 2 | 3 | 5 | 7 | 2 |
| Head tail | 1 | 1 | 2 | 3 | 65 | 1 |

These are small numerical fixtures, not long-context or trained-model evidence.
Native tests also exercise D=1/7/64/65/127/256, sliced/permuted Q/K/V, broadcast
biases, empty queries, structural future masking, cross-device rejection,
downstream guard propagation, scale signs/zero, large finite scores, and ownership.

## Files and reproduction

- `browser.json`: captured DOM report from the actual browser probe.
- `native.json`: extracted per-case errors and completed native test results.
- `SHA256SUMS`: repository-relative hashes of reports, fixture, generator and
  the implementation used for the observation; verify from repository root.
- [Frozen oracle](../../../crates/st-backend-wgpu/tests/fixtures/resident_attention_torch.json):
  all inputs and expected outputs, generated with PyTorch 2.12.1.
- [API and commands](../../../docs/resident_zspace_attention.md): build, native
  tests and browser probe.

Regenerate the independent CPU reference in an environment with PyTorch:

```bash
env SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0 \
  python3 -I tools/generate_resident_attention_torch_fixture.py \
  --output crates/st-backend-wgpu/tests/fixtures/resident_attention_torch.json
```

These environment variables disable optional global patches before Python
startup. Inputs are deterministic analytic values, not a random sample. The
reference uses the CPU math SDPA backend, no dropout, explicit offset masks and
no model/data downloads. Bias addition and reduction order may differ in rounding.

Rust 1.97.0 passed 42 contract tests, 66 resident tensor tests on the real Metal
adapter, and 12 attention-filtered tests including the legacy shader tests.
Strict native Clippy uses the installed Rust 1.98.0 toolchain: 1.97's Clippy
rejects pre-existing `chunks_exact_to_as_chunks` allow attributes in unrelated
files. Both native and wasm32 all-target strict Clippy passed on 1.98.0, along
with the repository-pinned formatter check. No global toolchain configuration
or unrelated lint code was changed.

The first browser attempt failed although native/Naga validation passed: a
decimal f32-MAX literal was rejected by the browser WGSL compiler. This was fixed
using the exact bit pattern in both attention shaders, and all 20 browser cases
were rerun successfully. The probe captures GPU validation errors, not just
potentially zero-filled outputs.

No Python facade, decoder graph, attention backward, automatic geometric-bias
producer or cache manager is claimed here. No speedup, training improvement,
general framework equivalence, or CUDA result follows from these fixtures.
