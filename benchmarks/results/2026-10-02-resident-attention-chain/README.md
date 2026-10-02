# Frozen QKV / Z-RBF Attention / Output Projection

**Scope: full-chain forward correctness, not throughput or model quality.**

This connects existing NN projection parameters and the existing Rust Z-RBF
product kernel to resident WGPU attention. Native and browser clients execute
the same Rust orchestration. Input activations are uploaded once per scenario;
the probe only reads back after the final output projection. Geometry is host
metadata, uploaded separately and reused. GPU packing, merging, validation and
multiple submissions remain; this is not a single fused kernel.

## Matched Controls

| Scenario | Batch | Sequence | Input width | QKV width | Heads | Output width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `small_tail` | 2 | 5 | 6 | 6 | 2 | 4 |
| `transformer_width` | 1 | 17 | 32 | 32 | 4 | 32 |

Each scenario has plain, zero-geometry and nonzero Z-RBF conditions, both
unmasked and causal: 12 cases in total. All input values, four projection
weights/biases, token indices and reference outputs are frozen in the fixture.
The geometry formula is independently evaluated in PyTorch, then separately
checked against `ZRBFAttention::kernel_bias`. The output reference includes all
projections and the identical additive bias; plain PyTorch is not used as the
numerical oracle for a different, biased function.

| Path | Output cases passed | Largest output absolute error | Largest geometry error |
| --- | ---: | ---: | ---: |
| Native Metal / Apple M4 | 12/12 | 1.30385160446167e-8 | 5.960464477539063e-8 |
| Browser WebGPU / WASM | 12/12 | 1.6763806343078613e-8 | 0 |

Every output element must satisfy `abs(actual - expected) <= 3e-6 + 3e-5 * abs(expected)`
and be finite. These small analytic values do not establish long-context
numerical stability, speed, general PyTorch equivalence or trained-model quality.
Browser adapter metadata exposes only `BrowserWebGpu` / `Other`; its physical
GPU identity is not inferred from the native machine or the observed errors.

## Verification And Limits

Execution source: `ab2fa7cef8e67a0f707d981aa1dab34040a7eeee`.
Reference: PyTorch 2.12.1, CPU float32, math SDPA, one thread, no dropout.
Rust builds/tests use 1.98.0; formatting uses the repository's
`nightly-2026-04-15`. No model or corpus downloads are needed.

- 43 kernel-contract tests passed, including N-D axis-selection views.
- 66 resident-tensor tests passed on the real native GPU with runtime tests enabled.
- 14 NN attention unit tests and the 12-case native integration test passed.
- The final wasm32 build passed and the local browser probe passed all 12 cases
  plus both geometry comparisons. Browser execution is not claimed as a CI run.
- CPU-only `st-nn --no-default-features` check, pinned formatting, fixture
  regeneration byte comparison and whitespace checks passed.
- Native strict Clippy passed for `st-kernel-contracts` and `st-backend-wgpu`.
  Ordinary all-target `st-nn` Clippy passed with existing warnings. Making all
  NN warnings fatal still fails in unrelated legacy code; this is not a
  workspace-wide strict-lint success. No unrelated warnings were suppressed.

Tests additionally check the legacy Z-RBF mean, COW parameter freezing,
column-major parameter import, rectangular geometry and head order, invalid
geometry rejection, noncontiguous inputs, retained outputs and deferred
non-finite guards across repeated forwards.

The compiled plan is a frozen inference snapshot. It does not update source
weights, return variance/entropy, own a KV cache, implement GQA or backward,
or expose a new Python facade. Head dimension is limited to 256. Geometry-bias
storage is quadratic and its CPU construction is not included in any timing
claim. There are no timings or CUDA observations in this record.

## Reproduction And Provenance

- [API and commands](../../../docs/resident_zspace_attention.md#full-chain-verification)
  cover native and browser execution, fixture regeneration and the runtime-test gate.
- [Fixture](../../../crates/st-nn/tests/fixtures/attention_chain_torch.json)
  records all inputs and expected outputs; the generator is a test oracle, not
  a second production implementation of Z-space policy.
- `native.json` is extracted from the completed native test log;
  `browser.json` is the final browser DOM report. Raw validation logs stay local.
- `SHA256SUMS` uses paths relative to this result directory for these reports and
  the README; run `shasum -a 256 -c SHA256SUMS` from this directory.
- `source-sha256.json` binds relevant source/fixture hashes and Cargo.lock to
  the execution revision. Resolve those paths **at that revision**, not at a
  later working tree. Older primitive evidence remains frozen at its own revision.

The next performance test should time this complete chain with resident inputs
and, separately, uploads/readback included, using matched shapes, controls and
hardware. Numerical agreement does not itself establish an optimization win.
