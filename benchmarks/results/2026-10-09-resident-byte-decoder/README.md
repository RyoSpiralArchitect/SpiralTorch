# Resident byte decoder: synthetic full-model learning correctness

This is a numerical and ownership validation, **not** a language-quality or
performance benchmark. Two tiny independent CPU-f32 PyTorch models are compared
against the same Rust training implementation on native WGPU and browser WASM.
The configurations/seeds differ, so their losses are not a geometry ablation.

## Conditions and results

Both models use B=2, T=4, width=4, two attention heads, FF width=6, 256 byte
symbols, learned byte/position embeddings and a LayerNorm/linear output head.
The first has one plain residual block and 18 parameter tensors. The second has
two distinct blocks, a Topos gate in the second block, position-only external
Z/pair score biases in both blocks, and 31 parameter tensors. External biases
are frozen during SGD; they are not a learned Z-Space encoder.

| Observation | Plain / 18 parameters | Topos + geometry / 31 parameters |
| --- | ---: | ---: |
| Maximum logit absolute error | 6.333e-8 | 7.451e-8 |
| Maximum parameter-VJP absolute error | 9.835e-7 | 2.236e-7 |
| Maximum embedding-output VJP absolute error | 9.686e-7 | 1.193e-7 |
| Maximum parameter error over all 16 updates | 4.694e-7 | 1.342e-7 |
| First pre-update CE | 5.46512985 | 5.62051439 |
| Sixteenth pre-update CE | 3.72382355 | 4.10000134 |

Native Metal reports Apple M4. The browser reports BrowserWebGpu / Other with
no adapter name; do not infer a reported device identity from the native run.
The result fields for both clients are identical in this observation. This
does not imply equivalence on all WebGPU devices.

The frozen elementwise tolerance is `3e-6 + 5e-5 * abs(reference)`. Every parameter
gradient and every parameter value after each update is compared, not only the
maximum above. Four external bias VJPs are checked in the geometric model.
All 16 training updates are queued before loss/receipt/parameter readback.
Initial arbitrary-cotangent VJPs are compared separately; per-step training
logits and VJPs are not all exported in this fixture.

Prefix outputs are unchanged under suffix replacement, changing a different
document leaves the other row unchanged, and prefix-alone execution matches its
extended counterpart (observed maximum errors: zero). Prefix-only loss has zero
future-byte/embedding-output gradients, with a positive suffix sensitivity
control. This tests the default model and these position-only biases, not the
causality of arbitrary external geometry producers.

Nine ownership controls per model pass: foreign same-counter tape rejection,
superseded tape rejection, retained output under changed input, final-block
invalid-bias preflight, good/bad/good VJPs, retained gradients under failure and
changed cotangent, all-or-none update rejection, stale gradients, and recovery.
Logical gradient shapes and per-block optional bias presence are checked.

Additional native checks: all 70 resident unit tests pass, including candidate
overflow at every one of 12 parameter slots in a smaller two-block byte model,
and isolated late byte/position scatter failures at zero and nonzero SGD rates.
CPU-only byte-plan tests: 4 pass. Existing residual attention regression: all 60
VJP conditions, two 32-update trajectories and 12 guard checks pass unchanged.

Formatting and diff-whitespace checks pass. An additional strict Clippy run on
the local 1.97 toolchain does **not** pass: dependency linting stops on seven
pre-existing `clippy::chunks_exact_to_as_chunks` attributes unknown to that
toolchain; a `--no-deps -D warnings` run reaches st-nn and stops on 23 warnings
in unchanged language/layers/z_rba/zspace_coherence code. No lint allowances or
unrelated source cleanups were added. This is not a claim of a clean Clippy gate.

## Reproduce and inspect

- [Rust contract and commands](../../../docs/resident_byte_decoder.md)
- [Native report](native.json)
- [Browser report](browser.json)
- [Source, fixture, executable and private-log hashes](validation.json)
- [Frozen oracle generator](../../../tools/generate_resident_byte_decoder_torch_fixture.py)

The fixture was generated using deterministic, single-threaded PyTorch 2.12.1 on
CPU/f32, without SpiralTorch imports. Generation refuses to overwrite a file;
normal tests use the checked-in fixture rather than regenerating it. Native and
browser probes use the same source and fixture. Exact source hashes identify the
tested worktree relative to its parent commit; build logs and tested executables
are retained locally, with hashes here instead of machine-local paths.

Independent source review identified two test weaknesses: an identical-input
retention control and missing non-parameter gradient shape assertions. Both were
strengthened and reviewed again before the final native/browser runs. Fixture
bytes and numeric tolerances did not change. No language corpus, training-quality
claim, throughput claim, causal geometric encoder, checkpoint/resume API, or
public Python/JavaScript full-model facade is implied by these results.
