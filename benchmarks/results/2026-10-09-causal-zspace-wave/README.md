# Causal Z-Space wave: state, pullback and resident learning

This record validates a causal complex-state filter and its interior
Poincare-ball coordinate chart, **not** geometric Attention, language quality,
throughput, a full streaming decoder or an advantage over ordinary fine-tuning.
The Rust CPU contract, native WGPU and browser WASM execute the same operation.

## Fixed comparisons

The independent deterministic single-thread CPU-f32 PyTorch 2.12.1 fixture has
ten cases with B=1..3, T=1..31 and C=2..8. Curvatures include -0.01, -0.3,
-0.75, -1, -1.25, -2 and -100. Feature-only, terminal-state-only and mixed
cotangents, zero initial states and saturated parameters are covered. Every
resident input and cotangent is exercised as a noncontiguous offset view.

The frozen elementwise tolerance is `3e-6 + 8e-5 * abs(reference)`. The fixture
SHA-256 is `6d999fa45310ada0514624a003712c699f5d81418b947e8722013cd038ece721`.
Neither fixture bytes nor tolerance changed after failures. Tests use the
checked-in fixture; the generator does not import SpiralTorch.

| Observation | Native Metal | Browser WebGPU |
| --- | ---: | ---: |
| Maximum forward absolute error | 4.769e-7 | 4.769e-7 |
| Maximum VJP absolute error | 1.550e-6 | 1.550e-6 |
| Maximum parameter/gradient error over 16 updates | 1.193e-7 | 1.193e-7 |
| First pre-update combined MSE | 0.0654248521 | 0.0654248521 |
| Sixteenth pre-update combined MSE | 0.0050896695 | 0.0050896697 |
| Two-chunk VJP maximum absolute error | 3.726e-9 | 3.726e-9 |

The native adapter reports Apple M4 / Metal. The browser reports
BrowserWebGpu / Other without an adapter name. Numerical result fields are not
all identical; these observations do not prove equivalence on every device.

All four VJPs (drive, raw decay, raw phase and initial state) and both outputs
are compared elementwise. Sixteen SGD updates learn decay and phase against a
synthetic feature/terminal-state target. Every pre-update loss and both
parameter gradients, and every post-update parameter, are checked after all
updates have been queued. This is primitive parameter learning, not next-byte
CE or full-model encoder training.

## Adversarial controls and rejected implementations

Nine independent f64 analytic cases cover axis, diagonal and oblique radial
seeds, positive/negative one-ULP perturbations, negative orientation, a large
exponent, a rotation-contraction cancellation and mixed cotangent exponents.
Each checks all four CPU/GPU VJPs. The large-exponent test additionally requires
a nonzero tiny derivative and a relative-error bound, so the ordinary absolute
tolerance cannot hide its disappearance. The one-ULP phase derivatives are
approximately +47.274685 and -47.274685; the mixed-exponent phase derivative is
approximately 1.477334.

An initial direct normalized-vector subtraction lost the radial chart
derivative at a large state. A compensated f32 repair passed seven controls but
still erased a recoverable small cotangent and invented a phase derivative
when chart components were narrowed before rotation contraction. Independent
source review identified these latter two defects. The final implementation
keeps per-scalar extended adjoints throughout chart pullback and BPTT, reusing
the existing GPU `Wide` arithmetic and using f64 internal CPU adjoints. These
are not arbitrary-precision operations or shader-f64. No near-radial deadzone,
test-specific branch or looser tolerance was introduced.

An earlier symmetric changed-cotangent sensitivity control was inert; the probe was
corrected to use an asymmetric cotangent. Two intermediate test builds also
failed because an analytic control lacked explicit float types. These failed
candidates remain represented by private log hashes rather than being relabeled
as passing runs. Final independent source review found no remaining actionable
issue in its bounded scope; runtime verification was performed separately.

Prefix outputs and future gradients pass causal controls. Two-chunk (2+3)
forward and reverse-state propagation match the unchunked operation within
tolerance. This does not cover Attention KV caches or universal bitwise chunk
equivalence: public state/cotangent seams are f32.

Guards cover four forward operands, two cotangents, good/bad/good isolation,
retained results, late batch-reduction invalidation of all four VJPs and
all-or-none update rejection at learning rates 0 and 0.125. Four negative
controls verify that the rejection checker rejects incomplete evidence.

## Reproduce and inspect

- [Rust contract and native/browser commands](../../../docs/causal_zspace_wave.md)
- [Native report](native.json)
- [Browser report](browser.json)
- [Source, fixture, executable and private-log hashes](validation.json)
- [Independent PyTorch generator](../../../tools/generate_causal_zspace_wave_torch_fixture.py)

The final regression suite and exact commands are enumerated in `validation.json`.
Build logs and tested executable/WASM artifacts are retained locally; only
results, validation records, hashes and rerun instructions are public. Source
hashes describe the tested worktree relative to its parent commit, excluding
unrelated human-deleted benchmark archives.

The next integration must evaluate genuine Poincare metric bias on these chart
coordinates, propagate both endpoint cotangents and learned head gains, and
place the encoder and decoder under one parameter owner. A Euclidean residual
path alone must not disguise a disconnected metric branch.
