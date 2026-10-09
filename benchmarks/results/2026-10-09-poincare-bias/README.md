# Causal Poincare pair-bias correctness

This is a metric/gradient validation, **not** full-model training, language
quality, memory-efficiency or a speed comparison. PyTorch is an independent
numerical reference here, not a timed competitor. The same Rust probe executes
on native WGPU and browser WASM/WebGPU.

## Fixed oracle

The deterministic single-thread CPU-f32 PyTorch 2.12.1 fixture has eight cases:
T=1, general multi-batch/head shapes, curvatures -0.01/-0.3/-0.75/-1/-100,
coincident positions, nearby points, and points approaching the ball boundary.
All resident coordinates, gains and cotangents use strided/offset views.
Every score, both-endpoint coordinate VJP and head-gain VJP is checked
elementwise against the fixture. CPU values/VJPs are checked independently.

The frozen tolerance is `3e-6 + 8e-5 * abs(reference)`. Fixture SHA-256:
`eceb17d0afefdaa0b01a7389978516e87aadd72a2975b65e2ed1484730d02255`.
Neither fixture nor tolerance was changed after generation. The generator
uses stock PyTorch under isolated Python, with explicit CPU tensors and no
SpiralTorch imports.

Native and browser reports are linked below. Near-boundary coordinate gradients
reach approximately 163, so their largest absolute error (approximately 1.23e-4)
must be read with the fixed elementwise relative criterion, not compared to
the absolute tolerance alone.

## Range, causality and failure controls

For `u=2^-24`, the exact f32 point
`[1-u, 2^-12*(1-u), 2^-12, u*(1-u)]` has positive ball margin
`u^3*(1-u)`. Pairing it with its negative gives squared distance approximately
10523.842841 despite the intermediate distance ratio exceeding f32 range.
Plain CPU f64 norm-then-subtraction initially rejected this point. The failing
test was retained, then the CPU margin was repaired with exact f32 squares,
curvature-product residuals and signed compensated summation. The GPU's existing
extended representation already passed this particular boundary control.

Seven additional controls require **nonzero** gradients and relative error
below 8e-5: tiny separation, tiny gain with a large cotangent, large gain with
tiny distance, one-ULP separation, and three values around the small-v series
switch. With gain=-110, curvature=-1e-30, coordinates 0 and 5e14, and cotangent
1e30, a gain VJP near -2.038e12 must survive; prematurely narrowing the gain
would erase it. No epsilon deadzone or clipping substitutes for the metric.

Changing suffix coordinates leaves prefix scores unchanged. Prefix-only
cotangents plus nonzero masked-future cotangents yield zero future coordinate
gradients, with a positive suffix-sensitivity control. Separate documents and
exact coincidence also have CPU regression tests.

Invalid coordinate/gain/cotangent families, outside/on-ball points at T=1,
good/bad/good calls and retained tapes are checked. A late gradient overflow
must invalidate both returned VJPs. Binding those gradients to a two-parameter
owner rejects the whole update at rates 0 and 0.125 without changing either
parameter. Four negative controls test the rejection checker itself. This
two-parameter rejection probe is not integration into the full decoder owner.

Independent read-only review confirmed the distance derivatives and GPU/host
range, caching and guard design. It identified the CPU boundary issue above;
the focused repair was reviewed again without remaining findings. Runtime
checks were executed separately by the parent agent. An initial strict lint
run rejected a needless Vec in a probe; it was replaced by an array rather
than weakening lint rules.

## Reproduce and inspect

- [Math, scope and commands](../../../docs/poincare_metric_attention.md)
- [Native report](native.json)
- [Browser report](browser.json)
- [Source, fixture, executable and private-log hashes](validation.json)
- [Independent reference generator](../../../tools/generate_poincare_bias_torch_fixture.py)

Build logs and tested executable/WASM artifacts are retained locally; only
results, verification records, hashes and rerun instructions are published.
Exact regression commands and counts are recorded in `validation.json`.
Whole-workspace linting, all-device equivalence, Riemannian optimization,
decoder KV caches and public Python/JavaScript model facades are not implied.

Next-byte learning still requires integrating encoder projection, wave
decay/phase and per-block gains under the byte decoder's single parameter owner,
then comparing full-model gradients and update trajectories. The direct
Euclidean path must not mask a disconnected metric branch.
