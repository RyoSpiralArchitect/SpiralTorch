# Causal Poincare metric bias

`st_kernel_contracts::poincare` defines a causal squared-distance score bias and
its Euclidean-coordinate VJP. `ResidentTensor::causal_poincare_bias` executes it
without host readback on native WGPU and browser WASM/WebGPU. JavaScript only
loads the compiled Rust probe; it does not reimplement metric semantics.

This metric primitive also participates in the optional learned encoder in
[ResidentByteDecoder](resident_byte_decoder.md). The
[causal wave](causal_zspace_wave.md) supplies suitable interior chart coordinates;
the metric must not instead receive arbitrary projected/residual vectors.

## Distance and pullback

For curvature `-c < 0`, coordinates `x,y` must satisfy `c*dot(x,x) < 1` and
`c*dot(y,y) < 1`. Starting with the Poincare-ball distance in
[Nickel and Kiela, equation 1](https://arxiv.org/html/1705.08039#S3.E1), rescaling
the curvature and using `acosh(1+2v) = 2*asinh(sqrt(v))` gives:

```text
A = 1 - c*dot(x,x)
B = 1 - c*dot(y,y)
v = c*dot(x-y,x-y)/(A*B)
distance_squared = 4*asinh(sqrt(v))^2/c

K(v) = asinh(sqrt(v))/(sqrt(v)*sqrt(1+v)), K(0)=1
gradient_x = 8*K(v)*((x-y)/(A*B) + v*x/A)
gradient_y = 8*K(v)*((y-x)/(A*B) + v*y/B)
```

The gradients above are obtained by differentiating the displayed expression.
They are generally not negatives of one another. They are coordinate
derivatives for backpropagation through the chart, not Riemannian optimization
directions; do not multiply them by an inverse metric before ordinary VJP
composition.

Inputs are coordinates `[B,T,C]` and raw gain `[H]`. Output is `[B,H,T,T]`:

```text
bias[b,h,q,k] = -softplus(raw_gain[h])*distance_squared(z[b,q],z[b,k])  if k<=q
bias[b,h,q,k] = 0                                                   if k>q
```

Future **bias** is zero, not a masking sentinel. The consuming Attention must
still apply its structural causal mask. `raw_gain=0` means `softplus(0)=log(2)`,
not a disabled metric. Disabling geometry must be an explicit model choice.

`backward(score_cotangent)` returns coordinate and raw-gain VJPs. Both endpoint
roles and all heads are summed, not averaged. Cotangents from every consuming
block must in turn be summed before causal-wave BPTT. Curvature is a frozen
hyperparameter in this API, not a learned input with a missing gradient.

## Numerical and execution boundary

The operation rejects nonfinite inputs, points outside/on the ball and
nonrepresentable returned f32 scores/gradients. It never clips coordinates or
silently projects them back into the ball. A fixed epsilon must not create a
zero-distance/zero-gradient region near coincidence.

CPU margin evaluation preserves product residuals and compensated sums before
narrowing `1-c*norm_squared`: even exact f32 inputs can have a positive margin
that plain f64 norm-then-subtraction loses. Remaining CPU metric arithmetic and
gradient contractions use f64 internally. GPU reuses existing extended-range
`Wide` arithmetic for margins, coefficient caching and gradient accumulation.
Large-v logarithms are exponent-aware; tiny gains stay extended until their
products are formed. This is not shader-f64 or arbitrary precision.

For `v < 2^-10`, the GPU evaluates fourth-order polynomials for
`asinh(sqrt(v))^2/v` and its corresponding derivative factor `K(v)`. The first
omitted terms are O(v^5), below 1e-15 on that interval before f32 arithmetic
roundoff. The exact zero limit and one-ULP separations are tested; values on
both sides of the switch retain nonzero VJPs.

Four extended coefficients consume **64 bytes per batch/query/key pair**.
Forward has O(B*T^2*C + B*H*T^2) work. Backward first reduces head cotangents once
per causal pair, then reuses that scale for each coordinate component. Its work
is O(B*T^2*(C+H)), not O(B*T^2*C*H). The extra dispatch stores one extended
scale, **16 bytes per batch/query/key pair**, in a private tail of that
backward's gradient allocation. Returned views expose only gradients; the
scratch storage lives as long as their shared allocation. The forward tape
is not mutated and the storage-binding count is unchanged. There is no new
host readback or floating-point atomic reduction. This correctness-first materialized
path is not a memory-efficient/streaming Attention kernel or a speed result.
Binding, storage, grid and backward-output limits are preflighted before
forward. Arbitrary validated strided/offset operands are packed on the GPU.

The immutable tape captures coordinates, gains, curvature and coefficients.
Whole-operation guards join inputs, packed intermediates and every returned
gradient. Masked cotangents must still be finite; masking is not a NaN escape.
The shared probe forces a late gradient overflow and checks that binding the
returned VJPs to SGD rejects the whole update at both zero and nonzero rates,
without changing either parameter. Negative controls check the rejection
validator itself, not merely the operation under test.
Standalone tapes are reusable. A surrounding model must additionally enforce
owner/revision and latest-forward rules before binding these gradients.

## Run the checks

```sh
cargo test --locked -p st-kernel-contracts poincare
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release \
  -p st-backend-wgpu poincare -- --test-threads=1
cargo build --locked -p st-backend-wgpu --target wasm32-unknown-unknown \
  --release --example poincare_bias_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/poincare_bias_browser.wasm \
  --target web --out-dir target/poincare-bias-web
python3 -I -S -m http.server 8771 --bind 127.0.0.1
```

Open `/crates/st-backend-wgpu/tests/poincare_bias_browser.html` in a WebGPU
browser on that server. Native linker environment flags may need to be unset
for a wasm32 build. Compilation alone is not a browser execution test.

The frozen independent CPU-f32 PyTorch fixture has eight conditions. Additional
analytic/CPU controls cover a margin near 2^-72, a representable distance with
v beyond f32 range, tiny separations/gains, one-ULP separation and the series
switch. Prefix, suffix-sensitivity, guard and retained-tape controls are shared
by native and browser probes. These tests do not establish language quality,
an advantage over ordinary fine-tuning or full-model metric learning.

The byte decoder's `with_causal_geometry` composition places byte/position
tables, encoder projection, wave decay/phase, per-block/head metric gains,
residual blocks and output head under one parameter owner. Its full-model
probe checks next-byte CE through the metric path; the standalone results below
remain primitive-only evidence and are not retroactively full-model results.

The [native/browser evidence bundle](../benchmarks/results/2026-10-09-poincare-bias/README.md)
records frozen criteria, failures and repairs, runtime results and artifact hashes.
The [pair-seed cache review repair](../benchmarks/results/2026-10-09-poincare-pair-seed-cache/README.md)
adds bounded scratch preflight and retained, distinct-seed backward controls
without changing that historical record or claiming a measured speedup.
