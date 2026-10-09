# Causal Z-Space wave primitive

This is the learned state/chart seam between a dictionary-free byte model and
future geometric Attention. It is not yet a whole-model encoder integration.
The fixed 256-byte alphabet in the [byte decoder](resident_byte_decoder.md) is
dictionary-free, not an absence of discrete symbols.

## One Rust contract, two execution clients

`st_kernel_contracts::causal_wave` owns the CPU specification and pullback.
`st_backend_wgpu::resident_tensor::causal_wave` implements the same operation on
native WGPU and browser WASM/WebGPU. The browser test calls the Rust operation;
JavaScript does not reconstruct the recurrence or its derivatives. Python and
public JavaScript facade methods for this primitive are not provided yet.

Inputs are a finite drive `[B,T,2P]`, raw decay `[P]`, raw phase `[P]`, and explicit
raw initial state `[B,2P]`. Empty axes, odd channel counts, invalid curvature,
length mismatches and nonportable addressing are errors. Pairs of adjacent
channels are real/imaginary components of a complex state.

```text
rho[p]   = 0.99 * sigmoid(raw_decay[p])
theta[p] = pi * tanh(raw_phase[p])
s[b,t,p] = rho[p] * rotate(theta[p], s[b,t-1,p])
           + (1-rho[p]) * drive[b,t,p]

radius = 0.95 / sqrt(-curvature), curvature < 0
z[b,t] = radius * s[b,t] / sqrt(1 + dot(s[b,t], s[b,t]))
```

The norm is over **all `2P` coordinates**, not independent two-dimensional
disks. Curvature and the interior ratio have different roles. The returned
`z` is a smooth interior Poincare-ball coordinate chart, not an exponential
map. The recurrence and SGD are Euclidean parameter operations, not a geodesic
recurrence or Riemannian optimizer.

The caller supplies/reset state at document boundaries. There is no global
state, token-boundary inference, full-text DFT, hidden reset or implicit
truncated BPTT. Each output at time `t` depends only on its initial state and
drive through `t`.

## Pullback and ownership

Both returned paths are differentiable: the chart features and the final raw
state. `backward(feature_cotangent, terminal_cotangent)` returns drive, raw-decay,
raw-phase and initial-state VJPs. Supply a zero terminal cotangent only when the
loss has no continuation-state dependency. Parameter reductions sum across
time and batch; they do not average.

```rust
use st_kernel_contracts::causal_wave::{CausalWaveForward, CausalWaveSpec};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = CausalWaveSpec::new([1, 3, 2], -1.0)?;
    let forward = CausalWaveForward::new(
        spec,
        &[0.1, -0.2, 0.3, 0.2, -0.1, 0.4],
        &[0.0], &[0.2], &[0.0, 0.0],
    )?;
    let gradient = forward.backward(&[1.0; 6], &[0.0; 2])?;
    assert_eq!(gradient.raw_phase.len(), 1);
    Ok(())
}
```

The resident API has the same shapes and explicit cotangents:

```rust,ignore
let forward = drive.causal_zspace_wave(&raw_decay, &raw_phase, &initial_state, -1.0)?;
let gradient = forward.backward(&feature_cotangent, &terminal_cotangent)?;
// gradient.drive(), raw_decay(), raw_phase(), initial_state()
```

This primitive owns an immutable, reusable tape, not an optimizer. Its decay
and phase gradients can be bound to a `ResidentParameters` snapshot. The shared
probe queues 16 such all-or-none SGD updates before observing any result.
Full-model integration must put the encoder, embeddings, blocks and output
head under the **same** parameter owner rather than create a second optimizer.

For two chunks, pass the first final state into the second forward. Backward
the second chunk first, feed its initial-state VJP into the first chunk's
terminal cotangent, then add the two parameter VJPs. This proves state/filter
chunk equivalence, not a streaming decoder: Attention still needs its own
position/KV-cache contract, and graph-autograd workspace/tape lifetime must be
handled separately.

## Numerical and failure boundaries

The forward chart scales the state before taking its squared norm. For large
states, evaluating `g - unit * dot(unit,g)` directly loses the small radial
eigenvalue. The pullback instead uses an extended-precision pivot decomposition:

```text
e[i] = (g[i]*s[p] - g[p]*s[i]) / s[p]  (p is a maximum-magnitude coordinate)
q = 1 + dot(s,s)
J*g = R * (e - s*dot(s,e)/q) / sqrt(q)
      + R * (g[p]/s[p])*s / q^(3/2)
```

At zero state the derivative is simply `R*g`. CPU uses f64 internal adjoints;
GPU reuses the existing LayerNorm/Attention `Wide` arithmetic: a three-component
significand with a per-scalar extended exponent, not shader-f64. Product
differences use integer-defined products and compensated sums, not an
assumption that WGSL `fma` is fused. Adjoints stay extended through the reverse
scan and rotation contraction; narrowing each chart component earlier invents
phase gradients even for a mathematically radial seed. Global cotangent scaling
is avoided because it can erase a small but recoverable component.

This remains O(`B*T*C`) rather than evaluating a dense Jacobian, but carries
extra arithmetic and 16-byte/value adjoint scratch. No nearly-parallel residual
is snapped to zero. Public inputs/VJPs and per-batch parameter partials are f32.
These implementations are not arbitrary-precision; chunk seams use public f32
cotangents, and tested chunk equivalence is tolerance-bounded, not universally
bitwise identical.

The GPU scans each complex channel over time, projects whole rows, performs
reverse scans, and deterministically reduces parameter partials over batch.
No floating-point atomics or host readback are used in forward/backward.
Arbitrary validated strided/offset inputs are packed on the GPU. Storage,
bindings, uniforms and dispatch limits are checked before execution, including
backward scratch requirements before forward.

Any detected nonfinite input, forward intermediate or returned derivative invalidates the whole
output family. Packed-input guards are joined again after dispatch. A late
parameter-reduction overflow invalidates all four VJPs, including otherwise
finite drive/initial-state gradients. Retained results must not be mutated by
later successful or failed calls. The optimizer rejects invalid gradients at
both zero and nonzero learning rates without a partial update.

## Verification and next seam

The checked-in independent single-thread CPU-f32 PyTorch oracle covers ten
shapes/curvatures, all four VJPs, and sixteen parameter-learning updates.
Separate f64 analytic controls exercise large axis-aligned/diagonal/oblique
states, one-ULP cotangent perturbations and a representable tiny derivative.
The reference fixture and elementwise tolerance are not regenerated/relaxed
after failures. Results and exact provenance are recorded in
[the validation record](../benchmarks/results/2026-10-09-causal-zspace-wave/README.md).

```sh
cargo test --locked -p st-kernel-contracts causal_wave
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release \
  -p st-backend-wgpu causal_wave -- --test-threads=1
cargo build --locked -p st-backend-wgpu --target wasm32-unknown-unknown \
  --release --example causal_zspace_wave_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/causal_zspace_wave_browser.wasm \
  --target web --out-dir target/causal-zspace-wave-web
python3 -I -S -m http.server 8771 --bind 127.0.0.1
```

Open `/crates/st-backend-wgpu/tests/causal_zspace_wave_browser.html` on the local
server in a WebGPU browser. The browser is an actual execution check; compiling
WASM alone is not equivalent. Native link environment flags may need to be
unset for a wasm32 cross-build.

The [Poincare metric primitive](poincare_metric_attention.md) provides genuine
squared-distance pair bias on chart coordinates, with derivatives into both
endpoints and learned per-head gains. The optional
[byte decoder geometry plan](resident_byte_decoder.md#the-geometry-boundary)
now integrates them under one full-model owner. Metric cotangents are summed
across all heads/blocks before this wave's backward, with next-byte CE reaching
decay, phase and projection parameters. Its full-model probe includes a detached
geometry control so a direct Euclidean path cannot disguise a missing pullback.
The standalone wave evidence above remains primitive-only; neither that evidence
nor the tiny full-model correctness probe establishes language-quality gains,
an advantage over ordinary FT, or throughput.
