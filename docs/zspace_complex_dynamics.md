# Complex Z-space Dynamics

The complex-state Schrodinger API preserves both quadratures between steps and
exposes their real Euclidean vector-Jacobian product (VJP). Rust owns the
equation, validation, and gradients. Python passes passive mappings; WASM passes
bounded JSON. The existing real-input forward/VJP v1 contracts are unchanged.

## One Step

```python
import spiraltorch as st

request = {
    "forward_request": {
        "input": [0.8, -0.3, 0.4],
        "potential": [0.4, -0.5, 0.3],
        "standard_normal": [0.0, 0.0, 0.0],
        "rows": 1,
        "features": 3,
        "config": {"time_step": 0.3, "hopping_rate": 0.6},
    },
    "input_imaginary": [0.2, 0.5, -0.6],
}
receipt = st.zspace_stochastic_schrodinger_complex_step(request)
st.validate_zspace_stochastic_schrodinger_complex(receipt)

# Never discard the imaginary state when composing transitions.
next_request = {
    "forward_request": {
        **receipt["request"]["forward_request"],
        "input": receipt["step"]["output_real"],
    },
    "input_imaginary": receipt["step"]["output_imaginary"],
}
next_receipt = st.zspace_stochastic_schrodinger_complex_step(next_request)
```

The nested `forward_request` has the same shape as the real-input API, while
`input_imaginary` is mandatory. Both state arrays have `rows * features` elements;
`potential` has `features` elements and is shared across rows. Gaussian witnesses
are explicit: zero witnesses do not secretly sample noise.

## Learning Through A Trajectory

Pass `cotangent={"real": [...], "imaginary": [...]}` to request gradients for
a real-valued loss. The receipt's `gradient` contains `grad_input_real`,
`grad_input_imaginary`, and `grad_potential`. Config and Gaussian witnesses are
held fixed; no derivative with respect to time, loss, hopping, or noise is claimed.

To backpropagate through multiple steps, process their canonical requests in
reverse order, passing both input cotangents to the preceding step. If potential
is shared over time, sum its gradient contributions from every step, in addition
to the row reduction already performed by Rust.

Run the deterministic fitting example against an installed wheel:

```bash
python examples/zspace_complex_trajectory.py
```

It fits a shared potential through three complex transitions using Rust VJPs.
This is a numerical learning check, not evidence of improved language quality or
identifiability of a physical system. The example preserves final-state output
and does not use decoding tricks to lower its squared-error loss.

## Rust And WASM

Rust users can call `apply_stochastic_schrodinger_complex_step` and
`backward_stochastic_schrodinger_complex_step` with borrowed
`StochasticSchrodingerComplexInput` from `st_core::dynamics::stochastic_schrodinger`.
For the same owned/replayable contract as Python, use
`run_zspace_stochastic_schrodinger_complex_step` from
`st_core::runtime::zspace_stochastic_schrodinger`.

WASM exports `zspaceStochasticSchrodingerComplexStepJson` and
`validateZspaceStochasticSchrodingerComplexJson`. Stringify the same request,
then parse the returned receipt. Runtime catalog v4 includes this surface.

The kernel evaluates `d * D(phi/2) * H_pair * D(phi/2) * psi` with
`phi = potential * dt + standard_normal * noise_scale * sqrt(dt)` and
`d = exp(-loss_rate * dt / 2)`. `H_pair` is a unitary hopping block on disjoint
adjacent feature pairs; an odd last feature receives only the diagonal phase.
It is the stated discrete split operator, not an exact solver for an arbitrary
Hamiltonian. State and gradients are f32, with f64 complex arithmetic and gradient
accumulation. Transcendentals use pinned Rust `libm` 0.2.16 rather than host libm:
native `expf` differed from WASM by one ULP in the fixed-seed cross-client test.
The receipt binds this arithmetic profile. Norm checks fail closed on invalid
numerical transitions; replay comparisons are not weakened to hide platform drift.

The executable cross-client check includes fixed edge cases and 64 pseudorandom
cases (seed 173), comparing complete receipts and replaying them both ways:

```bash
wasm-pack build bindings/st-wasm --target nodejs --out-dir /tmp/st-complex-wasm
python tools/verify_zspace_complex_clients.py --wasm-module /tmp/st-complex-wasm/spiraltorch_wasm.js
```

The protocol admits up to 262,144 complex values and 65,536 features. Its
receipt-plus-gradient ingress permits 96 MiB, 3,000,512 JSON nodes, and depth 10.
It runs the scalar Rust kernel natively or in WASM; it does not claim dispatch
through the separate real-quadrature WGPU kernels.
