# General Relativity Couplings inside Z-space

SpiralTorch treats the Z-space runtime as a cooperative geometric fabric.
Introducing general relativity extends that fabric with Lorentzian curvature
signals that bias the scheduler, tensors, and telemetry streams.  The
`st_core::theory::general_relativity` module implements the bridge.

## 1. Relativistic patches

A **relativistic patch** captures a local metric \(g_{\mu\nu}\) and its inverse
\(g^{\mu\nu}\).  Callers construct the patch with `RelativisticPatch::new` or
use the convenience `RelativisticPatch::minkowski()` when starting from flat
spacetime.  The constructor symmetrises the metric and verifies it is
invertible, ensuring downstream curvature computations remain numerically
stable.

```rust
use nalgebra::Matrix4;
use st_core::theory::general_relativity::RelativisticPatch;

let mut metric = Matrix4::zeros();
metric[(0, 0)] = 1.0;
metric[(1, 1)] = -1.0;
metric[(2, 2)] = -1.0;
metric[(3, 3)] = -1.2; // anisotropic time warp in Z-space
let patch = RelativisticPatch::new(metric);
```

## 2. Christoffel symbols and curvature

Provide first-order derivatives of the metric \(\partial_\sigma g_{\mu\nu}\) to
`RelativisticPatch::christoffel` to produce the connection coefficients
\(\Gamma^\mu_{\nu\rho}\).  Partial derivatives of the connection itself feed
`RelativisticPatch::ricci`, yielding the Ricci tensor and the scalar curvature.
The Einstein tensor follows automatically via
`RelativisticPatch::einstein_tensor`.

The helpers accept dense `nalgebra::Matrix4<f64>` blocks so they can run
alongside the existing microlocal solvers or be filled from telemetry-driven
finite differences.

## 3. Projecting curvature into Z-space pulses

`RelativisticPatch::to_zpulse` converts the Einstein tensor into a
`ZPulse` tagged with `ZSource::GW`.  Temporal energy density feeds the "Here"
band while mixed temporal-spatial components drive the Above/Beneath contrast.
The resulting pulse slots into the cooperative scheduler and biases the
roundtable using the scalar curvature as a Z-bias term.

```rust
use st_core::theory::general_relativity::{RelativisticPatch, Rank2Tensor};

let patch = RelativisticPatch::minkowski();
let mut ricci: Rank2Tensor = Rank2Tensor::zeros();
ricci[(0, 0)] = 2.5; // synthetic stress-energy deposit
let einstein = patch.einstein_tensor(&ricci);
let pulse = patch.to_zpulse(&einstein, 1024, 0.75);
assert_eq!(pulse.source, st_core::theory::zpulse::ZSource::GW);
```

## 4. Coupling with higher-level planners

Relativistic pulses can be fed into the existing `ZEmitter` infrastructure.
Downstream planners observe the scalar curvature via the `z_bias` field and can
adjust barycentre objectives, attention schedules, or hypergradient pacing
accordingly.  Because the Einstein tensor enters as a conventional `ZPulse`,
no additional plumbing is required for graph planners, PSI synchronisers, or
Canvas feedback loops.

## 5. Numerical stability hints

- Always clamp or regularise noisy telemetry before forming the metric.
- Supply consistent derivative tensors: \(\partial_\sigma g_{\mu\nu} = \partial_\sigma g_{\nu\mu}\).
- When approximating derivatives numerically, prefer symmetric differences so
  the resulting Einstein tensor remains well-conditioned.
- The projection uses absolute values for Above/Beneath to respect orientation
  changes; fine-tune the mapping if your workload favours signed fluxes.

With these additions, SpiralTorch treats relativistic curvature as a first-class
signal inside the Z-space runtime, letting gravitational dynamics steer the same
pipelines that already power microlocal inference and Maxwell-coded feedback.
