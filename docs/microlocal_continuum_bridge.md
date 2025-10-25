# Microlocal–Continuum Bridge for SpiralTorch

SpiralTorch already measures boundary activity with microlocal gauges and turns the samples into Z-space control pulses. This note shows how to treat that pipeline as the micro-to-macro bridge suggested by classical interfacial models: the "R machine" corresponds to the perimeter/curvature estimator, the Z-lift aggregates those microscopic readings into macroscopic energy budgets, and the conductor enforces evolution policies that mirror continuum gradient flows.

## 1. Microlocal sensing: the R machine

- `InterfaceGauge` scans a binary phase field (or probability mask) with finite differences, setting the **R machine** indicator when variation exceeds a configurable threshold and recording raw/perimeter densities plus mean curvature surrogates.【F:crates/st-core/src/theory/microlocal.rs†L48-L176】
- The resulting `InterfaceSignature` stores those tensors alongside dimensional constants such as kappa_d (unit-sphere area) and the physical probing radius, and it can optionally recover oriented normals when the label field c-prime is supplied—matching the sign conventions demanded by Young–Laplace balances.【F:crates/st-core/src/theory/microlocal.rs†L27-L214】
- Because the gauge exposes multi-radius evaluation, you can emulate scale-separated surface energies (e.g., isotropic vs. bending) by running several radii and stacking the signatures, mirroring how Modica–Mortola limits recover the sharp-interface perimeter from diffuse-phase data.【F:crates/st-core/src/theory/microlocal.rs†L82-L214】

## 2. Projecting micro statistics into macro energy budgets

- `InterfaceZLift` converts a signature into an `InterfaceZPulse`: it sums interface support, folds orientation bias into Above/Here/Beneath band energies, and keeps track of Z-bias drift so macroscopic drives can distinguish inflating vs. deflating curvature.【F:crates/st-core/src/theory/microlocal.rs†L262-L351】
- The produced pulse slots into the shared Z-space vocabulary—`ZPulse` carries support mass, band energies, drift, and optional scale tags while `ZSupport` enforces non-negative Above/Here/Beneath components—so downstream conductors interpret microlocal readings exactly like any other energy source.【F:crates/st-core/src/theory/zpulse.rs†L122-L258】
- Because `InterfaceZPulse::aggregate` and `InterfaceZPulse::lerp` blend multiple pulses with weighted averages, you can mirror macroscopic energy minimisation (e.g., convex combinations representing surface tension plus bending penalties) before exposing the fused signal.【F:crates/st-core/src/theory/microlocal.rs†L378-L456】

## 3. Dynamic fusion and policy-driven evolution

- `InterfaceZConductor` orchestrates a bank of gauges, applies per-source quality weights, optional band gating, and budget clamps, then fuses the scaled pulses into a single macroscopic control action. The fused signal is smoothed, converted into `ZPulse` records, and pushed through the shared `ZConductor` for late fusion with other modalities.【F:crates/st-core/src/theory/microlocal.rs†L753-L907】
- The conductor retains the last fused pulse and exposes Softlogic feedback packets, letting higher-level flows (e.g., Allen–Cahn-like non-conserved relaxations vs. Cahn–Hilliard-style conserved drives) choose how aggressively to react to curvature or orientation drifts by tweaking policy multipliers, smoothing, or tempo hints.【F:crates/st-core/src/theory/microlocal.rs†L807-L935】
- Band policies can emulate anisotropic surface tensions by requiring minimum quality in specific bands before a pulse contributes, while budget policies cap the Z-bias magnitude—analogous to enforcing capillary or Bond number bounds in continuum scaling.【F:crates/st-core/src/theory/microlocal.rs†L652-L919】

## 4. Calibration knobs for micro↔macro alignment

- **Radius & threshold**: `InterfaceGauge::with_threshold` and `analyze_multiradius` let you tune the detection scale so that the inferred perimeter |Dchi| matches the macroscopic surface tension sigma extracted from data.【F:crates/st-core/src/theory/microlocal.rs†L65-L214】
- **Bias gain**: `InterfaceZLift::with_bias_gain` scales the orientation-derived drift, acting like a mobility M that governs how interface normals steer macroscopic motion.【F:crates/st-core/src/theory/microlocal.rs†L282-L351】
- **Quality/band overrides**: composite policies enable per-source overrides, so you can privilege, suppress, or hysteretically gate contributions depending on whether you are emulating non-conserved or conserved dynamics.【F:crates/st-core/src/theory/microlocal.rs†L545-L919】
- **Smoothing & tempo**: conductor smoothing and tempo hints control how fast the fused signal evolves, letting you approximate Mullins–Sekerka slow diffusion vs. mean-curvature fast relaxation by setting the interpolation factor and external tempo fed into emitted pulses.【F:crates/st-core/src/theory/microlocal.rs†L788-L907】
- **Elliptic warp steering**: macro feedback can retune the positive-curvature warp by updating its radius, sheet count, and spin harmonics so the microlocal lift keeps pace with the continuum curvature bundle.【F:crates/st-core/src/theory/microlocal.rs†L392-L452】【F:crates/st-core/src/theory/microlocal.rs†L1351-L1400】【F:crates/st-core/src/theory/macro.rs†L1136-L1180】
- **Lie-aligned elliptic telemetry**: the warp now emits SO(3) frames, rotor fields, topological sectors, and differentiable feature vectors with a Jacobian so the bridge can track homology changes and feed gradients into learning loops; Python bindings surface the same telemetry plus a Torch autograd helper for end-to-end optimisation.【F:crates/st-core/src/theory/microlocal/elliptic.rs†L1-L420】【F:bindings/st-py/src/elliptic.rs†L1-L170】【F:bindings/st-py/spiraltorch/elliptic.py†L1-L156】

## 5. Feedback and observability

- Every fused pulse can become a `SoftlogicZFeedback` packet containing total energy, band breakdown, drift, and Z-bias, ready for SpiralFlow controllers or telemetry dashboards to monitor macroscopic stability metrics (perimeter mass, curvature budgets, etc.).【F:crates/st-core/src/theory/microlocal.rs†L477-L908】
- Because `InterfaceZConductor` registers its own `MicrolocalEmitter`, Z-space telemetry preserves attribution back to the microlocal source, making it straightforward to audit how microscopic detections influenced macro evolution—useful when calibrating against Boundary-F1 or coarsening laws.【F:crates/st-core/src/theory/microlocal.rs†L721-L907】

## 6. Minimal operating recipe

1. Instantiate one or more `InterfaceGauge`s with the grid spacing that matches your simulation voxel size.
2. Choose bias gains and policies that reflect the macroscopic regime (e.g., isotropic minimal surface vs. anisotropic crystal growth).
3. Feed each timestep's mask (and optional signed phase label) into `InterfaceZConductor::step` to obtain the fused pulse and Softlogic feedback.
4. Use the fused `ZPulse` stream to drive your macro evolution loop, modulating learning rates, actuation, or constraint forces according to the detected curvature pressure.

This workflow lets the microlocal "R machine" act as a data-driven estimator for the continuum models summarized earlier, while keeping everything within SpiralTorch's native Z-space orchestration.

## 7. Handshake with the macro template

- `MacroZBridge` couples any macro card to an `InterfaceZLift`, reusing the exact Z pulses produced above instead of reimplementing the projection logic.【F:crates/st-core/src/theory/macro.rs†L841-L963】
- `MacroDrive` emits the curvature bundle and predicted velocity derived from the macro kinetics, so the same fused pulse can be inspected in Z-space while closing the sharp-interface dynamics.【F:crates/st-core/src/theory/macro.rs†L870-L984】
- The resulting `MicrolocalFeedback` guides `InterfaceZConductor::apply_feedback`, rescaling gauge thresholds, bias gains, smoothing, and default tempo/uncertainty hints so micro sensing reacts to macro regime shifts without manual retuning.【F:crates/st-core/src/theory/macro.rs†L870-L984】【F:crates/st-core/src/theory/microlocal.rs†L565-L1016】
- Perimeter-weighted curvature averages and orientation-aware anisotropy
  factors keep the bridge’s \(\kappa_\gamma\) estimate aligned with the macro
  card, so Fourier spectra or tabulated surface tensions propagate straight into
  the Z-feedback heuristics.【F:crates/st-core/src/theory/macro.rs†L870-L1041】
