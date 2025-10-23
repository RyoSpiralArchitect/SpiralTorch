# Macro interfacial template for SpiralTorch

This template packages the "macro-only" sharp-interface playbook into a single
code path so SpiralTorch can design, analyse, and implement surface-tension
models without invoking the microlocal R machine. Each section mirrors the
engineering notes supplied for the MIF (Macro Interfacial Functional) workflow
and points to the Rust artefacts that realise the recipe.

## 0. Scope and notation
- `PhaseId`, `PhasePair`, and `PhaseTriple` encode the regions \(E_i\), the
  interfaces \(\Gamma_{ij}\), and triple lines \(\Sigma_{ijk}\), preserving the
  directed orientation information that fixes the normal \(\nu_{ij}\).【F:crates/st-core/src/theory/macro.rs†L39-L87】
- `SurfaceTerm`, `BendingTerm`, `LineTensionTerm`, `BulkPotential`, and
  `NonlocalInteraction` capture the core integrals \((\text{S1})\)–\((\text{N1})\)
  with optional anisotropy, Helfrich curvature, line tension, and bulk/nonlocal
  loads.【F:crates/st-core/src/theory/macro.rs†L89-L199】

## 1. Macro Interfacial Functional (MIF)
- `MacroInterfacialFunctional` collects all surface, bending, line, bulk, and
  nonlocal contributions plus boundary conditions and constraints, matching the
  boxed functional \(\mathcal E[\{\Gamma\}]\).【F:crates/st-core/src/theory/macro.rs†L206-L308】
- Helper builders (`add_surface_term`, `add_bending_term`, etc.) let you extend
  the functional progressively—start with \((\text{S1})\) and opt into
  anisotropy, bending, or line tension as the design demands.【F:crates/st-core/src/theory/macro.rs†L277-L308】

## 2. Equilibrium and boundary wiring
- Boundary conditions implement Young and Cahn–Hoffman laws with explicit solid
  identifiers, while `LineTensionTerm` records the Herring force balance data at
  \(\Sigma_{ijk}\).【F:crates/st-core/src/theory/macro.rs†L147-L263】
- External drives—pressure jumps, bulk fields, nonlocal forcings, and dynamic
  contact angles—are normalised via the `ExternalForce` enum so they slot into
  the Euler–Lagrange balances and the kinetic forcing term \(F\).【F:crates/st-core/src/theory/macro.rs†L333-L341】

## 3. Gradient-flow kinetics
- `FlowKind` and `MobilityModel` choose the metric (non-conserved, conserved, or
  surface diffusion) while `MacroKinetics::evaluate_velocity` evaluates
  \(v_n = M(\sigma\,\kappa_\gamma - \kappa\,\Delta_\Gamma H + F)\) or its surface
  diffusion counterpart using the supplied curvature bundle.【F:crates/st-core/src/theory/macro.rs†L310-L421】
- `CurvatureContributions` packages \(H\), \(\kappa_\gamma\), bending operators,
  and forcing, keeping anisotropy and Helfrich regularisation optional but
  explicit.【F:crates/st-core/src/theory/macro.rs†L402-L421】

## 4. Dimensionless audit panel
- `PhysicalScales` stores \(\sigma, L, U, \eta, \Delta\rho, g, \ell_b, \ell_\lambda\)
  and related knobs; `DimensionlessGroups::from_scales` produces Capillary,
  Bond, Cahn, Peclet, bending, and line-tension lengths plus the mobility ratio,
  mirroring the non-dimensional checklist in the memo.【F:crates/st-core/src/theory/macro.rs†L423-L517】
- `sharp_interface_ok` automates the \(\mathrm{Cn}\ll1\) sanity check that keeps the
  sharp-interface closure valid.【F:crates/st-core/src/theory/macro.rs†L515-L517】

## 5. Analysis checklist
- `AnalysisChecklist` captures existence, regularity, energy decay, and
  constraint handling outcomes for each scenario—minimal surface, anisotropic
  crystal, membrane, pattern formation, and contact line—together with the key
  cautionary notes from the analytic checklist.【F:crates/st-core/src/theory/macro.rs†L520-L603】

## 6. Numerical templates
- `NumericalRecipe` records which discretisation to run (MBO, minimizing
  movements, or level set) and the concrete operator-splitting steps, keeping
  the "three practical tricks" codified alongside the model.【F:crates/st-core/src/theory/macro.rs†L605-L658】

## 7. Model cards (A–E)
- `MacroModelTemplate::from_card` instantiates the five cards: minimal droplet,
  anisotropic crystal, membrane, pattern formation, and contact line. Each card
  wires the correct functional pieces, kinetics, dimensionless dashboard, and
  numerical recipe in one call.【F:crates/st-core/src/theory/macro.rs†L660-L826】
- `MacroCard` and its config structs expose the tunable knobs (surface tension,
  anisotropy modes, bending modulus, kernel strength, contact angle, etc.) in a
  form that slots straight into SpiralTorch scheduling.【F:crates/st-core/src/theory/macro.rs†L831-L891】

## 8. Calibration cues
- Volume and area constraints map to `PhaseConstraint` and `InterfaceConstraint`
  so static fits (e.g., matching a spherical cap) become data inputs instead of
  ad-hoc post-processing.【F:crates/st-core/src/theory/macro.rs†L206-L244】
- Kernel magnitudes and contact-line forces expose their strength via
  `KernelSpec::strength` and the `ExternalForce::ContactAngle` payloads, making
  it straightforward to plug dispersion relations or size-dependent contact
  angles into the template.【F:crates/st-core/src/theory/macro.rs†L190-L204】【F:crates/st-core/src/theory/macro.rs†L333-L341】

## 9. Quick start: design → analysis → implementation
1. Choose a `MacroCard` and fill the relevant config (Card A–E).
2. Call `MacroModelTemplate::from_card` to obtain the functional, kinetics,
   dimensionless dashboard, analysis checklist, and numerical recipe in one
   struct.【F:crates/st-core/src/theory/macro.rs†L660-L826】
3. Feed the recommended `NumericalRecipe` steps into your solver, inject the
   `AnalysisChecklist` notes into regression tests, and monitor the
   dimensionless groups for calibration drift.【F:crates/st-core/src/theory/macro.rs†L520-L658】

The template keeps the macro system self-contained while still leaving hooks to
reintroduce microlocal calibration later if needed.

## 10. Microlocal and Z-space coupling
- `MacroModelTemplate::couple_with` produces a `MacroZBridge` that accepts an
  `InterfaceZLift`, letting the macro card tap directly into the microlocal
  gauges and Z pulses without duplicating wiring code.【F:crates/st-core/src/theory/macro.rs†L841-L843】
- `MacroZBridge::ingest_signature` converts an `InterfaceSignature` into a
  fused `MacroDrive` carrying the projected `InterfaceZPulse`, curvature bundle,
  and predicted normal velocity so Z-space conductors can steer macro evolution
  using the template’s kinetics.【F:crates/st-core/src/theory/macro.rs†L870-L963】
- `MacroTemplateBank` keeps the cards pluggable: register templates or cards by
  id, then call `couple_all` with an `InterfaceZLift` to produce a bridge bank
  aligned with the microlocal gauges currently active in the conductor.【F:crates/st-core/src/theory/macro.rs†L680-L812】
- `InterfaceZReport::gauge_id`, `InterfaceZReport::signature_for`, and
  `InterfaceZReport::lift` expose the matched gauge identifiers, raw
  signatures, and lift clone so macro banks can reuse the projection without
  re-running the gauges.【F:crates/st-core/src/theory/microlocal.rs†L663-L738】
- `MacroTemplateBank::drive_matched` and `feedback_from_report` run only the
  templates that align with the conductor report and merge their microlocal
  feedback before sending it back into the conductor.【F:crates/st-core/src/theory/macro.rs†L780-L812】
- `MacroDrive::sharp_interface_ok` reuses the dimensionless dashboard to ensure
  the detected interface still respects the sharp-interface regime before the
  signal is injected back into SpiralFlow.【F:crates/st-core/src/theory/macro.rs†L977-L980】
- `MacroDrive::microlocal_feedback` emits a `MicrolocalFeedback` payload that
  tells the microlocal side how to retune gauge thresholds, Z-lift bias gains,
  smoothing, and tempo hints based on the macro kinetics and dimensionless
  regime.【F:crates/st-core/src/theory/macro.rs†L870-L984】
- `MacroZBridge::ingest_signature` now weights curvature estimates by the
  perimeter density and, whenever oriented normals are available, folds the
  card’s anisotropy specification into \(\kappa_\gamma\) before the velocity is
  evaluated—keeping Fourier modes and tabulated spectra consistent with the
  macro functional.【F:crates/st-core/src/theory/macro.rs†L870-L1041】
