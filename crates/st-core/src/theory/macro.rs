// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Macro-scale interfacial templates used to keep the sharp-interface
//! description "closeable" without invoking the microlocal gauges.
//!
//! The design mirrors the engineering memo captured in
//! `docs/macro_interfacial_template.md`: start with the Macro Interfacial
//! Functional (MIF), attach Euler–Lagrange and boundary conditions, wire the
//! gradient-flow kinetics, and then expose dimensionless audit knobs plus a
//! ready-to-run numerical recipe.  The goal is to provide a single
//! programmable artefact that turns the paper template into code so the rest
//! of SpiralTorch can instantiate droplets, crystalline fronts, membranes, or
//! pattern-forming interfaces without re-deriving the bookkeeping each time.

use crate::theory::microlocal::{
    InterfaceSignature, InterfaceZLift, InterfaceZPulse, InterfaceZReport, MicrolocalFeedback,
};
use crate::theory::microlocal_bank::GaugeBank;
use ndarray::{indices, ArrayD, Dimension, IxDyn};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Identifier for a macroscopic phase or boundary participant.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhaseId(String);

impl PhaseId {
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PhaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Phase pair representing an interface Γ_{ij}.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhasePair {
    pub i: PhaseId,
    pub j: PhaseId,
}

impl PhasePair {
    pub fn new<I: Into<String>, J: Into<String>>(i: I, j: J) -> Self {
        Self {
            i: PhaseId::new(i),
            j: PhaseId::new(j),
        }
    }
}

/// Phase triple representing a triple junction Σ_{ijk}.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhaseTriple {
    pub phases: [PhaseId; 3],
}

impl PhaseTriple {
    pub fn new<I: Into<String>, J: Into<String>, K: Into<String>>(i: I, j: J, k: K) -> Self {
        Self {
            phases: [PhaseId::new(i), PhaseId::new(j), PhaseId::new(k)],
        }
    }
}

/// Surface tension contribution (S1) with optional anisotropy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceTerm {
    pub pair: PhasePair,
    pub sigma: f64,
    pub anisotropy: AnisotropySpec,
}

impl SurfaceTerm {
    pub fn new(pair: PhasePair, sigma: f64, anisotropy: AnisotropySpec) -> Self {
        Self {
            pair,
            sigma: sigma.max(0.0),
            anisotropy,
        }
    }
}

/// Anisotropy descriptor for γ(ν).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AnisotropySpec {
    /// γ(ν) ≡ 1 (isotropic surface tension).
    Isotropic,
    /// γ(ν) expanded in planar Fourier modes (useful for 2D crystals).
    Fourier { modes: Vec<(usize, f64)> },
    /// Tabulated weights sampled on a quadrature grid of the unit sphere.
    Tabulated {
        weights: Vec<f64>,
        description: String,
    },
    /// Free-form description when the concrete implementation lives elsewhere.
    Custom { note: String },
}

impl Default for AnisotropySpec {
    fn default() -> Self {
        Self::Isotropic
    }
}

/// Bending/regularisation contribution (S2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BendingTerm {
    pub pair: PhasePair,
    pub kappa: f64,
    pub spontaneous_curvature: Option<f64>,
}

impl BendingTerm {
    pub fn new(pair: PhasePair, kappa: f64, spontaneous_curvature: Option<f64>) -> Self {
        Self {
            pair,
            kappa: kappa.max(0.0),
            spontaneous_curvature,
        }
    }
}

/// Line tension (S3) along triple lines.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineTensionTerm {
    pub triple: PhaseTriple,
    pub lambda: f64,
}

impl LineTensionTerm {
    pub fn new(triple: PhaseTriple, lambda: f64) -> Self {
        Self {
            triple,
            lambda: lambda.max(0.0),
        }
    }
}

/// Bulk potential (B1) acting on a phase volume.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BulkPotential {
    pub phase: PhaseId,
    pub potential: PotentialSpec,
}

/// Supported bulk potential templates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PotentialSpec {
    /// Constant offset representing a pressure reference.
    Constant(f64),
    /// Linear potential U(x) = g·x (gravity, acceleration fields).
    Linear { gradient: [f64; 3] },
    /// User-supplied description.
    Custom { note: String },
}

/// Non-local kernel (N1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NonlocalInteraction {
    pub phase: PhaseId,
    pub kernel: KernelSpec,
}

/// Supported non-local kernels used for pattern formation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum KernelSpec {
    Gaussian { strength: f64, length: f64 },
    PowerLaw { strength: f64, exponent: f64 },
    Custom { strength: f64, note: String },
}

impl KernelSpec {
    pub fn strength(&self) -> f64 {
        match *self {
            KernelSpec::Gaussian { strength, .. }
            | KernelSpec::PowerLaw { strength, .. }
            | KernelSpec::Custom { strength, .. } => strength,
        }
    }
}

/// Macro-level constraints (volume, area, topology).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Constraints {
    pub volumes: Vec<PhaseConstraint>,
    pub areas: Vec<InterfaceConstraint>,
    pub topology: Option<String>,
}

/// Volume constraint |E_i| = V_i.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseConstraint {
    pub phase: PhaseId,
    pub target: f64,
}

impl PhaseConstraint {
    pub fn new(phase: PhaseId, target: f64) -> Self {
        Self {
            phase,
            target: target.max(0.0),
        }
    }
}

/// Area constraint for a specific interface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterfaceConstraint {
    pub pair: PhasePair,
    pub target: f64,
}

impl InterfaceConstraint {
    pub fn new(pair: PhasePair, target: f64) -> Self {
        Self {
            pair,
            target: target.max(0.0),
        }
    }
}

/// Boundary condition applied on a solid wall.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundaryCondition {
    pub phase: PhaseId,
    pub kind: BoundaryKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoundaryKind {
    Young {
        solid: PhaseId,
        equilibrium_angle: f64,
    },
    CahnHoffman {
        solid: PhaseId,
        anisotropy: AnisotropySpec,
    },
}

/// Macro Interfacial Functional (MIF) container.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MacroInterfacialFunctional {
    pub surface_terms: Vec<SurfaceTerm>,
    pub bending_terms: Vec<BendingTerm>,
    pub line_tensions: Vec<LineTensionTerm>,
    pub bulk_potentials: Vec<BulkPotential>,
    pub nonlocal_terms: Vec<NonlocalInteraction>,
    pub boundary_conditions: Vec<BoundaryCondition>,
    pub constraints: Constraints,
}

impl MacroInterfacialFunctional {
    pub fn minimal(pair: PhasePair, sigma: f64) -> Self {
        Self {
            surface_terms: vec![SurfaceTerm::new(pair, sigma, AnisotropySpec::Isotropic)],
            ..Default::default()
        }
    }

    pub fn add_surface_term(&mut self, term: SurfaceTerm) {
        self.surface_terms.push(term);
    }

    pub fn add_bending_term(&mut self, term: BendingTerm) {
        self.bending_terms.push(term);
    }

    pub fn add_line_tension(&mut self, term: LineTensionTerm) {
        self.line_tensions.push(term);
    }

    pub fn add_bulk_potential(&mut self, term: BulkPotential) {
        self.bulk_potentials.push(term);
    }

    pub fn add_nonlocal_term(&mut self, term: NonlocalInteraction) {
        self.nonlocal_terms.push(term);
    }

    pub fn add_boundary_condition(&mut self, bc: BoundaryCondition) {
        self.boundary_conditions.push(bc);
    }
}

/// Flow family used to interpret the gradient flow.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FlowKind {
    NonConserved,
    Conserved,
    SurfaceDiffusion,
}

/// Mobility descriptor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MobilityModel {
    Scalar(f64),
    Surface(f64),
}

impl MobilityModel {
    pub fn value(&self) -> f64 {
        match *self {
            MobilityModel::Scalar(m) | MobilityModel::Surface(m) => m.max(0.0),
        }
    }
}

/// External contributions bundled into the driving force F.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExternalForce {
    PressureJump(f64),
    BulkPotential(f64),
    Nonlocal(f64),
    ContactAngle { equilibrium: f64, friction: f64 },
    Custom { note: String, magnitude: f64 },
}

/// Macro kinetics v_n = M (σ κ_γ − κ ΔΓ H + F).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MacroKinetics {
    pub flow: FlowKind,
    pub mobility: MobilityModel,
    pub external_forces: Vec<ExternalForce>,
}

impl MacroKinetics {
    pub fn non_conserved(mobility: f64) -> Self {
        Self {
            flow: FlowKind::NonConserved,
            mobility: MobilityModel::Scalar(mobility),
            external_forces: Vec::new(),
        }
    }

    pub fn conserved(mobility: f64) -> Self {
        Self {
            flow: FlowKind::Conserved,
            mobility: MobilityModel::Scalar(mobility),
            external_forces: Vec::new(),
        }
    }

    pub fn surface_diffusion(surface_mobility: f64) -> Self {
        Self {
            flow: FlowKind::SurfaceDiffusion,
            mobility: MobilityModel::Surface(surface_mobility),
            external_forces: Vec::new(),
        }
    }

    pub fn with_force(mut self, force: ExternalForce) -> Self {
        self.external_forces.push(force);
        self
    }

    /// Aggregates the external forcing term using the supplied Z-space pulse if available.
    pub fn forcing_from_pulse(&self, pulse: Option<&InterfaceZPulse>) -> f64 {
        self.external_forces
            .iter()
            .map(|force| match *force {
                ExternalForce::PressureJump(delta_p) => delta_p as f64,
                ExternalForce::BulkPotential(value)
                | ExternalForce::Nonlocal(value)
                | ExternalForce::Custom {
                    magnitude: value, ..
                } => value as f64,
                ExternalForce::ContactAngle {
                    equilibrium,
                    friction,
                } => {
                    if let Some(pulse) = pulse {
                        let deviation = pulse.z_bias as f64 - equilibrium as f64;
                        -(friction as f64) * deviation
                    } else {
                        0.0
                    }
                }
            })
            .sum()
    }

    /// Evaluates the normal velocity for the supplied contributions.
    pub fn evaluate_velocity(&self, contributions: &CurvatureContributions) -> f64 {
        let mobility = self.mobility.value();
        match self.flow {
            FlowKind::NonConserved | FlowKind::Conserved => {
                let curvature = contributions
                    .anisotropic_curvature
                    .unwrap_or(contributions.mean_curvature);
                let bending = contributions.bending_operator.unwrap_or(0.0);
                mobility * (curvature - bending + contributions.forcing)
            }
            FlowKind::SurfaceDiffusion => {
                let surface_term = contributions
                    .bending_operator
                    .unwrap_or(contributions.mean_curvature);
                -mobility * surface_term
            }
        }
    }
}

/// Curvature and forcing bundle required to evaluate v_n.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CurvatureContributions {
    pub mean_curvature: f64,
    pub anisotropic_curvature: Option<f64>,
    pub bending_operator: Option<f64>,
    pub forcing: f64,
}

impl CurvatureContributions {
    pub fn with_anisotropy(mut self, value: f64) -> Self {
        self.anisotropic_curvature = Some(value);
        self
    }

    pub fn with_bending(mut self, value: f64) -> Self {
        self.bending_operator = Some(value);
        self
    }
}

/// Physical scales used to derive dimensionless groups.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalScales {
    pub sigma: f64,
    pub reference_length: Option<f64>,
    pub velocity_scale: Option<f64>,
    pub viscosity: Option<f64>,
    pub density_difference: Option<f64>,
    pub gravity: Option<f64>,
    pub interface_thickness: Option<f64>,
    pub diffusion: Option<f64>,
    pub bending_modulus: Option<f64>,
    pub line_tension: Option<f64>,
    pub mobility_ratio: Option<f64>,
}

impl PhysicalScales {
    pub fn with_sigma(sigma: f64) -> Self {
        Self {
            sigma,
            reference_length: None,
            velocity_scale: None,
            viscosity: None,
            density_difference: None,
            gravity: None,
            interface_thickness: None,
            diffusion: None,
            bending_modulus: None,
            line_tension: None,
            mobility_ratio: None,
        }
    }
}

/// Dimensionless groups extracted from the physical scales.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DimensionlessGroups {
    pub capillary: Option<f64>,
    pub bond: Option<f64>,
    pub cahn: Option<f64>,
    pub peclet: Option<f64>,
    pub bending_length: Option<f64>,
    pub line_tension_length: Option<f64>,
    pub mobility_ratio: Option<f64>,
}

impl DimensionlessGroups {
    pub fn from_scales(scales: &PhysicalScales) -> Self {
        let sigma = scales.sigma.max(f64::EPSILON);
        let capillary = match (scales.viscosity, scales.velocity_scale) {
            (Some(eta), Some(u)) => Some((eta * u) / sigma),
            _ => None,
        };
        let bond = match (
            scales.density_difference,
            scales.gravity,
            scales.reference_length,
        ) {
            (Some(delta_rho), Some(g), Some(l)) => Some(delta_rho * g * l * l / sigma),
            _ => None,
        };
        let cahn = match (scales.interface_thickness, scales.reference_length) {
            (Some(eps), Some(l)) if l > 0.0 => Some((eps / l).abs()),
            _ => None,
        };
        let peclet = match (
            scales.velocity_scale,
            scales.reference_length,
            scales.diffusion,
        ) {
            (Some(u), Some(l), Some(d)) if d > 0.0 => Some(u * l / d),
            _ => None,
        };
        let bending_length = match (scales.bending_modulus, scales.sigma > 0.0) {
            (Some(kappa), true) if kappa > 0.0 => Some((kappa / sigma).sqrt()),
            _ => None,
        };
        let line_tension_length = match (scales.line_tension, scales.sigma > 0.0) {
            (Some(lambda), true) if lambda > 0.0 => Some(lambda / sigma),
            _ => None,
        };
        Self {
            capillary,
            bond,
            cahn,
            peclet,
            bending_length,
            line_tension_length,
            mobility_ratio: scales.mobility_ratio,
        }
    }

    pub fn sharp_interface_ok(&self) -> bool {
        self.cahn.map(|cn| cn < 0.1).unwrap_or(true)
    }
}

/// Analysis checklist derived from the macro template.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalysisChecklist {
    pub existence: bool,
    pub regularity: bool,
    pub energy_decay: bool,
    pub constraint_handling: bool,
    pub notes: Vec<String>,
}

impl AnalysisChecklist {
    pub fn minimal_surface() -> Self {
        Self {
            existence: true,
            regularity: false,
            energy_decay: true,
            constraint_handling: true,
            notes: vec![
                "Lower semicontinuity of perimeter with volume constraint grants minimisers."
                    .into(),
                "Mean curvature flow preserves monotone energy decay under time-splitting.".into(),
            ],
        }
    }

    pub fn anisotropic(strongly_convex: bool, bending_regulariser: bool) -> Self {
        let mut notes = vec!["Cahn–Hoffman vector enforces force balance on facets.".into()];
        if strongly_convex {
            notes.push("Strongly convex γ smooths facets → classical solutions.".into());
        } else {
            notes.push("Non-convex γ requires facet tracking or small κ regulariser.".into());
        }
        if bending_regulariser {
            notes.push("Small κ suppresses facet pinning and helps numerical stability.".into());
        }
        Self {
            existence: true,
            regularity: strongly_convex || bending_regulariser,
            energy_decay: true,
            constraint_handling: true,
            notes,
        }
    }

    pub fn membrane() -> Self {
        Self {
            existence: true,
            regularity: true,
            energy_decay: true,
            constraint_handling: true,
            notes: vec![
                "Willmore–Helfrich flow handled via operator splitting.".into(),
                "Surface Laplacian of mean curvature requires C1 meshes or LS regularisation."
                    .into(),
            ],
        }
    }

    pub fn pattern() -> Self {
        Self {
            existence: true,
            regularity: false,
            energy_decay: true,
            constraint_handling: false,
            notes: vec![
                "Non-local kernels need FFT-friendly quadrature or FMM.".into(),
                "Convex-splitting ensures monotone decay for σ|Γ| + K * χ.".into(),
            ],
        }
    }

    pub fn contact_line() -> Self {
        Self {
            existence: true,
            regularity: false,
            energy_decay: true,
            constraint_handling: true,
            notes: vec![
                "Dynamic contact angle law needs friction ζ_ℓ calibration.".into(),
                "Line tension enters Herring balance via λ κ_g.".into(),
            ],
        }
    }
}

/// Numerical recipe recommended by the template.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericalRecipe {
    pub algorithm: Algorithm,
    pub steps: Vec<String>,
}

impl NumericalRecipe {
    pub fn mbo(enforce_volume: bool) -> Self {
        let mut steps = vec![
            r"Diffuse mask with Gaussian kernel G_{\sqrt{2Δt}}.".into(),
            "Threshold back to a sharp interface.".into(),
        ];
        if enforce_volume {
            steps.push("Adjust threshold to keep |E_i| constant.".into());
        }
        Self {
            algorithm: Algorithm::MerrimanBenceOsher,
            steps,
        }
    }

    pub fn minimizing_movements() -> Self {
        Self {
            algorithm: Algorithm::MinimizingMovements,
            steps: vec![
                "Solve argmin Γ { E[Γ] + (2Δt)^{-1} d(Γ, Γ^k)^2 }.".into(),
                "Update constraints via Lagrange multipliers.".into(),
            ],
        }
    }

    pub fn level_set(split_biharmonic: bool) -> Self {
        let mut steps = vec![
            "Evolve φ with curvature-dependent velocity.".into(),
            "Reinitialize signed distance to control stiffness.".into(),
        ];
        if split_biharmonic {
            steps.push("Operator-split bending as two diffusion passes.".into());
        }
        Self {
            algorithm: Algorithm::LevelSet,
            steps,
        }
    }
}

/// Supported discretisation algorithms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Algorithm {
    MerrimanBenceOsher,
    MinimizingMovements,
    LevelSet,
}

/// Complete macro template tying design-analysis-implementation together.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MacroModelTemplate {
    pub functional: MacroInterfacialFunctional,
    pub kinetics: MacroKinetics,
    pub nondimensional: DimensionlessGroups,
    pub analysis: AnalysisChecklist,
    pub numerics: NumericalRecipe,
}

/// Registry holding named [`MacroModelTemplate`] instances so macro designs can
/// be swapped in and out alongside microlocal gauge banks.
#[derive(Clone, Debug, Default)]
pub struct MacroTemplateBank {
    inner: GaugeBank<MacroModelTemplate>,
}

impl MacroTemplateBank {
    /// Creates an empty template registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a template with the provided identifier. Returns `false` if an
    /// entry with the same identifier already exists.
    pub fn register(&mut self, id: impl Into<String>, template: MacroModelTemplate) -> bool {
        self.inner.register(id, template)
    }

    /// Registers a [`MacroCard`] by converting it into a template first.
    pub fn register_card(&mut self, id: impl Into<String>, card: MacroCard) -> bool {
        self.register(id, MacroModelTemplate::from_card(card))
    }

    /// Builder-style registration that returns `self` for chaining.
    pub fn with_template(mut self, id: impl Into<String>, template: MacroModelTemplate) -> Self {
        let _ = self.register(id, template);
        self
    }

    /// Builder-style helper that accepts a [`MacroCard`].
    pub fn with_card(mut self, id: impl Into<String>, card: MacroCard) -> Self {
        let _ = self.register_card(id, card);
        self
    }

    /// Returns an immutable reference to the named template.
    pub fn get(&self, id: &str) -> Option<&MacroModelTemplate> {
        self.inner.get(id)
    }

    /// Returns a mutable reference to the named template.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut MacroModelTemplate> {
        self.inner.get_mut(id)
    }

    /// Removes a template from the registry and returns it if present.
    pub fn remove(&mut self, id: &str) -> Option<MacroModelTemplate> {
        self.inner.remove(id)
    }

    /// Iterates over registered identifiers and templates in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MacroModelTemplate)> {
        self.inner.iter()
    }

    /// Iterates over registered identifiers and mutable templates.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut MacroModelTemplate)> {
        self.inner.iter_mut()
    }

    /// Returns the registered identifiers.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.inner.ids()
    }

    /// Number of registered templates.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when no templates are registered.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clones the registered templates into a vector preserving insertion order.
    pub fn to_vec(&self) -> Vec<MacroModelTemplate> {
        self.inner.to_vec()
    }

    /// Consumes the bank and returns the templates in insertion order.
    pub fn into_vec(self) -> Vec<MacroModelTemplate> {
        self.inner.into_vec()
    }

    /// Couples the named template with the supplied microlocal lift.
    pub fn couple(&self, id: &str, lift: InterfaceZLift) -> Option<MacroZBridge> {
        self.get(id)
            .cloned()
            .map(|template| template.couple_with(lift))
    }

    /// Couples every registered template with the supplied microlocal lift and
    /// returns the resulting bridge registry.
    pub fn couple_all(&self, lift: &InterfaceZLift) -> GaugeBank<MacroZBridge> {
        let mut bank = GaugeBank::new();
        for (id, template) in self.inner.entries() {
            let bridge = template.clone().couple_with(lift.clone());
            let _ = bank.register(id.as_ref(), bridge);
        }
        bank
    }

    /// Runs all templates whose identifiers match gauges reported by the conductor
    /// and returns the resulting drives keyed by template id.
    pub fn drive_matched(&self, report: &InterfaceZReport) -> GaugeBank<MacroDrive> {
        let mut drives = GaugeBank::new();
        let lift = report.lift();
        for (id, template) in self.inner.iter() {
            if let Some(signature) = report.signature_for(id) {
                let bridge = template.clone().couple_with(lift.clone());
                let drive = bridge.ingest_signature(signature);
                let _ = drives.register(id, drive);
            }
        }
        drives
    }

    /// Aggregates microlocal feedback emitted by the matched drives. Returns
    /// `None` when no templates were matched.
    pub fn feedback_from_report(&self, report: &InterfaceZReport) -> Option<MicrolocalFeedback> {
        let drives = self.drive_matched(report);
        let mut combined: Option<MicrolocalFeedback> = None;
        for (_, drive) in drives.iter() {
            let feedback = drive.microlocal_feedback().clone();
            combined = Some(match combined {
                Some(existing) => existing.merge(&feedback),
                None => feedback,
            });
        }
        combined
    }
}

impl IntoIterator for MacroTemplateBank {
    type Item = MacroModelTemplate;
    type IntoIter = <GaugeBank<MacroModelTemplate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl MacroModelTemplate {
    pub fn from_card(card: MacroCard) -> Self {
        match card {
            MacroCard::Minimal(config) => {
                let mut functional =
                    MacroInterfacialFunctional::minimal(config.phase_pair.clone(), config.sigma);
                if let Some(volume) = config.volume {
                    functional
                        .constraints
                        .volumes
                        .push(PhaseConstraint::new(config.phase_pair.i.clone(), volume));
                }
                let kinetics = MacroKinetics::non_conserved(config.mobility);
                let scales = config
                    .physical_scales
                    .unwrap_or_else(|| PhysicalScales::with_sigma(config.sigma));
                let nondimensional = DimensionlessGroups::from_scales(&scales);
                let analysis = AnalysisChecklist::minimal_surface();
                let numerics = NumericalRecipe::mbo(config.volume.is_some());
                Self {
                    functional,
                    kinetics,
                    nondimensional,
                    analysis,
                    numerics,
                }
            }
            MacroCard::AnisotropicCrystal(config) => {
                let mut functional =
                    MacroInterfacialFunctional::minimal(config.phase_pair.clone(), config.sigma);
                functional.surface_terms[0].anisotropy = config.anisotropy.clone();
                if let Some(volume) = config.volume {
                    functional
                        .constraints
                        .volumes
                        .push(PhaseConstraint::new(config.phase_pair.i.clone(), volume));
                }
                if let Some(kappa) = config.bending_regularisation {
                    functional.add_bending_term(BendingTerm::new(
                        config.phase_pair.clone(),
                        kappa,
                        None,
                    ));
                }
                let kinetics = MacroKinetics::non_conserved(config.mobility);
                let scales = config
                    .physical_scales
                    .unwrap_or_else(|| PhysicalScales::with_sigma(config.sigma));
                let nondimensional = DimensionlessGroups::from_scales(&scales);
                let analysis = AnalysisChecklist::anisotropic(
                    matches!(
                        config.anisotropy,
                        AnisotropySpec::Tabulated { .. } | AnisotropySpec::Fourier { .. }
                    ),
                    config.bending_regularisation.is_some(),
                );
                let numerics = NumericalRecipe::mbo(config.volume.is_some());
                Self {
                    functional,
                    kinetics,
                    nondimensional,
                    analysis,
                    numerics,
                }
            }
            MacroCard::Membrane(config) => {
                let mut functional =
                    MacroInterfacialFunctional::minimal(config.phase_pair.clone(), config.sigma);
                functional.add_bending_term(BendingTerm::new(
                    config.phase_pair.clone(),
                    config.bending_modulus,
                    config.spontaneous_curvature,
                ));
                let kinetics = MacroKinetics::non_conserved(config.mobility).with_force(
                    ExternalForce::BulkPotential(config.pressure_jump.unwrap_or(0.0)),
                );
                let mut scales = config
                    .physical_scales
                    .unwrap_or_else(|| PhysicalScales::with_sigma(config.sigma));
                scales.bending_modulus = Some(config.bending_modulus);
                let nondimensional = DimensionlessGroups::from_scales(&scales);
                let analysis = AnalysisChecklist::membrane();
                let numerics = NumericalRecipe::level_set(true);
                Self {
                    functional,
                    kinetics,
                    nondimensional,
                    analysis,
                    numerics,
                }
            }
            MacroCard::PatternFormation(config) => {
                let mut functional = MacroInterfacialFunctional::minimal(
                    PhasePair::new(config.phase.clone().to_string() + "_in", "out"),
                    config.sigma,
                );
                functional.add_nonlocal_term(NonlocalInteraction {
                    phase: config.phase.clone(),
                    kernel: config.kernel.clone(),
                });
                let kinetics = MacroKinetics::non_conserved(config.mobility)
                    .with_force(ExternalForce::Nonlocal(config.kernel.strength()));
                let scales = config
                    .physical_scales
                    .unwrap_or_else(|| PhysicalScales::with_sigma(config.sigma));
                let nondimensional = DimensionlessGroups::from_scales(&scales);
                let analysis = AnalysisChecklist::pattern();
                let numerics = NumericalRecipe::minimizing_movements();
                Self {
                    functional,
                    kinetics,
                    nondimensional,
                    analysis,
                    numerics,
                }
            }
            MacroCard::ContactLine(config) => {
                let mut functional =
                    MacroInterfacialFunctional::minimal(config.phase_pair.clone(), config.sigma);
                functional.add_boundary_condition(BoundaryCondition {
                    phase: config.phase_pair.i.clone(),
                    kind: BoundaryKind::Young {
                        solid: config.solid.clone(),
                        equilibrium_angle: config.equilibrium_angle,
                    },
                });
                if let Some(lambda) = config.line_tension {
                    functional.add_line_tension(LineTensionTerm::new(
                        PhaseTriple::new(
                            config.phase_pair.i.as_str(),
                            config.phase_pair.j.as_str(),
                            config.solid.as_str(),
                        ),
                        lambda,
                    ));
                }
                let kinetics = MacroKinetics::non_conserved(config.mobility).with_force(
                    ExternalForce::ContactAngle {
                        equilibrium: config.equilibrium_angle,
                        friction: config.contact_friction,
                    },
                );
                let mut scales = config
                    .physical_scales
                    .unwrap_or_else(|| PhysicalScales::with_sigma(config.sigma));
                scales.line_tension = config.line_tension;
                let nondimensional = DimensionlessGroups::from_scales(&scales);
                let analysis = AnalysisChecklist::contact_line();
                let numerics = NumericalRecipe::minimizing_movements();
                Self {
                    functional,
                    kinetics,
                    nondimensional,
                    analysis,
                    numerics,
                }
            }
        }
    }

    /// Couples the template with a microlocal lift to produce Z-space aware drives.
    pub fn couple_with(self, lift: InterfaceZLift) -> MacroZBridge {
        MacroZBridge::new(self, lift)
    }
}

/// Bridges macro templates with microlocal signatures and Z pulses.
#[derive(Clone)]
pub struct MacroZBridge {
    template: MacroModelTemplate,
    lift: InterfaceZLift,
}

impl MacroZBridge {
    pub fn new(template: MacroModelTemplate, lift: InterfaceZLift) -> Self {
        Self { template, lift }
    }

    pub fn template(&self) -> &MacroModelTemplate {
        &self.template
    }

    fn average_field(field: &ArrayD<f32>) -> f64 {
        if field.len() == 0 {
            0.0
        } else {
            field.iter().map(|v| *v as f64).sum::<f64>() / field.len() as f64
        }
    }

    fn weighted_average(field: &ArrayD<f32>, weights: Option<&ArrayD<f32>>) -> f64 {
        if let Some(weights) = weights {
            let mut weighted = 0.0f64;
            let mut total = 0.0f64;
            for (value, weight) in field.iter().zip(weights.iter()) {
                let w = *weight as f64;
                if w <= 0.0 {
                    continue;
                }
                weighted += (*value as f64) * w;
                total += w;
            }
            if total > 0.0 {
                return weighted / total;
            }
        }
        Self::average_field(field)
    }

    fn average_normal(
        signature: &InterfaceSignature,
        weights: Option<&ArrayD<f32>>,
    ) -> Option<Vec<f64>> {
        let orient = signature.orientation.as_ref()?;
        let components = orient.shape().first().copied().unwrap_or(0);
        if components == 0 {
            return None;
        }

        let mut accum = vec![0.0f64; components];
        let mut total = 0.0f64;
        for idx in indices(signature.r_machine.raw_dim()) {
            let idx_dyn = IxDyn(idx.slice());
            let weight = weights
                .map(|w| w[&idx_dyn] as f64)
                .unwrap_or_else(|| signature.r_machine[&idx_dyn] as f64);
            if weight <= 0.0 {
                continue;
            }
            total += weight;
            for axis in 0..components {
                let mut orient_idx = Vec::with_capacity(orient.ndim());
                orient_idx.push(axis);
                orient_idx.extend_from_slice(idx.slice());
                accum[axis] += orient[IxDyn(&orient_idx)] as f64 * weight;
            }
        }

        if total <= 0.0 {
            return None;
        }

        for value in &mut accum {
            *value /= total;
        }

        let norm = accum.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= 1e-12 {
            None
        } else {
            for value in &mut accum {
                *value /= norm;
            }
            Some(accum)
        }
    }

    fn evaluate_anisotropy(spec: &AnisotropySpec, normal: &[f64]) -> Option<f64> {
        match spec {
            AnisotropySpec::Isotropic => Some(1.0),
            AnisotropySpec::Fourier { modes } => {
                if normal.len() < 2 {
                    return None;
                }
                let nx = normal.get(1).copied().unwrap_or(0.0);
                let ny = normal.get(0).copied().unwrap_or(0.0);
                if nx.abs() + ny.abs() <= 1e-12 {
                    return None;
                }
                let theta = ny.atan2(nx);
                let mut gamma = 1.0f64;
                for (k, amplitude) in modes {
                    let multiplier = (*k as f64) * theta;
                    gamma += amplitude * multiplier.cos();
                }
                Some(gamma.max(0.0))
            }
            AnisotropySpec::Tabulated { weights, .. } => {
                if weights.is_empty() {
                    Some(1.0)
                } else {
                    let avg = weights.iter().copied().sum::<f64>() / weights.len() as f64;
                    Some(avg.max(0.0))
                }
            }
            AnisotropySpec::Custom { .. } => None,
        }
    }

    fn anisotropy_factor_for_normal(&self, normal: &[f64]) -> Option<f64> {
        let mut weighted = 0.0f64;
        let mut sigma_total = 0.0f64;
        for term in &self.template.functional.surface_terms {
            let sigma = term.sigma as f64;
            if sigma <= 0.0 {
                continue;
            }
            let factor = Self::evaluate_anisotropy(&term.anisotropy, normal).unwrap_or(1.0);
            weighted += sigma * factor;
            sigma_total += sigma;
        }

        if sigma_total > 0.0 {
            Some(weighted / sigma_total)
        } else {
            None
        }
    }

    fn derive_feedback(
        &self,
        contributions: &CurvatureContributions,
        velocity: f64,
        pulse: &InterfaceZPulse,
    ) -> MicrolocalFeedback {
        let mut threshold_scale = 1.0f32;
        if pulse.support < 1e-6 {
            threshold_scale *= 0.9;
        }
        if !self.template.nondimensional.sharp_interface_ok() {
            threshold_scale *= 0.75;
        }
        if let Some(cahn) = self.template.nondimensional.cahn {
            if cahn < 0.02 {
                threshold_scale *= 1.1;
            } else if cahn > 0.1 {
                threshold_scale *= 0.8;
            }
        }

        if let Some(anisotropic) = contributions.anisotropic_curvature {
            let base = contributions.mean_curvature.abs().max(1e-6);
            let ratio = (anisotropic.abs() / base).max(0.0);
            if ratio > 1.5 {
                threshold_scale *= 0.85;
            } else if ratio < 0.5 {
                threshold_scale *= 1.05;
            }
        }

        let mut smoothing = if let Some(cap) = self.template.nondimensional.capillary {
            let cap = cap.max(0.0);
            let ratio = cap / (1.0 + cap);
            ratio.clamp(0.2, 0.8) as f32
        } else if threshold_scale < 1.0 {
            0.6
        } else {
            0.3
        };

        if let Some(anisotropic) = contributions.anisotropic_curvature {
            if anisotropic.abs() > contributions.mean_curvature.abs() {
                smoothing = (smoothing + 0.1).min(0.9);
            }
        }

        let vel_mag = velocity.abs() as f32;
        let mut bias_gain = 1.0f32;
        if let Some(bond) = self.template.nondimensional.bond {
            if bond > 1.0 {
                bias_gain *= 0.8;
            } else if bond < 0.1 {
                bias_gain *= 1.1;
            }
        }
        if vel_mag > 1.0 {
            bias_gain *= 1.0 + (vel_mag.min(5.0) - 1.0) * 0.15;
        } else if vel_mag < 0.05 {
            bias_gain *= 0.85;
        }

        let mut feedback = MicrolocalFeedback::default()
            .with_bias_gain(bias_gain)
            .with_smoothing(smoothing)
            .with_tempo_hint(vel_mag.max(0.01));

        let stderr = contributions.forcing.abs() as f32;
        if stderr > 0.0 {
            feedback = feedback.with_stderr_hint(stderr);
        }

        if (threshold_scale - 1.0).abs() > 1e-3 {
            feedback = feedback.with_threshold_scale(threshold_scale);
        }

        feedback
    }

    /// Projects a microlocal signature into Z-space and evaluates the macro drive.
    pub fn ingest_signature(&self, signature: &InterfaceSignature) -> MacroDrive {
        let pulse = self.lift.project(signature);
        let weights = Some(&signature.perimeter_density);
        let mean_curvature = Self::weighted_average(&signature.mean_curvature, weights);
        let signed_curvature = signature
            .signed_mean_curvature
            .as_ref()
            .map(|field| Self::weighted_average(field, weights));
        let normal = Self::average_normal(signature, weights);
        let anisotropy_factor = normal
            .as_ref()
            .and_then(|normal| self.anisotropy_factor_for_normal(normal));
        let anisotropic_curvature = match (signed_curvature, anisotropy_factor) {
            (Some(value), Some(factor)) => Some(value * factor),
            (Some(value), None) => Some(value),
            (None, Some(factor)) => Some(mean_curvature * factor),
            (None, None) => None,
        };
        let bending_operator = self
            .template
            .functional
            .bending_terms
            .iter()
            .find_map(|term| term.spontaneous_curvature)
            .map(|h0| mean_curvature - h0);
        let forcing = self.template.kinetics.forcing_from_pulse(Some(&pulse));
        let contributions = CurvatureContributions {
            mean_curvature,
            anisotropic_curvature,
            bending_operator,
            forcing,
        };
        let velocity = self.template.kinetics.evaluate_velocity(&contributions);
        let feedback = self.derive_feedback(&contributions, velocity, &pulse);
        MacroDrive {
            pulse,
            contributions,
            velocity,
            dimensionless: self.template.nondimensional.clone(),
            feedback,
        }
    }
}

/// Result of pushing a microlocal signature through a macro template.
#[derive(Clone, Debug)]
pub struct MacroDrive {
    pub pulse: InterfaceZPulse,
    pub contributions: CurvatureContributions,
    pub velocity: f64,
    pub dimensionless: DimensionlessGroups,
    feedback: MicrolocalFeedback,
}

impl MacroDrive {
    pub fn sharp_interface_ok(&self) -> bool {
        self.dimensionless.sharp_interface_ok()
    }

    pub fn microlocal_feedback(&self) -> &MicrolocalFeedback {
        &self.feedback
    }
}

/// Macro model cards (A–E) mirroring the memo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MacroCard {
    Minimal(MinimalCardConfig),
    AnisotropicCrystal(AnisotropicCardConfig),
    Membrane(MembraneCardConfig),
    PatternFormation(PatternCardConfig),
    ContactLine(ContactCardConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinimalCardConfig {
    pub phase_pair: PhasePair,
    pub sigma: f64,
    pub mobility: f64,
    pub volume: Option<f64>,
    pub physical_scales: Option<PhysicalScales>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnisotropicCardConfig {
    pub phase_pair: PhasePair,
    pub sigma: f64,
    pub anisotropy: AnisotropySpec,
    pub mobility: f64,
    pub volume: Option<f64>,
    pub bending_regularisation: Option<f64>,
    pub physical_scales: Option<PhysicalScales>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MembraneCardConfig {
    pub phase_pair: PhasePair,
    pub sigma: f64,
    pub bending_modulus: f64,
    pub spontaneous_curvature: Option<f64>,
    pub mobility: f64,
    pub pressure_jump: Option<f64>,
    pub physical_scales: Option<PhysicalScales>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternCardConfig {
    pub phase: PhaseId,
    pub sigma: f64,
    pub kernel: KernelSpec,
    pub mobility: f64,
    pub physical_scales: Option<PhysicalScales>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactCardConfig {
    pub phase_pair: PhasePair,
    pub sigma: f64,
    pub equilibrium_angle: f64,
    pub line_tension: Option<f64>,
    pub contact_friction: f64,
    pub mobility: f64,
    pub solid: PhaseId,
    pub physical_scales: Option<PhysicalScales>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::microlocal::{
        InterfaceGauge, InterfaceSignature, InterfaceZConductor, InterfaceZLift,
        MicrolocalGaugeBank,
    };
    use crate::util::math::LeechProjector;
    use ndarray::{array, ArrayD, IxDyn};

    #[test]
    fn template_bank_registers_and_couples_templates() {
        let minimal = MinimalCardConfig {
            phase_pair: PhasePair::new("A", "B"),
            sigma: 0.1,
            mobility: 1.0,
            volume: None,
            physical_scales: None,
        };
        let template = MacroModelTemplate::from_card(MacroCard::Minimal(minimal.clone()));
        let mut bank = MacroTemplateBank::new();
        assert!(bank.register("minimal", template.clone()));
        assert!(!bank.register("minimal", template.clone()));

        let membrane = MembraneCardConfig {
            phase_pair: PhasePair::new("A", "C"),
            sigma: 0.2,
            bending_modulus: 0.05,
            spontaneous_curvature: Some(0.1),
            mobility: 0.5,
            pressure_jump: Some(0.02),
            physical_scales: None,
        };
        bank = bank.with_card("membrane", MacroCard::Membrane(membrane));
        assert_eq!(bank.len(), 2);
        assert!(bank.get("membrane").is_some());

        let lift = InterfaceZLift::new(&[1.0], LeechProjector::new(24, 0.15));
        let bridge = bank
            .couple("minimal", lift.clone())
            .expect("bridge should be produced");
        assert_eq!(bridge.template().kinetics.flow, FlowKind::NonConserved);

        let coupled = bank.couple_all(&lift);
        assert_eq!(coupled.len(), 2);
        let ids: Vec<_> = coupled.ids().collect();
        assert_eq!(ids, vec!["minimal", "membrane"]);
        let (first_id, first_bridge) = coupled.iter().next().expect("bridge missing");
        assert_eq!(first_id, "minimal");
        assert_eq!(first_bridge.template().functional.surface_terms.len(), 1);
    }

    #[test]
    fn dimensionless_groups_compute_expected_values() {
        let scales = PhysicalScales {
            sigma: 0.072,
            reference_length: Some(0.01),
            velocity_scale: Some(0.02),
            viscosity: Some(1.0e-3),
            density_difference: Some(100.0),
            gravity: Some(9.81),
            interface_thickness: Some(1.0e-6),
            diffusion: Some(1.0e-9),
            bending_modulus: Some(1.0e-19),
            line_tension: Some(1.0e-7),
            mobility_ratio: Some(0.5),
        };
        let groups = DimensionlessGroups::from_scales(&scales);
        assert!((groups.capillary.unwrap() - 2.7777778e-4).abs() < 1e-10);
        assert!((groups.bond.unwrap() - 1.3625).abs() < 1e-4);
        assert!((groups.cahn.unwrap() - 1.0e-4).abs() < 1e-10);
        assert!((groups.peclet.unwrap() - 2.0e5).abs() < 1e-2);
        assert!((groups.bending_length.unwrap() - 1.1785113e-9).abs() < 1e-15);
        assert!((groups.line_tension_length.unwrap() - 1.3888889e-6).abs() < 1e-12);
        assert_eq!(groups.mobility_ratio, Some(0.5));
        assert!(groups.sharp_interface_ok());
    }

    #[test]
    fn minimal_card_builds_expected_template() {
        let config = MinimalCardConfig {
            phase_pair: PhasePair::new("A", "B"),
            sigma: 0.1,
            mobility: 1.0,
            volume: Some(1.0),
            physical_scales: None,
        };
        let template = MacroModelTemplate::from_card(MacroCard::Minimal(config));
        assert_eq!(template.functional.surface_terms.len(), 1);
        assert_eq!(template.functional.bending_terms.len(), 0);
        assert_eq!(template.functional.constraints.volumes.len(), 1);
        assert_eq!(template.kinetics.flow, FlowKind::NonConserved);
        assert_eq!(template.numerics.algorithm, Algorithm::MerrimanBenceOsher);
    }

    #[test]
    fn kinetics_evaluates_velocity_with_anisotropy_and_bending() {
        let kinetics = MacroKinetics::non_conserved(2.0);
        let contributions = CurvatureContributions {
            mean_curvature: 0.5,
            anisotropic_curvature: Some(0.6),
            bending_operator: Some(0.1),
            forcing: 0.2,
        };
        let velocity = kinetics.evaluate_velocity(&contributions);
        assert!((velocity - 1.4).abs() < 1e-12);
    }

    #[test]
    fn bridge_converts_microlocal_signature_into_drive() {
        let config = MinimalCardConfig {
            phase_pair: PhasePair::new("A", "B"),
            sigma: 0.1,
            mobility: 1.0,
            volume: None,
            physical_scales: {
                let mut scales = PhysicalScales::with_sigma(0.1);
                scales.reference_length = Some(1.0);
                scales.velocity_scale = Some(0.5);
                scales.viscosity = Some(0.2);
                scales.interface_thickness = Some(0.2);
                scales.density_difference = Some(1.0);
                scales.gravity = Some(9.81);
                Some(scales)
            },
        };
        let template = MacroModelTemplate::from_card(MacroCard::Minimal(config));
        let lift = InterfaceZLift::new(&[1.0], LeechProjector::new(24, 0.1));
        let bridge = template.couple_with(lift);

        let shape = IxDyn(&[1, 1, 1]);
        let r_machine = ArrayD::from_elem(shape.clone(), 1.0f32);
        let raw_density = ArrayD::from_elem(shape.clone(), 0.2f32);
        let perimeter_density = ArrayD::from_elem(shape.clone(), 0.1f32);
        let mean_curvature = ArrayD::from_elem(shape.clone(), 0.5f32);
        let signed_mean = ArrayD::from_elem(shape.clone(), 0.55f32);
        let signature = InterfaceSignature {
            r_machine,
            raw_density,
            perimeter_density,
            mean_curvature,
            signed_mean_curvature: Some(signed_mean),
            orientation: None,
            kappa_d: 1.0,
            radius: 1,
            physical_radius: 1.0,
        };

        let drive = bridge.ingest_signature(&signature);
        assert!(drive.velocity > 0.0);
        assert!((drive.contributions.mean_curvature - 0.5).abs() < 1e-6);
        assert!(drive.pulse.support > 0.0);
        assert!(!drive.sharp_interface_ok());
        let feedback = drive.microlocal_feedback();
        assert!(feedback.bias_gain.unwrap() > 0.0);
        assert!(feedback.smoothing.unwrap() >= 0.2);
        assert!(feedback.tempo_hint.unwrap() > 0.0);
        assert!(feedback.threshold_scale.unwrap() < 1.0);
    }

    #[test]
    fn bridge_uses_perimeter_weighting_for_curvature() {
        let config = MinimalCardConfig {
            phase_pair: PhasePair::new("A", "B"),
            sigma: 0.2,
            mobility: 1.0,
            volume: None,
            physical_scales: None,
        };
        let template = MacroModelTemplate::from_card(MacroCard::Minimal(config));
        let lift = InterfaceZLift::new(&[1.0], LeechProjector::new(24, 0.2));
        let bridge = template.couple_with(lift);

        let shape = IxDyn(&[1, 2]);
        let r_machine = ArrayD::from_shape_vec(shape.clone(), vec![1.0f32, 0.0f32]).unwrap();
        let raw_density = ArrayD::zeros(shape.clone());
        let perimeter_density =
            ArrayD::from_shape_vec(shape.clone(), vec![1.0f32, 0.0f32]).unwrap();
        let mean_curvature = ArrayD::from_shape_vec(shape.clone(), vec![2.0f32, 20.0f32]).unwrap();
        let signed_mean = ArrayD::from_shape_vec(shape.clone(), vec![2.0f32, 20.0f32]).unwrap();

        let signature = InterfaceSignature {
            r_machine,
            raw_density,
            perimeter_density,
            mean_curvature,
            signed_mean_curvature: Some(signed_mean),
            orientation: None,
            kappa_d: 1.0,
            radius: 1,
            physical_radius: 1.0,
        };

        let drive = bridge.ingest_signature(&signature);
        assert!((drive.contributions.mean_curvature - 2.0).abs() < 1e-6);
        assert!(
            (drive
                .contributions
                .anisotropic_curvature
                .expect("anisotropic curvature missing")
                - 2.0)
                .abs()
                < 1e-6
        );
        assert!((drive.velocity - 2.0).abs() < 1e-6);
    }

    #[test]
    fn bridge_applies_fourier_anisotropy_from_orientation() {
        let pair = PhasePair::new("A", "B");
        let mut functional = MacroInterfacialFunctional::minimal(pair.clone(), 0.3);
        functional.surface_terms[0].anisotropy = AnisotropySpec::Fourier {
            modes: vec![(4, 0.5)],
        };
        let kinetics = MacroKinetics::non_conserved(1.0);
        let template = MacroModelTemplate {
            functional,
            kinetics,
            nondimensional: DimensionlessGroups::default(),
            analysis: AnalysisChecklist::minimal_surface(),
            numerics: NumericalRecipe::mbo(false),
        };
        let lift = InterfaceZLift::new(&[1.0, 0.0], LeechProjector::new(24, 0.5));
        let bridge = MacroZBridge::new(template, lift);

        let shape = IxDyn(&[1, 1]);
        let r_machine = ArrayD::from_elem(shape.clone(), 1.0f32);
        let raw_density = ArrayD::from_elem(shape.clone(), 0.1f32);
        let perimeter_density = ArrayD::from_elem(shape.clone(), 1.0f32);
        let mean_curvature = ArrayD::from_elem(shape.clone(), 2.0f32);
        let signed_mean = ArrayD::from_elem(shape.clone(), 2.0f32);
        let orientation = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.0f32, 1.0f32]).unwrap();

        let signature = InterfaceSignature {
            r_machine,
            raw_density,
            perimeter_density,
            mean_curvature,
            signed_mean_curvature: Some(signed_mean),
            orientation: Some(orientation),
            kappa_d: 1.0,
            radius: 1,
            physical_radius: 1.0,
        };

        let drive = bridge.ingest_signature(&signature);
        let anisotropic = drive
            .contributions
            .anisotropic_curvature
            .expect("anisotropic curvature missing");
        assert!((drive.contributions.mean_curvature - 2.0).abs() < 1e-6);
        assert!((anisotropic - 3.0).abs() < 1e-6);
        assert!((drive.velocity - 3.0).abs() < 1e-6);
    }

    #[test]
    fn drive_matched_emits_feedback_for_matching_gauge() {
        let mut gauges = MicrolocalGaugeBank::new();
        gauges.register("minimal", InterfaceGauge::new(1.0, 1.0));
        let lift = InterfaceZLift::new(&[1.0], LeechProjector::new(24, 0.25));
        let mut conductor = InterfaceZConductor::from_bank(gauges, lift);
        let mask = array![[0.0, 0.0], [0.0, 1.0]].into_dyn();
        let report = conductor.step(&mask, None, None, None);
        assert_eq!(report.gauge_id(0), Some("minimal"));

        let card = MinimalCardConfig {
            phase_pair: PhasePair::new("A", "B"),
            sigma: 0.1,
            mobility: 1.0,
            volume: None,
            physical_scales: None,
        };
        let mut templates = MacroTemplateBank::new();
        templates.register_card("minimal", MacroCard::Minimal(card));

        let drives = templates.drive_matched(&report);
        assert_eq!(drives.len(), 1);
        let (_, drive) = drives.iter().next().expect("drive missing");
        let microlocal_feedback = drive.microlocal_feedback();
        assert!(
            microlocal_feedback.threshold_scale.is_some()
                || microlocal_feedback.bias_gain.is_some()
        );

        let thresholds_before = conductor.gauge_thresholds();
        let bias_before = conductor.bias_gain();
        let feedback = templates
            .feedback_from_report(&report)
            .expect("feedback missing");
        conductor.apply_feedback(&feedback);
        let thresholds_after = conductor.gauge_thresholds();
        let bias_after = conductor.bias_gain();
        assert!(thresholds_before != thresholds_after || (bias_before - bias_after).abs() > 1e-6);
    }
}
