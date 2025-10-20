// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Open-cartesian topos guards that keep the pure tensor stack loop-free and
//! numerically stable even in the presence of extreme curvatures.
//!
//! The implementation focuses on three guarantees that were repeatedly flagged
//! as weaknesses in the original stack:
//!
//! * **Numerical safety** – All tensors are validated for finite components and
//!   projected through a saturation window so NaNs and infinities are rewritten
//!   into bounded values.
//! * **Loop freedom** – Traversals through fractal depths are capped by an
//!   "open cartesian" horizon which ensures self-referential rewrites never
//!   re-enter the same stratum.
//! * **Solver determinism** – The conjugate gradient solver exposes explicit
//!   tolerance and iteration limits so hyperbolic Jacobians cannot silently
//!   diverge.
//!
//! The module intentionally stays allocation-light so the new guards can be used
//! from both CPU-only and WASM environments without fighting the borrow checker.

use super::{fractal::FractalPatch, PureResult, Tensor, TensorError};
use core::{cmp, f64::consts::PI as PI64};

const DEFAULT_MODALITY_PERMEABILITY: f32 = 0.12;
const DEFAULT_GRAPH_PERMEABILITY: f32 = 0.08;

fn validate_permeability(label: &'static str, permeability: f32) -> PureResult<()> {
    if !permeability.is_finite() || permeability < 0.0 || permeability > 1.0 {
        return Err(TensorError::InvalidValue { label });
    }
    Ok(())
}

fn permeable_clamp(value: f32, limit: f32, permeability: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    if limit <= 0.0 {
        return 0.0;
    }
    let magnitude = value.abs();
    if magnitude <= limit {
        return value;
    }
    let sign = value.signum();
    if permeability <= f32::EPSILON {
        return sign * limit;
    }
    let headroom = limit * permeability;
    let softened = 1.0 - (-((magnitude - limit) / (headroom + limit))).exp();
    sign * (limit + headroom * softened.min(1.0))
}

/// Numerically guards the Lawvere–Tierney topology that keeps probabilistic data j-closed.
#[derive(Clone, Copy, Debug)]
pub struct LawvereTierneyGuard {
    density_min: f32,
    density_max: f32,
    mass_tolerance: f32,
}

impl LawvereTierneyGuard {
    /// Creates a new guard ensuring densities stay within the provided window and
    /// that normalisations land within `mass_tolerance` of unit mass.
    pub fn new(density_min: f32, density_max: f32, mass_tolerance: f32) -> PureResult<Self> {
        if !density_min.is_finite() || !density_max.is_finite() || !mass_tolerance.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lawvere_tierney_params",
                value: f32::NAN,
            });
        }
        if density_min <= 0.0 || density_max < density_min {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_density_window",
            });
        }
        if mass_tolerance <= 0.0 || mass_tolerance >= 1.0 {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_mass_tolerance",
            });
        }
        Ok(Self {
            density_min,
            density_max,
            mass_tolerance,
        })
    }

    /// Returns the minimum density admitted by the guard.
    pub fn density_min(&self) -> f32 {
        self.density_min
    }

    /// Returns the maximum density admitted by the guard.
    pub fn density_max(&self) -> f32 {
        self.density_max
    }

    /// Returns the tolerance allowed when projecting to unit mass.
    pub fn mass_tolerance(&self) -> f32 {
        self.mass_tolerance
    }

    fn guard_density(&self, density: f32) -> PureResult<()> {
        if !density.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lawvere_tierney_density",
                value: density,
            });
        }
        if density <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "density_non_positive",
            });
        }
        if density < self.density_min || density > self.density_max {
            return Err(TensorError::InvalidValue {
                label: "density_window_violation",
            });
        }
        Ok(())
    }

    fn guard_mass(&self, mass: f32) -> PureResult<()> {
        if !mass.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lawvere_tierney_mass",
                value: mass,
            });
        }
        if mass <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_mass_non_positive",
            });
        }
        if mass < self.density_min {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_mass_too_small",
            });
        }
        if mass > 1.0 + self.mass_tolerance {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_mass_too_large",
            });
        }
        Ok(())
    }

    fn guard_cover_mass(&self, total_mass: f32) -> PureResult<()> {
        if !total_mass.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lawvere_tierney_cover_mass",
                value: total_mass,
            });
        }
        if total_mass <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_cover_empty",
            });
        }
        let deviation = (total_mass - 1.0).abs();
        if deviation > self.mass_tolerance {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_cover_mass_violation",
            });
        }
        Ok(())
    }

    /// Projects a probability slice to the guarded subtopos by clipping
    /// non-finite values, enforcing the density window, and re-normalising.
    pub fn project_slice(
        &self,
        label: &'static str,
        slice: &mut [f32],
        saturation: f32,
    ) -> PureResult<()> {
        if slice.is_empty() {
            return Err(TensorError::EmptyInput(label));
        }
        let mut sum = 0.0f32;
        for value in slice.iter_mut() {
            if !value.is_finite() {
                *value = 0.0;
            }
            *value = value.clamp(0.0, saturation).min(self.density_max);
            sum += *value;
        }
        if sum <= 0.0 {
            return Err(TensorError::NonFiniteValue { label, value: sum });
        }
        for value in slice.iter_mut() {
            *value /= sum;
            if *value > 0.0 && *value < self.density_min {
                *value = self.density_min;
            }
        }
        let renorm_sum: f32 = slice.iter().sum();
        if renorm_sum <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label,
                value: renorm_sum,
            });
        }
        for value in slice.iter_mut() {
            *value /= renorm_sum;
            if *value > self.density_max {
                *value = self.density_max;
            }
            if *value > 0.0 {
                if *value + self.mass_tolerance < self.density_min {
                    return Err(TensorError::InvalidValue {
                        label: "lawvere_tierney_probability_density_floor",
                    });
                }
            }
        }
        let final_sum: f32 = slice.iter().sum();
        let deviation = (final_sum - 1.0).abs();
        if deviation > self.mass_tolerance {
            let scale = 1.0 / final_sum;
            for value in slice.iter_mut() {
                *value *= scale;
            }
        }
        let final_sum: f32 = slice.iter().sum();
        let deviation = (final_sum - 1.0).abs();
        if deviation > self.mass_tolerance {
            return Err(TensorError::InvalidValue {
                label: "lawvere_tierney_probability_mass",
            });
        }
        Ok(())
    }
}

/// Open box in a negatively curved Z-space site.
#[derive(Clone, Debug)]
pub struct ZBox {
    centers: Vec<Vec<f32>>,
    radii: Vec<f32>,
    density: f32,
}

impl ZBox {
    /// Builds a new κ-box. Each factor is described by a centre and a radius.
    pub fn new(centers: Vec<Vec<f32>>, radii: Vec<f32>, density: f32) -> PureResult<Self> {
        if centers.is_empty() || radii.is_empty() {
            return Err(TensorError::EmptyInput("zbox_factors"));
        }
        if centers.len() != radii.len() {
            return Err(TensorError::DataLength {
                expected: centers.len(),
                got: radii.len(),
            });
        }
        for center in centers.iter() {
            if center.is_empty() {
                return Err(TensorError::EmptyInput("zbox_center"));
            }
            if center.iter().any(|c| !c.is_finite()) {
                return Err(TensorError::NonFiniteValue {
                    label: "zbox_center",
                    value: f32::NAN,
                });
            }
        }
        for radius in radii.iter() {
            if !radius.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "zbox_radius",
                    value: *radius,
                });
            }
            if *radius <= 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "zbox_radius_non_positive",
                });
            }
        }
        if !density.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zbox_density",
                value: density,
            });
        }
        Ok(Self {
            centers,
            radii,
            density,
        })
    }

    /// Returns the number of factors composing the κ-box.
    pub fn arity(&self) -> usize {
        self.radii.len()
    }

    /// Returns the density weight assigned to the box.
    pub fn density(&self) -> f32 {
        self.density
    }

    /// Returns the dimension of the ambient Z-space for the i-th factor.
    pub fn factor_dimension(&self, index: usize) -> usize {
        self.centers[index].len()
    }

    /// Computes the total hyperbolic volume of the κ-box.
    pub fn hyperbolic_volume(&self, curvature: f32) -> PureResult<f32> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        let mut volume = 1.0f32;
        for (i, radius) in self.radii.iter().enumerate() {
            let dim = self.factor_dimension(i);
            let factor = hyperbolic_ball_volume(curvature, *radius, dim)?;
            volume *= factor.max(0.0);
        }
        Ok(volume)
    }

    /// Returns the probability mass assigned to this κ-box.
    pub fn probability_mass(&self, curvature: f32) -> PureResult<f32> {
        Ok(self.density * self.hyperbolic_volume(curvature)?)
    }

    fn validate_radius_window(&self, min: f32, max: f32) -> PureResult<()> {
        if min <= 0.0 || max <= 0.0 || max < min {
            return Err(TensorError::InvalidValue {
                label: "zbox_radius_window",
            });
        }
        for radius in self.radii.iter() {
            if *radius < min || *radius > max {
                return Err(TensorError::InvalidValue {
                    label: "zbox_radius_window_violation",
                });
            }
        }
        Ok(())
    }
}

/// Guards the κ-box site attached to an open-cartesian topos.
#[derive(Clone, Debug)]
pub struct ZBoxSite {
    curvature: f32,
    radius_min: f32,
    radius_max: f32,
    guard: LawvereTierneyGuard,
}

impl ZBoxSite {
    /// Builds the default κ-box site for the provided curvature.
    pub fn default_for(curvature: f32) -> PureResult<Self> {
        let guard = LawvereTierneyGuard::new(1e-6, 1e3, 1e-5)?;
        Ok(Self {
            curvature,
            radius_min: 1e-3,
            radius_max: 64.0,
            guard,
        })
    }

    /// Adjusts the admissible radius window.
    pub fn with_radius_window(mut self, min: f32, max: f32) -> PureResult<Self> {
        if !min.is_finite() || !max.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zbox_radius_window",
                value: f32::NAN,
            });
        }
        if min <= 0.0 || max <= 0.0 || max < min {
            return Err(TensorError::InvalidValue {
                label: "zbox_radius_window",
            });
        }
        self.radius_min = min;
        self.radius_max = max;
        Ok(self)
    }

    /// Replaces the internal Lawvere–Tierney guard.
    pub fn with_guard(mut self, guard: LawvereTierneyGuard) -> Self {
        self.guard = guard;
        self
    }

    /// Returns the curvature parameter for the site.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the Lawvere–Tierney guard used by the site.
    pub fn guard(&self) -> &LawvereTierneyGuard {
        &self.guard
    }

    /// Ensures a single κ-box is admissible.
    pub fn guard_box(&self, zbox: &ZBox) -> PureResult<()> {
        zbox.validate_radius_window(self.radius_min, self.radius_max)?;
        self.guard.guard_density(zbox.density())?;
        let mass = zbox.probability_mass(self.curvature)?;
        self.guard.guard_mass(mass)?;
        Ok(())
    }

    /// Ensures a cover of κ-boxes is admissible and mass-preserving.
    pub fn guard_cover(&self, cover: &[ZBox]) -> PureResult<()> {
        if cover.is_empty() {
            return Err(TensorError::EmptyInput("zbox_cover"));
        }
        let mut mass = 0.0f32;
        for zbox in cover.iter() {
            self.guard_box(zbox)?;
            mass += zbox.probability_mass(self.curvature)?;
        }
        self.guard.guard_cover_mass(mass)
    }
}

fn hyperbolic_ball_volume(curvature: f32, radius: f32, dimension: usize) -> PureResult<f32> {
    if dimension == 0 {
        return Err(TensorError::EmptyInput("hyperbolic_dimension"));
    }
    let kappa = curvature;
    if kappa >= 0.0 {
        return Err(TensorError::NonHyperbolicCurvature { curvature: kappa });
    }
    let lambda = (-kappa).sqrt();
    let n = dimension as i32;
    let omega = unit_sphere_surface((dimension - 1) as i32);
    let steps = 64;
    let h = radius / steps as f32;
    let mut integral = 0.0f64;
    for i in 0..=steps {
        let t = i as f32 * h;
        let sinh_term = ((lambda * t) as f64).sinh() / lambda as f64;
        let power = if n == 1 {
            1.0
        } else {
            sinh_term.powi((n - 1).max(0))
        };
        let weight = if i == 0 || i == steps {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        integral += weight * power;
    }
    integral *= (h as f64) / 3.0;
    let volume = omega * integral;
    if !volume.is_finite() || volume <= 0.0 {
        return Err(TensorError::InvalidValue {
            label: "hyperbolic_volume",
        });
    }
    Ok(volume as f32)
}

fn unit_sphere_surface(dimension: i32) -> f64 {
    match dimension {
        -1 => 2.0,
        0 => 2.0,
        1 => 2.0 * PI64,
        2 => 4.0 * PI64,
        d if d >= 3 => {
            let n = (d + 1) as f64;
            2.0 * PI64.powf(n / 2.0) / gamma(n / 2.0)
        }
        _ => 2.0,
    }
}

fn gamma(z: f64) -> f64 {
    if z == 0.5 {
        return PI64.sqrt();
    }
    if z == 1.0 {
        return 1.0;
    }
    if z == 2.0 {
        return 1.0;
    }
    if z.fract() == 0.5 {
        let n = (z - 0.5) as usize;
        let mut numerator = 1.0f64;
        for k in 1..=n {
            numerator *= (2 * k) as f64;
        }
        let mut denominator = 1.0f64;
        for k in 1..=n {
            denominator *= (2 * k - 1) as f64;
        }
        numerator / denominator * PI64.sqrt()
    } else if z.fract() == 0.0 {
        let n = z as usize - 1;
        (1..=n).fold(1.0f64, |acc, v| acc * v as f64)
    } else {
        lanczos_gamma(z)
    }
}

fn lanczos_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const P: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        PI64 / ((PI64 * z).sin() * lanczos_gamma(1.0 - z))
    } else {
        let z = z - 1.0;
        let mut x = P[0];
        for (i, p) in P.iter().enumerate().skip(1) {
            x += p / (z + i as f64);
        }
        let t = z + G + 0.5;
        (2.0 * PI64).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

/// Maintains safety envelopes for tensors travelling through the pure stack.
#[derive(Clone, Debug)]
pub struct OpenCartesianTopos {
    curvature: f32,
    tolerance: f32,
    saturation: f32,
    porosity: f32,
    max_depth: usize,
    max_volume: usize,
    site: ZBoxSite,
}

impl OpenCartesianTopos {
    /// Builds a new guard. `curvature` must remain negative, `tolerance` and
    /// `saturation` must be positive. `max_depth` and `max_volume` are expressed
    /// in absolute counts rather than logarithms so callers can wire them to
    /// dataset or topology specific limits.
    pub fn new(
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PureResult<Self> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if tolerance <= 0.0 {
            return Err(TensorError::NonPositiveTolerance { tolerance });
        }
        if saturation <= 0.0 {
            return Err(TensorError::NonPositiveSaturation { saturation });
        }
        if max_depth == 0 {
            return Err(TensorError::EmptyInput("topos max depth"));
        }
        if max_volume == 0 {
            return Err(TensorError::EmptyInput("topos max volume"));
        }
        let site = ZBoxSite::default_for(curvature)?;
        Ok(Self {
            curvature,
            tolerance,
            saturation,
            porosity: 0.2,
            max_depth,
            max_volume,
            site,
        })
    }

    /// Returns the curvature enforced by the topos.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Tolerance applied when inverting Jacobians or measuring residuals.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Returns the saturation limit used to absorb overflows.
    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    /// Returns the permeability applied while saturating values.
    pub fn porosity(&self) -> f32 {
        self.porosity
    }

    /// Maximum permitted traversal depth before the guard considers the topos
    /// closed for the current rewrite.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Maximum tensor volume allowed inside the topos envelope.
    pub fn max_volume(&self) -> usize {
        self.max_volume
    }

    /// Returns the κ-box site guarded by this topos.
    pub fn site(&self) -> &ZBoxSite {
        &self.site
    }

    /// Replaces the κ-box site guard, returning a new topos instance.
    pub fn with_site(mut self, site: ZBoxSite) -> PureResult<Self> {
        if (site.curvature() - self.curvature).abs() > self.tolerance {
            return Err(TensorError::CurvatureMismatch {
                expected: self.curvature,
                got: site.curvature(),
            });
        }
        self.site = site;
        Ok(self)
    }

    /// Replaces the porosity used during saturation.
    pub fn with_porosity(mut self, porosity: f32) -> PureResult<Self> {
        if !porosity.is_finite() || porosity < 0.0 || porosity > 1.0 {
            return Err(TensorError::PorosityOutOfRange { porosity });
        }
        self.porosity = porosity;
        Ok(self)
    }

    /// Ensures the provided tensor stays finite and within the permitted volume.
    pub fn guard_tensor(&self, label: &'static str, tensor: &Tensor) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        let volume = rows.saturating_mul(cols);
        if volume > self.max_volume {
            return Err(TensorError::TensorVolumeExceeded {
                label,
                volume,
                max_volume: self.max_volume,
            });
        }
        self.guard_slice(label, tensor.data())
    }

    /// Ensures the provided κ-box satisfies the site guard.
    pub fn guard_zbox(&self, zbox: &ZBox) -> PureResult<()> {
        self.site.guard_box(zbox)
    }

    /// Ensures a κ-box cover is admissible and mass-preserving.
    pub fn guard_cover(&self, cover: &[ZBox]) -> PureResult<()> {
        self.site.guard_cover(cover)
    }

    /// Ensures a buffer remains finite.
    pub fn guard_slice(&self, label: &'static str, slice: &[f32]) -> PureResult<()> {
        for &value in slice {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        Ok(())
    }

    /// Normalises a probability slice while keeping it within the topos saturation window.
    pub fn guard_probability_slice(
        &self,
        label: &'static str,
        slice: &mut [f32],
    ) -> PureResult<()> {
        self.site
            .guard()
            .project_slice(label, slice, self.saturation)
    }

    /// Normalises a probability tensor in-place.
    pub fn guard_probability_tensor(
        &self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        self.guard_probability_slice(label, tensor.data_mut())
    }

    /// Catches runaway recursion depth before it can trigger a feedback loop.
    pub fn ensure_loop_free(&self, depth: usize) -> PureResult<()> {
        if depth >= self.max_depth {
            return Err(TensorError::LoopDetected {
                depth,
                max_depth: self.max_depth,
            });
        }
        Ok(())
    }

    /// Validates a fractal patch before it is ingested by other pure modules.
    pub fn guard_fractal_patch(&self, label: &'static str, patch: &FractalPatch) -> PureResult<()> {
        self.ensure_loop_free(patch.depth() as usize)?;
        self.guard_tensor(label, patch.relation())
    }

    /// Saturates a scalar into the finite window enforced by the topos.
    pub fn saturate(&self, value: f32) -> f32 {
        porous_mix(value, self.saturation, self.porosity)
    }

    /// Saturates an entire slice in-place.
    pub fn saturate_slice(&self, slice: &mut [f32]) {
        for value in slice.iter_mut() {
            *value = self.saturate(*value);
        }
    }
}

/// Local envelope that specialises an [`OpenCartesianTopos`] guard for a specific modality.
#[derive(Clone, Copy, Debug)]
pub struct ModalityProfile {
    max_volume: usize,
    local_saturation: Option<f32>,
    permeability: f32,
}

impl ModalityProfile {
    /// Creates a new modality envelope. `max_volume` must be non-zero and `local_saturation`,
    /// when provided, must be positive.
    pub fn new(max_volume: usize, local_saturation: Option<f32>) -> PureResult<Self> {
        if max_volume == 0 {
            return Err(TensorError::EmptyInput("modality_max_volume"));
        }
        if let Some(saturation) = local_saturation {
            if saturation <= 0.0 || !saturation.is_finite() {
                return Err(TensorError::NonPositiveSaturation { saturation });
            }
        }
        Ok(Self {
            max_volume,
            local_saturation,
            permeability: DEFAULT_MODALITY_PERMEABILITY,
        })
    }

    /// Returns the maximum tensor volume admitted by the modality.
    pub fn max_volume(&self) -> usize {
        self.max_volume
    }

    /// Returns the permeability applied when clamping modality values.
    pub fn permeability(&self) -> f32 {
        self.permeability
    }

    /// Overrides the permeability while preserving other settings.
    pub fn with_permeability(mut self, permeability: f32) -> PureResult<Self> {
        validate_permeability("modality_permeability", permeability)?;
        self.permeability = permeability;
        Ok(self)
    }

    fn effective_saturation(&self, topos: &OpenCartesianTopos) -> f32 {
        self.local_saturation
            .unwrap_or_else(|| topos.saturation())
            .min(topos.saturation())
            .max(0.0)
    }

    fn guard_volume(&self, label: &'static str, volume: usize) -> PureResult<()> {
        if volume > self.max_volume {
            return Err(TensorError::TensorVolumeExceeded {
                label,
                volume,
                max_volume: self.max_volume,
            });
        }
        Ok(())
    }

    fn rewrite_slice(
        &self,
        topos: &OpenCartesianTopos,
        label: &'static str,
        slice: &mut [f32],
    ) -> PureResult<()> {
        let saturation = self.effective_saturation(topos);
        for value in slice.iter_mut() {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label,
                    value: *value,
                });
            }
            *value = permeable_clamp(*value, saturation, self.permeability);
        }
        topos.guard_slice(label, slice)
    }

    fn rewrite_tensor(
        &self,
        topos: &OpenCartesianTopos,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        self.guard_volume(label, rows.saturating_mul(cols))?;
        self.rewrite_slice(topos, label, tensor.data_mut())?;
        topos.guard_tensor(label, tensor)
    }

    fn validate_for_topos(&self, topos: &OpenCartesianTopos) -> PureResult<()> {
        if self.max_volume > topos.max_volume() {
            return Err(TensorError::InvalidValue {
                label: "modality_volume_exceeds_topos",
            });
        }
        Ok(())
    }
}

/// Guard configuration for structured graph data rewritten through an open-cartesian topos.
#[derive(Clone, Copy, Debug)]
pub struct GraphGuardProfile {
    max_nodes: usize,
    max_edges: usize,
    max_degree: usize,
    symmetry_tolerance: f32,
    activation_threshold: f32,
    edge_saturation: Option<f32>,
    permeability: f32,
}

impl GraphGuardProfile {
    /// Creates a graph guard profile. Node, edge, and degree counts must be positive.
    pub fn new(
        max_nodes: usize,
        max_edges: usize,
        max_degree: usize,
        symmetry_tolerance: f32,
        activation_threshold: f32,
        edge_saturation: Option<f32>,
    ) -> PureResult<Self> {
        if max_nodes == 0 {
            return Err(TensorError::EmptyInput("graph_max_nodes"));
        }
        if max_edges == 0 {
            return Err(TensorError::EmptyInput("graph_max_edges"));
        }
        if max_degree == 0 {
            return Err(TensorError::EmptyInput("graph_max_degree"));
        }
        if symmetry_tolerance < 0.0 || !symmetry_tolerance.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "graph_symmetry_tolerance",
            });
        }
        if activation_threshold < 0.0 || !activation_threshold.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "graph_activation_threshold",
            });
        }
        if let Some(saturation) = edge_saturation {
            if saturation <= 0.0 || !saturation.is_finite() {
                return Err(TensorError::NonPositiveSaturation { saturation });
            }
        }
        Ok(Self {
            max_nodes,
            max_edges,
            max_degree,
            symmetry_tolerance,
            activation_threshold,
            edge_saturation,
            permeability: DEFAULT_GRAPH_PERMEABILITY,
        })
    }

    /// Maximum number of graph nodes admitted by the guard.
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
    }

    /// Returns the permeability admitted while guarding adjacency tensors.
    pub fn permeability(&self) -> f32 {
        self.permeability
    }

    fn validate_for_topos(&self, topos: &OpenCartesianTopos) -> PureResult<()> {
        let max_possible = topos.max_volume();
        if self.max_edges > max_possible {
            return Err(TensorError::InvalidValue {
                label: "graph_edges_exceed_topos_volume",
            });
        }
        Ok(())
    }

    fn expected_len(&self, node_count: usize) -> PureResult<usize> {
        node_count
            .checked_mul(node_count)
            .ok_or(TensorError::InvalidDimensions {
                rows: node_count,
                cols: node_count,
            })
    }

    fn saturate_value(&self, topos: &OpenCartesianTopos, value: f32) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        let limit = self
            .edge_saturation
            .unwrap_or_else(|| topos.saturation())
            .min(topos.saturation())
            .max(0.0);
        permeable_clamp(value, limit, self.permeability)
    }

    fn guard_shape(&self, node_count: usize, len: usize) -> PureResult<usize> {
        if node_count == 0 {
            return Err(TensorError::EmptyInput("graph_nodes"));
        }
        if node_count > self.max_nodes {
            return Err(TensorError::InvalidValue {
                label: "graph_max_nodes_exceeded",
            });
        }
        let expected = self.expected_len(node_count)?;
        if len != expected {
            return Err(TensorError::DataLength { expected, got: len });
        }
        Ok(expected)
    }

    fn guard_degree(&self, degree: usize) -> PureResult<()> {
        if degree > self.max_degree {
            return Err(TensorError::InvalidValue {
                label: "graph_degree_exceeded",
            });
        }
        Ok(())
    }

    fn guard_edge_budget(&self, edge_count: usize) -> PureResult<usize> {
        if edge_count <= self.max_edges {
            return Ok(0);
        }
        let slack = ((self.max_edges as f32) * self.permeability).ceil() as usize;
        let overflow = edge_count - self.max_edges;
        if overflow > slack {
            return Err(TensorError::TensorVolumeExceeded {
                volume: edge_count,
                max_volume: self.max_edges,
            });
        }
        Ok(overflow)
    }

    /// Overrides the permeability used when smoothing adjacency saturation and edge budgets.
    pub fn with_permeability(mut self, permeability: f32) -> PureResult<Self> {
        validate_permeability("graph_permeability", permeability)?;
        self.permeability = permeability;
        Ok(self)
    }
}

/// Structured report emitted by [`MultiModalToposGuard::guard_graph_adjacency`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphGuardReport {
    /// Number of undirected edges whose absolute weight exceeded the activation threshold.
    pub edge_count: usize,
    /// Maximum node degree observed while guarding the adjacency matrix.
    pub max_degree: usize,
    /// Number of asymmetric edge pairs detected beyond the configured tolerance.
    pub symmetry_violations: usize,
    /// Number of entries that were saturated into the admissible range.
    pub saturated_entries: usize,
    /// Number of edges exceeding the configured budget but tolerated by permeability.
    pub edge_overflow: usize,
}

/// Reward boundary used to detect runaway reinforcement learning signals.
#[derive(Clone, Copy, Debug)]
pub struct RewardBoundary {
    lower: f32,
    upper: f32,
    hysteresis: f32,
}

impl RewardBoundary {
    /// Creates a new reward boundary. `lower` must be below `upper` and hysteresis must be
    /// non-negative.
    pub fn new(lower: f32, upper: f32, hysteresis: f32) -> PureResult<Self> {
        if !lower.is_finite() || !upper.is_finite() || !hysteresis.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "reward_boundary",
                value: f32::NAN,
            });
        }
        if lower >= upper {
            return Err(TensorError::InvalidValue {
                label: "reward_boundary_window",
            });
        }
        if hysteresis < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "reward_boundary_hysteresis",
            });
        }
        Ok(Self {
            lower,
            upper,
            hysteresis,
        })
    }

    /// Returns the lower reward bound.
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Returns the upper reward bound.
    pub fn upper(&self) -> f32 {
        self.upper
    }

    fn clamp(&self, value: f32) -> f32 {
        value.clamp(self.lower, self.upper)
    }

    fn breached_lower(&self, value: f32) -> bool {
        value < self.lower - self.hysteresis
    }

    fn breached_upper(&self, value: f32) -> bool {
        value > self.upper + self.hysteresis
    }
}

/// Signal emitted when reward traces cross configured boundaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardBoundarySignal {
    /// Index of the first sample that breached the lower boundary, if any.
    pub lower_breach_index: Option<usize>,
    /// Index of the first sample that breached the upper boundary, if any.
    pub upper_breach_index: Option<usize>,
    /// Number of reward samples that were clamped back into the admissible window.
    pub clamped: usize,
    /// Minimum reward observed before clamping.
    pub min_observed: f32,
    /// Maximum reward observed before clamping.
    pub max_observed: f32,
}

impl Default for RewardBoundarySignal {
    fn default() -> Self {
        Self {
            lower_breach_index: None,
            upper_breach_index: None,
            clamped: 0,
            min_observed: f32::INFINITY,
            max_observed: f32::NEG_INFINITY,
        }
    }
}

impl RewardBoundarySignal {
    /// Finalises the observed range by normalising infinities for empty traces.
    fn finalise(mut self) -> Self {
        if !self.min_observed.is_finite() {
            self.min_observed = 0.0;
        }
        if !self.max_observed.is_finite() {
            self.max_observed = 0.0;
        }
        self
    }
}

/// Multi-modal guard that specialises an [`OpenCartesianTopos`] for audio, vision, text, graph,
/// and reinforcement-learning reward traces.
#[derive(Clone, Copy, Debug)]
pub struct MultiModalToposGuard<'a> {
    topos: &'a OpenCartesianTopos,
    text: ModalityProfile,
    audio: ModalityProfile,
    vision: ModalityProfile,
    graph: GraphGuardProfile,
    reward: RewardBoundary,
}

impl<'a> MultiModalToposGuard<'a> {
    /// Builds a multi-modal guard from explicit modality profiles and a reward boundary.
    pub fn from_profiles(
        topos: &'a OpenCartesianTopos,
        text: ModalityProfile,
        audio: ModalityProfile,
        vision: ModalityProfile,
        graph: GraphGuardProfile,
        reward: RewardBoundary,
    ) -> PureResult<Self> {
        graph.validate_for_topos(topos)?;
        text.validate_for_topos(topos)?;
        audio.validate_for_topos(topos)?;
        vision.validate_for_topos(topos)?;
        Ok(Self {
            topos,
            text,
            audio,
            vision,
            graph,
            reward,
        })
    }

    /// Builds a new multi-modal guard anchored to the provided topos with conservative defaults.
    pub fn new(topos: &'a OpenCartesianTopos) -> PureResult<Self> {
        let text = ModalityProfile::new(topos.max_volume().min(16384).max(1), None)?
            .with_permeability(0.18)?;
        let audio = ModalityProfile::new(
            topos.max_volume().min(65536).max(1),
            Some(topos.saturation()),
        )?
        .with_permeability(0.1)?;
        let vision = ModalityProfile::new(topos.max_volume().min(262144).max(1), None)?
            .with_permeability(0.15)?;
        let graph = GraphGuardProfile::new(
            cmp::min(2048, topos.max_volume()),
            cmp::min(topos.max_volume(), 131_072),
            1024,
            1e-3,
            1e-4,
            None,
        )?
        .with_permeability(0.1)?;
        let reward = RewardBoundary::new(-topos.saturation(), topos.saturation(), 0.05)?;
        Self::from_profiles(topos, text, audio, vision, graph, reward)
    }

    /// Returns the underlying topos guard.
    pub fn topos(&self) -> &'a OpenCartesianTopos {
        self.topos
    }

    /// Returns a rewrite monad anchored to the multi-modal guard.
    pub fn monad(&self) -> RewriteMonad<'a> {
        RewriteMonad::new(self.topos)
    }

    /// Creates a multi-modal atlas that shares this guard's modality envelopes.
    pub fn atlas(&self) -> MultiModalAtlas<'a> {
        MultiModalAtlas::new(*self)
    }

    /// Cultivates a tensor biome that preserves the guard's modality profiles.
    pub fn cultivate_biome(&self) -> MultiModalBiome {
        MultiModalBiome::new(*self)
    }

    /// Overrides the text modality profile.
    pub fn with_text_profile(mut self, profile: ModalityProfile) -> PureResult<Self> {
        profile.validate_for_topos(self.topos)?;
        self.text = profile;
        Ok(self)
    }

    /// Overrides the audio modality profile.
    pub fn with_audio_profile(mut self, profile: ModalityProfile) -> PureResult<Self> {
        profile.validate_for_topos(self.topos)?;
        self.audio = profile;
        Ok(self)
    }

    /// Overrides the vision modality profile.
    pub fn with_vision_profile(mut self, profile: ModalityProfile) -> PureResult<Self> {
        profile.validate_for_topos(self.topos)?;
        self.vision = profile;
        Ok(self)
    }

    /// Overrides the graph guard profile.
    pub fn with_graph_profile(mut self, profile: GraphGuardProfile) -> PureResult<Self> {
        profile.validate_for_topos(self.topos)?;
        self.graph = profile;
        Ok(self)
    }

    /// Overrides the reward boundary detector.
    pub fn with_reward_boundary(mut self, boundary: RewardBoundary) -> PureResult<Self> {
        self.reward = boundary;
        Ok(self)
    }

    /// Guards language or text tensors by applying modality-specific saturation before
    /// delegating to the underlying topos guard.
    pub fn guard_text_tensor(&self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        self.text.rewrite_tensor(self.topos, label, tensor)
    }

    /// Guards audio waveforms represented as contiguous slices.
    pub fn guard_audio_waveform(
        &self,
        label: &'static str,
        waveform: &mut [f32],
    ) -> PureResult<()> {
        self.audio.guard_volume(label, waveform.len())?;
        self.audio.rewrite_slice(self.topos, label, waveform)
    }

    /// Guards image or vision tensors by applying the configured vision profile.
    pub fn guard_vision_tensor(&self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        self.vision.rewrite_tensor(self.topos, label, tensor)
    }

    /// Guards graph adjacency matrices and reports symmetry and activation statistics.
    pub fn guard_graph_adjacency(
        &self,
        adjacency: &mut [f32],
        node_count: usize,
    ) -> PureResult<GraphGuardReport> {
        let expected = self.graph.guard_shape(node_count, adjacency.len())?;
        if let Some(budget) = self.graph.max_edges.checked_mul(2) {
            let soft_budget = (budget as f32 * (1.0 + self.graph.permeability()))
                .ceil()
                .max(budget as f32) as usize;
            if expected > soft_budget {
                return Err(TensorError::InvalidValue {
                    label: "graph_edge_budget",
                });
            }
        }
        let mut edge_count = 0usize;
        let mut max_degree = 0usize;
        let mut symmetry_violations = 0usize;
        let mut saturated_entries = 0usize;
        for i in 0..node_count {
            let mut degree = 0usize;
            for j in 0..node_count {
                let idx = i * node_count + j;
                let clamped = self.graph.saturate_value(self.topos, adjacency[idx]);
                if clamped != adjacency[idx] {
                    adjacency[idx] = clamped;
                    saturated_entries += 1;
                }
                if clamped.abs() >= self.graph.activation_threshold {
                    degree += 1;
                    if i < j {
                        edge_count += 1;
                    }
                }
                if i < j {
                    let sym_idx = j * node_count + i;
                    let sym = adjacency[sym_idx];
                    if (clamped - sym).abs() > self.graph.symmetry_tolerance {
                        symmetry_violations += 1;
                    }
                }
            }
            self.graph.guard_degree(degree)?;
            max_degree = cmp::max(max_degree, degree);
        }
        if edge_count > self.graph.max_edges {
            return Err(TensorError::TensorVolumeExceeded {
                label: "graph_edges",
                volume: edge_count,
                max_volume: self.graph.max_edges,
            });
        }
        self.topos.guard_slice("graph_adjacency", adjacency)?;
        Ok(GraphGuardReport {
            edge_count,
            max_degree,
            symmetry_violations,
            saturated_entries,
            edge_overflow: overflow,
        })
    }

    /// Guards reinforcement learning reward traces and detects boundary breaches.
    pub fn guard_reward_trace(&self, rewards: &mut [f32]) -> PureResult<RewardBoundarySignal> {
        if rewards.is_empty() {
            return Err(TensorError::EmptyInput("reward_trace"));
        }
        let mut signal = RewardBoundarySignal::default();
        for (idx, reward) in rewards.iter_mut().enumerate() {
            if !reward.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "reward_trace",
                    value: *reward,
                });
            }
            signal.min_observed = signal.min_observed.min(*reward);
            signal.max_observed = signal.max_observed.max(*reward);
            let saturated = self.topos.saturate(*reward);
            if self.reward.breached_lower(saturated) && signal.lower_breach_index.is_none() {
                signal.lower_breach_index = Some(idx);
            }
            if self.reward.breached_upper(saturated) && signal.upper_breach_index.is_none() {
                signal.upper_breach_index = Some(idx);
            }
            let clamped = self.reward.clamp(saturated);
            if (clamped - *reward).abs() > f32::EPSILON {
                signal.clamped += 1;
            }
            *reward = clamped;
        }
        self.topos.guard_slice("reward_trace", rewards)?;
        Ok(signal.finalise())
    }
}

/// Atlas that keeps track of modality-aware traversals across an open-cartesian topos.
#[derive(Clone, Debug)]
pub struct MultiModalAtlas<'a> {
    guard: MultiModalToposGuard<'a>,
    atlas: ToposAtlas<'a>,
}

impl<'a> MultiModalAtlas<'a> {
    pub(crate) fn from_parts(guard: MultiModalToposGuard<'a>, atlas: ToposAtlas<'a>) -> Self {
        Self { guard, atlas }
    }

    /// Creates a fresh multi-modal atlas from a guard.
    pub fn new(guard: MultiModalToposGuard<'a>) -> Self {
        Self::from_parts(guard, ToposAtlas::new(guard.topos))
    }

    /// Returns the underlying topos.
    pub fn topos(&self) -> &'a OpenCartesianTopos {
        self.guard.topos
    }

    /// Returns the rewrite monad anchored to this atlas.
    pub fn monad(&self) -> RewriteMonad<'a> {
        self.atlas.monad()
    }

    /// Returns the visited depth recorded by the underlying atlas.
    pub fn depth(&self) -> usize {
        self.atlas.depth()
    }

    /// Returns the accumulated tensor volume rewritten through this atlas.
    pub fn visited_volume(&self) -> usize {
        self.atlas.visited_volume()
    }

    /// Remaining admissible volume before the guard saturates.
    pub fn remaining_volume(&self) -> usize {
        self.atlas.remaining_volume()
    }

    /// Borrows the inner atlas.
    pub fn inner(&self) -> &ToposAtlas<'a> {
        &self.atlas
    }

    /// Borrows the inner atlas mutably.
    pub fn inner_mut(&mut self) -> &mut ToposAtlas<'a> {
        &mut self.atlas
    }

    /// Consumes the multi-modal wrapper and returns the underlying atlas.
    pub fn into_inner(self) -> ToposAtlas<'a> {
        self.atlas
    }

    /// Guards language tensors while tracking atlas traversal.
    pub fn guard_text_tensor(
        &mut self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        self.guard
            .text
            .rewrite_tensor(self.guard.topos, label, tensor)?;
        self.atlas.guard_tensor(label, tensor)
    }

    /// Lifts a text tensor through the atlas after applying modality saturation.
    pub fn lift_text_tensor(&mut self, label: &'static str, tensor: Tensor) -> PureResult<Tensor> {
        let mut tensor = tensor;
        self.guard
            .text
            .rewrite_tensor(self.guard.topos, label, &mut tensor)?;
        self.atlas.lift_tensor(label, tensor)
    }

    /// Guards audio waveforms using the atlas.
    pub fn guard_audio_waveform(
        &mut self,
        label: &'static str,
        waveform: &mut [f32],
    ) -> PureResult<()> {
        self.guard.audio.guard_volume(label, waveform.len())?;
        self.guard
            .audio
            .rewrite_slice(self.guard.topos, label, waveform)?;
        self.atlas.guard_slice(label, waveform)
    }

    /// Guards vision tensors while accounting for atlas traversal volume.
    pub fn guard_vision_tensor(
        &mut self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        self.guard
            .vision
            .rewrite_tensor(self.guard.topos, label, tensor)?;
        self.atlas.guard_tensor(label, tensor)
    }

    /// Lifts a vision tensor through the atlas after modality saturation.
    pub fn lift_vision_tensor(
        &mut self,
        label: &'static str,
        tensor: Tensor,
    ) -> PureResult<Tensor> {
        let mut tensor = tensor;
        self.guard
            .vision
            .rewrite_tensor(self.guard.topos, label, &mut tensor)?;
        self.atlas.lift_tensor(label, tensor)
    }

    /// Guards graph adjacency matrices and reuses the atlas guard for the resulting slice.
    pub fn guard_graph_adjacency(
        &mut self,
        adjacency: &mut [f32],
        node_count: usize,
    ) -> PureResult<GraphGuardReport> {
        let report = self.guard.guard_graph_adjacency(adjacency, node_count)?;
        self.atlas.guard_slice("graph_adjacency", adjacency)?;
        Ok(report)
    }

    /// Guards reinforcement-learning reward traces through the atlas.
    pub fn guard_reward_trace(&mut self, rewards: &mut [f32]) -> PureResult<RewardBoundarySignal> {
        let signal = self.guard.guard_reward_trace(rewards)?;
        self.atlas.guard_slice("reward_trace", rewards)?;
        Ok(signal)
    }

    /// Guards a fractal patch by delegating to the underlying atlas.
    pub fn guard_fractal_patch(
        &mut self,
        label: &'static str,
        patch: &FractalPatch,
    ) -> PureResult<()> {
        self.atlas.guard_fractal_patch(label, patch)
    }
}

/// Tensor biome that preserves modality envelopes while absorbing shoots.
#[derive(Clone, Debug)]
pub struct MultiModalBiome {
    biome: TensorBiome,
    text: ModalityProfile,
    audio: ModalityProfile,
    vision: ModalityProfile,
    graph: GraphGuardProfile,
    reward: RewardBoundary,
}

impl MultiModalBiome {
    /// Cultivates a biome from the provided guard.
    pub fn new(guard: MultiModalToposGuard<'_>) -> Self {
        Self {
            biome: TensorBiome::new(guard.topos.clone()),
            text: guard.text,
            audio: guard.audio,
            vision: guard.vision,
            graph: guard.graph,
            reward: guard.reward,
        }
    }

    fn guard_ref(&self) -> MultiModalToposGuard<'_> {
        MultiModalToposGuard {
            topos: self.biome.topos(),
            text: self.text,
            audio: self.audio,
            vision: self.vision,
            graph: self.graph,
            reward: self.reward,
        }
    }

    fn absorb_with_profile(
        &mut self,
        label: &'static str,
        mut tensor: Tensor,
        profile: ModalityProfile,
        weight: f32,
    ) -> PureResult<()> {
        let guard = self.guard_ref();
        profile.rewrite_tensor(guard.topos, label, &mut tensor)?;
        self.biome.absorb_weighted(label, tensor, weight)
    }

    /// Returns the underlying open-cartesian topos.
    pub fn topos(&self) -> &OpenCartesianTopos {
        self.biome.topos()
    }

    /// Returns a rewrite monad anchored to the biome's guard.
    pub fn monad(&self) -> RewriteMonad<'_> {
        self.biome.monad()
    }

    /// Creates a multi-modal atlas sharing the biome's profiles.
    pub fn atlas(&self) -> MultiModalAtlas<'_> {
        MultiModalAtlas::from_parts(self.guard_ref(), self.biome.atlas())
    }

    /// Absorbs a text tensor into the biome.
    pub fn absorb_text(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_text_weighted(label, tensor, 1.0)
    }

    /// Absorbs a text tensor with an explicit weight.
    pub fn absorb_text_weighted(
        &mut self,
        label: &'static str,
        tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        self.absorb_with_profile(label, tensor, self.text, weight)
    }

    /// Absorbs a vision tensor into the biome.
    pub fn absorb_vision(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_vision_weighted(label, tensor, 1.0)
    }

    /// Absorbs a vision tensor with an explicit weight.
    pub fn absorb_vision_weighted(
        &mut self,
        label: &'static str,
        tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        self.absorb_with_profile(label, tensor, self.vision, weight)
    }

    /// Absorbs an audio tensor into the biome.
    pub fn absorb_audio(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_audio_weighted(label, tensor, 1.0)
    }

    /// Absorbs an audio tensor with an explicit weight.
    pub fn absorb_audio_weighted(
        &mut self,
        label: &'static str,
        tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        self.absorb_with_profile(label, tensor, self.audio, weight)
    }

    /// Delegates graph guarding to the preserved profiles.
    pub fn guard_graph_adjacency(
        &self,
        adjacency: &mut [f32],
        node_count: usize,
    ) -> PureResult<GraphGuardReport> {
        self.guard_ref()
            .guard_graph_adjacency(adjacency, node_count)
    }

    /// Guards reward traces through the biome's guard.
    pub fn guard_reward_trace(&self, rewards: &mut [f32]) -> PureResult<RewardBoundarySignal> {
        self.guard_ref().guard_reward_trace(rewards)
    }

    /// Borrows the inner tensor biome.
    pub fn inner(&self) -> &TensorBiome {
        &self.biome
    }

    /// Borrows the inner tensor biome mutably.
    pub fn inner_mut(&mut self) -> &mut TensorBiome {
        &mut self.biome
    }

    /// Consumes the wrapper and returns the inner tensor biome.
    pub fn into_inner(self) -> TensorBiome {
        self.biome
    }

    /// Number of shoots currently stored in the biome.
    pub fn len(&self) -> usize {
        self.biome.len()
    }

    /// Whether the biome has absorbed any shoots.
    pub fn is_empty(&self) -> bool {
        self.biome.is_empty()
    }

    /// Total accumulated weight across all shoots.
    pub fn total_weight(&self) -> f32 {
        self.biome.total_weight()
    }

    /// Returns the weights assigned to each shoot.
    pub fn weights(&self) -> &[f32] {
        self.biome.weights()
    }

    /// Harvests the canopy averaged across all shoots.
    pub fn canopy(&self) -> PureResult<Tensor> {
        self.biome.canopy()
    }

    /// Clears all shoots from the biome.
    pub fn clear(&mut self) {
        self.biome.clear();
    }

    /// Returns a view over the stored shoots.
    pub fn shoots(&self) -> &[Tensor] {
        self.biome.shoots()
    }

    /// Stacks all shoots into a dense tensor.
    pub fn stack(&self) -> PureResult<Tensor> {
        self.biome.stack()
    }

    /// Delegates fractal patch absorption to the underlying biome.
    pub fn absorb_fractal_patch(&mut self, patch: &FractalPatch) -> PureResult<()> {
        self.biome.absorb_fractal_patch(patch)
    }

    /// Renormalises the shoot weights through the preserved guard.
    pub fn renormalise_weights(&mut self) -> PureResult<()> {
        self.biome.renormalise_weights()
    }
}

/// Tracks shared topos state across tensor rewrites, fractal traversals, and measure updates.
#[derive(Debug, Clone)]
pub struct ToposAtlas<'a> {
    topos: &'a OpenCartesianTopos,
    monad: RewriteMonad<'a>,
    visited_volume: usize,
    depth: usize,
}

impl<'a> ToposAtlas<'a> {
    /// Creates a new atlas anchored to a shared open-cartesian topos.
    pub fn new(topos: &'a OpenCartesianTopos) -> Self {
        Self {
            topos,
            monad: RewriteMonad::new(topos),
            visited_volume: 0,
            depth: 0,
        }
    }

    /// Returns the underlying guard.
    pub fn topos(&self) -> &'a OpenCartesianTopos {
        self.topos
    }

    /// Returns the rewrite monad anchored to this atlas.
    pub fn monad(&self) -> RewriteMonad<'a> {
        self.monad
    }

    fn observe_volume(&mut self, label: &'static str, volume: usize) -> PureResult<()> {
        let projected = self.visited_volume.saturating_add(volume);
        if projected > self.topos.max_volume() {
            return Err(TensorError::TensorVolumeExceeded {
                label,
                volume: projected,
                max_volume: self.topos.max_volume(),
            });
        }
        self.visited_volume = projected;
        Ok(())
    }

    /// Guards a tensor and records the total traversed volume.
    pub fn guard_tensor(&mut self, label: &'static str, tensor: &Tensor) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        self.observe_volume(label, rows.saturating_mul(cols))?;
        self.monad.guard_tensor(label, tensor)
    }

    /// Rewrites a tensor in-place while tracking the traversed volume.
    pub fn guard_tensor_mut(&mut self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        self.observe_volume(label, rows.saturating_mul(cols))?;
        self.monad.rewrite_tensor(label, tensor)
    }

    /// Lifts an owned tensor into the atlas, returning the rewritten value.
    pub fn lift_tensor(&mut self, label: &'static str, tensor: Tensor) -> PureResult<Tensor> {
        let (rows, cols) = tensor.shape();
        self.observe_volume(label, rows.saturating_mul(cols))?;
        self.monad.lift_tensor(label, tensor)
    }

    /// Guards a slice without affecting the tracked volume.
    pub fn guard_slice(&self, label: &'static str, slice: &[f32]) -> PureResult<()> {
        self.monad.guard_slice(label, slice)
    }

    /// Rewrites a mutable slice while keeping volume untouched.
    pub fn guard_slice_mut(&self, label: &'static str, slice: &mut [f32]) -> PureResult<()> {
        self.monad.rewrite_slice(label, slice)
    }

    /// Normalises a probability slice within the atlas.
    pub fn guard_probability_slice(
        &self,
        label: &'static str,
        slice: &mut [f32],
    ) -> PureResult<()> {
        self.monad.guard_probability_slice(label, slice)
    }

    /// Normalises a probability tensor within the atlas.
    pub fn guard_probability_tensor(
        &self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        self.monad.guard_probability_tensor(label, tensor)
    }

    /// Registers the observed depth and guards the underlying relation tensor.
    pub fn guard_fractal_patch(
        &mut self,
        label: &'static str,
        patch: &FractalPatch,
    ) -> PureResult<()> {
        self.observe_depth(patch.depth() as usize)?;
        self.guard_tensor(label, patch.relation())
    }

    /// Updates the maximum visited depth.
    pub fn observe_depth(&mut self, depth: usize) -> PureResult<()> {
        self.topos.ensure_loop_free(depth)?;
        if depth > self.depth {
            self.depth = depth;
        }
        Ok(())
    }

    /// Returns the deepest stratum observed by the atlas.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the accumulated volume guarded by the atlas.
    pub fn visited_volume(&self) -> usize {
        self.visited_volume
    }

    /// Remaining admissible tensor volume before the atlas saturates.
    pub fn remaining_volume(&self) -> usize {
        self.topos.max_volume().saturating_sub(self.visited_volume)
    }
}

/// Minimal monadic helper that rewrites values through the enclosing topos.
#[derive(Clone, Copy, Debug)]
pub struct RewriteMonad<'a> {
    topos: &'a OpenCartesianTopos,
}

impl<'a> RewriteMonad<'a> {
    /// Wraps a guard for repeated rewrites.
    pub fn new(topos: &'a OpenCartesianTopos) -> Self {
        Self { topos }
    }

    /// Returns the underlying topos guard.
    pub fn topos(&self) -> &'a OpenCartesianTopos {
        self.topos
    }

    /// Returns the Lawvere–Tierney guard of the enclosed topos.
    pub fn lawvere_guard(&self) -> &LawvereTierneyGuard {
        self.topos.site().guard()
    }

    /// Rewrites a scalar by saturating it into the open-cartesian window.
    pub fn rewrite_scalar(&self, value: f32) -> f32 {
        self.topos.saturate(value)
    }

    /// Rewrites a mutable slice and validates the result.
    pub fn rewrite_slice(&self, label: &'static str, slice: &mut [f32]) -> PureResult<()> {
        self.topos.saturate_slice(slice);
        self.topos.guard_slice(label, slice)
    }

    /// Guards a read-only slice without saturation.
    pub fn guard_slice(&self, label: &'static str, slice: &[f32]) -> PureResult<()> {
        self.topos.guard_slice(label, slice)
    }

    /// Rewrites a tensor and re-validates its envelope.
    pub fn rewrite_tensor(&self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        self.topos.saturate_slice(tensor.data_mut());
        self.topos.guard_tensor(label, tensor)
    }

    /// Guards an immutable tensor reference.
    pub fn guard_tensor(&self, label: &'static str, tensor: &Tensor) -> PureResult<()> {
        self.topos.guard_tensor(label, tensor)
    }

    /// Ensures a κ-box is admissible for the enclosed topos.
    pub fn guard_zbox(&self, zbox: &ZBox) -> PureResult<()> {
        self.topos.guard_zbox(zbox)
    }

    /// Ensures a κ-box cover remains mass-preserving.
    pub fn guard_cover(&self, cover: &[ZBox]) -> PureResult<()> {
        self.topos.guard_cover(cover)
    }

    /// Normalises a probability slice through the topos window.
    pub fn guard_probability_slice(
        &self,
        label: &'static str,
        slice: &mut [f32],
    ) -> PureResult<()> {
        self.topos.guard_probability_slice(label, slice)
    }

    /// Normalises a probability tensor through the topos window.
    pub fn guard_probability_tensor(
        &self,
        label: &'static str,
        tensor: &mut Tensor,
    ) -> PureResult<()> {
        self.topos.guard_probability_tensor(label, tensor)
    }

    /// Lifts an owned tensor into the monadic context and returns the guarded value.
    pub fn lift_tensor(&self, label: &'static str, mut tensor: Tensor) -> PureResult<Tensor> {
        self.rewrite_tensor(label, &mut tensor)?;
        Ok(tensor)
    }

    /// Applies a closure to a tensor before rewriting it through the guard.
    pub fn bind_tensor<F>(
        &self,
        label: &'static str,
        mut tensor: Tensor,
        f: F,
    ) -> PureResult<Tensor>
    where
        F: FnOnce(&mut Tensor) -> PureResult<()>,
    {
        f(&mut tensor)?;
        self.rewrite_tensor(label, &mut tensor)?;
        Ok(tensor)
    }

    /// Applies a closure to a mutable slice before rewriting it through the guard.
    pub fn bind_slice<F>(&self, label: &'static str, slice: &mut [f32], f: F) -> PureResult<()>
    where
        F: FnOnce(&mut [f32]) -> PureResult<()>,
    {
        f(slice)?;
        self.rewrite_slice(label, slice)
    }

    /// Cultivates a fresh tensor biome anchored to this monad's guard.
    pub fn cultivate_biome(&self) -> TensorBiome {
        TensorBiome::new(self.topos.clone())
    }

    /// Absorbs an owned tensor into an existing biome using the monadic guard.
    pub fn absorb_into_biome(
        &self,
        biome: &mut TensorBiome,
        label: &'static str,
        mut tensor: Tensor,
    ) -> PureResult<()> {
        self.rewrite_tensor(label, &mut tensor)?;
        biome.push_shoot(tensor, 1.0)
    }

    /// Absorbs a weighted tensor into an existing biome through the monad guard.
    pub fn absorb_weighted_into_biome(
        &self,
        biome: &mut TensorBiome,
        label: &'static str,
        mut tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        self.rewrite_tensor(label, &mut tensor)?;
        biome.push_shoot(tensor, weight)
    }
}

/// Organises tensors rewritten through an open topos into a living "biome".
///
/// The biome behaves like a minimal monad: every tensor absorbed into it is
/// rewritten through the enclosing `OpenCartesianTopos`, saturated into the
/// safety window, and retained as a new shoot.  When the caller is ready to
/// harvest the emergent meaning, the biome collapses all shoots into a guarded
/// canopy tensor that stays within the same topos envelope.
#[derive(Clone, Debug)]
pub struct TensorBiome {
    topos: OpenCartesianTopos,
    shoots: Vec<Tensor>,
    weights: Vec<f32>,
    total_weight: f32,
    shape: Option<(usize, usize)>,
}

impl TensorBiome {
    /// Wraps a biome around an open-cartesian topos.
    pub fn new(topos: OpenCartesianTopos) -> Self {
        Self {
            topos,
            shoots: Vec::new(),
            weights: Vec::new(),
            total_weight: 0.0,
            shape: None,
        }
    }

    /// Returns the guard topos.
    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    /// Returns a rewrite monad anchored to the biome's guard.
    pub fn monad(&self) -> RewriteMonad<'_> {
        RewriteMonad::new(&self.topos)
    }

    /// Returns an atlas anchored to the biome's guard.
    pub fn atlas(&self) -> ToposAtlas<'_> {
        ToposAtlas::new(&self.topos)
    }

    /// Number of shoots currently living inside the biome.
    pub fn len(&self) -> usize {
        self.shoots.len()
    }

    /// Whether the biome is empty.
    pub fn is_empty(&self) -> bool {
        self.shoots.is_empty()
    }

    /// Total accumulated weight across all shoots.
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// Returns the individual weights that were assigned to each shoot.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Absorbs a tensor into the biome, rewriting it through the guard topos.
    pub fn absorb(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_weighted(label, tensor, 1.0)
    }

    /// Absorbs a tensor with an explicit weight that skews the canopy average.
    pub fn absorb_weighted(
        &mut self,
        label: &'static str,
        mut tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(TensorError::NonPositiveWeight { weight });
        }
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor(label, &mut tensor)?;
        self.push_shoot(tensor, weight)
    }

    /// Absorbs a tensor produced by a monadic builder.
    pub fn absorb_with<F>(&mut self, label: &'static str, build: F) -> PureResult<()>
    where
        F: FnOnce(RewriteMonad<'_>) -> PureResult<Tensor>,
    {
        let tensor = build(self.monad())?;
        self.absorb(label, tensor)
    }

    /// Absorbs a weighted tensor produced by a monadic builder.
    pub fn absorb_weighted_with<F>(
        &mut self,
        label: &'static str,
        weight: f32,
        build: F,
    ) -> PureResult<()>
    where
        F: FnOnce(RewriteMonad<'_>) -> PureResult<Tensor>,
    {
        let tensor = build(self.monad())?;
        self.absorb_weighted(label, tensor, weight)
    }

    fn push_shoot(&mut self, tensor: Tensor, weight: f32) -> PureResult<()> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(TensorError::NonPositiveWeight { weight });
        }
        let shape = tensor.shape();
        if let Some(expected) = self.shape {
            if expected != shape {
                return Err(TensorError::ShapeMismatch {
                    left: expected,
                    right: shape,
                });
            }
        } else {
            self.shape = Some(shape);
        }
        self.total_weight += weight;
        self.shoots.push(tensor);
        self.weights.push(weight);
        Ok(())
    }

    /// Applies a guarded rewrite to every shoot living inside the biome.
    pub fn bind_shoots<F>(&mut self, label: &'static str, mut f: F) -> PureResult<()>
    where
        F: FnMut(RewriteMonad<'_>, &mut Tensor) -> PureResult<()>,
    {
        let topos = self.topos.clone();
        let monad = RewriteMonad::new(&topos);
        for shoot in &mut self.shoots {
            f(monad, shoot)?;
            monad.rewrite_tensor(label, shoot)?;
        }
        Ok(())
    }

    /// Builds a new biome by mapping the current shoots through a monadic builder.
    pub fn map_shoots<F>(&self, label: &'static str, mut f: F) -> PureResult<TensorBiome>
    where
        F: FnMut(RewriteMonad<'_>, &Tensor) -> PureResult<Tensor>,
    {
        let topos = self.topos.clone();
        let monad = RewriteMonad::new(&topos);
        let mut biome = TensorBiome::new(topos.clone());
        for (shoot, &weight) in self.shoots.iter().zip(self.weights.iter()) {
            let mut mapped = f(monad, shoot)?;
            monad.rewrite_tensor(label, &mut mapped)?;
            biome.push_shoot(mapped, weight)?;
        }
        Ok(biome)
    }

    /// Renormalises the shoot weights through the Lawvere–Tierney guard.
    pub fn renormalise_weights(&mut self) -> PureResult<()> {
        if self.weights.is_empty() {
            return Err(TensorError::EmptyInput("tensor_biome_weights"));
        }
        let topos = self.topos.clone();
        let monad = RewriteMonad::new(&topos);
        monad.guard_probability_slice("tensor_biome_weights", &mut self.weights)?;
        self.total_weight = self.weights.iter().sum();
        Ok(())
    }

    /// Absorbs a fractal relation patch directly into the biome canopy.
    pub fn absorb_fractal_patch(&mut self, patch: &FractalPatch) -> PureResult<()> {
        let mut atlas = self.atlas();
        atlas.guard_fractal_patch("tensor_biome_fractal_patch", patch)?;
        let relation = atlas.lift_tensor(
            "tensor_biome_fractal_patch_relation",
            patch.relation().clone(),
        )?;
        self.absorb_weighted("tensor_biome_fractal_patch", relation, patch.weight())
    }

    /// Clears all shoots from the biome while preserving the topos.
    pub fn clear(&mut self) {
        self.shoots.clear();
        self.weights.clear();
        self.total_weight = 0.0;
        self.shape = None;
    }

    /// Harvests the biome by averaging all shoots into a guarded canopy tensor.
    pub fn canopy(&self) -> PureResult<Tensor> {
        let (rows, cols) = self.shape.ok_or(TensorError::EmptyInput("tensor_biome"))?;
        if self.is_empty() {
            return Err(TensorError::EmptyInput("tensor_biome"));
        }
        if self.total_weight <= 0.0 {
            return Err(TensorError::NonPositiveWeight {
                weight: self.total_weight,
            });
        }
        let mut acc = Tensor::zeros(rows, cols)?;
        for (shoot, &weight) in self.shoots.iter().zip(self.weights.iter()) {
            acc.add_scaled(shoot, weight)?;
        }
        let mut canopy = acc.scale(1.0 / self.total_weight)?;
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("tensor_biome_canopy", &mut canopy)?;
        Ok(canopy)
    }

    /// Returns a snapshot of the current shoots.
    pub fn shoots(&self) -> &[Tensor] {
        &self.shoots
    }

    /// Stacks all shoots along the row dimension, yielding a dense tensor.
    pub fn stack(&self) -> PureResult<Tensor> {
        let (rows, cols) = self.shape.ok_or(TensorError::EmptyInput("tensor_biome"))?;
        if self.is_empty() {
            return Err(TensorError::EmptyInput("tensor_biome"));
        }
        let mut data = Vec::with_capacity(self.shoots.len() * rows * cols);
        for shoot in &self.shoots {
            data.extend_from_slice(shoot.data());
        }
        Tensor::from_vec(self.shoots.len() * rows, cols, data)
    }
}

/// Deterministic conjugate gradient solver that respects the open-cartesian guard.
pub struct ConjugateGradientSolver<'a> {
    topos: &'a OpenCartesianTopos,
    tolerance: f32,
    max_iterations: usize,
}

impl<'a> ConjugateGradientSolver<'a> {
    /// Creates a solver with an explicit tolerance and iteration cap.
    pub fn new(
        topos: &'a OpenCartesianTopos,
        tolerance: f32,
        max_iterations: usize,
    ) -> PureResult<Self> {
        if tolerance <= 0.0 {
            return Err(TensorError::NonPositiveTolerance { tolerance });
        }
        if max_iterations == 0 {
            return Err(TensorError::EmptyInput("conjugate gradient max iterations"));
        }
        Ok(Self {
            topos,
            tolerance,
            max_iterations,
        })
    }

    /// Returns the solver tolerance.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Solves a linear system `Ax = b` using repeated matrix-vector products.
    ///
    /// The callback receives the candidate vector and must write `A * src` into
    /// `dst`. The solver enforces explicit tolerances so hyperbolic Jacobians do
    /// not diverge.
    pub fn solve<F>(&self, mut matvec: F, b: &[f32], x: &mut [f32]) -> PureResult<usize>
    where
        F: FnMut(&[f32], &mut [f32]),
    {
        if b.len() != x.len() {
            return Err(TensorError::DataLength {
                expected: b.len(),
                got: x.len(),
            });
        }
        if b.is_empty() {
            return Err(TensorError::EmptyInput("conjugate gradient rhs"));
        }
        self.topos.guard_slice("cg_rhs", b)?;
        self.topos.guard_slice("cg_initial", x)?;
        let mut r = vec![0.0f32; b.len()];
        let mut p = vec![0.0f32; b.len()];
        let mut ap = vec![0.0f32; b.len()];
        matvec(x, &mut ap);
        for ((r_i, p_i), (&b_i, &ap_i)) in
            r.iter_mut().zip(p.iter_mut()).zip(b.iter().zip(ap.iter()))
        {
            *r_i = b_i - ap_i;
            *p_i = *r_i;
        }
        let mut rsold = dot(&r, &r);
        let tol = self.tolerance.max(self.topos.tolerance());
        if rsold.sqrt() <= tol {
            return Ok(0);
        }
        for iter in 0..self.max_iterations {
            matvec(&p, &mut ap);
            let denom = dot(&p, &ap);
            if denom.abs() <= tol {
                return Err(TensorError::ConjugateGradientDiverged {
                    residual: rsold.sqrt(),
                    tolerance: tol,
                });
            }
            let alpha = rsold / denom;
            for (x_i, p_i) in x.iter_mut().zip(p.iter()) {
                *x_i = self.topos.saturate(*x_i + alpha * *p_i);
            }
            for (r_i, ap_i) in r.iter_mut().zip(ap.iter()) {
                *r_i -= alpha * *ap_i;
            }
            let rsnew = dot(&r, &r);
            if rsnew.sqrt() <= tol {
                self.topos.guard_slice("cg_solution", x)?;
                return Ok(iter + 1);
            }
            let beta = rsnew / rsold.max(tol * tol);
            for (p_i, r_i) in p.iter_mut().zip(r.iter()) {
                *p_i = self.topos.saturate(*r_i + beta * *p_i);
            }
            rsold = rsnew;
        }
        Err(TensorError::ConjugateGradientDiverged {
            residual: rsold.sqrt(),
            tolerance: tol,
        })
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure::fractal::FractalPatch;

    fn demo_topos() -> OpenCartesianTopos {
        OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 4096).unwrap()
    }

    #[test]
    fn topos_porosity_softens_extremes() {
        let rigid = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 4096)
            .unwrap()
            .with_porosity(0.0)
            .unwrap();
        let porous = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 4096)
            .unwrap()
            .with_porosity(0.9)
            .unwrap();
        let sample = rigid.saturation() * 4.0;
        let rigid_value = rigid.saturate(sample);
        let porous_value = porous.saturate(sample);
        assert!(porous_value.abs() < rigid_value.abs());
        assert!(porous_value.abs() > rigid.saturation() * 0.8);
    }

    #[test]
    fn reward_boundary_porosity_delays_breach() {
        let topos = demo_topos();
        let strict_boundary = RewardBoundary::with_porosity(-0.5, 0.5, 0.05, 0.0).unwrap();
        let porous_boundary = RewardBoundary::with_porosity(-0.5, 0.5, 0.05, 0.8).unwrap();
        let strict_guard = MultiModalToposGuard::new(&topos)
            .unwrap()
            .with_reward_boundary(strict_boundary)
            .unwrap();
        let porous_guard = MultiModalToposGuard::new(&topos)
            .unwrap()
            .with_reward_boundary(porous_boundary)
            .unwrap();
        let mut strict_trace = vec![0.56f32];
        let mut porous_trace = vec![0.56f32];
        let strict_signal = strict_guard.guard_reward_trace(&mut strict_trace).unwrap();
        let porous_signal = porous_guard.guard_reward_trace(&mut porous_trace).unwrap();
        assert!(strict_signal.upper_breach_index.is_some());
        assert!(porous_signal.upper_breach_index.is_none());
        assert!(strict_trace[0] <= 0.5 + 1e-6);
        assert!(porous_trace[0] < 0.5);
    }

    #[test]
    fn graph_profile_rejects_invalid_porosity() {
        let profile = GraphGuardProfile::new(8, 12, 4, 1e-3, 0.05, Some(1.0)).unwrap();
        let err = profile.with_porosity(1.5).unwrap_err();
        assert!(matches!(err, TensorError::PorosityOutOfRange { .. }));
    }

    #[test]
    fn zbox_cover_respects_mass() {
        let topos = demo_topos();
        let centers_a = vec![vec![0.0f32, 0.0]];
        let centers_b = vec![vec![0.25f32, -0.1]];
        let radii = vec![0.4f32];
        let base = ZBox::new(centers_a.clone(), radii.clone(), 1.0).unwrap();
        let volume = base.hyperbolic_volume(topos.curvature()).unwrap();
        let density = 0.5 / volume;
        let box_a = ZBox::new(centers_a.clone(), radii.clone(), density).unwrap();
        let box_b = ZBox::new(centers_b.clone(), radii.clone(), density).unwrap();
        topos.guard_cover(&[box_a.clone(), box_b.clone()]).unwrap();
        let heavy_density = 0.8 / volume;
        let heavy = ZBox::new(centers_b, radii, heavy_density).unwrap();
        let err = topos.guard_cover(&[box_a, heavy]).unwrap_err();
        assert!(matches!(err, TensorError::InvalidValue { .. }));
    }

    #[test]
    fn topos_rejects_non_finite_values() {
        let topos = demo_topos();
        let tensor = Tensor::from_vec(1, 2, vec![1.0, f32::INFINITY]).unwrap();
        let err = topos.guard_tensor("nonfinite", &tensor).unwrap_err();
        matches!(err, TensorError::NonFiniteValue { .. });
    }

    #[test]
    fn biome_absorbs_and_harvests() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        let big = topos.saturation() * 2.0;
        biome
            .absorb(
                "biome_shoot_a",
                Tensor::from_vec(1, 2, vec![big, 0.5]).unwrap(),
            )
            .unwrap();
        biome
            .absorb(
                "biome_shoot_b",
                Tensor::from_vec(1, 2, vec![-big, 1.0]).unwrap(),
            )
            .unwrap();
        let canopy = biome.canopy().unwrap();
        assert_eq!(canopy.shape(), (1, 2));
        let data = canopy.data();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.75).abs() < 1e-6);
        assert_eq!(biome.total_weight(), 2.0);
    }

    #[test]
    fn biome_detects_shape_mismatch() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        biome
            .absorb(
                "biome_shape_a",
                Tensor::from_vec(2, 1, vec![0.1, 0.2]).unwrap(),
            )
            .unwrap();
        let err = biome
            .absorb(
                "biome_shape_b",
                Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap(),
            )
            .unwrap_err();
        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }

    #[test]
    fn biome_weighted_canopy_respects_shoot_weights() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        biome
            .absorb_weighted(
                "weighted_a",
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                1.0,
            )
            .unwrap();
        biome
            .absorb_weighted(
                "weighted_b",
                Tensor::from_vec(1, 1, vec![3.0]).unwrap(),
                3.0,
            )
            .unwrap();
        let canopy = biome.canopy().unwrap();
        assert_eq!(canopy.data(), &[2.5]);
        assert_eq!(biome.weights(), &[1.0, 3.0]);
        assert!((biome.total_weight() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn biome_stack_concatenates_shoots() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        biome
            .absorb("stack_a", Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap())
            .unwrap();
        biome
            .absorb("stack_b", Tensor::from_vec(1, 2, vec![0.3, 0.4]).unwrap())
            .unwrap();
        let stacked = biome.stack().unwrap();
        assert_eq!(stacked.shape(), (2, 2));
        assert_eq!(stacked.data(), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn rewrite_monad_saturates_values() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let mut tensor = Tensor::from_vec(1, 2, vec![20.0, -20.0]).unwrap();
        monad.rewrite_tensor("rewrite", &mut tensor).unwrap();
        assert!(tensor.data().iter().all(|v| v.abs() <= topos.saturation()));
    }

    #[test]
    fn rewrite_monad_surfaces_lawvere_guard() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let guard = monad.lawvere_guard();
        assert!(guard.density_min() > 0.0);
        assert!(guard.density_max() > guard.density_min());
    }

    #[test]
    fn rewrite_monad_lift_and_bind_tensor() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let lifted = monad
            .lift_tensor(
                "lift",
                Tensor::from_vec(1, 2, vec![topos.saturation() * 4.0, 0.25]).unwrap(),
            )
            .unwrap();
        assert!(lifted
            .data()
            .iter()
            .all(|v| v.is_finite() && v.abs() <= topos.saturation()));

        let bound = monad
            .bind_tensor(
                "bind",
                Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap(),
                |tensor| {
                    let update = Tensor::from_vec(1, 2, vec![0.3, 0.4]).unwrap();
                    tensor.add_scaled(&update, 1.0)
                },
            )
            .unwrap();
        assert_eq!(bound.shape(), (1, 2));
        assert!(bound.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn topos_normalises_probability_slices() {
        let topos = demo_topos();
        let mut slice = vec![2.0, -1.0, 0.5];
        topos
            .guard_probability_slice("probability_guard", &mut slice)
            .unwrap();
        assert!(slice.iter().all(|v| *v >= 0.0));
        let sum: f32 = slice.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn atlas_tracks_volume_and_depth() {
        let topos = demo_topos();
        let mut atlas = ToposAtlas::new(&topos);
        let tensor = Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap();
        atlas.guard_tensor("atlas_tensor", &tensor).unwrap();
        assert_eq!(atlas.visited_volume(), 2);
        assert_eq!(atlas.remaining_volume(), topos.max_volume() - 2);
        let patch = FractalPatch::new(Tensor::from_vec(1, 2, vec![0.3, 0.4]).unwrap(), 1.0, 1.0, 1)
            .unwrap();
        atlas.guard_fractal_patch("atlas_patch", &patch).unwrap();
        assert_eq!(atlas.depth(), 1);
        assert_eq!(atlas.visited_volume(), 4);
    }

    #[test]
    fn atlas_lifts_tensor_through_monad() {
        let topos = demo_topos();
        let mut atlas = ToposAtlas::new(&topos);
        let lifted = atlas
            .lift_tensor(
                "atlas_lift",
                Tensor::from_vec(1, 2, vec![topos.saturation() * 5.0, 0.5]).unwrap(),
            )
            .unwrap();
        assert!(lifted.data().iter().all(|v| v.abs() <= topos.saturation()));
        assert_eq!(atlas.visited_volume(), 2);
    }

    #[test]
    fn biome_absorbs_fractal_patches() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        let patch =
            FractalPatch::new(Tensor::from_vec(1, 1, vec![2.0]).unwrap(), 2.0, 1.0, 0).unwrap();
        biome.absorb_fractal_patch(&patch).unwrap();
        assert_eq!(biome.len(), 1);
        let canopy = biome.canopy().unwrap();
        assert_eq!(canopy.data(), &[2.0]);
    }

    #[test]
    fn biome_absorb_with_monadic_builder() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        biome
            .absorb_with("monadic", |monad| {
                monad.lift_tensor(
                    "monadic_build",
                    Tensor::from_vec(1, 2, vec![topos.saturation() * 3.0, 0.5]).unwrap(),
                )
            })
            .unwrap();
        assert_eq!(biome.len(), 1);
        let canopy = biome.canopy().unwrap();
        assert_eq!(canopy.shape(), (1, 2));
    }

    #[test]
    fn biome_absorb_weighted_with_monadic_builder() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        biome
            .absorb_weighted_with("weighted_monadic", 2.0, |monad| {
                monad.bind_tensor(
                    "weighted_monadic_build",
                    Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                    |tensor| tensor.add_scaled(&Tensor::from_vec(1, 1, vec![1.0]).unwrap(), 1.0),
                )
            })
            .unwrap();
        let canopy = biome.canopy().unwrap();
        assert_eq!(canopy.data(), &[2.0]);
        assert!((biome.total_weight() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn biome_bind_shoots_rewrites_each_shoot() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        biome
            .absorb(
                "bind_shoot",
                Tensor::from_vec(1, 1, vec![topos.saturation() * 4.0]).unwrap(),
            )
            .unwrap();
        biome
            .bind_shoots("bind_shoots", |monad, shoot| {
                let update = monad.lift_tensor(
                    "bind_shoot_update",
                    Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
                )?;
                shoot.add_scaled(&update, 1.0)
            })
            .unwrap();
        let canopy = biome.canopy().unwrap();
        assert!(canopy.data()[0].abs() <= topos.saturation());
    }

    #[test]
    fn biome_map_shoots_reuses_weights() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        biome
            .absorb_weighted("map_a", Tensor::from_vec(1, 1, vec![1.0]).unwrap(), 2.0)
            .unwrap();
        biome
            .absorb_weighted("map_b", Tensor::from_vec(1, 1, vec![2.0]).unwrap(), 3.0)
            .unwrap();
        let base_canopy = biome.canopy().unwrap();
        let mapped = biome
            .map_shoots("map_transform", |monad, shoot| {
                monad.bind_tensor("map_transform_build", shoot.clone(), |tensor| {
                    tensor.add_scaled(&Tensor::from_vec(1, 1, vec![1.0]).unwrap(), 1.0)
                })
            })
            .unwrap();
        assert_eq!(mapped.len(), biome.len());
        assert!((mapped.total_weight() - biome.total_weight()).abs() < 1e-6);
        let mapped_canopy = mapped.canopy().unwrap();
        assert!((mapped_canopy.data()[0] - (base_canopy.data()[0] + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn biome_renormalises_weights_preserves_canopy() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        biome
            .absorb_weighted("renorm_a", Tensor::from_vec(1, 1, vec![1.0]).unwrap(), 2.0)
            .unwrap();
        biome
            .absorb_weighted("renorm_b", Tensor::from_vec(1, 1, vec![3.0]).unwrap(), 3.0)
            .unwrap();
        let canopy_before = biome.canopy().unwrap();
        biome.renormalise_weights().unwrap();
        let sum: f32 = biome.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((biome.total_weight() - 1.0).abs() < 1e-6);
        let canopy_after = biome.canopy().unwrap();
        assert!((canopy_before.data()[0] - canopy_after.data()[0]).abs() < 1e-6);
    }

    #[test]
    fn monad_cultivates_biome_and_absorbs_weighted() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let mut biome = monad.cultivate_biome();
        monad
            .absorb_weighted_into_biome(
                &mut biome,
                "monad_absorb",
                Tensor::from_vec(1, 1, vec![topos.saturation() * 6.0]).unwrap(),
                2.0,
            )
            .unwrap();
        assert_eq!(biome.len(), 1);
        assert!((biome.total_weight() - 2.0).abs() < 1e-6);
        let canopy = biome.canopy().unwrap();
        assert!(canopy.data()[0].abs() <= topos.saturation());
    }

    #[test]
    fn conjugate_gradient_converges_with_guard() {
        let topos = demo_topos();
        let solver = ConjugateGradientSolver::new(&topos, 1e-5, 32).unwrap();
        let matrix = [4.0f32, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0];
        let mut matvec = |src: &[f32], dst: &mut [f32]| {
            dst.fill(0.0);
            for row in 0..3 {
                for col in 0..3 {
                    dst[row] += matrix[row * 3 + col] * src[col];
                }
            }
        };
        let b = [1.0f32, 2.0, 3.0];
        let mut x = [0.0f32; 3];
        let iterations = solver.solve(&mut matvec, &b, &mut x).unwrap();
        assert!(iterations > 0);
        let mut residual = [0.0f32; 3];
        matvec(&x, &mut residual);
        for (res, rhs) in residual.iter_mut().zip(b.iter()) {
            *res -= rhs;
        }
        let norm: f32 = residual.iter().map(|v| v * v).sum();
        assert!(norm.sqrt() <= solver.tolerance().max(topos.tolerance()));
    }

    #[test]
    fn multi_modal_text_profile_clamps_values() {
        let topos = demo_topos();
        let guard = MultiModalToposGuard::new(&topos)
            .unwrap()
            .with_text_profile(ModalityProfile::new(16, Some(0.25)).unwrap())
            .unwrap();
        let mut tensor = Tensor::from_vec(4, 4, vec![10.0; 16]).unwrap();
        guard
            .guard_text_tensor("multi_modal_text", &mut tensor)
            .unwrap();
        let headroom = 0.25 * (1.0 + DEFAULT_MODALITY_PERMEABILITY);
        assert!(tensor
            .data()
            .iter()
            .all(|&value| value <= headroom + 1e-6 && value >= 0.25 - 1e-6));
    }

    #[test]
    fn multi_modal_graph_guard_reports_symmetry() {
        let topos = demo_topos();
        let guard = MultiModalToposGuard::new(&topos)
            .unwrap()
            .with_graph_profile(GraphGuardProfile::new(8, 12, 4, 1e-3, 0.05, Some(1.0)).unwrap())
            .unwrap();
        let mut adjacency = vec![0.0f32; 16];
        adjacency[0 * 4 + 1] = 0.2;
        adjacency[1 * 4 + 0] = 0.1;
        let report = guard.guard_graph_adjacency(&mut adjacency, 4).unwrap();
        assert_eq!(report.edge_count, 1);
        assert_eq!(report.symmetry_violations, 1);
        assert_eq!(report.edge_overflow, 0);
        assert!(adjacency.iter().all(|value| value.abs() <= 1.0));
    }

    #[test]
    fn multi_modal_reward_boundary_detects_breaches() {
        let topos = demo_topos();
        let guard = MultiModalToposGuard::new(&topos)
            .unwrap()
            .with_reward_boundary(RewardBoundary::new(-0.5, 0.5, 0.1).unwrap())
            .unwrap();
        let mut rewards = vec![-0.9, 0.2, 0.8];
        let signal = guard.guard_reward_trace(&mut rewards).unwrap();
        assert_eq!(signal.lower_breach_index, Some(0));
        assert_eq!(signal.upper_breach_index, Some(2));
        assert_eq!(signal.clamped, 2);
        assert!((signal.min_observed + 0.9).abs() < 1e-6);
        assert!((signal.max_observed - 0.8).abs() < 1e-6);
        assert!(rewards.iter().all(|value| *value >= -0.5 && *value <= 0.5));
    }

    #[test]
    fn multi_modal_atlas_tracks_volume_and_lifts() {
        let topos = demo_topos();
        let guard = MultiModalToposGuard::new(&topos).unwrap();
        let mut atlas = guard.atlas();
        let mut text = Tensor::from_vec(2, 4, vec![topos.saturation() * 4.0; 8]).unwrap();
        atlas
            .guard_text_tensor("multi_modal_atlas_text", &mut text)
            .unwrap();
        assert_eq!(atlas.visited_volume(), 8);
        let text_limit = atlas.guard.text.effective_saturation(atlas.guard.topos);
        let text_headroom = text_limit * (1.0 + atlas.guard.text.permeability());
        assert!(text
            .data()
            .iter()
            .all(|&value| value.abs() <= text_headroom + 1e-5));
        let lifted = atlas
            .lift_vision_tensor(
                "multi_modal_atlas_vision",
                Tensor::from_vec(1, 4, vec![topos.saturation() * 3.0; 4]).unwrap(),
            )
            .unwrap();
        let vision_limit = atlas.guard.vision.effective_saturation(atlas.guard.topos);
        let vision_headroom = vision_limit * (1.0 + atlas.guard.vision.permeability());
        assert!(lifted
            .data()
            .iter()
            .all(|&value| value.abs() <= vision_headroom + 1e-5));
        assert_eq!(atlas.visited_volume(), 12);
    }

    #[test]
    fn multi_modal_biome_absorbs_with_preserved_profiles() {
        let topos = demo_topos();
        let guard = MultiModalToposGuard::new(&topos)
            .unwrap()
            .with_text_profile(
                ModalityProfile::new(64, Some(0.5))
                    .unwrap()
                    .with_permeability(0.2)
                    .unwrap(),
            )
            .unwrap();
        let mut biome = guard.cultivate_biome();
        biome
            .absorb_text(
                "multi_modal_biome_text",
                Tensor::from_vec(1, 4, vec![2.0; 4]).unwrap(),
            )
            .unwrap();
        biome
            .absorb_vision_weighted(
                "multi_modal_biome_vision",
                Tensor::from_vec(1, 4, vec![5.0; 4]).unwrap(),
                2.0,
            )
            .unwrap();
        assert_eq!(biome.len(), 2);
        assert!((biome.total_weight() - 3.0).abs() < 1e-6);
        let canopy = biome.canopy().unwrap();
        let text_limit = biome.text.effective_saturation(biome.topos());
        let text_headroom = text_limit * (1.0 + biome.text.permeability());
        let vision_limit = biome.vision.effective_saturation(biome.topos());
        let vision_headroom = vision_limit * (1.0 + biome.vision.permeability());
        let max_headroom = text_headroom.max(vision_headroom);
        assert!(canopy
            .data()
            .iter()
            .all(|&value| value.abs() <= max_headroom + 1e-5));
        let mut atlas = biome.atlas();
        let mut waveform = vec![topos.saturation() * 6.0; 4];
        atlas
            .guard_audio_waveform("multi_modal_biome_audio", &mut waveform)
            .unwrap();
        let audio_limit = biome.audio.effective_saturation(biome.topos());
        let audio_headroom = audio_limit * (1.0 + biome.audio.permeability());
        assert!(waveform
            .iter()
            .all(|&value| value.abs() <= audio_headroom + 1e-5));
        let _lifted = atlas
            .lift_text_tensor(
                "multi_modal_biome_lift",
                Tensor::from_vec(1, 4, vec![topos.saturation() * 2.5; 4]).unwrap(),
            )
            .unwrap();
        assert_eq!(atlas.visited_volume(), 4);
    }
}
