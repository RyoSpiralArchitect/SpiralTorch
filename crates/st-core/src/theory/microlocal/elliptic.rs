// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::f32::consts::{PI, TAU};

use crate::telemetry::hub::SoftlogicEllipticSample;

const EPSILON: f32 = 1e-6;

/// Lie group frame describing the SO(3) rotation aligning the +Z pole with the
/// sampled orientation.
#[derive(Clone, Copy, Debug)]
pub struct LieFrame {
    quaternion: [f32; 4],
    rotation: [f32; 9],
}

impl LieFrame {
    pub fn identity() -> Self {
        Self {
            quaternion: [1.0, 0.0, 0.0, 0.0],
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn from_axis_angle(axis: [f32; 3], angle: f32) -> Self {
        let half = 0.5 * angle;
        let (s, c) = half.sin_cos();
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let quaternion = if norm <= EPSILON {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [
                c,
                axis[0] / norm * s,
                axis[1] / norm * s,
                axis[2] / norm * s,
            ]
        };
        Self::from_quaternion(quaternion)
    }

    pub fn from_matrix(matrix: [f32; 9]) -> Self {
        let quaternion = matrix_to_quaternion(matrix);
        Self::from_quaternion(quaternion)
    }

    pub fn from_quaternion(mut quaternion: [f32; 4]) -> Self {
        let mut norm = 0.0f32;
        for q in &quaternion {
            norm += q * q;
        }
        norm = norm.sqrt();
        if norm <= EPSILON {
            quaternion = [1.0, 0.0, 0.0, 0.0];
        } else {
            for q in quaternion.iter_mut() {
                *q /= norm;
            }
            if quaternion[0] < 0.0 {
                for q in quaternion.iter_mut() {
                    *q = -*q;
                }
            }
        }
        let rotation = quaternion_to_matrix(quaternion);
        Self {
            quaternion,
            rotation,
        }
    }

    pub fn quaternion(&self) -> [f32; 4] {
        self.quaternion
    }

    pub fn rotation_matrix(&self) -> [f32; 9] {
        self.rotation
    }

    pub fn apply(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.rotation[0] * v[0] + self.rotation[1] * v[1] + self.rotation[2] * v[2],
            self.rotation[3] * v[0] + self.rotation[4] * v[1] + self.rotation[5] * v[2],
            self.rotation[6] * v[0] + self.rotation[7] * v[1] + self.rotation[8] * v[2],
        ]
    }

    pub fn apply_inverse(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.rotation[0] * v[0] + self.rotation[3] * v[1] + self.rotation[6] * v[2],
            self.rotation[1] * v[0] + self.rotation[4] * v[1] + self.rotation[7] * v[2],
            self.rotation[2] * v[0] + self.rotation[5] * v[1] + self.rotation[8] * v[2],
        ]
    }

    pub fn inverse(&self) -> Self {
        let [w, x, y, z] = self.quaternion;
        let quaternion = [w, -x, -y, -z];
        let rotation = [
            self.rotation[0],
            self.rotation[3],
            self.rotation[6],
            self.rotation[1],
            self.rotation[4],
            self.rotation[7],
            self.rotation[2],
            self.rotation[5],
            self.rotation[8],
        ];
        Self {
            quaternion,
            rotation,
        }
    }

    pub fn compose(&self, other: &LieFrame) -> Self {
        let [w1, x1, y1, z1] = self.quaternion;
        let [w2, x2, y2, z2] = other.quaternion;
        let quaternion = [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ];
        Self::from_quaternion(quaternion)
    }

    pub fn relative_to(&self, reference: &LieFrame) -> Self {
        reference.inverse().compose(self)
    }

    pub fn log(&self) -> [f32; 3] {
        let [w, x, y, z] = self.quaternion;
        let norm_v = (x * x + y * y + z * z).sqrt();
        if norm_v <= EPSILON {
            return [0.0, 0.0, 0.0];
        }
        let angle = 2.0 * norm_v.atan2(w);
        let scale = angle / norm_v;
        [x * scale, y * scale, z * scale]
    }

    pub fn exp(tangent: [f32; 3]) -> Self {
        let theta =
            (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
        if theta <= EPSILON {
            return Self::identity();
        }
        let axis = [tangent[0] / theta, tangent[1] / theta, tangent[2] / theta];
        Self::from_axis_angle(axis, theta)
    }

    pub fn transport(&self, vector: [f32; 3], target: &LieFrame) -> [f32; 3] {
        let world = self.apply(vector);
        target.apply_inverse(world)
    }
}

impl Default for LieFrame {
    fn default() -> Self {
        Self::identity()
    }
}

/// Differential of the elliptic warp with respect to the input orientation.
#[derive(Clone, Debug, Default)]
pub struct EllipticDifferential {
    pub features: [f32; 9],
    pub jacobian: [[f32; 3]; 9],
}

impl EllipticDifferential {
    pub fn feature_slice(&self) -> &[f32; 9] {
        &self.features
    }

    pub fn jacobian(&self) -> &[[f32; 3]; 9] {
        &self.jacobian
    }
}

/// Positive-curvature warp that remaps microlocal orientations onto an elliptic
/// Z-frame.
#[derive(Clone, Debug)]
pub struct EllipticWarp {
    curvature_radius: f32,
    sheet_count: usize,
    spin_harmonics: usize,
}

impl EllipticWarp {
    /// Creates a warp anchored to the provided curvature radius.
    pub fn new(curvature_radius: f32) -> Self {
        let radius = curvature_radius.max(EPSILON);
        Self {
            curvature_radius: radius,
            sheet_count: 2,
            spin_harmonics: 1,
        }
    }

    /// Configures the number of discrete sheets representing the χ axis.
    pub fn with_sheet_count(mut self, sheet_count: usize) -> Self {
        self.sheet_count = sheet_count.max(1);
        self
    }

    /// Configures the number of spin harmonics applied while computing ν.
    pub fn with_spin_harmonics(mut self, harmonics: usize) -> Self {
        self.spin_harmonics = harmonics.max(1);
        self
    }

    /// Returns the curvature radius associated with the warp.
    pub fn curvature_radius(&self) -> f32 {
        self.curvature_radius
    }

    /// Returns the number of χ sheets encoded by the warp.
    pub fn sheet_count(&self) -> usize {
        self.sheet_count
    }

    /// Returns the number of spin harmonics applied to the ν axis.
    pub fn spin_harmonics(&self) -> usize {
        self.spin_harmonics
    }

    /// Maximum geodesic radius reachable on the warp.
    pub fn max_geodesic(&self) -> f32 {
        self.curvature_radius * PI
    }

    /// Maps an orientation vector to elliptic telemetry describing the warped
    /// coordinates. Returns `None` when the orientation is degenerate.
    pub fn map_orientation(&self, orientation: &[f32]) -> Option<EllipticTelemetry> {
        self.map_orientation_with_differential(orientation)
            .map(|(telemetry, _)| telemetry)
    }

    /// Maps an orientation vector to elliptic telemetry and returns the forward differential
    /// of the differentiable features with respect to the provided orientation. The differential
    /// can be wired into autograd tapes.
    pub fn map_orientation_with_differential(
        &self,
        orientation: &[f32],
    ) -> Option<(EllipticTelemetry, EllipticDifferential)> {
        if orientation.is_empty() {
            return None;
        }

        let mut dir = [0.0f32; 3];
        for (dst, &value) in dir.iter_mut().zip(orientation.iter()).take(3) {
            *dst = value;
        }
        let norm_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= EPSILON {
            return None;
        }

        for component in dir.iter_mut() {
            *component /= norm;
        }

        let (telemetry, differential) = self.build_telemetry_from_unit(dir, norm);
        Some((telemetry, differential))
    }

    fn build_telemetry_from_unit(
        &self,
        dir: [f32; 3],
        source_norm: f32,
    ) -> (EllipticTelemetry, EllipticDifferential) {
        let jac_unit = unit_normalization_jacobian(dir, source_norm);
        let dir_z = dir[2].clamp(-1.0, 1.0);
        let polar = dir_z.acos();
        let sin_theta = (1.0 - dir_z * dir_z).sqrt();
        let geodesic_radius = polar * self.curvature_radius;
        let denom_radius = (self.curvature_radius * PI).max(EPSILON);
        let mut normalized_radius = (geodesic_radius / denom_radius).clamp(0.0, 1.0);

        let azimuth = dir[1].atan2(dir[0]);
        let spin_gain = self.spin_harmonics.max(1) as f32;
        let spin_phase = azimuth / PI;
        let spin_alignment = (spin_phase * spin_gain).sin();

        let sheet_count = self.sheet_count.max(1);
        let sheet_phase_raw = (azimuth + PI) / TAU;
        let sheet_phase = sheet_phase_raw.fract();
        let sheet_f = sheet_phase * sheet_count as f32;
        let mut sheet_index = sheet_f.floor() as usize;
        if sheet_index >= sheet_count {
            sheet_index = sheet_count - 1;
        }
        let sheet_position = if sheet_count <= 1 {
            0.0
        } else {
            sheet_phase.clamp(0.0, 1.0)
        };

        let normal_bias = dir_z;

        let base = [0.0, 0.0, 1.0];
        let axis = [
            base[1] * dir[2] - base[2] * dir[1],
            base[2] * dir[0] - base[0] * dir[2],
            base[0] * dir[1] - base[1] * dir[0],
        ];
        let lie_frame = LieFrame::from_axis_angle(axis, polar);
        let lie_log = lie_frame.log();
        let flow_vector = lie_frame.apply([1.0, 0.0, 0.0]);

        let curvature_tensor = curvature_tensor_from_direction(dir, self.curvature_radius);

        let sin_theta_safe = sin_theta.max(EPSILON);
        let rotor_unit = if sin_theta <= EPSILON {
            [0.0, 0.0, 0.0]
        } else {
            [-dir[1] / sin_theta_safe, dir[0] / sin_theta_safe, 0.0]
        };
        let rotor_field = [
            rotor_unit[0] * polar,
            rotor_unit[1] * polar,
            rotor_unit[2] * polar,
        ];

        let rotor_norm = (rotor_field[0] * rotor_field[0]
            + rotor_field[1] * rotor_field[1]
            + rotor_field[2] * rotor_field[2])
            .sqrt();
        let spin_magnitude = (spin_alignment * spin_alignment + EPSILON).sqrt();
        let resonance_heat = normalized_radius * spin_magnitude;
        let noise_denom = polar + EPSILON;
        let mut raw_noise = 1.0 - rotor_norm / noise_denom;
        if !raw_noise.is_finite() {
            raw_noise = 0.0;
        }
        let noise_density = raw_noise.clamp(0.0, 1.0);

        let orientation_parity = if normal_bias >= 0.0 { 0 } else { 1 };
        let topological_sector = ((sheet_index as u32) << 1) | orientation_parity;
        let homology_index =
            compute_homology_index(sheet_index, sheet_count, spin_alignment, normalized_radius);
        let rotor_transport = lie_frame.apply_inverse(rotor_field);

        let features = [
            geodesic_radius,
            normalized_radius,
            spin_alignment,
            sheet_position,
            normal_bias,
            resonance_heat,
            rotor_field[0],
            rotor_field[1],
            rotor_field[2],
        ];

        // Jacobians
        let mut d_polar = [0.0; 3];
        if sin_theta > EPSILON {
            for (d_polar_j, &jac) in d_polar.iter_mut().zip(jac_unit[2].iter()) {
                *d_polar_j = -jac / sin_theta_safe;
            }
        }

        let mut d_geodesic = [0.0; 3];
        for j in 0..3 {
            d_geodesic[j] = self.curvature_radius * d_polar[j];
        }

        let mut d_normalized = [0.0; 3];
        let base_normalized = geodesic_radius / denom_radius;
        if base_normalized > 0.0 && base_normalized < 1.0 {
            for j in 0..3 {
                d_normalized[j] = d_geodesic[j] / denom_radius;
            }
        } else {
            normalized_radius = normalized_radius.clamp(0.0, 1.0);
        }

        let denom_xy = (dir[0] * dir[0] + dir[1] * dir[1]).max(EPSILON);
        let mut d_azimuth = [0.0; 3];
        let azimuth_coeff_x = -dir[1] / denom_xy;
        let azimuth_coeff_y = dir[0] / denom_xy;
        for ((d_azimuth_j, &jac0), &jac1) in d_azimuth
            .iter_mut()
            .zip(jac_unit[0].iter())
            .zip(jac_unit[1].iter())
        {
            *d_azimuth_j = azimuth_coeff_x * jac0 + azimuth_coeff_y * jac1;
        }

        let mut d_spin = [0.0; 3];
        let cos_input = (spin_phase * spin_gain).cos();
        for j in 0..3 {
            d_spin[j] = cos_input * spin_gain / PI * d_azimuth[j];
        }

        let mut d_sheet_position = [0.0; 3];
        if sheet_count > 1 && sheet_position > 0.0 && sheet_position < 1.0 {
            for j in 0..3 {
                d_sheet_position[j] = d_azimuth[j] / TAU;
            }
        }

        let mut d_normal_bias = [0.0; 3];
        d_normal_bias.copy_from_slice(&jac_unit[2]);

        let mut d_rotor_x = [0.0; 3];
        let mut d_rotor_y = [0.0; 3];
        let mut d_rotor_z = [0.0; 3];
        if sin_theta > EPSILON {
            let mut d_sin_theta = [0.0; 3];
            let sin_theta_coeff = -dir[2] / sin_theta_safe;
            for (d_sin_theta_j, &jac) in d_sin_theta.iter_mut().zip(jac_unit[2].iter()) {
                *d_sin_theta_j = sin_theta_coeff * jac;
            }
            for j in 0..3 {
                let d_dir0 = jac_unit[0][j];
                let d_dir1 = jac_unit[1][j];
                let d_axis_unit_x = -d_dir1 / sin_theta_safe
                    + dir[1] / (sin_theta_safe * sin_theta_safe) * d_sin_theta[j];
                let d_axis_unit_y = d_dir0 / sin_theta_safe
                    - dir[0] / (sin_theta_safe * sin_theta_safe) * d_sin_theta[j];
                d_rotor_x[j] = d_axis_unit_x * polar + rotor_unit[0] * d_polar[j];
                d_rotor_y[j] = d_axis_unit_y * polar + rotor_unit[1] * d_polar[j];
                d_rotor_z[j] = rotor_unit[2] * d_polar[j];
            }
        }

        let mut d_rotor_norm = [0.0; 3];
        if rotor_norm > EPSILON {
            for j in 0..3 {
                let delta = rotor_field[0] * d_rotor_x[j]
                    + rotor_field[1] * d_rotor_y[j]
                    + rotor_field[2] * d_rotor_z[j];
                d_rotor_norm[j] = delta / rotor_norm;
            }
        }

        let mut d_resonance = [0.0; 3];
        let spin_magnitude_safe = spin_magnitude.max(EPSILON);
        for j in 0..3 {
            let d_spin_mag = (spin_alignment / spin_magnitude_safe) * d_spin[j];
            d_resonance[j] = normalized_radius * d_spin_mag + spin_magnitude * d_normalized[j];
        }

        let mut jacobian = [[0.0; 3]; 9];
        jacobian[0] = d_geodesic;
        jacobian[1] = d_normalized;
        jacobian[2] = d_spin;
        jacobian[3] = d_sheet_position;
        jacobian[4] = d_normal_bias;
        jacobian[5] = d_resonance;
        jacobian[6] = d_rotor_x;
        jacobian[7] = d_rotor_y;
        jacobian[8] = d_rotor_z;

        let differential = EllipticDifferential { features, jacobian };

        let telemetry = EllipticTelemetry {
            curvature_radius: self.curvature_radius,
            geodesic_radius,
            normalized_radius,
            spin_alignment,
            sheet_index,
            sheet_position,
            normal_bias,
            sheet_count,
            topological_sector,
            homology_index,
            rotor_field,
            flow_vector,
            curvature_tensor,
            resonance_heat,
            noise_density,
            lie_frame,
            lie_log,
            rotor_transport,
        };

        (telemetry, differential)
    }
}

/// Telemetry describing an elliptic Z-frame projection.
#[derive(Clone, Debug)]
pub struct EllipticTelemetry {
    pub curvature_radius: f32,
    pub geodesic_radius: f32,
    pub normalized_radius: f32,
    pub spin_alignment: f32,
    pub sheet_index: usize,
    pub sheet_position: f32,
    pub normal_bias: f32,
    pub sheet_count: usize,
    pub topological_sector: u32,
    pub homology_index: u32,
    pub rotor_field: [f32; 3],
    pub flow_vector: [f32; 3],
    pub curvature_tensor: [[f32; 3]; 3],
    pub resonance_heat: f32,
    pub noise_density: f32,
    pub lie_frame: LieFrame,
    pub lie_log: [f32; 3],
    pub rotor_transport: [f32; 3],
}

impl Default for EllipticTelemetry {
    fn default() -> Self {
        Self {
            curvature_radius: 1.0,
            geodesic_radius: 0.0,
            normalized_radius: 0.0,
            spin_alignment: 0.0,
            sheet_index: 0,
            sheet_position: 0.0,
            normal_bias: 1.0,
            sheet_count: 1,
            topological_sector: 0,
            homology_index: 0,
            rotor_field: [0.0; 3],
            flow_vector: [1.0, 0.0, 0.0],
            curvature_tensor: [[0.0; 3]; 3],
            resonance_heat: 0.0,
            noise_density: 0.0,
            lie_frame: LieFrame::identity(),
            lie_log: [0.0; 3],
            rotor_transport: [0.0; 3],
        }
    }
}

impl EllipticTelemetry {
    /// Normalised geodesic radius within \([0, 1]\).
    pub fn normalized_radius(&self) -> f32 {
        self.normalized_radius.clamp(0.0, 1.0)
    }

    /// Interpolates two telemetry samples.
    pub fn lerp(&self, other: &EllipticTelemetry, t: f32) -> EllipticTelemetry {
        let clamped = t.clamp(0.0, 1.0);
        let sheet_count = self.sheet_count.max(other.sheet_count).max(1);
        let sheet_position =
            super::lerp(self.sheet_position, other.sheet_position, clamped).clamp(0.0, 1.0);
        let sheet_index = ((sheet_position * sheet_count as f32).round() as usize)
            .min(sheet_count.saturating_sub(1));
        let normalized_radius =
            super::lerp(self.normalized_radius, other.normalized_radius, clamped).clamp(0.0, 1.0);
        let mut quaternion = [0.0f32; 4];
        let qa = self.lie_frame.quaternion();
        let qb = other.lie_frame.quaternion();
        for (slot, (&a, &b)) in quaternion.iter_mut().zip(qa.iter().zip(qb.iter())) {
            *slot = super::lerp(a, b, clamped);
        }
        let lie_frame = LieFrame::from_quaternion(quaternion);
        let mut rotor_field = [0.0; 3];
        let mut flow_vector = [0.0; 3];
        for i in 0..3 {
            rotor_field[i] = super::lerp(self.rotor_field[i], other.rotor_field[i], clamped);
            flow_vector[i] = super::lerp(self.flow_vector[i], other.flow_vector[i], clamped);
        }
        let mut lie_log = [0.0; 3];
        let mut rotor_transport = [0.0; 3];
        for i in 0..3 {
            lie_log[i] = super::lerp(self.lie_log[i], other.lie_log[i], clamped);
            rotor_transport[i] =
                super::lerp(self.rotor_transport[i], other.rotor_transport[i], clamped);
        }
        let mut curvature_tensor = [[0.0; 3]; 3];
        for (out_row, (self_row, other_row)) in curvature_tensor
            .iter_mut()
            .zip(self.curvature_tensor.iter().zip(other.curvature_tensor.iter()))
        {
            for (out, (&a, &b)) in out_row.iter_mut().zip(self_row.iter().zip(other_row.iter())) {
                *out = super::lerp(a, b, clamped);
            }
        }
        let topological_sector = if clamped < 0.5 {
            self.topological_sector
        } else {
            other.topological_sector
        };
        let homology_index = if clamped < 0.5 {
            self.homology_index
        } else {
            other.homology_index
        };
        EllipticTelemetry {
            curvature_radius: super::lerp(self.curvature_radius, other.curvature_radius, clamped)
                .max(EPSILON),
            geodesic_radius: super::lerp(self.geodesic_radius, other.geodesic_radius, clamped)
                .max(0.0),
            normalized_radius,
            spin_alignment: super::lerp(self.spin_alignment, other.spin_alignment, clamped)
                .clamp(-1.0, 1.0),
            sheet_index,
            sheet_position,
            normal_bias: super::lerp(self.normal_bias, other.normal_bias, clamped).clamp(-1.0, 1.0),
            sheet_count,
            topological_sector,
            homology_index,
            rotor_field,
            flow_vector,
            curvature_tensor,
            resonance_heat: super::lerp(self.resonance_heat, other.resonance_heat, clamped),
            noise_density: super::lerp(self.noise_density, other.noise_density, clamped)
                .clamp(0.0, 1.0),
            lie_frame,
            lie_log,
            rotor_transport,
        }
    }

    /// Returns event tags that summarise the elliptic telemetry.
    pub fn event_tags(&self) -> [String; 5] {
        [
            format!("elliptic.sheet:{:02}", self.sheet_index),
            format!("elliptic.sector:{:02}", self.topological_sector),
            format!("elliptic.radius:{:.4}", self.normalized_radius()),
            format!("elliptic.spin:{:.3}", self.spin_alignment),
            format!("elliptic.heat:{:.3}", self.resonance_heat),
        ]
    }
}

impl From<&EllipticTelemetry> for SoftlogicEllipticSample {
    fn from(telemetry: &EllipticTelemetry) -> Self {
        SoftlogicEllipticSample {
            curvature_radius: telemetry.curvature_radius,
            geodesic_radius: telemetry.geodesic_radius,
            normalized_radius: telemetry.normalized_radius(),
            spin_alignment: telemetry.spin_alignment,
            sheet_index: telemetry.sheet_index as u32,
            sheet_position: telemetry.sheet_position,
            normal_bias: telemetry.normal_bias,
            sheet_count: telemetry.sheet_count as u32,
            topological_sector: telemetry.topological_sector,
            homology_index: telemetry.homology_index,
            rotor_field: telemetry.rotor_field,
            flow_vector: telemetry.flow_vector,
            curvature_tensor: telemetry.curvature_tensor,
            resonance_heat: telemetry.resonance_heat,
            noise_density: telemetry.noise_density,
            quaternion: telemetry.lie_frame.quaternion(),
            rotation: telemetry.lie_frame.rotation_matrix(),
            lie_log: telemetry.lie_log,
            rotor_transport: telemetry.rotor_transport,
        }
    }
}

#[derive(Default)]
pub(crate) struct EllipticAccumulator {
    curvature_sum: f32,
    radius_sum: f32,
    normalized_sum: f32,
    bias_sum: f32,
    spin_sum: f32,
    sheet_sum: f32,
    rotor_sum: [f32; 3],
    flow_sum: [f32; 3],
    lie_log_sum: [f32; 3],
    transport_sum: [f32; 3],
    tensor_sum: [[f32; 3]; 3],
    heat_sum: f32,
    noise_sum: f32,
    quat_sum: [f32; 4],
    weight: f32,
    sheet_count: usize,
    topological_sector: u32,
    topological_weight: f32,
    homology_index: u32,
    homology_weight: f32,
}

impl EllipticAccumulator {
    pub(crate) fn accumulate(&mut self, telemetry: &EllipticTelemetry, weight: f32) {
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        self.curvature_sum += telemetry.curvature_radius * weight;
        self.radius_sum += telemetry.geodesic_radius * weight;
        self.normalized_sum += telemetry.normalized_radius * weight;
        self.bias_sum += telemetry.normal_bias * weight;
        self.spin_sum += telemetry.spin_alignment * weight;
        self.sheet_sum += telemetry.sheet_position * weight;
        for i in 0..3 {
            self.rotor_sum[i] += telemetry.rotor_field[i] * weight;
            self.flow_sum[i] += telemetry.flow_vector[i] * weight;
            self.lie_log_sum[i] += telemetry.lie_log[i] * weight;
            self.transport_sum[i] += telemetry.rotor_transport[i] * weight;
        }
        for i in 0..3 {
            for j in 0..3 {
                self.tensor_sum[i][j] += telemetry.curvature_tensor[i][j] * weight;
            }
        }
        self.heat_sum += telemetry.resonance_heat * weight;
        self.noise_sum += telemetry.noise_density * weight;
        let quaternion = telemetry.lie_frame.quaternion();
        for (slot, value) in self.quat_sum.iter_mut().zip(quaternion.iter()) {
            *slot += value * weight;
        }
        self.weight += weight;
        if telemetry.sheet_count > self.sheet_count {
            self.sheet_count = telemetry.sheet_count;
        }
        if telemetry.topological_sector == self.topological_sector {
            self.topological_weight += weight;
        } else if self.topological_weight <= 0.0 || weight > self.topological_weight {
            self.topological_sector = telemetry.topological_sector;
            self.topological_weight = weight;
        }
        if telemetry.homology_index == self.homology_index {
            self.homology_weight += weight;
        } else if self.homology_weight <= 0.0 || weight > self.homology_weight {
            self.homology_index = telemetry.homology_index;
            self.homology_weight = weight;
        }
    }

    pub(crate) fn finish(self) -> Option<EllipticTelemetry> {
        if self.weight <= 0.0 {
            return None;
        }
        let sheet_count = self.sheet_count.max(1);
        let sheet_position = (self.sheet_sum / self.weight).clamp(0.0, 1.0);
        let sheet_index = ((sheet_position * sheet_count as f32).round() as usize)
            .min(sheet_count.saturating_sub(1));
        let mut quaternion = self.quat_sum;
        for q in quaternion.iter_mut() {
            *q /= self.weight;
        }
        let lie_frame = LieFrame::from_quaternion(quaternion);
        let mut rotor_field = [0.0f32; 3];
        let mut flow_vector = [0.0f32; 3];
        let mut lie_log = [0.0f32; 3];
        let mut rotor_transport = [0.0f32; 3];
        for i in 0..3 {
            rotor_field[i] = self.rotor_sum[i] / self.weight;
            flow_vector[i] = self.flow_sum[i] / self.weight;
            lie_log[i] = self.lie_log_sum[i] / self.weight;
            rotor_transport[i] = self.transport_sum[i] / self.weight;
        }
        let mut curvature_tensor = [[0.0f32; 3]; 3];
        for (out_row, in_row) in curvature_tensor.iter_mut().zip(self.tensor_sum.iter()) {
            for (out, value) in out_row.iter_mut().zip(in_row.iter()) {
                *out = *value / self.weight;
            }
        }
        Some(EllipticTelemetry {
            curvature_radius: (self.curvature_sum / self.weight).max(EPSILON),
            geodesic_radius: (self.radius_sum / self.weight).max(0.0),
            normalized_radius: (self.normalized_sum / self.weight).clamp(0.0, 1.0),
            spin_alignment: (self.spin_sum / self.weight).clamp(-1.0, 1.0),
            sheet_index,
            sheet_position,
            normal_bias: (self.bias_sum / self.weight).clamp(-1.0, 1.0),
            sheet_count,
            topological_sector: self.topological_sector,
            homology_index: self.homology_index,
            rotor_field,
            flow_vector,
            curvature_tensor,
            resonance_heat: self.heat_sum / self.weight,
            noise_density: (self.noise_sum / self.weight).clamp(0.0, 1.0),
            lie_frame,
            lie_log,
            rotor_transport,
        })
    }
}

fn unit_normalization_jacobian(dir: [f32; 3], source_norm: f32) -> [[f32; 3]; 3] {
    let mut jac = [[0.0f32; 3]; 3];
    let denom = source_norm.max(EPSILON);
    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            jac[i][j] = (delta - dir[i] * dir[j]) / denom;
        }
    }
    jac
}

fn quaternion_to_matrix(q: [f32; 4]) -> [f32; 9] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

fn matrix_to_quaternion(m: [f32; 9]) -> [f32; 4] {
    let trace = m[0] + m[4] + m[8];
    let mut q = [0.0f32; 4];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt().max(EPSILON) * 2.0;
        q[0] = 0.25 * s;
        q[1] = (m[7] - m[5]) / s;
        q[2] = (m[2] - m[6]) / s;
        q[3] = (m[3] - m[1]) / s;
    } else if m[0] > m[4] && m[0] > m[8] {
        let s = (1.0 + m[0] - m[4] - m[8]).sqrt().max(EPSILON) * 2.0;
        q[0] = (m[7] - m[5]) / s;
        q[1] = 0.25 * s;
        q[2] = (m[1] + m[3]) / s;
        q[3] = (m[2] + m[6]) / s;
    } else if m[4] > m[8] {
        let s = (1.0 + m[4] - m[0] - m[8]).sqrt().max(EPSILON) * 2.0;
        q[0] = (m[2] - m[6]) / s;
        q[1] = (m[1] + m[3]) / s;
        q[2] = 0.25 * s;
        q[3] = (m[5] + m[7]) / s;
    } else {
        let s = (1.0 + m[8] - m[0] - m[4]).sqrt().max(EPSILON) * 2.0;
        q[0] = (m[3] - m[1]) / s;
        q[1] = (m[2] + m[6]) / s;
        q[2] = (m[5] + m[7]) / s;
        q[3] = 0.25 * s;
    }
    q
}

fn curvature_tensor_from_direction(dir: [f32; 3], radius: f32) -> [[f32; 3]; 3] {
    let inv = 1.0 / radius.max(EPSILON);
    let mut tensor = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            tensor[i][j] = dir[i] * dir[j] * inv;
        }
    }
    tensor
}

fn compute_homology_index(
    sheet_index: usize,
    sheet_count: usize,
    spin_alignment: f32,
    normalized_radius: f32,
) -> u32 {
    let sheet = sheet_index as u32 & 0x3FF;
    let sheets = sheet_count as u32 & 0x3FF;
    let spin_bucket =
        ((spin_alignment.clamp(-1.0, 1.0) * 32767.0).round() as i32).wrapping_add(32768) as u32;
    let radius_bucket = (normalized_radius.clamp(0.0, 1.0) * 65535.0).round() as u32;
    (sheet << 22) ^ (sheets << 12) ^ (spin_bucket << 1) ^ radius_bucket
}
