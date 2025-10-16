// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

//! Microlocal boundary gauges used to detect interfaces that survive the
//! blow-up limit without committing to a global label.
//!
//! The constructions in this module mirror the BV/varifold correspondence
//! outlined in the SpiralTorch sheaf notes: only the presence of an interface
//! (the `R` machine) remains gauge invariant in the microlocal limit, while
//! oriented data such as co-normals requires an external label `c′` to fix a
//! sign.  `InterfaceGauge` provides a practical discretisation that measures
//! the local total-variation density inside shrinking metric balls, emits the
//! stable boundary indicator, and optionally reconstructs the oriented normal
//! field when a signed phase label is supplied.

use ndarray::{indices, ArrayD, IxDyn};
use statrs::function::gamma::gamma;
use std::f64::consts::PI;

/// Result of running an [`InterfaceGauge`] on a binary phase field.
#[derive(Debug, Clone)]
pub struct InterfaceSignature {
    /// Gauge invariant boundary detector (1 = interface present).
    pub r_machine: ArrayD<f32>,
    /// Raw (0–1) estimate of the perimeter density relative to the flat lattice.
    pub raw_density: ArrayD<f32>,
    /// Perimeter density normalised by \(|D\chi_E|(B_\varepsilon)/\varepsilon^{d-1}|\),
    /// mapped to \{0, \kappa_d\} to match the finite perimeter blow-up.
    pub perimeter_density: ArrayD<f32>,
    /// Optional unit normal field (only populated when `c_prime` was provided).
    pub orientation: Option<ArrayD<f32>>,
    /// Surface measure of the unit \((d-1)\)-sphere used for the normalisation.
    pub kappa_d: f32,
    /// Radius in lattice steps used to accumulate the blow-up statistics.
    pub radius: isize,
}

impl InterfaceSignature {
    /// Returns `true` when any interface cell was detected.
    pub fn has_interface(&self) -> bool {
        self.r_machine.iter().any(|v| *v > 0.0)
    }
}

/// Microlocal interface detector approximating the BV blow-up.
#[derive(Debug, Clone)]
pub struct InterfaceGauge {
    grid_spacing: f32,
    physical_radius: f32,
    threshold: f32,
}

impl InterfaceGauge {
    /// Constructs a new interface gauge.
    ///
    /// * `grid_spacing` – physical size of one lattice step (used for gradients).
    /// * `physical_radius` – radius of the metric ball used in the blow-up probe.
    pub fn new(grid_spacing: f32, physical_radius: f32) -> Self {
        Self {
            grid_spacing: grid_spacing.max(f32::EPSILON),
            physical_radius: physical_radius.max(f32::EPSILON),
            threshold: 0.25,
        }
    }

    /// Adjusts the contrast threshold used when deciding if two samples belong
    /// to different phases. Defaults to `0.25` which is robust for \{0,1\} masks.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.max(0.0);
        self
    }

    /// Evaluates the gauge on a binary mask, returning the interface signature.
    pub fn analyze(&self, mask: &ArrayD<f32>) -> InterfaceSignature {
        self.analyze_with_label(mask, None)
    }

    /// Evaluates the gauge on a binary mask using an optional signed label `c′`
    /// to reconstruct oriented normals.
    pub fn analyze_with_label(
        &self,
        mask: &ArrayD<f32>,
        c_prime: Option<&ArrayD<f32>>,
    ) -> InterfaceSignature {
        let dim = mask.ndim();
        assert!(dim > 0, "mask must have positive dimension");
        if let Some(label) = c_prime {
            assert_eq!(label.shape(), mask.shape(), "c′ must match mask shape");
        }

        let radius = self.radius_in_steps();
        let offsets = generate_offsets(dim, radius);
        let shape = mask.shape().to_vec();
        let raw_dim = mask.raw_dim();
        let mut r_machine = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut raw_density = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut perimeter_density = ArrayD::<f32>::zeros(raw_dim.clone());
        let mut orientation = c_prime.map(|_| {
            let mut full_shape = Vec::with_capacity(dim + 1);
            full_shape.push(dim);
            full_shape.extend_from_slice(&shape);
            ArrayD::<f32>::zeros(IxDyn(&full_shape))
        });

        let threshold = self.threshold;
        let kappa_d = unit_sphere_area(dim);

        for idx in indices(raw_dim.clone()) {
            let idx_slice = idx.slice();
            let idx_dyn = IxDyn(idx_slice);
            let center = mask[&idx_dyn];
            let mut min_val = center;
            let mut max_val = center;

            for off in offsets.iter().filter(|off| off.iter().any(|&o| o != 0)) {
                if let Some(neigh) = neighbor_sample(mask, idx_slice, &shape, off) {
                    min_val = min_val.min(neigh);
                    max_val = max_val.max(neigh);
                }
            }

            let has_interface = (max_val - min_val) >= threshold;
            let r_val = if has_interface { 1.0 } else { 0.0 };
            r_machine[&idx_dyn] = r_val;

            let (raw_val, discrete_jump) =
                local_raw_density(mask, idx_slice, &shape, threshold, center);
            raw_density[&idx_dyn] = raw_val;
            perimeter_density[&idx_dyn] = if discrete_jump { kappa_d } else { 0.0 };

            if let (Some(label_field), Some(ref mut orient)) = (c_prime, orientation.as_mut()) {
                if has_interface {
                    let normal = oriented_normal(
                        label_field,
                        idx_slice,
                        &shape,
                        self.grid_spacing,
                        threshold,
                    );
                    if let Some(n) = normal {
                        for axis in 0..dim {
                            let mut coord = Vec::with_capacity(dim + 1);
                            coord.push(axis);
                            coord.extend_from_slice(idx_slice);
                            orient[IxDyn(&coord)] = n[axis];
                        }
                    }
                }
            }
        }

        InterfaceSignature {
            r_machine,
            raw_density,
            perimeter_density,
            orientation,
            kappa_d,
            radius,
        }
    }

    fn radius_in_steps(&self) -> isize {
        let steps = (self.physical_radius / self.grid_spacing).ceil() as isize;
        steps.max(1)
    }
}

fn unit_sphere_area(dim: usize) -> f32 {
    let d = dim as f64;
    let area = 2.0 * PI.powf(d / 2.0) / gamma(d / 2.0);
    area as f32
}

fn generate_offsets(dim: usize, radius: isize) -> Vec<Vec<isize>> {
    fn recurse(acc: &mut Vec<Vec<isize>>, cur: &mut Vec<isize>, dim: usize, radius: isize) {
        if cur.len() == dim {
            acc.push(cur.clone());
            return;
        }
        for step in -radius..=radius {
            cur.push(step);
            recurse(acc, cur, dim, radius);
            cur.pop();
        }
    }
    let mut acc = Vec::new();
    let mut scratch = Vec::with_capacity(dim);
    recurse(&mut acc, &mut scratch, dim, radius);
    acc
}

fn neighbor_sample(
    field: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    offset: &[isize],
) -> Option<f32> {
    let mut coord = Vec::with_capacity(idx.len());
    for (axis, &delta) in offset.iter().enumerate() {
        let pos = idx[axis] as isize + delta;
        if pos < 0 || pos >= shape[axis] as isize {
            return None;
        }
        coord.push(pos as usize);
    }
    Some(field[IxDyn(&coord)])
}

fn local_raw_density(
    mask: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    threshold: f32,
    center: f32,
) -> (f32, bool) {
    let dim = idx.len();
    let mut diff_count = 0.0f32;
    let mut neighbor_total = 0.0f32;
    for axis in 0..dim {
        if idx[axis] + 1 < shape[axis] {
            neighbor_total += 1.0;
            let mut coord = idx.to_vec();
            coord[axis] += 1;
            let neigh = mask[IxDyn(&coord)];
            if (neigh - center).abs() >= threshold {
                diff_count += 1.0;
            }
        }
    }
    if neighbor_total > 0.0 {
        let raw = diff_count / neighbor_total;
        (raw, diff_count > 0.0)
    } else {
        (0.0, false)
    }
}

fn oriented_normal(
    label: &ArrayD<f32>,
    idx: &[usize],
    shape: &[usize],
    h: f32,
    threshold: f32,
) -> Option<Vec<f32>> {
    let dim = idx.len();
    let mut grad = vec![0.0f32; dim];
    let center = label[IxDyn(idx)];
    for axis in 0..dim {
        let forward = if idx[axis] + 1 < shape[axis] {
            let mut coord = idx.to_vec();
            coord[axis] += 1;
            label[IxDyn(&coord)]
        } else {
            center
        };
        let backward = if idx[axis] > 0 {
            let mut coord = idx.to_vec();
            coord[axis] -= 1;
            label[IxDyn(&coord)]
        } else {
            center
        };
        grad[axis] = (forward - backward) / (2.0 * h);
    }
    let norm_sq = grad.iter().map(|g| (*g as f64).powi(2)).sum::<f64>() as f32;
    if norm_sq.sqrt() < threshold {
        return None;
    }
    let norm = norm_sq.sqrt().max(f32::EPSILON);
    for g in &mut grad {
        *g /= norm;
    }
    Some(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn detects_boundary_presence() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let sig = gauge.analyze(&mask);
        assert!(sig.has_interface());
        assert_eq!(sig.kappa_d, 2.0 * std::f32::consts::PI);
        assert!((sig.perimeter_density[IxDyn(&[1, 1])] - sig.kappa_d).abs() < 1e-5);
        assert_eq!(sig.perimeter_density[IxDyn(&[0, 0])], 0.0);
    }

    #[test]
    fn oriented_normals_require_label() {
        let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
        let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
        let gauge = InterfaceGauge::new(1.0, 1.0);
        let sig = gauge.analyze_with_label(&mask, Some(&c_prime));
        let orient = sig.orientation.expect("orientation missing");
        let normal_y = orient[IxDyn(&[0, 1, 1])];
        let normal_x = orient[IxDyn(&[1, 1, 1])];
        assert!(normal_y.abs() > 0.5);
        assert!(normal_x.abs() < 1e-3);
    }
}
