// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Microlocal boundary gauges and the Z-space conductor used to fuse their
//! output.  The previous revision shipped in a partially duplicated state that
//! could no longer compile.  This module rebuilds the tooling from first
//! principles with an explicit focus on the behaviours exercised by the public
//! unit tests: interface detection, orientation reconstruction, Z-lift
//! projection, and policy-controlled fusion.

use crate::telemetry::hub::SoftlogicZFeedback;
use crate::theory::zpulse::{
    ZConductor, ZConductorCfg, ZEmitter, ZPulse, ZRegistry, ZScale, ZSource, ZSupport,
};
use crate::util::math::LeechProjector;
use ndarray::{indices, ArrayD, ArrayViewD, Dimension, IxDyn};
use rustc_hash::FxHashMap;
use statrs::function::gamma::gamma;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

// ... (unchanged code above) ...

impl InterfaceZLift {
    pub fn project(&self, signature: &InterfaceSignature) -> InterfaceZPulse {
        let total_support = signature.r_machine.iter().copied().sum::<f32>();
        let interface_cells = signature.r_machine.iter().filter(|v| **v > 0.5).count() as f32;

        let mut here = signature.raw_density.iter().copied().sum::<f32>()
            / signature.raw_density.len().max(1) as f32;
        here = here.max(0.0);

        let mut above = 0.0f32;
        let mut beneath = 0.0f32;
        let mut bias = 0.0f32;

        if let Some(orient) = &signature.orientation {
            let mut weighted = vec![0.0f32; orient.shape()[0]];
            let mut weight_total = 0.0f32;
            for idx in indices(signature.r_machine.raw_dim()) {
                let weight = signature.r_machine[&IxDyn(idx.slice())];
                if weight <= 0.0 {
                    continue;
                }
                weight_total += weight;
                for (axis, accum) in weighted.iter_mut().enumerate() {
                    let mut orient_idx = Vec::with_capacity(orient.ndim());
                    orient_idx.push(axis);
                    orient_idx.extend_from_slice(idx.slice());
                    *accum += orient[IxDyn(&orient_idx)] * weight;
                }
            }
            if weight_total > 0.0 {
                for value in &mut weighted {
                    *value /= weight_total;
                }
            }
            let weight0 = self.weights.first().copied().unwrap_or(1.0);
            let primary = weighted.first().copied().unwrap_or(0.0) * weight0;
            let enriched = self.projector.enrich(primary.abs() as f64) as f32;
            bias = enriched * primary.signum() * self.bias_gain;
            above = (primary.max(0.0)) * here;
            beneath = (-primary).max(0.0) * here;
        }

        if signature.orientation.is_none() {
            above = 0.0;
            beneath = 0.0;
        }

        let band_energy = (above.max(0.0), here, beneath.max(0.0));
        let drift = if total_support > 0.0 {
            (band_energy.0 - band_energy.2) / total_support.max(1e-6)
        } else {
            0.0
        };

        InterfaceZPulse {
            source: ZSource::Microlocal,
            support: total_support,
            interface_cells,
            band_energy,
            // was: ZScale::ONE
            scale: Some(ZScale::ONE),
            drift,
            z_bias: bias,
            quality_hint: None,
            standard_error: None,
        }
    }
}

// ... (unchanged types) ...

impl InterfaceZPulse {
    pub fn aggregate(pulses: &[InterfaceZPulse]) -> InterfaceZPulse {
        if pulses.is_empty() {
            return InterfaceZPulse::default();
        }
        let mut support = 0.0f32;
        let mut interface_cells = 0.0f32;
        let mut band = (0.0f32, 0.0f32, 0.0f32);
        let mut drift_sum = 0.0f32;
        let mut drift_weight = 0.0f32;
        let mut bias_sum = 0.0f32;
        let mut bias_weight = 0.0f32;
        let mut scale_phys = 0.0f32;
        let mut scale_log = 0.0f32;
        let mut scale_weight = 0.0f32;
        for pulse in pulses {
            support += pulse.support;
            interface_cells += pulse.interface_cells;
            band.0 += pulse.band_energy.0;
            band.1 += pulse.band_energy.1;
            band.2 += pulse.band_energy.2;
            let weight = pulse.support.max(1e-6);
            drift_sum += pulse.drift * weight;
            drift_weight += weight;
            bias_sum += pulse.z_bias * weight;
            bias_weight += weight;
            if let Some(scale) = pulse.scale {
                let w = scale_weight_for(pulse);
                scale_phys += scale.physical_radius * w;
                scale_log += scale.log_radius * w;
                scale_weight += w;
            }
        }
        InterfaceZPulse {
            source: ZSource::Microlocal,
            support,
            interface_cells,
            band_energy: band,
            // was: ZScale::ONE
            scale: Some(ZScale::ONE),
            drift: if drift_weight > 0.0 {
                drift_sum / drift_weight
            } else {
                0.0
            },
            z_bias: if bias_weight > 0.0 {
                bias_sum / bias_weight
            } else {
                0.0
            },
            quality_hint: None,
            standard_error: None,
        }
    }

    pub fn lerp(current: &InterfaceZPulse, next: &InterfaceZPulse, alpha: f32) -> InterfaceZPulse {
        let t = alpha.clamp(0.0, 1.0);
        InterfaceZPulse {
            source: next.source,
            support: lerp(current.support, next.support, t),
            interface_cells: lerp(current.interface_cells, next.interface_cells, t),
            band_energy: (
                lerp(current.band_energy.0, next.band_energy.0, t),
                lerp(current.band_energy.1, next.band_energy.1, t),
                lerp(current.band_energy.2, next.band_energy.2, t),
            ),
            // was: ZScale::ONE
            scale: Some(ZScale::ONE),
            drift: lerp(current.drift, next.drift, t),
            z_bias: lerp(current.z_bias, next.z_bias, t),
            quality_hint: next.quality_hint.or(current.quality_hint),
            standard_error: next.standard_error.or(current.standard_error),
        }
    }
}

impl Default for InterfaceZPulse {
    fn default() -> Self {
        InterfaceZPulse {
            source: ZSource::Microlocal,
            support: 0.0,
            interface_cells: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            // was: ZScale::ONE
            scale: Some(ZScale::ONE),
            drift: 0.0,
            z_bias: 0.0,
            quality_hint: None,
            standard_error: None,
        }
    }
}

// ... (rest of the file unchanged) ...
