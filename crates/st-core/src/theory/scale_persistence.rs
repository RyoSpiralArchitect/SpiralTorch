// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Bridges Z-pulse telemetry into the scale persistence pipeline.

use ndarray::Array2;
use st_frac::scale_stack::{ScaleStack, ScaleStackError, SemanticMetric};

use super::zpulse::ZPulse;

const FEATURE_DIM: usize = 8;

fn pulse_features(pulse: &ZPulse) -> [f32; FEATURE_DIM] {
    [
        pulse.total_energy(),
        pulse.drift,
        pulse.z_bias,
        pulse.support_mass(),
        pulse.tempo,
        pulse.quality,
        pulse.latency_ms,
        pulse.density_fluctuation(),
    ]
}

/// Builds a [`ScaleStack`] describing semantic coherence across a pulse trace.
pub fn zpulse_scale_stack(
    pulses: &[ZPulse],
    scales: &[f64],
    threshold: f32,
    metric: SemanticMetric,
) -> Result<ScaleStack, ScaleStackError> {
    if pulses.is_empty() {
        let empty = Array2::<f32>::zeros((0, FEATURE_DIM));
        return ScaleStack::from_semantic_field(
            empty.view().into_dyn(),
            scales,
            threshold,
            1,
            metric,
        );
    }
    let mut data = Vec::with_capacity(pulses.len() * FEATURE_DIM);
    for pulse in pulses {
        data.extend_from_slice(&pulse_features(pulse));
    }
    let array = Array2::from_shape_vec((pulses.len(), FEATURE_DIM), data).map_err(|_| {
        ScaleStackError::ShapeMismatch {
            expected: vec![pulses.len(), FEATURE_DIM],
            actual: vec![pulses.len(), FEATURE_DIM],
        }
    })?;
    ScaleStack::from_semantic_field(array.view().into_dyn(), scales, threshold, 1, metric)
}

/// Convenience helper returning coherence breakpoints for Z-pulse sequences.
pub fn zpulse_coherence_levels(
    pulses: &[ZPulse],
    scales: &[f64],
    threshold: f32,
    metric: SemanticMetric,
    levels: &[f64],
) -> Result<Vec<Option<f64>>, ScaleStackError> {
    let stack = zpulse_scale_stack(pulses, scales, threshold, metric)?;
    Ok(stack.coherence_profile(levels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::zpulse::{ZPulse, ZSource, ZSupport};
    use st_frac::scale_stack::InterfaceMode;

    fn sample_pulse(drift: f32, energy: f32) -> ZPulse {
        let band_energy = (energy, 0.0, 0.0);
        ZPulse {
            source: ZSource::Microlocal,
            ts: 0,
            tempo: 1.0,
            band_energy,
            density_fluctuation: ZPulse::density_fluctuation_for(band_energy),
            drift,
            z_bias: 0.0,
            support: ZSupport::default(),
            scale: None,
            quality: 0.5,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }

    #[test]
    fn zpulse_bridge_produces_semantic_stack() {
        let pulses = vec![
            sample_pulse(0.0, 1.0),
            sample_pulse(0.0, 1.0),
            sample_pulse(1.0, 2.0),
            sample_pulse(1.0, 2.0),
        ];
        let stack =
            zpulse_scale_stack(&pulses, &[1.0, 2.0], 0.25, SemanticMetric::Euclidean).unwrap();
        assert!(matches!(stack.mode(), InterfaceMode::Semantic { .. }));
        assert!(stack.samples()[0].gate_mean > 0.0);
    }

    #[test]
    fn zpulse_coherence_levels_return_values() {
        let pulses = vec![
            sample_pulse(0.0, 1.0),
            sample_pulse(0.0, 1.0),
            sample_pulse(1.0, 2.0),
            sample_pulse(1.0, 2.0),
        ];
        let levels = zpulse_coherence_levels(
            &pulses,
            &[1.0, 2.0, 3.0],
            0.25,
            SemanticMetric::Euclidean,
            &[0.25, 0.5],
        )
        .unwrap();
        assert_eq!(levels.len(), 2);
    }
}
