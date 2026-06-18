// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Bridges Z-pulse telemetry into the scale persistence pipeline.

use ndarray::Array2;
use st_frac::scale_stack::{ScaleStack, ScaleStackError, SemanticMetric};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

use super::zpulse::{ZPulse, ZSource};

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
    let stack = if pulses.is_empty() {
        let empty = Array2::<f32>::zeros((0, FEATURE_DIM));
        ScaleStack::from_semantic_field(empty.view().into_dyn(), scales, threshold, 1, metric)?
    } else {
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
        ScaleStack::from_semantic_field(array.view().into_dyn(), scales, threshold, 1, metric)?
    };
    emit_zpulse_scale_stack_meta(pulses, scales, threshold, metric, &stack);
    Ok(stack)
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
    let profile = stack.coherence_profile(levels);
    emit_zpulse_coherence_levels_meta(pulses, scales, threshold, metric, levels, &profile);
    Ok(profile)
}

fn finite_meta_f64(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn finite_meta_f32(value: f32) -> f64 {
    if value.is_finite() {
        value as f64
    } else {
        0.0
    }
}

fn metric_label(metric: SemanticMetric) -> &'static str {
    match metric {
        SemanticMetric::Euclidean => "euclidean",
        SemanticMetric::Cosine => "cosine",
    }
}

fn source_label(source: &ZSource) -> &'static str {
    match source {
        ZSource::Microlocal => "microlocal",
        ZSource::Maxwell => "maxwell",
        ZSource::Graph => "graph",
        ZSource::Desire => "desire",
        ZSource::GW => "gw",
        ZSource::RealGrad => "realgrad",
        ZSource::Other(_) => "other",
    }
}

fn pulse_feature_summary(pulses: &[ZPulse]) -> serde_json::Map<String, serde_json::Value> {
    let mut payload = serde_json::Map::new();
    if pulses.is_empty() {
        payload.insert("energy_mean".into(), 0.0.into());
        payload.insert("energy_max".into(), 0.0.into());
        payload.insert("drift_abs_mean".into(), 0.0.into());
        payload.insert("z_bias_abs_mean".into(), 0.0.into());
        payload.insert("support_mean".into(), 0.0.into());
        payload.insert("quality_mean".into(), 0.0.into());
        payload.insert("latency_abs_mean".into(), 0.0.into());
        payload.insert("density_mean".into(), 0.0.into());
        payload.insert("source_microlocal_count".into(), 0.into());
        payload.insert("source_maxwell_count".into(), 0.into());
        payload.insert("source_graph_count".into(), 0.into());
        payload.insert("source_desire_count".into(), 0.into());
        payload.insert("source_other_count".into(), 0.into());
        return payload;
    }

    let mut energy_sum = 0.0f32;
    let mut energy_max = 0.0f32;
    let mut drift_abs_sum = 0.0f32;
    let mut z_bias_abs_sum = 0.0f32;
    let mut support_sum = 0.0f32;
    let mut quality_sum = 0.0f32;
    let mut latency_abs_sum = 0.0f32;
    let mut density_sum = 0.0f32;
    let mut microlocal = 0usize;
    let mut maxwell = 0usize;
    let mut graph = 0usize;
    let mut desire = 0usize;
    let mut other = 0usize;
    for pulse in pulses {
        let energy = pulse.total_energy().max(0.0);
        energy_sum += energy;
        energy_max = energy_max.max(energy);
        drift_abs_sum += pulse.drift.abs();
        z_bias_abs_sum += pulse.z_bias.abs();
        support_sum += pulse.support_mass().max(0.0);
        quality_sum += pulse.quality.clamp(0.0, 1.0);
        latency_abs_sum += pulse.latency_ms.abs();
        density_sum += pulse.density_fluctuation().clamp(0.0, 1.0);
        match source_label(&pulse.source) {
            "microlocal" => microlocal += 1,
            "maxwell" => maxwell += 1,
            "graph" => graph += 1,
            "desire" => desire += 1,
            _ => other += 1,
        }
    }
    let count = pulses.len() as f32;
    payload.insert(
        "energy_mean".into(),
        finite_meta_f32(energy_sum / count).into(),
    );
    payload.insert("energy_max".into(), finite_meta_f32(energy_max).into());
    payload.insert(
        "drift_abs_mean".into(),
        finite_meta_f32(drift_abs_sum / count).into(),
    );
    payload.insert(
        "z_bias_abs_mean".into(),
        finite_meta_f32(z_bias_abs_sum / count).into(),
    );
    payload.insert(
        "support_mean".into(),
        finite_meta_f32(support_sum / count).into(),
    );
    payload.insert(
        "quality_mean".into(),
        finite_meta_f32(quality_sum / count).into(),
    );
    payload.insert(
        "latency_abs_mean".into(),
        finite_meta_f32(latency_abs_sum / count).into(),
    );
    payload.insert(
        "density_mean".into(),
        finite_meta_f32(density_sum / count).into(),
    );
    payload.insert("source_microlocal_count".into(), microlocal.into());
    payload.insert("source_maxwell_count".into(), maxwell.into());
    payload.insert("source_graph_count".into(), graph.into());
    payload.insert("source_desire_count".into(), desire.into());
    payload.insert("source_other_count".into(), other.into());
    payload
}

fn scale_stack_summary(stack: &ScaleStack) -> (f64, f64, f64, f64, usize, f64) {
    let samples = stack.samples();
    let gate_first = samples
        .first()
        .map(|sample| sample.gate_mean)
        .unwrap_or(0.0);
    let gate_final = samples.last().map(|sample| sample.gate_mean).unwrap_or(0.0);
    let mut gate_sum = 0.0f64;
    let mut gate_max = 0.0f64;
    for sample in samples {
        gate_sum += sample.gate_mean;
        gate_max = gate_max.max(sample.gate_mean);
    }
    let gate_mean = if samples.is_empty() {
        0.0
    } else {
        gate_sum / samples.len() as f64
    };
    let bins = stack.persistence_measure();
    let persistence_mass = bins.iter().map(|bin| bin.mass).sum::<f64>();
    (
        gate_first,
        gate_final,
        gate_mean,
        gate_max,
        bins.len(),
        persistence_mass,
    )
}

fn emit_zpulse_scale_stack_meta(
    pulses: &[ZPulse],
    scales: &[f64],
    threshold: f32,
    metric: SemanticMetric,
    stack: &ScaleStack,
) {
    let (gate_first, gate_final, gate_mean, gate_max, persistence_bins, persistence_mass) =
        scale_stack_summary(stack);
    emit_tensor_op(
        "zpulse_scale_stack",
        &[pulses.len(), FEATURE_DIM],
        &[stack.samples().len(), 2],
    );
    emit_tensor_op_meta("zpulse_scale_stack", || {
        let mut payload = pulse_feature_summary(pulses);
        payload.insert("backend".into(), "cpu".into());
        payload.insert("requested_backend".into(), "auto".into());
        payload.insert("kind".into(), "st_core_zpulse_scale_stack".into());
        payload.insert("pulse_count".into(), pulses.len().into());
        payload.insert("feature_dim".into(), FEATURE_DIM.into());
        payload.insert("scale_count".into(), scales.len().into());
        payload.insert(
            "scale_min".into(),
            finite_meta_f64(scales.first().copied().unwrap_or(0.0)).into(),
        );
        payload.insert(
            "scale_max".into(),
            finite_meta_f64(scales.last().copied().unwrap_or(0.0)).into(),
        );
        payload.insert("threshold".into(), finite_meta_f32(threshold).into());
        payload.insert("metric".into(), metric_label(metric).into());
        payload.insert("sample_count".into(), stack.samples().len().into());
        payload.insert("gate_first".into(), finite_meta_f64(gate_first).into());
        payload.insert("gate_final".into(), finite_meta_f64(gate_final).into());
        payload.insert("gate_mean".into(), finite_meta_f64(gate_mean).into());
        payload.insert("gate_max".into(), finite_meta_f64(gate_max).into());
        payload.insert("persistence_bins".into(), persistence_bins.into());
        payload.insert(
            "persistence_mass".into(),
            finite_meta_f64(persistence_mass).into(),
        );
        payload.into()
    });
}

fn emit_zpulse_coherence_levels_meta(
    pulses: &[ZPulse],
    scales: &[f64],
    threshold: f32,
    metric: SemanticMetric,
    levels: &[f64],
    profile: &[Option<f64>],
) {
    let resolved = profile.iter().filter(|value| value.is_some()).count();
    let first_resolved = profile.iter().flatten().next().copied().unwrap_or(0.0);
    let last_resolved = profile
        .iter()
        .rev()
        .flatten()
        .next()
        .copied()
        .unwrap_or(0.0);
    emit_tensor_op(
        "zpulse_coherence_levels",
        &[pulses.len(), levels.len()],
        &[resolved.max(1), 1],
    );
    emit_tensor_op_meta("zpulse_coherence_levels", || {
        let mut payload = pulse_feature_summary(pulses);
        payload.insert("backend".into(), "cpu".into());
        payload.insert("requested_backend".into(), "auto".into());
        payload.insert("kind".into(), "st_core_zpulse_coherence_levels".into());
        payload.insert("pulse_count".into(), pulses.len().into());
        payload.insert("scale_count".into(), scales.len().into());
        payload.insert("level_count".into(), levels.len().into());
        payload.insert("resolved_count".into(), resolved.into());
        payload.insert(
            "resolved_ratio".into(),
            finite_meta_f64(if levels.is_empty() {
                0.0
            } else {
                resolved as f64 / levels.len() as f64
            })
            .into(),
        );
        payload.insert(
            "first_level".into(),
            finite_meta_f64(levels.first().copied().unwrap_or(0.0)).into(),
        );
        payload.insert(
            "last_level".into(),
            finite_meta_f64(levels.last().copied().unwrap_or(0.0)).into(),
        );
        payload.insert(
            "first_resolved_scale".into(),
            finite_meta_f64(first_resolved).into(),
        );
        payload.insert(
            "last_resolved_scale".into(),
            finite_meta_f64(last_resolved).into(),
        );
        payload.insert("threshold".into(), finite_meta_f32(threshold).into());
        payload.insert("metric".into(), metric_label(metric).into());
        payload.into()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::zpulse::{ZPulse, ZSource, ZSupport};
    use st_frac::scale_stack::InterfaceMode;
    use std::sync::{Arc, Mutex};

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

    #[test]
    fn zpulse_scale_stack_and_levels_emit_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut pulses = vec![
            sample_pulse(0.0, 1.0),
            sample_pulse(0.0, 1.0),
            sample_pulse(1.0, 2.0),
            sample_pulse(1.0, 2.0),
        ];
        pulses[2].source = ZSource::Maxwell;
        let levels = zpulse_coherence_levels(
            &pulses,
            &[1.0, 2.0, 3.0],
            0.25,
            SemanticMetric::Euclidean,
            &[0.25, 0.5],
        )
        .unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(levels.len(), 2);
        let events = events.lock().unwrap();
        let stack_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zpulse_scale_stack"
                    && data["kind"] == "st_core_zpulse_scale_stack"
                    && data["pulse_count"] == 4
                    && data["scale_count"] == 3
                    && data["source_maxwell_count"] == 1
            })
            .expect("zpulse_scale_stack metadata event");
        assert_eq!(stack_meta.1["backend"], "cpu");
        assert_eq!(stack_meta.1["requested_backend"], "auto");
        assert_eq!(stack_meta.1["pulse_count"], 4);
        assert_eq!(stack_meta.1["feature_dim"], 8);
        assert_eq!(stack_meta.1["scale_count"], 3);
        assert_eq!(stack_meta.1["metric"], "euclidean");
        assert_eq!(stack_meta.1["source_microlocal_count"], 3);
        assert_eq!(stack_meta.1["source_maxwell_count"], 1);
        assert!(stack_meta.1["energy_mean"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(stack_meta.1["gate_final"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(stack_meta.1["persistence_mass"].as_f64().unwrap_or(0.0) > 0.0);

        let level_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zpulse_coherence_levels"
                    && data["kind"] == "st_core_zpulse_coherence_levels"
                    && data["pulse_count"] == 4
                    && data["level_count"] == 2
            })
            .expect("zpulse_coherence_levels metadata event");
        assert_eq!(level_meta.1["backend"], "cpu");
        assert_eq!(level_meta.1["pulse_count"], 4);
        assert_eq!(level_meta.1["level_count"], 2);
        assert!(level_meta.1["resolved_count"].as_u64().unwrap_or(0) > 0);
        assert!(level_meta.1["resolved_ratio"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(level_meta.1["first_resolved_scale"].as_f64().unwrap_or(0.0) > 0.0);
    }
}
