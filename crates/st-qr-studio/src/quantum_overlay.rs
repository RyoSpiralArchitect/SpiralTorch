// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use serde::{Deserialize, Serialize};
use st_core::maxwell::MaxwellZPulse;
use std::f64::consts::PI;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumOverlayConfig {
    curvature: f64,
    qubits: usize,
    packing_bias: f64,
    leech_shells: usize,
}

impl QuantumOverlayConfig {
    pub fn new(curvature: f64, qubits: usize) -> Self {
        let curvature = if curvature.is_finite() {
            curvature
        } else {
            -1.0
        };
        let qubits = qubits.max(1);
        Self {
            curvature,
            qubits,
            packing_bias: 0.35,
            leech_shells: 24,
        }
    }

    pub fn with_packing_bias(mut self, packing_bias: f64) -> Self {
        if packing_bias.is_finite() {
            self.packing_bias = packing_bias.clamp(0.0, 1.0);
        }
        self
    }

    pub fn with_leech_shells(mut self, shells: usize) -> Self {
        self.leech_shells = shells.max(1);
        self
    }

    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    pub fn qubits(&self) -> usize {
        self.qubits
    }

    pub fn leech_shells(&self) -> usize {
        self.leech_shells
    }

    pub fn packing_bias(&self) -> f64 {
        self.packing_bias
    }
}

impl Default for QuantumOverlayConfig {
    fn default() -> Self {
        Self::new(-1.0, 24)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZResonance {
    pub spectrum: Vec<f64>,
    pub eta_hint: f32,
    pub shell_weights: Vec<f64>,
}

impl ZResonance {
    pub fn from_pulses(pulses: &[MaxwellZPulse]) -> Self {
        if pulses.is_empty() {
            return Self::from_spectrum(Vec::new(), 0.0);
        }
        let mut spectrum = Vec::with_capacity(pulses.len());
        let mut shell_weights = Vec::with_capacity(pulses.len());
        let mut eta_acc = 0.0f32;
        for pulse in pulses {
            let energy = pulse.band_energy;
            let shell = energy.0 as f64 + energy.1 as f64 + energy.2 as f64;
            shell_weights.push(shell.max(0.0));
            spectrum.push(pulse.mean);
            eta_acc += pulse.z_bias.abs().max(0.0);
        }
        let eta_hint = if pulses.is_empty() {
            0.0
        } else {
            (eta_acc / pulses.len() as f32).min(2.0)
        };
        Self {
            spectrum,
            eta_hint,
            shell_weights,
        }
    }

    pub fn from_spectrum(spectrum: Vec<f64>, eta_hint: f32) -> Self {
        let shells = if spectrum.is_empty() {
            Vec::new()
        } else {
            spectrum
                .iter()
                .enumerate()
                .map(|(idx, value)| value.abs() * (idx as f64 + 1.0))
                .collect()
        };
        Self {
            spectrum,
            eta_hint: eta_hint.max(0.0),
            shell_weights: shells,
        }
    }

    pub fn shell_weights(&self) -> &[f64] {
        &self.shell_weights
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZOverlayCircuit {
    config: QuantumOverlayConfig,
    weights: Vec<f64>,
    eta_bar: f64,
    packing_pressure: f64,
}

impl ZOverlayCircuit {
    pub(crate) fn from_config(config: &QuantumOverlayConfig, resonance: &ZResonance) -> Self {
        let shells = normalize_shells(resonance, config.leech_shells());
        let curvature = config.curvature().abs().max(1e-6);
        let mut weights = Vec::with_capacity(config.qubits());
        for idx in 0..config.qubits() {
            let shell = shells[idx % shells.len()].abs();
            let hyper = (shell * curvature).tanh();
            let leech_phase = (idx as f64 / config.leech_shells() as f64) * 2.0 * PI;
            let packing = (leech_phase.sin() * config.packing_bias()).abs();
            weights.push(hyper * (1.0 - config.packing_bias()) + packing * config.packing_bias());
        }
        let packing_pressure = shells.iter().copied().sum::<f64>() / shells.len().max(1) as f64;
        let eta_bar = (resonance.eta_hint as f64 + packing_pressure).tanh().abs();
        Self {
            config: config.clone(),
            weights,
            eta_bar,
            packing_pressure,
        }
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn eta_bar(&self) -> f64 {
        self.eta_bar
    }

    pub fn packing_pressure(&self) -> f64 {
        self.packing_pressure
    }

    pub fn measure(&self, threshold: f64) -> QuantumMeasurement {
        let threshold = if threshold.is_finite() {
            threshold
        } else {
            0.0
        };
        let mut indexed: Vec<(usize, f64)> = self.weights.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut active = Vec::new();
        let mut logits = Vec::with_capacity(self.weights.len());
        for (idx, weight) in indexed.iter() {
            if weight >= &threshold || active.is_empty() {
                active.push(*idx);
            }
            logits.push(*weight as f32);
        }
        QuantumMeasurement {
            active_qubits: active,
            eta_bar: self.eta_bar as f32,
            policy_logits: logits,
            packing_pressure: self.packing_pressure as f32,
        }
    }

    pub fn synthesize(config: &QuantumOverlayConfig, resonance: &ZResonance) -> Self {
        Self::from_config(config, resonance)
    }
}

fn normalize_shells(resonance: &ZResonance, shells: usize) -> Vec<f64> {
    if resonance.shell_weights().is_empty() {
        let shells = shells.max(1);
        return (0..shells).map(|idx| 1.0 / (idx as f64 + 1.0)).collect();
    }
    let mut weights = resonance.shell_weights().to_vec();
    if weights.len() < shells {
        let mut idx = 0;
        while weights.len() < shells {
            weights.push(weights[idx % resonance.shell_weights().len()].abs());
            idx += 1;
        }
    }
    let norm = weights
        .iter()
        .copied()
        .fold(0.0, |acc, value| acc + value.abs());
    if norm <= 0.0 {
        return vec![1.0; shells.max(1)];
    }
    weights.into_iter().map(|value| value / norm).collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumMeasurement {
    active_qubits: Vec<usize>,
    eta_bar: f32,
    policy_logits: Vec<f32>,
    packing_pressure: f32,
}

impl QuantumMeasurement {
    pub fn active_qubits(&self) -> &[usize] {
        &self.active_qubits
    }

    pub fn eta_bar(&self) -> f32 {
        self.eta_bar
    }

    pub fn policy_logits(&self) -> &[f32] {
        &self.policy_logits
    }

    pub fn packing_pressure(&self) -> f32 {
        self.packing_pressure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_measures_qubits() {
        let config = QuantumOverlayConfig::new(-1.0, 4);
        let pulses = vec![MaxwellZPulse {
            blocks: 1,
            mean: 0.5,
            standard_error: 0.1,
            z_score: 2.0,
            band_energy: (0.2, 0.4, 0.6),
            z_bias: 0.7,
        }];
        let resonance = ZResonance::from_pulses(&pulses);
        let circuit = ZOverlayCircuit::from_config(&config, &resonance);
        let measurement = circuit.measure(0.1);
        assert!(!measurement.active_qubits().is_empty());
        assert!(measurement.eta_bar() > 0.0);
        assert_eq!(measurement.policy_logits().len(), config.qubits());
    }
}
