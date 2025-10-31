use std::collections::HashMap;

use crate::sensors::FusedFrame;

/// Encodes an instinctive objective via a quadratic tolerance band.
#[derive(Debug, Clone)]
pub struct Desire {
    pub target_norm: f32,
    pub tolerance: f32,
    pub weight: f32,
}

impl Desire {
    pub fn energy(&self, norm: f32) -> f32 {
        let delta = (norm - self.target_norm).abs() - self.tolerance;
        let penalty = delta.max(0.0);
        penalty * self.weight
    }
}

/// Aggregates channel energy contributions per desire field.
#[derive(Debug, Clone)]
pub struct EnergyReport {
    pub total: f32,
    pub per_channel: HashMap<String, f32>,
}

impl EnergyReport {
    pub fn zero() -> Self {
        Self {
            total: 0.0,
            per_channel: HashMap::new(),
        }
    }
}

/// Collection of instincts spanning sensor channels.
#[derive(Debug, Clone, Default)]
pub struct DesireLagrangianField {
    desires: HashMap<String, Desire>,
}

impl DesireLagrangianField {
    pub fn new(desires: HashMap<String, Desire>) -> Self {
        Self { desires }
    }

    pub fn energy(&self, frame: &FusedFrame) -> EnergyReport {
        let mut report = EnergyReport::zero();
        for (name, desire) in &self.desires {
            if let Some(norm) = frame.norm(name) {
                let channel_energy = desire.energy(norm);
                report.total += channel_energy;
                report.per_channel.insert(name.clone(), channel_energy);
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensors::SensorFusionHub;

    #[test]
    fn desire_field_accumulates_energy() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("pose", 1).unwrap();
        let frame = hub
            .fuse(&HashMap::from([("pose".to_string(), vec![0.7])]))
            .unwrap();
        let mut desires = HashMap::new();
        desires.insert(
            "pose".to_string(),
            Desire {
                target_norm: 0.2,
                tolerance: 0.05,
                weight: 2.0,
            },
        );
        let field = DesireLagrangianField::new(desires);
        let energy = field.energy(&frame);
        assert!(energy.total > 0.0);
        assert!(energy.per_channel.get("pose").unwrap() > &0.0);
    }
}
