// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Shared gravitational field primitives for Z-space dynamics.

use std::collections::HashMap;

/// Encodes the physics regime for a gravitational source.
#[derive(Debug, Clone)]
pub enum GravityRegime {
    /// Classical Newtonian gravity.
    Newtonian,
    /// Relativistic gravity damped by finite light speed.
    Relativistic { speed_of_light: f32 },
}

impl GravityRegime {
    fn relativistic_factor(&self, radius: f32) -> f32 {
        match self {
            GravityRegime::Newtonian => 1.0,
            GravityRegime::Relativistic { speed_of_light } => {
                let c = (*speed_of_light).max(1e-6);
                1.0 / (1.0 + radius / c)
            }
        }
    }
}

/// Mass descriptor for a gravitational source in Z-space.
#[derive(Debug, Clone)]
pub struct GravityWell {
    pub mass: f32,
    pub regime: GravityRegime,
}

impl GravityWell {
    pub fn new(mass: f32, regime: GravityRegime) -> Self {
        Self { mass, regime }
    }
}

/// Aggregate gravitational field covering multiple channels.
#[derive(Debug, Clone)]
pub struct GravityField {
    constant: f32,
    wells: HashMap<String, GravityWell>,
}

impl GravityField {
    pub fn new(constant: f32) -> Self {
        Self {
            constant,
            wells: HashMap::new(),
        }
    }

    pub fn with_wells(constant: f32, wells: HashMap<String, GravityWell>) -> Self {
        Self { constant, wells }
    }

    pub fn constant(&self) -> f32 {
        self.constant
    }

    pub fn wells(&self) -> &HashMap<String, GravityWell> {
        &self.wells
    }

    pub fn wells_mut(&mut self) -> &mut HashMap<String, GravityWell> {
        &mut self.wells
    }

    pub fn add_well(&mut self, channel: impl Into<String>, well: GravityWell) {
        self.wells.insert(channel.into(), well);
    }

    /// Compute the potential for a channel at a given metric radius.
    pub fn potential(&self, channel: &str, radius: f32) -> Option<f32> {
        let well = self.wells.get(channel)?;
        if radius <= 1e-6 {
            return Some(0.0);
        }
        let base = -self.constant * well.mass / radius;
        let factor = well.regime.relativistic_factor(radius);
        Some(base * factor)
    }
}

impl Default for GravityField {
    fn default() -> Self {
        const G: f32 = 6.67430e-11_f32;
        Self::new(G)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn potential_returns_value_for_registered_channel() {
        let mut field = GravityField::default();
        field.add_well("pose", GravityWell::new(10.0, GravityRegime::Newtonian));
        let potential = field.potential("pose", 2.0).unwrap();
        assert!(potential.is_sign_negative());
    }

    #[test]
    fn relativistic_regime_dampens_potential() {
        let mut field = GravityField::default();
        field.add_well(
            "pose",
            GravityWell::new(
                10.0,
                GravityRegime::Relativistic {
                    speed_of_light: 10.0,
                },
            ),
        );
        let newtonian = field
            .wells()
            .get("pose")
            .map(|well| -field.constant() * well.mass / 2.0)
            .unwrap();
        let relativistic = field.potential("pose", 2.0).unwrap();
        assert!(relativistic > newtonian);
    }
}
