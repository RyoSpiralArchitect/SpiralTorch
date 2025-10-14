/// Aggregate band energy used by the free-energy scoring proxy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BandEnergy {
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
}

impl BandEnergy {
    /// Returns the L1 magnitude across all bands.
    pub fn l1(&self) -> f32 {
        self.above.abs() + self.here.abs() + self.beneath.abs()
    }

    /// Normalises the band energies so they sum to one. When every component is
    /// zero the method returns an even split.
    pub fn norm(self) -> Self {
        let sum = self.l1();
        if sum <= f32::EPSILON {
            return Self {
                above: 1.0 / 3.0,
                here: 1.0 / 3.0,
                beneath: 1.0 / 3.0,
            };
        }
        Self {
            above: (self.above / sum).clamp(0.0, 1.0),
            here: (self.here / sum).clamp(0.0, 1.0),
            beneath: (self.beneath / sum).clamp(0.0, 1.0),
        }
    }
}

/// Context passed into [`score_with_free_energy`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FeCtx {
    pub loss_before: f32,
    pub loss_after: f32,
    pub step_ms: f32,
    pub mem_mb: f32,
    pub retry: f32,
    pub band: BandEnergy,
    pub entropy: f32,
}

/// Computes a free-energy inspired utility score used to rank plans.
pub fn score_with_free_energy(ctx: &FeCtx, beta: f32) -> f32 {
    let beta = beta.max(0.0);
    let delta_l = ctx.loss_after - ctx.loss_before;
    let entropy = ctx.entropy.max(0.0);
    let free_energy = delta_l + beta * entropy;

    let norm = ctx.band.norm();
    let novelty = norm.above - norm.beneath;
    let stability = (1.0 - norm.here).clamp(0.0, 1.0);

    let step_penalty = 0.0025 * ctx.step_ms.max(0.0);
    let mem_penalty = 0.001 * ctx.mem_mb.max(0.0);
    let retry_penalty = 0.5 * ctx.retry.max(0.0);

    -free_energy - step_penalty - mem_penalty - retry_penalty + 0.2 * novelty - 0.1 * stability
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_entropy_is_penalised() {
        let ctx = FeCtx {
            loss_before: 0.5,
            loss_after: 0.25,
            entropy: 1.2,
            step_ms: 10.0,
            mem_mb: 128.0,
            retry: 0.0,
            band: BandEnergy {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
            },
        };
        let tight = score_with_free_energy(&ctx, 0.0);
        let loose = score_with_free_energy(&ctx, 0.8);
        assert!(loose < tight);
    }

    #[test]
    fn novelty_bias_pushes_above_band() {
        let base = FeCtx {
            loss_before: 0.4,
            loss_after: 0.3,
            entropy: 0.4,
            step_ms: 12.0,
            mem_mb: 64.0,
            retry: 0.0,
            band: BandEnergy {
                above: 0.2,
                here: 0.6,
                beneath: 0.2,
            },
        };
        let novel = FeCtx {
            band: BandEnergy {
                above: 0.6,
                here: 0.3,
                beneath: 0.1,
            },
            ..base
        };
        assert!(score_with_free_energy(&novel, 0.3) > score_with_free_energy(&base, 0.3));
    }
}
