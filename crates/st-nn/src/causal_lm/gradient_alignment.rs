//! Compare sealed, unmodified loss gradients without stepping or decoding the model.
use super::sum::Sum;
use super::*;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct CausalLmGradientPairStats {
    pub elements: usize,
    pub left_l2: f64,
    pub right_l2: f64,
    pub dot: f64,
    /// Absent when either gradient is zero.
    pub cosine: Option<f64>,
    /// L2 norm of the equally weighted arithmetic mean gradient.
    pub mean_l2: f64,
    /// ||left + right|| / (||left|| + ||right||), absent only if both are zero.
    pub retained_fraction: Option<f64>,
    /// First derivatives of the two losses along -(left + right)/2. Negative means descent.
    pub left_loss_slope: f64,
    pub right_loss_slope: f64,
}

#[derive(Debug, Serialize)]
pub struct CausalLmGradientAlignment {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub geometry_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub left_target_tokens: usize,
    pub right_target_tokens: usize,
    pub total: CausalLmGradientPairStats,
    pub parameters: BTreeMap<String, BTreeMap<String, CausalLmGradientPairStats>>,
}

#[derive(Default)]
struct Moments {
    n: usize,
    left2: Sum,
    right2: Sum,
    dot: Sum,
    mean2: Sum,
    left_slope: Sum,
    right_slope: Sum,
}

impl Moments {
    fn measure(left: &[f32], right: &[f32]) -> CausalLmResult<Self> {
        if left.len() != right.len() || left.is_empty() {
            return Err(
                "gradient alignment requires matching nonempty parameter dimensions".into(),
            );
        }
        let mut s = Self {
            n: left.len(),
            ..Self::default()
        };
        for (&a, &b) in left.iter().zip(right) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite parameter in gradient alignment".into());
            }
            let (a, b) = (f64::from(a), f64::from(b));
            let mean = (a + b) * 0.5;
            s.left2.add(a * a);
            s.right2.add(b * b);
            s.dot.add(a * b);
            // Do not reconstruct these from the Gram matrix: nearly opposite
            // vectors can lose their entire residual to subtraction.
            s.mean2.add(mean * mean);
            s.left_slope.add(-a * mean);
            s.right_slope.add(-b * mean);
        }
        Ok(s)
    }

    fn merge(&mut self, other: &Self) -> CausalLmResult<()> {
        self.n = self
            .n
            .checked_add(other.n)
            .ok_or("gradient alignment count overflow")?;
        self.left2.merge(&other.left2);
        self.right2.merge(&other.right2);
        self.dot.merge(&other.dot);
        self.mean2.merge(&other.mean2);
        self.left_slope.merge(&other.left_slope);
        self.right_slope.merge(&other.right_slope);
        Ok(())
    }

    fn finish(self) -> CausalLmGradientPairStats {
        let left_l2 = self.left2.value().sqrt();
        let right_l2 = self.right2.value().sqrt();
        let dot = self.dot.value();
        let mean_l2 = self.mean2.value().sqrt();
        CausalLmGradientPairStats {
            elements: self.n,
            left_l2,
            right_l2,
            dot,
            mean_l2,
            cosine: (left_l2 > 0. && right_l2 > 0.)
                .then(|| (dot / (left_l2 * right_l2)).clamp(-1., 1.)),
            retained_fraction: (left_l2 + right_l2 > 0.)
                .then(|| (2. * mean_l2 / (left_l2 + right_l2)).min(1.)),
            left_loss_slope: self.left_slope.value(),
            right_loss_slope: self.right_slope.value(),
        }
    }
}

impl NativeCausalLmAdaptationGradients {
    /// Inputs/objectives may differ. The weight owners, readout policy and
    /// full set of differentiated coordinates must match. No model work occurs.
    pub fn alignment_with(&self, other: &Self) -> CausalLmResult<CausalLmGradientAlignment> {
        if self.base_model_identity() != other.base_model_identity()
            || self.lora_artifact_id() != other.lora_artifact_id()
            || self.geometry_artifact_id() != other.geometry_artifact_id()
            || self.readout_accumulator() != other.readout_accumulator()
        {
            return Err("gradient alignment requires the same snapshot and readout policy".into());
        }
        let (left, right) = (self.parameters(), other.parameters());
        if left.is_empty() || left.keys().ne(right.keys()) {
            return Err("gradient alignment requires identical differentiated placements".into());
        }
        let mut total = Moments::default();
        let mut parameters = BTreeMap::new();
        for (site, group) in left {
            let other_group = &right[site];
            if group.is_empty() || group.keys().ne(other_group.keys()) {
                return Err("gradient alignment requires identical differentiated tensors".into());
            }
            let mut stats = BTreeMap::new();
            for (name, values) in group {
                let local = Moments::measure(values, &other_group[name])?;
                total.merge(&local)?;
                stats.insert(name.clone(), local.finish());
            }
            parameters.insert(site.clone(), stats);
        }
        Ok(CausalLmGradientAlignment {
            base_model_identity: self.base_model_identity().to_owned(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            geometry_artifact_id: self.geometry_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator(),
            left_target_tokens: self.target_tokens(),
            right_target_tokens: other.target_tokens(),
            total: total.finish(),
            parameters,
        })
    }
}

#[cfg(test)]
mod tests;
