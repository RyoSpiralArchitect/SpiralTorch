//! Matched counterfactual supervision on a fixed-left, trainable-right attention projection.
use super::*;
use sha2::{Digest, Sha256};
use st_tensor::PackedLayout;
use std::collections::HashSet;

mod answer;
pub use answer::{
    CausalLmAnswerBinding, CausalLmAnswerCorrection, CausalLmAnswerGuard,
    CausalLmAnswerGuardStatus, CausalLmAnswerGuardStep, CausalLmAnswerPoint,
    CausalLmAnswerProjectionStep, CausalLmAnswerProtection, CausalLmAnswerTrial,
    CausalLmLossGradient,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmQuartetObjective {
    /// Average the four diagonal-versus-adjacent off-diagonal logistic losses.
    Pairwise,
    /// Apply logistic loss to (aa - ab - ba + bb) / 2. Additive row and column
    /// effects cancel *before* the nonlinearity, not just in the mean margin.
    Interaction,
}

impl CausalLmQuartetObjective {
    fn mean_seed(self, gaps: &[f64], groups: &[[usize; 4]]) -> (f64, Vec<f64>) {
        let mut coefficients: Vec<_> = (0..gaps.len()).map(|_| sum::Sum::default()).collect();
        let mut total = sum::Sum::default();
        for group in groups {
            let (loss, deriv) = self.evaluate(group.map(|i| gaps[i]));
            total.add(loss);
            for (&i, g) in group.iter().zip(deriv) {
                coefficients[i].add(g);
            }
        }
        let n = groups.len() as f64;
        (
            total.value() / n,
            coefficients.into_iter().map(|s| s.value() / n).collect(),
        )
    }
    fn evaluate(self, [a, b, c, d]: [f64; 4]) -> (f64, [f64; 4]) {
        use scalar_output::logistic;
        match self {
            Self::Interaction => {
                let (loss, g) = logistic(((a - b) + (d - c)) * 0.5);
                (loss, [g * 0.5, -g * 0.5, -g * 0.5, g * 0.5])
            }
            Self::Pairwise => {
                let [(l0, g0), (l1, g1), (l2, g2), (l3, g3)] =
                    [a - c, d - b, a - b, d - c].map(logistic);
                (
                    (l0 + l1 + l2 + l3) * 0.25,
                    [
                        (g0 + g2) * 0.25,
                        -(g1 + g2) * 0.25,
                        -(g0 + g3) * 0.25,
                        (g1 + g3) * 0.25,
                    ],
                )
            }
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmQuartetStep {
    pub loss: f64,
    pub gradient: Vec<f64>,
    pub gradient_norm: f64,
    pub prompt_evaluations: usize,
    pub quartets: usize,
}

fn validate_groups(prompts: &[Vec<usize>], groups: &[[usize; 4]]) -> CausalLmResult<()> {
    if !(1..=64).contains(&groups.len()) {
        return Err("invalid quartet count".into());
    }
    let mut used = vec![false; prompts.len()];
    for group in groups {
        for (j, &i) in group.iter().enumerate() {
            if i >= used.len() || group[..j].contains(&i) {
                return Err("invalid quartet indices".into());
            }
            used[i] = true;
        }
    }
    if used.contains(&false) || prompts.iter().collect::<HashSet<_>>().len() != prompts.len() {
        return Err("unused or duplicated quartet prompt".into());
    }
    Ok(())
}

/// Learns W' = round_f32(W + left * right^T) in one attention site. The private
/// model is always repacked from the immutable origin, never from a rounded update.
#[derive(Clone, Debug)]
pub struct NativeCausalLmRankOneProjection {
    origin: Arc<NativeCausalLm>,
    current: NativeCausalLm,
    spec: CausalLmProjectionSpec,
    left: Arc<[f64]>,
    right: Vec<f64>,
}

/// A branchable serving prefix bound to one exact projection recipe. A training
/// update never changes an already-prefilled prefix.
#[derive(Clone, Debug)]
pub struct NativeCausalLmProjectionPrefix {
    owner: Arc<str>,
    cache: NativeCausalLmContext,
}

impl NativeCausalLmProjectionPrefix {
    pub(super) fn from_snapshot(
        model: &NativeCausalLm,
        owner: String,
        prompt: Vec<usize>,
        max_cache_values: usize,
    ) -> CausalLmResult<Self> {
        Ok(Self {
            owner: owner.into(),
            cache: NativeCausalLmContext::new(model.clone(), prompt, max_cache_values)?,
        })
    }
    pub fn projection_id(&self) -> &str {
        &self.owner
    }
    pub fn token_ids(&self) -> &[usize] {
        self.cache.token_ids()
    }
    pub fn cache_values(&self) -> usize {
        self.cache.cache_values()
    }
    pub fn max_cache_values(&self) -> usize {
        self.cache.max_cache_values()
    }
    pub fn logits(&self) -> &[f32] {
        self.cache.logits_tensor().data()
    }
    /// Advance a new branch without replaying or altering the common prefix.
    pub fn branch(&self, token: usize) -> CausalLmResult<Self> {
        Ok(Self {
            owner: self.owner.clone(),
            cache: self.cache.preview(token)?,
        })
    }
    /// One raw projected key at its source position, concatenated in head order.
    /// A lagged block can reuse this key at subsequent visible value slots.
    pub fn key_row(&self, layer: usize, position: usize) -> CausalLmResult<Vec<f32>> {
        self.cache.key_row(layer, position)
    }
    /// Return suffix logits after substituting only the past key at a matched
    /// position. Past values, other keys and causal geometry histories start
    /// factual. Descendant states can change; neither original prefix changes.
    pub fn probe_key_from(
        &self,
        donor: &Self,
        layer: usize,
        position: usize,
        suffix: &[usize],
    ) -> CausalLmResult<Vec<f32>> {
        if self.owner != donor.owner {
            return Err("different projection snapshot in key probe".into());
        }
        self.cache
            .probe_key_from(&donor.cache, layer, position, suffix)
    }
}

impl NativeCausalLmRankOneProjection {
    pub fn new(
        model: &NativeCausalLm,
        layer: usize,
        site: CausalLmProjectionSite,
        left: &[f64],
        right: &[f64],
    ) -> CausalLmResult<Self> {
        let w = model.config.width;
        if layer >= model.config.layers || right.len() != w || right.iter().any(|x| !x.is_finite())
        {
            return Err("invalid rank-one projection layer or right factor".into());
        }
        model.validate_output_left(Some(left))?;
        product(
            &[w, w, if site.qkv_part().is_some() { 3 } else { 1 }],
            (1 << 24).min(model.limits.max_parameter_elements),
            "rank-one packed projection",
        )?;
        let mut state = Self {
            origin: Arc::new(model.clone()),
            current: model.clone(),
            left: left.into(),
            // Contracting with a fixed left vector has diagonal-sized capture cost.
            spec: CausalLmProjectionSpec {
                layer,
                site,
                basis: CausalLmProjectionBasis::Diagonal,
            },
            right: vec![0.; w],
        };
        state.install(right.to_vec())?;
        Ok(state)
    }

    fn install(&mut self, right: Vec<f64>) -> CausalLmResult<()> {
        if right.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite rank-one update".into());
        }
        if right == self.right {
            return Ok(());
        }
        let mut next = (*self.origin).clone();
        if right.iter().any(|&x| x != 0.) {
            let w = self.left.len();
            let block = &mut next.blocks[self.spec.layer];
            let (dense, cols, part) = match self.spec.site.qkv_part() {
                Some(p) => (&mut block.qkv, 3 * w, p),
                None => (&mut block.output, w, 0),
            };
            if dense.weight.layout() != PackedLayout::ColMajor
                || (dense.weight.inner(), dense.weight.cols()) != (w, cols)
            {
                return Err("unexpected packed projection shape".into());
            }
            let mut packed = dense.weight.as_slice().to_vec();
            for (col, &v) in packed[part * w * w..(part + 1) * w * w]
                .chunks_exact_mut(w)
                .zip(&right)
            {
                for (x, &u) in col.iter_mut().zip(self.left.iter()) {
                    let delta = u * v;
                    if delta != 0. {
                        *x = (f64::from(*x) + delta) as f32;
                        if !x.is_finite() {
                            return Err("nonfinite merged projection weight".into());
                        }
                    }
                }
            }
            dense.weight = PackedB::from_tensor_transpose(
                &Tensor::from_vec(cols, w, packed)?,
                dense.weight.tile(),
            )?;
            dense.backward_weight = Default::default();
        }
        self.current = next;
        self.right = right;
        Ok(())
    }

    pub fn right(&self) -> &[f64] {
        &self.right
    }

    /// Identity of the exact serving recipe, distinct from the frozen base ID.
    pub fn projection_id(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"spiraltorch.rank_one_projection_prefix.v1\0");
        for s in [
            self.origin.base_model_identity(),
            self.origin.lora_artifact_id().unwrap_or(""),
            self.origin.geometry_artifact_id().unwrap_or(""),
            self.origin.readout_accumulator().as_str(),
            self.spec.site.as_str(),
        ] {
            h.update((s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        }
        h.update((self.spec.layer as u64).to_le_bytes());
        h.update((self.left.len() as u64).to_le_bytes());
        for v in self.left.iter().chain(&self.right) {
            h.update(v.to_bits().to_le_bytes());
        }
        format!("sha256:{:x}", h.finalize())
    }

    pub fn prefill(
        &self,
        prompt: Vec<usize>,
        max_cache_values: usize,
    ) -> CausalLmResult<NativeCausalLmProjectionPrefix> {
        NativeCausalLmProjectionPrefix::from_snapshot(
            &self.current,
            self.projection_id(),
            prompt,
            max_cache_values,
        )
    }

    pub fn last_logits(&self, prompt: &[usize]) -> CausalLmResult<Vec<f32>> {
        Ok(self
            .current
            .logits(prompt, [1, prompt.len()], true)?
            .data()
            .to_vec())
    }

    pub fn contrast_gradient(
        &self,
        prompt: &[usize],
        ids: [usize; 2],
    ) -> CausalLmResult<(CausalLmTokenContrast, Vec<f64>)> {
        self.current
            .fixed_left_contrast(prompt, ids[0], ids[1], self.spec, &self.left)
    }

    /// One normalized descent step. Quartets index the unique prompt roster in
    /// [aa, ab, ba, bb] order. Reusing a quartet changes its objective weight,
    /// not the number of decoder VJPs. Any error leaves the projection unchanged.
    pub fn step_quartets(
        &mut self,
        prompts: &[Vec<usize>],
        ids: [usize; 2],
        quartets: &[[usize; 4]],
        objective: CausalLmQuartetObjective,
        length: f64,
    ) -> CausalLmResult<CausalLmQuartetStep> {
        if !(1..=MAX_OUTPUT_CONTRAST_ROWS).contains(&prompts.len())
            || !(1..=64).contains(&quartets.len())
            || !length.is_finite()
            || length <= 0.
        {
            return Err("invalid quartet roster, count or step length".into());
        }
        self.current.validate_output_roster_steps(
            self.spec.layer,
            &prompts.iter().map(Vec::len).collect::<Vec<_>>(),
            ids[0],
            ids[1],
            true,
        )?;
        validate_groups(prompts, quartets)?;
        if prompts
            .iter()
            .flatten()
            .any(|&x| x >= self.current.config.vocabulary)
        {
            return Err("unused, duplicated or invalid quartet prompt".into());
        }
        let rows = prompts
            .iter()
            .map(|p| self.contrast_gradient(p, ids))
            .collect::<CausalLmResult<Vec<_>>>()?;
        let (loss, coefficients) = objective.mean_seed(
            &rows
                .iter()
                .map(|r| r.0.target_minus_alternative)
                .collect::<Vec<_>>(),
            quartets,
        );
        let mut grad: Vec<_> = (0..self.right.len()).map(|_| sum::Sum::default()).collect();
        for (v, (_, row)) in coefficients.into_iter().zip(rows) {
            for (s, g) in grad.iter_mut().zip(row) {
                s.add(v * g);
            }
        }
        let gradient: Vec<_> = grad.iter().map(sum::Sum::value).collect();
        let norm = gradient.iter().fold(0.0_f64, |a, &g| a.hypot(g));
        if !loss.is_finite() || !norm.is_finite() {
            return Err("nonfinite quartet objective or gradient".into());
        }
        self.descend(&gradient, norm, length)?;
        Ok(CausalLmQuartetStep {
            loss,
            gradient,
            gradient_norm: norm,
            prompt_evaluations: prompts.len(),
            quartets: quartets.len(),
        })
    }

    fn descend(&mut self, gradient: &[f64], norm: f64, length: f64) -> CausalLmResult<()> {
        if norm > 0. {
            self.install(
                self.right
                    .iter()
                    .zip(gradient)
                    .map(|(&v, &g)| v - length * (g / norm))
                    .collect(),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interaction_removes_main_effects_before_the_logistic() {
        let x = [3., -0.5, 1., 2.];
        let moved = [4., 4.5, -1., 4.]; // additive row [1,-2], column [0,4]
        let (loss, g) = CausalLmQuartetObjective::Interaction.evaluate(x);
        assert_eq!(
            (loss, g),
            CausalLmQuartetObjective::Interaction.evaluate(moved)
        );
        assert_ne!(
            CausalLmQuartetObjective::Pairwise.evaluate(x).0,
            CausalLmQuartetObjective::Pairwise.evaluate(moved).0
        );
        for obj in [
            CausalLmQuartetObjective::Interaction,
            CausalLmQuartetObjective::Pairwise,
        ] {
            let (_, g) = obj.evaluate(x);
            for j in 0..4 {
                let mut a = x;
                let mut b = x;
                a[j] += 0.00001;
                b[j] -= 0.00001;
                assert!(((obj.evaluate(a).0 - obj.evaluate(b).0) / 0.00002 - g[j]).abs() < 1e-10);
            }
        }
    }
}
