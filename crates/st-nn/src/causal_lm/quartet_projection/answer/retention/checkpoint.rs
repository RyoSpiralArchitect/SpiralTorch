//! Rebuild finite guards from their initial state and an externally pinned roster.
use super::*;

const SCHEMA: &str = "spiraltorch.native_conditional_retention.v3";

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Baseline {
    margin_bits: Vec<u64>,
    mass: Option<MassBaseline>,
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct MassBaseline {
    mean_pair_mass_bits: u64,
    minimum_mean_pair_mass_bits: u64,
    maximum_mean_outside_mass_bits: u64,
    minimum_log_mean_pair_mass_bits: Option<u64>,
}

impl Baseline {
    fn validate_encoding(&self, with_mass: bool) -> CausalLmResult<()> {
        let margins = self.margin_bits.iter().copied().map(f64::from_bits);
        if margins.clone().any(|m| !m.is_finite())
            || !margins.clone().any(|m| m > 0.)
            || with_mass != self.mass.is_some()
        {
            return Err("invalid conditional retention checkpoint baseline".into());
        }
        if let Some(mass) = &self.mass {
            let mean = f64::from_bits(mass.mean_pair_mass_bits);
            let minimum = f64::from_bits(mass.minimum_mean_pair_mass_bits);
            let outside = f64::from_bits(mass.maximum_mean_outside_mass_bits);
            let log_minimum = mass.minimum_log_mean_pair_mass_bits.map(f64::from_bits);
            if !(0.0..=1.0).contains(&mean)
                || !(0.0..=mean).contains(&minimum)
                || !(0.0..=1.0).contains(&outside)
                || log_minimum.is_some_and(|x| !x.is_finite() || x >= f64::MIN_POSITIVE.ln())
            {
                return Err("invalid conditional retention checkpoint mass floor".into());
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema: String,
    layer: usize,
    site: String,
    roster_id: String,
    initial_projection_id: String,
    projection_id: String,
    left_bits: Vec<u64>,
    initial_right_bits: Vec<u64>,
    right_bits: Vec<u64>,
    max_mean_mass_drop_bits: Option<u64>,
    baseline: Baseline,
}

/// Byte-admitted restore plan. It has performed no decoder work and
/// still requires the caller's original, ordered training roster.
pub struct CausalLmConditionalRetentionRestore {
    source: NativeCausalLm,
    plan: Record,
    spec: CausalLmProjectionSpec,
    left: Vec<f64>,
    initial_right: Vec<f64>,
    right: Vec<f64>,
    allowed_drop: Option<f64>,
}

pub(super) fn fingerprint(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn canonical_digest(id: &str) -> bool {
    id.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

pub(super) fn roster_id(prompts: &[Vec<usize>], pairs: &[[usize; 2]]) -> String {
    let mut h = Sha256::new();
    h.update(b"spiraltorch.conditional_retention_roster.v1\0");
    h.update((prompts.len() as u64).to_le_bytes());
    for (prompt, pair) in prompts.iter().zip(pairs) {
        h.update((prompt.len() as u64).to_le_bytes());
        for &token in prompt.iter().chain(pair) {
            h.update((token as u64).to_le_bytes());
        }
    }
    format!("sha256:{:x}", h.finalize())
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

impl CausalLmConditionalMarginRetention {
    fn checkpoint_baseline(&self) -> Baseline {
        Baseline {
            margin_bits: bits(&self.baseline),
            mass: self.mass.as_ref().map(|floor| MassBaseline {
                mean_pair_mass_bits: floor.baseline.to_bits(),
                minimum_mean_pair_mass_bits: floor.minimum.to_bits(),
                maximum_mean_outside_mass_bits: floor.maximum_outside.to_bits(),
                minimum_log_mean_pair_mass_bits: floor.log_pair_limit().map(f64::to_bits),
            }),
        }
    }
}

impl NativeCausalLmRankOneProjection {
    pub const MAX_RETENTION_CHECKPOINT_BYTES: usize = 1 << 20;

    /// Save current factors, the original guard recipe, and a digest of the
    /// ordered training roster. This includes no prompt IDs or cached activations.
    pub fn checkpoint_with_retention(
        &self,
        guard: &CausalLmConditionalMarginRetention,
    ) -> CausalLmResult<(Vec<u8>, String)> {
        guard.verify_owner(self)?;
        let payload = serde_json::to_vec(&Record {
            schema: SCHEMA.into(),
            layer: self.spec.layer,
            site: self.spec.site.as_str().into(),
            roster_id: guard.roster_id.clone(),
            initial_projection_id: guard.initial_projection_id.clone(),
            projection_id: self.projection_id(),
            left_bits: bits(&self.left),
            initial_right_bits: bits(&guard.initial_right),
            right_bits: bits(self.right()),
            max_mean_mass_drop_bits: guard.allowed_drop.map(f64::to_bits),
            baseline: guard.checkpoint_baseline(),
        })?;
        if payload.len() > Self::MAX_RETENTION_CHECKPOINT_BYTES {
            return Err("conditional retention checkpoint exceeds byte budget".into());
        }
        let pin = fingerprint(&payload);
        Ok((payload, pin))
    }

    /// Check an externally pinned checkpoint and its decoder recipes before
    /// copying a training roster or evaluating any of its rows.
    pub fn admit_retained_checkpoint(
        source: &NativeCausalLm,
        payload: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<CausalLmConditionalRetentionRestore> {
        if payload.len() > Self::MAX_RETENTION_CHECKPOINT_BYTES
            || !canonical_digest(expected_id)
            || fingerprint(payload) != expected_id
        {
            return Err("conditional retention checkpoint length or digest mismatch".into());
        }
        let plan: Record = serde_json::from_slice(payload)?;
        let w = source.config.width;
        if plan.schema != SCHEMA
            || plan.layer >= source.config.layers
            || [&plan.left_bits, &plan.initial_right_bits, &plan.right_bits]
                .iter()
                .any(|v| v.len() != w)
            || !(1..=MAX_OUTPUT_CONTRAST_ROWS).contains(&plan.baseline.margin_bits.len())
        {
            return Err("invalid conditional retention checkpoint schema or dimensions".into());
        }
        if !canonical_digest(&plan.roster_id) {
            return Err("invalid conditional retention checkpoint roster digest".into());
        }
        let site: CausalLmProjectionSite = plan.site.parse()?;
        let allowed_drop = plan.max_mean_mass_drop_bits.map(f64::from_bits);
        if allowed_drop.is_some_and(|drop| !(0.0..=1.0).contains(&drop)) {
            return Err("invalid conditional retention checkpoint mass allowance".into());
        }
        plan.baseline.validate_encoding(allowed_drop.is_some())?;
        let decode = |v: &[u64]| v.iter().copied().map(f64::from_bits).collect::<Vec<_>>();
        let left = decode(&plan.left_bits);
        let initial_right = decode(&plan.initial_right_bits);
        let right = decode(&plan.right_bits);
        source.validate_output_left(Some(&left))?;
        if initial_right.iter().chain(&right).any(|v| !v.is_finite()) {
            return Err("nonfinite conditional retention checkpoint factors".into());
        }
        if Self::recipe_id(source, plan.layer, site, &left, &initial_right)
            != plan.initial_projection_id
            || Self::recipe_id(source, plan.layer, site, &left, &right) != plan.projection_id
        {
            return Err(
                "conditional retention checkpoint belongs to a different projection recipe".into(),
            );
        }
        Ok(CausalLmConditionalRetentionRestore {
            spec: CausalLmProjectionSpec {
                layer: plan.layer,
                site,
                basis: CausalLmProjectionBasis::Diagonal,
            },
            source: source.clone(),
            plan,
            left,
            initial_right,
            right,
            allowed_drop,
        })
    }
}

impl CausalLmConditionalRetentionRestore {
    pub fn roster_len(&self) -> usize {
        self.plan.baseline.margin_bits.len()
    }

    /// Admit the pair metadata before a binding extracts any prompt tokens.
    /// The full ordered roster digest is still verified during restore.
    pub fn admit_roster_pairs(&self, pairs: &[[usize; 2]]) -> CausalLmResult<()> {
        if pairs.len() != self.roster_len() {
            return Err("conditional retention checkpoint roster length changed".into());
        }
        for &[gold, alternative] in pairs {
            self.source.check_contrast_candidates(gold, alternative)?;
        }
        Ok(())
    }

    /// Admit exactly the expected number of rows and their prefix workspace.
    pub fn roster_input_count(&self, steps: &[usize]) -> CausalLmResult<usize> {
        if steps.len() != self.roster_len() {
            return Err("conditional retention checkpoint roster length changed".into());
        }
        let tokens = steps
            .iter()
            .try_fold(0usize, |sum, &n| -> CausalLmResult<usize> {
                self.source
                    .projection_work_dimensions(n, self.spec, false)?;
                sum.checked_add(n)
                    .ok_or_else(|| "restored retention roster token overflow".into())
            })?;
        product(
            &[tokens, self.source.config.width],
            1 << 24,
            "restored retention prefix elements",
        )?;
        Ok(tokens)
    }

    /// Recompute the baseline, then check saved current weights against it.
    /// Repair or a new, weaker baseline is never attempted on mismatch.
    pub fn restore(
        self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
    ) -> CausalLmResult<(
        NativeCausalLmRankOneProjection,
        CausalLmConditionalMarginRetention,
    )> {
        if prompts.len() != self.roster_len() {
            return Err("conditional retention checkpoint roster length changed".into());
        }
        self.admit_roster_pairs(pairs)?;
        self.roster_input_count(&prompts.iter().map(Vec::len).collect::<Vec<_>>())?;
        if roster_id(prompts, pairs) != self.plan.roster_id {
            return Err("conditional retention checkpoint roster changed".into());
        }
        let mut state = NativeCausalLmRankOneProjection::new(
            &self.source,
            self.spec.layer,
            self.spec.site,
            &self.left,
            &self.initial_right,
        )?;
        let mut guard = state.prepare_conditional_retention(prompts, pairs, self.allowed_drop)?;
        if guard.checkpoint_baseline() != self.plan.baseline {
            return Err("conditional retention checkpoint baseline does not reproduce".into());
        }
        state.install(self.right)?;
        if state.projection_id() != self.plan.projection_id
            || !guard.check(&state)?.permits_commit()
        {
            return Err(
                "conditional retention checkpoint current state violates the saved guard".into(),
            );
        }
        guard.advance(state.projection_id());
        Ok((state, guard))
    }
}
