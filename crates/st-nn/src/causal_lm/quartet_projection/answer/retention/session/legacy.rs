//! Read-only bridge for historical family-ablation checkpoints. Not a resume path.
use super::*;

const REPEATED: &str = "spiraltorch.native_conditional_answer_session.v2";
const FAMILY: &str = "spiraltorch.native_conditional_answer_session.v4";
const RETAINED: &str = "spiraltorch.native_conditional_retention.v2";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyHeader {
    schema: String,
    plan_id: String,
    guard_pin: String,
    pool_rows: usize,
    total_batches: u64,
    #[serde(default, deserialize_with = "present")]
    epochs: Option<u32>,
    position: u64,
    accepted_updates: u64,
    length_bits: u64,
    max_trials: usize,
    mass_inward_cosine_bits: Option<u64>,
    max_prefix_values: usize,
    #[serde(default, deserialize_with = "present")]
    family_size: Option<usize>,
}

fn present<'de, T: Deserialize<'de>, D: serde::Deserializer<'de>>(
    input: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(input).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyRetention {
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
    baseline: LegacyBaseline,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyBaseline {
    margin_bits: Vec<u64>,
    mean_pair_mass_bits: Option<u64>,
    minimum_mean_pair_mass_bits: Option<u64>,
    maximum_mean_outside_mass_bits: Option<u64>,
}

impl LegacyBaseline {
    fn validate(&self, drop: Option<u64>) -> CausalLmResult<()> {
        let margins = self.margin_bits.iter().copied().map(f64::from_bits);
        if !(1..=MAX_OUTPUT_CONTRAST_ROWS).contains(&self.margin_bits.len())
            || margins.clone().any(|v| !v.is_finite())
            || !margins.clone().any(|v| v > 0.)
        {
            return Err("invalid legacy conditional guard margin encoding".into());
        }
        let values = (
            drop.map(f64::from_bits),
            self.mean_pair_mass_bits.map(f64::from_bits),
            self.minimum_mean_pair_mass_bits.map(f64::from_bits),
            self.maximum_mean_outside_mass_bits.map(f64::from_bits),
        );
        match values {
            (None, None, None, None) => Ok(()),
            (Some(drop), Some(mean), Some(floor), outside)
                if (0.0..=1.0).contains(&drop)
                    && (0.0..=1.0).contains(&mean)
                    && (0.0..=mean).contains(&floor)
                    && (floor >= 0.5) == outside.is_some()
                    && outside.is_none_or(|v| (0.0..=1.0).contains(&v)) =>
            {
                Ok(())
            }
            _ => Err("invalid legacy conditional guard mass encoding".into()),
        }
    }
}

impl NativeCausalLmConditionalAnswerSession {
    /// Read weights from pinned historical v2 repeated and v4 family checkpoints
    /// containing a v2 retention record. Does not validate the training roster,
    /// reconstruct a guard, or allow training resume.
    pub fn serving_from_legacy_checkpoint(
        source: &NativeCausalLm,
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<NativeCausalLmConditionalServingSnapshot> {
        let (header, retained) = frame(bytes, expected_id)?;
        let header: LegacyHeader = serde_json::from_slice(header)?;
        let epochs = match (header.schema.as_str(), header.epochs, header.family_size) {
            (REPEATED, Some(n), None) if n > 1 => n,
            (FAMILY, epochs, Some(size))
                if (2..=MAX_OUTPUT_CONTRAST_ROWS).contains(&size)
                    && epochs.is_none_or(|n| n > 1) =>
            {
                epochs.unwrap_or(1)
            }
            _ => return Err("unsupported legacy conditional session format".into()),
        };
        let batches = usize::try_from(header.total_batches / u64::from(epochs))?;
        if Self::scheduled_positions(batches, epochs)? != header.total_batches
            || !(1..=MAX_CONDITIONAL_BANK_ROWS).contains(&header.pool_rows)
            || !(1..=MAX_PREPARED_PREFIX_VALUES).contains(&header.max_prefix_values)
            || header.position > header.total_batches
            || header.accepted_updates > header.position
            || !canonical_digest(&header.plan_id)
        {
            return Err("invalid legacy conditional session policy or cursor".into());
        }
        CausalLmConditionalAnswerConfig {
            length: f64::from_bits(header.length_bits),
            max_trials: header.max_trials,
            mass_inward_cosine: header.mass_inward_cosine_bits.map(f64::from_bits),
        }
        .validate()?;
        if !canonical_digest(&header.guard_pin)
            || retained.len() > NativeCausalLmRankOneProjection::MAX_RETENTION_CHECKPOINT_BYTES
            || fingerprint(retained) != header.guard_pin
        {
            return Err("legacy conditional retention checkpoint length or digest mismatch".into());
        }
        let state: LegacyRetention = serde_json::from_slice(retained)?;
        if state.schema != RETAINED
            || state.layer >= source.config.layers
            || !canonical_digest(&state.roster_id)
            || [
                &state.left_bits,
                &state.initial_right_bits,
                &state.right_bits,
            ]
            .iter()
            .any(|v| v.len() != source.config.width)
        {
            return Err("invalid legacy conditional retention schema or dimensions".into());
        }
        state.baseline.validate(state.max_mean_mass_drop_bits)?;
        let decode = |v: Vec<u64>| v.into_iter().map(f64::from_bits).collect::<Vec<_>>();
        let (left, initial, right) = (
            decode(state.left_bits),
            decode(state.initial_right_bits),
            decode(state.right_bits),
        );
        source.validate_output_left(Some(&left))?;
        if initial.iter().chain(&right).any(|v| !v.is_finite()) {
            return Err("nonfinite legacy conditional projection factors".into());
        }
        let site = state.site.parse()?;
        let layer = state.layer;
        let recipe = |v| NativeCausalLmRankOneProjection::recipe_id(source, layer, site, &left, v);
        if recipe(&initial) != state.initial_projection_id || recipe(&right) != state.projection_id
        {
            return Err(
                "legacy conditional checkpoint belongs to a different projection recipe".into(),
            );
        }
        let projection = NativeCausalLmRankOneProjection::new(source, layer, site, &left, &right)?;
        Ok(NativeCausalLmConditionalServingSnapshot {
            projection_id: state.projection_id,
            model: projection.current,
            checkpoint_id: expected_id.to_owned(),
            position: header.position,
            total_batches: header.total_batches,
        })
    }
}
