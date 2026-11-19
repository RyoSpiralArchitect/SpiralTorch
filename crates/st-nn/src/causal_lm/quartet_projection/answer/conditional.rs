//! Optimize two-candidate choice itself, optionally retaining candidate mass.
use super::observation::candidate_mass_loss;
use super::*;
use crate::causal_lm::scalar_output::logistic;
use std::sync::Weak;

const MAX_PREPARED_PREFIX_VALUES: usize = 1 << 24;
/// Maximum number of distinct prompts in an in-memory conditional prefix bank.
pub const MAX_CONDITIONAL_BANK_ROWS: usize = 256;

struct ConditionalPrefix {
    steps: usize,
    hidden: Tensor,
}

struct ConditionalAnswerRow {
    pair: [usize; 2],
    prefix: ConditionalPrefix,
}

/// Bounded label-free roster of immutable pre-projection prompt states.
/// Row identities are positional and valid only within this bank.
pub struct CausalLmConditionalAnswerPrefixBank {
    origin: Weak<NativeCausalLm>,
    spec: CausalLmProjectionSpec,
    rows: Vec<ConditionalPrefix>,
    prefix_values: usize,
}

impl CausalLmConditionalAnswerPrefixBank {
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Retained f32 values, excluding metadata and temporary workspace.
    pub fn prefix_values(&self) -> usize {
        self.prefix_values
    }
}

/// Prompt prefixes strictly before the trainable projection. Bound to one
/// in-memory projection origin; usable across right-factor updates and clones.
pub struct CausalLmPreparedConditionalAnswerBatch {
    batch: PreparedConditionalRows,
    prefix_values: usize,
}

struct PreparedConditionalRows {
    origin: Weak<NativeCausalLm>,
    spec: CausalLmProjectionSpec,
    rows: Vec<ConditionalAnswerRow>,
    tokens: usize,
}

impl CausalLmPreparedConditionalAnswerBatch {
    pub fn len(&self) -> usize {
        self.batch.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batch.rows.is_empty()
    }

    /// Total retained f32 values (one width-sized row per input token).
    pub fn prefix_values(&self) -> usize {
        self.prefix_values
    }
}

enum ConditionalAnswerBatch<'a> {
    Raw(&'a [Vec<usize>], &'a [[usize; 2]]),
    Prepared(&'a PreparedConditionalRows),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalLmConditionalAnswerConfig {
    /// Initial normalized right-factor step length; trials halve it.
    pub length: f64,
    pub max_trials: usize,
    /// None: direct conditional descent. Some: require finite-batch candidate
    /// mass non-regression and aim this far into its first-order descent cone.
    pub mass_inward_cosine: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CausalLmConditionalAnswerPoint {
    pub conditional_ce: f64,
    /// Mean -log(P(gold)+P(alternative)), not minus log of the batch mean.
    pub candidate_mass_nll: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalAnswerTrial {
    pub point: CausalLmConditionalAnswerPoint,
    pub projection_id: String,
    pub distance: f64,
    pub required_conditional_drop: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmConditionalAnswerStep {
    pub before: CausalLmConditionalAnswerPoint,
    pub after: CausalLmConditionalAnswerPoint,
    pub trials: Vec<CausalLmConditionalAnswerTrial>,
    pub committed: bool,
    pub projected: bool,
    pub conditional_gradient_l2: f64,
    pub mass_gradient_l2: Option<f64>,
    pub gradient_forwards: usize,
    pub projection_pullbacks: usize,
    pub trial_forwards: usize,
    /// Frozen block calls skipped by replaying only the trial suffix.
    pub reused_prefix_block_calls: usize,
    /// Additional frozen block calls skipped in this step's gradient forward.
    /// Preparation itself is not included.
    #[serde(skip_serializing_if = "is_zero")]
    pub reused_prepared_prefix_block_calls: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

impl CausalLmConditionalAnswerConfig {
    fn validate(self) -> CausalLmResult<()> {
        if !self.length.is_finite()
            || self.length <= 0.
            || !(1..=8).contains(&self.max_trials)
            || self
                .mass_inward_cosine
                .is_some_and(|x| !(0.0..1.0).contains(&x))
        {
            return Err("invalid conditional answer step configuration".into());
        }
        Ok(())
    }
}

fn norm(xs: &[f64]) -> f64 {
    xs.iter().fold(0.0_f64, |n, &x| n.hypot(x))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut result = sum::Sum::default();
    for (&x, &y) in a.iter().zip(b) {
        result.add(x * y);
    }
    result.value()
}

/// Return a unit gradient-space direction (the actual step subtracts it).
fn direction(choice: &[f64], mass: Option<(&[f64], f64)>) -> Option<(Vec<f64>, bool)> {
    let n = norm(choice);
    if n == 0. || !n.is_finite() {
        return None;
    }
    let choice: Vec<_> = choice.iter().map(|v| v / n).collect();
    let Some((mass, inward)) = mass else {
        return Some((choice, false));
    };
    let n = norm(mass);
    if n == 0. {
        return Some((choice, false));
    }
    let mass: Vec<_> = mass.iter().map(|v| v / n).collect();
    let cos = dot(&choice, &mass);
    if cos >= inward {
        return Some((choice, false));
    }
    let tangent: Vec<_> = choice.iter().zip(&mass).map(|(c, m)| c - cos * m).collect();
    let n = norm(&tangent);
    if n <= 1e-12 {
        return None;
    }
    let scale = (1. - inward * inward).sqrt() / n;
    let projected: Vec<_> = tangent
        .iter()
        .zip(mass)
        .map(|(t, m)| scale * t + inward * m)
        .collect();
    (dot(&choice, &projected) > 0.).then_some((projected, true))
}

#[derive(Default)]
struct PointSum {
    ce: sum::Sum,
    mass: sum::Sum,
}

impl PointSum {
    fn add(&mut self, ce: f64, mass: f64) {
        self.ce.add(ce);
        self.mass.add(mass);
    }
    fn finish(self, n: usize) -> CausalLmConditionalAnswerPoint {
        CausalLmConditionalAnswerPoint {
            conditional_ce: self.ce.mean(n as u64),
            candidate_mass_nll: self.mass.mean(n as u64),
        }
    }
}

impl NativeCausalLmRankOneProjection {
    /// Admit a labelled roster's lengths before a binding reads its token IDs.
    pub fn conditional_answer_input_count(
        &self,
        steps: &[usize],
        backward: bool,
    ) -> CausalLmResult<usize> {
        if !(1..=MAX_OUTPUT_CONTRAST_ROWS).contains(&steps.len()) {
            return Err("invalid conditional answer roster length".into());
        }
        product(
            &[steps.len(), self.current.config.width],
            output_margin::MAX_BANK_VALUES,
            "conditional answer roster",
        )?;
        let tokens = self.conditional_prompt_tokens(steps.iter().copied(), backward)?;
        self.admit_conditional_capture(tokens, false)?;
        Ok(tokens)
    }

    /// Admit the step's configuration and actual number of pullbacks before token copies.
    pub fn conditional_answer_step_input_count(
        &self,
        steps: &[usize],
        config: CausalLmConditionalAnswerConfig,
    ) -> CausalLmResult<usize> {
        config.validate()?;
        let tokens = self.conditional_answer_input_count(steps, true)?;
        self.admit_conditional_capture(tokens, config.mass_inward_cosine.is_some())?;
        Ok(tokens)
    }

    fn admit_conditional_capture(&self, tokens: usize, guarded: bool) -> CausalLmResult<()> {
        product(
            &[tokens, self.current.config.width, 1 + usize::from(guarded)],
            output_margin::MAX_CAPTURE_TERMS,
            "conditional answer capture work",
        )?;
        Ok(())
    }

    /// Admit a prefix bank's storage and lengths before reading token IDs.
    pub fn conditional_answer_bank_values(
        &self,
        steps: &[usize],
        max_prefix_values: usize,
    ) -> CausalLmResult<usize> {
        self.conditional_bank_values(steps.len(), steps.iter().copied(), max_prefix_values)
    }

    fn conditional_prompt_tokens(
        &self,
        mut steps: impl Iterator<Item = usize>,
        backward: bool,
    ) -> CausalLmResult<usize> {
        steps.try_fold(0usize, |total, n| {
            self.current
                .projection_work_dimensions(n, self.spec, backward)?;
            total
                .checked_add(n)
                .ok_or_else(|| "conditional prompt token overflow".into())
        })
    }

    fn conditional_bank_values(
        &self,
        count: usize,
        steps: impl Iterator<Item = usize>,
        max_prefix_values: usize,
    ) -> CausalLmResult<usize> {
        if !(1..=MAX_CONDITIONAL_BANK_ROWS).contains(&count) {
            return Err("conditional answer prefix bank requires 1 to 256 distinct prompts".into());
        }
        product(
            &[
                self.conditional_prompt_tokens(steps, true)?,
                self.current.config.width,
            ],
            max_prefix_values.min(MAX_PREPARED_PREFIX_VALUES),
            "conditional answer prefix bank values",
        )
    }

    fn conditional_prefix(&self, prompt: &[usize]) -> CausalLmResult<ConditionalPrefix> {
        let steps = prompt.len();
        let shape = [1, steps];
        let m = &self.current;
        let hidden = m.forward_blocks(m.embed(prompt, shape)?, shape, 0..self.spec.layer)?;
        Ok(ConditionalPrefix { steps, hidden })
    }

    /// Prepare distinct prompts in caller order, without committing to any
    /// answer labels. The caller controls the storage cap in f32 values;
    /// admission is completed before any decoder work.
    pub fn prepare_conditional_answer_prefix_bank(
        &self,
        prompts: &[Vec<usize>],
        max_prefix_values: usize,
    ) -> CausalLmResult<CausalLmConditionalAnswerPrefixBank> {
        let m = &self.current;
        let prefix_values = self.conditional_bank_values(
            prompts.len(),
            prompts.iter().map(Vec::len),
            max_prefix_values,
        )?;
        if prompts
            .iter()
            .flatten()
            .any(|&id| id >= m.config.vocabulary)
        {
            return Err("invalid prefix bank prompt token".into());
        }
        if prompts.iter().collect::<HashSet<_>>().len() != prompts.len() {
            return Err("duplicate prompt in conditional answer prefix bank".into());
        }
        let rows = prompts
            .iter()
            .map(|prompt| self.conditional_prefix(prompt))
            .collect::<CausalLmResult<_>>()?;
        Ok(CausalLmConditionalAnswerPrefixBank {
            origin: Arc::downgrade(&self.origin),
            spec: self.spec,
            rows,
            prefix_values,
        })
    }

    fn verify_conditional_origin(
        &self,
        origin: &Weak<NativeCausalLm>,
        spec: CausalLmProjectionSpec,
    ) -> CausalLmResult<()> {
        if spec != self.spec || !Weak::ptr_eq(origin, &Arc::downgrade(&self.origin)) {
            return Err(
                "prepared conditional answer data belongs to a different projection origin".into(),
            );
        }
        Ok(())
    }

    fn select_conditional_rows(
        &self,
        bank: &CausalLmConditionalAnswerPrefixBank,
        indices: &[usize],
        pairs: &[[usize; 2]],
    ) -> CausalLmResult<PreparedConditionalRows> {
        self.verify_conditional_origin(&bank.origin, bank.spec)?;
        let m = &self.current;
        if indices.len() != pairs.len() || !(1..=MAX_OUTPUT_CONTRAST_ROWS).contains(&indices.len())
        {
            return Err("invalid indexed conditional answer roster".into());
        }
        product(
            &[indices.len(), m.config.width],
            output_margin::MAX_BANK_VALUES,
            "indexed conditional answer roster",
        )?;
        let mut tokens = 0usize;
        let mut rows = Vec::with_capacity(indices.len());
        for (&index, &pair) in indices.iter().zip(pairs) {
            m.check_contrast_candidates(pair[0], pair[1])?;
            let row = bank
                .rows
                .get(index)
                .ok_or("conditional answer bank index out of range")?;
            tokens = tokens
                .checked_add(row.steps)
                .ok_or("indexed conditional answer token overflow")?;
            rows.push(ConditionalAnswerRow {
                pair,
                prefix: ConditionalPrefix {
                    steps: row.steps,
                    hidden: row.hidden.clone(),
                },
            });
        }
        Ok(PreparedConditionalRows {
            origin: bank.origin.clone(),
            spec: bank.spec,
            rows,
            tokens,
        })
    }

    /// Use rows in the supplied order, including repeated indices. Each
    /// occurrence contributes to the batch loss with its own ordered pair.
    pub fn step_indexed_conditional_answers(
        &mut self,
        bank: &CausalLmConditionalAnswerPrefixBank,
        indices: &[usize],
        pairs: &[[usize; 2]],
        config: CausalLmConditionalAnswerConfig,
    ) -> CausalLmResult<CausalLmConditionalAnswerStep> {
        let batch = self.select_conditional_rows(bank, indices, pairs)?;
        self.conditional_step(ConditionalAnswerBatch::Prepared(&batch), config, None)
            .map(|(r, _)| r)
    }

    /// Indexed equivalent of a finite retained update. The bank does not
    /// store or consume retention decisions.
    pub fn step_indexed_conditional_answers_retaining(
        &mut self,
        bank: &CausalLmConditionalAnswerPrefixBank,
        indices: &[usize],
        pairs: &[[usize; 2]],
        config: CausalLmConditionalAnswerConfig,
        retention: &mut CausalLmConditionalMarginRetention,
    ) -> CausalLmResult<CausalLmRetainedConditionalAnswerStep> {
        let batch = self.select_conditional_rows(bank, indices, pairs)?;
        self.conditional_step_retaining(ConditionalAnswerBatch::Prepared(&batch), config, retention)
    }

    /// Admit an entire labelled batch before allocating bounded prefix state.
    /// Only the embedding and blocks strictly before the modified layer are
    /// captured; the latest suffix, logits and gradients are still evaluated
    /// on every update. The limit is per prepared batch, not per session.
    pub fn prepare_conditional_answer_batch(
        &self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
        max_prefix_values: usize,
    ) -> CausalLmResult<CausalLmPreparedConditionalAnswerBatch> {
        let tokens = self.admit_conditional_batch(prompts, pairs)?;
        let m = &self.current;
        let prefix_values = product(
            &[tokens, m.config.width],
            max_prefix_values.min(MAX_PREPARED_PREFIX_VALUES),
            "prepared conditional answer prefix values",
        )?;
        let rows = prompts
            .iter()
            .zip(pairs)
            .map(|(prompt, &pair)| {
                Ok(ConditionalAnswerRow {
                    pair,
                    prefix: self.conditional_prefix(prompt)?,
                })
            })
            .collect::<CausalLmResult<Vec<_>>>()?;
        Ok(CausalLmPreparedConditionalAnswerBatch {
            batch: PreparedConditionalRows {
                origin: Arc::downgrade(&self.origin),
                spec: self.spec,
                rows,
                tokens,
            },
            prefix_values,
        })
    }

    /// Conditional descent with the same finite tests as the uncached step.
    pub fn step_prepared_conditional_answers(
        &mut self,
        prepared: &CausalLmPreparedConditionalAnswerBatch,
        config: CausalLmConditionalAnswerConfig,
    ) -> CausalLmResult<CausalLmConditionalAnswerStep> {
        self.conditional_step(
            ConditionalAnswerBatch::Prepared(&prepared.batch),
            config,
            None,
        )
        .map(|(r, _)| r)
    }

    /// Combine a prepared update batch with an independently owned retention
    /// guard. Preparing the batch does not reset or bypass the guard.
    pub fn step_prepared_conditional_answers_retaining(
        &mut self,
        prepared: &CausalLmPreparedConditionalAnswerBatch,
        config: CausalLmConditionalAnswerConfig,
        retention: &mut CausalLmConditionalMarginRetention,
    ) -> CausalLmResult<CausalLmRetainedConditionalAnswerStep> {
        self.conditional_step_retaining(
            ConditionalAnswerBatch::Prepared(&prepared.batch),
            config,
            retention,
        )
    }

    /// Fit conditional CE on explicit [gold, alternative] pairs. Commit only a
    /// measured decrease in conditional CE and, when guarded, non-increase in
    /// mass loss. Guarantees cover the supplied batch, not other prompts.
    /// An error or exhausted search never changes the serving identity.
    pub fn step_conditional_answers(
        &mut self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
        config: CausalLmConditionalAnswerConfig,
    ) -> CausalLmResult<CausalLmConditionalAnswerStep> {
        self.conditional_step(ConditionalAnswerBatch::Raw(prompts, pairs), config, None)
            .map(|(r, _)| r)
    }

    /// As for ordinary conditional learning, with an additional finite
    /// sign check on every anchored prompt and an optional mean-mass check
    /// on the finite roster before a proposal may commit. The roster limits
    /// acceptance, not the construction of the mini-batch search direction.
    pub fn step_conditional_answers_retaining(
        &mut self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
        config: CausalLmConditionalAnswerConfig,
        retention: &mut CausalLmConditionalMarginRetention,
    ) -> CausalLmResult<CausalLmRetainedConditionalAnswerStep> {
        self.conditional_step_retaining(
            ConditionalAnswerBatch::Raw(prompts, pairs),
            config,
            retention,
        )
    }

    fn conditional_step_retaining(
        &mut self,
        batch: ConditionalAnswerBatch<'_>,
        config: CausalLmConditionalAnswerConfig,
        retention: &mut CausalLmConditionalMarginRetention,
    ) -> CausalLmResult<CausalLmRetainedConditionalAnswerStep> {
        retention.verify_owner(self)?;
        let (step, checks) = self.conditional_step(batch, config, Some(retention))?;
        if step.committed {
            retention.advance(self.projection_id());
        }
        Ok(CausalLmRetainedConditionalAnswerStep {
            step,
            retention: checks,
        })
    }

    fn admit_conditional_batch(
        &self,
        prompts: &[Vec<usize>],
        pairs: &[[usize; 2]],
    ) -> CausalLmResult<usize> {
        let m = &self.current;
        if prompts.len() != pairs.len() || pairs.len() > MAX_OUTPUT_CONTRAST_ROWS {
            return Err("invalid conditional answer roster".into());
        }
        for &[y, a] in pairs {
            m.check_contrast_candidates(y, a)?;
        }
        let gold: Vec<_> = pairs.iter().map(|p| p[0]).collect();
        self.admit_answers(prompts, &gold, None, true)?;
        Ok(prompts.iter().map(Vec::len).sum())
    }

    fn conditional_step(
        &mut self,
        batch: ConditionalAnswerBatch<'_>,
        config: CausalLmConditionalAnswerConfig,
        retention: Option<&CausalLmConditionalMarginRetention>,
    ) -> CausalLmResult<(
        CausalLmConditionalAnswerStep,
        Vec<Option<CausalLmConditionalRetentionCheck>>,
    )> {
        config.validate()?;
        let (n, tokens, from_cache) = match &batch {
            ConditionalAnswerBatch::Raw(prompts, pairs) => (
                prompts.len(),
                self.admit_conditional_batch(prompts, pairs)?,
                false,
            ),
            ConditionalAnswerBatch::Prepared(prepared) => {
                self.verify_conditional_origin(&prepared.origin, prepared.spec)?;
                (prepared.rows.len(), prepared.tokens, true)
            }
        };
        let m = &self.current;
        let guarded = config.mass_inward_cosine.is_some();
        let w = m.config.width;
        self.admit_conditional_capture(tokens, guarded)?;

        let mut point = PointSum::default();
        let mut choice: Vec<_> = (0..w).map(|_| sum::Sum::default()).collect();
        let mut mass = guarded.then(|| (0..w).map(|_| sum::Sum::default()).collect::<Vec<_>>());
        let first = self.spec.layer;
        let mut computed = if from_cache {
            Vec::new()
        } else {
            Vec::with_capacity(n)
        };
        for i in 0..n {
            let cached = match &batch {
                ConditionalAnswerBatch::Raw(prompts, pairs) => {
                    computed.push(ConditionalAnswerRow {
                        pair: pairs[i],
                        prefix: self.conditional_prefix(&prompts[i])?,
                    });
                    &computed[i]
                }
                ConditionalAnswerBatch::Prepared(batch) => &batch.rows[i],
            };
            let [y, a] = cached.pair;
            let row = m.projection_last_row_from_hidden(
                cached.prefix.hidden.clone(),
                cached.prefix.steps,
                first,
                self.spec,
            )?;
            let z = row.logits.data();
            let (ce, slope) = logistic(f64::from(z[y]) - f64::from(z[a]));
            let mass_loss = if let Some(out) = &mut mass {
                let (loss, gradient) = self.candidate_mass_gradient(row.clone(), [y, a])?;
                for (s, g) in out.iter_mut().zip(gradient) {
                    s.add(g);
                }
                loss
            } else {
                candidate_mass_loss(z, [y, a])?
            };
            point.add(ce, mass_loss);
            let mut v = vec![0.; z.len()];
            v[y] = 1.;
            v[a] = -1.;
            let seed = Tensor::from_vec(1, z.len(), v)?;
            for (s, g) in choice
                .iter_mut()
                .zip(m.fixed_left_seed(row, &seed, self.spec, &self.left)?)
            {
                s.add(slope * g);
            }
        }
        let rows = match batch {
            ConditionalAnswerBatch::Raw(..) => &computed,
            ConditionalAnswerBatch::Prepared(batch) => &batch.rows,
        };
        let mean = |s: Vec<sum::Sum>| s.into_iter().map(|v| v.mean(n as u64)).collect::<Vec<_>>();
        let choice = mean(choice);
        let mass = mass.map(mean);
        let before = point.finish(n);
        let mut result = CausalLmConditionalAnswerStep {
            before,
            after: before,
            trials: Vec::new(),
            committed: false,
            projected: false,
            conditional_gradient_l2: norm(&choice),
            mass_gradient_l2: mass.as_ref().map(|v| norm(v)),
            gradient_forwards: n,
            projection_pullbacks: n * (1 + usize::from(guarded)),
            trial_forwards: 0,
            reused_prefix_block_calls: 0,
            reused_prepared_prefix_block_calls: if from_cache { n * first } else { 0 },
        };
        if !result.conditional_gradient_l2.is_finite()
            || result.mass_gradient_l2.is_some_and(|x| !x.is_finite())
            || !before.conditional_ce.is_finite()
            || !before.candidate_mass_nll.is_finite()
        {
            return Err("nonfinite conditional answer objective".into());
        }
        let constraint = mass.as_deref().zip(config.mass_inward_cosine);
        let mut checks = Vec::new();
        let Some((direction, projected)) = direction(&choice, constraint) else {
            return Ok((result, checks));
        };
        result.projected = projected;
        for i in 0..config.max_trials {
            let length = config.length * 0.5_f64.powi(i as i32);
            let right: Vec<_> = self
                .right()
                .iter()
                .zip(&direction)
                .map(|(&x, &d)| x - length * d)
                .collect();
            let displacement: Vec<_> = self
                .right()
                .iter()
                .zip(&right)
                .map(|(&x, &y)| x - y)
                .collect();
            let distance = norm(&displacement);
            if distance == 0. {
                break;
            }
            let required = 1e-4 * dot(&choice, &displacement).max(0.);
            if !distance.is_finite() || !required.is_finite() {
                return Err("nonfinite conditional displacement".into());
            }
            let mut candidate = self.clone();
            candidate.install(right)?;
            let mut point = PointSum::default();
            // Only this projection changes, so the prefix strictly before it
            // is identical for every rejected or committed proposal.
            let m = &candidate.current;
            for row in rows {
                let [y, a] = row.pair;
                let shape = [1, row.prefix.steps];
                let h =
                    m.forward_blocks(row.prefix.hidden.clone(), shape, first..m.config.layers)?;
                let last = h.gather_rows(&[row.prefix.steps - 1])?;
                let normalized = m.final_norm.forward(&last, m.config.norm_epsilon)?;
                let logits = m.project_readout(&normalized)?;
                let z = logits.data();
                point.add(
                    logistic(f64::from(z[y]) - f64::from(z[a])).0,
                    candidate_mass_loss(z, [y, a])?,
                );
            }
            let after = point.finish(n);
            let drop = before.conditional_ce - after.conditional_ce;
            let ordinary_accepted = drop > 0.
                && drop >= required
                && (!guarded || after.candidate_mass_nll <= before.candidate_mass_nll);
            let mut accepted = ordinary_accepted;
            if let Some(deck) = retention {
                let check = ordinary_accepted
                    .then(|| deck.check(&candidate))
                    .transpose()?;
                accepted &= check
                    .as_ref()
                    .is_none_or(CausalLmConditionalRetentionCheck::permits_commit);
                checks.push(check);
            }
            result.trial_forwards += n;
            result.reused_prefix_block_calls += n * first;
            result.trials.push(CausalLmConditionalAnswerTrial {
                point: after,
                projection_id: candidate.projection_id(),
                distance,
                required_conditional_drop: required,
                accepted,
            });
            if accepted {
                *self = candidate;
                result.after = after;
                result.committed = true;
                break;
            }
        }
        Ok((result, checks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_direction_gains_on_both_objectives_or_holds() {
        let pair = [-0.6, 0.8];
        let mass = [1., 0.];
        let (g, projected) = direction(&pair, Some((&mass, 0.05))).unwrap();
        assert!(projected && (g[0] - 0.05).abs() < 1e-15);
        assert!(dot(&pair, &g) > 0. && (norm(&g) - 1.).abs() < 1e-15);
        assert!(direction(&[-1., 0.], Some((&mass, 0.05))).is_none());
        assert_eq!(
            direction(&mass, Some((&mass, 0.05))),
            Some((mass.to_vec(), false))
        );
    }
}
