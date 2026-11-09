//! One answer per unpadded sequence. Length groups retain the caller's order.
use super::*;
use crate::language_windows::{equal_width_groups, CausalDocumentWindow};

pub(super) struct AnswerGradientMean {
    width: usize,
    samples: u64,
    loss: sum::Sum,
    values: Vec<sum::Sum>,
}

impl AnswerGradientMean {
    pub(super) fn new(width: usize) -> Self {
        Self {
            width,
            samples: 0,
            loss: Default::default(),
            values: (0..3 * width * width)
                .map(|_| sum::Sum::default())
                .collect(),
        }
    }
    pub(super) fn add(&mut self, loss: f32, gradient: &Tensor, samples: usize) {
        // Only checked, nonempty answer_gradients results reach this reducer.
        debug_assert_eq!(gradient.shape(), (3 * self.width, self.width));
        self.samples += samples as u64;
        self.loss.add(f64::from(loss) * samples as f64);
        for (a, &g) in self.values.iter_mut().zip(gradient.data()) {
            a.add(f64::from(g) * samples as f64);
        }
    }
    pub(super) fn finish(self) -> CausalLmResult<(f32, Tensor)> {
        if self.samples == 0 {
            return Err("empty QKV gradient mean".into());
        }
        let values = self
            .values
            .into_iter()
            .map(|s| s.mean(self.samples) as f32)
            .collect();
        Ok((
            self.loss.mean(self.samples) as f32,
            Tensor::from_vec(3 * self.width, self.width, values)?,
        ))
    }
}

impl NativeCausalLmQkvAdamW {
    /// Admission bound on sequences in one logical answer update.
    pub const MAX_ANSWER_SEQUENCES: usize = 4096;

    /// Admit the entire logical batch before a client copies token storage.
    pub fn answer_sequence_input_count(
        &self,
        lengths: &[usize],
        max_rows: usize,
    ) -> CausalLmResult<usize> {
        self.answer_sequence_groups(lengths, max_rows)
            .map(|(_, count)| count)
    }

    fn answer_sequence_groups(
        &self,
        lengths: &[usize],
        max_rows: usize,
    ) -> CausalLmResult<(Vec<std::ops::Range<usize>>, usize)> {
        if lengths.is_empty() || lengths.len() > Self::MAX_ANSWER_SEQUENCES || max_rows == 0 {
            return Err("invalid QKV sequence roster or microbatch limit".into());
        }
        let c = &self.model.config;
        let mut total = 0usize;
        let mut windows = Vec::with_capacity(lengths.len());
        for (i, &length) in lengths.iter().enumerate() {
            if length == 0 || length > c.positions {
                return Err("invalid unpadded QKV prefix length".into());
            }
            total = total
                .checked_add(length)
                .ok_or("QKV sequence token count overflow")?;
            windows.push(CausalDocumentWindow {
                document: i as u32,
                start: 0,
                length: u32::try_from(length)?,
            });
        }
        product(
            &[total, 3, c.width, c.width],
            1 << 30,
            "logical QKV gradient work",
        )?;
        let groups: Vec<_> = equal_width_groups(&windows, max_rows).collect();
        if groups.len() > 1024 {
            return Err("QKV sequence microbatch count exceeds limit".into());
        }
        for group in &groups {
            self.answer_gradient_rows([group.len(), lengths[group.start]])?;
        }
        Ok((groups, total))
    }

    /// Mean full-vocabulary answer CE and derivative for ragged unpadded prefixes.
    ///
    /// Each sequence contributes one final-position answer, irrespective of its
    /// prefix length. Only adjacent equal-length sequences are batched; there is
    /// no padding, truncation, reordering, EOS insertion or document concatenation.
    /// All groups are structurally admitted before the first forward pass.
    /// Changing max_rows can change floating-point rounding; preserve it and
    /// the exact input order when reproducing a checkpoint continuation.
    pub fn answer_sequence_gradients(
        &self,
        prompts: &[&[usize]],
        gold: &[usize],
        max_rows: usize,
    ) -> CausalLmResult<(f32, Tensor)> {
        self.answer_sequence_gradients_with_label_smoothing(prompts, gold, max_rows, 0.)
    }

    /// Full-vocabulary smoothed CE, with one answer-weighted contribution per prefix.
    /// Uses the shared tensor CE/VJP contract: target=(1-epsilon)*one_hot+epsilon/V.
    /// The returned scalar is the training objective, not unsmoothed gold NLL.
    pub fn answer_sequence_gradients_with_label_smoothing(
        &self,
        prompts: &[&[usize]],
        gold: &[usize],
        max_rows: usize,
        label_smoothing: f64,
    ) -> CausalLmResult<(f32, Tensor)> {
        CrossEntropyConfig {
            label_smoothing,
            ..CrossEntropyConfig::default()
        }
        .validate()?;
        if prompts.is_empty()
            || prompts.len() > Self::MAX_ANSWER_SEQUENCES
            || prompts.len() != gold.len()
            || max_rows == 0
        {
            return Err("invalid QKV sequence roster or microbatch limit".into());
        }
        let c = &self.model.config;
        let lengths = prompts.iter().map(|q| q.len()).collect::<Vec<_>>();
        let (groups, _) = self.answer_sequence_groups(&lengths, max_rows)?;
        for (q, &label) in prompts.iter().zip(gold) {
            if label >= c.vocabulary || q.iter().any(|&x| x >= c.vocabulary) {
                return Err("invalid unpadded QKV prefix or answer".into());
            }
        }
        let mut mean = AnswerGradientMean::new(c.width);
        for group in groups {
            let shape = [group.len(), lengths[group.start]];
            let ids = prompts[group.clone()].concat();
            let (loss, gradient) =
                self.answer_gradients_smoothed(&ids, shape, &gold[group.clone()], label_smoothing)?;
            mean.add(loss, &gradient, group.len());
        }
        mean.finish()
    }

    /// Commit exactly one atomic AdamW update after every length group succeeds.
    pub fn step_answer_sequences(
        &mut self,
        prompts: &[&[usize]],
        gold: &[usize],
        max_rows: usize,
        step: AdamWStep,
    ) -> CausalLmResult<CausalLmQkvStep> {
        let (loss, gradient) = self.answer_sequence_gradients(prompts, gold, max_rows)?;
        self.apply_answer_gradient(loss, gradient, step)
    }

    /// Opt-in smoothed learning, with unchanged atomic optimizer/checkpoint semantics.
    /// Record epsilon with the caller's batch schedule for exact continuation;
    /// optimizer checkpoints do not own the next objective, batch or learning rate.
    /// Evaluate plain gold NLL separately; no smoothing is applied to inference.
    pub fn step_answer_sequences_with_label_smoothing(
        &mut self,
        prompts: &[&[usize]],
        gold: &[usize],
        max_rows: usize,
        label_smoothing: f64,
        step: AdamWStep,
    ) -> CausalLmResult<CausalLmQkvSmoothedStep> {
        let (objective_loss, gradient) = self.answer_sequence_gradients_with_label_smoothing(
            prompts,
            gold,
            max_rows,
            label_smoothing,
        )?;
        let (steps, gradient_l2, gradient_clip) = self.apply_gradient(gradient, step)?;
        Ok(CausalLmQkvSmoothedStep {
            steps,
            objective_loss,
            label_smoothing,
            gradient_l2,
            gradient_clip,
        })
    }
}
