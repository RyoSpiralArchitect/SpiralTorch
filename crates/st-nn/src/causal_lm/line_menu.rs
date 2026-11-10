//! Train against complete terminated alternatives, not a restricted vocabulary at the head.
pub(super) use super::line_set::LineMenu;
use super::*;

pub const MAX_LINE_MENU_CANDIDATES: usize = 32;
pub const MAX_LINE_MENU_TARGETS: usize = 4096;
const MAX_PADDED_IDS: usize = 65_536;

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmLineMenuReport {
    pub gold_index: usize,
    pub winner_index: usize,
    pub shared_content_tokens: usize,
    pub prefix_nodes: usize,
    pub decoder_lanes: usize,
    pub stop_id: usize,
    /// Includes one stop per candidate. All vectors follow the supplied lexicographic order.
    pub target_counts: Vec<usize>,
    pub line_nlls: Vec<f64>,
    pub suffix_nlls: Vec<f64>,
    pub menu_probabilities: Vec<f64>,
    /// Sum of rival probabilities, not one minus a rounded gold probability.
    pub rival_mass: f64,
    pub gold_mean_nll: f64,
    pub menu_nll: f64,
    pub mix: f64,
    pub loss: f32,
}

impl NativeCausalLm {
    /// Admit lengths (including padded decoder work and selected logits) before copying foreign token arrays.
    pub fn line_menu_shape(
        &self,
        prompt_len: usize,
        content_lengths: &[usize],
        gold_index: usize,
        stop_id: usize,
        mix: f64,
        geometry_gradients: bool,
    ) -> CausalLmResult<[usize; 2]> {
        Self::validate_ranking(content_lengths.len(), gold_index, mix)?;
        self.line_set_shape(prompt_len, content_lengths, stop_id, geometry_gradients)
    }

    fn validate_ranking(count: usize, gold_index: usize, mix: f64) -> CausalLmResult<()> {
        if count < 2 || gold_index >= count || !(0.0..=1.0).contains(&mix) {
            return Err("invalid complete-line menu size, gold, stop, prompt or mix".into());
        }
        Ok(())
    }

    /// Bound logical input even when prefix sharing later makes the physical batch smaller.
    pub(super) fn terminated_shape(
        &self,
        prompt_len: usize,
        content_lengths: &[usize],
        stop_id: usize,
    ) -> CausalLmResult<([usize; 2], usize)> {
        let n = content_lengths.len();
        if !(1..=MAX_LINE_MENU_CANDIDATES).contains(&n)
            || prompt_len == 0
            || stop_id >= self.config.vocabulary
            || content_lengths.contains(&0)
        {
            return Err(
                "invalid complete-line support size, stop, prompt or content length".into(),
            );
        }
        let targets = content_lengths
            .iter()
            .try_fold(n, |total, &v| total.checked_add(v))
            .ok_or("line target count overflow")?;
        if targets > MAX_LINE_MENU_TARGETS {
            return Err("complete-line target budget exceeded".into());
        }
        let width = prompt_len
            .checked_add(*content_lengths.iter().max().unwrap())
            .and_then(|v| v.checked_add(1))
            .ok_or("padded line width overflow")?;
        let shape = [n, width];
        product(&shape, MAX_PADDED_IDS, "padded complete-line IDs")?;
        Ok((shape, targets))
    }

    pub(super) fn admit_terminated_work(
        &self,
        shape: [usize; 2],
        readout_rows: usize,
        geometry_gradients: bool,
    ) -> CausalLmResult<()> {
        if geometry_gradients {
            self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
            self.adaptation_labeled_input_count(shape, true)?;
        } else {
            self.loss_objective_input_count(shape, true)?;
        }
        self.logit_count(readout_rows)?;
        Ok(())
    }

    fn prepare_line_menu(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        gold: usize,
        stop: usize,
        mix: f64,
        training: bool,
    ) -> CausalLmResult<(PreparedCausalWindow, LineMenu)> {
        Self::validate_ranking(candidates.len(), gold, mix)?;
        let (batch, support) = self.prepare_line_set(prompt, candidates, stop, training)?;
        Ok((batch, LineMenu::new(support, candidates, gold, mix)))
    }

    pub(super) fn validate_terminated_contents(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        stop: usize,
    ) -> CausalLmResult<()> {
        if prompt.iter().any(|&v| v >= self.config.vocabulary)
            || candidates.windows(2).any(|p| p[0] >= p[1])
            || candidates
                .iter()
                .flatten()
                .any(|&v| v == stop || v >= self.config.vocabulary)
        {
            return Err("complete-line contents must be distinct, sorted token sequences without the stop ID; token IDs must be in range".into());
        }
        Ok(())
    }

    /// Forward-only loss on one prompt and sorted complete content alternatives (stop supplied separately).
    pub fn line_menu_loss(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        gold_index: usize,
        stop_id: usize,
        mix: f64,
    ) -> CausalLmResult<CausalLmLineMenuReport> {
        let (batch, plan) =
            self.prepare_line_menu(prompt, candidates, gold_index, stop_id, mix, false)?;
        let hidden = self.decode_hidden(&batch.input, batch.shape)?;
        plan.score(&self.causal_logits(hidden, &batch)?)
    }

    /// Geometry-only VJP of exactly the forward-only line-menu loss. Base and any LoRA are frozen.
    pub fn geometry_line_menu_gradients(
        &self,
        prompt: &[usize],
        candidates: &[Vec<usize>],
        gold_index: usize,
        stop_id: usize,
        mix: f64,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let (mut batch, plan) =
            self.prepare_line_menu(prompt, candidates, gold_index, stop_id, mix, true)?;
        batch.competition = Some(objective::PreparedCompetition::LineMenu(plan));
        self.geometry_gradients_for_competition(&batch)
    }
}
