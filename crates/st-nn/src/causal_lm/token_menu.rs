//! A single next-token choice, in contrast to a menu of complete terminated answers.
use super::*;

impl NativeCausalLm {
    /// Admit lengths and the geometry owner before a client copies prompt or choice IDs.
    pub fn geometry_token_menu_shape(
        &self,
        prompt_len: usize,
        choices: usize,
        gold_index: usize,
        mix: f64,
    ) -> CausalLmResult<[usize; 2]> {
        self.validate_adaptation_mode(CausalLmAdaptationMode::Geometry)?;
        Self::validate_ranking(choices, gold_index, mix)?;
        if prompt_len == 0 || choices > MAX_LINE_MENU_CANDIDATES.min(self.config.vocabulary) {
            return Err("invalid next-token menu or empty prompt".into());
        }
        let width = prompt_len
            .checked_add(1)
            .ok_or("next-token window overflow")?;
        let shape = [1, width];
        self.adaptation_labeled_input_count(shape, true)?;
        Ok(shape)
    }

    /// Mix full-vocabulary gold CE with CE conditional on the supplied sorted token IDs.
    /// Exactly one prediction is supervised; neither the prompt nor any terminator is a target.
    pub fn geometry_token_menu_gradients(
        &self,
        prompt: &[usize],
        choices: &[usize],
        gold_index: usize,
        mix: f64,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let shape = self.geometry_token_menu_shape(prompt.len(), choices.len(), gold_index, mix)?;
        if choices.windows(2).any(|p| p[0] >= p[1])
            || choices.iter().any(|&id| id >= self.config.vocabulary)
        {
            return Err("expected distinct sorted in-vocabulary next-token choices".into());
        }
        let gold = choices[gold_index];
        let mut window = Vec::with_capacity(shape[1]);
        window.extend_from_slice(prompt);
        window.push(gold);
        let mut labels = vec![-100; shape[1]];
        labels[prompt.len()] = gold as i64;
        let contrast = DeclaredRivalRow {
            position: prompt.len(),
            target: gold,
            rivals: choices.iter().copied().filter(|&id| id != gold).collect(),
        };
        self.geometry_declared_rival_gradients(&window, shape, Some(&labels), &[contrast], mix)
    }
}
