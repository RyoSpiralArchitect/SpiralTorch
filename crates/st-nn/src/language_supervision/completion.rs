//! The answer is scored in both documents; only training includes an explicit terminator.
use super::CausalTargetAlignment;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct CausalCompletion {
    schema: &'static str,
    training_ids: Vec<u32>,
    training_labels: Vec<i64>,
    prompt_end: usize,
    answer_end: usize,
}

impl CausalCompletion {
    /// Token counts, not source offsets. Called before reading foreign elements.
    pub fn admit_counts(
        prompt: usize,
        prompt_answer: usize,
        training: usize,
    ) -> Result<(), &'static str> {
        if prompt == 0
            || prompt >= prompt_answer
            || prompt_answer > training
            || training > CausalTargetAlignment::MAX_TOKENS
        {
            return Err("completion requires a nonempty prompt and answer within the token budget");
        }
        Ok(())
    }

    fn check_prefixes(
        prompt: &[u32],
        prompt_answer: &[u32],
        training: &[u32],
    ) -> Result<(), &'static str> {
        Self::admit_counts(prompt.len(), prompt_answer.len(), training.len())?;
        if !prompt_answer.starts_with(prompt) || !training.starts_with(prompt_answer) {
            return Err("independent prompt or prompt-plus-answer token IDs differ from the training prefix");
        }
        Ok(())
    }

    fn build(training_ids: Vec<u32>, prompt_end: usize, answer_end: usize) -> Self {
        let mut training_labels = vec![-100; prompt_end];
        training_labels.extend(training_ids[prompt_end..].iter().copied().map(i64::from));
        Self {
            schema: "spiraltorch.causal_completion.v1",
            training_ids,
            training_labels,
            prompt_end,
            answer_end,
        }
    }

    /// No text codec assumed. The last input is prompt + answer + optional terminator.
    pub fn from_tokenizations(
        prompt: &[u32],
        prompt_answer: &[u32],
        training: Vec<u32>,
    ) -> Result<Self, &'static str> {
        Self::check_prefixes(prompt, prompt_answer, &training)?;
        Ok(Self::build(training, prompt.len(), prompt_answer.len()))
    }

    /// Only the answer is marked in the source alignment. A trailing unmarked
    /// run, if present, is the caller's explicit terminator, never evaluation content.
    pub fn from_alignment(
        alignment: &CausalTargetAlignment,
        prompt: &[u32],
        prompt_answer: &[u32],
    ) -> Result<Self, &'static str> {
        Self::check_prefixes(prompt, prompt_answer, alignment.token_ids())?;
        match alignment.runs() {
            [p, a] | [p, a, _]
                if p[1] as usize == prompt.len() && a[1] as usize == prompt_answer.len() => {}
            _ => {
                return Err(
                    "completion prefixes must cover exactly the aligned prompt and single answer",
                )
            }
        }
        Ok(Self::build(
            alignment.token_ids().to_vec(),
            prompt.len(),
            prompt_answer.len(),
        ))
    }

    pub fn prompt_ids(&self) -> &[u32] {
        &self.training_ids[..self.prompt_end]
    }
    pub fn answer_end(&self) -> usize {
        self.answer_end
    }
    pub fn training_document(&self) -> (&[u32], &[i64]) {
        (&self.training_ids, &self.training_labels)
    }
    pub fn evaluation_document(&self) -> (&[u32], &[i64]) {
        (
            &self.training_ids[..self.answer_end],
            &self.training_labels[..self.answer_end],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_views_keep_the_answer_out_of_generation_and_suffix_out_of_evaluation() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/causal_completion.json"))
                .unwrap();
        let input = &f["input"];
        let ids: Vec<u32> = serde_json::from_value(input["training_ids"].clone()).unwrap();
        let prefix: Vec<u32> = serde_json::from_value(input["prompt_ids"].clone()).unwrap();
        let content: Vec<u32> = serde_json::from_value(input["prompt_answer_ids"].clone()).unwrap();
        let offsets: Vec<[u32; 2]> =
            serde_json::from_value(input["source"]["offsets"].clone()).unwrap();
        let alignment = CausalTargetAlignment::new(7, ids.clone(), &offsets, &[[3, 6]]).unwrap();
        for view in [
            CausalCompletion::from_tokenizations(&prefix, &content, ids).unwrap(),
            CausalCompletion::from_alignment(&alignment, &prefix, &content).unwrap(),
        ] {
            assert_eq!(serde_json::to_value(&view).unwrap(), f["expected"]);
            assert_eq!(view.prompt_ids(), &prefix);
            assert_eq!(view.evaluation_document().0, &content);
            assert_eq!(
                view.evaluation_document().1,
                &[-100, -100, -100, -100, 5, 6, 7, 8]
            );
        }
        let v =
            CausalCompletion::from_tokenizations(&[0], &[0, u32::MAX], vec![0, u32::MAX]).unwrap();
        assert_eq!(v.evaluation_document(), v.training_document());
        assert_eq!(v.training_document().1, &[-100, i64::from(u32::MAX)]);
    }

    #[test]
    fn rejects_divergent_prefixes_oversize_and_ambiguous_source_spans() {
        for (p, a, t) in [(0, 1, 2), (1, 1, 2), (2, 3, 2), (1, 2, usize::MAX)] {
            assert!(CausalCompletion::admit_counts(p, a, t).is_err());
        }
        assert!(CausalCompletion::from_tokenizations(&[2], &[1, 2], vec![1, 2, 3]).is_err());
        assert!(CausalCompletion::from_tokenizations(&[1], &[1, 3], vec![1, 2, 3]).is_err());
        let offsets = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]];
        for spans in [vec![[2, 3]], vec![[1, 3], [4, 5]], vec![[1, 4]]] {
            let a = CausalTargetAlignment::new(5, vec![1, 2, 3, 4, 5], &offsets, &spans).unwrap();
            assert!(CausalCompletion::from_alignment(&a, &[1], &[1, 2, 3]).is_err());
        }
    }
}
