use super::*;
use st_nn::causal_lm::{CausalLmScalarSearch, NativeCausalLmScalarOutput, MAX_SCALAR_OUTPUT_PAIRS};

#[pyclass(name = "NativeCausalLmScalarOutput", module = "spiraltorch", frozen)]
pub(super) struct PyScalarOutput(pub(super) NativeCausalLmScalarOutput);

struct PairInputs {
    pos: Vec<Vec<usize>>,
    neg: Vec<Vec<usize>>,
    t: usize,
    a: usize,
}

fn number(x: &Bound<'_, PyAny>) -> PyResult<f64> {
    reject_bool(x)?;
    x.extract()
}

fn inputs(
    model: &NativeCausalLmScalarOutput,
    positive: &Bound<'_, PyAny>,
    negative: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<PairInputs> {
    let n = positive.len()?;
    if n == 0 || n > MAX_SCALAR_OUTPUT_PAIRS || negative.len()? != n {
        return Err(error("invalid scalar pair batch size"));
    }
    let (t, a) = (dimension(target)?, dimension(other)?);
    let unpack = |rows: &Bound<'_, PyAny>| -> PyResult<Vec<Vec<usize>>> {
        (0..n)
            .map(|i| {
                let ids = rows.get_item(i)?;
                tokens(
                    &ids,
                    model.pair_prompt_count(ids.len()?, t, a).map_err(error)?,
                )
            })
            .collect()
    };
    Ok(PairInputs {
        pos: unpack(positive)?,
        neg: unpack(negative)?,
        t,
        a,
    })
}

#[pymethods]
impl PyScalarOutput {
    #[getter]
    fn coefficient(&self) -> f64 {
        self.0.coefficient()
    }
    #[getter]
    fn steps(&self) -> u64 {
        self.0.steps()
    }
    fn checkpoint_json(&self) -> PyResult<String> {
        self.0.checkpoint_json().map_err(error)
    }
    fn last_logits_f32le<'py>(
        &self,
        py: Python<'py>,
        prompt: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let ids = tokens(prompt, self.0.prompt_count(prompt.len()?).map_err(error)?)?;
        let v = py.detach(|| self.0.last_logits(&ids)).map_err(error)?;
        PyBytes::new_with(py, v.len() * 4, |buf| {
            for (b, x) in buf.chunks_exact_mut(4).zip(v) {
                b.copy_from_slice(&x.to_le_bytes());
            }
            Ok(())
        })
    }
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(positive,negative,target_id,alternative_id,*,learning_rate,max_step,max_abs))]
    fn step_pairs(
        &self,
        py: Python<'_>,
        positive: &Bound<'_, PyAny>,
        negative: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        alternative_id: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        max_step: &Bound<'_, PyAny>,
        max_abs: &Bound<'_, PyAny>,
    ) -> PyResult<(Self, Py<PyAny>)> {
        let (rate, step, limit) = (number(learning_rate)?, number(max_step)?, number(max_abs)?);
        self.0.validate_sgd(rate, step, limit).map_err(error)?;
        let PairInputs { pos, neg, t, a } =
            inputs(&self.0, positive, negative, target_id, alternative_id)?;
        let (s, r) = py
            .detach(|| self.0.step_pairs(&pos, &neg, t, a, rate, step, limit))
            .map_err(error)?;
        Ok((
            Self(s),
            crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)?,
        ))
    }

    fn pair_objective(
        &self,
        py: Python<'_>,
        positive: &Bound<'_, PyAny>,
        negative: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        alternative_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let PairInputs { pos, neg, t, a } =
            inputs(&self.0, positive, negative, target_id, alternative_id)?;
        let r = py
            .detach(|| self.0.pair_objective(&pos, &neg, t, a))
            .map_err(error)?;
        crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(positive,negative,target_id,alternative_id,*,learning_rate,max_step,max_abs,max_trials,min_decrease))]
    fn step_pairs_backtracking(
        &self,
        py: Python<'_>,
        positive: &Bound<'_, PyAny>,
        negative: &Bound<'_, PyAny>,
        target_id: &Bound<'_, PyAny>,
        alternative_id: &Bound<'_, PyAny>,
        learning_rate: &Bound<'_, PyAny>,
        max_step: &Bound<'_, PyAny>,
        max_abs: &Bound<'_, PyAny>,
        max_trials: &Bound<'_, PyAny>,
        min_decrease: &Bound<'_, PyAny>,
    ) -> PyResult<(Self, Py<PyAny>)> {
        let search = CausalLmScalarSearch {
            learning_rate: number(learning_rate)?,
            max_step: number(max_step)?,
            max_abs: number(max_abs)?,
            max_trials: dimension(max_trials)?,
            min_decrease: number(min_decrease)?,
        };
        search.validate(&self.0).map_err(error)?;
        let PairInputs { pos, neg, t, a } =
            inputs(&self.0, positive, negative, target_id, alternative_id)?;
        let (s, r) = py
            .detach(|| self.0.step_pairs_backtracking(&pos, &neg, t, a, search))
            .map_err(error)?;
        Ok((
            Self(s),
            crate::json::json_to_py(py, &serde_json::to_value(r).map_err(error)?)?,
        ))
    }
}
