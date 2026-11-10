use super::*;
use st_nn::causal_lm::MAX_LINE_MENU_CANDIDATES;

pub(super) struct Inputs {
    pub prompt: Vec<usize>,
    pub lines: Vec<Vec<usize>>,
    pub stop: usize,
}

pub(super) fn subset(
    model: &NativeCausalLm,
    prompt: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    accepted: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    training: bool,
) -> PyResult<(Inputs, Vec<usize>)> {
    let n = choices.len()?;
    let k = accepted.len()?;
    if !(2..=MAX_LINE_MENU_CANDIDATES).contains(&n) || k == 0 || k >= n {
        return Err(error("expected a nonempty proper subset of a bounded menu"));
    }
    let indexes = tokens(accepted, k)?;
    NativeCausalLm::validate_line_subset(n, &indexes).map_err(error)?;
    Ok((
        collect(model, prompt, choices, stop, None, training)?,
        indexes,
    ))
}

pub(super) fn extract(
    model: &NativeCausalLm,
    prompt: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    gold: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    mix: &Bound<'_, PyAny>,
    training: bool,
) -> PyResult<(Inputs, usize, f64)> {
    let gold = dimension(gold)?;
    reject_bool(mix)?;
    let mix = mix.extract::<f64>()?;
    let inputs = collect(model, prompt, choices, stop, Some((gold, mix)), training)?;
    Ok((inputs, gold, mix))
}

pub(super) fn collect(
    model: &NativeCausalLm,
    prompt: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    menu: Option<(usize, f64)>,
    training: bool,
) -> PyResult<Inputs> {
    if training {
        model
            .validate_adaptation_mode(st_nn::causal_lm::CausalLmAdaptationMode::Geometry)
            .map_err(error)?;
    }
    let stop = dimension(stop)?;
    let n = choices.len()?;
    if !((if menu.is_some() { 2 } else { 1 })..=MAX_LINE_MENU_CANDIDATES).contains(&n) {
        return Err(error("invalid complete-line candidate count"));
    }
    let prompt_len = prompt.len()?;
    let items: Vec<_> = (0..n)
        .map(|i| choices.get_item(i))
        .collect::<PyResult<_>>()?;
    let sizes: Vec<_> = items.iter().map(|p| p.len()).collect::<PyResult<_>>()?;
    match menu {
        Some((gold, mix)) => model.line_menu_shape(prompt_len, &sizes, gold, stop, mix, training),
        None => model.line_set_shape(prompt_len, &sizes, stop, training),
    }
    .map_err(error)?;
    let prompt = tokens(prompt, prompt_len)?;
    let lines = items
        .iter()
        .zip(sizes)
        .map(|(p, n)| tokens(p, n))
        .collect::<PyResult<_>>()?;
    Ok(Inputs {
        prompt,
        lines,
        stop,
    })
}
