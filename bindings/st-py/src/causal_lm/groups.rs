//! Admit all group sizes before extracting any sealed gradient handle.
use super::*;
use st_nn::causal_lm::{CausalLmResult, NativeCausalLmAdaptationGradients};

pub(super) fn apply<'py, R: serde::Serialize + Send>(
    py: Python<'py>,
    groups: &Bound<'py, PyAny>,
    n: usize,
    primary: Option<&Bound<'py, PyAny>>,
    admit: impl FnOnce(&[usize]) -> CausalLmResult<()>,
    run: impl FnOnce(&[&[&NativeCausalLmAdaptationGradients]]) -> CausalLmResult<(NativeCausalLm, R)>
        + Send,
) -> PyResult<(PyNativeCausalLm, Py<PyAny>)> {
    let mut lists = (0..n)
        .map(|i| groups.get_item(i))
        .collect::<PyResult<Vec<_>>>()?;
    if let Some(first) = primary {
        lists.insert(0, first.clone());
    }
    let sizes = lists
        .iter()
        .map(|l| l.len())
        .collect::<PyResult<Vec<_>>>()?;
    admit(&sizes).map_err(error)?;
    let owners = lists
        .iter()
        .zip(sizes)
        .map(|(l, n)| {
            (0..n)
                .map(|i| {
                    l.get_item(i)?
                        .cast_into::<PyNativeCausalLmAdaptationGradients>()
                        .map_err(Into::into)
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let refs: Vec<Vec<_>> = owners
        .iter()
        .map(|g| g.iter().map(|x| &x.get().inner).collect())
        .collect();
    let slices: Vec<_> = refs.iter().map(Vec::as_slice).collect();
    let (inner, report) = py.detach(|| run(&slices)).map_err(error)?;
    Ok((
        PyNativeCausalLm { inner },
        crate::json::json_to_py(py, &serde_json::to_value(report).map_err(error)?)?,
    ))
}
