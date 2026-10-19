//! Shared bounded document transport; callers supply Rust-owned admission.
use super::{error, loss_labels, tokens};
use pyo3::prelude::*;
use st_nn::language_windows::MAX_DOCUMENTS;

type Documents = (Vec<Vec<usize>>, Option<Vec<Vec<i64>>>);

pub(super) fn documents(
    value: &Bound<'_, PyAny>,
    labels: Option<&Bound<'_, PyAny>>,
    admit: impl FnOnce(&[usize], bool) -> PyResult<()>,
) -> PyResult<Documents> {
    let count = value.len()?;
    if count > MAX_DOCUMENTS {
        return Err(error("document count exceeds budget"));
    }
    if labels
        .map(|value| value.len())
        .transpose()?
        .is_some_and(|length| length != count)
    {
        return Err(error("document and label counts differ"));
    }
    let mut rows = Vec::with_capacity(count);
    let mut lengths = Vec::with_capacity(count);
    for index in 0..count {
        let row = value.get_item(index)?;
        lengths.push(row.len()?);
        rows.push(row);
    }
    admit(&lengths, labels.is_some())?;
    let mut target_rows = labels.map(|_| Vec::with_capacity(count));
    if let (Some(value), Some(rows)) = (labels, &mut target_rows) {
        for (index, &length) in lengths.iter().enumerate() {
            let row = value.get_item(index)?;
            if row.len()? != length {
                return Err(error("document label length differs"));
            }
            rows.push(row);
        }
    }
    let ids = rows
        .iter()
        .zip(&lengths)
        .map(|(row, &n)| tokens(row, n))
        .collect::<PyResult<_>>()?;
    let labels = target_rows
        .map(|rows| {
            rows.iter()
                .zip(&lengths)
                .map(|(row, &n)| loss_labels(Some(row), n).map(Option::unwrap))
                .collect::<PyResult<_>>()
        })
        .transpose()?;
    Ok((ids, labels))
}
