//! Shared bounded document transport; callers supply Rust-owned admission.
use super::{error, loss_labels, tokens};
use pyo3::prelude::*;
use st_nn::language_windows::MAX_DOCUMENTS;

type Documents = (Vec<Vec<usize>>, Option<Vec<Vec<i64>>>);

pub(super) fn documents(
    value: &Bound<'_, PyAny>,
    labels: Option<&Bound<'_, PyAny>>,
    max_tokens: usize,
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
    let mut target_rows = labels.map(|_| Vec::with_capacity(count));
    let mut lengths = Vec::with_capacity(count);
    let mut remaining = max_tokens;
    for index in 0..count {
        let row = value.get_item(index)?;
        let length = row.len()?;
        remaining = remaining
            .checked_sub(length)
            .ok_or_else(|| error("document token count exceeds budget"))?;
        if let (Some(value), Some(targets)) = (labels, &mut target_rows) {
            let label_row = value.get_item(index)?;
            if label_row.len()? != length {
                return Err(error("document label length differs"));
            }
            targets.push(label_row);
        }
        lengths.push(length);
        rows.push(row);
    }
    // Admission may detach from Python. Keep both sides of each row pair pinned;
    // only read their scalar values once the common Rust plan has been admitted.
    admit(&lengths, labels.is_some())?;
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
