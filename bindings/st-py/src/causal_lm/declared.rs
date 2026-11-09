use super::*;
use st_nn::causal_lm::{DeclaredRivalRow, MAX_DECLARED_RIVAL_IDS, MAX_RIVAL_CLASSES};

pub(super) fn extract(
    value: &Bound<'_, PyAny>,
    shape: [usize; 2],
    count: usize,
    classes: usize,
    mix: f64,
) -> PyResult<Vec<DeclaredRivalRow>> {
    DeclaredRivalRow::validate_plan(&[], count, classes, mix).map_err(error)?;
    let n = value.len()?;
    if n > (count - shape[0]).min(MAX_DECLARED_RIVAL_IDS) {
        return Err(error("too many declared target positions"));
    }
    let mut result = Vec::with_capacity(n);
    let mut remaining = MAX_DECLARED_RIVAL_IDS;
    for i in 0..n {
        let item = value.get_item(i)?;
        if item.len()? != 3 {
            return Err(error("expected (position, target, rivals)"));
        }
        let position = dimension(&item.get_item(0)?)?;
        let target = dimension(&item.get_item(1)?)?;
        if position >= count || position % shape[1] == 0 {
            return Err(error("declared position must have a causal predictor"));
        }
        let rivals = item.get_item(2)?;
        let size = rivals.len()?;
        if size == 0
            || size
                > remaining
                    .min(MAX_RIVAL_CLASSES)
                    .min(classes.saturating_sub(1))
        {
            return Err(error("invalid number of declared rival IDs"));
        }
        remaining -= size;
        result.push(DeclaredRivalRow {
            position,
            target,
            rivals: tokens(&rivals, size)?,
        });
    }
    DeclaredRivalRow::validate_plan(&result, count, classes, mix).map_err(error)?;
    Ok(result)
}
