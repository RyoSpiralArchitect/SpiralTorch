//! Input admission only. Rust owns the packed decoder, loss, seed and gradient identity.
use super::*;
use st_nn::causal_lm::CausalLmPairedGapObjective as Objective;

pub(super) struct Inputs {
    pub first: Vec<usize>,
    pub second: Vec<usize>,
    pub ids: [usize; 2],
    pub objective: Objective,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect(
    model: &NativeCausalLm,
    first: &Bound<'_, PyAny>,
    second: &Bound<'_, PyAny>,
    ids: &Bound<'_, PyAny>,
    name: &str,
    center: Option<&Bound<'_, PyAny>>,
    weight: Option<&Bound<'_, PyAny>>,
    training: bool,
) -> PyResult<Inputs> {
    let objective = match (name, center, weight) {
        ("binary", None, None) => Objective::Binary,
        ("rank", None, None) => Objective::Rank,
        ("weakest_lane", None, None) => Objective::WeakestLane,
        ("anchored_rank", Some(c), Some(w)) => {
            reject_bool(c)?;
            reject_bool(w)?;
            Objective::AnchoredRank { reference_center:c.extract()?, stiffness:w.extract()? }
        }
        _ => return Err(error("expected binary, rank or weakest_lane without an anchor, or anchored_rank with reference_center and stiffness")),
    };
    if ids.len()? != 2 {
        return Err(error("expected two competing token IDs"));
    }
    let ids = [dimension(&ids.get_item(0)?)?, dimension(&ids.get_item(1)?)?];
    let (a, b) = (first.len()?, second.len()?);
    model
        .paired_token_gap_shape(a, b, ids, objective, training)
        .map_err(error)?;
    Ok(Inputs {
        first: tokens(first, a)?,
        second: tokens(second, b)?,
        ids,
        objective,
    })
}
