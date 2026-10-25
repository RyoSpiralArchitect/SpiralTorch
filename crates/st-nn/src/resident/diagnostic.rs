//! Host-only provenance for opt-in captures of private execution graphs.
use super::InferenceError;
use st_backend_wgpu::resident_training::{
    graph::{GraphForward, ResidentGraphAutograd},
    TrainingError,
};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GraphDiagnosticBinding {
    pub model_slot: usize,
    pub flattened: bool,
}

pub(crate) struct GraphDiagnosticSource<'a> {
    pub name: String,
    pub bindings: Vec<GraphDiagnosticBinding>,
    pub graph: &'a ResidentGraphAutograd,
    pub forward: &'a GraphForward,
}

pub(crate) fn diagnostic_bindings(
    range: Range<usize>,
    flattened_router: Option<usize>,
    parameter_count: usize,
) -> Result<Vec<GraphDiagnosticBinding>, InferenceError> {
    let count = range
        .end
        .checked_sub(range.start)
        .and_then(|n| n.checked_add(usize::from(flattened_router.is_some())))
        .ok_or(TrainingError::ParameterLayout)?;
    if count != parameter_count {
        return Err(TrainingError::ParameterLayout.into());
    }
    let mut bindings: Vec<_> = range
        .map(|model_slot| GraphDiagnosticBinding {
            model_slot,
            flattened: false,
        })
        .collect();
    if let Some(model_slot) = flattened_router {
        bindings.push(GraphDiagnosticBinding {
            model_slot,
            flattened: true,
        });
    }
    Ok(bindings)
}

impl<'a> GraphDiagnosticSource<'a> {
    pub fn new(
        name: String,
        range: Range<usize>,
        flattened_router: Option<usize>,
        graph: &'a ResidentGraphAutograd,
        forward: &'a GraphForward,
    ) -> Result<Self, InferenceError> {
        Ok(Self {
            name,
            bindings: diagnostic_bindings(range, flattened_router, graph.parameter_count())?,
            graph,
            forward,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_diagnostic_binding_keeps_appended_router_order() {
        let bindings = diagnostic_bindings(8..10, Some(7), 3).unwrap();
        assert_eq!(
            bindings.iter().map(|b| b.model_slot).collect::<Vec<_>>(),
            [8, 9, 7]
        );
        assert_eq!(
            bindings.iter().map(|b| b.flattened).collect::<Vec<_>>(),
            [false, false, true]
        );
    }

    #[test]
    fn diagnostic_binding_rejects_reversed_and_incomplete_ranges() {
        assert!(diagnostic_bindings(Range { start: 9, end: 8 }, None, 0).is_err());
        assert!(diagnostic_bindings(8..10, Some(7), 2).is_err());
        assert!(diagnostic_bindings(8..10, None, 3).is_err());
        assert!(diagnostic_bindings(0..usize::MAX, Some(0), 0).is_err());
    }

    #[test]
    fn diagnostic_binding_allows_parameterless_graphs() {
        assert!(diagnostic_bindings(12..12, None, 0).unwrap().is_empty());
    }
}
