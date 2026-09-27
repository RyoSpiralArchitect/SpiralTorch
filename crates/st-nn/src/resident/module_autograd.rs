//! Exact resident VJPs for module-owned, parameterized subgraphs.

use super::module_forward::{same_operations, OperationStamp};
use super::*;
use st_backend_wgpu::{
    resident_tensor::ResidentTensor,
    resident_training::graph::{GraphGradients, ResidentGraphAutograd},
};
use std::cell::RefCell;

/// Host submission/cache diagnostics, not a numerical-acceptance receipt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentAutogradStats {
    pub compilations: u64,
    pub cache_hits: u64,
    pub submitted_vjps: u64,
}

struct Cached {
    operations: Vec<InferenceOp>,
    stamps: Vec<OperationStamp>,
    graph: ResidentGraphAutograd,
}

#[derive(Default)]
struct State {
    current: Option<Cached>,
    stats: ResidentAutogradStats,
}

/// Reuses one frozen graph until its logical input shape, device, or parameter
/// values change. Returned gradients own GPU captures beyond cache replacement.
#[derive(Default)]
pub struct ResidentModuleAutogradCache(RefCell<State>);

impl std::fmt::Debug for ResidentModuleAutogradCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentModuleAutogradCache")
            .field("stats", &self.stats())
            .finish()
    }
}

impl ResidentModuleAutogradCache {
    pub fn stats(&self) -> ResidentAutogradStats {
        self.0.borrow().stats
    }

    pub fn clear(&self) {
        self.0.borrow_mut().current = None;
    }

    /// Return an exact graph VJP in operation parameter order. No host
    /// readback, implicit parameter update, or loss reduction is performed.
    pub fn vjp(
        &self,
        operations: Vec<InferenceOp>,
        input: &ResidentTensor,
        cotangent: &ResidentTensor,
    ) -> Result<GraphGradients, InferenceError> {
        require_uncommitted_route()?;
        let layout = NdLayout::contiguous(input.layout().shape())?;
        let mut state = self.0.borrow_mut();
        let reusable = state.current.as_mut().is_some_and(|cached| {
            cached.graph.input_layout() == &layout
                && cached
                    .graph
                    .tensor_device()
                    .runtime()
                    .context()
                    .shares_handles_with(input.device().runtime().context())
                && same_operations(&cached.operations, &operations, &mut cached.stamps)
        });
        let next_vjps = state
            .stats
            .submitted_vjps
            .checked_add(1)
            .ok_or(st_backend_wgpu::resident_graph::GraphInferenceError::CounterOverflow)?;
        let next_hits = if reusable {
            state
                .stats
                .cache_hits
                .checked_add(1)
                .ok_or(st_backend_wgpu::resident_graph::GraphInferenceError::CounterOverflow)?
        } else {
            state.stats.cache_hits
        };
        if !reusable {
            let next_compilations = state
                .stats
                .compilations
                .checked_add(1)
                .ok_or(st_backend_wgpu::resident_graph::GraphInferenceError::CounterOverflow)?;
            let frozen: Vec<_> = operations.iter().map(InferenceOp::snapshot).collect();
            let plan = InferencePlan::from_operations(layout, frozen.clone())?;
            let graph = plan.compile_graph_autograd_wgpu(input.device().runtime().clone())?;
            state.current = Some(Cached {
                operations: frozen,
                stamps: operations.iter().map(OperationStamp::capture).collect(),
                graph,
            });
            state.stats.compilations = next_compilations;
        }
        let cached = state.current.as_mut().expect("compiled resident graph");
        cached.graph.set_input_tensor(input)?;
        let forward = cached.graph.forward()?;
        let gradients = cached.graph.backward(&forward, cotangent)?;
        state.stats.cache_hits = next_hits;
        state.stats.submitted_vjps = next_vjps;
        Ok(gradients)
    }
}
