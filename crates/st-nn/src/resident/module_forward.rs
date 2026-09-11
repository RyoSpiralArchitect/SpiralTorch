//! Reuse the resident graph behind the original Module, without host readback.
use super::*;
use st_backend_wgpu::{
    resident_graph::{GraphInferenceError, ResidentGraph},
    resident_tensor::ResidentTensor,
};
use std::cell::RefCell;

/// Host submission/cache diagnostics, not GPU completion or validation receipts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ResidentForwardStats {
    pub compilations: u64,
    pub cache_hits: u64,
    pub submitted_forwards: u64,
}

struct Cached {
    operations: Vec<InferenceOp>,
    graph: ResidentGraph,
}

#[derive(Default)]
struct State {
    current: Option<Cached>,
    stats: ResidentForwardStats,
}

/// One bounded, replaceable workspace per Module. Returned tensors own their
/// GPU captures and survive workspace rebuild/clear and Module destruction.
#[derive(Default)]
pub struct ResidentForwardCache(RefCell<State>);

impl std::fmt::Debug for ResidentForwardCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentForwardCache")
            .field("stats", &self.stats())
            .finish()
    }
}

pub(crate) fn require_uncommitted_route() -> Result<(), InferenceError> {
    if crate::execution::current_backend_policy()
        .is_some_and(|p| p.runtime_plan_output_sha256().is_some())
    {
        return Err(InferenceError::ResidentForwardPolicy);
    }
    Ok(())
}

pub(crate) fn unary_forward(
    input: &ResidentTensor,
    op: st_kernel_contracts::elementwise::ElementwiseOp,
) -> Result<ResidentTensor, InferenceError> {
    require_uncommitted_route()?;
    Ok(input.apply(op, None).map_err(GraphInferenceError::from)?)
}

fn same_tensor(a: &Tensor, b: &Tensor) -> bool {
    a.shape() == b.shape()
        && a.layout() == b.layout()
        && ((a.is_snapshot() && b.is_snapshot() && a.data().as_ptr() == b.data().as_ptr())
            || a.data()
                .iter()
                .zip(b.data())
                .all(|(a, b)| a.to_bits() == b.to_bits()))
}

fn same_operations(a: &[InferenceOp], b: &[InferenceOp]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| match (a, b) {
            (
                InferenceOp::Linear {
                    weight: aw,
                    bias: ab,
                },
                InferenceOp::Linear {
                    weight: bw,
                    bias: bb,
                },
            ) => same_tensor(aw, bw) && same_tensor(ab, bb),
            (InferenceOp::Scale { gain: a }, InferenceOp::Scale { gain: b }) => same_tensor(a, b),
            (InferenceOp::Gelu, InferenceOp::Gelu) | (InferenceOp::Relu, InferenceOp::Relu) => true,
            _ => false,
        })
}

impl ResidentForwardCache {
    pub fn stats(&self) -> ResidentForwardStats {
        self.0.borrow().stats
    }

    /// Release this workspace, without invalidating already returned tensors.
    /// Lifetime counters are retained; the next forward compiles again.
    pub fn clear(&self) {
        self.0.borrow_mut().current = None;
    }

    /// Submit checked descriptors on the input device, reusing an exact matching
    /// graph. Unknown/invalid operations fail; no implicit host readback occurs.
    pub fn forward(
        &self,
        operations: Vec<InferenceOp>,
        input: &ResidentTensor,
    ) -> Result<ResidentTensor, InferenceError> {
        require_uncommitted_route()?;
        if operations.is_empty() {
            return Ok(input.clone());
        }
        let layout = NdLayout::contiguous(input.layout().shape())?;
        let mut state = self.0.borrow_mut();
        let reuse = state.current.as_ref().is_some_and(|cached| {
            cached.graph.input_layout() == &layout
                && cached
                    .graph
                    .tensor_device()
                    .runtime()
                    .context()
                    .shares_handles_with(input.device().runtime().context())
                && same_operations(&cached.operations, &operations)
        });
        let increment = |value: u64| {
            value
                .checked_add(1)
                .ok_or(GraphInferenceError::CounterOverflow)
        };
        let submissions = increment(state.stats.submitted_forwards)?;
        if reuse {
            state.stats.cache_hits = increment(state.stats.cache_hits)?;
        } else {
            let compilations = increment(state.stats.compilations)?;
            // Freeze once on a miss. Mutable/foreign values are compared by bits
            // on every reuse; pointer identity alone cannot detect producer writes.
            let frozen: Vec<_> = operations.iter().map(InferenceOp::snapshot).collect();
            let plan = InferencePlan::from_operations(layout, frozen.clone())?;
            let graph = plan.compile_graph_wgpu(input.device().runtime().clone())?;
            state.current = Some(Cached {
                operations: frozen,
                graph,
            });
            state.stats.compilations = compilations;
        }
        let cached = state.current.as_mut().unwrap();
        cached.graph.set_input_tensor(input)?;
        cached.graph.dispatch()?;
        state.stats.submitted_forwards = submissions;
        Ok(state.current.as_ref().unwrap().graph.output_tensor()?)
    }
}

#[cfg(test)]
mod tests;
