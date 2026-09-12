//! Reuse the resident graph behind the original Module, without host readback.
use super::*;
use st_backend_wgpu::{
    resident_graph::{GraphInferenceError, ResidentGraph},
    resident_tensor::{ResidentTensor, TensorReadback},
};
use st_tensor::TensorContentStamp;
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
    stamps: Vec<OperationStamp>,
    graph: ResidentGraph,
}

enum OperationStamp {
    Linear {
        weight: Option<TensorContentStamp>,
        bias: Option<TensorContentStamp>,
    },
    Scale(Option<TensorContentStamp>),
    Gelu,
    Relu,
}

impl OperationStamp {
    fn capture(operation: &InferenceOp) -> Self {
        match operation {
            InferenceOp::Linear { weight, bias } => Self::Linear {
                weight: weight.content_stamp(),
                bias: bias.content_stamp(),
            },
            InferenceOp::Scale { gain } => Self::Scale(gain.content_stamp()),
            InferenceOp::Gelu => Self::Gelu,
            InferenceOp::Relu => Self::Relu,
        }
    }
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
    // The tensor binding remains active even under an uncommitted nested NN
    // policy, and can also be installed directly by a Rust caller.
    if st_tensor::execution::current_execution_plan_binding().is_some() {
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

pub(crate) fn unary_snapshot(
    input: &ResidentTensor,
    op: st_kernel_contracts::elementwise::ElementwiseOp,
) -> Result<TensorReadback, InferenceError> {
    require_uncommitted_route()?;
    Ok(input
        .apply_snapshot(op, None)
        .map_err(GraphInferenceError::from)?)
}

fn same_tensor(a: &Tensor, b: &Tensor) -> bool {
    #[cfg(test)]
    tests::EXACT_COMPARISONS.with(|count| count.set(count.get() + 1));
    a.shape() == b.shape()
        && a.layout() == b.layout()
        && ((a.is_snapshot() && b.is_snapshot() && a.data().as_ptr() == b.data().as_ptr())
            || a.data()
                .iter()
                .zip(b.data())
                .all(|(a, b)| a.to_bits() == b.to_bits()))
}

fn same_parameter(
    frozen: &Tensor,
    current: &Tensor,
    stamp: &mut Option<TensorContentStamp>,
) -> bool {
    if stamp.as_ref().is_some_and(|stamp| stamp.matches(current)) {
        return true;
    }
    *stamp = None;
    if !same_tensor(frozen, current) {
        return false;
    }
    // Equal-value replacement can acquire a fresh witness without recompiling.
    *stamp = current.content_stamp();
    true
}

fn same_operations(a: &[InferenceOp], b: &[InferenceOp], stamps: &mut [OperationStamp]) -> bool {
    a.len() == b.len()
        && a.len() == stamps.len()
        && a.iter()
            .zip(b)
            .zip(stamps)
            .all(|((a, b), stamp)| match (a, b, stamp) {
                (
                    InferenceOp::Linear {
                        weight: aw,
                        bias: ab,
                    },
                    InferenceOp::Linear {
                        weight: bw,
                        bias: bb,
                    },
                    OperationStamp::Linear { weight, bias },
                ) => same_parameter(aw, bw, weight) && same_parameter(ab, bb, bias),
                (
                    InferenceOp::Scale { gain: a },
                    InferenceOp::Scale { gain: b },
                    OperationStamp::Scale(stamp),
                ) => same_parameter(a, b, stamp),
                (InferenceOp::Gelu, InferenceOp::Gelu, OperationStamp::Gelu)
                | (InferenceOp::Relu, InferenceOp::Relu, OperationStamp::Relu) => true,
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

    /// Terminal capture using the same parameters, cache and guard semantics.
    /// An empty sequence captures the input without counting an NN forward.
    pub fn snapshot(
        &self,
        operations: Vec<InferenceOp>,
        input: &ResidentTensor,
    ) -> Result<TensorReadback, InferenceError> {
        Ok(self
            .forward(operations, input)?
            .snapshot()
            .map_err(GraphInferenceError::from)?)
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
        let reuse = state.current.as_mut().is_some_and(|cached| {
            cached.graph.input_layout() == &layout
                && cached
                    .graph
                    .tensor_device()
                    .runtime()
                    .context()
                    .shares_handles_with(input.device().runtime().context())
                && same_operations(&cached.operations, &operations, &mut cached.stamps)
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
            // Freeze on a miss. Owned values use revocable weak witnesses;
            // external values still require exact comparison on every reuse.
            let frozen: Vec<_> = operations.iter().map(InferenceOp::snapshot).collect();
            let plan = InferencePlan::from_operations(layout, frozen.clone())?;
            let graph = plan.compile_graph_wgpu(input.device().runtime().clone())?;
            state.current = Some(Cached {
                operations: frozen,
                stamps: operations.iter().map(OperationStamp::capture).collect(),
                graph,
            });
            state.stats.compilations = compilations;
        }
        let cached = state.current.as_mut().unwrap();
        let output = cached.graph.forward_tensor(input)?;
        state.stats.submitted_forwards = submissions;
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
