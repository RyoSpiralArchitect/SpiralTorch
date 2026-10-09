//! Compose existing graph operations and attention with two explicit residuals.
use super::*;

#[cfg(feature = "wgpu")]
mod training;
#[cfg(feature = "wgpu")]
pub use training::{
    ResidentResidualAttentionForward, ResidentResidualAttentionTraining,
    ResidentResidualAttentionVjp,
};

/// Immutable composition `y = x + attention(pre(x)); output = y + feed_forward(y)`.
///
/// A pre-LayerNorm transformer block uses a LayerNorm `pre` and a
/// LayerNorm -> Linear -> GELU -> Linear `feed_forward`. Both branches are
/// existing portable InferencePlans, so explicit Topos/Scaler operations keep
/// their semantics and trainable parameters instead of being silently discarded.
/// This is one block, not token embedding, a KV cache, or a complete decoder.
#[derive(Clone, Debug)]
pub struct ResidualAttentionPlan {
    pre: InferencePlan,
    attention: AttentionInferencePlan,
    feed_forward: InferencePlan,
}

impl ResidualAttentionPlan {
    /// Clone already frozen plans. Exact logical layouts must match at every
    /// edge and residual; equal flattened element counts are not sufficient.
    pub fn from_plans(
        pre: &InferencePlan,
        attention: &AttentionInferencePlan,
        feed_forward: &InferencePlan,
    ) -> Result<Self, InferenceError> {
        if pre.input_layout().rank() != 3
            || pre.output_layout() != attention.input_layout()
            || attention.output_layout() != pre.input_layout()
            || feed_forward.input_layout() != pre.input_layout()
            || feed_forward.output_layout() != pre.input_layout()
        {
            return Err(InferenceError::Attention(
                "residual attention requires matching logical layouts at every branch",
            ));
        }
        pre.graph_definition()?;
        feed_forward.graph_definition()?;
        Ok(Self {
            pre: pre.clone(),
            attention: attention.clone(),
            feed_forward: feed_forward.clone(),
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.pre.input_layout()
    }

    pub fn output_layout(&self) -> &NdLayout {
        self.feed_forward.output_layout()
    }

    pub fn attention_spec(&self) -> AttentionSpec {
        self.attention.attention_spec()
    }
}

#[cfg(test)]
mod tests;
