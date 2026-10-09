//! Causal byte-model composition with one full-model parameter owner.
use super::*;
use std::ops::Range;

mod geometry;
pub use geometry::{ByteDecoderGeometryParameterLayout, ByteDecoderGeometryPlan};

#[cfg(feature = "wgpu")]
mod training;
#[cfg(feature = "wgpu")]
pub use training::{
    ByteDecoderBias, ByteDecoderBiasGradient, ResidentByteBatch, ResidentByteDecoder,
    ResidentByteDecoderForward, ResidentByteDecoderVjp,
};

pub const BYTE_LM_VOCAB: usize = 256;

/// Each row is one already-selected document window of T+1 bytes. Rows never
/// attend to one another. No concatenation, padding, EOS, UTF-8 decoding or
/// normalization is implicit; the caller must not cross a document boundary.
#[derive(Clone, Debug)]
pub struct ByteLmBatch {
    shape: [usize; 2],
    input: Vec<u8>,
    targets: Vec<u8>,
}

impl ByteLmBatch {
    /// Select T+1 bytes without crossing a document boundary. `selections`
    /// contains (document index, byte offset), not character/token offsets.
    pub fn from_documents(
        documents: &[&[u8]],
        selections: &[(usize, usize)],
        steps: usize,
    ) -> Result<Self, InferenceError> {
        if steps == 0 {
            return Err(InferenceError::ByteDecoder("steps must be nonzero"));
        }
        let span = steps
            .checked_add(1)
            .ok_or(InferenceError::PortableAddressSpace)?;
        let windows = selections
            .iter()
            .map(|&(document, start)| {
                let end = start
                    .checked_add(span)
                    .ok_or(InferenceError::PortableAddressSpace)?;
                documents
                    .get(document)
                    .and_then(|d| d.get(start..end))
                    .ok_or(InferenceError::ByteDecoder(
                        "window crosses a document boundary or selection is invalid",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_windows(&windows)
    }

    pub fn from_windows(windows: &[&[u8]]) -> Result<Self, InferenceError> {
        let len = windows
            .first()
            .ok_or(InferenceError::ByteDecoder("empty batch"))?
            .len();
        if len < 2 || windows.iter().any(|row| row.len() != len) {
            return Err(InferenceError::ByteDecoder(
                "windows need equal lengths of at least two bytes",
            ));
        }
        let shape = [windows.len(), len - 1];
        let count = shape[0]
            .checked_mul(shape[1])
            .ok_or(InferenceError::PortableAddressSpace)?;
        u32::try_from(count).map_err(|_| InferenceError::PortableAddressSpace)?;
        let mut input = Vec::new();
        let mut targets = Vec::new();
        for values in [&mut input, &mut targets] {
            values
                .try_reserve_exact(count)
                .map_err(|_| InferenceError::ByteDecoder("batch allocation"))?;
        }
        for window in windows {
            input.extend_from_slice(&window[..len - 1]);
            targets.extend_from_slice(&window[1..]);
        }
        Ok(Self {
            shape,
            input,
            targets,
        })
    }
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn input_bytes(&self) -> &[u8] {
        &self.input
    }
    pub fn target_bytes(&self) -> &[u8] {
        &self.targets
    }
}

/// Untied parameter order: token table, position table, optional geometry,
/// each residual block in order, then the head graph. Ranges remain explicit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteDecoderParameterLayout {
    geometry: Option<ByteDecoderGeometryParameterLayout>,
    blocks: Vec<Range<usize>>,
    head: Range<usize>,
}

impl ByteDecoderParameterLayout {
    pub fn token_embedding(&self) -> usize {
        0
    }
    pub fn position_embedding(&self) -> usize {
        1
    }
    pub fn blocks(&self) -> &[Range<usize>] {
        &self.blocks
    }
    pub fn geometry(&self) -> Option<&ByteDecoderGeometryParameterLayout> {
        self.geometry.as_ref()
    }
    pub fn head(&self) -> Range<usize> {
        self.head.clone()
    }
    pub fn len(&self) -> usize {
        self.head.end
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Frozen learned byte/position tables, causal residual blocks and a tokenwise
/// 256-way head. Positions reset to zero in every fixed-size window. This is a
/// full-window training plan, not a streaming/KV-cache or tied-weight decoder.
#[derive(Clone, Debug)]
pub struct ByteDecoderPlan {
    token: GraphParameter,
    position: GraphParameter,
    geometry: Option<ByteDecoderGeometryPlan>,
    blocks: Vec<ResidualAttentionPlan>,
    head: InferencePlan,
    parameters: ByteDecoderParameterLayout,
}

fn freeze_table(table: &Tensor) -> Result<GraphParameter, InferenceError> {
    let shape = table.shape();
    let table = table.to_layout(Layout::RowMajor)?;
    if table.data().iter().any(|v| !v.is_finite()) {
        return Err(InferenceError::ByteDecoder(
            "embedding parameters must be finite",
        ));
    }
    let len = shape
        .0
        .checked_mul(shape.1)
        .ok_or(InferenceError::PortableAddressSpace)?;
    u32::try_from(len).map_err(|_| InferenceError::PortableAddressSpace)?;
    Ok(GraphParameter {
        role: ParameterRole::Weight,
        shape: vec![shape.0, shape.1],
        values: table.data().to_vec(),
    })
}

impl ByteDecoderPlan {
    pub fn from_plans(
        token_table: &Tensor,
        position_table: &Tensor,
        blocks: &[ResidualAttentionPlan],
        head: &InferencePlan,
    ) -> Result<Self, InferenceError> {
        let first = blocks.first().ok_or(InferenceError::ByteDecoder(
            "at least one causal block is required",
        ))?;
        let input = first.input_layout();
        let shape = input.shape();
        if shape.len() != 3 || input.is_empty() || !input.is_contiguous() || input.offset() != 0 {
            return Err(InferenceError::InvalidLayout);
        }
        if token_table.shape() != (BYTE_LM_VOCAB, shape[2])
            || position_table.shape().0 < shape[1]
            || position_table.shape().1 != shape[2]
        {
            return Err(InferenceError::ByteDecoder(
                "byte/position table shapes do not match the blocks",
            ));
        }
        let mut offset = 2usize;
        let mut ranges = Vec::new();
        for block in blocks {
            if block.input_layout() != input
                || block.output_layout() != input
                || block.attention_spec().mask() != (AttentionMask::Causal { query_offset: 0 })
            {
                return Err(InferenceError::ByteDecoder(
                    "blocks must preserve layout and use full-sequence causal masking",
                ));
            }
            let end = offset
                .checked_add(block.parameter_count()?)
                .ok_or(InferenceError::PortableAddressSpace)?;
            ranges.push(offset..end);
            offset = end;
        }
        if head.input_layout() != input
            || head.output_layout().shape() != [shape[0], shape[1], BYTE_LM_VOCAB]
        {
            return Err(InferenceError::ByteDecoder(
                "head must produce [batch, sequence, 256] logits",
            ));
        }
        // GraphDefinition only admits tokenwise stages; no sequence-mixing
        // reductions/views can silently bypass attention's causal boundary.
        let head_count = head.graph_definition()?.parameters().len();
        let end = offset
            .checked_add(head_count)
            .ok_or(InferenceError::PortableAddressSpace)?;
        Ok(Self {
            token: freeze_table(token_table)?,
            position: freeze_table(position_table)?,
            geometry: None,
            blocks: blocks.to_vec(),
            head: head.clone(),
            parameters: ByteDecoderParameterLayout {
                geometry: None,
                blocks: ranges,
                head: offset..end,
            },
        })
    }

    pub fn input_layout(&self) -> &NdLayout {
        self.blocks[0].input_layout()
    }
    pub fn output_layout(&self) -> &NdLayout {
        self.head.output_layout()
    }
    pub fn parameter_layout(&self) -> &ByteDecoderParameterLayout {
        &self.parameters
    }
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
    pub fn position_capacity(&self) -> usize {
        self.position.shape[0]
    }
    pub fn embedding_width(&self) -> usize {
        self.token.shape[1]
    }
}

#[cfg(test)]
mod tests;
