//! Complete model values and topology, independent of a GPU allocation.
use super::*;
use crate::resident::portable::GraphRecord;
use serde::{Deserialize, Serialize};

pub const BYTE_DECODER_CHECKPOINT_SCHEMA: &str = "spiraltorch.nn.byte_decoder_checkpoint.v1";

/// A complete fixed-window byte model and its attempted SGD revision.
/// Plain SGD has no momentum slots. Data cursors, learning-rate schedules,
/// external biases, runtime/kernel options and acceptance history are caller
/// state, not part of this model checkpoint. Restoring creates a fresh owner.
#[derive(Clone, Debug)]
pub struct ByteDecoderCheckpoint {
    plan: ByteDecoderPlan,
    attempted_revision: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointRecord {
    schema: String,
    update_rule: String,
    window_state: String,
    // A decimal string stays exact even after a browser parses the JSON.
    attempted_revision: String,
    model: ModelRecord,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Table {
    shape: [u32; 2],
    values: Vec<f32>,
}

impl Table {
    fn from_parameter(p: &GraphParameter) -> Result<Self, InferenceError> {
        Ok(Self {
            shape: [
                p.shape[0]
                    .try_into()
                    .map_err(|_| InferenceError::PortableAddressSpace)?,
                p.shape[1]
                    .try_into()
                    .map_err(|_| InferenceError::PortableAddressSpace)?,
            ],
            values: p.values.clone(),
        })
    }

    fn into_tensor(self) -> Result<Tensor, InferenceError> {
        let [rows, cols] = self.shape;
        let len = rows
            .checked_mul(cols)
            .ok_or(InferenceError::PortableAddressSpace)?;
        if len == 0 || len as usize != self.values.len() {
            return Err(InferenceError::ByteDecoder(
                "checkpoint table shape mismatch",
            ));
        }
        Ok(Tensor::from_vec(rows as usize, cols as usize, self.values)?)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometryRecord {
    projection: GraphRecord,
    curvature: f32,
    raw_decay: Vec<f32>,
    raw_phase: Vec<f32>,
    raw_gains: Vec<Vec<f32>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockRecord {
    pre: GraphRecord,
    heads: u32,
    qkv: GraphRecord,
    output: GraphRecord,
    feed_forward: GraphRecord,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelRecord {
    token: Table,
    position: Table,
    geometry: Option<GeometryRecord>,
    blocks: Vec<BlockRecord>,
    head: GraphRecord,
}

fn graph_record(plan: &InferencePlan) -> Result<GraphRecord, InferenceError> {
    GraphRecord::from_graph(&plan.graph_definition()?)
}

impl ModelRecord {
    fn from_plan(plan: &ByteDecoderPlan) -> Result<Self, InferenceError> {
        Ok(Self {
            token: Table::from_parameter(&plan.token)?,
            position: Table::from_parameter(&plan.position)?,
            geometry: plan
                .geometry
                .as_ref()
                .map(|g| {
                    Ok::<_, InferenceError>(GeometryRecord {
                        projection: graph_record(g.projection())?,
                        curvature: g.curvature(),
                        raw_decay: g.raw_decay().to_vec(),
                        raw_phase: g.raw_phase().to_vec(),
                        raw_gains: g.raw_gains().to_vec(),
                    })
                })
                .transpose()?,
            blocks: plan
                .blocks
                .iter()
                .map(|block| {
                    let (pre, attention, feed_forward) = block.plan_parts();
                    let [qkv, output] = attention.projection_parts();
                    Ok(BlockRecord {
                        pre: graph_record(pre)?,
                        heads: attention.attention_spec().query_shape()[1]
                            .try_into()
                            .map_err(|_| InferenceError::PortableAddressSpace)?,
                        qkv: graph_record(qkv)?,
                        output: graph_record(output)?,
                        feed_forward: graph_record(feed_forward)?,
                    })
                })
                .collect::<Result<_, InferenceError>>()?,
            head: graph_record(&plan.head)?,
        })
    }

    fn into_plan(self) -> Result<ByteDecoderPlan, InferenceError> {
        // Generic graph lowering retains a layout per stage. Reject foreign
        // ranks before a tiny JSON can amplify a huge rank across many stages.
        let graphs = std::iter::once(&self.head)
            .chain(self.geometry.iter().map(|g| &g.projection))
            .chain(
                self.blocks
                    .iter()
                    .flat_map(|b| [&b.pre, &b.qkv, &b.output, &b.feed_forward]),
            );
        if graphs.into_iter().any(|g| g.input_rank() != 3) {
            return Err(InferenceError::ByteDecoder(
                "checkpoint graph rank must be three",
            ));
        }
        let blocks = self
            .blocks
            .into_iter()
            .map(|b| {
                let attention = AttentionInferencePlan::from_fused_projection_plans(
                    b.heads as usize,
                    AttentionMask::Causal { query_offset: 0 },
                    b.qkv.into_plan()?,
                    b.output.into_plan()?,
                )?;
                ResidualAttentionPlan::from_plans(
                    &b.pre.into_plan()?,
                    &attention,
                    &b.feed_forward.into_plan()?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = ByteDecoderPlan::from_plans(
            &self.token.into_tensor()?,
            &self.position.into_tensor()?,
            &blocks,
            &self.head.into_plan()?,
        )?;
        match self.geometry {
            None => Ok(plan),
            Some(g) => plan.with_causal_geometry(ByteDecoderGeometryPlan::new(
                &g.projection.into_plan()?,
                &g.raw_decay,
                &g.raw_phase,
                &g.raw_gains,
                g.curvature,
            )?),
        }
    }

    #[cfg(feature = "wgpu")]
    fn values_mut(&mut self) -> impl Iterator<Item = &mut Vec<f32>> {
        [&mut self.token.values, &mut self.position.values]
            .into_iter()
            .chain(self.geometry.iter_mut().flat_map(|g| {
                g.projection
                    .values_mut()
                    .chain([&mut g.raw_decay, &mut g.raw_phase])
                    .chain(g.raw_gains.iter_mut())
            }))
            .chain(self.blocks.iter_mut().flat_map(|b| {
                b.pre
                    .values_mut()
                    .chain(b.qkv.values_mut())
                    .chain(b.output.values_mut())
                    .chain(b.feed_forward.values_mut())
            }))
            .chain(self.head.values_mut())
    }
}

impl ByteDecoderPlan {
    /// Frozen plans are initial values, not the live resident model's weights.
    pub fn initial_checkpoint(&self) -> ByteDecoderCheckpoint {
        ByteDecoderCheckpoint {
            plan: self.clone(),
            attempted_revision: 0,
        }
    }
}

impl ByteDecoderCheckpoint {
    pub fn plan(&self) -> &ByteDecoderPlan {
        &self.plan
    }

    /// Counts attempts, including guarded rejections; not accepted updates.
    pub fn attempted_revision(&self) -> u64 {
        self.attempted_revision
    }

    pub fn to_json(&self) -> Result<String, InferenceError> {
        Ok(serde_json::to_string(&CheckpointRecord {
            schema: BYTE_DECODER_CHECKPOINT_SCHEMA.to_owned(),
            update_rule: "stateless_sgd.v1".to_owned(),
            window_state: "reset_positions_and_geometry.v1".to_owned(),
            attempted_revision: self.attempted_revision.to_string(),
            model: ModelRecord::from_plan(&self.plan)?,
        })?)
    }

    pub fn from_json(payload: &str) -> Result<Self, InferenceError> {
        Self::from_json_with_limit(payload, DEFAULT_MAX_PLAN_JSON_BYTES)
    }

    /// Validates the complete topology before any GPU allocation. Native and
    /// wasm32 use the same fixed-width graph/table address checks.
    pub fn from_json_with_limit(payload: &str, max_bytes: usize) -> Result<Self, InferenceError> {
        if payload.len() > max_bytes {
            return Err(InferenceError::JsonLimit {
                actual: payload.len(),
                limit: max_bytes,
            });
        }
        let record: CheckpointRecord = serde_json::from_str(payload)?;
        if record.schema != BYTE_DECODER_CHECKPOINT_SCHEMA {
            return Err(InferenceError::Schema(record.schema));
        }
        if record.update_rule != "stateless_sgd.v1"
            || record.window_state != "reset_positions_and_geometry.v1"
        {
            return Err(InferenceError::ByteDecoder(
                "unsupported checkpoint learning semantics",
            ));
        }
        let attempted_revision = record
            .attempted_revision
            .parse::<u64>()
            .map_err(|_| InferenceError::ByteDecoder("invalid attempted revision"))?;
        if attempted_revision.to_string() != record.attempted_revision {
            return Err(InferenceError::ByteDecoder(
                "noncanonical attempted revision",
            ));
        }
        Ok(Self {
            plan: record.model.into_plan()?,
            attempted_revision,
        })
    }
}

// The live owner keeps only topology/roles/shapes, never a second full set of
// initial host weights. Empty value vectors are private, never importable plans.
#[cfg(feature = "wgpu")]
pub(super) struct CheckpointTemplate(ModelRecord);

#[cfg(feature = "wgpu")]
impl CheckpointTemplate {
    pub(super) fn new(plan: &ByteDecoderPlan) -> Result<Self, InferenceError> {
        let mut record = ModelRecord::from_plan(plan)?;
        for values in record.values_mut() {
            *values = Vec::new();
        }
        Ok(Self(record))
    }
}

#[cfg(feature = "wgpu")]
mod gpu;
#[cfg(feature = "wgpu")]
pub use gpu::ByteDecoderCheckpointReadback;

#[cfg(test)]
mod tests;
