//! Complete model values and topology, independent of a GPU allocation.
use super::*;
use crate::resident::portable::GraphRecord;
use serde::{Deserialize, Serialize};
use st_kernel_contracts::adamw::AdamWState;
mod optimizer;
use optimizer::AdamWRecord;

pub const BYTE_DECODER_CHECKPOINT_SCHEMA: &str = "spiraltorch.nn.byte_decoder_checkpoint.v1";
pub const BYTE_DECODER_METRIC_CHECKPOINT_SCHEMA: &str = "spiraltorch.nn.byte_decoder_checkpoint.v2";
pub const BYTE_DECODER_HEADWISE_CHECKPOINT_SCHEMA: &str =
    "spiraltorch.nn.byte_decoder_checkpoint.v3";
pub const BYTE_DECODER_MIXTURE_CHECKPOINT_SCHEMA: &str =
    "spiraltorch.nn.byte_decoder_checkpoint.v4";
pub const BYTE_DECODER_QUERY_MIXTURE_CHECKPOINT_SCHEMA: &str =
    "spiraltorch.nn.byte_decoder_checkpoint.v5";
pub const BYTE_DECODER_ADAMW_CHECKPOINT_SCHEMA: &str = "spiraltorch.nn.byte_decoder_checkpoint.v6";

/// A complete fixed-window byte model and its attempted update revision.
/// AdamW checkpoints also own moments and accepted per-tensor clocks.
/// Data cursors, learning-rate/decay schedules,
/// external biases, runtime/kernel options and acceptance history are caller
/// state, not part of this model checkpoint. Restoring creates a fresh owner.
#[derive(Clone, Debug)]
pub struct ByteDecoderCheckpoint {
    plan: ByteDecoderPlan,
    attempted_revision: u64,
    optimizer: Option<AdamWState>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointRecord {
    schema: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "explicit_model_schema"
    )]
    model_schema: Option<String>,
    update_rule: String,
    window_state: String,
    // A decimal string stays exact even after a browser parses the JSON.
    attempted_revision: String,
    model: ModelRecord,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optimizer::explicit_optimizer"
    )]
    optimizer: Option<AdamWRecord>,
}

fn explicit_model_schema<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<String>, D::Error> {
    String::deserialize(d).map(Some)
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
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "explicit_metric"
    )]
    pair_metric: Option<ByteDecoderPairMetric>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "explicit_head_metrics"
    )]
    head_pair_metrics: Option<Vec<Vec<ByteDecoderPairMetric>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "explicit_mixture"
    )]
    mixture: Option<MixtureRecord>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixtureRecord {
    metrics: Vec<ByteDecoderPairMetric>,
    scales: Vec<Vec<f32>>,
    raw_logits: Vec<Vec<f32>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "explicit_query_projections"
    )]
    query_projections: Option<Vec<GraphRecord>>,
}

fn explicit_query_projections<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<Vec<GraphRecord>>, D::Error> {
    Vec::<GraphRecord>::deserialize(d).map(Some)
}

fn explicit_mixture<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<MixtureRecord>, D::Error> {
    MixtureRecord::deserialize(d).map(Some)
}

fn explicit_metric<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<ByteDecoderPairMetric>, D::Error> {
    ByteDecoderPairMetric::deserialize(d).map(Some)
}

fn explicit_head_metrics<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<Vec<Vec<ByteDecoderPairMetric>>>, D::Error> {
    Vec::<Vec<ByteDecoderPairMetric>>::deserialize(d).map(Some)
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
    fn checkpoint_schema(&self) -> &'static str {
        if self
            .geometry
            .as_ref()
            .and_then(|g| g.mixture.as_ref())
            .is_some_and(|m| m.query_projections.is_some())
        {
            return BYTE_DECODER_QUERY_MIXTURE_CHECKPOINT_SCHEMA;
        }
        if self.geometry.as_ref().is_some_and(|g| g.mixture.is_some()) {
            return BYTE_DECODER_MIXTURE_CHECKPOINT_SCHEMA;
        }
        if self
            .geometry
            .as_ref()
            .is_some_and(|g| g.head_pair_metrics.is_some())
        {
            return BYTE_DECODER_HEADWISE_CHECKPOINT_SCHEMA;
        }
        if self
            .geometry
            .as_ref()
            .is_some_and(|g| g.pair_metric.is_some())
        {
            BYTE_DECODER_METRIC_CHECKPOINT_SCHEMA
        } else {
            BYTE_DECODER_CHECKPOINT_SCHEMA
        }
    }
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
                        pair_metric: g
                            .pair_metric()
                            .filter(|&m| m != ByteDecoderPairMetric::PoincareSquared),
                        head_pair_metrics: g
                            .head_pair_metrics()
                            .filter(|_| g.pair_metric().is_none())
                            .map(<[_]>::to_vec),
                        mixture: g
                            .mixture()
                            .map(|m| {
                                Ok::<_, InferenceError>(MixtureRecord {
                                    metrics: m.metrics().to_vec(),
                                    scales: m.scales().to_vec(),
                                    raw_logits: m.raw_logits().to_vec(),
                                    query_projections: if m.query_projections().is_empty() {
                                        None
                                    } else {
                                        Some(
                                            m.query_projections()
                                                .iter()
                                                .map(graph_record)
                                                .collect::<Result<Vec<_>, _>>()?,
                                        )
                                    },
                                })
                            })
                            .transpose()?,
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
            .chain(self.geometry.iter().flat_map(|g| {
                g.mixture
                    .iter()
                    .flat_map(|m| m.query_projections.iter().flatten())
            }))
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
            Some(g) => {
                let geometry = ByteDecoderGeometryPlan::new(
                    &g.projection.into_plan()?,
                    &g.raw_decay,
                    &g.raw_phase,
                    &g.raw_gains,
                    g.curvature,
                )?;
                if let Some(mixture) = g.mixture {
                    if g.pair_metric.is_some() || g.head_pair_metrics.is_some() {
                        return Err(InferenceError::ByteDecoder(
                            "mixture conflicts with fixed metric routing",
                        ));
                    }
                    let geometry = geometry.with_mixture(
                        &mixture.metrics,
                        &mixture.raw_logits,
                        &mixture.scales,
                    )?;
                    let geometry = if let Some(projections) = mixture.query_projections {
                        let projections = projections
                            .into_iter()
                            .map(GraphRecord::into_plan)
                            .collect::<Result<Vec<_>, _>>()?;
                        geometry.with_query_projections(&projections)?
                    } else {
                        geometry
                    };
                    return plan.with_causal_geometry(geometry);
                }
                let geometry = match g.head_pair_metrics {
                    Some(metrics) => {
                        if g.pair_metric.is_some() {
                            return Err(InferenceError::ByteDecoder(
                                "conflicting geometry metric routes",
                            ));
                        }
                        let routed = geometry.with_head_pair_metrics(&metrics)?;
                        if routed.pair_metric().is_some() {
                            return Err(InferenceError::ByteDecoder(
                                "uniform routing requires the canonical metric schema",
                            ));
                        }
                        routed
                    }
                    None => geometry.with_pair_metric(
                        g.pair_metric
                            .unwrap_or(ByteDecoderPairMetric::PoincareSquared),
                    ),
                };
                plan.with_causal_geometry(geometry)
            }
        }
    }

    fn values_mut(&mut self) -> impl Iterator<Item = &mut Vec<f32>> {
        [&mut self.token.values, &mut self.position.values]
            .into_iter()
            .chain(self.geometry.iter_mut().flat_map(|g| {
                g.projection
                    .values_mut()
                    .chain([&mut g.raw_decay, &mut g.raw_phase])
                    .chain(g.raw_gains.iter_mut())
                    .chain(g.mixture.iter_mut().flat_map(|m| {
                        m.raw_logits.iter_mut().chain(
                            m.query_projections
                                .iter_mut()
                                .flatten()
                                .flat_map(GraphRecord::values_mut),
                        )
                    }))
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
            optimizer: None,
        }
    }
}

impl ByteDecoderCheckpoint {
    pub(crate) fn with_zero_adamw(
        mut self,
        config: st_kernel_contracts::adamw::AdamWConfig,
    ) -> Result<Self, InferenceError> {
        if self.optimizer.is_some() {
            return Err(InferenceError::ByteDecoder(
                "optimizer history already exists",
            ));
        }
        self.optimizer = Some(AdamWState {
            config,
            slots: self
                .plan
                .parameter_shapes()?
                .into_iter()
                .map(|shape| {
                    let len = shape.iter().product();
                    st_kernel_contracts::adamw::AdamWSlot {
                        shape,
                        first: vec![0.; len],
                        second: vec![0.; len],
                        steps: 0,
                    }
                })
                .collect(),
        });
        Ok(self)
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn adamw_json(&self) -> Result<serde_json::Value, InferenceError> {
        Ok(match &self.optimizer {
            Some(state) => serde_json::to_value(AdamWRecord::from_state(state)?)?,
            None => serde_json::Value::Null,
        })
    }

    pub(crate) fn geometry_parameters_match(&self, other: &Self) -> Result<bool, InferenceError> {
        match (&self.plan.geometry, &other.plan.geometry) {
            (None, None) => Ok(true),
            (Some(a), Some(b)) => {
                let (a, b) = (a.parameter_values()?, b.parameter_values()?);
                Ok(a.len() == b.len()
                    && a.iter().zip(&b).all(|(a, b)| {
                        a.shape == b.shape
                            && a.values.len() == b.values.len()
                            && a.values
                                .iter()
                                .zip(&b.values)
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                    }))
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn escaped_json_size_bound(&self) -> Result<usize, InferenceError> {
        let mut model = ModelRecord::from_plan(&self.plan)?;
        let mut count = 0usize;
        for values in model.values_mut() {
            count = count
                .checked_add(values.len())
                .ok_or(InferenceError::PortableAddressSpace)?;
            values.clear();
        }
        let mut optimizer = self
            .optimizer
            .as_ref()
            .map(AdamWRecord::from_state)
            .transpose()?;
        if let Some(optimizer) = &mut optimizer {
            for slot in &mut optimizer.slots {
                count = count
                    .checked_add(slot.first.len())
                    .and_then(|n| n.checked_add(slot.second.len()))
                    .ok_or(InferenceError::PortableAddressSpace)?;
                slot.first.clear();
                slot.second.clear();
            }
        }
        let empty = serde_json::to_string(&CheckpointRecord {
            schema: self.schema(&model).to_owned(),
            model_schema: self
                .optimizer
                .as_ref()
                .map(|_| model.checkpoint_schema().to_owned()),
            update_rule: self.update_rule().to_owned(),
            window_state: "reset_positions_and_geometry.v1".to_owned(),
            attempted_revision: u64::MAX.to_string(),
            model,
            optimizer,
        })?;
        // serde_json's finite-float formatter has a 24-byte buffer. Reserve 32
        // per value including separators; digits never need string escaping.
        let escaped_bytes = serde_json::to_string(&empty)?.len();
        count
            .checked_mul(32)
            .and_then(|n| n.checked_add(escaped_bytes))
            .ok_or(InferenceError::PortableAddressSpace)
    }

    pub(crate) fn topology_json(&self) -> Result<String, InferenceError> {
        let mut record = ModelRecord::from_plan(&self.plan)?;
        for values in record.values_mut() {
            values.clear();
        }
        Ok(serde_json::to_string(&record)?)
    }

    pub fn plan(&self) -> &ByteDecoderPlan {
        &self.plan
    }

    /// Counts attempts, including guarded rejections; not accepted updates.
    pub fn attempted_revision(&self) -> u64 {
        self.attempted_revision
    }

    pub fn adamw_state(&self) -> Option<&AdamWState> {
        self.optimizer.as_ref()
    }

    fn schema(&self, model: &ModelRecord) -> &'static str {
        if self.optimizer.is_some() {
            BYTE_DECODER_ADAMW_CHECKPOINT_SCHEMA
        } else {
            model.checkpoint_schema()
        }
    }

    fn update_rule(&self) -> &'static str {
        if self.optimizer.is_some() {
            "adamw_zero_rate_freeze.v1"
        } else {
            "stateless_sgd.v1"
        }
    }

    pub fn to_json(&self) -> Result<String, InferenceError> {
        let model = ModelRecord::from_plan(&self.plan)?;
        Ok(serde_json::to_string(&CheckpointRecord {
            schema: self.schema(&model).to_owned(),
            model_schema: self
                .optimizer
                .as_ref()
                .map(|_| model.checkpoint_schema().to_owned()),
            update_rule: self.update_rule().to_owned(),
            window_state: "reset_positions_and_geometry.v1".to_owned(),
            attempted_revision: self.attempted_revision.to_string(),
            model,
            optimizer: self
                .optimizer
                .as_ref()
                .map(AdamWRecord::from_state)
                .transpose()?,
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
        if (record.optimizer.is_some()
            && record.model_schema.as_deref() != Some(record.model.checkpoint_schema()))
            || (record.optimizer.is_none() && record.model_schema.is_some())
        {
            return Err(InferenceError::ByteDecoder(
                "checkpoint optimizer/model topology discriminator differs",
            ));
        }
        let expected_schema = if record.optimizer.is_some() {
            BYTE_DECODER_ADAMW_CHECKPOINT_SCHEMA
        } else {
            record.model.checkpoint_schema()
        };
        if record.schema != expected_schema
            || record
                .model
                .geometry
                .as_ref()
                .is_some_and(|g| g.pair_metric == Some(ByteDecoderPairMetric::PoincareSquared))
        {
            return Err(InferenceError::Schema(record.schema));
        }
        let expected_rule = if record.optimizer.is_some() {
            "adamw_zero_rate_freeze.v1"
        } else {
            "stateless_sgd.v1"
        };
        if record.update_rule != expected_rule
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
        let plan = record.model.into_plan()?;
        let optimizer = record
            .optimizer
            .map(|o| o.into_state(&plan, attempted_revision))
            .transpose()?;
        Ok(Self {
            plan,
            attempted_revision,
            optimizer,
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
mod optimizer_tests;
#[cfg(test)]
mod tests;
