// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{RoundtableBandSignal, ZSpaceGraphNetwork};
use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::schedule::GradientBands;
use crate::{PureResult, Tensor, TensorError};
use st_core::backend::device_caps::DeviceCaps;
use st_core::ops::zspace_round::RoundtableBand;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorUtilBackend};
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::ops::Range;

/// Graph-level pooling strategy applied after node-wise message passing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphReadout {
    /// Average every node embedding into one graph embedding.
    #[default]
    Mean,
    /// Sum every node embedding into one graph embedding.
    Sum,
    /// Take the feature-wise maximum node activation.
    Max,
}

impl GraphReadout {
    /// Builds a mean-pooling readout.
    pub fn mean() -> Self {
        Self::Mean
    }

    /// Builds a sum-pooling readout.
    pub fn sum() -> Self {
        Self::Sum
    }

    /// Builds a max-pooling readout.
    pub fn max() -> Self {
        Self::Max
    }

    /// Stable label used in tensor backend traces.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Sum => "sum",
            Self::Max => "max",
        }
    }

    /// Pools node embeddings into a single graph embedding.
    pub fn forward(&self, node_features: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = validate_node_features(node_features)?;
        let values = rows.saturating_mul(cols);
        let tensor_util_backend = current_tensor_util_backend_for_values(values);
        let (output, meta_backend, kernel) = match self {
            Self::Mean => {
                let data = node_features
                    .try_sum_axis0_scaled_with_backend(1.0 / rows as f32, tensor_util_backend)?;
                (
                    Tensor::from_vec(1, cols, data)?,
                    "composite",
                    "graph_readout.mean.sum_axis0_scaled",
                )
            }
            Self::Sum => {
                let data = node_features.try_sum_axis0_with_backend(tensor_util_backend)?;
                (
                    Tensor::from_vec(1, cols, data)?,
                    "composite",
                    "graph_readout.sum.sum_axis0",
                )
            }
            Self::Max => {
                let mut data = vec![f32::NEG_INFINITY; cols];
                for row in 0..rows {
                    let offset = row * cols;
                    for col in 0..cols {
                        data[col] = data[col].max(node_features.data()[offset + col]);
                    }
                }
                (Tensor::from_vec(1, cols, data)?, "cpu", "scalar")
            }
        };
        emit_graph_readout_meta(
            "graph_readout",
            *self,
            rows,
            cols,
            output.shape(),
            meta_backend,
            tensor_util_backend,
            kernel,
        );
        Ok(output)
    }

    /// Broadcasts graph-level gradients back to node embeddings.
    pub fn backward(&self, node_features: &Tensor, grad_graph: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = validate_node_features(node_features)?;
        if grad_graph.shape() != (1, cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_graph.shape(),
                right: (1, cols),
            });
        }
        let values = rows.saturating_mul(cols);
        let tensor_util_backend = current_tensor_util_backend_for_values(values);
        let mut output = Tensor::zeros(rows, cols)?;
        let mut meta_backend = "cpu";
        let mut kernel = "scalar";
        match self {
            Self::Mean => {
                let scale = 1.0 / rows as f32;
                let row = grad_graph
                    .data()
                    .iter()
                    .map(|value| value * scale)
                    .collect::<Vec<_>>();
                output.add_row_inplace_with_backend(&row, tensor_util_backend)?;
                meta_backend = "composite";
                kernel = "graph_readout_backward.mean.add_row";
            }
            Self::Sum => {
                output.add_row_inplace_with_backend(grad_graph.data(), tensor_util_backend)?;
                meta_backend = "composite";
                kernel = "graph_readout_backward.sum.add_row";
            }
            Self::Max => {
                let max_values = self.forward(node_features)?;
                let mut tie_counts = vec![0usize; cols];
                for row in 0..rows {
                    let offset = row * cols;
                    for col in 0..cols {
                        if node_features.data()[offset + col] == max_values.data()[col] {
                            tie_counts[col] += 1;
                        }
                    }
                }
                let data = output.data_mut();
                for row in 0..rows {
                    let offset = row * cols;
                    for col in 0..cols {
                        if node_features.data()[offset + col] == max_values.data()[col] {
                            data[offset + col] = grad_graph.data()[col] / tie_counts[col] as f32;
                        }
                    }
                }
            }
        }
        emit_graph_readout_meta(
            "graph_readout_backward",
            *self,
            rows,
            cols,
            output.shape(),
            meta_backend,
            tensor_util_backend,
            kernel,
        );
        Ok(output)
    }
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn emit_graph_readout_meta(
    op_name: &'static str,
    readout: GraphReadout,
    node_rows: usize,
    cols: usize,
    output_shape: (usize, usize),
    backend: &'static str,
    requested_backend: TensorUtilBackend,
    kernel: &'static str,
) {
    emit_tensor_op(
        op_name,
        &[node_rows, cols],
        &[output_shape.0, output_shape.1],
    );
    emit_tensor_op_meta(op_name, || {
        serde_json::json!({
            "backend": backend,
            "requested_backend": tensor_util_backend_label(requested_backend),
            "kernel": kernel,
            "readout": readout.as_str(),
            "node_rows": node_rows,
            "cols": cols,
            "values": node_rows.saturating_mul(cols),
            "output_rows": output_shape.0,
            "output_cols": output_shape.1,
        })
    });
}

fn validate_node_features(node_features: &Tensor) -> PureResult<(usize, usize)> {
    let (rows, cols) = node_features.shape();
    if rows == 0 || cols == 0 {
        return Err(TensorError::InvalidDimensions { rows, cols });
    }
    Ok((rows, cols))
}

/// Row offsets describing how a row-concatenated tensor should be split into graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphBatch {
    offsets: Vec<usize>,
}

impl GraphBatch {
    /// Builds a batch from per-graph node counts.
    pub fn from_node_counts<I>(counts: I) -> PureResult<Self>
    where
        I: IntoIterator<Item = usize>,
    {
        let mut offsets = vec![0usize];
        let mut total = 0usize;
        for count in counts {
            if count == 0 {
                return Err(TensorError::InvalidDimensions { rows: 0, cols: 1 });
            }
            total = total.checked_add(count).ok_or(TensorError::InvalidValue {
                label: "graph_batch_rows",
            })?;
            offsets.push(total);
        }
        if offsets.len() == 1 {
            return Err(TensorError::EmptyInput("graph_batch"));
        }
        Ok(Self { offsets })
    }

    /// Builds a batch for row-concatenated fixed-size graphs.
    pub fn fixed(nodes_per_graph: NonZeroUsize, total_rows: usize) -> PureResult<Self> {
        if total_rows == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: total_rows,
                cols: 1,
            });
        }
        let nodes = nodes_per_graph.get();
        if total_rows % nodes != 0 {
            return Err(TensorError::InvalidValue {
                label: "graph_batch_fixed_rows",
            });
        }
        Self::from_node_counts(std::iter::repeat(nodes).take(total_rows / nodes))
    }

    /// Returns the row offsets, including the leading zero and trailing total rows.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Returns the number of graphs represented by the batch.
    pub fn graph_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Returns the total number of rows covered by the batch.
    pub fn total_rows(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0)
    }

    /// Returns the row range for a graph in the batch.
    pub fn segment(&self, graph_index: usize) -> Option<Range<usize>> {
        if graph_index >= self.graph_count() {
            return None;
        }
        Some(self.offsets[graph_index]..self.offsets[graph_index + 1])
    }

    /// Ensures a tensor's rows match this graph batch.
    pub fn validate_tensor(&self, tensor: &Tensor) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        if rows != self.total_rows() {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape(),
                right: (self.total_rows(), cols),
            });
        }
        Ok(())
    }
}

/// Per-graph readout telemetry captured from a row-concatenated graph batch.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphBatchReadoutEntry {
    /// Index of the graph within the batch.
    pub graph_index: usize,
    /// Inclusive start row in the row-concatenated input tensor.
    pub row_start: usize,
    /// Exclusive end row in the row-concatenated input tensor.
    pub row_end: usize,
    /// L2 norm of the graph's node features after message passing.
    pub node_l2: f32,
    /// L2 norm of the graph-level prediction after readout.
    pub prediction_l2: f32,
}

/// Summary of graph-level readout telemetry for a batched forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphBatchReadoutTrace {
    /// Readout strategy used to produce the graph predictions.
    pub readout: GraphReadout,
    /// Number of graph segments in the batch.
    pub graph_count: usize,
    /// Total input rows covered by the graph batch.
    pub total_rows: usize,
    /// Shape of the graph-level prediction tensor.
    pub output_shape: (usize, usize),
    /// Per-graph readout records in batch order.
    pub entries: Vec<GraphBatchReadoutEntry>,
}

impl GraphBatchReadoutTrace {
    /// Compares traced graph predictions against graph-level targets.
    pub fn compare_predictions(
        &self,
        prediction: &Tensor,
        target: &Tensor,
    ) -> PureResult<GraphBatchReadoutErrorTrace> {
        if prediction.shape() != self.output_shape {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: self.output_shape,
            });
        }
        if target.shape() != self.output_shape {
            return Err(TensorError::ShapeMismatch {
                left: target.shape(),
                right: self.output_shape,
            });
        }
        if self.entries.len() != self.graph_count {
            return Err(TensorError::InvalidValue {
                label: "graph_batch_readout_entries",
            });
        }
        let (_, cols) = prediction.shape();
        let mut total_mse = 0.0f32;
        let mut total_target_mean_square = 0.0f32;
        let mut entries = Vec::with_capacity(self.graph_count);
        for entry in &self.entries {
            if entry.graph_index >= self.graph_count {
                return Err(TensorError::InvalidValue {
                    label: "graph_batch_readout_graph_index",
                });
            }
            let prediction_row = tensor_row(prediction, entry.graph_index)?;
            let target_row = tensor_row(target, entry.graph_index)?;
            let residual = prediction_row.sub_with_backend(
                &target_row,
                current_tensor_util_backend_for_values(prediction_row.data().len()),
            )?;
            let residual_l2_squared = residual.squared_l2_norm_with_backend(
                current_tensor_util_backend_for_values(residual.data().len()),
            )?;
            let mean_squared_error = residual_l2_squared / cols as f32;
            let target_l2_squared = target_row.squared_l2_norm_with_backend(
                current_tensor_util_backend_for_values(target_row.data().len()),
            )?;
            let target_mean_square = target_l2_squared / cols as f32;
            let normalized_mean_squared_error =
                normalized_mean_squared_error(mean_squared_error, target_mean_square);
            total_mse += mean_squared_error;
            total_target_mean_square += target_mean_square;
            entries.push(GraphBatchReadoutErrorEntry {
                graph_index: entry.graph_index,
                prediction_l2: tensor_l2(&prediction_row)?,
                target_l2: target_l2_squared.sqrt(),
                target_mean_square,
                residual_l2: residual_l2_squared.sqrt(),
                mean_squared_error,
                normalized_mean_squared_error,
            });
        }
        let mean_squared_error = total_mse / self.graph_count.max(1) as f32;
        let target_mean_square = total_target_mean_square / self.graph_count.max(1) as f32;
        Ok(GraphBatchReadoutErrorTrace {
            graph_count: self.graph_count,
            output_shape: self.output_shape,
            mean_squared_error,
            target_mean_square,
            normalized_mean_squared_error: normalized_mean_squared_error(
                mean_squared_error,
                target_mean_square,
            ),
            entries,
        })
    }
}

/// Per-graph prediction/target comparison derived from a readout trace.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphBatchReadoutErrorEntry {
    /// Index of the graph within the batch.
    pub graph_index: usize,
    /// L2 norm of the graph-level prediction row.
    pub prediction_l2: f32,
    /// L2 norm of the graph-level target row.
    pub target_l2: f32,
    /// Mean square energy of the graph-level target row.
    pub target_mean_square: f32,
    /// L2 norm of `prediction - target`.
    pub residual_l2: f32,
    /// Mean squared error for this graph row.
    pub mean_squared_error: f32,
    /// Mean squared error normalized by target mean-square energy.
    pub normalized_mean_squared_error: Option<f32>,
}

/// Batch-level prediction/target comparison derived from a readout trace.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphBatchReadoutErrorTrace {
    /// Number of graph segments in the comparison.
    pub graph_count: usize,
    /// Shape of the prediction and target tensors.
    pub output_shape: (usize, usize),
    /// Average per-graph mean squared error.
    pub mean_squared_error: f32,
    /// Average per-graph target mean-square energy.
    pub target_mean_square: f32,
    /// Average MSE normalized by average target mean-square energy.
    pub normalized_mean_squared_error: Option<f32>,
    /// Per-graph prediction/target comparison records.
    pub entries: Vec<GraphBatchReadoutErrorEntry>,
}

fn normalized_mean_squared_error(mse: f32, target_mean_square: f32) -> Option<f32> {
    if !mse.is_finite() || !target_mean_square.is_finite() {
        return None;
    }
    if target_mean_square > f32::EPSILON {
        return Some(mse / target_mean_square);
    }
    if mse.abs() <= f32::EPSILON {
        return Some(0.0);
    }
    None
}

fn tensor_rows(tensor: &Tensor, range: Range<usize>) -> PureResult<Tensor> {
    let (rows, cols) = tensor.shape();
    if range.start >= range.end || range.end > rows {
        return Err(TensorError::InvalidValue {
            label: "tensor_row_range",
        });
    }
    let start = range.start * cols;
    let end = range.end * cols;
    Tensor::from_vec(
        range.end - range.start,
        cols,
        tensor.data()[start..end].to_vec(),
    )
}

fn tensor_row(tensor: &Tensor, row: usize) -> PureResult<Tensor> {
    tensor_rows(tensor, row..row + 1)
}

fn tensor_l2(tensor: &Tensor) -> PureResult<f32> {
    Ok(tensor
        .squared_l2_norm_with_backend(current_tensor_util_backend_for_values(tensor.data().len()))?
        .sqrt())
}

/// Trainable graph-level regressor that wraps a node-wise graph network with a readout head.
#[derive(Debug)]
pub struct ZSpaceGraphRegressor {
    network: ZSpaceGraphNetwork,
    readout: GraphReadout,
    cache: RefCell<Option<Tensor>>,
}

impl ZSpaceGraphRegressor {
    /// Creates a graph-level model from an existing graph network and readout strategy.
    pub fn new(network: ZSpaceGraphNetwork, readout: GraphReadout) -> Self {
        Self {
            network,
            readout,
            cache: RefCell::new(None),
        }
    }

    /// Returns the wrapped node-wise graph network.
    pub fn network(&self) -> &ZSpaceGraphNetwork {
        &self.network
    }

    /// Returns the wrapped node-wise graph network mutably.
    pub fn network_mut(&mut self) -> &mut ZSpaceGraphNetwork {
        &mut self.network
    }

    /// Returns the active graph readout strategy.
    pub fn readout(&self) -> GraphReadout {
        self.readout
    }

    /// Replaces the graph readout strategy.
    pub fn set_readout(&mut self, readout: GraphReadout) {
        self.readout = readout;
    }

    fn take_cache(&self) -> Option<Tensor> {
        self.cache.replace(None)
    }

    fn store_cache(&self, node_features: Tensor) {
        self.cache.replace(Some(node_features));
    }
}

impl Module for ZSpaceGraphRegressor {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let node_features = self.network.forward(input)?;
        let graph_prediction = self.readout.forward(&node_features)?;
        self.store_cache(node_features);
        Ok(graph_prediction)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let node_features = match self.take_cache() {
            Some(cache) => cache,
            None => self.network.forward(input)?,
        };
        let grad_nodes = self.readout.backward(&node_features, grad_output)?;
        self.network.backward(input, &grad_nodes)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.network.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.network.visit_parameters_mut(visitor)
    }

    fn backward_bands(&mut self, input: &Tensor, bands: &GradientBands) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let mut total = Tensor::zeros(rows, cols)?;
        for (band, grad) in bands.iter_labeled() {
            let backend = current_tensor_util_backend_for_values(grad.data().len());
            if grad.squared_l2_norm_with_backend(backend)? == 0.0 {
                continue;
            }
            self.begin_backward_band_pass(band, grad)?;
            let result = (|| {
                let _ = self.forward(input)?;
                self.backward(input, grad)
            })();
            self.end_backward_band_pass(band)?;
            let contribution = result?;
            let backend = current_tensor_util_backend_for_values(total.data().len());
            total.add_scaled_with_backend(&contribution, 1.0, backend)?;
        }
        Ok(total)
    }

    fn begin_backward_band_pass(
        &mut self,
        band: RoundtableBand,
        gradient: &Tensor,
    ) -> PureResult<()> {
        self.network.begin_backward_band_pass(band, gradient)
    }

    fn end_backward_band_pass(&mut self, band: RoundtableBand) -> PureResult<()> {
        self.network.end_backward_band_pass(band)
    }

    fn apply_roundtable_band(&mut self, signal: &RoundtableBandSignal) -> PureResult<()> {
        self.network.apply_roundtable_band(signal)
    }

    fn clear_roundtable_band(&mut self) -> PureResult<()> {
        self.network.clear_roundtable_band()
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        self.network.infuse_text(text)
    }

    fn set_training(&mut self, training: bool) -> PureResult<()> {
        self.network.set_training(training)
    }

    fn psi_probe(&self) -> Option<f32> {
        self.network.psi_probe()
    }

    fn preferred_device(&self) -> Option<DeviceCaps> {
        self.network.preferred_device()
    }
}

/// Graph-level regressor for row-concatenated batches of fixed-size graphs.
#[derive(Debug)]
pub struct ZSpaceGraphBatchRegressor {
    network: ZSpaceGraphNetwork,
    readout: GraphReadout,
    nodes_per_graph: NonZeroUsize,
}

impl ZSpaceGraphBatchRegressor {
    /// Creates a batched graph-level model for fixed-size graphs.
    pub fn new(
        network: ZSpaceGraphNetwork,
        readout: GraphReadout,
        nodes_per_graph: NonZeroUsize,
    ) -> Self {
        Self {
            network,
            readout,
            nodes_per_graph,
        }
    }

    /// Returns the wrapped node-wise graph network.
    pub fn network(&self) -> &ZSpaceGraphNetwork {
        &self.network
    }

    /// Returns the wrapped node-wise graph network mutably.
    pub fn network_mut(&mut self) -> &mut ZSpaceGraphNetwork {
        &mut self.network
    }

    /// Returns the active graph readout strategy.
    pub fn readout(&self) -> GraphReadout {
        self.readout
    }

    /// Replaces the graph readout strategy.
    pub fn set_readout(&mut self, readout: GraphReadout) {
        self.readout = readout;
    }

    /// Returns how many node rows belong to each graph in the default fixed-size batch mode.
    pub fn nodes_per_graph(&self) -> NonZeroUsize {
        self.nodes_per_graph
    }

    /// Runs graph-level prediction using an explicit graph batch descriptor.
    pub fn forward_batch(&self, input: &Tensor, batch: &GraphBatch) -> PureResult<Tensor> {
        Ok(self.forward_batch_impl(input, batch, false)?.0)
    }

    /// Runs graph-level prediction while collecting per-graph readout telemetry.
    pub fn forward_with_trace(
        &self,
        input: &Tensor,
    ) -> PureResult<(Tensor, GraphBatchReadoutTrace)> {
        let batch = self.fixed_batch_for(input)?;
        self.forward_batch_with_trace(input, &batch)
    }

    /// Runs graph-level prediction with an explicit graph batch descriptor while tracing readout.
    pub fn forward_batch_with_trace(
        &self,
        input: &Tensor,
        batch: &GraphBatch,
    ) -> PureResult<(Tensor, GraphBatchReadoutTrace)> {
        let (output, trace) = self.forward_batch_impl(input, batch, true)?;
        let trace = trace.ok_or(TensorError::InvalidValue {
            label: "graph_batch_readout_trace",
        })?;
        Ok((output, trace))
    }

    fn forward_batch_impl(
        &self,
        input: &Tensor,
        batch: &GraphBatch,
        collect_trace: bool,
    ) -> PureResult<(Tensor, Option<GraphBatchReadoutTrace>)> {
        batch.validate_tensor(input)?;
        let mut predictions = Vec::with_capacity(batch.graph_count());
        let mut entries = collect_trace.then(|| Vec::with_capacity(batch.graph_count()));
        for graph_index in 0..batch.graph_count() {
            let segment = batch
                .segment(graph_index)
                .ok_or(TensorError::InvalidValue {
                    label: "graph_batch_segment",
                })?;
            let row_start = segment.start;
            let row_end = segment.end;
            let graph_input = tensor_rows(input, segment)?;
            let node_features = self.network.forward(&graph_input)?;
            let prediction = self.readout.forward(&node_features)?;
            if let Some(entries) = entries.as_mut() {
                entries.push(GraphBatchReadoutEntry {
                    graph_index,
                    row_start,
                    row_end,
                    node_l2: tensor_l2(&node_features)?,
                    prediction_l2: tensor_l2(&prediction)?,
                });
            }
            predictions.push(prediction);
        }
        let output = Tensor::cat_rows(&predictions)?;
        let trace = entries.map(|entries| GraphBatchReadoutTrace {
            readout: self.readout,
            graph_count: batch.graph_count(),
            total_rows: batch.total_rows(),
            output_shape: output.shape(),
            entries,
        });
        Ok((output, trace))
    }

    /// Backpropagates graph-level gradients using an explicit graph batch descriptor.
    pub fn backward_batch(
        &mut self,
        input: &Tensor,
        batch: &GraphBatch,
        grad_output: &Tensor,
    ) -> PureResult<Tensor> {
        batch.validate_tensor(input)?;
        let (grad_rows, grad_cols) = grad_output.shape();
        if grad_rows != batch.graph_count() {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (batch.graph_count(), grad_cols),
            });
        }
        let mut graph_gradients = Vec::with_capacity(batch.graph_count());
        for graph_index in 0..batch.graph_count() {
            let segment = batch
                .segment(graph_index)
                .ok_or(TensorError::InvalidValue {
                    label: "graph_batch_segment",
                })?;
            let graph_input = tensor_rows(input, segment)?;
            let node_features = self.network.forward(&graph_input)?;
            let graph_grad = tensor_row(grad_output, graph_index)?;
            let grad_nodes = self.readout.backward(&node_features, &graph_grad)?;
            graph_gradients.push(self.network.backward(&graph_input, &grad_nodes)?);
        }
        Tensor::cat_rows(&graph_gradients)
    }

    fn fixed_batch_for(&self, input: &Tensor) -> PureResult<GraphBatch> {
        GraphBatch::fixed(self.nodes_per_graph, input.shape().0)
    }
}

impl Module for ZSpaceGraphBatchRegressor {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let batch = self.fixed_batch_for(input)?;
        self.forward_batch(input, &batch)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let batch = self.fixed_batch_for(input)?;
        self.backward_batch(input, &batch, grad_output)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.network.visit_parameters(visitor)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.network.visit_parameters_mut(visitor)
    }

    fn backward_bands(&mut self, input: &Tensor, bands: &GradientBands) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let mut total = Tensor::zeros(rows, cols)?;
        for (band, grad) in bands.iter_labeled() {
            let backend = current_tensor_util_backend_for_values(grad.data().len());
            if grad.squared_l2_norm_with_backend(backend)? == 0.0 {
                continue;
            }
            self.begin_backward_band_pass(band, grad)?;
            let result = self.backward(input, grad);
            self.end_backward_band_pass(band)?;
            let contribution = result?;
            let backend = current_tensor_util_backend_for_values(total.data().len());
            total.add_scaled_with_backend(&contribution, 1.0, backend)?;
        }
        Ok(total)
    }

    fn begin_backward_band_pass(
        &mut self,
        band: RoundtableBand,
        gradient: &Tensor,
    ) -> PureResult<()> {
        self.network.begin_backward_band_pass(band, gradient)
    }

    fn end_backward_band_pass(&mut self, band: RoundtableBand) -> PureResult<()> {
        self.network.end_backward_band_pass(band)
    }

    fn apply_roundtable_band(&mut self, signal: &RoundtableBandSignal) -> PureResult<()> {
        self.network.apply_roundtable_band(signal)
    }

    fn clear_roundtable_band(&mut self) -> PureResult<()> {
        self.network.clear_roundtable_band()
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        self.network.infuse_text(text)
    }

    fn set_training(&mut self, training: bool) -> PureResult<()> {
        self.network.set_training(training)
    }

    fn psi_probe(&self) -> Option<f32> {
        self.network.psi_probe()
    }

    fn preferred_device(&self) -> Option<DeviceCaps> {
        self.network.preferred_device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::{GraphContext, GraphLayerSpec, ZSpaceGraphNetworkBuilder};
    use crate::{Dataset, MeanSquaredError, ModuleTrainer, RoundtableConfig};
    use st_core::backend::device_caps::DeviceCaps;
    use std::num::NonZeroUsize;
    use std::sync::{Arc, Mutex, OnceLock};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn sample_context() -> GraphContext {
        let adjacency =
            Tensor::from_vec(3, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        GraphContext::from_adjacency(adjacency).unwrap()
    }

    fn sample_node_features() -> Tensor {
        Tensor::from_vec(3, 2, vec![1.0, -1.0, 3.0, 0.5, -2.0, 0.5]).unwrap()
    }

    fn sample_network() -> ZSpaceGraphNetwork {
        let mut builder = ZSpaceGraphNetworkBuilder::new(
            sample_context(),
            NonZeroUsize::new(2).unwrap(),
            -1.0,
            0.05,
        );
        builder.push_layer(GraphLayerSpec::new(NonZeroUsize::new(2).unwrap()));
        builder.build("readout_regressor").unwrap()
    }

    #[test]
    fn readout_pools_node_features() {
        let nodes = sample_node_features();
        let mean = GraphReadout::Mean.forward(&nodes).unwrap();
        assert_eq!(mean.shape(), (1, 2));
        assert!((mean.data()[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!(mean.data()[1].abs() < 1e-6);

        let sum = GraphReadout::Sum.forward(&nodes).unwrap();
        assert_eq!(sum.data(), &[2.0, 0.0]);

        let max = GraphReadout::Max.forward(&nodes).unwrap();
        assert_eq!(max.data(), &[3.0, 0.5]);
    }

    #[test]
    fn readout_backward_distributes_graph_gradients() {
        let nodes = sample_node_features();
        let grad = Tensor::from_vec(1, 2, vec![0.6, -0.3]).unwrap();

        let mean_grad = GraphReadout::Mean.backward(&nodes, &grad).unwrap();
        assert_eq!(mean_grad.shape(), (3, 2));
        assert!((mean_grad.data()[0] - 0.2).abs() < 1e-6);
        assert!((mean_grad.data()[1] + 0.1).abs() < 1e-6);

        let sum_grad = GraphReadout::Sum.backward(&nodes, &grad).unwrap();
        assert_eq!(sum_grad.data()[0], 0.6);
        assert_eq!(sum_grad.data()[1], -0.3);

        let max_grad = GraphReadout::Max.backward(&nodes, &grad).unwrap();
        assert_eq!(max_grad.data(), &[0.0, 0.0, 0.6, -0.15, 0.0, -0.15]);
    }

    #[test]
    fn readout_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let nodes = sample_node_features();
        let grad = Tensor::from_vec(1, 2, vec![0.6, -0.3]).unwrap();
        let _ = GraphReadout::Mean.forward(&nodes).unwrap();
        let _ = GraphReadout::Mean.backward(&nodes, &grad).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "graph_readout"
                    && data["backend"] == "composite"
                    && data["readout"] == "mean"
                    && data["node_rows"] == 3
                    && data["cols"] == 2
                    && data["kernel"] == "graph_readout.mean.sum_axis0_scaled"
            })
            .expect("graph readout metadata event");
        assert_eq!(forward.1["requested_backend"], "auto");

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "graph_readout_backward"
                    && data["backend"] == "composite"
                    && data["readout"] == "mean"
                    && data["node_rows"] == 3
                    && data["cols"] == 2
                    && data["kernel"] == "graph_readout_backward.mean.add_row"
            })
            .expect("graph readout backward metadata event");
        assert_eq!(backward.1["requested_backend"], "auto");
        assert!(events
            .iter()
            .any(|(op_name, data)| *op_name == "sum_axis0_scaled"
                && data["backend"] == "cpu"
                && data["requested_backend"] == "auto"));
        assert!(events
            .iter()
            .any(|(op_name, data)| *op_name == "add_row_inplace"
                && data["backend"] == "cpu"
                && data["requested_backend"] == "auto"));
    }

    #[test]
    fn graph_regressor_maps_graph_gradients_through_network() {
        let mut model = ZSpaceGraphRegressor::new(sample_network(), GraphReadout::Mean);
        let input = Tensor::from_vec(3, 2, vec![1.0, 0.25, 0.5, -0.5, -0.25, 1.0]).unwrap();
        let prediction = model.forward(&input).unwrap();
        assert_eq!(prediction.shape(), (1, 2));

        let grad_graph = Tensor::from_vec(1, 2, vec![0.1, -0.2]).unwrap();
        let grad_input = model.backward(&input, &grad_graph).unwrap();
        assert_eq!(grad_input.shape(), (3, 2));
        assert!(grad_input.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn graph_regressor_replays_band_gradients() {
        let mut model = ZSpaceGraphRegressor::new(sample_network(), GraphReadout::Mean);
        let input = Tensor::from_vec(3, 2, vec![1.0, 0.25, 0.5, -0.5, -0.25, 1.0]).unwrap();
        let bands = GradientBands::from_components(
            Tensor::from_vec(1, 2, vec![0.05, 0.0]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.0, -0.1]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.02, 0.03]).unwrap(),
        )
        .unwrap();
        let grad_input = model.backward_bands(&input, &bands).unwrap();
        assert_eq!(grad_input.shape(), (3, 2));
        assert!(grad_input.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn graph_batch_describes_fixed_segments() {
        let batch = GraphBatch::fixed(NonZeroUsize::new(3).unwrap(), 9).unwrap();
        assert_eq!(batch.graph_count(), 3);
        assert_eq!(batch.total_rows(), 9);
        assert_eq!(batch.offsets(), &[0, 3, 6, 9]);
        assert_eq!(batch.segment(1), Some(3..6));
        assert!(GraphBatch::fixed(NonZeroUsize::new(3).unwrap(), 10).is_err());
    }

    #[test]
    fn graph_batch_regressor_maps_multiple_graphs() {
        let mut model = ZSpaceGraphBatchRegressor::new(
            sample_network(),
            GraphReadout::Mean,
            NonZeroUsize::new(3).unwrap(),
        );
        let graph_a = Tensor::from_vec(3, 2, vec![1.0, 0.25, 0.5, -0.5, -0.25, 1.0]).unwrap();
        let graph_b = Tensor::from_vec(3, 2, vec![-0.5, 0.75, 0.25, 0.5, 1.0, -1.0]).unwrap();
        let input = Tensor::cat_rows(&[graph_a, graph_b]).unwrap();
        let prediction = model.forward(&input).unwrap();
        assert_eq!(prediction.shape(), (2, 2));

        let grad_graph = Tensor::from_vec(2, 2, vec![0.1, -0.2, -0.05, 0.15]).unwrap();
        let grad_input = model.backward(&input, &grad_graph).unwrap();
        assert_eq!(grad_input.shape(), (6, 2));
        assert!(grad_input.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn graph_batch_regressor_traces_multiple_graph_readouts() {
        let model = ZSpaceGraphBatchRegressor::new(
            sample_network(),
            GraphReadout::Mean,
            NonZeroUsize::new(3).unwrap(),
        );
        let graph_a = Tensor::from_vec(3, 2, vec![1.0, 0.25, 0.5, -0.5, -0.25, 1.0]).unwrap();
        let graph_b = Tensor::from_vec(3, 2, vec![-0.5, 0.75, 0.25, 0.5, 1.0, -1.0]).unwrap();
        let input = Tensor::cat_rows(&[graph_a, graph_b]).unwrap();

        let (prediction, trace) = model.forward_with_trace(&input).unwrap();

        assert_eq!(prediction.shape(), (2, 2));
        assert_eq!(trace.readout, GraphReadout::Mean);
        assert_eq!(trace.graph_count, 2);
        assert_eq!(trace.total_rows, 6);
        assert_eq!(trace.output_shape, (2, 2));
        assert_eq!(trace.entries.len(), 2);
        assert_eq!(trace.entries[0].row_start, 0);
        assert_eq!(trace.entries[0].row_end, 3);
        assert_eq!(trace.entries[1].row_start, 3);
        assert_eq!(trace.entries[1].row_end, 6);
        assert!(trace.entries.iter().all(|entry| entry.node_l2.is_finite()));
        assert!(trace
            .entries
            .iter()
            .all(|entry| entry.prediction_l2.is_finite()));

        let target = Tensor::from_vec(2, 2, vec![0.1, -0.2, -0.05, 0.15]).unwrap();
        let errors = trace.compare_predictions(&prediction, &target).unwrap();
        assert_eq!(errors.graph_count, 2);
        assert_eq!(errors.output_shape, (2, 2));
        assert_eq!(errors.entries.len(), 2);
        assert!(errors.mean_squared_error.is_finite());
        assert!(errors.target_mean_square.is_finite());
        assert!(errors.normalized_mean_squared_error.is_some());
        assert!(errors
            .entries
            .iter()
            .all(|entry| entry.residual_l2.is_finite()));
        assert!(errors
            .entries
            .iter()
            .all(|entry| entry.target_mean_square.is_finite()
                && entry.normalized_mean_squared_error.is_some()));
    }

    #[test]
    fn graph_batch_regressor_replays_batched_band_gradients() {
        let mut model = ZSpaceGraphBatchRegressor::new(
            sample_network(),
            GraphReadout::Mean,
            NonZeroUsize::new(3).unwrap(),
        );
        let graph_a = Tensor::from_vec(3, 2, vec![1.0, 0.25, 0.5, -0.5, -0.25, 1.0]).unwrap();
        let graph_b = Tensor::from_vec(3, 2, vec![-0.5, 0.75, 0.25, 0.5, 1.0, -1.0]).unwrap();
        let input = Tensor::cat_rows(&[graph_a, graph_b]).unwrap();
        let bands = GradientBands::from_components(
            Tensor::from_vec(2, 2, vec![0.05, 0.0, 0.02, 0.0]).unwrap(),
            Tensor::from_vec(2, 2, vec![0.0, -0.1, 0.0, 0.03]).unwrap(),
            Tensor::from_vec(2, 2, vec![0.02, 0.03, -0.04, 0.01]).unwrap(),
        )
        .unwrap();
        let grad_input = model.backward_bands(&input, &bands).unwrap();
        assert_eq!(grad_input.shape(), (6, 2));
        assert!(grad_input.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn graph_batch_regressor_trains_with_dataloader_batches() {
        let mut model = ZSpaceGraphBatchRegressor::new(
            sample_network(),
            GraphReadout::Mean,
            NonZeroUsize::new(3).unwrap(),
        );
        model.attach_hypergrad(-1.0, 1e-2).unwrap();
        let graph_a = Tensor::from_vec(3, 2, vec![1.0, 0.25, 0.5, -0.5, -0.25, 1.0]).unwrap();
        let graph_b = Tensor::from_vec(3, 2, vec![-0.5, 0.75, 0.25, 0.5, 1.0, -1.0]).unwrap();
        let target_a = GraphReadout::Mean.forward(&graph_a).unwrap();
        let target_b = GraphReadout::Mean.forward(&graph_b).unwrap();
        let dataset = Dataset::from_vec(vec![(graph_a, target_a), (graph_b, target_b)]);
        let loader = dataset.loader().batched(2);
        let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 1e-2, 1e-2);
        let schedule = trainer.roundtable(
            2,
            2,
            RoundtableConfig::default()
                .with_top_k(1)
                .with_mid_k(1)
                .with_bottom_k(1),
        );
        let mut loss = MeanSquaredError::new();
        let stats = trainer
            .train_epoch(&mut model, &mut loss, loader.iter(), &schedule)
            .unwrap();
        assert_eq!(stats.batches, 1);
        assert!(stats.average_loss.is_finite());
    }
}
