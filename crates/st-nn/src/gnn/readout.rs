// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{RoundtableBandSignal, ZSpaceGraphNetwork};
use crate::module::{Module, Parameter};
use crate::schedule::GradientBands;
use crate::{PureResult, Tensor, TensorError};
use st_core::backend::device_caps::DeviceCaps;
use st_core::ops::zspace_round::RoundtableBand;
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

    /// Pools node embeddings into a single graph embedding.
    pub fn forward(&self, node_features: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = validate_node_features(node_features)?;
        match self {
            Self::Mean => {
                let mut data = column_sums(node_features);
                let scale = 1.0 / rows as f32;
                for value in &mut data {
                    *value *= scale;
                }
                Tensor::from_vec(1, cols, data)
            }
            Self::Sum => Tensor::from_vec(1, cols, column_sums(node_features)),
            Self::Max => {
                let mut data = vec![f32::NEG_INFINITY; cols];
                for row in 0..rows {
                    let offset = row * cols;
                    for col in 0..cols {
                        data[col] = data[col].max(node_features.data()[offset + col]);
                    }
                }
                Tensor::from_vec(1, cols, data)
            }
        }
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
        let mut data = vec![0.0f32; rows * cols];
        match self {
            Self::Mean => {
                let scale = 1.0 / rows as f32;
                for row in 0..rows {
                    let offset = row * cols;
                    for col in 0..cols {
                        data[offset + col] = grad_graph.data()[col] * scale;
                    }
                }
            }
            Self::Sum => {
                for row in 0..rows {
                    let offset = row * cols;
                    for col in 0..cols {
                        data[offset + col] = grad_graph.data()[col];
                    }
                }
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
        Tensor::from_vec(rows, cols, data)
    }
}

fn validate_node_features(node_features: &Tensor) -> PureResult<(usize, usize)> {
    let (rows, cols) = node_features.shape();
    if rows == 0 || cols == 0 {
        return Err(TensorError::InvalidDimensions { rows, cols });
    }
    Ok((rows, cols))
}

fn column_sums(node_features: &Tensor) -> Vec<f32> {
    let (rows, cols) = node_features.shape();
    let mut data = vec![0.0f32; cols];
    for row in 0..rows {
        let offset = row * cols;
        for col in 0..cols {
            data[col] += node_features.data()[offset + col];
        }
    }
    data
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
            if grad.squared_l2_norm() == 0.0 {
                continue;
            }
            self.begin_backward_band_pass(band, grad)?;
            let result = (|| {
                let _ = self.forward(input)?;
                self.backward(input, grad)
            })();
            self.end_backward_band_pass(band)?;
            let contribution = result?;
            total.add_scaled(&contribution, 1.0)?;
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
        batch.validate_tensor(input)?;
        let mut predictions = Vec::with_capacity(batch.graph_count());
        for graph_index in 0..batch.graph_count() {
            let segment = batch
                .segment(graph_index)
                .ok_or(TensorError::InvalidValue {
                    label: "graph_batch_segment",
                })?;
            let graph_input = tensor_rows(input, segment)?;
            let node_features = self.network.forward(&graph_input)?;
            predictions.push(self.readout.forward(&node_features)?);
        }
        Tensor::cat_rows(&predictions)
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
            if grad.squared_l2_norm() == 0.0 {
                continue;
            }
            self.begin_backward_band_pass(band, grad)?;
            let result = self.backward(input, grad);
            self.end_backward_band_pass(band)?;
            let contribution = result?;
            total.add_scaled(&contribution, 1.0)?;
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
