//! Explicit graph-only evidence; neither a model checkpoint nor a training hook.
use super::*;
use crate::resident::diagnostic::{GraphDiagnosticBinding, GraphDiagnosticSource};
use st_backend_wgpu::resident_training::graph::{
    GraphForwardDiagnostic, GraphForwardDiagnosticReadback, GRAPH_FORWARD_DIAGNOSTIC_MAX_BYTES,
};
use std::collections::HashSet;

mod json;

/// A private execution graph's captured bindings in graph parameter order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteDecoderNamedGraphDiagnostic {
    pub name: String,
    /// Absolute model slots. A contextual query appends its router table after
    /// its projection parameters; its captured router shape is flattened.
    pub model_slots: Vec<usize>,
    pub diagnostic: GraphForwardDiagnostic,
}

/// Graph bindings, inputs, stage outputs and guards from one model forward.
///
/// This excludes direct parameter values, optimizer state, the full causal-wave
/// and attention tapes, targets and loss. Some wave features appear as query
/// graph inputs. `direct_model_slots` explicitly accounts for uncopied slots.
/// Raw invalid values remain evidence, not acceptance or reproducibility proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteDecoderGraphDiagnostic {
    pub submitted_forward: u64,
    pub parameter_revision: u64,
    pub captured_bytes: usize,
    pub model_parameter_shapes: Vec<Vec<usize>>,
    /// Model slots used directly, rather than copied into a captured graph.
    /// Their values are NOT included in this diagnostic.
    pub direct_model_slots: Vec<usize>,
    pub graphs: Vec<ByteDecoderNamedGraphDiagnostic>,
}

struct NamedReadback {
    name: String,
    model_slots: Vec<usize>,
    readback: GraphForwardDiagnosticReadback,
}

/// Owns submitted staging copies without borrowing the model or its forward.
/// Reading consumes the snapshot; subsequent model updates cannot change it.
pub struct ByteDecoderGraphDiagnosticReadback {
    metadata: ByteDecoderGraphDiagnostic,
    graphs: Vec<NamedReadback>,
}

impl ByteDecoderGraphDiagnosticReadback {
    pub fn captured_bytes(&self) -> usize {
        self.metadata.captured_bytes
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<ByteDecoderGraphDiagnostic, InferenceError> {
        let mut result = self.metadata;
        for graph in self.graphs {
            result.graphs.push(ByteDecoderNamedGraphDiagnostic {
                name: graph.name,
                model_slots: graph.model_slots,
                diagnostic: graph.readback.read()?,
            });
        }
        Ok(result)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<ByteDecoderGraphDiagnostic, InferenceError> {
        let mut result = self.metadata;
        for graph in self.graphs {
            result.graphs.push(ByteDecoderNamedGraphDiagnostic {
                name: graph.name,
                model_slots: graph.model_slots,
                diagnostic: graph.readback.read_async().await?,
            });
        }
        Ok(result)
    }
}

fn graph_count(blocks: usize, geometry: bool, queries: usize) -> Result<usize, InferenceError> {
    let count = blocks
        .checked_mul(4)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_add(usize::from(geometry)))
        .and_then(|n| n.checked_add(queries))
        .ok_or(TrainingError::Overflow)?;
    if count > BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_GRAPHS {
        return Err(InferenceError::ByteDecoder("graph diagnostic graph limit"));
    }
    Ok(count)
}

fn aggregate_bytes(sizes: &[usize], max_bytes: usize) -> Result<usize, InferenceError> {
    if max_bytes == 0 || max_bytes > BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES {
        return Err(InferenceError::ByteDecoder("graph diagnostic byte budget"));
    }
    if sizes.is_empty() || sizes.len() > BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_GRAPHS {
        return Err(InferenceError::ByteDecoder("graph diagnostic graph limit"));
    }
    let mut total = 0usize;
    for &bytes in sizes {
        if bytes == 0 || bytes > GRAPH_FORWARD_DIAGNOSTIC_MAX_BYTES {
            return Err(InferenceError::ByteDecoder(
                "graph diagnostic per-graph budget",
            ));
        }
        total = total.checked_add(bytes).ok_or(TrainingError::Overflow)?;
    }
    if total > max_bytes {
        return Err(InferenceError::ByteDecoder("graph diagnostic byte budget"));
    }
    Ok(total)
}

fn mark_slot(seen: &mut [bool], slot: usize) -> Result<(), InferenceError> {
    match seen.get_mut(slot) {
        Some(used) if !*used => {
            *used = true;
            Ok(())
        }
        _ => Err(TrainingError::ParameterLayout.into()),
    }
}

fn complete_partition(seen: &[bool]) -> Result<(), InferenceError> {
    if seen.iter().any(|used| !used) {
        return Err(TrainingError::ParameterLayout.into());
    }
    Ok(())
}

fn validate_binding_shapes(
    model_shapes: &[Vec<usize>],
    seen: &mut [bool],
    bindings: &[GraphDiagnosticBinding],
    graph_shapes: &[Vec<usize>],
) -> Result<(), InferenceError> {
    if bindings.len() != graph_shapes.len() || seen.len() != model_shapes.len() {
        return Err(TrainingError::ParameterLayout.into());
    }
    for (binding, graph_shape) in bindings.iter().zip(graph_shapes) {
        let model_shape = model_shapes
            .get(binding.model_slot)
            .ok_or(TrainingError::ParameterLayout)?;
        let matches = if binding.flattened {
            // The sole shape exception is the query's [heads, components]
            // router table, appended as a flat pointwise bias.
            model_shape.len() == 2
                && graph_shape.len() == 1
                && model_shape[0].checked_mul(model_shape[1]) == Some(graph_shape[0])
        } else {
            model_shape == graph_shape
        };
        if !matches {
            return Err(TrainingError::ParameterLayout.into());
        }
        mark_slot(seen, binding.model_slot)?;
    }
    Ok(())
}

struct CapturePlan<'a> {
    metadata: ByteDecoderGraphDiagnostic,
    graphs: Vec<GraphDiagnosticSource<'a>>,
}

impl ResidentByteDecoder {
    fn graph_diagnostic_plan<'a>(
        &'a self,
        forward: &'a ResidentByteDecoderForward,
        max_bytes: usize,
    ) -> Result<CapturePlan<'a>, InferenceError> {
        require_uncommitted_route()?;
        if self.latest != Some(forward.submission)
            || !self.parameters.is_current(&forward.parameters)
            || self.bound_revision != Some(forward.parameters.revision())
        {
            return Err(TrainingError::StaleForward.into());
        }
        if max_bytes == 0 || max_bytes > BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES {
            return Err(InferenceError::ByteDecoder("graph diagnostic byte budget"));
        }
        let layout = &self.parameter_layout;
        if self.blocks.len() != forward.blocks.len()
            || self.blocks.len() != layout.blocks().len()
            || forward.parameters.values().len() != layout.len()
        {
            return Err(TrainingError::ParameterLayout.into());
        }
        let queries = layout.geometry().map_or(0, |g| g.query_projections().len());
        let expected = graph_count(self.blocks.len(), self.geometry.is_some(), queries)?;
        let mut direct_model_slots = vec![layout.token_embedding(), layout.position_embedding()];
        let mut graphs = Vec::with_capacity(expected);
        match (&self.geometry, &forward.geometry, layout.geometry()) {
            (Some(geometry), Some(tape), Some(geometry_layout)) => {
                graphs.extend(geometry.diagnostic_graphs(
                    tape,
                    geometry_layout,
                    self.blocks.len(),
                )?);
                direct_model_slots
                    .extend([geometry_layout.raw_decay(), geometry_layout.raw_phase()]);
                direct_model_slots.extend(geometry_layout.raw_gains());
                if geometry_layout.query_projections().is_empty() {
                    if let Some(routers) = geometry_layout.mixture_logits() {
                        direct_model_slots.extend(routers);
                    }
                }
            }
            (None, None, None) => {}
            _ => return Err(TrainingError::ParameterLayout.into()),
        }
        for (index, ((block, tape), range)) in self
            .blocks
            .iter()
            .zip(&forward.blocks)
            .zip(layout.blocks())
            .enumerate()
        {
            graphs.extend(block.diagnostic_graphs(
                tape,
                &format!("block.{index}"),
                range.clone(),
            )?);
        }
        graphs.push(GraphDiagnosticSource::new(
            "head".into(),
            layout.head(),
            None,
            &self.head,
            &forward.head,
        )?);
        if graphs.len() != expected {
            return Err(TrainingError::ParameterLayout.into());
        }
        let model_parameter_shapes: Vec<_> = forward
            .parameters
            .values()
            .iter()
            .map(|p| p.layout().shape().to_vec())
            .collect();
        let mut seen = vec![false; model_parameter_shapes.len()];
        for &slot in &direct_model_slots {
            mark_slot(&mut seen, slot)?;
        }
        let mut names = HashSet::new();
        let mut sizes = Vec::with_capacity(expected);
        for graph in &graphs {
            if !names.insert(&graph.name) {
                return Err(InferenceError::ByteDecoder(
                    "duplicate graph diagnostic name",
                ));
            }
            // These APIs validate each inner token and inspect host metadata;
            // no diagnostic COPY is submitted until the whole plan succeeds.
            sizes.push(graph.graph.forward_diagnostic_bytes(graph.forward)?);
            let shapes = graph
                .graph
                .forward_diagnostic_parameter_shapes(graph.forward)?;
            validate_binding_shapes(&model_parameter_shapes, &mut seen, &graph.bindings, &shapes)?;
        }
        complete_partition(&seen)?;
        let captured_bytes = aggregate_bytes(&sizes, max_bytes)?;
        Ok(CapturePlan {
            metadata: ByteDecoderGraphDiagnostic {
                submitted_forward: forward.submission,
                parameter_revision: forward.parameters.revision(),
                captured_bytes,
                model_parameter_shapes,
                direct_model_slots,
                graphs: Vec::new(),
            },
            graphs,
        })
    }

    /// Host-only validation and staging-payload size of this forward's graphs.
    /// Enforces both per-graph and combined caps without copying or mapping.
    pub fn graph_diagnostic_bytes(
        &self,
        forward: &ResidentByteDecoderForward,
    ) -> Result<usize, InferenceError> {
        Ok(self
            .graph_diagnostic_plan(forward, BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES)?
            .metadata
            .captured_bytes)
    }

    /// Freeze all actual graph bindings and stage values, with explicit mapping.
    ///
    /// Every identity, shape, slot partition and total budget is checked before
    /// the first copy. A later device/allocation failure is not transactional:
    /// earlier copies may already have been submitted. There is no retry.
    /// Call after the chosen forward (and its loss submission), before backward
    /// if a pre-backward boundary is required. A latest token alone does not
    /// establish that boundary. Capture changes scheduling; preserve the original
    /// outcome separately. No capture is inserted into ordinary training.
    pub fn graph_diagnostic_snapshot(
        &self,
        forward: &ResidentByteDecoderForward,
        max_bytes: usize,
    ) -> Result<ByteDecoderGraphDiagnosticReadback, InferenceError> {
        let plan = self.graph_diagnostic_plan(forward, max_bytes)?;
        let mut graphs = Vec::with_capacity(plan.graphs.len());
        for graph in plan.graphs {
            let readback = graph
                .graph
                .forward_diagnostic_snapshot(graph.forward, GRAPH_FORWARD_DIAGNOSTIC_MAX_BYTES)?;
            graphs.push(NamedReadback {
                name: graph.name,
                model_slots: graph.bindings.iter().map(|b| b.model_slot).collect(),
                readback,
            });
        }
        Ok(ByteDecoderGraphDiagnosticReadback {
            metadata: plan.metadata,
            graphs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident::diagnostic::diagnostic_bindings;

    #[test]
    fn diagnostic_graph_count_is_checked_and_keeps_all_four_block_graphs() {
        assert_eq!(graph_count(1, false, 0).unwrap(), 5);
        assert_eq!(graph_count(1, true, 0).unwrap(), 6);
        assert_eq!(graph_count(1, true, 1).unwrap(), 7);
        assert_eq!(graph_count(50, true, 50).unwrap(), 252);
        assert!(graph_count(51, true, 51).is_err());
        assert!(graph_count(usize::MAX, false, 0).is_err());
        assert!(graph_count(1, true, usize::MAX).is_err());
    }

    #[test]
    fn diagnostic_aggregate_budget_rejects_partial_and_oversized_requests() {
        let graph_max = GRAPH_FORWARD_DIAGNOSTIC_MAX_BYTES;
        let max = BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_BYTES;
        assert_eq!(aggregate_bytes(&[graph_max; 4], max).unwrap(), max);
        assert_eq!(aggregate_bytes(&[4; 256], 1024).unwrap(), 1024);
        assert!(aggregate_bytes(&[graph_max; 4], max - 1).is_err());
        assert!(aggregate_bytes(&[graph_max; 5], max).is_err());
        assert!(aggregate_bytes(&[graph_max + 1], max).is_err());
        assert!(aggregate_bytes(&[0], max).is_err());
        assert!(aggregate_bytes(&[], max).is_err());
        assert!(aggregate_bytes(&[4; 257], max).is_err());
        assert!(aggregate_bytes(&[4], 0).is_err());
        assert!(aggregate_bytes(&[4], max + 1).is_err());
        assert!(aggregate_bytes(&[usize::MAX], max).is_err());
    }

    #[test]
    fn diagnostic_partition_accounts_for_direct_and_contextual_router_slots() {
        // Seven graphs, including the non-monotonic [8, 9, 7] query mapping.
        // Values are deliberately absent: this is a pure metadata fixture.
        let mut shapes = vec![vec![2]; 26];
        shapes[7] = vec![2, 3];
        let mut seen = vec![false; shapes.len()];
        for slot in [0, 1, 4, 5, 6] {
            mark_slot(&mut seen, slot).unwrap();
        }
        for (range, router, count) in [
            (2..4, None, 2),
            (8..10, Some(7), 3),
            (10..12, None, 2),
            (12..14, None, 2),
            (14..16, None, 2),
            (16..22, None, 6),
            (22..26, None, 4),
        ] {
            let bindings = diagnostic_bindings(range, router, count).unwrap();
            let graph_shapes: Vec<_> = bindings
                .iter()
                .map(|b| {
                    if b.flattened {
                        vec![6]
                    } else {
                        shapes[b.model_slot].clone()
                    }
                })
                .collect();
            validate_binding_shapes(&shapes, &mut seen, &bindings, &graph_shapes).unwrap();
        }
        complete_partition(&seen).unwrap();
    }

    #[test]
    fn diagnostic_partition_rejects_duplicates_missing_and_out_of_range_slots() {
        let mut seen = vec![false; 3];
        mark_slot(&mut seen, 0).unwrap();
        assert!(mark_slot(&mut seen, 0).is_err());
        assert!(mark_slot(&mut seen, 3).is_err());
        assert!(complete_partition(&seen).is_err());
        mark_slot(&mut seen, 1).unwrap();
        mark_slot(&mut seen, 2).unwrap();
        complete_partition(&seen).unwrap();

        let binding = diagnostic_bindings(0..1, None, 1).unwrap();
        let mut seen = vec![true];
        assert!(validate_binding_shapes(&[vec![2]], &mut seen, &binding, &[vec![2]]).is_err());
    }

    #[test]
    fn diagnostic_shape_exception_is_only_a_two_dimensional_router_flatten() {
        let direct = diagnostic_bindings(0..1, None, 1).unwrap();
        let router = diagnostic_bindings(1..1, Some(0), 1).unwrap();
        assert!(validate_binding_shapes(&[vec![2, 3]], &mut [false], &direct, &[vec![6]]).is_err());
        validate_binding_shapes(&[vec![2, 3]], &mut [false], &router, &[vec![6]]).unwrap();
        assert!(
            validate_binding_shapes(&[vec![2, 3]], &mut [false], &router, &[vec![3, 2]]).is_err()
        );
        assert!(validate_binding_shapes(&[vec![2, 3]], &mut [false], &router, &[vec![5]]).is_err());
        assert!(
            validate_binding_shapes(&[vec![1, 2, 3]], &mut [false], &router, &[vec![6]]).is_err()
        );
        assert!(
            validate_binding_shapes(&[vec![usize::MAX, 2]], &mut [false], &router, &[vec![0]])
                .is_err()
        );
        assert!(validate_binding_shapes(&[vec![2]], &mut [], &direct, &[vec![2]]).is_err());
        assert!(validate_binding_shapes(&[vec![2]], &mut [false], &direct, &[]).is_err());
        let missing = diagnostic_bindings(1..2, None, 1).unwrap();
        assert!(validate_binding_shapes(&[vec![2]], &mut [false], &missing, &[vec![2]]).is_err());
        validate_binding_shapes(&[], &mut [], &[], &[]).unwrap();
    }
}
