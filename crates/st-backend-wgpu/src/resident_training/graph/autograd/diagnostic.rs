//! Explicit copies of a latest forward's actual bindings, not owner checkpoints.
use super::*;

/// Hard ceiling per graph capture, including both guards and every value.
/// Callers composing captures must also enforce their own aggregate budget.
/// This is a staging-payload limit, not a bound on peak host memory.
pub const GRAPH_FORWARD_DIAGNOSTIC_MAX_BYTES: usize = 16 * 1024 * 1024;

/// Raw f32 bit patterns and guards captured from an actual graph forward.
/// Invalid values are evidence: NaN payloads, signed zero and nonzero guards
/// are preserved, not normalized or promoted to numerical acceptance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphForwardDiagnostic {
    pub input_generation: u64,
    pub submitted_forward: u64,
    /// This graph's rebinding counter, not its enclosing model's revision.
    pub parameter_revision: u64,
    /// Cumulative counter. It does not prove a pre-backward capture boundary.
    pub submitted_backwards_at_capture: u64,
    pub parameter_shapes: Vec<Vec<usize>>,
    pub parameter_bits: Vec<Vec<u32>>,
    /// Input first, then each stage's output. The terminal output is the
    /// owning prediction, never the unused terminal scratch allocation.
    pub activation_shapes: Vec<Vec<usize>>,
    pub activation_bits: Vec<Vec<u32>>,
    pub forward_flags: Vec<u32>,
    pub prediction_flags: u32,
}

#[derive(Clone)]
struct CaptureLayout {
    parameter_shapes: Vec<Vec<usize>>,
    activation_shapes: Vec<Vec<usize>>,
    lengths: Vec<usize>,
    bytes: usize,
}

fn capture_bytes(lengths: &[usize], max_bytes: usize) -> Result<usize, TrainingError> {
    if max_bytes == 0 || max_bytes > GRAPH_FORWARD_DIAGNOSTIC_MAX_BYTES {
        return Err(TensorError::Limit("graph diagnostic byte budget").into());
    }
    let bytes = lengths.iter().try_fold(0usize, |sum, &len| {
        len.checked_mul(4)
            .and_then(|bytes| sum.checked_add(bytes))
            .ok_or(TrainingError::Overflow)
    })?;
    if bytes > max_bytes {
        return Err(TensorError::Limit("graph diagnostic byte budget").into());
    }
    Ok(bytes)
}

impl CaptureLayout {
    fn new(definition: &GraphDefinition, max_bytes: usize) -> Result<Self, TrainingError> {
        let parameter_shapes = definition
            .parameters()
            .iter()
            .map(|p| p.shape.clone())
            .collect();
        let activation_shapes = definition
            .layouts()
            .iter()
            .map(|l| l.shape().to_vec())
            .collect();
        let flags = definition
            .stages()
            .len()
            .checked_add(4)
            .ok_or(TrainingError::Overflow)?;
        let lengths: Vec<_> = [flags, 1]
            .into_iter()
            .chain(definition.parameters().iter().map(|p| p.values.len()))
            .chain(definition.layouts().iter().map(|l| l.len()))
            .collect();
        let bytes = capture_bytes(&lengths, max_bytes)?;
        Ok(Self {
            parameter_shapes,
            activation_shapes,
            lengths,
            bytes,
        })
    }

    fn decode(
        self,
        words: Vec<Vec<u32>>,
        input_generation: u64,
        submitted_forward: u64,
        parameter_revision: u64,
        submitted_backwards_at_capture: u64,
    ) -> Result<GraphForwardDiagnostic, TrainingError> {
        if words.len() != self.lengths.len()
            || words.iter().zip(&self.lengths).any(|(w, &n)| w.len() != n)
        {
            return Err(TrainingError::InvalidReadback);
        }
        let mut words = words
            .into_iter()
            .map(|v| v.into_iter().map(u32::from_le).collect::<Vec<_>>());
        let forward_flags = words.next().ok_or(TrainingError::InvalidReadback)?;
        let prediction_flags = words.next().ok_or(TrainingError::InvalidReadback)?[0];
        let parameter_bits = words.by_ref().take(self.parameter_shapes.len()).collect();
        let activation_bits = words.collect();
        Ok(GraphForwardDiagnostic {
            input_generation,
            submitted_forward,
            parameter_revision,
            submitted_backwards_at_capture,
            parameter_shapes: self.parameter_shapes,
            parameter_bits,
            activation_shapes: self.activation_shapes,
            activation_bits,
            forward_flags,
            prediction_flags,
        })
    }
}

/// Copies are already submitted; mapping is explicit and consumes this snapshot.
/// Later graph rebinding, forwards or owner drop cannot alter its staging data.
pub struct GraphForwardDiagnosticReadback {
    layout: CaptureLayout,
    batch: runtime::ReadbackBatch<u32>,
    input_generation: u64,
    submitted_forward: u64,
    parameter_revision: u64,
    submitted_backwards_at_capture: u64,
}

impl GraphForwardDiagnosticReadback {
    pub fn captured_bytes(&self) -> usize {
        self.layout.bytes
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<GraphForwardDiagnostic, TrainingError> {
        self.layout.decode(
            self.batch.read()?,
            self.input_generation,
            self.submitted_forward,
            self.parameter_revision,
            self.submitted_backwards_at_capture,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<GraphForwardDiagnostic, TrainingError> {
        self.layout.decode(
            self.batch.read_async().await?,
            self.input_generation,
            self.submitted_forward,
            self.parameter_revision,
            self.submitted_backwards_at_capture,
        )
    }
}

impl ResidentGraphAutograd {
    /// Host metadata for checking a model-to-graph binding map before copying.
    /// These are the graph's fixed shapes, including explicit reshaped bindings.
    pub fn forward_diagnostic_parameter_shapes(
        &self,
        forward: &GraphForward,
    ) -> Result<Vec<Vec<usize>>, TrainingError> {
        self.forward_diagnostic_bytes(forward)?;
        Ok(self
            .graph
            .definition
            .parameters()
            .iter()
            .map(|p| p.shape.clone())
            .collect())
    }

    /// Host-only preflight for a diagnostic capture of the latest owned forward.
    /// No copies, mapping, waits, parameter changes or acceptance checks occur.
    pub fn forward_diagnostic_bytes(&self, forward: &GraphForward) -> Result<usize, TrainingError> {
        if !self
            .current
            .as_ref()
            .is_some_and(|id| Shared::ptr_eq(id, &forward.identity))
        {
            return Err(TrainingError::StaleForward);
        }
        Ok(CaptureLayout::new(&self.graph.definition, GRAPH_FORWARD_DIAGNOSTIC_MAX_BYTES)?.bytes)
    }

    /// Freeze actual graph bindings and stage values of the latest forward.
    ///
    /// This is opt-in instrumentation, not a model checkpoint or a repair. Call
    /// after the target forward, not immediately after the preceding optimizer
    /// update: graph rebinding and input assignment happen inside that forward.
    /// Copies change GPU scheduling; retain the original run outcome separately.
    /// Guards and raw bits are returned even when the forward was numerically
    /// invalid. Never use the existence of a capture as proof of acceptance.
    /// This is not the complete backward tape. A caller requiring a pre-backward
    /// boundary must invoke this before backward; backward itself does not make
    /// the forward token stale or overwrite the selected forward values.
    pub fn forward_diagnostic_snapshot(
        &self,
        forward: &GraphForward,
        max_bytes: usize,
    ) -> Result<GraphForwardDiagnosticReadback, TrainingError> {
        self.forward_diagnostic_bytes(forward)?;
        let g = &self.graph;
        let layout = CaptureLayout::new(&g.definition, max_bytes)?;
        let mut sources = vec![
            (&self.forward_validation, layout.lengths[0]),
            (forward.prediction.flags(), 1),
        ];
        sources.extend(
            g.parameters
                .iter()
                .zip(g.definition.parameters())
                .map(|(b, p)| (b, p.values.len())),
        );
        // The terminal producer bypasses activations.last(), which is stale.
        sources.extend(
            g.activations
                .iter()
                .zip(g.definition.layouts())
                .take(g.nodes.len())
                .map(|(b, l)| (b, l.len())),
        );
        sources.push((forward.prediction.values(), g.output_layout().len()));
        let batch = runtime::ReadbackBatch::copy(
            g.device.runtime().context(),
            &sources,
            "graph.forward_diagnostic",
        )?;
        Ok(GraphForwardDiagnosticReadback {
            layout,
            batch,
            input_generation: forward.identity.generation,
            submitted_forward: forward.identity.submission,
            parameter_revision: forward.identity.parameters.revision,
            submitted_backwards_at_capture: self.backwards,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition() -> GraphDefinition {
        GraphDefinition::new(
            NdLayout::contiguous(&[1, 2]).unwrap(),
            vec![GraphStage::Linear {
                weight: 0,
                bias: 1,
                gelu: false,
            }],
            vec![
                GraphParameter {
                    role: ParameterRole::Weight,
                    shape: vec![2, 2],
                    values: vec![0.; 4],
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![2],
                    values: vec![0.; 2],
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn graph_diagnostic_budget_includes_guards_parameters_and_input_output() {
        let definition = definition();
        let layout = CaptureLayout::new(&definition, 64).unwrap();
        assert_eq!(layout.lengths, [5, 1, 4, 2, 2, 2]);
        assert_eq!(layout.bytes, 64);
        assert!(CaptureLayout::new(&definition, 63).is_err());
        assert!(CaptureLayout::new(&definition, 0).is_err());
        assert!(CaptureLayout::new(&definition, GRAPH_FORWARD_DIAGNOSTIC_MAX_BYTES + 1).is_err());
        assert!(matches!(
            capture_bytes(&[usize::MAX], 64),
            Err(TrainingError::Overflow)
        ));
    }

    #[test]
    fn graph_diagnostic_decode_preserves_invalid_bits_and_host_identity() {
        let layout = CaptureLayout::new(&definition(), 64).unwrap();
        let mut words: Vec<_> = layout.lengths.iter().map(|&n| vec![0u32; n]).collect();
        words[0][1] = 3u32.to_le();
        words[1][0] = 7u32.to_le();
        words[2][0] = 0x8000_0000u32.to_le();
        words[5][0] = 0x7fc0_1234u32.to_le();
        let state = layout.decode(words, 4, 5, 6, 7).unwrap();
        assert_eq!(
            (
                state.input_generation,
                state.submitted_forward,
                state.parameter_revision,
                state.submitted_backwards_at_capture
            ),
            (4, 5, 6, 7)
        );
        assert_eq!(state.parameter_bits[0][0], 0x8000_0000);
        assert_eq!(state.activation_bits[1][0], 0x7fc0_1234);
        assert_eq!(state.forward_flags, [0, 3, 0, 0, 0]);
        assert_eq!(state.prediction_flags, 7);
    }

    #[test]
    fn graph_diagnostic_decode_rejects_missing_extra_and_truncated_segments() {
        let layout = CaptureLayout::new(&definition(), 64).unwrap();
        let words: Vec<_> = layout.lengths.iter().map(|&n| vec![0; n]).collect();
        for malformed in [
            words[..5].to_vec(),
            [words.clone(), vec![vec![]]].concat(),
            {
                let mut w = words.clone();
                w[1].clear();
                w
            },
        ] {
            assert!(matches!(
                layout.clone().decode(malformed, 0, 0, 0, 0),
                Err(TrainingError::InvalidReadback)
            ));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "Requires a separately authorized real-GPU diagnostic run"]
    fn graph_diagnostic_gpu_freezes_rebound_values_and_real_terminal_output() {
        let (runtime, _) =
            runtime::ensure_default_runtime_blocking("graph.forward.diagnostic").unwrap();
        assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
        let make = || {
            ResidentGraphAutograd::new(
                runtime.clone(),
                definition(),
                Default::default(),
                MatmulKernel::Scalar,
                Default::default(),
            )
            .unwrap()
        };
        let mut graph = make();
        let other = make();
        let device = graph.tensor_device().clone();
        graph
            .set_parameter_tensors(&[
                device.upload(&[2, 2], &[1., 0., 0., 1.]).unwrap(),
                device.upload(&[2], &[1., 2.]).unwrap(),
            ])
            .unwrap();
        graph.upload(&[3., 4.]).unwrap();
        let first = graph.forward().unwrap();
        assert_eq!(graph.forward_diagnostic_bytes(&first).unwrap(), 64);
        assert!(matches!(
            other.forward_diagnostic_snapshot(&first, 64),
            Err(TrainingError::StaleForward)
        ));
        assert!(graph.forward_diagnostic_snapshot(&first, 63).is_err());
        let frozen = graph.forward_diagnostic_snapshot(&first, 64).unwrap();
        assert_eq!(frozen.captured_bytes(), 64);
        graph
            .set_parameter_tensors(&[
                device.upload(&[2, 2], &[2., 0., 0., 2.]).unwrap(),
                device.upload(&[2], &[0., 0.]).unwrap(),
            ])
            .unwrap();
        assert!(matches!(
            graph.forward_diagnostic_snapshot(&first, 64),
            Err(TrainingError::StaleForward)
        ));
        graph.upload(&[8., 9.]).unwrap();
        let second = graph.forward().unwrap();
        let next = graph.forward_diagnostic_snapshot(&second, 64).unwrap();
        // No map occurs until both forwards and captures have been submitted.
        drop((graph, other, first, second, device));
        let frozen = frozen.read().unwrap();
        let next = next.read().unwrap();
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            frozen.parameter_bits,
            vec![bits(&[1., 0., 0., 1.]), bits(&[1., 2.])]
        );
        assert_eq!(
            frozen.activation_bits,
            vec![bits(&[3., 4.]), bits(&[4., 6.])]
        );
        assert_eq!(
            next.activation_bits,
            vec![bits(&[8., 9.]), bits(&[16., 18.])]
        );
        assert_eq!(frozen.parameter_revision, 1);
        assert_eq!(next.parameter_revision, 2);
        assert!(frozen.forward_flags.iter().all(|&v| v == 0));
        assert_eq!(frozen.prediction_flags, 0);
    }
}
