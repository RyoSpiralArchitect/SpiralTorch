use super::direct_io_tests::{graph, runtime};
use super::*;
use st_kernel_contracts::{
    graph::{GraphParameter, ParameterRole},
    pointwise::{PointwiseChain, PointwiseStep},
};

fn relu() -> GraphStage {
    GraphStage::Pointwise {
        chain: PointwiseChain::new(1, vec![PointwiseStep::named("relu", None).unwrap()]).unwrap(),
        parameters: vec![],
    }
}

fn guarded_definition(bad: usize) -> GraphDefinition {
    let mut stages = Vec::new();
    let mut parameters = Vec::new();
    for block in 0..12 {
        let value = if block == bad { -f32::MAX } else { 1. };
        let id = parameters.len();
        if block % 2 == 0 {
            parameters.push(GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![1],
                values: vec![value],
            });
            stages.push(GraphStage::Pointwise {
                chain: PointwiseChain::new(
                    2,
                    vec![PointwiseStep::named("multiply", Some(1)).unwrap()],
                )
                .unwrap(),
                parameters: vec![id],
            });
        } else {
            parameters.push(GraphParameter {
                role: ParameterRole::Weight,
                shape: vec![1, 1],
                values: vec![value],
            });
            parameters.push(GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![1],
                values: vec![0.],
            });
            stages.push(GraphStage::Linear {
                weight: id,
                bias: id + 1,
                gelu: false,
            });
        }
        // Negative overflow is masked immediately, but both stage guards must
        // survive. All earlier stages pass the positive input through.
        stages.push(relu());
    }
    GraphDefinition::new(NdLayout::contiguous(&[2, 1]).unwrap(), stages, parameters).unwrap()
}

#[test]
fn shared_guard_preserves_every_stage_and_queued_snapshot_across_reuse() {
    let Some(runtime) = runtime() else { return };
    for bad in [0, 1, 6, 11] {
        let definition = guarded_definition(bad);
        for direct in [false, true] {
            let mut graph = graph(&runtime, definition.clone());
            let device = graph.tensor_device().clone();
            if direct {
                graph
                    .forward_tensor(&device.upload(&[2, 1], &[2.; 2]).unwrap())
                    .unwrap();
            } else {
                graph.upload(&[2.; 2]).unwrap();
                graph.dispatch().unwrap();
            }
            let pending = graph.snapshot().unwrap();
            let owned = graph.output_tensor().unwrap();
            let mut raw = graph.snapshot().unwrap();
            let bytes = raw
                .staging
                .read(
                    &raw.context,
                    std::time::Duration::from_secs(30),
                    "graph.guard.test",
                )
                .unwrap();
            let flags: Vec<_> = bytes[8..]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            assert_eq!(flags.len(), 25);
            for (stage, &flag) in flags.iter().enumerate() {
                assert_eq!(
                    flag != 0,
                    stage == bad * 2 || stage == bad * 2 + 1,
                    "bad={bad} direct={direct} stage={stage}"
                );
            }
            let safe = device.upload(&[2, 1], &[0.; 2]).unwrap();
            for _ in 0..8 {
                drop(graph.forward_tensor(&safe).unwrap());
            }
            assert_eq!(graph.snapshot().unwrap().read().unwrap(), vec![0.; 2]);
            assert!(
                matches!(pending.read(), Err(GraphInferenceError::NonFinite { stage, .. }) if stage == bad * 2)
            );
            assert!(owned.snapshot().unwrap().read().is_err());
        }
    }
}

#[test]
fn indexed_pointwise_guard_keeps_other_words_and_inherits_empty_input_failures() {
    let Some(runtime) = runtime() else { return };
    let device = TensorDevice::new(runtime).unwrap();
    let context = device.runtime().context();
    let gpu = context.device();
    assert!(PointwisePlan::new_with_flag_slot(
        device.clone(),
        PointwiseChain::new(1, vec![PointwiseStep::named("relu", None).unwrap()]).unwrap(),
        vec![NdLayout::contiguous(&[1]).unwrap()],
        u32::MAX,
    )
    .is_err());
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    for shape in [vec![], vec![0], vec![3, 1]] {
        for slot in [0, 3, 31] {
            for inherited in [false, true] {
                let layout = NdLayout::contiguous(&shape).unwrap();
                if layout.is_empty() && !inherited {
                    continue;
                }
                let chain =
                    PointwiseChain::new(1, vec![PointwiseStep::named("relu", None).unwrap()])
                        .unwrap();
                let plan = PointwisePlan::new_with_flag_slot(
                    device.clone(),
                    chain,
                    vec![layout.clone()],
                    slot,
                )
                .unwrap();
                let values =
                    vec![if inherited { 1. } else { f32::NEG_INFINITY }; layout.len().max(1)];
                let input = runtime::upload_slice(gpu, "guard.input", &values, usage).unwrap();
                let output =
                    runtime::empty_buffer::<f32>(gpu, "guard.output", layout.len().max(1), usage)
                        .unwrap();
                let upstream =
                    runtime::upload_slice(gpu, "guard.upstream", &[u32::from(inherited)], usage)
                        .unwrap();
                let mut expected: Vec<u32> = (0..33).map(|i| 0x40 + i).collect();
                expected[slot as usize] = 0;
                let flags = runtime::upload_slice(gpu, "guard.flags", &expected, usage).unwrap();
                let pool = runtime::ReadbackPool::new::<u32>(context.clone(), 33).unwrap();
                let mut snapshot = pool.checkout("guard.words");
                let mut encoder = gpu.create_command_encoder(&Default::default());
                plan.encode_into(&mut encoder, &[&input], &output, &upstream, &flags);
                encoder.copy_buffer_to_buffer(&flags, 0, snapshot.buffer(), 0, flags.size());
                context.queue().submit(Some(encoder.finish()));
                let bytes = snapshot
                    .read(context, std::time::Duration::from_secs(30), "guard.words")
                    .unwrap();
                let actual: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                expected[slot as usize] = crate::resident_tensor::INVALID_TENSOR_FLAG;
                assert_eq!(
                    actual, expected,
                    "shape={shape:?} slot={slot} inherited={inherited}"
                );
            }
        }
    }
}

#[test]
fn terminal_guard_observes_late_workgroups_and_survives_queued_valid_forwards() {
    let Some(runtime) = runtime() else { return };
    let shape = [3, 7, 65];
    let width = 65;
    let count = shape.iter().product::<usize>();
    let mut input = vec![0.; count];
    input[count - 1] = 2.;
    for dense in [false, true] {
        for bad_stage in [0, 17, 33] {
            let mut stages = vec![relu(); bad_stage];
            let mut values = vec![1.; width];
            values[width - 1] = -f32::MAX;
            let parameters = if dense {
                let mut weights = vec![0.; width * width];
                for i in 0..width {
                    weights[i * width + i] = values[i];
                }
                stages.push(GraphStage::Linear {
                    weight: 0,
                    bias: 1,
                    gelu: false,
                });
                vec![
                    GraphParameter {
                        role: ParameterRole::Weight,
                        shape: vec![width, width],
                        values: weights,
                    },
                    GraphParameter {
                        role: ParameterRole::Bias,
                        shape: vec![width],
                        values: vec![0.; width],
                    },
                ]
            } else {
                stages.push(GraphStage::Pointwise {
                    chain: PointwiseChain::new(
                        2,
                        vec![PointwiseStep::named("multiply", Some(1)).unwrap()],
                    )
                    .unwrap(),
                    parameters: vec![0],
                });
                vec![GraphParameter {
                    role: ParameterRole::Gain,
                    shape: vec![width],
                    values,
                }]
            };
            stages.push(relu());
            let definition =
                GraphDefinition::new(NdLayout::contiguous(&shape).unwrap(), stages, parameters)
                    .unwrap();
            let mut graph = graph(&runtime, definition);
            let device = graph.tensor_device().clone();
            let bad = device.upload(&shape, &input).unwrap();
            let safe = device.upload(&shape, &vec![0.; count]).unwrap();
            let old = graph.forward_tensor(&bad).unwrap();
            let pending = graph.snapshot().unwrap();
            let pending_owner = old.snapshot().unwrap();
            for _ in 0..16 {
                drop(graph.forward_tensor(&safe).unwrap());
            }
            assert_eq!(graph.snapshot().unwrap().read().unwrap(), vec![0.; count]);
            assert!(
                matches!(pending.read(), Err(GraphInferenceError::NonFinite { stage, .. }) if stage == bad_stage)
            );
            assert!(pending_owner.read().is_err());
            assert!(old.snapshot().unwrap().read().is_err());
            // The legacy separate output-capture path remains an independent
            // reference for the same stage flags, after switching API routes.
            graph.upload(&input).unwrap();
            graph.dispatch().unwrap();
            assert!(
                matches!(graph.snapshot().unwrap().read(), Err(GraphInferenceError::NonFinite { stage, .. }) if stage == bad_stage)
            );
            assert!(graph
                .output_tensor()
                .unwrap()
                .snapshot()
                .unwrap()
                .read()
                .is_err());
            drop(old);
            for _ in 0..16 {
                drop(graph.forward_tensor(&safe).unwrap());
            }
            assert_eq!(
                graph
                    .output_tensor()
                    .unwrap()
                    .snapshot()
                    .unwrap()
                    .read()
                    .unwrap(),
                vec![0.; count]
            );
        }
    }
}
