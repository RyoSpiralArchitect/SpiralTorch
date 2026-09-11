use super::*;
use crate::resident_graph::direct_io_tests::{definition, graph, read, runtime};
use crate::resident_tensor::pointwise::PointwiseInputs;
use st_kernel_contracts::{
    graph::{GraphParameter, ParameterRole},
    pointwise::{PointwiseChain, PointwiseExecution, PointwiseStep},
};

fn scale(runtime: &WgpuRuntime, gain: f32) -> ResidentGraph {
    let multiply = GraphStage::Pointwise {
        chain: PointwiseChain::new(2, vec![PointwiseStep::named("multiply", Some(1)).unwrap()])
            .unwrap(),
        parameters: vec![0],
    };
    let relu = GraphStage::Pointwise {
        chain: PointwiseChain::new(1, vec![PointwiseStep::named("relu", None).unwrap()]).unwrap(),
        parameters: vec![],
    };
    graph(
        runtime,
        GraphDefinition::new(
            NdLayout::contiguous(&[2, 1]).unwrap(),
            vec![multiply, relu],
            vec![GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![1],
                values: vec![gain],
            }],
        )
        .unwrap(),
    )
}

#[test]
fn retention_caps_count_and_actual_output_bytes_without_overflow() {
    assert_eq!(retention_limit(1), 4);
    assert_eq!(retention_limit(MAX_OUTPUT_BYTES as usize / 16 - 1), 4);
    assert_eq!(retention_limit(MAX_OUTPUT_BYTES as usize / 8 - 1), 2);
    assert_eq!(retention_limit(MAX_OUTPUT_BYTES as usize / 4 - 1), 1);
    assert_eq!(retention_limit(MAX_OUTPUT_BYTES as usize / 4), 0);
    assert_eq!(retention_limit(usize::MAX), 0);
}

#[test]
fn outputs_at_the_byte_budget_are_not_retained_on_the_real_device() {
    let Some(runtime) = runtime() else { return };
    let elements = MAX_OUTPUT_BYTES as usize / 4;
    let def = GraphDefinition::new(
        NdLayout::contiguous(&[elements]).unwrap(),
        vec![GraphStage::Pointwise {
            chain: PointwiseChain::new(1, vec![PointwiseStep::named("relu", None).unwrap()])
                .unwrap(),
            parameters: vec![],
        }],
        vec![],
    )
    .unwrap();
    let mut graph = graph(&runtime, def);
    let input = graph
        .tensor_device()
        .upload(&[elements], &vec![1.; elements])
        .unwrap();
    for _ in 0..2 {
        let output = graph.forward_tensor(&input).unwrap();
        assert_eq!(read(&output.narrow(0, 0, 2).unwrap()), vec![1.; 2]);
        assert!(
            graph.output_slots.is_empty(),
            "the owning guard also counts against the byte budget"
        );
    }
    assert_eq!(graph.direct_stats.allocations, 2);
    assert_eq!(graph.direct_stats.reuses, 0);
}

#[test]
fn fixed_input_reuses_outputs_and_bindings_but_rejects_stale_binding_keys() {
    let Some(runtime) = runtime() else { return };
    let mut graph = graph(&runtime, definition("pp", false));
    let device = graph.tensor_device().clone();
    let input = device.upload(&[2, 3, 3], &[1.; 18]).unwrap();
    for _ in 0..20 {
        drop(graph.forward_tensor(&input).unwrap());
    }
    assert_eq!(read(&graph.output_tensor().unwrap()), vec![1.; 18]);
    assert_eq!(
        graph.direct_stats,
        DirectStats {
            allocations: 2,
            reuses: 18,
            input_bindings: 1
        }
    );
    assert_eq!(graph.output_slots.len(), 2);
    graph.dispatch().unwrap();
    graph.forward_tensor(&input).unwrap();
    assert_eq!(graph.direct_stats.input_bindings, 1);
    let before = graph.direct_stats;
    assert!(graph
        .forward_tensor(&input.reshape(&[6, 3]).unwrap())
        .is_err());
    assert_eq!(graph.direct_stats, before);
    graph.upload(&[3.; 18]).unwrap();
    assert!(graph.input_binding.is_none());
    assert_eq!(read(&graph.forward_tensor(&input).unwrap()), vec![1.; 18]);
    let other = device.upload(&[2, 3, 3], &[2.; 18]).unwrap();
    graph.set_input_tensor(&other).unwrap();
    assert!(graph.input_binding.is_none());
    assert_eq!(read(&graph.forward_tensor(&input).unwrap()), vec![1.; 18]);
    assert_eq!(read(&graph.forward_tensor(&other).unwrap()), vec![2.; 18]);
    assert_eq!(graph.direct_stats.input_bindings, 4);
}

#[test]
fn recurrent_outputs_recycle_with_pending_snapshots_but_never_observed_views() {
    let Some(runtime) = runtime() else { return };
    for retain in [false, true] {
        let mut graph = scale(&runtime, 2.);
        let mut current = graph
            .tensor_device()
            .upload(&[2, 1], &[0.125, 0.25])
            .unwrap();
        let mut view = None;
        let mut snapshot = None;
        let mut inputs = PointwiseInputs::new();
        for i in 0..20 {
            current = graph.forward_tensor(&current).unwrap();
            if i == 0 {
                snapshot = Some(current.snapshot().unwrap());
                if retain {
                    view = Some(current.narrow(0, 0, 1).unwrap());
                    inputs.add(&current).unwrap();
                }
            }
        }
        let allocations = if retain { 4 } else { 3 };
        assert_eq!(graph.direct_stats.allocations, allocations);
        assert_eq!(graph.direct_stats.reuses, 20 - allocations);
        assert_eq!(graph.direct_stats.input_bindings, 20);
        assert_eq!(read(&current), vec![131072., 262144.]);
        assert_eq!(snapshot.unwrap().read().unwrap(), vec![0.25, 0.5]);
        drop(graph);
        if let Some(view) = view {
            assert_eq!(read(&view), vec![0.25]);
            let plan = inputs
                .compile(
                    PointwiseChain::new(1, vec![PointwiseStep::named("identity", None).unwrap()])
                        .unwrap(),
                )
                .unwrap();
            assert_eq!(
                read(&inputs.run(&plan, PointwiseExecution::Fused).unwrap()),
                vec![0.25, 0.5]
            );
        }
        assert_eq!(read(&current), vec![131072., 262144.]);
    }
}

#[test]
fn held_versions_beyond_budget_allocate_separately_and_later_recycle() {
    let Some(runtime) = runtime() else { return };
    let mut graph = scale(&runtime, 2.);
    let device = graph.tensor_device().clone();
    let mut outputs = Vec::new();
    for i in 0..12 {
        let input = device.upload(&[2, 1], &[i as f32; 2]).unwrap();
        let output = graph.forward_tensor(&input).unwrap();
        assert!(outputs.iter().all(|old| !output.shares_storage_with(old)));
        outputs.push(output);
    }
    assert_eq!(graph.output_slots.len(), MAX_OUTPUT_SLOTS);
    assert_eq!(graph.direct_stats.allocations, 12);
    assert_eq!(graph.direct_stats.reuses, 0);
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(read(output), vec![2. * i as f32; 2]);
    }
    drop(outputs);
    let reused = graph
        .forward_tensor(&device.upload(&[2, 1], &[7.; 2]).unwrap())
        .unwrap();
    assert_eq!(graph.direct_stats.allocations, 12);
    assert_eq!(graph.direct_stats.reuses, 1);
    assert_eq!(read(&reused), vec![14.; 2]);
}

#[test]
fn recycled_flags_reset_while_old_snapshots_and_consumers_keep_their_error() {
    let Some(runtime) = runtime() else { return };
    for retain_consumer in [false, true] {
        let mut producer = scale(&runtime, f32::MAX);
        let device = producer.tensor_device().clone();
        let bad = producer
            .forward_tensor(&device.upload(&[2, 1], &[-2.; 2]).unwrap())
            .unwrap();
        let snapshot = bad.snapshot().unwrap();
        let mut consumer = scale(&runtime, 0.);
        if retain_consumer {
            consumer.set_input_tensor(&bad).unwrap();
        }
        drop(bad);
        let good = device.upload(&[2, 1], &[0.; 2]).unwrap();
        for _ in 0..20 {
            drop(producer.forward_tensor(&good).unwrap());
        }
        assert!(producer.direct_stats.reuses > 15);
        assert_eq!(read(&producer.output_tensor().unwrap()), vec![0.; 2]);
        assert!(snapshot.read().is_err());
        if retain_consumer {
            consumer.dispatch().unwrap();
            assert!(consumer.snapshot().unwrap().read().is_err());
        }
    }
}

#[test]
fn another_thread_retaining_a_version_prevents_its_reuse() {
    let Some(runtime) = runtime() else { return };
    let mut graph = scale(&runtime, 2.);
    let device = graph.tensor_device().clone();
    let first = graph
        .forward_tensor(&device.upload(&[2, 1], &[1.; 2]).unwrap())
        .unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    // The consumer intentionally reads only after the producer's GPU work has
    // completed; concurrent ownership, not competing GPU timing, is the test.
    let reader = std::thread::spawn(move || {
        receive.recv().unwrap();
        read(&first)
    });
    let input = device.upload(&[2, 1], &[7.; 2]).unwrap();
    for _ in 0..20 {
        drop(graph.forward_tensor(&input).unwrap());
    }
    assert_eq!(read(&graph.output_tensor().unwrap()), vec![14.; 2]);
    send.send(()).unwrap();
    assert_eq!(reader.join().unwrap(), vec![2.; 2]);
}
