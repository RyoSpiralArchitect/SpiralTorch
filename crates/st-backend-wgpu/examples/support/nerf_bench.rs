//! Matched complete-ray rendering. Timings include the stated terminal map,
//! not isolated GPU execution. Raw data stays in the caller's output artifact.
use serde_json::{json, Value};
use st_backend_wgpu::{
    nerf::{NerfError, NerfRay, RaySampling, ResidentNerf, ResidentRays},
    resident_graph::ResidentGraph,
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{ResidentTensor, TensorError},
    runtime::WgpuRuntime,
};
use st_kernel_contracts::{
    graph::{GraphDefinition, GraphParameter, GraphStage, ParameterRole},
    layout::NdLayout,
    pointwise::{PointwiseChain, PointwiseStep},
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WARMUP: usize = 3;
const BLOCKS: usize = 9;
const SEED: u32 = 17;

async fn read(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(snapshot.read()?)
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(snapshot.read_async().await?)
    }
}

fn graph(rows: usize, count: usize, hidden: usize) -> Result<GraphDefinition> {
    let dims = if hidden == 0 {
        vec![3, 4]
    } else {
        vec![3, hidden, 4]
    };
    let mut parameters = Vec::new();
    let mut stages = Vec::new();
    for (layer, pair) in dims.windows(2).enumerate() {
        let [input, output] = [pair[0], pair[1]];
        let weight = parameters.len();
        parameters.push(GraphParameter {
            role: ParameterRole::Weight,
            shape: vec![input, output],
            values: (0..input * output)
                .map(|i| ((i * 7 + layer * 3) % 23) as f32 / 128. - 11. / 128.)
                .collect(),
        });
        parameters.push(GraphParameter {
            role: ParameterRole::Bias,
            shape: vec![output],
            values: if layer + 2 == dims.len() {
                vec![0.7, 0.2, 0.4, -0.1]
            } else {
                (0..output).map(|i| (i % 7) as f32 / 32. - 0.05).collect()
            },
        });
        stages.push(GraphStage::Linear {
            weight,
            bias: weight + 1,
            gelu: false,
        });
        if layer + 2 != dims.len() {
            stages.push(GraphStage::Pointwise {
                chain: PointwiseChain::new(1, vec![PointwiseStep::named("relu", None)?])?,
                parameters: vec![],
            });
        }
    }
    Ok(GraphDefinition::new(
        NdLayout::contiguous(&[rows, count, 3])?,
        stages,
        parameters,
    )?)
}

#[derive(Clone, Copy)]
enum Connection {
    Staged,
    Separate,
    Single,
    Packed,
    Rows,
}

#[derive(Clone, Copy)]
pub enum Comparison {
    StagedDirect,
    Submissions,
    InputRows,
}

fn render(
    nerf: &ResidentNerf,
    rays: &ResidentRays,
    count: usize,
    connection: Connection,
    graph: &mut ResidentGraph,
) -> Result<ResidentTensor> {
    let mode = RaySampling::Stratified { seed: SEED };
    match connection {
        Connection::Packed | Connection::Rows => {
            let samples = nerf.sample(rays, count, mode)?;
            let input = samples.positions()?;
            let field = if matches!(connection, Connection::Packed) {
                graph.forward_tensor_packed(&input)?
            } else {
                graph.forward_tensor(&input)?
            };
            Ok(nerf.composite(&samples, &field)?)
        }
        Connection::Single => Ok(nerf.render_graph_single_submission(rays, count, mode, graph)?),
        Connection::Separate => Ok(nerf.render_graph(rays, count, mode, graph)?),
        Connection::Staged => {
            let samples = nerf.sample(rays, count, mode)?;
            graph.set_input_tensor(&samples.positions()?)?;
            graph.dispatch()?;
            Ok(nerf.composite(&samples, &graph.output_tensor()?)?)
        }
    }
}

fn close(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.len() != b.len() {
        return Err("ray output shape mismatch".into());
    }
    let mut maximum = 0f64;
    for (&a, &b) in a.iter().zip(b) {
        let delta = (f64::from(a) - f64::from(b)).abs();
        if !a.is_finite() || !b.is_finite() || delta > 4e-7 + 4e-6 * f64::from(b).abs() {
            return Err(format!("paired mismatch {a} versus {b}").into());
        }
        maximum = maximum.max(delta);
    }
    Ok(maximum)
}

pub async fn run(runtime: WgpuRuntime, now: fn() -> f64, comparison: Comparison) -> Result<Value> {
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("software adapter is not real-GPU coverage".into());
    }
    let adapter = format!("{:?}", runtime.adapter_info());
    let (connections, names, schema, comparison_name) = match comparison {
        Comparison::Submissions => (
            [Connection::Separate, Connection::Single],
            ["separate", "single"],
            "spiraltorch.nerf_submit_bench.v1",
            "separate_direct_vs_single_submission",
        ),
        Comparison::InputRows => (
            [Connection::Packed, Connection::Rows],
            ["packed", "rows"],
            "spiraltorch.nerf_row_input_bench.v1",
            "packed_input_vs_row_addressing",
        ),
        Comparison::StagedDirect => (
            [Connection::Staged, Connection::Separate],
            ["staged", "direct"],
            "spiraltorch.nerf_direct_bench.v1",
            "staged_vs_direct",
        ),
    };
    let direct_render = if matches!(comparison, Comparison::Submissions) {
        ResidentNerf::render_graph_single_submission
    } else {
        ResidentNerf::render_graph
    };
    let mut cases = Vec::new();
    let mut guard_checks = 0;
    for (rows, count) in [(1, 1), (1, 64), (65, 64), (256, 64), (1024, 64), (256, 256)] {
        for hidden in [0, 32] {
            let definition = graph(rows, count, hidden)?;
            let parameter_data: Vec<_> = definition
                .parameters()
                .iter()
                .map(|p| json!({"shape":p.shape,"values":p.values,"role":p.role.as_str()}))
                .collect();
            let make_graph = || {
                ResidentGraph::new(
                    runtime.clone(),
                    definition.clone(),
                    MatmulTile::default(),
                    MatmulKernel::Register2x2,
                    MatmulAccumulation::Sequential,
                )
            };
            let mut graphs = [make_graph()?, make_graph()?];
            let nerf = ResidentNerf::new(graphs[0].tensor_device().clone())?;
            let rays: Vec<_> = (0..rows)
                .map(|i| NerfRay {
                    origin: [0.1 + i as f32 / 1024., -0.2, 0.3],
                    direction: [0.4, 0.5, -0.6],
                    near: -0.25,
                    far: 1. + (i % 7) as f32 * 0.125,
                })
                .collect();
            let uploaded = nerf.upload_rays(&rays)?;
            let reference = read(&render(
                &nerf,
                &uploaded,
                count,
                connections[0],
                &mut graphs[0],
            )?)
            .await?;
            close(
                &reference,
                &read(&render(
                    &nerf,
                    &uploaded,
                    count,
                    connections[1],
                    &mut graphs[1],
                )?)
                .await?,
            )?;
            let before = graphs[1].submitted_dispatches();
            if !matches!(
                direct_render(&nerf, &uploaded, 0, RaySampling::Midpoint, &mut graphs[1]),
                Err(NerfError::Empty)
            ) || !matches!(
                direct_render(
                    &nerf,
                    &uploaded,
                    count + 1,
                    RaySampling::Midpoint,
                    &mut graphs[1]
                ),
                Err(NerfError::GraphShape)
            ) || graphs[1].submitted_dispatches() != before
            {
                return Err("graph admission mutated workspace".into());
            }
            let held = render(&nerf, &uploaded, count, connections[1], &mut graphs[1])?;
            let bad = nerf.upload_rays(&vec![
                NerfRay {
                    origin: [0.; 3],
                    direction: [f32::MAX; 3],
                    near: 8.,
                    far: 16.
                };
                rows
            ])?;
            let rejected =
                direct_render(&nerf, &bad, count, RaySampling::Midpoint, &mut graphs[1])?;
            if !read(&rejected).await.is_err_and(|e| {
                matches!(
                    e.downcast_ref::<TensorError>(),
                    Some(TensorError::NonFinite)
                )
            }) {
                return Err("graph hid invalid sampled positions".into());
            }
            close(&read(&held).await?, &reference)?;
            guard_checks += 1;
            let mut intervals = Vec::new();
            let mut last = [Vec::new(), Vec::new()];
            for burst in [1, 4] {
                for block in 0..WARMUP + BLOCKS {
                    let order = if (block + rows + hidden) % 2 == 0 {
                        [0, 1]
                    } else {
                        [1, 0]
                    };
                    for route in order {
                        let graph = &mut graphs[route];
                        let before = graph.submitted_dispatches();
                        let start = now();
                        let mut output = None;
                        for _ in 0..burst {
                            output =
                                Some(render(&nerf, &uploaded, count, connections[route], graph)?);
                        }
                        let values = read(output.as_ref().unwrap()).await?;
                        let elapsed_ms = now() - start;
                        let error = close(&values, &reference)?;
                        if graph.submitted_dispatches() - before != burst as u64
                            || !elapsed_ms.is_finite()
                            || elapsed_ms <= 0.
                        {
                            return Err("invalid completed-render interval".into());
                        }
                        last[route] = values;
                        if block >= WARMUP {
                            intervals.push(json!({"block":block-WARMUP,"burst":burst,
                                "route":names[route],
                                "order":order,"elapsed_ms":elapsed_ms,"max_abs_error":error}));
                        }
                    }
                }
            }
            cases.push(json!({"rays":rows,"samples":count,"hidden":hidden,"seed":SEED,
                "ray_inputs":rays.iter().map(|r| [r.origin[0],r.origin[1],r.origin[2],
                    r.direction[0],r.direction[1],r.direction[2],r.near,r.far]).collect::<Vec<_>>(),
                "parameters":parameter_data,"reference":reference,"last_outputs":last,"intervals":intervals}));
        }
    }
    Ok(json!({"schema":schema,"status":"passed",
        "comparison":comparison_name,
        "guard_cases":guard_checks,
        "adapter":adapter,"warmup":WARMUP,"blocks":BLOCKS,"bursts":[1,4],"cases":cases,
        "kernel":"register_2x2","accumulation":"sequential",
        "boundary":"Prepared identical rays/parameters; every iteration samples rays, evaluates the NN and composites. One owning RGBA/guard snapshot read at the end of each 1/4-render interval; queue completion and map included. Setup/compilation/uploads and numerical comparisons/serialization excluded. Paired correctness only until independent control is run; no isolated GPU timing or training claim."}))
}
