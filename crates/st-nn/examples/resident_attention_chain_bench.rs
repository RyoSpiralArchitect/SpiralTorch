//! Bounded full-chain timings against a generated, independent Torch fixture.
#[cfg(not(target_arch = "wasm32"))]
#[path = "support/attention_projection.rs"]
mod projection;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use serde_json::{json, Value};
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime};
    use st_kernel_contracts::attention::AttentionMask;
    use st_nn::{
        resident::AttentionInferencePlan,
        z_rba::attention::{SimpleZFrame, ZIndex, ZMetricWeights, ZRBFAttention},
        Tensor,
    };
    use st_tensor::NdLayout;
    use std::time::{Duration, Instant};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    fn floats(value: &Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn close(actual: &[f32], expected: &[f32]) -> Result<f32> {
        if actual.len() != expected.len() {
            return Err("output length differs".into());
        }
        let mut error = 0f32;
        for (&a, &b) in actual.iter().zip(expected) {
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 3e-6 + 3e-5 * b.abs() {
                return Err(format!("output differs: {a} != {b}").into());
            }
            error = error.max((a - b).abs());
        }
        Ok(error)
    }
    fn synchronize(runtime: &runtime::WgpuRuntime) -> Result<()> {
        runtime::submit_with_timeout(
            runtime.context().device(),
            runtime.context().queue(),
            std::iter::empty(),
            Duration::from_secs(30),
            "attention.bench.sync",
        )?;
        Ok(())
    }

    pub fn main() -> Result<()> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if !(4..=5).contains(&args.len()) {
            return Err("usage: <fixture.json> <samples>=3 <warmup>=1 <burst=1..64> [scalar|register8|register16]".into());
        }
        let projection = args.get(4).map_or("scalar", String::as_str);
        let (tile, projection_kernel) = crate::projection::options(projection)?;
        let samples = args[1].parse::<usize>()?;
        let warmup = args[2].parse::<usize>()?;
        let burst = args[3].parse::<usize>()?;
        if samples < 3 || warmup == 0 || !(1..=64).contains(&burst) {
            return Err("invalid timing recipe".into());
        }
        let fixture: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
        if fixture["schema"] != "spiraltorch.attention_chain_torch.v1" {
            return Err("wrong fixture schema".into());
        }
        let (runtime, _) = runtime::ensure_default_runtime_blocking("attention.chain.bench")?;
        if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
            return Err("real GPU required".into());
        }
        let device = TensorDevice::new(runtime.clone())?;
        let mut cases = Vec::new();
        for scenario in fixture["scenarios"].as_array().ok_or("missing scenarios")? {
            let shape: Vec<_> = scenario["input_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            let (batch, tokens, inner) = (shape[0], shape[1], shape[2]);
            let heads = scenario["heads"].as_u64().unwrap() as usize;
            let width = scenario["width"].as_u64().unwrap() as usize;
            let out = scenario["output_width"].as_u64().unwrap() as usize;
            let weights: Vec<_> = (0..4)
                .map(|i| {
                    Tensor::from_vec(
                        if i < 3 { inner } else { width },
                        if i < 3 { width } else { out },
                        floats(&scenario["weights"][i]),
                    )
                })
                .collect::<std::result::Result<_, _>>()?;
            let biases: Vec<_> = (0..4)
                .map(|i| {
                    Tensor::from_vec(
                        1,
                        if i < 3 { width } else { out },
                        floats(&scenario["biases"][i]),
                    )
                })
                .collect::<std::result::Result<_, _>>()?;
            let indices: Vec<_> = scenario["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| ZIndex {
                    band: v[0].as_u64().unwrap() as usize,
                    sheet: v[1].as_u64().unwrap() as usize,
                    echo: v[2].as_u64().unwrap() as usize,
                })
                .collect();
            let dims = scenario["frame_shape"].as_array().unwrap();
            let frame = SimpleZFrame::new(
                dims[0].as_u64().unwrap() as usize,
                dims[1].as_u64().unwrap() as usize,
                dims[2].as_u64().unwrap() as usize,
            );
            let geometry = ZRBFAttention::new(width, heads, ZMetricWeights::default(), true)?;
            let kernel = geometry.kernel_bias(&frame, &indices, &indices)?;
            let geometry_error = close(kernel.data(), &floats(&scenario["expected_kernel"]))?;
            let host_input = floats(&scenario["input"]);
            let resident_input = device.upload(&shape, &host_input)?;
            for case in scenario["cases"].as_array().unwrap() {
                let mask = if case["causal"].as_bool().unwrap() {
                    AttentionMask::Causal { query_offset: 0 }
                } else {
                    AttentionMask::None
                };
                let plan = AttentionInferencePlan::from_parameters(
                    NdLayout::contiguous(&shape)?,
                    heads,
                    mask,
                    std::array::from_fn(|i| (&weights[i], &biases[i])),
                )?;
                let mut graph = plan.compile_wgpu_with_options(
                    runtime.clone(),
                    tile,
                    projection_kernel,
                    Default::default(),
                )?;
                let host_bias = case["geometry_strength"].as_f64().map(|s| {
                    kernel
                        .data()
                        .iter()
                        .map(|v| v * s as f32)
                        .collect::<Vec<_>>()
                });
                let upload_bias = || -> Result<_> {
                    Ok(host_bias
                        .as_ref()
                        .map(|v| {
                            device
                                .upload(&[1, heads, tokens, tokens], v)?
                                .broadcast_to(&[batch, heads, tokens, tokens])
                        })
                        .transpose()?)
                };
                let resident_bias = upload_bias()?;
                let expected = floats(&case["expected"]);
                close(
                    &graph
                        .forward_merged_heads(&resident_input, None, resident_bias.as_ref())?
                        .snapshot()?
                        .read()?,
                    &expected,
                )?;
                let mut timings = Vec::new();
                for block in 0..warmup + samples {
                    for slot in 0..2 {
                        let resident = (block + slot) % 2 == 0;
                        let forwards = if resident { burst } else { 1 };
                        synchronize(&runtime)?;
                        let mut outputs = Vec::with_capacity(forwards);
                        let start = Instant::now();
                        let host_output = if resident {
                            for _ in 0..forwards {
                                outputs.push(graph.forward_merged_heads(
                                    &resident_input,
                                    None,
                                    resident_bias.as_ref(),
                                )?);
                            }
                            synchronize(&runtime)?;
                            None
                        } else {
                            let input = device.upload(&shape, &host_input)?;
                            let bias = upload_bias()?;
                            Some(
                                graph
                                    .forward_merged_heads(&input, None, bias.as_ref())?
                                    .snapshot()?
                                    .read()?,
                            )
                        };
                        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                        let mut max_error = 0f32;
                        if let Some(output) = host_output {
                            max_error = close(&output, &expected)?;
                        }
                        for output in outputs {
                            max_error =
                                max_error.max(close(&output.snapshot()?.read()?, &expected)?);
                        }
                        if !elapsed_ms.is_finite() || elapsed_ms <= 0. {
                            return Err("invalid elapsed time".into());
                        }
                        if block >= warmup {
                            timings.push(json!({"block":block-warmup,"slot":slot,"route":if resident {"resident"} else {"host_to_host"},"forwards":forwards,"elapsed_ms":elapsed_ms,"max_abs_error":max_error}));
                        }
                    }
                }
                let name = format!(
                    "{}/{}",
                    scenario["name"].as_str().unwrap(),
                    case["name"].as_str().unwrap()
                );
                eprintln!("completed {name}");
                cases.push(json!({"name":name,"shape":shape,"heads":heads,"width":width,"geometry_max_abs_error":geometry_error,"samples":timings}));
            }
        }
        if cases.is_empty() {
            return Err("empty fixture".into());
        }
        println!(
            "{}",
            json!({"schema":"spiraltorch.attention_chain_bench.v1","status":"passed","engine":"st_wgpu","projection":projection,"output_path":"direct_merged_heads","adapter":format!("{:?}",runtime.adapter_info()),"warmup":warmup,"samples_per_route":samples,"burst":burst,"cases":cases,"boundary":"Resident: fixed input/bias/weights, burst forwards, queue completion included, reads/checks excluded. Host-to-host: fresh input and bias upload plus owning output read each forward, weights remain resident. Setup, compile and geometry construction excluded. This is a host-timed inference comparison of the explicit merged-head path, not the default NN route, kernel timestamps or training."})
        );
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::main()
}
#[cfg(target_arch = "wasm32")]
fn main() {}
