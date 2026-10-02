//! Ignored research probe: timestamps the existing attention/guard passes.
//! QKV is produced by the same resident graph primitive as NN, not host math.

use super::*;
use crate::{
    resident_graph::ResidentGraph,
    resident_matmul::{MatmulKernel, MatmulTile},
    runtime::timestamps::{PassTimestampRecorder, TimestampErrorScopes},
};
use serde_json::{json, Value};
use st_kernel_contracts::graph::{GraphDefinition, GraphParameter, GraphStage, ParameterRole};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SAMPLES: usize = 25;
const WARMUP: usize = 50;

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn linear(
    runtime: WgpuRuntime,
    shape: &[usize],
    width: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    tile: MatmulTile,
    kernel: MatmulKernel,
) -> Result<ResidentGraph> {
    Ok(ResidentGraph::new(
        runtime,
        GraphDefinition::new(
            NdLayout::contiguous(shape)?,
            vec![GraphStage::Linear {
                weight: 0,
                bias: 1,
                gelu: false,
            }],
            vec![
                GraphParameter {
                    role: ParameterRole::Weight,
                    shape: vec![*shape.last().unwrap(), width],
                    values: weights,
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![width],
                    values: bias,
                },
            ],
        )?,
        tile,
        kernel,
        Default::default(),
    )?)
}

fn merge(tensor: ResidentTensor, shape: [usize; 4], direct: bool) -> Result<ResidentTensor> {
    let [batch, heads, queries, dim] = shape;
    Ok(if direct {
        tensor
    } else {
        tensor
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[batch, queries, heads * dim])?
    })
}

fn close(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err("output length".into());
    }
    let mut error = 0f32;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 3e-6 + 3e-5 * b.abs() {
            return Err(format!("reference mismatch: {a} != {b}").into());
        }
        error = error.max((a - b).abs());
    }
    Ok(error)
}

#[test]
#[ignore = "explicit GPU timestamp experiment; requires frozen fixture and fresh output paths"]
fn profile_fixture_attention_passes() -> Result<()> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return Err("explicit real-GPU test opt-in required".into());
    }
    let fixture: Value = serde_json::from_slice(&std::fs::read(std::env::var(
        "SPIRALTORCH_ATTENTION_PROFILE_FIXTURE",
    )?)?)?;
    let output_path = std::env::var("SPIRALTORCH_ATTENTION_PROFILE_OUTPUT")?;
    if std::path::Path::new(&output_path).exists() {
        return Err("fresh output path required".into());
    }
    if fixture["schema"] != "spiraltorch.attention_chain_torch.v1" {
        return Err("fixture schema".into());
    }
    let runtime = WgpuRuntime::request_profiled_headless_blocking("attention.isolated_pass_probe")?;
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("real GPU required".into());
    }
    let device = TensorDevice::new(runtime.clone())?;
    let mut records = Vec::new();
    let mut case_count = 0;
    for (projection, tile, kernel) in [
        ("scalar", MatmulTile::default(), MatmulKernel::Scalar),
        (
            "register16",
            MatmulTile::new(16, 16, 16)?,
            MatmulKernel::Register2x2,
        ),
    ] {
        for scenario in fixture["scenarios"].as_array().ok_or("scenarios")? {
            let shape: Vec<_> = scenario["input_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            let (batch, queries, inner) = (shape[0], shape[1], shape[2]);
            let heads = scenario["heads"].as_u64().unwrap() as usize;
            let width = scenario["width"].as_u64().unwrap() as usize;
            let out = scenario["output_width"].as_u64().unwrap() as usize;
            if queries == 0 || heads == 0 || !width.is_multiple_of(heads) {
                return Err("probe shape".into());
            }
            let dim = width / heads;
            let q_shape = [batch, heads, queries, dim];
            let mut packed_weights = vec![0.; inner * 3 * width];
            let mut packed_bias = Vec::new();
            for part in 0..3 {
                let weights = floats(&scenario["weights"][part]);
                for row in 0..inner {
                    packed_weights
                        [row * 3 * width + part * width..row * 3 * width + (part + 1) * width]
                        .copy_from_slice(&weights[row * width..(row + 1) * width]);
                }
                packed_bias.extend(floats(&scenario["biases"][part]));
            }
            let mut qkv = linear(
                runtime.clone(),
                &shape,
                3 * width,
                packed_weights,
                packed_bias,
                tile,
                kernel,
            )?;
            let mut output = linear(
                runtime.clone(),
                &[batch, queries, width],
                out,
                floats(&scenario["weights"][3]),
                floats(&scenario["biases"][3]),
                tile,
                kernel,
            )?;
            let input = device.upload(&shape, &floats(&scenario["input"]))?;
            let projected = qkv
                .forward_tensor(&input)?
                .reshape(&[batch, queries, 3, heads, dim])?;
            let q = projected.select(2, 0)?.permute(&[0, 2, 1, 3])?;
            let k = projected.select(2, 1)?.permute(&[0, 2, 1, 3])?;
            let v = projected.select(2, 2)?.permute(&[0, 2, 1, 3])?;
            for case in scenario["cases"].as_array().unwrap() {
                case_count += 1;
                let mask = if case["causal"].as_bool().unwrap() {
                    AttentionMask::Causal { query_offset: 0 }
                } else {
                    AttentionMask::None
                };
                let scale = 1. / (dim as f32).sqrt();
                let pair = case["geometry_strength"]
                    .as_f64()
                    .map(|strength| {
                        let values: Vec<_> = floats(&scenario["expected_kernel"])
                            .iter()
                            .map(|v| v * strength as f32)
                            .collect();
                        device
                            .upload(&[1, heads, queries, queries], &values)?
                            .broadcast_to(&[batch, heads, queries, queries])
                    })
                    .transpose()?;
                let expected = floats(&case["expected"]);
                for sample in 0..WARMUP {
                    for slot in 0..2 {
                        let direct = (sample + slot) % 2 == 1;
                        let attended = q.attention_forward(
                            &k,
                            &v,
                            scale,
                            mask,
                            None,
                            pair.as_ref(),
                            if direct {
                                OutputOrder::MergedHeads
                            } else {
                                OutputOrder::HeadMajor
                            },
                            &mut PassTimestampCursor::default(),
                        )?;
                        let result = output.forward_tensor(&merge(attended, q_shape, direct)?)?;
                        close(&result.snapshot()?.read()?, &expected)?;
                    }
                }
                let context = runtime.context();
                let scopes = TimestampErrorScopes::try_new(context.clone())?;
                let recorder = PassTimestampRecorder::new(context.clone(), (SAMPLES * 4) as u32)?;
                let mut cursor = PassTimestampCursor::new(&recorder);
                let mut retained = Vec::new();
                for sample in 0..SAMPLES {
                    for slot in 0..2 {
                        let direct = (sample + slot) % 2 == 1;
                        let tensor = q.attention_forward(
                            &k,
                            &v,
                            scale,
                            mask,
                            None,
                            pair.as_ref(),
                            if direct {
                                OutputOrder::MergedHeads
                            } else {
                                OutputOrder::HeadMajor
                            },
                            &mut cursor,
                        )?;
                        retained.push((sample, slot, direct, tensor));
                    }
                }
                cursor.finish();
                let mut encoder = context.device().create_command_encoder(&Default::default());
                let mut readback = recorder.resolve(&mut encoder);
                context.queue().submit(Some(encoder.finish()));
                readback.validate(scopes.finish());
                let times = readback.read()?;
                if times.passes.len() != retained.len() * 2 {
                    return Err("timestamp coverage".into());
                }
                for ((sample, slot, direct, tensor), times_pair) in
                    retained.into_iter().zip(times.passes.as_chunks::<2>().0)
                {
                    let result = output.forward_tensor(&merge(tensor, q_shape, direct)?)?;
                    let error = close(&result.snapshot()?.read()?, &expected)?;
                    records.push(json!({
                        "projection": projection, "scenario": scenario["name"], "case": case["name"],
                        "input_shape": shape, "sample": sample, "slot": slot,
                        "output": if direct {"merged_heads"} else {"head_major"},
                        "attention_ns": times_pair[0].elapsed_ns, "guard_ns": times_pair[1].elapsed_ns,
                        "timestamp_period_ns": times.timestamp_period_ns, "max_abs_error": error,
                    }));
                }
            }
        }
    }
    if case_count != 36 || records.len() != 36 * SAMPLES * 2 {
        return Err("fixture coverage".into());
    }
    let report = json!({
        "schema": "spiraltorch.attention_pass_probe.v1", "status": "passed",
        "adapter": format!("{:?}", runtime.adapter_info()), "samples_per_order": SAMPLES,
        "warmup_pairs": WARMUP, "records": records,
        "scope": "Attention and inherited-guard GPU pass timestamps only; fixed resident graph-produced QKV and fixture geometry. Output Linear verifies every output against Torch. QKV/output projections, head packing, host work, copies, query resolution and validation are not timed. Isolated diagnostic, not whole-chain throughput or causation proof.",
    });
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}
