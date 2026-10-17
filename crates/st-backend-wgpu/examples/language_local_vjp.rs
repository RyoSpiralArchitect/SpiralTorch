//! Read a bounded private operand fixture on stdin; replay shared GPU kernels.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel},
    resident_tensor::{attention::AttentionMask, ResidentTensor, TensorDevice},
    resident_training::graph::ResidentGraphAutograd,
    runtime,
};
use st_kernel_contracts::{
    graph::{GraphDefinition, GraphParameter, GraphStage, ParameterRole},
    layout::NdLayout,
};
use std::io::Read;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn floats(v: &Value) -> Result<Vec<f32>> {
    v.as_array()
        .ok_or("expected float array")?
        .iter()
        .map(|v| {
            let n = v.as_f64().ok_or("expected numeric value")? as f32;
            if !n.is_finite() {
                return Err("nonfinite operand".into());
            }
            Ok(n)
        })
        .collect()
}
fn read(t: &ResidentTensor) -> Result<Vec<f32>> {
    Ok(t.snapshot()?.read()?)
}

fn main() -> Result<()> {
    let mut raw = Vec::new();
    std::io::stdin()
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    if raw.len() > 64 * 1024 * 1024 {
        return Err("fixture exceeds 64 MiB".into());
    }
    let fixture: Value = serde_json::from_slice(&raw)?;
    if fixture["schema"] != "spiraltorch.language_local_vjp.fixture.v1" {
        return Err("invalid fixture schema".into());
    }
    let cases = fixture["cases"].as_array().ok_or("missing cases")?;
    if cases.is_empty() || cases.len() > 128 {
        return Err("expected 1..=128 cases".into());
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("language.local.vjp")?;
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("GPU required".into());
    }
    let device = TensorDevice::new(runtime.clone())?;
    let mut output = Vec::new();
    for case in cases {
        let shape: Vec<usize> = serde_json::from_value(case["shape"].clone())?;
        let mut modes = serde_json::Map::new();
        match case["kind"].as_str() {
            Some("linear") => {
                let [rows, inner, cols] = <[usize; 3]>::try_from(shape.as_slice())?;
                let definition = GraphDefinition::new(
                    NdLayout::contiguous(&[rows, inner])?,
                    vec![GraphStage::Linear {
                        weight: 0,
                        bias: 1,
                        gelu: false,
                    }],
                    vec![
                        GraphParameter {
                            role: ParameterRole::Weight,
                            shape: vec![inner, cols],
                            values: floats(&case["weight"])?,
                        },
                        GraphParameter {
                            role: ParameterRole::Bias,
                            shape: vec![cols],
                            values: floats(&case["bias"])?,
                        },
                    ],
                )?;
                for policy in [
                    MatmulAccumulation::Sequential,
                    MatmulAccumulation::Tiled,
                    MatmulAccumulation::Compensated,
                ] {
                    let mut graph = ResidentGraphAutograd::new(
                        runtime.clone(),
                        definition.clone(),
                        Default::default(),
                        MatmulKernel::Scalar,
                        policy,
                    )?;
                    graph.upload(&floats(&case["input"])?)?;
                    let seed = device.upload(&[rows, cols], &floats(&case["upstream"])?)?;
                    let f = graph.forward()?;
                    let g = graph.backward(&f, &seed)?;
                    modes.insert(policy.as_str().into(),json!({"forward":read(f.prediction())?, "input":read(g.input_gradient())?,
                        "weight":read(&g.parameter_gradients()[0])?,"bias":read(&g.parameter_gradients()[1])?}));
                }
            }
            Some("attention") => {
                let [b, h, t, _] = <[usize; 4]>::try_from(shape.as_slice())?;
                let upload = |key| -> Result<ResidentTensor> {
                    Ok(device.upload(&shape, &floats(&case[key])?)?)
                };
                let (q, k, v, seed) = (
                    upload("query")?,
                    upload("key")?,
                    upload("value")?,
                    upload("upstream")?,
                );
                let pair = if case["pair"].is_null() {
                    None
                } else {
                    Some(device.upload(&[b, h, t, t], &floats(&case["pair"])?)?)
                };
                let scale = case["scale"].as_f64().ok_or("missing scale")? as f32;
                let mask = AttentionMask::Causal { query_offset: 0 };
                let f = q.scaled_dot_attention(&k, &v, scale, mask, None, pair.as_ref())?;
                let g =
                    q.scaled_dot_attention_vjp(&k, &v, &seed, scale, mask, None, pair.as_ref())?;
                let mut result = json!({"forward":read(&f)?,"query":read(&g.query)?,"key":read(&g.key)?,"value":read(&g.value)?});
                if let Some(pair) = g.pair_bias {
                    result["pair"] = json!(read(&pair)?);
                }
                modes.insert("native".into(), result);
            }
            _ => return Err("unsupported operation".into()),
        }
        output.push(
            json!({"name":case["name"],"batch_index":case["batch_index"],"kind":case["kind"],
            "operation":case["operation"],"shape":shape,"modes":modes}),
        );
    }
    println!(
        "{}",
        json!({"schema":"spiraltorch.language_local_vjp.native.v1",
        "fixture_sha256":format!("{:x}",Sha256::digest(&raw)),"adapter":format!("{:?}",runtime.adapter_info()),"cases":output})
    );
    Ok(())
}
