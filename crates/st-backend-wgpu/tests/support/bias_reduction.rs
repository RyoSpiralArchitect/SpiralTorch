use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_matmul::MatmulKernel,
    resident_tensor::{ResidentTensor, TensorError},
    resident_training::graph::ResidentGraphAutograd,
    runtime::WgpuRuntime,
};
use st_kernel_contracts::{
    graph::{GraphDefinition, GraphParameter, GraphStage, ParameterRole},
    layout::NdLayout,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn read(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    Ok(values)
}

fn graph(runtime: WgpuRuntime, rows: usize) -> Result<ResidentGraphAutograd> {
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[rows, 1])?,
        vec![GraphStage::Linear {
            weight: 0,
            bias: 1,
            gelu: false,
        }],
        vec![
            GraphParameter {
                role: ParameterRole::Weight,
                shape: vec![1, 2],
                values: vec![0.; 2],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![2],
                values: vec![0.; 2],
            },
        ],
    )?;
    let mut graph = ResidentGraphAutograd::new(
        runtime,
        definition,
        Default::default(),
        MatmulKernel::Scalar,
        Default::default(),
    )?;
    graph.upload(&vec![0.; rows])?;
    Ok(graph)
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let tiny = 2f32.powi(-25);
    let large = 2f32.powi(24);
    let mut tail = vec![tiny; 257];
    tail[0] = 1.;
    tail[256] = -1.;
    let cases = [
        ("small_residual", vec![1., tiny, -1.], tiny),
        ("unit_residual", vec![large, 1., -large], 1.),
        ("nested_residual", vec![large, 1., tiny, -large, -1.], tiny),
        (
            "small_gradient",
            vec![2f32.powi(-12), 2f32.powi(-40), -2f32.powi(-12)],
            2f32.powi(-40),
        ),
        (
            "repeated_cancellation",
            [1., tiny, -1.].repeat(64),
            64. * tiny,
        ),
        ("long_tail", tail, 255. * tiny),
    ];
    let mut checks = Vec::new();
    for (name, values, expected) in cases {
        let rows = values.len();
        let mut graph = graph(runtime.clone(), rows)?;
        let seed: Vec<_> = values
            .iter()
            .zip(values.iter().rev())
            .flat_map(|(&a, &b)| [a, b])
            .collect();
        let seed = graph.tensor_device().upload(&[rows, 2], &seed)?;
        let forward = graph.forward()?;
        let gradients = graph.backward(&forward, &seed)?;
        let bias = read(&gradients.parameter_gradients()[1]).await?;
        let weight = read(&gradients.parameter_gradients()[0]).await?;
        let input = read(gradients.input_gradient()).await?;
        let passed =
            bias == [expected, expected] && weight == [0., 0.] && input.iter().all(|&v| v == 0.);
        checks.push(
            json!({"name": name, "rows": rows, "bias_forward_reverse": bias,
                          "expected": expected, "passed": passed}),
        );
    }
    for sign in [1f32, -1.] {
        let mut graph = graph(runtime.clone(), 4)?;
        let values = [f32::MAX, 2f32.powi(102), 2f32.powi(102), -2f32.powi(77)];
        let seed: Vec<_> = values.into_iter().flat_map(|v| [sign * v; 2]).collect();
        let seed = graph.tensor_device().upload(&[4, 2], &seed)?;
        let forward = graph.forward()?;
        let gradients = graph.backward(&forward, &seed)?;
        let bias = read(&gradients.parameter_gradients()[1])
            .await
            .map_err(|error| format!("finite collapse boundary {sign}: {error}"))?;
        let expected = sign * f32::MAX;
        checks.push(json!({"name":format!("finite_collapse_boundary_{sign}"),
                          "rows":4, "bias_same_order":bias, "expected":expected,
                          "passed":bias == [expected, expected]}));
    }
    let mut guards = Vec::new();
    for (name, values, inherited) in [
        ("nan_seed", [1., f32::NAN, -1.], false),
        ("infinite_seed", [1., f32::INFINITY, -1.], false),
        ("overflowing_prefix", [f32::MAX, f32::MAX, -f32::MAX], false),
        (
            "overflowing_final",
            [f32::MAX, 2f32.powi(102), 2f32.powi(102)],
            false,
        ),
        ("inherited_overflow", [1., f32::MAX, -1.], true),
    ] {
        let mut graph = graph(runtime.clone(), 3)?;
        let seed: Vec<_> = values.into_iter().flat_map(|v| [v, v]).collect();
        let forward = graph.forward()?;
        let (rejected, boundary) = match graph.tensor_device().upload(&[3, 2], &seed) {
            Err(TensorError::NonFinite) => (true, "upload"),
            Err(error) => return Err(error.into()),
            Ok(seed) => {
                let seed = if inherited {
                    seed.mul(&graph.tensor_device().upload(&[], &[2.])?)?
                } else {
                    seed
                };
                let gradients = graph.backward(&forward, &seed)?;
                let rejected = matches!(read(&gradients.parameter_gradients()[1]).await,
                    Err(error) if matches!(error.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)));
                (rejected, "guarded_read")
            }
        };
        let valid = graph.tensor_device().upload(&[3, 2], &[1.; 6])?;
        let recovery = graph.backward(&forward, &valid)?;
        let recovered = read(&recovery.parameter_gradients()[1]).await? == [3., 3.];
        guards.push(
            json!({"name":name, "rejected":rejected, "recovered":recovered, "boundary":boundary}),
        );
    }
    let mut oracle = Vec::new();
    let fixture = include_str!("../../../st-kernel-contracts/tests/fixtures/f32_sum.txt");
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let mut fields = line.split_whitespace();
        let name = fields.next().ok_or("missing fixture name")?;
        let expected = fields.next().ok_or("missing fixture expectation")?;
        // Guard cases are exercised with typed rejection and recovery above.
        let Ok(expected) = u32::from_str_radix(expected, 16) else {
            continue;
        };
        let values: Vec<_> = fields
            .map(|s| u32::from_str_radix(s, 16).map(f32::from_bits))
            .collect::<std::result::Result<_, _>>()?;
        let mut graph = graph(runtime.clone(), values.len())?;
        let seed: Vec<_> = values.iter().flat_map(|&v| [v; 2]).collect();
        let seed = graph.tensor_device().upload(&[values.len(), 2], &seed)?;
        let forward = graph.forward()?;
        let gradients = graph.backward(&forward, &seed)?;
        let observed = read(&gradients.parameter_gradients()[1]).await?;
        let bits: Vec<_> = observed.into_iter().map(f32::to_bits).collect();
        oracle.push(json!({"name":name, "expected_bits":expected, "actual_bits":bits, "passed":bits == [expected; 2]}));
    }
    Ok(json!({
        "schema": "spiraltorch.dense_bias_reduction.validation.v1",
        "adapter": format!("{:?}", runtime.adapter_info()),
        "backend": format!("{:?}", runtime.adapter_info().backend),
        "browser_hardware_execution": if cfg!(target_arch="wasm32") {Some("unverified")} else {None},
        "passed": checks.iter().all(|c| c["passed"] == true) &&
            guards.iter().all(|c| c["rejected"] == true && c["recovered"] == true) &&
            !oracle.is_empty() && oracle.iter().all(|c| c["passed"] == true),
        "checks": checks, "guards": guards, "integer_oracle":oracle,
        "scope": "exact synthetic cancellation and rejection/recovery; no throughput or quality claim"
    }))
}
