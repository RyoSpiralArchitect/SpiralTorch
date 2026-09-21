use serde_json::{json, Value};
use st_backend_wgpu::{
    nerf::{NerfError, NerfRay, RaySampling, ResidentNerf},
    resident_graph::ResidentGraph,
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{ResidentTensor, TensorDevice, TensorError},
    runtime::WgpuRuntime,
};
use st_kernel_contracts::{
    graph::{GraphDefinition, GraphParameter, GraphStage, ParameterRole},
    layout::NdLayout,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WEIGHTS: [f32; 12] = [
    0.2, 0.1, -0.05, 0.3, -0.1, 0.25, 0.3, 0.02, 0.4, -0.2, 0.1, 0.15,
];
const BIAS: [f32; 4] = [0.7, 0.2, 0.4, -0.1];

async fn read(tensor: &ResidentTensor) -> std::result::Result<Vec<f32>, TensorError> {
    let snapshot = tensor.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        snapshot.read()
    }
    #[cfg(target_arch = "wasm32")]
    {
        snapshot.read_async().await
    }
}

fn offset(ray: u32, sample: u32, mode: RaySampling) -> f64 {
    let RaySampling::Stratified { seed } = mode else {
        return 0.5;
    };
    let mut bits = seed
        ^ ray
            .wrapping_mul(0x9e3779b9)
            .wrapping_add(sample.wrapping_mul(0x85ebca6b));
    bits = (bits ^ (bits >> 16)).wrapping_mul(0x7feb352d);
    bits = (bits ^ (bits >> 15)).wrapping_mul(0x846ca68b);
    bits ^= bits >> 16;
    f64::from(bits >> 8) / 16777216.
}

fn reference(rays: &[NerfRay], samples: usize, varying: bool, mode: RaySampling) -> Vec<f64> {
    let mut result = Vec::new();
    for (index, ray) in rays.iter().enumerate() {
        let width = (f64::from(ray.far) - f64::from(ray.near)) / samples as f64;
        let mut rgba = [0f64; 4];
        let mut trans = 1f64;
        for i in 0..samples {
            let t = f64::from(ray.near) + (i as f64 + offset(index as u32, i as u32, mode)) * width;
            let mut value = BIAS;
            if varying {
                for d in 0..3 {
                    let x = (f64::from(ray.origin[d]) + f64::from(ray.direction[d]) * t) as f32;
                    for c in 0..4 {
                        value[c] += x * WEIGHTS[d * 4 + c];
                    }
                }
            }
            let tau = f64::from(value[0].max(0.)) * width;
            let weight = trans * -(-tau).exp_m1();
            for c in 0..3 {
                rgba[c] += weight * f64::from(value[c + 1]);
            }
            rgba[3] += weight;
            trans *= (-tau).exp();
        }
        result.extend(rgba);
    }
    result
}

fn compare(actual: &[f32], reference: &[f64]) -> Result<f64> {
    if actual.len() != reference.len() {
        return Err("NeRF output length mismatch".into());
    }
    let mut maximum = 0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let error = (f64::from(actual) - reference).abs();
        if !actual.is_finite() || error > 4e-7 + 4e-6 * reference.abs() {
            return Err(format!("NeRF mismatch {actual} vs {reference}").into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("software adapter is not real-GPU NeRF coverage".into());
    }
    let adapter = format!("{:?}", runtime.adapter_info());
    let mut cases = Vec::new();
    for rows in [1, 65, 256] {
        for count in [1, 8, 64] {
            for varying in [false, true] {
                for mode in [RaySampling::Midpoint, RaySampling::Stratified { seed: 17 }] {
                    let definition = GraphDefinition::new(
                        NdLayout::contiguous(&[rows, count, 3])?,
                        vec![GraphStage::Linear {
                            weight: 0,
                            bias: 1,
                            gelu: false,
                        }],
                        vec![
                            GraphParameter {
                                role: ParameterRole::Weight,
                                shape: vec![3, 4],
                                values: if varying {
                                    WEIGHTS.to_vec()
                                } else {
                                    vec![0.; 12]
                                },
                            },
                            GraphParameter {
                                role: ParameterRole::Bias,
                                shape: vec![4],
                                values: BIAS.to_vec(),
                            },
                        ],
                    )?;
                    let mut graph = ResidentGraph::new(
                        runtime.clone(),
                        definition,
                        MatmulTile::default(),
                        MatmulKernel::Scalar,
                        MatmulAccumulation::Sequential,
                    )?;
                    let nerf = ResidentNerf::new(graph.tensor_device().clone())?;
                    let rays: Vec<_> = (0..rows)
                        .map(|i| NerfRay {
                            origin: [0.1 + i as f32 / 1024., -0.2, 0.3],
                            direction: [0.4, 0.5, -0.6],
                            near: -0.25,
                            far: 1. + (i % 7) as f32 * 0.125,
                        })
                        .collect();
                    let resident = nerf.upload_rays(&rays)?;
                    let samples = nerf.sample(&resident, count, mode)?;
                    graph.set_input_tensor(&samples.positions()?)?;
                    graph.dispatch()?;
                    let rgba = nerf.composite(&samples, &graph.output_tensor()?)?;
                    let expected = reference(&rays, count, varying, mode);
                    // First and only observation in this complete ray/NN/render chain.
                    let actual = read(&rgba).await?;
                    let maximum = compare(&actual, &expected)?;
                    cases.push(json!({"rays":rows,"samples":count,"varying":varying,
                        "mode":format!("{mode:?}"),"max_abs_error":maximum,
                        "seed":match mode { RaySampling::Midpoint => None, RaySampling::Stratified { seed } => Some(seed) },
                        "ray_inputs":rays.iter().map(|r| [r.origin[0],r.origin[1],r.origin[2],r.direction[0],r.direction[1],r.direction[2],r.near,r.far]).collect::<Vec<_>>(),
                        "rgba":actual,"reference":expected}));
                }
            }
        }
    }
    let device = TensorDevice::new(runtime)?;
    let nerf = ResidentNerf::new(device.clone())?;
    let ray = NerfRay {
        origin: [0.; 3],
        direction: [0., 0., 1.],
        near: 0.,
        far: 1.,
    };
    let rays = nerf.upload_rays(&[ray])?;
    let samples = nerf.sample(&rays, 2, RaySampling::Midpoint)?;
    let shape_rejected = matches!(
        nerf.composite(&samples, &device.upload(&[2, 4], &[1.; 8])?),
        Err(NerfError::FieldShape)
    );
    let zero_rejected = matches!(
        nerf.sample(&rays, 0, RaySampling::Midpoint),
        Err(NerfError::Empty)
    );
    let tiny = nerf.upload_rays(&[NerfRay {
        far: f32::MIN_POSITIVE,
        ..ray
    }])?;
    let subnormal_rejected = matches!(
        nerf.sample(&tiny, 2, RaySampling::Midpoint),
        Err(NerfError::WidthUnderflow)
    );
    let bad = device
        .upload(&[1, 2, 4], &[-f32::MAX; 8])?
        .mul(&device.upload(&[], &[2.])?)?
        .relu()?;
    let inherited_rejected = matches!(
        read(&nerf.composite(&samples, &bad)?).await,
        Err(TensorError::NonFinite)
    );
    let valid = device.upload(&[1, 2, 4], &[1.; 8])?;
    let old = nerf.composite(&samples, &valid)?;
    let replacement = nerf.sample(&rays, 2, RaySampling::Stratified { seed: 99 })?;
    let _ = nerf.composite(&replacement, &valid)?;
    let original = read(&old).await?;
    let retained_version = compare(&original, &[-(-1f64).exp_m1(); 4]).is_ok();
    let guards = json!({"shape_rejected":shape_rejected,"zero_rejected":zero_rejected,
        "subnormal_width_rejected":subnormal_rejected,"inherited_error_rejected":inherited_rejected,
        "retained_version":retained_version});
    if !guards
        .as_object()
        .unwrap()
        .values()
        .all(|v| v == &Value::Bool(true))
    {
        return Err(format!("NeRF guard failure: {guards}").into());
    }
    Ok(
        json!({"status":"passed","adapter":adapter,"cases":cases,"guards":guards,"weights":WEIGHTS,"bias":BIAS,
        "boundary":"Real resident sample -> NN graph -> composite, one terminal observation; correctness fixture, not a performance or training benchmark"}),
    )
}
