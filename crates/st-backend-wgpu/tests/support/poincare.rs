//! Frozen independent PyTorch metric reference on native and browser clients.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice, TensorError, INVALID_TENSOR_FLAG},
    resident_training::{parameters::ResidentParameters, TrainingError},
    runtime::WgpuRuntime,
};
use st_kernel_contracts::poincare::{PoincareBiasForward, PoincareBiasSpec};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn data(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
async fn read(t: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = t.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    Ok(values)
}
fn close(a: &[f32], b: &[f32], label: &str) -> Result<f64> {
    if a.len() != b.len() {
        return Err(format!("{label}: length mismatch").into());
    }
    let mut maximum = 0f64;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let delta = (f64::from(a) - f64::from(b)).abs();
        if !a.is_finite() || !b.is_finite() || delta > 3e-6 + 8e-5 * f64::from(b).abs() {
            return Err(format!("{label}[{i}]: {a} != {b}").into());
        }
        maximum = maximum.max(delta);
    }
    Ok(maximum)
}
fn strided(d: &TensorDevice, s: &[usize], v: &[f32]) -> Result<ResidentTensor> {
    let mut shape = s.to_vec();
    shape.push(2);
    Ok(d.upload(
        &shape,
        &v.iter().flat_map(|&v| [0.123, v]).collect::<Vec<_>>(),
    )?
    .select(s.len(), 1)?)
}
async fn rejected(t: &ResidentTensor) -> Result<()> {
    if !matches!(read(t).await,Err(e) if matches!(e.downcast_ref::<TensorError>(),Some(TensorError::NonFinite)))
    {
        return Err("Poincare operation lost its whole-family guard".into());
    }
    Ok(())
}

fn require_update_rejection(status: std::result::Result<u64, TrainingError>) -> Result<()> {
    if !matches!(status, Err(TrainingError::Rejected {stage:0,flags}) if flags & INVALID_TENSOR_FLAG != 0)
    {
        return Err("metric update was accepted or rejected for the wrong reason".into());
    }
    Ok(())
}

async fn analytic(d: &TensorDevice) -> Result<Value> {
    let u = 2f32.powi(-24);
    let x = [
        1. - u,
        2f32.powi(-12) * (1. - u),
        2f32.powi(-12),
        u * (1. - u),
    ];
    let a = f64::from(u).powi(3) * (1. - f64::from(u));
    let distance = 4. * (4. * (1. - a) / (a * a)).sqrt().asinh().powi(2);
    let points: Vec<_> = x.into_iter().chain(x.map(|x| -x)).collect();
    let spec = PoincareBiasSpec::new([1, 2, 4], 1, -1.)?;
    let cpu = PoincareBiasForward::new(spec, &points, &[0.])?;
    let f =
        strided(d, &[1, 2, 4], &points)?.causal_poincare_bias(&strided(d, &[1], &[0.])?, -1.)?;
    let seed = [0., 0., 1., 0.];
    let expected = cpu.backward(&seed)?;
    let g = f.backward(&strided(d, &[1, 1, 2, 2], &seed)?)?;
    let score = read(f.scores()).await?;
    let analytic_score = -2f64.ln() * distance;
    if (f64::from(score[2]) / analytic_score - 1.).abs() > 8e-5 {
        return Err("near-boundary metric mismatch".into());
    }
    close(
        &read(g.coordinates()).await?,
        &expected.coordinates,
        "near-boundary coordinate VJP",
    )?;
    close(
        &read(g.raw_gain()).await?,
        &expected.raw_gain,
        "near-boundary gain VJP",
    )?;
    let mut cases = Vec::new();
    let switch = ((1f64 / 1024.) / (1. + 1. / 1024.)).sqrt() as f32;
    for (c, x, gain, seed, name) in [
        (-1., [0., 1e-25], 0., 1e30, "tiny_separation"),
        (-1e-30, [0., 5e14], -110., 1e30, "tiny_gain_large_seed"),
        (-1., [0., 1e-20], 1e20, 1., "large_gain_tiny_distance"),
        (
            -1.,
            [0.4, f32::from_bits(0.4f32.to_bits() + 1)],
            0.,
            1.,
            "one_ulp_separation",
        ),
        (
            -1.,
            [0., f32::from_bits(switch.to_bits() - 1)],
            0.,
            1.,
            "series_below",
        ),
        (-1., [0., switch], 0., 1., "series_at"),
        (
            -1.,
            [0., f32::from_bits(switch.to_bits() + 1)],
            0.,
            1.,
            "series_above",
        ),
    ] {
        let spec = PoincareBiasSpec::new([1, 2, 1], 1, c)?;
        let cpu = PoincareBiasForward::new(spec, &x, &[gain])?;
        let expected = cpu.backward(&[0., 0., seed, 0.])?;
        let f = strided(d, &[1, 2, 1], &x)?.causal_poincare_bias(&strided(d, &[1], &[gain])?, c)?;
        let g = f.backward(&strided(d, &[1, 1, 2, 2], &[0., 0., seed, 0.])?)?;
        let (gx, gg) = (read(g.coordinates()).await?, read(g.raw_gain()).await?);
        let mut error = 0f64;
        for (a, b) in gx
            .iter()
            .chain(&gg)
            .zip(expected.coordinates.iter().chain(&expected.raw_gain))
        {
            let relative = (f64::from(*a) / f64::from(*b) - 1.).abs();
            if !a.is_finite() || !b.is_finite() || *a == 0. || *b == 0. || relative > 8e-5 {
                return Err(format!("{name}: {a} != {b}").into());
            }
            error = error.max(relative);
        }
        cases.push(json!({"name":name,"relative_error":error,"nonzero":true}));
    }
    Ok(
        json!({"near_boundary":true,"analytic_squared_distance":distance,"large_v":true,"extended_cases":cases}),
    )
}

async fn causality_and_guards(d: &TensorDevice) -> Result<Value> {
    let shape = [2, 3, 2];
    let x = [0.1, 0.2, 0.3, -0.2, 0.4, 0.1, -0.2, 0.1, 0.2, 0.3, 0.1, 0.4];
    let gain = d.upload(&[2], &[-0.4, 0.7])?;
    let f = d.upload(&shape, &x)?.causal_poincare_bias(&gain, -1.)?;
    let saved = read(f.scores()).await?;
    let mut altered = x;
    altered[4] = -0.4;
    altered[5] = -0.3;
    altered[10] = -0.3;
    altered[11] = -0.2;
    let other = d
        .upload(&shape, &altered)?
        .causal_poincare_bias(&gain, -1.)?;
    let changed = read(other.scores()).await?;
    let mut sensitivity = false;
    let mut seed = vec![0.; 36];
    for b in 0..2 {
        for h in 0..2 {
            for q in 0..3 {
                for k in 0..3 {
                    let i = ((b * 2 + h) * 3 + q) * 3 + k;
                    if q < 2 && saved[i] != changed[i] {
                        return Err("suffix affected prefix metric".into());
                    }
                    if q == 2 && (saved[i] - changed[i]).abs() > 1e-5 {
                        sensitivity = true;
                    }
                    if q < 2 || k > q {
                        seed[i] = 0.3;
                    }
                }
            }
        }
    }
    if !sensitivity {
        return Err("metric suffix control inert".into());
    }
    let cot = d.upload(&[2, 2, 3, 3], &seed)?;
    let g = f.backward(&cot)?;
    let retained = read(g.coordinates()).await?;
    for b in 0..2 {
        if retained[b * 6 + 4..b * 6 + 6].iter().any(|&v| v != 0.) {
            return Err("future coordinate gradient".into());
        }
    }
    let huge = d.upload(&[1], &[f32::MAX])?;
    let invalid = huge.mul(&huge)?;
    for operand in 0..3 {
        let points = d.upload(&shape, &x)?;
        let mut args = [points, gain.clone(), cot.clone()];
        args[operand] = d.guard_together(&[&args[operand], &invalid])?.remove(0);
        let bad = args[0].causal_poincare_bias(&args[1], -1.)?;
        if operand < 2 {
            rejected(bad.scores()).await?;
        }
        let bad = bad.backward(&args[2])?;
        rejected(bad.coordinates()).await?;
        rejected(bad.raw_gain()).await?;
    }
    for outside in [1., 1.01] {
        let bad = d
            .upload(&[1, 1, 2], &[outside, 0.])?
            .causal_poincare_bias(&gain, -1.)?;
        rejected(bad.scores()).await?;
    }
    let points = d.upload(&[1, 2, 1], &[0., 0.8])?;
    let raw = d.upload(&[1], &[0.])?;
    let stable = points.causal_poincare_bias(&raw, -1.)?;
    let bad = stable.backward(&d.upload(&[1, 1, 2, 2], &[0., 0., f32::MAX, 0.])?)?;
    rejected(bad.coordinates()).await?;
    rejected(bad.raw_gain()).await?;
    for rate in [0., 0.125] {
        let mut owner = ResidentParameters::new(vec![points.clone(), raw.clone()])?;
        let bound = owner
            .snapshot()
            .bind_gradients(vec![bad.coordinates().clone(), bad.raw_gain().clone()])?;
        let receipt = owner.sgd(&bound, rate)?.snapshot()?;
        #[cfg(target_arch = "wasm32")]
        let status = receipt.read_async().await;
        #[cfg(not(target_arch = "wasm32"))]
        let status = receipt.read();
        require_update_rejection(status)?;
        let snapshot = owner.snapshot();
        if read(&snapshot.values()[0]).await? != [0., 0.8]
            || read(&snapshot.values()[1]).await? != [0.]
        {
            return Err("partial metric parameter update".into());
        }
    }
    for status in [
        Ok(1),
        Err(TrainingError::InvalidReadback),
        Err(TrainingError::Rejected {
            stage: 1,
            flags: INVALID_TENSOR_FLAG,
        }),
        Err(TrainingError::Rejected { stage: 0, flags: 1 }),
    ] {
        if require_update_rejection(status).is_ok() {
            return Err("invalid metric rejection checker".into());
        }
    }
    close(&read(f.scores()).await?, &saved, "retained scores")?;
    close(&read(g.coordinates()).await?, &retained, "retained VJP")?;
    close(
        &read(f.backward(&cot)?.coordinates()).await?,
        &retained,
        "good bad good",
    )?;
    Ok(
        json!({"prefix_invariant":true,"future_gradient_zero":true,"suffix_sensitive":true,"guarded_operands":3,"outside_ball_rejections":2,"retained":true,"good_bad_good":true,"late_gradient_whole_family":true,"atomic_rejections":2,"rejection_negative_controls":4}),
    )
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let adapter = format!("{:?}", runtime.adapter_info());
    let d = TensorDevice::new(runtime)?;
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/poincare_bias_torch.json"))?;
    let mut checks = Vec::new();
    for c in fixture["cases"].as_array().unwrap() {
        let shape: Vec<_> = c["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect();
        let heads = c["heads"].as_u64().unwrap() as usize;
        let curvature = c["curvature"].as_f64().unwrap() as f32;
        let spec = PoincareBiasSpec::new(shape.clone().try_into().unwrap(), heads, curvature)?;
        let x = data(&c["coordinates"]);
        let gain = data(&c["raw_gain"]);
        let seed = data(&c["seed"]);
        let cpu = PoincareBiasForward::new(spec, &x, &gain)?;
        let cpu_g = cpu.backward(&seed)?;
        let f = strided(&d, &shape, &x)?
            .causal_poincare_bias(&strided(&d, &[heads], &gain)?, curvature)?;
        let g = f.backward(&strided(&d, &spec.score_shape(), &seed)?)?;
        if f.scores().layout().shape() != spec.score_shape()
            || g.coordinates().layout().shape() != shape
            || g.raw_gain().layout().shape() != [heads]
        {
            return Err("metric gradient shapes".into());
        }
        let scores = close(&read(f.scores()).await?, &data(&c["scores"]), "score")?;
        let coordinates = close(
            &read(g.coordinates()).await?,
            &data(&c["coordinates_vjp"]),
            "coordinate VJP",
        )?;
        let gain = close(
            &read(g.raw_gain()).await?,
            &data(&c["raw_gain_vjp"]),
            "gain VJP",
        )?;
        let cpu_error = close(cpu.scores(), &data(&c["scores"]), "CPU score")?
            .max(close(
                &cpu_g.coordinates,
                &data(&c["coordinates_vjp"]),
                "CPU coordinate VJP",
            )?)
            .max(close(
                &cpu_g.raw_gain,
                &data(&c["raw_gain_vjp"]),
                "CPU gain VJP",
            )?);
        checks.push(json!({"name":c["name"],"score_max_abs":scores,"coordinates_vjp_max_abs":coordinates,"gain_vjp_max_abs":gain,"cpu_max_abs":cpu_error,"strided":true,"shapes":true}));
    }
    let analytic = analytic(&d).await?;
    let guards = causality_and_guards(&d).await?;
    Ok(
        json!({"schema":"spiraltorch.poincare_bias.validation.v1","adapter":adapter,"passed":true,"checks":checks,"analytic":analytic,"guards":guards,"scope":"Metric primitive correctness, not full-model learning, language quality or throughput"}),
    )
}
