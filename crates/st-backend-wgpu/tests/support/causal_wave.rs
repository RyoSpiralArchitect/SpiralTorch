//! Same immutable Torch oracle and streaming/guard controls on both clients.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice, TensorError, INVALID_TENSOR_FLAG},
    resident_training::{parameters::ResidentParameters, TrainingError},
    runtime::WgpuRuntime,
};
use st_kernel_contracts::causal_wave::{CausalWaveForward, CausalWaveSpec};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn values(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn shape(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}
async fn read(t: &ResidentTensor) -> Result<Vec<f32>> {
    let s = t.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let v = s.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let v = s.read()?;
    Ok(v)
}
fn close(a: &[f32], b: &[f32], label: &str) -> Result<f64> {
    if a.len() != b.len() {
        return Err(format!("{label}: length mismatch").into());
    }
    let mut max = 0f64;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let delta = (f64::from(a) - f64::from(b)).abs();
        if !a.is_finite() || !b.is_finite() || delta > 3e-6 + 8e-5 * f64::from(b).abs() {
            return Err(format!("{label}[{i}]: {a} != {b}").into());
        }
        max = max.max(delta);
    }
    Ok(max)
}
fn strided(d: &TensorDevice, shape: &[usize], values: &[f32]) -> Result<ResidentTensor> {
    let mut storage_shape = shape.to_vec();
    storage_shape.push(2);
    let storage: Vec<_> = values.iter().flat_map(|&v| [0.321, v]).collect();
    Ok(d.upload(&storage_shape, &storage)?.select(shape.len(), 1)?)
}
async fn rejected(t: &ResidentTensor) -> Result<()> {
    if !matches!(read(t).await,Err(e) if matches!(e.downcast_ref::<TensorError>(),Some(TensorError::NonFinite)))
    {
        return Err("wave lost a whole-operation guard".into());
    }
    Ok(())
}
fn require_rejection(status: std::result::Result<u64, TrainingError>) -> Result<()> {
    if !matches!(status,Err(TrainingError::Rejected{stage:0,flags}) if flags & INVALID_TENSOR_FLAG != 0)
    {
        return Err("wave update rejected for wrong reason or was accepted".into());
    }
    Ok(())
}

async fn streaming(d: &TensorDevice, c: &Value) -> Result<Value> {
    let s = shape(&c["shape"]);
    let (b, t, cols) = (s[0], s[1], s[2]);
    let cut = 2;
    let curvature = c["curvature"].as_f64().unwrap() as f32;
    let x = strided(d, &s, &values(&c["drive"]))?;
    let decay = strided(d, &[cols / 2], &values(&c["raw_decay"]))?;
    let phase = strided(d, &[cols / 2], &values(&c["raw_phase"]))?;
    let initial = strided(d, &[b, cols], &values(&c["initial_state"]))?;
    let seed = strided(d, &s, &values(&c["feature_seed"]))?;
    let end = strided(d, &[b, cols], &values(&c["terminal_seed"]))?;
    let full = x.causal_zspace_wave(&decay, &phase, &initial, curvature)?;
    let first = x
        .narrow(1, 0, cut)?
        .causal_zspace_wave(&decay, &phase, &initial, curvature)?;
    let last = x.narrow(1, cut, t - cut)?.causal_zspace_wave(
        &decay,
        &phase,
        first.final_state(),
        curvature,
    )?;
    let joined = ResidentTensor::concatenate(&[first.features(), last.features()], 1)?;
    let feature_error = close(
        &read(&joined).await?,
        &read(full.features()).await?,
        "chunk features",
    )?;
    let state_error = close(
        &read(last.final_state()).await?,
        &read(full.final_state()).await?,
        "chunk final state",
    )?;
    let gf = full.backward(&seed, &end)?;
    let gr = last.backward(&seed.narrow(1, cut, t - cut)?, &end)?;
    let gl = first.backward(&seed.narrow(1, 0, cut)?, gr.initial_state())?;
    let gx = ResidentTensor::concatenate(&[gl.drive(), gr.drive()], 1)?;
    let gd = gl.raw_decay().add(gr.raw_decay())?;
    let gp = gl.raw_phase().add(gr.raw_phase())?;
    let mut vjp_error = 0f64;
    for (a, b) in [
        (&gx, gf.drive()),
        (&gd, gf.raw_decay()),
        (&gp, gf.raw_phase()),
        (gl.initial_state(), gf.initial_state()),
    ] {
        vjp_error = vjp_error.max(close(&read(a).await?, &read(b).await?, "chunk BPTT")?);
    }
    let mut changed = values(&c["drive"]);
    for batch in 0..b {
        changed[(batch * t + cut) * cols..(batch + 1) * t * cols].fill(0.8);
    }
    let altered = d
        .upload(&s, &changed)?
        .causal_zspace_wave(&decay, &phase, &initial, curvature)?;
    let prefix_error = close(
        &read(&altered.features().narrow(1, 0, cut)?).await?,
        &read(first.features()).await?,
        "causal prefix",
    )?;
    if !read(altered.final_state())
        .await?
        .iter()
        .zip(read(full.final_state()).await?)
        .any(|(a, b)| (a - b).abs() > 1e-6)
    {
        return Err("suffix sensitivity control is inert".into());
    }
    let mut prefix_seed = values(&c["feature_seed"]);
    for batch in 0..b {
        prefix_seed[(batch * t + cut) * cols..(batch + 1) * t * cols].fill(0.);
    }
    let g = full.backward(
        &d.upload(&s, &prefix_seed)?,
        &d.upload(&[b, cols], &vec![0.; b * cols])?,
    )?;
    if read(&g.drive().narrow(1, cut, t - cut)?)
        .await?
        .iter()
        .any(|&v| v != 0.)
    {
        return Err("future drive has a prefix gradient".into());
    }
    close(
        &read(full.features()).await?,
        &values(&c["features"]),
        "retained forward after changed input",
    )?;
    Ok(
        json!({"feature_max_abs_error":feature_error,"state_max_abs_error":state_error,"vjp_max_abs_error":vjp_error,"prefix_max_abs_error":prefix_error,"future_gradient_zero":true,"sensitivity_control":true,"retained_output":true}),
    )
}

async fn guards(d: &TensorDevice) -> Result<Value> {
    let raw = d.upload(&[1], &[0.])?;
    let x = d.upload(&[1, 2, 2], &[0.2; 4])?;
    let state = d.upload(&[1, 2], &[0.1; 2])?;
    let huge = d.upload(&[1], &[f32::MAX])?;
    let invalid = huge.mul(&huge)?;
    let inputs = [x.clone(), raw.clone(), raw.clone(), state.clone()];
    for i in 0..4 {
        let mut args = inputs.clone();
        args[i] = d.guard_together(&[&args[i], &invalid])?.remove(0);
        let f = args[0].causal_zspace_wave(&args[1], &args[2], &args[3], -1.)?;
        rejected(f.features()).await?;
        rejected(f.final_state()).await?;
    }
    let f = x.causal_zspace_wave(&raw, &raw, &state, -1.)?;
    // A symmetric seed is orthogonal to the rotational derivative here.
    let seed = d.upload(&[1, 2, 2], &[0.3, -0.1, 0.2, 0.4])?;
    let end = d.upload(&[1, 2], &[0.2, -0.3])?;
    let g = f.backward(&seed, &end)?;
    let saved = read(g.raw_phase()).await?;
    for i in 0..2 {
        let mut seeds = [seed.clone(), end.clone()];
        seeds[i] = d.guard_together(&[&seeds[i], &invalid])?.remove(0);
        let bad = f.backward(&seeds[0], &seeds[1])?;
        for t in [
            bad.drive(),
            bad.raw_decay(),
            bad.raw_phase(),
            bad.initial_state(),
        ] {
            rejected(t).await?;
        }
        if saved != read(g.raw_phase()).await? {
            return Err("failed backward changed retained gradient".into());
        }
    }
    let repeat = f.backward(&seed, &end)?;
    close(&read(repeat.raw_phase()).await?, &saved, "good bad good")?;
    let other = f.backward(
        &d.upload(&[1, 2, 2], &[-0.3, 0.1, -0.2, -0.4])?,
        &d.upload(&[1, 2], &[-0.2, 0.3])?,
    )?;
    if !read(other.raw_phase())
        .await?
        .iter()
        .zip(&saved)
        .any(|(a, b)| (a - b).abs() > 1e-6)
    {
        return Err("changed cotangent control is inert".into());
    }
    if read(g.raw_phase()).await? != saved {
        return Err("changed cotangent mutated retained gradient".into());
    }
    let init = d.upload(&[16, 2], &[1., 0.].repeat(16))?;
    let big = d
        .upload(&[16, 1, 2], &[0.; 32])?
        .causal_zspace_wave(&raw, &raw, &init, -1.)?;
    let bad = big.backward(
        &d.upload(&[16, 1, 2], &[0.; 32])?,
        &d.upload(&[16, 2], &[1e38, 0.].repeat(16))?,
    )?;
    for t in [
        bad.drive(),
        bad.raw_decay(),
        bad.raw_phase(),
        bad.initial_state(),
    ] {
        rejected(t).await?;
    }
    let mut rejected_steps = 0;
    for rate in [0., 0.125] {
        let mut owner = ResidentParameters::new(vec![raw.clone(), raw.clone()])?;
        let before = owner.snapshot();
        let bound =
            before.bind_gradients(vec![bad.raw_decay().clone(), bad.raw_phase().clone()])?;
        let receipt = owner.sgd(&bound, rate)?.snapshot()?;
        #[cfg(target_arch = "wasm32")]
        let status = receipt.read_async().await;
        #[cfg(not(target_arch = "wasm32"))]
        let status = receipt.read();
        require_rejection(status)?;
        for value in owner.snapshot().values() {
            if read(value).await? != [0.] {
                return Err("partial wave parameter update".into());
            }
        }
        rejected_steps += 1;
    }
    let large = d.upload(&[1, 1, 2], &[1e30, -1e30])?.causal_zspace_wave(
        &raw,
        &raw,
        &d.upload(&[1, 2], &[0.; 2])?,
        -1.,
    )?;
    let norm = read(large.features())
        .await?
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    if !(0.949..0.951).contains(&norm) {
        return Err("large finite chart is outside its interior radius".into());
    }
    let mut negatives = 0;
    for status in [
        Ok(1),
        Err(TrainingError::InvalidReadback),
        Err(TrainingError::Rejected {
            stage: 1,
            flags: INVALID_TENSOR_FLAG,
        }),
        Err(TrainingError::Rejected { stage: 0, flags: 1 }),
    ] {
        if require_rejection(status).is_ok() {
            return Err("invalid rejection validator".into());
        }
        negatives += 1;
    }
    Ok(
        json!({"forward_operand_guards":4,"cotangent_guards":2,"good_bad_good":true,"retained_gradients":true,"late_reduction_whole_family":true,"atomic_rejections":rejected_steps,"rejection_negative_controls":negatives,"large_state_chart":true}),
    )
}

async fn train(d: &TensorDevice, fixture: &Value) -> Result<Value> {
    let learning = &fixture["learning"];
    let c = &fixture["cases"][2];
    let s = shape(&c["shape"]);
    let state_shape = [s[0], s[2]];
    if learning["steps"] != 16 || learning["trace"].as_array().map(Vec::len) != Some(16) {
        return Err("incomplete learning oracle".into());
    }
    let x = d.upload(&s, &values(&c["drive"]))?;
    let initial = d.upload(&state_shape, &values(&c["initial_state"]))?;
    let target = d.upload(&s, &values(&learning["target"]))?;
    let end_target = d.upload(&state_shape, &values(&learning["terminal_target"]))?;
    let mut owner = ResidentParameters::new(vec![
        d.upload(&[s[2] / 2], &values(&c["raw_decay"]))?,
        d.upload(&[s[2] / 2], &values(&c["raw_phase"]))?,
    ])?;
    let mut pending = Vec::new();
    for _ in 0..16 {
        let snapshot = owner.snapshot();
        let p = snapshot.values();
        let f = x.causal_zspace_wave(
            &p[0],
            &p[1],
            &initial,
            c["curvature"].as_f64().unwrap() as f32,
        )?;
        let lf = f.features().mean_squared_error(&target)?;
        let ls = f.final_state().mean_squared_error(&end_target)?;
        let vjp = f.backward(lf.prediction_gradient(), ls.prediction_gradient())?;
        let gradients =
            snapshot.bind_gradients(vec![vjp.raw_decay().clone(), vjp.raw_phase().clone()])?;
        let update = owner.sgd(&gradients, learning["rate"].as_f64().unwrap() as f32)?;
        pending.push((lf, ls, vjp, update, owner.snapshot()));
    }
    let mut trace = Vec::new();
    for (step, (lf, ls, g, update, p)) in pending.into_iter().enumerate() {
        let e = &learning["trace"][step];
        let receipt = update.snapshot()?;
        #[cfg(target_arch = "wasm32")]
        let revision = receipt.read_async().await?;
        #[cfg(not(target_arch = "wasm32"))]
        let revision = receipt.read()?;
        if revision != step as u64 + 1 {
            return Err("wave owner revision mismatch".into());
        }
        let fl = read(lf.value()).await?;
        let sl = read(ls.value()).await?;
        close(
            &fl,
            &[e["feature_loss"].as_f64().unwrap() as f32],
            "feature MSE",
        )?;
        close(
            &sl,
            &[e["terminal_loss"].as_f64().unwrap() as f32],
            "terminal MSE",
        )?;
        let mut max = 0f64;
        for (t, key) in [
            (g.raw_decay(), "raw_decay_gradient"),
            (g.raw_phase(), "raw_phase_gradient"),
            (&p.values()[0], "raw_decay"),
            (&p.values()[1], "raw_phase"),
        ] {
            max = max.max(close(&read(t).await?, &values(&e[key]), key)?);
        }
        trace.push(json!({"revision":revision,"feature_loss":fl[0],"terminal_loss":sl[0],"parameter_and_gradient_max_abs_error":max}));
    }
    Ok(json!({"steps":16,"trace":trace,"observation_after_all_updates":true}))
}

async fn radial_controls(d: &TensorDevice) -> Result<Value> {
    let k = 134_217_728f32;
    let oblique_seed = [10000. * k, 3. * k];
    let mut up = oblique_seed;
    up[1] = f32::from_bits(up[1].to_bits() + 1);
    let mut down = oblique_seed;
    down[1] = f32::from_bits(down[1].to_bits() - 1);
    let mut checks = Vec::new();
    for (name, x, initial, seed) in [
        ("axis", [16384., 0.], [0.; 2], [1e8, 0.]),
        (
            "diagonal",
            [10000., 10000.],
            [10000., 10000.],
            [10000. * k; 2],
        ),
        ("oblique", [10000., 3.], [10000., 3.], oblique_seed),
        ("oblique_up", [10000., 3.], [10000., 3.], up),
        ("oblique_down", [10000., 3.], [10000., 3.], down),
        (
            "oblique_negative",
            [10000., -3.],
            [10000., -3.],
            [10000. * k, -3. * k],
        ),
        ("large_exponent", [1e20, 0.], [1e20, 0.], [1e38, 0.]),
        (
            "oblique_phase",
            [10000., 6000.],
            [10000., 6000.],
            [10000. * k, 6000. * k],
        ),
        ("mixed_exponents", [10000., 0.], [10000., 0.], [1e38, 1e-10]),
    ] {
        let curvature = if name == "mixed_exponents" {
            -1e-20f32
        } else {
            -1.
        };
        let radius = 0.95f32 / (-curvature).sqrt();
        let rho = 0.99f32 * 0.5;
        let s = x
            .iter()
            .zip(initial)
            .map(|(&x, p)| f64::from(rho * p + (1. - rho) * x))
            .collect::<Vec<_>>();
        let g = seed.map(f64::from);
        // Independent f64 pivot identity. Products of two f32 inputs fit its
        // significand; no small residual is snapped to zero.
        let residual = [0., (g[1] * s[0] - g[0] * s[1]) / s[0]];
        let q = 1. + s[0] * s[0] + s[1] * s[1];
        let root = q.sqrt();
        let dot = s[1] * residual[1];
        let chart: Vec<_> = (0..2)
            .map(|i| {
                f64::from(radius) / root * (residual[i] - s[i] * dot / q + (g[0] / s[0]) * s[i] / q)
            })
            .collect();
        let expected_drive: Vec<_> = chart
            .iter()
            .map(|v| (v * f64::from(1. - rho)) as f32)
            .collect();
        let expected_initial: Vec<_> = chart.iter().map(|v| (v * f64::from(rho)) as f32).collect();
        let expected_decay = (chart
            .iter()
            .enumerate()
            .map(|(i, g)| g * f64::from(initial[i] - x[i]))
            .sum::<f64>()
            * f64::from(0.99f32 * 0.5 * 0.5)) as f32;
        let expected_phase = ((chart[1] * f64::from(initial[0]) - chart[0] * f64::from(initial[1]))
            * f64::from(rho)
            * f64::from(std::f32::consts::PI)) as f32;
        let expected = [
            expected_drive,
            vec![expected_decay],
            vec![expected_phase],
            expected_initial,
        ];
        let spec = CausalWaveSpec::new([1, 1, 2], curvature)?;
        let cpu = CausalWaveForward::new(spec, &x, &[0.], &[0.], &initial)?;
        let cg = cpu.backward(&seed, &[0.; 2])?;
        let raw = d.upload(&[1], &[0.])?;
        let f = d.upload(&[1, 1, 2], &x)?.causal_zspace_wave(
            &raw,
            &raw,
            &d.upload(&[1, 2], &initial)?,
            curvature,
        )?;
        let vjp = f.backward(&d.upload(&[1, 1, 2], &seed)?, &d.upload(&[1, 2], &[0.; 2])?)?;
        let mut max = 0f64;
        for (i, (gpu, cpu)) in [
            (vjp.drive(), &cg.drive),
            (vjp.raw_decay(), &cg.raw_decay),
            (vjp.raw_phase(), &cg.raw_phase),
            (vjp.initial_state(), &cg.initial_state),
        ]
        .into_iter()
        .enumerate()
        {
            let gpu = read(gpu).await?;
            max = max
                .max(close(&gpu, &expected[i], name)?)
                .max(close(cpu, &expected[i], name)?);
            if name == "large_exponent" && (i == 0 || i == 3) {
                for actual in [gpu[0], cpu[0]] {
                    let error = (f64::from(actual) - f64::from(expected[i][0])).abs();
                    if actual <= 0. || error > 8e-5 * f64::from(expected[i][0]).abs() {
                        return Err("representable radial derivative underflowed".into());
                    }
                }
            }
        }
        if ["oblique_up", "oblique_down", "mixed_exponents"].contains(&name)
            && expected_phase.abs() < 1.
        {
            return Err("one-ULP cotangent control is inert".into());
        }
        checks.push(
            json!({"name":name,"all_four_vjps":true,"cpu_and_gpu_max_abs_error":max,
            "expected_phase_gradient":expected_phase}),
        );
    }
    Ok(
        json!({"cases":checks,"f64_analytic_reference":true,"one_ulp_controls":2,"large_exponent_nonzero":true}),
    )
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/causal_zspace_wave_torch.json"))?;
    if fixture["schema"] != "spiraltorch.causal_zspace_wave.torch_fixture.v1"
        || fixture["cases"].as_array().map(Vec::len) != Some(10)
    {
        return Err("incomplete causal wave fixture".into());
    }
    let device = TensorDevice::new(runtime.clone())?;
    let mut checks = Vec::new();
    for c in fixture["cases"].as_array().unwrap() {
        let s = shape(&c["shape"]);
        let spec = CausalWaveSpec::new(
            s.clone().try_into().unwrap(),
            c["curvature"].as_f64().unwrap() as f32,
        )?;
        let x = values(&c["drive"]);
        let d = values(&c["raw_decay"]);
        let p = values(&c["raw_phase"]);
        let init = values(&c["initial_state"]);
        let fs = values(&c["feature_seed"]);
        let ts = values(&c["terminal_seed"]);
        let cpu = CausalWaveForward::new(spec, &x, &d, &p, &init)?;
        let cg = cpu.backward(&fs, &ts)?;
        let drive = strided(&device, &s, &x)?;
        let decay = strided(&device, &[s[2] / 2], &d)?;
        let phase = strided(&device, &[s[2] / 2], &p)?;
        let state = strided(&device, &[s[0], s[2]], &init)?;
        let f = drive.causal_zspace_wave(&decay, &phase, &state, spec.curvature())?;
        let g = f.backward(
            &strided(&device, &s, &fs)?,
            &strided(&device, &[s[0], s[2]], &ts)?,
        )?;
        let mut errors = serde_json::Map::new();
        let mut cpu_error = 0f64;
        for (tensor, expected, host, expected_shape, key) in [
            (
                f.features(),
                &c["features"],
                cpu.features(),
                s.clone(),
                "features",
            ),
            (
                f.final_state(),
                &c["final_state"],
                cpu.final_state(),
                vec![s[0], s[2]],
                "final_state",
            ),
            (
                g.drive(),
                &c["gradients"]["drive"],
                cg.drive.as_slice(),
                s.clone(),
                "drive_vjp",
            ),
            (
                g.raw_decay(),
                &c["gradients"]["raw_decay"],
                cg.raw_decay.as_slice(),
                vec![s[2] / 2],
                "raw_decay_vjp",
            ),
            (
                g.raw_phase(),
                &c["gradients"]["raw_phase"],
                cg.raw_phase.as_slice(),
                vec![s[2] / 2],
                "raw_phase_vjp",
            ),
            (
                g.initial_state(),
                &c["gradients"]["initial_state"],
                cg.initial_state.as_slice(),
                vec![s[0], s[2]],
                "initial_state_vjp",
            ),
        ] {
            if tensor.layout().shape() != expected_shape {
                return Err("wave result logical shape mismatch".into());
            }
            let expected = values(expected);
            errors.insert(
                key.into(),
                json!(close(&read(tensor).await?, &expected, key)?),
            );
            cpu_error = cpu_error.max(close(host, &expected, key)?);
        }
        checks.push(json!({"name":c["name"],"errors":errors,"cpu_max_abs_error":cpu_error,"logical_shapes":true,"all_inputs_and_cotangents_strided":true}));
    }
    Ok(
        json!({"schema":"spiraltorch.causal_zspace_wave.validation.v1","passed":true,"adapter":format!("{:?}",runtime.adapter_info()),
        "checks":checks,"streaming":streaming(&device,&fixture["cases"][2]).await?,"guards":guards(&device).await?,"learning":train(&device,&fixture).await?,"radial":radial_controls(&device).await?,
        "scope":"causal state/chart primitive correctness and parameter learning; not full decoder streaming, geometric attention or language quality"}),
    )
}
