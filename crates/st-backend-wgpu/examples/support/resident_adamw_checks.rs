use serde_json::Value;
use st_backend_wgpu::{
    resident_tensor::TensorDevice,
    resident_training::{
        parameters::{ResidentParameterUpdate, ResidentParameters},
        TrainingError,
    },
};
use st_kernel_contracts::{
    adamw::{AdamWConfig, AdamWState, AdamWStep},
    elementwise::ElementwiseOp,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn close(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len()
        || a.iter()
            .zip(b)
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 3e-6 + 5e-5 * b.abs())
    {
        return Err(format!("AdamW Torch mismatch: {a:?} versus {b:?}").into());
    }
    Ok(())
}
async fn receipt(u: &ResidentParameterUpdate) -> std::result::Result<u64, TrainingError> {
    let snapshot = u.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    return snapshot.read();
    #[cfg(target_arch = "wasm32")]
    return snapshot.read_async().await;
}
async fn state(owner: &ResidentParameters) -> Result<AdamWState> {
    let snapshot = owner.adamw_snapshot()?.ok_or("missing optimizer")?;
    #[cfg(not(target_arch = "wasm32"))]
    let s = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let s = snapshot.read_async().await?;
    Ok(s)
}
async fn values(owner: &ResidentParameters) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::new();
    for p in owner.snapshot().values() {
        let snap = p.snapshot()?;
        #[cfg(not(target_arch = "wasm32"))]
        let p = snap.read()?;
        #[cfg(target_arch = "wasm32")]
        let p = snap.read_async().await?;
        out.push(p);
    }
    Ok(out)
}
fn bits(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter()
        .map(|x| x.iter().map(|x| x.to_bits()).collect())
        .collect()
}
fn state_bits(s: &AdamWState) -> Vec<Vec<u32>> {
    s.slots
        .iter()
        .map(|s| {
            s.first
                .iter()
                .chain(&s.second)
                .map(|x| x.to_bits())
                .chain(std::iter::once(s.steps))
                .collect()
        })
        .collect()
}

pub async fn run(device: &TensorDevice) -> Result<Value> {
    scalar_resume(device).await?;
    let f: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/resident_adamw_torch.json"
    )))?;
    let config = floats(&f["config"]);
    let config = AdamWConfig::new(config[0], config[1], config[2])?;
    let shapes: Vec<Vec<usize>> = f["shapes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            s.as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    let initial = f["initial"].as_array().unwrap();
    let upload = |data: &Vec<Value>| -> Result<Vec<_>> {
        data.iter()
            .zip(&shapes)
            .map(|(v, s)| Ok(device.upload(s, &floats(v))?))
            .collect()
    };
    let mut owner = ResidentParameters::new(upload(initial)?)?;
    owner.enable_adamw(config)?;
    assert!(owner.enable_adamw(config).is_err());
    let retained = owner.adamw_snapshot()?.unwrap();
    let mut resumed = None;
    for (i, step) in f["trajectory"].as_array().unwrap().iter().enumerate() {
        let rates = floats(&step["rates"]);
        let decay = floats(&step["decay"]);
        let policy = rates
            .iter()
            .zip(decay)
            .map(|(&r, d)| AdamWStep::new(r, d))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let g = upload(step["gradients"].as_array().unwrap())?;
        for current in std::iter::once(&mut owner).chain(resumed.iter_mut()) {
            let before_values = bits(&values(current).await?);
            let before_state = state_bits(&state(current).await?);
            let gradient = current.snapshot().bind_gradients(g.clone())?;
            assert!(current.sgd(&gradient, 0.01).is_err());
            assert!(current.adamw_with_steps(&gradient, &[]).is_err());
            assert_eq!(
                receipt(&current.adamw_with_steps(&gradient, &policy)?).await?,
                (i + 1) as u64
            );
            assert!(matches!(
                current.adamw_with_steps(&gradient, &policy),
                Err(TrainingError::ParameterVersion)
            ));
            let s = state(current).await?;
            let actual_values = values(current).await?;
            for (j, actual) in actual_values.iter().enumerate() {
                if rates[j] == 0.0 {
                    assert_eq!(bits(&actual_values)[j], before_values[j]);
                    assert_eq!(state_bits(&s)[j], before_state[j]);
                }
                close(actual, &floats(&step["values"][j]))?;
                close(&s.slots[j].first, &floats(&step["slots"][j]["first"]))?;
                close(&s.slots[j].second, &floats(&step["slots"][j]["second"]))?;
                assert_eq!(
                    u64::from(s.slots[j].steps),
                    step["slots"][j]["steps"].as_u64().unwrap()
                );
            }
        }
        if let Some(resumed) = &resumed {
            assert_eq!(bits(&values(&owner).await?), bits(&values(resumed).await?));
            assert_eq!(
                state_bits(&state(&owner).await?),
                state_bits(&state(resumed).await?)
            );
        }
        if i == 5 {
            let snapshot = owner.snapshot();
            let mut other = ResidentParameters::from_restored_values(
                snapshot.values().to_vec(),
                snapshot.revision(),
            )?;
            other.restore_adamw(&state(&owner).await?)?;
            resumed = Some(other);
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    let retained = retained.read()?;
    #[cfg(target_arch = "wasm32")]
    let retained = retained.read_async().await?;
    assert!(retained
        .slots
        .iter()
        .all(|s| s.steps == 0 && s.first.iter().chain(&s.second).all(|x| x.to_bits() == 0)));
    // Reject a late optimizer overflow, or an inherited invalid frozen gradient.
    for frozen in [false, true] {
        let snapshot = owner.snapshot();
        let before = bits(&values(&owner).await?);
        let history = state_bits(&state(&owner).await?);
        let mut gs = snapshot
            .values()
            .iter()
            .map(|p| device.upload(p.layout().shape(), &vec![0.25; p.layout().len()]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        gs[2] = if frozen {
            device
                .upload(&[1], &[1.0])?
                .apply(ElementwiseOp::Divide, Some(&device.upload(&[1], &[0.0])?))?
        } else {
            device.upload(&[1], &[f32::from_bits(0x5f800000)])?
        };
        let gradient = snapshot.bind_gradients(gs)?;
        let mut policy = vec![AdamWStep::new(0.01, 0.1)?; shapes.len()];
        if frozen {
            policy[2] = AdamWStep::new(0.0, 0.0)?;
        }
        assert!(matches!(
            receipt(&owner.adamw_with_steps(&gradient, &policy)?).await,
            Err(TrainingError::Rejected { .. })
        ));
        assert_eq!(bits(&values(&owner).await?), before);
        assert_eq!(state_bits(&state(&owner).await?), history);
    }
    // Neither the overflow guard nor the update may divide outside WGSL's
    // guaranteed domain, even when the controls themselves are finite/normal.
    for (epsilon, rate, decay) in [
        (1e-8, 2.0, f32::from_bits(0x7f000000)),
        (f32::from_bits(0x7f000000), 0.01, 0.0),
    ] {
        let mut extreme = ResidentParameters::from_restored_values(
            owner.snapshot().values().to_vec(),
            owner.snapshot().revision(),
        )?;
        let mut history = state(&owner).await?;
        history.config = AdamWConfig::new(0.9, 0.999, epsilon)?;
        extreme.restore_adamw(&history)?;
        let before = bits(&values(&extreme).await?);
        let gradient = extreme.snapshot().bind_gradients(
            extreme
                .snapshot()
                .values()
                .iter()
                .map(|p| device.upload(p.layout().shape(), &vec![0.25; p.layout().len()]))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )?;
        assert!(matches!(
            receipt(&extreme.adamw(&gradient, AdamWStep::new(rate, decay)?)?).await,
            Err(TrainingError::Rejected { .. })
        ));
        assert_eq!(bits(&values(&extreme).await?), before);
        assert_eq!(state_bits(&state(&extreme).await?), state_bits(&history));
    }
    // Even division by one needs quotient headroom for WGSL's allowed error.
    let mut large_moment = state(&owner).await?;
    large_moment.config = AdamWConfig::new(f32::from_bits(0x3f7fffff), 0.0, 1.0)?;
    large_moment.slots[2].first.fill(f32::MAX);
    large_moment.slots[2].second.fill(0.0);
    large_moment.slots[2].steps = u32::MAX - 1;
    let mut extreme = ResidentParameters::from_restored_values(
        owner.snapshot().values().to_vec(),
        u64::from(u32::MAX - 1),
    )?;
    extreme.restore_adamw(&large_moment)?;
    let before = bits(&values(&extreme).await?);
    let gradient = extreme.snapshot().bind_gradients(
        extreme
            .snapshot()
            .values()
            .iter()
            .map(|p| device.upload(p.layout().shape(), &vec![0.0; p.layout().len()]))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    )?;
    assert!(matches!(
        receipt(&extreme.adamw(&gradient, AdamWStep::new(1.0, 0.0)?)?).await,
        Err(TrainingError::Rejected { .. })
    ));
    assert_eq!(bits(&values(&extreme).await?), before);
    assert_eq!(
        state_bits(&state(&extreme).await?),
        state_bits(&large_moment)
    );
    // Clock overflow rejects every tensor; freezing preserves its exact integer.
    let mut s = state(&owner).await?;
    s.slots[2].steps = u32::MAX;
    let mut exhausted = ResidentParameters::from_restored_values(
        owner.snapshot().values().to_vec(),
        u64::from(u32::MAX),
    )?;
    exhausted.restore_adamw(&s)?;
    let before = bits(&values(&exhausted).await?);
    let gradient = exhausted.snapshot().bind_gradients(
        exhausted
            .snapshot()
            .values()
            .iter()
            .map(|p| device.upload(p.layout().shape(), &vec![0.25; p.layout().len()]))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    )?;
    assert!(matches!(
        receipt(&exhausted.adamw(&gradient, AdamWStep::new(0.01, 0.0)?)?).await,
        Err(TrainingError::Rejected { .. })
    ));
    assert_eq!(bits(&values(&exhausted).await?), before);
    assert_eq!(state_bits(&state(&exhausted).await?), state_bits(&s));
    let mut edge = ResidentParameters::new(vec![device.upload(&[1], &[1.0])?])?;
    edge.enable_adamw(AdamWConfig::new(
        f32::from_bits(0x3f7fffff),
        f32::from_bits(0x3f7fffff),
        1.0,
    )?)?;
    let gradient = edge
        .snapshot()
        .bind_gradients(vec![device.upload(&[1], &[1.0])?])?;
    receipt(&edge.adamw(&gradient, AdamWStep::new(1.0, 0.0)?)?).await?;
    close(&values(&edge).await?[0], &[0.5])?;
    assert_eq!(state(&edge).await?.slots[0].steps, 1);
    let gradient = exhausted.snapshot().bind_gradients(
        exhausted
            .snapshot()
            .values()
            .iter()
            .map(|p| device.upload(p.layout().shape(), &vec![0.25; p.layout().len()]))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    )?;
    receipt(&exhausted.adamw(&gradient, AdamWStep::new(0.0, 0.0)?)?).await?;
    assert_eq!(bits(&values(&exhausted).await?), before);
    assert_eq!(state_bits(&state(&exhausted).await?), state_bits(&s));
    Ok(
        serde_json::json!({"status":"passed", "torch_updates":12, "torch":f["torch"],
        "same_device_resume":"bitwise", "atomic_rejection":true, "retained_snapshot":true,
        "scalar_resume_updates":3, "scalar_snapshot_restore":true,
        "frozen_gradient_poison":"guarded_divide_by_zero",
        "division_domain_rejection":true, "division_headroom_rejection":true}),
    )
}

async fn scalar_resume(device: &TensorDevice) -> Result<()> {
    let config = AdamWConfig::new(0.5, 0.75, 0.25)?;
    let step = AdamWStep::new(0.1, 0.2)?;
    let mut owner = ResidentParameters::new(vec![device.upload(&[], &[1.0])?])?;
    owner.enable_adamw(config)?;
    let retained = owner.adamw_snapshot()?.ok_or("missing scalar optimizer")?;
    let mut expected = (1.0, 0.0, 0.0, 0);
    for gradient in [0.5, -0.25, 1.5] {
        // Exercise both fresh and nonzero-clock restore, with rank preserved.
        let history = state(&owner).await?;
        assert!(history.slots[0].shape.is_empty());
        assert_eq!(history.slots[0].first.len(), 1);
        assert_eq!(history.slots[0].second.len(), 1);
        let snapshot = owner.snapshot();
        let mut restored = ResidentParameters::from_restored_values(
            snapshot.values().to_vec(),
            snapshot.revision(),
        )?;
        restored.restore_adamw(&history)?;
        let restored_history = state(&restored).await?;
        assert_eq!(restored_history.slots[0].shape, history.slots[0].shape);
        assert_eq!(state_bits(&restored_history), state_bits(&history));
        assert!(restored.snapshot().values()[0].layout().shape().is_empty());
        assert_eq!(
            bits(&values(&restored).await?),
            bits(&values(&owner).await?)
        );
        let candidate = step.candidate(
            config, expected.0, gradient, expected.1, expected.2, expected.3,
        )?;
        expected = (
            candidate.value,
            candidate.first,
            candidate.second,
            candidate.steps,
        );
        for current in [&mut owner, &mut restored] {
            let gradient = current
                .snapshot()
                .bind_gradients(vec![device.upload(&[], &[gradient])?])?;
            assert_eq!(
                receipt(&current.adamw(&gradient, step)?).await?,
                u64::from(expected.3)
            );
            assert!(current.snapshot().values()[0].layout().shape().is_empty());
            close(&values(current).await?[0], &[expected.0])?;
            let history = state(current).await?;
            close(&history.slots[0].first, &[expected.1])?;
            close(&history.slots[0].second, &[expected.2])?;
            assert_eq!(history.slots[0].steps, expected.3);
        }
        assert_eq!(
            bits(&values(&owner).await?),
            bits(&values(&restored).await?)
        );
        assert_eq!(
            state_bits(&state(&owner).await?),
            state_bits(&state(&restored).await?)
        );
    }
    #[cfg(not(target_arch = "wasm32"))]
    let retained = retained.read()?;
    #[cfg(target_arch = "wasm32")]
    let retained = retained.read_async().await?;
    assert!(retained.slots[0].shape.is_empty());
    assert_eq!(state_bits(&retained), vec![vec![0, 0, 0]]);
    Ok(())
}
