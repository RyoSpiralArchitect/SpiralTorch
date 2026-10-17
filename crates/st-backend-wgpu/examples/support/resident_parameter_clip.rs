use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice},
    resident_training::{parameters::ResidentParameters, TrainingError},
};
use st_kernel_contracts::{
    adamw::{AdamWConfig, AdamWStep},
    elementwise::ElementwiseOp,
    gradient_clip::GlobalNormClip,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn read(t: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = t.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    return Ok(snapshot.read()?);
    #[cfg(target_arch = "wasm32")]
    return Ok(snapshot.read_async().await?);
}

async fn accepted(
    update: st_backend_wgpu::resident_training::parameters::ResidentParameterUpdate,
) -> Result<u64> {
    let snapshot = update.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    return Ok(snapshot.read()?);
    #[cfg(target_arch = "wasm32")]
    return Ok(snapshot.read_async().await?);
}

fn close(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(a, b)| {
            !a.is_finite()
                || (f64::from(*a) - f64::from(*b)).abs() > 3e-6 * f64::from(b.abs()).max(1e-38)
        })
    {
        return Err(format!("clipped gradient differs: {actual:?} vs {expected:?}").into());
    }
    Ok(())
}

fn bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

pub async fn run(device: &TensorDevice) -> Result<serde_json::Value> {
    let max = f32::MAX;
    let cases = vec![
        (
            "global",
            vec![vec![3., -0.], vec![4.]],
            vec![true, true],
            1.,
        ),
        ("zero", vec![vec![0., -0.], vec![0.]], vec![true, true], 1.),
        (
            "norm_floor",
            vec![vec![1e-8], vec![-1e-8]],
            vec![true, true],
            1e-30,
        ),
        (
            "near_one",
            vec![vec![1.], vec![0.]],
            vec![true, true],
            f32::from_bits(1f32.to_bits() - 1),
        ),
        (
            "wide_norm",
            vec![vec![max, -max], vec![max]],
            vec![true, true],
            1.,
        ),
        (
            "tiny_scale",
            vec![vec![max, -max], vec![max]],
            vec![true, true],
            1e-30,
        ),
        (
            "frozen_excluded",
            vec![vec![3., 4.], vec![max]],
            vec![true, false],
            1.,
        ),
        (
            "all_frozen",
            vec![vec![3., -0.], vec![max]],
            vec![false, false],
            0.1,
        ),
        (
            "many_partials",
            vec![vec![1.; 1031], vec![-2.; 515]],
            vec![true, true],
            2.,
        ),
        (
            "floor_below",
            vec![vec![f32::from_bits(0x332a_aaaa); 9]],
            vec![true],
            1e-30,
        ),
        (
            "floor_equal",
            vec![vec![f32::from_bits(0x3300_0000); 16]],
            vec![true],
            1e-30,
        ),
        (
            "floor_above_9",
            vec![vec![f32::from_bits(0x332a_aaab); 9]],
            vec![true],
            1e-30,
        ),
        (
            "floor_above_33",
            vec![vec![f32::from_bits(0x32b2_416a); 33]],
            vec![true],
            1e-30,
        ),
        (
            "floor_above_257",
            vec![vec![f32::from_bits(0x31ff_8060); 257]],
            vec![true],
            1e-30,
        ),
        (
            "subnormal_noop",
            vec![vec![f32::from_bits(1), f32::from_bits(0x007f_ffff)]],
            vec![true],
            1e-30,
        ),
    ];
    let mut reports = Vec::new();
    for (name, raw, active, limit) in cases {
        let clip = GlobalNormClip::new(limit)?;
        let tensors = raw
            .iter()
            .map(|v| {
                let shape = [v.len()];
                device.upload(if v.len() == 1 { &[] } else { &shape }, v)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let zeros = tensors
            .iter()
            .map(|t| device.upload(t.layout().shape(), &vec![0.; t.layout().len()]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut owner = ResidentParameters::new(zeros)?;
        let retained = owner.snapshot();
        let bound = retained.bind_gradients(tensors)?;
        if !matches!(
            owner.clip_gradients(&bound, clip, &[]),
            Err(TrainingError::ParameterLayout)
        ) {
            return Err("invalid active count accepted".into());
        }
        let squared: f64 = raw
            .iter()
            .zip(&active)
            .filter(|(_, enabled)| **enabled)
            .flat_map(|(v, _)| v)
            .map(|&x| f64::from(x).powi(2))
            .sum();
        let factors = clip.factors(squared)?;
        let expected: Vec<Vec<f32>> = raw
            .iter()
            .zip(&active)
            .map(|(v, enabled)| {
                v.iter()
                    .map(|&x| {
                        if *enabled {
                            factors.as_slice().iter().fold(x, |v, f| v * f)
                        } else {
                            x
                        }
                    })
                    .collect()
            })
            .collect();
        let clipped = owner.clip_gradients(&bound, clip, &active)?;
        let mut observed = Vec::new();
        for (i, tensor) in clipped.values().iter().enumerate() {
            let actual = read(tensor).await?;
            close(&actual, &expected[i])?;
            if (!active[i] || factors.as_slice().is_empty()) && !bits(&actual, &raw[i]) {
                return Err("no-op/frozen clipping changed bits".into());
            }
            if !bits(&read(&bound.values()[i]).await?, &raw[i]) {
                return Err("clipping mutated an input snapshot".into());
            }
            observed.push(actual);
        }
        // The subnormal fixture checks bit-preserving clipping and a frozen
        // commit, not denormal arithmetic in the existing nonzero-rate SGD.
        let update_rate = if name == "subnormal_noop" { 0. } else { 0.25 };
        let rates: Vec<_> = active
            .iter()
            .map(|&a| if a { update_rate } else { 0. })
            .collect();
        if accepted(owner.sgd_with_rates_clipped(&bound, &rates, clip)?).await? != 1 {
            return Err("revision".into());
        }
        for (i, value) in owner.snapshot().values().iter().enumerate() {
            let wanted: Vec<_> = expected[i]
                .iter()
                .map(|x| if rates[i] != 0. { -rates[i] * x } else { 0. })
                .collect();
            close(&read(value).await?, &wanted)?;
            if !read(&retained.values()[i]).await?.iter().all(|x| *x == 0.) {
                return Err("old parameter changed".into());
            }
        }
        if !matches!(
            owner.clip_gradients(&bound, clip, &active),
            Err(TrainingError::ParameterVersion)
        ) {
            return Err("stale gradient accepted".into());
        }
        close(&read(&clipped.values()[0]).await?, &expected[0])?;
        reports.push(
            serde_json::json!({"name":name,"passed":true,"norm_f64":squared.sqrt(),"limit":limit,
                "inputs":raw,"active":active,"clipped":observed,"sgd_rates":rates}),
        );
    }
    // Strided gradients use their logical view, not underlying storage order.
    let strided = device
        .upload(&[2, 2], &[1., 2., 3., 4.])?
        .permute(&[1, 0])?;
    let mut owner = ResidentParameters::new(vec![device.upload(&[2, 2], &[0.; 4])?])?;
    let bound = owner.snapshot().bind_gradients(vec![strided])?;
    let other = ResidentParameters::new(owner.snapshot().values().to_vec())?.snapshot();
    let foreign = other.bind_gradients(bound.values().to_vec())?;
    let clip = GlobalNormClip::new(1.)?;
    if !matches!(
        owner.clip_gradients(&foreign, clip, &[true]),
        Err(TrainingError::ParameterVersion)
    ) {
        return Err("foreign owner accepted".into());
    }
    let clipped = owner.clip_gradients(&bound, clip, &[true])?;
    close(
        &read(&clipped.values()[0]).await?,
        &[
            1. / 30f32.sqrt(),
            3. / 30f32.sqrt(),
            2. / 30f32.sqrt(),
            4. / 30f32.sqrt(),
        ],
    )?;

    // Rejected inactive derivatives cannot be sanitized by exclusion from the norm.
    let mut owner = ResidentParameters::new(vec![
        device.upload(&[1], &[1.])?,
        device.upload(&[1], &[-0.])?,
    ])?;
    owner.enable_adamw(AdamWConfig::default())?;
    let original = owner.snapshot();
    let poisoned = device
        .upload(&[1], &[1.])?
        .apply(ElementwiseOp::Divide, Some(&device.upload(&[1], &[0.])?))?;
    let bad = original.bind_gradients(vec![device.upload(&[1], &[3.])?, poisoned])?;
    let steps = [AdamWStep::new(0.1, 0.0)?, AdamWStep::new(0.0, 0.0)?];
    if accepted(owner.adamw_with_steps_clipped(&bad, &steps, clip)?)
        .await
        .is_ok()
    {
        return Err("frozen invalid derivative accepted".into());
    }
    for (old, new) in original.values().iter().zip(owner.snapshot().values()) {
        if !bits(&read(old).await?, &read(new).await?) {
            return Err("rejected update changed weights".into());
        }
    }
    let snapshot = owner.adamw_snapshot()?.unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let state = snapshot.read()?;
    #[cfg(target_arch = "wasm32")]
    let state = snapshot.read_async().await?;
    if state
        .slots
        .iter()
        .any(|s| s.steps != 0 || s.first.iter().chain(&s.second).any(|x| *x != 0.))
    {
        return Err("rejected update changed AdamW history".into());
    }
    let valid = owner.snapshot().bind_gradients(vec![
        device.upload(&[1], &[3.])?,
        device.upload(&[1], &[max])?,
    ])?;
    accepted(owner.adamw_with_steps_clipped(&valid, &steps, clip)?).await?;
    let expected = steps[0].candidate(AdamWConfig::default(), 1., 1., 0., 0., 0)?;
    close(
        &read(&owner.snapshot().values()[0]).await?,
        &[expected.value],
    )?;
    if !bits(&read(&owner.snapshot().values()[1]).await?, &[-0.]) {
        return Err("frozen AdamW changed bits".into());
    }
    Ok(
        serde_json::json!({"schema":"spiraltorch.resident_parameter_clip.v1", "passed":true,
        "cases":reports,"strided":true,"foreign_and_stale":true,"frozen_guard_rejection":true,
        "adamw_recovery":true,"quality_claim":false}),
    )
}
