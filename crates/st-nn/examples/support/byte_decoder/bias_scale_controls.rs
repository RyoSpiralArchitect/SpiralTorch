//! One-time calibration, then the existing complete resident learner unchanged.
use super::*;
use sha2::{Digest, Sha256};
use st_kernel_contracts::causal_bias::{CausalBiasMoments, CausalBiasScaleMatch};

type Windows = Vec<Vec<Vec<u8>>>;
type Observations = Vec<Vec<Vec<f32>>>;
const TOLERANCE: f64 = 1e-5;

async fn measure(
    runtime: &WgpuRuntime,
    plan: &ByteDecoderPlan,
    windows: &Windows,
) -> Result<(Vec<CausalBiasMoments>, Observations)> {
    let mut model = plan.compile_training_wgpu(runtime.clone())?;
    let mut moments: Vec<CausalBiasMoments> = Vec::new();
    let mut observed = Vec::new();
    for rows in windows {
        let resident = model.prepare_batch(&batch(rows, 4)?)?;
        let forward = model.forward(&resident)?;
        if forward.parameter_revision() != 0
            || forward.geometry_pair_bias(plan.block_count()).is_some()
        {
            return Err("calibration changed revision or exposes a nonexistent block".into());
        }
        let mut scores = Vec::new();
        for block in 0..plan.block_count() {
            let bias = forward
                .geometry_pair_bias(block)
                .ok_or("missing owned bias")?;
            let values = read(bias).await?;
            let sample =
                CausalBiasMoments::from_scores(bias.layout().shape().try_into()?, &values)?;
            if let Some(m) = moments.get_mut(block) {
                m.merge(&sample)?;
            } else {
                moments.push(sample);
            }
            scores.push(values);
        }
        observed.push(scores);
    }
    if model.parameter_snapshot().revision() != 0 {
        return Err("calibration consumed a training update".into());
    }
    Ok((moments, observed))
}

async fn accessor_controls(
    runtime: &WgpuRuntime,
    p: &ByteDecoderPlan,
    case: &Value,
    windows: &Windows,
    expected: &Observations,
) -> Result<()> {
    let mut model = p.compile_training_wgpu(runtime.clone())?;
    let mut changed = windows.last().ok_or("missing calibration windows")?.clone();
    for row in &mut changed {
        *row.last_mut().ok_or("missing target")? ^= 255;
    }
    let resident = model.prepare_batch(&batch(&changed, 4)?)?;
    let external = biases(model.tensor_device(), case, 4)?;
    let forward = model.forward_with_external_biases(&resident, &borrowed(&external))?;
    for block in 0..p.block_count() {
        let actual = read(
            forward
                .geometry_pair_bias(block)
                .ok_or("missing owned bias")?,
        )
        .await?;
        if !same_bits(&actual, &expected.last().unwrap()[block]) {
            return Err(
                "targets or external scores contaminated owned geometry calibration".into(),
            );
        }
    }
    let mut ordinary_case = case.clone();
    ordinary_case["config"]
        .as_object_mut()
        .unwrap()
        .remove("causal_geometry");
    ordinary_case["parameters"]
        .as_array_mut()
        .unwrap()
        .retain(|parameter| !parameter["name"].as_str().unwrap().starts_with("geometry."));
    let mut ordinary = plan(&ordinary_case, 4)?.compile_training_wgpu(runtime.clone())?;
    let resident = ordinary.prepare_batch(&batch(&changed, 4)?)?;
    let forward = ordinary.forward(&resident)?;
    if forward.geometry_pair_bias(0).is_some() || forward.geometry_pair_bias(usize::MAX).is_some() {
        return Err("ordinary models expose nonexistent geometry scores".into());
    }
    Ok(())
}

pub(super) async fn run(runtime: WgpuRuntime, input: &str) -> Result<Value> {
    if format!("{:x}", Sha256::digest(input.as_bytes()))
        != "3eb8e538371d0aa968abfd1f9d3758b7f83464ce6b36e06fb0fe9c91f2e9a00b"
    {
        return Err("frozen calibration fixture SHA-256 mismatch".into());
    }
    let fixture: Value = serde_json::from_str(input)?;
    if fixture["schema"] != "spiraltorch.resident_byte_bias_scale.torch_fixture.v1"
        || fixture["tolerance"] != json!({"atol":3e-6,"rtol":5e-5,"geometry_relative_l2":0.002})
        || fixture["cases"].as_array().map(Vec::len) != Some(2)
    {
        return Err("invalid bias calibration fixture".into());
    }
    let mut calibrated_cases = Vec::new();
    let mut records = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        if case["config"]["causal_geometry"]["pair_metric"] != "euclidean_chord_squared.v1"
            || case["calibration"]["relative_tolerance"] != TOLERANCE
        {
            return Err("invalid calibration metric/tolerance".into());
        }
        let windows: Windows = serde_json::from_value(case["calibration"]["windows"].clone())?;
        if windows.len() != 2
            || windows
                .iter()
                .any(|b| b.len() != 2 || b.iter().any(|w| w.len() != 5))
        {
            return Err("incomplete fixed calibration window selection".into());
        }
        let mut uncalibrated = case.clone();
        uncalibrated["parameters"] = case["uncalibrated_parameters"].clone();
        let candidate = plan(&uncalibrated, 4)?;
        let mut reference_case = uncalibrated.clone();
        reference_case["config"]["causal_geometry"]["pair_metric"] = json!("poincare_squared.v1");
        let reference = plan(&reference_case, 4)?;
        let (target, reference_scores) = measure(&runtime, &reference, &windows).await?;
        let (before, candidate_scores) = measure(&runtime, &candidate, &windows).await?;
        let old_gains = candidate
            .causal_geometry()
            .ok_or("missing geometry")?
            .raw_gains();
        let fits = target
            .iter()
            .zip(&before)
            .zip(old_gains)
            .map(|((a, b), g)| CausalBiasScaleMatch::new(a, b, g, TOLERANCE))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let gains: Vec<_> = fits.iter().map(|f| f.raw_gains().to_vec()).collect();
        let fitted = candidate.clone().with_initial_geometry_raw_gains(&gains)?;
        let (after, fitted_scores) = measure(&runtime, &fitted, &windows).await?;
        let mut errors = Vec::new();
        let mut changed = false;
        for ((a, b), c) in target.iter().zip(&before).zip(&after) {
            for ((target, before), after) in a.rms().iter().zip(b.rms()).zip(c.rms()) {
                let error = (after / target - 1.).abs();
                if !error.is_finite() || error > TOLERANCE {
                    return Err(
                        "realized device bias RMS did not match; no iterative retuning allowed"
                            .into(),
                    );
                }
                changed |= (before / target - 1.).abs() > 1e-4;
                errors.push(error);
            }
        }
        if !changed {
            return Err("calibration positive control is insensitive".into());
        }
        accessor_controls(&runtime, &fitted, case, &windows, &fitted_scores).await?;
        let mut calibrated = uncalibrated;
        let slots = candidate.parameter_layout().geometry().unwrap().raw_gains();
        for (slot, gain) in slots.zip(&gains) {
            calibrated["parameters"][slot]["values"] = json!(gain);
        }
        let checkpoint = fitted.initial_checkpoint().to_json()?;
        // The unchanged oracle runner reconstructs this exact fitted plan. No
        // expected Torch output is ever used to set its weights or statistics.
        if plan(&calibrated, 4)?.initial_checkpoint().to_json()? != checkpoint {
            return Err("calibrated learner differs from the fitted plan".into());
        }
        records.push(json!({"name":case["name"], "shape":case["calibration"]["shape"],
            "windows":windows, "reference_scores":reference_scores, "candidate_scores":candidate_scores,
            "fitted_scores":fitted_scores, "reference_rms":target.iter().map(CausalBiasMoments::rms).collect::<Vec<_>>(),
            "candidate_rms":before.iter().map(CausalBiasMoments::rms).collect::<Vec<_>>(),
            "fitted_rms":after.iter().map(CausalBiasMoments::rms).collect::<Vec<_>>(),
            "valid_pairs_per_head":target[0].valid_pairs_per_head(), "relative_tolerance":TOLERANCE,
            "old_raw_gains":old_gains, "raw_gains":gains,
            "requested_scales":fits.iter().map(|f| f.requested_scales()).collect::<Vec<_>>(),
            "realized_relative_errors":errors, "before_checkpoint_json":candidate.initial_checkpoint().to_json()?,
            "calibrated_checkpoint_json":checkpoint, "no_updates_consumed":true,
            "targets_and_external_bias_excluded":true, "ordinary_and_out_of_range_absent":true,
            "positive_control":true}));
        calibrated_cases.push(calibrated);
    }
    let learning = run_cases(runtime, &calibrated_cases, &[23, 37], true, true).await?;
    for (record, check) in records.iter().zip(learning["checks"].as_array().unwrap()) {
        if record["calibrated_checkpoint_json"] != check["initial_checkpoint_json"] {
            return Err("calibration and actual learner initialization differ".into());
        }
    }
    Ok(
        json!({"schema":"spiraltorch.resident_byte_bias_scale.validation.v1", "passed":true,
        "calibration":records, "learning":learning,
        "scope":"one-time initial score-strength matching and synthetic training/resume; not differentiable normalization, matched training dynamics, corpus quality or speed"}),
    )
}
