//! Paired same-runtime control. Never changes the earlier mixed-runtime gate.
use super::*;
use st_nn::resident::ByteDecoderQueryPullback;

fn parameter_routes(
    full: &[Vec<f32>],
    stopped: &[Vec<f32>],
    repeated: &[Vec<f32>],
    changed_geometry: std::ops::Range<usize>,
    retain: bool,
    failures: &mut Vec<Value>,
) -> Result<Vec<usize>> {
    if full.len() != stopped.len() || full.len() != repeated.len() || full.is_empty() {
        return Err("truncated query route family".into());
    }
    let mut unchanged = Vec::new();
    for (slot, ((a, b), c)) in full.iter().zip(stopped).zip(repeated).enumerate() {
        if a.is_empty()
            || a.len() != b.len()
            || a.len() != c.len()
            || a.iter().chain(b).chain(c).any(|v| !v.is_finite())
        {
            return Err("invalid query route tensor".into());
        }
        if !same_bits(a, c) {
            return Err("stopped pullback poisoned a later full pullback".into());
        }
        if slot >= 2 && !changed_geometry.contains(&slot) {
            if same_bits(a, b) {
                unchanged.push(slot);
            } else if retain {
                failures.push(json!({"gate":"stopped_unchanged_parameter","slot":slot}));
            } else {
                return Err("query stop-gradient changed an unrelated parameter VJP".into());
            }
        }
    }
    Ok(unchanged)
}

#[test]
fn caller_retains_finite_unchanged_route_failure_only_in_v2() {
    let full = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];
    let mut stopped = full.clone();
    stopped[3][0] += 1e-7;
    let mut failures = Vec::new();
    let unchanged = parameter_routes(&full, &stopped, &full, 2..3, true, &mut failures).unwrap();
    assert!(unchanged.is_empty());
    assert_eq!(
        failures,
        vec![json!({"gate":"stopped_unchanged_parameter","slot":3})]
    );
    assert!(parameter_routes(&full, &stopped, &full, 2..3, false, &mut Vec::new()).is_err());
    stopped[3][0] = f32::NAN;
    assert!(parameter_routes(&full, &stopped, &full, 2..3, true, &mut Vec::new()).is_err());
    assert!(parameter_routes(&full, &full[..3], &full, 2..3, true, &mut Vec::new()).is_err());
    assert_eq!(
        parameter_routes(&full, &full, &full, 2..3, true, &mut Vec::new()).unwrap(),
        [3]
    );
}

fn stopped_parameter_check(
    actual: &[f32],
    expected: &[f32],
    slot: usize,
    geometry: bool,
    failures: &mut Vec<Value>,
) -> Result<Value> {
    if actual.is_empty() || expected.is_empty() {
        return Err("empty stopped parameter family".into());
    }
    let error = observed_close(
        actual,
        expected,
        &format!("stopped parameter VJP {slot}"),
        0,
        true,
        failures,
    )?;
    let relative = if geometry {
        match geometry_controls::relative_gradient(actual, expected) {
            Ok(relative) => Some(relative),
            Err(error) => {
                // observed_close already rejected malformed/nonfinite families.
                failures.push(json!({"gate":"stopped_geometry_relative_l2","slot":slot,
                    "threshold":0.002,"error":error.to_string()}));
                None
            }
        }
    } else {
        None
    };
    Ok(json!({"slot":slot,"max_abs_error":error,"relative_l2":relative}))
}

#[test]
fn finite_stopped_mismatches_are_retained_but_invalid_data_aborts() {
    let mut failures = Vec::new();
    let ok = stopped_parameter_check(&[1e-6], &[1e-6], 2, true, &mut failures).unwrap();
    assert_eq!(ok["relative_l2"], 0.);
    assert!(failures.is_empty());
    let bad = stopped_parameter_check(&[0.], &[1e-7], 2, true, &mut failures).unwrap();
    assert_eq!(bad["max_abs_error"], f64::from(1e-7f32));
    assert!(bad["relative_l2"].is_null());
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0]["gate"], "stopped_geometry_relative_l2");
    stopped_parameter_check(&[1.], &[0.], 0, false, &mut failures).unwrap();
    assert_eq!(failures.len(), 2);
    for (actual, expected) in [
        (vec![f32::NAN], vec![0.]),
        (vec![0.], vec![f32::INFINITY]),
        (vec![], vec![0.]),
        (vec![], vec![]),
    ] {
        assert!(stopped_parameter_check(&actual, &expected, 0, true, &mut failures).is_err());
    }
    assert_eq!(failures.len(), 2);
}

pub(super) async fn run(runtime: WgpuRuntime, fixture: &str) -> Result<Value> {
    let cases = projective_controls::cases(fixture)?;
    run_cases(runtime, &cases, false).await
}

pub(super) async fn run_cases(
    runtime: WgpuRuntime,
    cases: &[Value],
    module_residual: bool,
) -> Result<Value> {
    if cases.len() != 6 {
        return Err("paired control requires all six model cases".into());
    }
    let mut checks = Vec::new();
    for case in &cases[4..] {
        let plan = plan(case, 4)?;
        let mut model = plan.compile_training_wgpu(runtime.clone())?;
        let windows: Vec<Vec<u8>> = case["windows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| {
                sizes(w)
                    .into_iter()
                    .map(|v| u8::try_from(v).unwrap())
                    .collect()
            })
            .collect();
        let resident = model.prepare_batch(&batch(&windows, 4)?)?;
        let external = biases(model.tensor_device(), case, 4)?;
        let before = read_many(model.tensor_device(), model.parameter_snapshot().values()).await?;
        let forward = model.forward_with_external_biases(&resident, &borrowed(&external))?;
        let output = read(forward.prediction()).await?;
        close(&output, &data(&case["output"]), "paired logits")?;
        let seed = model
            .tensor_device()
            .upload(model.output_layout().shape(), &data(&case["cotangent"]))?;
        let full = model.backward(&forward, &seed)?;
        // Freeze each observation before another backward can touch any buffers.
        let full_values = read(full.embedding_output_gradient()).await?;
        let full_parameters = read_many(model.tensor_device(), full.parameter_gradients()).await?;
        let stopped = model.backward_with_query_pullback(
            &forward,
            &seed,
            ByteDecoderQueryPullback::StopCoordinates,
        )?;
        let stopped_values = read(stopped.embedding_output_gradient()).await?;
        let stopped_parameters =
            read_many(model.tensor_device(), stopped.parameter_gradients()).await?;
        let full_retained = read_many(model.tensor_device(), full.parameter_gradients()).await?;
        if !same_bits(&full_values, &read(full.embedding_output_gradient()).await?)
            || !same_family(&full_parameters, &full_retained)
        {
            return Err("stopped pullback overwrote a retained full VJP".into());
        }
        let repeated = model.backward(&forward, &seed)?;
        if full.query_pullback() != ByteDecoderQueryPullback::Full
            || stopped.query_pullback() != ByteDecoderQueryPullback::StopCoordinates
        {
            return Err("query pullback provenance lost".into());
        }
        let stopped_retained =
            read_many(model.tensor_device(), stopped.parameter_gradients()).await?;
        if !same_bits(
            &full_values,
            &read(repeated.embedding_output_gradient()).await?,
        ) || !same_bits(&output, &read(forward.prediction()).await?)
            || !same_bits(
                &stopped_values,
                &read(stopped.embedding_output_gradient()).await?,
            )
            || !same_family(&stopped_parameters, &stopped_retained)
        {
            return Err("query control changed a retained forward or full backward".into());
        }
        close(
            &full_values,
            &data(&case["embedding_output_gradient"]),
            "paired full embedding VJP",
        )?;
        let reference_stopped = data(&case["query_controls"]["router_detached_embedding_gradient"]);
        close(
            &stopped_values,
            &reference_stopped,
            "paired stopped embedding VJP",
        )?;
        let repeated_parameters =
            read_many(model.tensor_device(), repeated.parameter_gradients()).await?;
        let layout = plan
            .parameter_layout()
            .geometry()
            .ok_or("missing query geometry")?;
        let changed_geometry = layout.projection().start..layout.raw_gains().start;
        if full_parameters.len() != plan.parameter_layout().len()
            || stopped_parameters.len() != full_parameters.len()
            || repeated_parameters.len() != full_parameters.len()
        {
            return Err("truncated query pullback parameter family".into());
        }
        let mut stopped_errors = Vec::new();
        let mut stopped_failures = Vec::new();
        if module_residual {
            let reference = case["query_controls"]["router_detached_parameter_gradients"]
                .as_array()
                .ok_or("missing independent stopped parameter oracle")?;
            if reference.len() != stopped_parameters.len() {
                return Err("incomplete independent stopped parameter oracle".into());
            }
            for (slot, (actual, expected)) in stopped_parameters.iter().zip(reference).enumerate() {
                let expected = data(expected);
                stopped_errors.push(stopped_parameter_check(
                    actual,
                    &expected,
                    slot,
                    layout.all().contains(&slot),
                    &mut stopped_failures,
                )?);
            }
        }
        let unchanged = parameter_routes(
            &full_parameters,
            &stopped_parameters,
            &repeated_parameters,
            changed_geometry,
            module_residual,
            &mut stopped_failures,
        )?;
        for range in layout.query_projections() {
            for slot in range.clone() {
                geometry_controls::relative_gradient(
                    &full_parameters[slot],
                    &data(&case["parameter_gradients"][slot]),
                )?;
            }
        }
        let now = read_many(model.tensor_device(), model.parameter_snapshot().values()).await?;
        if before.len() != now.len() || before.iter().zip(&now).any(|(a, b)| !same_bits(a, b)) {
            return Err("paired query observation changed model parameters".into());
        }
        let difference = |a: &[f32], b: &[f32]| -> Result<Vec<f32>> {
            if a.len() != b.len() || a.is_empty() {
                return Err("invalid paired contrast shape".into());
            }
            let values: Vec<_> = a.iter().zip(b).map(|(a, b)| a - b).collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite paired contrast".into());
            }
            Ok(values)
        };
        let reference = difference(
            &data(&case["embedding_output_gradient"]),
            &reference_stopped,
        )?;
        let delta = difference(&full_values, &stopped_values)?;
        let paired = geometry_controls::relative_gradient(&delta, &reference);
        let legacy = geometry_controls::relative_gradient(
            &difference(&full_values, &reference_stopped)?,
            &reference,
        );
        for bad in [
            vec![0.; reference.len()],
            reference.iter().map(|v| -v).collect(),
        ] {
            if geometry_controls::relative_gradient(&bad, &reference).is_ok() {
                return Err("paired query negative control was accepted".into());
            }
        }
        checks.push(json!({"name":case["name"], "passed":paired.is_ok() && stopped_failures.is_empty(),
            "paired_relative_l2":paired.as_ref().ok(), "paired_error":paired.err().map(|e|e.to_string()),
            "legacy_mixed_runtime_passed":legacy.is_ok(), "legacy_mixed_runtime_error":legacy.err().map(|e|e.to_string()),
            "unchanged_parameter_slots":unchanged,"negative_controls_rejected":2,
            "forward_unchanged":true,"parameters_unchanged":true,"full_repeated_exact":true,
            "full_embedding_gradient":full_values,"stopped_embedding_gradient":stopped_values,
            "full_parameter_gradients":full_parameters,"stopped_parameter_gradients":stopped_parameters}));
        if module_residual {
            checks.last_mut().unwrap()["stopped_parameter_errors"] = json!(stopped_errors);
            checks.last_mut().unwrap()["stopped_parameter_failures"] = json!(stopped_failures);
        }
    }
    Ok(
        json!({"schema":if module_residual {"spiraltorch.resident_byte_query_pullback.validation.v2"}
            else {"spiraltorch.resident_byte_query_pullback.validation.v1"},
        "passed":checks.iter().all(|c|c["passed"] == true),"checks":checks,
        "adapter":format!("{:?}",runtime.adapter_info()),
        "scope":if module_residual {
            "paired query-input stop-gradient correctness; legacy_mixed_runtime fields describe the formula on this fixture, not historical v1 outcomes; no corpus quality or speed claim"
        } else {
            "paired query-input stop-gradient correctness, not the old six-case qualification, corpus quality or speed"
        }}),
    )
}

fn same_family(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_bits(a, b))
}
