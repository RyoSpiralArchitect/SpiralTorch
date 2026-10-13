//! New six-case contract; existing three-metric fixture protocols stay frozen.
use super::*;

fn control_values(value: &Value) -> Result<Vec<f32>> {
    let values = value
        .as_array()
        .ok_or("missing contextual control vector")?;
    if values.is_empty() {
        return Err("empty contextual control vector".into());
    }
    values
        .iter()
        .map(|value| {
            let value = value.as_f64().ok_or("nonnumeric contextual control")? as f32;
            if !value.is_finite() {
                return Err("nonfinite contextual control".into());
            }
            Ok(value)
        })
        .collect()
}

fn control_contrast(actual: &Value, baseline: &Value) -> Result<()> {
    let a = control_values(actual)?;
    let b = control_values(baseline)?;
    if a.len() != b.len() || a.iter().zip(&b).any(|(a, b)| !(a - b).is_finite()) {
        return Err("invalid contextual control contrast".into());
    }
    Ok(())
}

// Only finite numerical qualification failures may be retained. Validate the
// auxiliary controls before GPU work and their observed contrasts before check().
fn query_control_data(case: &Value, observed: Option<&Value>) -> Result<()> {
    let controls = &case["query_controls"];
    control_contrast(&case["output"], &controls["shared_output"])?;
    control_contrast(
        &case["embedding_output_gradient"],
        &controls["router_detached_embedding_gradient"],
    )?;
    if let Some(report) = observed {
        control_contrast(&report["output"], &controls["shared_output"])?;
        control_contrast(
            &report["embedding_output_gradient"],
            &controls["router_detached_embedding_gradient"],
        )?;
    }
    Ok(())
}

#[test]
fn malformed_or_nonfinite_query_controls_abort_instead_of_becoming_failures() {
    let valid = json!({"output":[1.], "embedding_output_gradient":[0.75],
        "query_controls":{"shared_output":[0.5], "router_detached_embedding_gradient":[0.25]}});
    assert!(query_control_data(&valid, Some(&valid)).is_ok());
    for key in ["shared_output", "router_detached_embedding_gradient"] {
        for bad in [
            json!([1e300]),
            json!(["bad"]),
            json!([]),
            json!([0., 1.]),
            Value::Null,
        ] {
            let mut case = valid.clone();
            case["query_controls"][key] = bad;
            assert!(query_control_data(&case, None).is_err());
        }
    }
    let mut bad = valid.clone();
    bad["query_controls"]["router_detached_embedding_gradient"] = json!([-f32::MAX]);
    let mut observed = valid.clone();
    observed["embedding_output_gradient"] = json!([f32::MAX]);
    assert!(query_control_data(&bad, None).is_ok());
    assert!(query_control_data(&bad, Some(&observed)).is_err());
    observed["output"] = json!([1e300]);
    assert!(query_control_data(&valid, Some(&observed)).is_err());
}

pub(super) fn cases(fixture_json: &str) -> Result<Vec<Value>> {
    let fixture: Value = serde_json::from_str(fixture_json)?;
    if fixture.get("execution_contract").is_some() {
        return Err("versioned execution contract cannot be evaluated as legacy v1".into());
    }
    cases_with_schema(
        fixture_json,
        "spiraltorch.resident_byte_geometry_projective.torch_fixture.v1",
    )
}

fn cases_with_schema(fixture_json: &str, schema: &str) -> Result<Vec<Value>> {
    let fixture: Value = serde_json::from_str(fixture_json)?;
    let cases = fixture["cases"]
        .as_array()
        .ok_or("missing projective cases")?;
    let metrics = json!([
        "poincare_squared.v1",
        "euclidean_chord_squared.v1",
        "categorical_fisher_rao_squared.v1",
        "complex_projective_chord_squared.v1"
    ]);
    if fixture["schema"] != schema
        || fixture["tolerance"] != json!({"atol":3e-6,"rtol":5e-5,"geometry_relative_l2":0.002})
        || cases.len() != 6
        || cases.iter().enumerate().any(|(i, c)| {
            let g = &c["config"]["causal_geometry"];
            c["config"]["heads"] != 2
                || c["config"]["blocks"].as_array().map(Vec::len) != Some(1 + i % 2)
                || g.get("head_pair_metrics").is_some()
                || if i < 2 {
                    g["pair_metric"] != "complex_projective_chord_squared.v1"
                        || g.get("mixture").is_some()
                } else {
                    g.get("pair_metric").is_some()
                        || g["mixture"]["metrics"] != metrics
                        || if i < 4 {
                            g["mixture"].get("query_projection").is_some()
                        } else {
                            g["mixture"]["query_projection"] != "linear.v1"
                                || c["query_controls"]["zero_init"] != false
                        }
                }
        })
    {
        return Err("incomplete or mislabeled projective correctness reference".into());
    }
    for case in &cases[4..] {
        query_control_data(case, None)?;
    }
    Ok(cases.clone())
}

fn module_contract() -> Value {
    json!({"residual":"projection_bias_before_residual.v1",
        "router_coordinate_vjp":"same_runtime_full_minus_stop.v1"})
}

pub(super) fn module_cases(fixture_json: &str) -> Result<Vec<Value>> {
    let fixture: Value = serde_json::from_str(fixture_json)?;
    if fixture["execution_contract"] != module_contract() {
        return Err("missing or mismatched module-residual execution contract".into());
    }
    let cases = cases_with_schema(
        fixture_json,
        "spiraltorch.resident_byte_geometry_projective.torch_fixture.v2",
    )?;
    let names = [
        "projective_causal_geometry_blocks1_metric_onlyTrue",
        "projective_causal_geometry_blocks2_metric_onlyFalse",
        "four_way_shared_causal_geometry_blocks1_metric_onlyTrue",
        "four_way_shared_causal_geometry_blocks2_metric_onlyFalse",
        "four_way_query_contextual_query_causal_geometry_blocks1_metric_onlyTrue",
        "four_way_query_contextual_query_causal_geometry_blocks2_metric_onlyFalse",
    ];
    for (case, name) in cases.iter().zip(names) {
        if case["name"] != name
            || case["learning"]["steps"] != 16
            || case["learning"]["rate"] != 0.125
        {
            return Err("mismatched module-residual case identity or optimizer".into());
        }
    }
    for case in &cases[4..] {
        let descriptors = case["parameters"].as_array().ok_or("missing parameters")?;
        let gradients = case["query_controls"]["router_detached_parameter_gradients"]
            .as_array()
            .ok_or("missing stopped parameter oracle")?;
        if gradients.len() != descriptors.len() || gradients.is_empty() {
            return Err("truncated stopped parameter oracle".into());
        }
        for (gradient, descriptor) in gradients.iter().zip(descriptors) {
            let values = control_values(gradient)?;
            if descriptor["values"].as_array().map(Vec::len) != Some(values.len()) {
                return Err("stopped parameter oracle shape mismatch".into());
            }
        }
    }
    Ok(cases)
}

pub(super) async fn run_module(runtime: WgpuRuntime, fixture_json: &str) -> Result<Value> {
    let cases = module_cases(fixture_json)?;
    let boundary = boundary(runtime.clone()).await?;
    let guard = query_controls::guard(&runtime, &cases[5]).await?;
    let mut report = run_cases_observed(
        runtime.clone(),
        &cases,
        &[23, 37, 24, 39, 26, 43],
        true,
        false,
        true,
        true,
    )
    .await?;
    let paired = query_pullback_controls::run_cases(runtime, &cases, true).await?;
    if report["checks"].as_array().map(Vec::len) != Some(6)
        || paired["checks"].as_array().map(Vec::len) != Some(2)
    {
        return Err("incomplete module-residual qualification".into());
    }
    for ((case, check), pullback) in cases[4..]
        .iter()
        .zip(
            report["checks"]
                .as_array_mut()
                .ok_or("missing checks")?
                .iter_mut()
                .skip(4),
        )
        .zip(paired["checks"].as_array().ok_or("missing paired checks")?)
    {
        query_control_data(case, Some(check))?;
        match query_controls::check_paired(case, check, pullback) {
            Ok(controls) => check["query_controls"] = controls,
            Err(error) => check["query_controls_error"] = json!(error.to_string()),
        }
    }
    report["passed"] = json!(
        report["passed"] == true
            && paired["passed"] == true
            && report["checks"]
                .as_array()
                .unwrap()
                .iter()
                .all(|c| c.get("query_controls_error").is_none())
    );
    report["schema"] = json!("spiraltorch.resident_byte_geometry_projective.validation.v2");
    report["execution_contract"] = module_contract();
    report["projective_boundary"] = boundary;
    report["query_guard"] = guard;
    report["paired_query"] = paired;
    report["scope"] = json!("module-grouped six-case qualification with same-runtime paired query controls; original v1 unchanged; no language-quality or speed claim");
    Ok(report)
}

#[cfg(test)]
mod module_contract_tests {
    use super::*;

    fn fixture() -> Value {
        let mut cases = Vec::new();
        for mode in ["projective", "four_way_shared", "four_way_query"] {
            for blocks in [1, 2] {
                let contextual = mode == "four_way_query";
                let name = format!(
                    "{mode}_{}causal_geometry_blocks{blocks}_metric_only{}",
                    if contextual { "contextual_query_" } else { "" },
                    if blocks == 1 { "True" } else { "False" }
                );
                let geometry = if mode == "projective" {
                    json!({"pair_metric":"complex_projective_chord_squared.v1"})
                } else {
                    let mut g = json!({"mixture":{"metrics":["poincare_squared.v1",
                        "euclidean_chord_squared.v1","categorical_fisher_rao_squared.v1",
                        "complex_projective_chord_squared.v1"]}});
                    if contextual {
                        g["mixture"]["query_projection"] = json!("linear.v1");
                    }
                    g
                };
                cases.push(json!({"name":name,"config":{"heads":2,
                    "blocks":vec![json!({}); blocks],"causal_geometry":geometry},
                    "parameters":[{"values":[0.]}],"output":[1.],"embedding_output_gradient":[0.75],
                    "learning":{"steps":16,"rate":0.125},
                    "query_controls":{"zero_init":false,"shared_output":[0.5],
                        "router_detached_embedding_gradient":[0.25],
                        "router_detached_parameter_gradients":[[0.]]}}));
            }
        }
        json!({"schema":"spiraltorch.resident_byte_geometry_projective.torch_fixture.v2",
            "execution_contract":module_contract(),"tolerance":{"atol":3e-6,"rtol":5e-5,"geometry_relative_l2":0.002},"cases":cases})
    }

    #[test]
    fn module_and_legacy_contracts_cannot_be_interchanged() {
        let mut f = fixture();
        assert!(module_cases(&f.to_string()).is_ok());
        assert!(cases(&f.to_string()).is_err());
        f["schema"] = json!("spiraltorch.resident_byte_geometry_projective.torch_fixture.v1");
        assert!(module_cases(&f.to_string()).is_err());
        assert!(cases(&f.to_string()).is_err());
        f.as_object_mut().unwrap().remove("execution_contract");
        assert!(cases(&f.to_string()).is_ok());
        assert!(module_cases(&f.to_string()).is_err());
    }

    #[test]
    fn malformed_module_contract_or_stopped_oracle_rejects() {
        for field in ["residual", "router_coordinate_vjp"] {
            let mut f = fixture();
            f["execution_contract"][field] = json!("different");
            assert!(module_cases(&f.to_string()).is_err());
        }
        for bad in [
            Value::Null,
            json!([]),
            json!([[0.], [0.]]),
            json!([[0., 0.]]),
            json!([[1e300]]),
        ] {
            let mut f = fixture();
            f["cases"][4]["query_controls"]["router_detached_parameter_gradients"] = bad;
            assert!(module_cases(&f.to_string()).is_err());
        }
        for bad in [json!("swapped"), Value::Null] {
            let mut f = fixture();
            f["cases"][0]["name"] = bad;
            assert!(module_cases(&f.to_string()).is_err());
        }
        let mut f = fixture();
        f["cases"][5]["learning"]["rate"] = json!(0.);
        assert!(module_cases(&f.to_string()).is_err());
    }
}

pub(super) async fn run(runtime: WgpuRuntime, fixture_json: &str) -> Result<Value> {
    let cases = cases(fixture_json)?;
    let boundary = boundary(runtime.clone()).await?;
    let guard = query_controls::guard(&runtime, &cases[5]).await?;
    let mut report = run_cases_observed(
        runtime,
        &cases,
        &[23, 37, 24, 39, 26, 43],
        true,
        false,
        true,
        false,
    )
    .await?;
    for (case, check) in cases
        .iter()
        .zip(report["checks"].as_array_mut().ok_or("missing checks")?)
        .skip(4)
    {
        query_control_data(case, Some(check))?;
        match query_controls::check(case, check) {
            Ok(controls) => check["query_controls"] = controls,
            Err(error) => check["query_controls_error"] = json!(error.to_string()),
        }
    }
    report["passed"] = json!(
        report["passed"] == true
            && report["checks"]
                .as_array()
                .unwrap()
                .iter()
                .all(|c| c.get("query_controls_error").is_none())
    );
    report["schema"] = json!("spiraltorch.resident_byte_geometry_projective.validation.v1");
    report["projective_boundary"] = boundary;
    report["query_guard"] = guard;
    Ok(report)
}

async fn boundary(runtime: WgpuRuntime) -> Result<Value> {
    use st_kernel_contracts::projective::{ProjectiveBiasForward, ProjectiveBiasSpec};
    let device = TensorDevice::new(runtime)?;
    let spec = ProjectiveBiasSpec::new([1, 2, 2], 1)?;
    let mut max_relative = 0f64;
    for (value, gain, cotangent) in [(1e-30, 1e30, 1.), (1e20, 0., 1e30), (1e-25, 0., 1e30)] {
        let x = [0., 0., value, 0.];
        let seed = [0., 0., cotangent, 0.];
        let cpu = ProjectiveBiasForward::new(spec, &x, &[gain])?.backward(&seed)?;
        let f = device
            .upload(&spec.shape(), &x)?
            .causal_projective_bias(&device.upload(&[1], &[gain])?)?;
        let g = f.backward(&device.upload(&spec.score_shape(), &seed)?)?;
        for (i, (a, b)) in read(g.coordinates())
            .await?
            .iter()
            .zip(cpu.coordinates)
            .enumerate()
        {
            if i % 2 == 0 {
                let relative = (f64::from(*a) / f64::from(b) - 1.).abs();
                if !relative.is_finite() || *a == 0. || relative > 8e-5 {
                    return Err("projective boundary gradient lost or inaccurate".into());
                }
                max_relative = max_relative.max(relative);
            } else if *a != 0. {
                return Err("spurious imaginary chart gradient".into());
            }
        }
    }
    let l = 2f32.powi(70);
    let gain = 2f32.powi(120);
    let d = 2f32.powi(29);
    let middle = 1025. * 2f32.powi(-36);
    let e = 2f32.powi(-100);
    for (cols, x, expected) in [
        (
            4,
            vec![1., 0., e, 0., -1., 0., e, 0.],
            vec![
                -2f32.powi(39),
                0.,
                2f32.powi(-61),
                0.,
                2f32.powi(39),
                0.,
                2f32.powi(-61),
                0.,
            ],
        ),
        (
            4,
            vec![l, 32., l, 0., l, 32., -l, 0.],
            vec![
                1025. * d,
                middle,
                -1025. * d,
                0.,
                1025. * d,
                middle,
                1025. * d,
                0.,
            ],
        ),
        (
            4,
            vec![l, 0., l, 0., l, 0., -l, 0.],
            vec![d, 0., -d, 0., d, 0., d, 0.],
        ),
        (
            6,
            vec![l, 0., 32., 0., l, 0., l, 0., 32., 0., -l, 0.],
            vec![
                1025. * d,
                0.,
                middle,
                0.,
                -1025. * d,
                0.,
                1025. * d,
                0.,
                middle,
                0.,
                1025. * d,
                0.,
            ],
        ),
    ] {
        let f = device
            .upload(&[1, 2, cols], &x)?
            .causal_projective_bias(&device.upload(&[1], &[gain])?)?;
        let g = f.backward(&device.upload(&[1, 1, 2, 2], &[0., 0., gain, 0.])?)?;
        for (a, b) in read(g.coordinates()).await?.iter().zip(expected) {
            if b == 0. {
                if *a != 0. {
                    return Err("unexpected imaginary cancellation gradient".into());
                }
            } else {
                let relative = (f64::from(*a) / f64::from(b) - 1.).abs();
                if !relative.is_finite() || relative > 8e-5 {
                    return Err("cancelled overlap erased the projective gradient".into());
                }
                max_relative = max_relative.max(relative);
            }
        }
    }
    let e = 2f32.powi(-121);
    let l = 2f32.powi(122);
    let f = device
        .upload(&[1, 2, 4], &[e, 0., 1., 0., 1., 0., l, 0.])?
        .causal_projective_bias(&device.upload(&[1], &[2f32.powi(60)])?)?;
    let g = f.backward(&device.upload(&[1, 1, 2, 2], &[0., 0., 2f32.powi(63), 0.])?)?;
    if read(g.coordinates()).await? != [-e, 0., l, 0., e, 0., -e, 0.] {
        return Err("projective base subtraction erased a recoverable endpoint".into());
    }
    Ok(
        json!({"cases":8,"analytic_cancellation_cases":5,"nonzero_endpoint_components":30,"max_relative_error":max_relative}),
    )
}
