use super::*;
use crate::resident::byte_study::{
    optimizer::tests::{designs, recipe, synthetic, wrap},
    tests::valid,
};

fn active() -> Value {
    json!({"kind":"active_global_l2.v1", "max_norm":0.01})
}

fn bind(mut request: Value, clip: Value) -> Value {
    request["schema"] = json!(REQUEST_SCHEMA);
    request["gradient_clip"] = clip;
    request
}

#[test]
fn explicit_clipping_is_orthogonal_to_all_geometry_designs_and_optimizers() {
    for design in designs() {
        let legacy = ByteCorpusStudy::from_json(design.to_string().as_bytes()).unwrap();
        assert_eq!(
            legacy.gradient_clip_recipe_json().unwrap(),
            r#"{"kind":"disabled.v1"}"#
        );
        for optimizer in [json!({"kind":"stateless_sgd.v1"}), recipe()] {
            let v9 = wrap(design.clone(), optimizer);
            assert!(valid(&v9));
            for policy in [json!({"kind":"disabled.v1"}), active()] {
                let request = bind(v9.clone(), policy);
                let study = ByteCorpusStudy::from_json(request.to_string().as_bytes()).unwrap();
                assert_eq!(study.design_schema(), legacy.design_schema());
                assert_eq!(
                    study.request.arms_per_seed(),
                    legacy.request.arms_per_seed()
                );
                assert_eq!(
                    study.request.checkpoint_schema(),
                    "spiraltorch.byte_corpus.checkpoint.v10"
                );
                assert_eq!(
                    study.request.report_schema(true),
                    "spiraltorch.byte_corpus.result.v10"
                );
                assert_eq!(
                    study.request.report_schema(false),
                    "spiraltorch.byte_corpus.partial.v10"
                );
                for cursor in 0..=study.total_updates() {
                    let saved = synthetic(&study, cursor).to_json().unwrap();
                    let decoded = study.checkpoint_from_json(saved.as_bytes()).unwrap();
                    study.validate_segment(Some(&decoded), cursor).unwrap();
                    assert!(saved.len() < study.checkpoint_size_bound().unwrap());
                }
            }
        }
    }
}

#[test]
fn clipping_cannot_be_implicit_lossy_null_or_ignored_by_legacy_requests() {
    let designs = designs();
    let base = bind(wrap(designs[0].clone(), recipe()), active());
    for field in ["gradient_clip", "optimizer", "design_schema"] {
        let mut value = base.clone();
        value.as_object_mut().unwrap().remove(field);
        assert!(!valid(&value));
        value[field] = Value::Null;
        assert!(!valid(&value));
    }
    for bad in [
        json!({}),
        json!({"kind":"unknown"}),
        json!({"kind":"active_global_l2.v1"}),
        json!({"kind":"disabled.v1","max_norm":1.}),
        json!({"kind":"active_global_l2.v1","max_norm":1.,"extra":0}),
    ] {
        let mut value = base.clone();
        value["gradient_clip"] = bad;
        assert!(!valid(&value));
    }
    for bad in [
        json!(0.),
        json!(-0.),
        json!(-1.),
        json!(true),
        Value::Null,
        json!(1e100),
        json!(1e-100),
        json!(1e-40),
    ] {
        let mut value = base.clone();
        value["gradient_clip"]["max_norm"] = bad;
        assert!(!valid(&value));
    }
    for mut legacy in designs.into_iter().chain([wrap(base_design(), recipe())]) {
        assert!(valid(&legacy));
        legacy["gradient_clip"] = active();
        assert!(!valid(&legacy));
        legacy["gradient_clip"] = Value::Null;
        assert!(!valid(&legacy));
    }
    for bad in [REQUEST_SCHEMA, optimizer::REQUEST_SCHEMA, "unknown"] {
        let mut value = base.clone();
        value["design_schema"] = json!(bad);
        assert!(!valid(&value));
    }
}

fn base_design() -> Value {
    crate::resident::byte_study::tests::fixture()
}

#[test]
fn resume_binds_the_exact_clipping_policy_and_envelope() {
    let request = bind(wrap(base_design(), recipe()), active());
    let study = ByteCorpusStudy::from_json(request.to_string().as_bytes()).unwrap();
    let checkpoint = synthetic(&study, 1);
    let saved = checkpoint.to_json().unwrap();
    for policy in [
        json!({"kind":"disabled.v1"}),
        json!({"kind":"active_global_l2.v1","max_norm":0.02}),
    ] {
        let mut changed = request.clone();
        changed["gradient_clip"] = policy;
        let other = ByteCorpusStudy::from_json(changed.to_string().as_bytes()).unwrap();
        assert!(other.checkpoint_from_json(saved.as_bytes()).is_err());
    }
    let mut changed = checkpoint.clone();
    changed.schema = "spiraltorch.byte_corpus.checkpoint.v9".into();
    assert!(study
        .checkpoint_from_json(changed.to_json().unwrap().as_bytes())
        .is_err());
    let mut v9 = request;
    v9["schema"] = json!(optimizer::REQUEST_SCHEMA);
    v9.as_object_mut().unwrap().remove("gradient_clip");
    let other = ByteCorpusStudy::from_json(v9.to_string().as_bytes()).unwrap();
    assert!(other.checkpoint_from_json(saved.as_bytes()).is_err());
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn learning_design() -> Value {
    let mut request = designs()[1].clone();
    request["config"]["width"] = json!(4);
    request["config"]["hidden"] = json!(4);
    // The structural fixture is all 0.1: its symmetry can erase geometry VJPs.
    // Keep paired arms identical but make positions, channels and tokens distinct.
    for case in request["cases"].as_array_mut().unwrap() {
        for parameter in case["parameters"].as_array_mut().unwrap() {
            let name = parameter["name"].as_str().unwrap().to_owned();
            let shape: Vec<usize> = match name.as_str() {
                "token_embedding" => vec![256, 4],
                "position_embedding" => vec![2, 4],
                "geometry.projection.weight" => vec![4, 2],
                _ if name.starts_with("geometry.") => parameter["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect(),
                _ => parameter["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| match n.as_u64().unwrap() {
                        2 => 4,
                        6 => 12,
                        n => n as usize,
                    })
                    .collect(),
            };
            let phase = name.bytes().map(u32::from).sum::<u32>() as f32;
            let values: Vec<f32> = (0..shape.iter().product())
                .map(|i| {
                    if name.ends_with(".gain") {
                        1.
                    } else if name.ends_with(".bias") {
                        0.
                    } else {
                        0.35 * (phase + i as f32 * 0.71).sin()
                    }
                })
                .collect();
            parameter["shape"] = json!(shape);
            parameter["values"] = json!(values);
        }
    }
    request
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn clipped_studies_update_geometry_freeze_and_resume_without_changing_legacy() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let runtime = st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.study.clip")
        .unwrap()
        .0;
    for optimizer in [json!({"kind":"stateless_sgd.v1"}), recipe()] {
        let v9 = wrap(learning_design(), optimizer);
        let legacy = ByteCorpusStudy::from_json(v9.to_string().as_bytes()).unwrap();
        let old = pollster::block_on(legacy.advance(runtime.clone(), None, 2)).unwrap();
        let disabled = ByteCorpusStudy::from_json(
            bind(v9.clone(), json!({"kind":"disabled.v1"}))
                .to_string()
                .as_bytes(),
        )
        .unwrap();
        let off = pollster::block_on(disabled.advance(runtime.clone(), None, 2)).unwrap();
        assert_eq!(off.report["cases"], old.report["cases"]);
        for (a, b) in off.checkpoint.cases.iter().zip(&old.checkpoint.cases) {
            assert_eq!(a.model_json, b.model_json);
        }
        let study = ByteCorpusStudy::from_json(bind(v9, active()).to_string().as_bytes()).unwrap();
        let full = pollster::block_on(study.advance(runtime.clone(), None, 2)).unwrap();
        assert_ne!(
            full.report["cases"][0]["final_parameters"], off.report["cases"][0]["final_parameters"],
            "clipping must affect actual learning"
        );
        let partial = pollster::block_on(study.advance(runtime.clone(), None, 1)).unwrap();
        let checkpoint = study
            .checkpoint_from_json(partial.checkpoint.to_json().unwrap().as_bytes())
            .unwrap();
        let resumed =
            pollster::block_on(study.advance(runtime.clone(), Some(&checkpoint), 2)).unwrap();
        assert_eq!(full.report, resumed.report);
        assert_eq!(
            full.checkpoint.to_json().unwrap(),
            resumed.checkpoint.to_json().unwrap()
        );
        let noop = pollster::block_on(study.advance(runtime.clone(), Some(&resumed.checkpoint), 2))
            .unwrap();
        assert_eq!(full.report, noop.report);
        assert_eq!(
            full.checkpoint.to_json().unwrap(),
            noop.checkpoint.to_json().unwrap()
        );
        for (case, model) in study
            .request
            .cases
            .iter()
            .zip(study.checked_models(&full.checkpoint).unwrap())
        {
            if case.geometry {
                assert_eq!(
                    model
                        .geometry_parameters_match(&study.request.initial_checkpoint(case).unwrap())
                        .unwrap(),
                    case.frozen(),
                    "geometry update policy for {}",
                    case.name
                );
            }
        }
    }
}
