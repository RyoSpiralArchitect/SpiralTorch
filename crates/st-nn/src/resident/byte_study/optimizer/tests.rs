use super::*;
use crate::resident::byte_study::tests::{fixture, saved, valid};

pub(in crate::resident::byte_study) fn recipe() -> Value {
    json!({"kind":"adamw_zero_rate_freeze.v1", "beta1":0.9, "beta2":0.999,
        "epsilon":1e-8, "weight_decay":0.01, "geometry_weight_decay":0.0})
}

pub(in crate::resident::byte_study) fn wrap(mut value: Value, optimizer: Value) -> Value {
    value["design_schema"] = value["schema"].clone();
    value["schema"] = json!(REQUEST_SCHEMA);
    value["optimizer"] = optimizer;
    value
}

pub(in crate::resident::byte_study) fn designs() -> Vec<Value> {
    let v1 = fixture();
    let mut v2 = v1.clone();
    v2["schema"] = json!("spiraltorch.byte_corpus.request.v2");
    for case in v2["cases"].as_array_mut().unwrap() {
        case["geometry_update"] = json!("train");
    }
    let mut frozen = v2["cases"][1].clone();
    frozen["name"] = json!("frozen");
    frozen["geometry_update"] = json!("frozen");
    v2["cases"].as_array_mut().unwrap().push(frozen);
    let v5 = headwise::tests::source();
    let mut v4 = v5.clone();
    v4["schema"] = json!("spiraltorch.byte_corpus.request.v4");
    v4["cases"].as_array_mut().unwrap().truncate(7);
    let mut v3 = v4.clone();
    v3["schema"] = json!("spiraltorch.byte_corpus.request.v3");
    v3.as_object_mut().unwrap().remove("bias_calibration");
    v3["cases"].as_array_mut().unwrap().truncate(5);
    for case in v3["cases"].as_array_mut().unwrap() {
        case.as_object_mut().unwrap().remove("bias_initialization");
    }
    let v7 = mixture::tests::prepared();
    let v8 = serde_json::from_str(
        &ByteCorpusQueryMixtureControls::from_json(v7.to_string().as_bytes())
            .unwrap()
            .prepare()
            .request_json,
    )
    .unwrap();
    vec![v1, v2, v3, v4, v5, headwise::tests::prepared(), v7, v8]
}

// Structural fixtures only: these records are not evidence of a training run.
pub(in crate::resident::byte_study) fn synthetic(
    study: &ByteCorpusStudy,
    cursor: usize,
) -> ByteCorpusStudyCheckpoint {
    let mut checkpoint = saved(study, cursor);
    for (case, record) in study.request.cases.iter().zip(&mut checkpoint.cases) {
        let initial = study.request.initial_checkpoint(case).unwrap();
        let mut model: Value = serde_json::from_str(&initial.to_json().unwrap()).unwrap();
        model["attempted_revision"] = json!(cursor.to_string());
        if let Some(slots) = model
            .get_mut("optimizer")
            .and_then(|o| o.get_mut("slots"))
            .and_then(Value::as_array_mut)
        {
            for (slot, parameter) in slots.iter_mut().zip(&case.parameters) {
                slot["steps"] = json!(if case.parameter_frozen(&parameter.name) {
                    0
                } else {
                    cursor
                });
            }
        }
        record.model_json = ByteDecoderCheckpoint::from_json(&model.to_string())
            .unwrap()
            .to_json()
            .unwrap();
    }
    checkpoint
}

#[test]
fn recipes_are_orthogonal_to_all_eight_geometry_designs() {
    for (i, design) in designs().into_iter().enumerate() {
        assert!(valid(&design), "legacy design {}", i + 1);
        let legacy = ByteCorpusStudy::from_json(design.to_string().as_bytes()).unwrap();
        assert_eq!(
            legacy.optimizer_recipe_json().unwrap(),
            r#"{"kind":"stateless_sgd.v1"}"#
        );
        for optimizer in [json!({"kind":"stateless_sgd.v1"}), recipe()] {
            let value = wrap(design.clone(), optimizer);
            let study = ByteCorpusStudy::from_json(value.to_string().as_bytes()).unwrap();
            assert_eq!(study.design_schema(), design["schema"].as_str().unwrap());
            assert_eq!(
                study.request.arms_per_seed(),
                legacy.request.arms_per_seed()
            );
            assert_eq!(
                study.request.report_schema(true),
                "spiraltorch.byte_corpus.result.v9"
            );
            assert_eq!(
                study.request.report_schema(false),
                "spiraltorch.byte_corpus.partial.v9"
            );
            assert_eq!(
                study.request.checkpoint_schema(),
                "spiraltorch.byte_corpus.checkpoint.v9"
            );
            for cursor in 0..=study.total_updates() {
                let checkpoint = synthetic(&study, cursor);
                study.validate_segment(Some(&checkpoint), cursor).unwrap();
                let encoded = checkpoint.to_json().unwrap();
                study.checkpoint_from_json(encoded.as_bytes()).unwrap();
                assert!(encoded.len() < study.checkpoint_size_bound().unwrap());
            }
            for (case, old) in study.request.cases.iter().zip(&legacy.request.cases) {
                assert_eq!(
                    study
                        .request
                        .initial_checkpoint(case)
                        .unwrap()
                        .topology_json()
                        .unwrap(),
                    legacy
                        .request
                        .initial_checkpoint(old)
                        .unwrap()
                        .topology_json()
                        .unwrap()
                );
            }
        }
    }
}

#[test]
fn optimizer_metadata_cannot_be_null_implicit_or_silently_ignored() {
    let base = wrap(fixture(), recipe());
    for key in ["optimizer", "design_schema"] {
        let mut v = base.clone();
        v.as_object_mut().unwrap().remove(key);
        assert!(!valid(&v));
        v[key] = Value::Null;
        assert!(!valid(&v));
        let mut legacy = fixture();
        legacy[key] = base[key].clone();
        assert!(!valid(&legacy));
        legacy[key] = Value::Null;
        assert!(!valid(&legacy));
    }
    for invalid in [
        json!({}),
        json!({"kind":"unknown"}),
        json!({"kind":"stateless_sgd.v1","beta1":0.9}),
        json!({"kind":"adamw_zero_rate_freeze.v1"}),
        Value::Null,
    ] {
        let mut v = base.clone();
        v["optimizer"] = invalid;
        assert!(!valid(&v));
    }
    for key in [
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "geometry_weight_decay",
    ] {
        for bad in [
            json!(-1.),
            json!(1e100),
            json!(1e-100),
            json!(1e-40),
            Value::Null,
            json!(true),
        ] {
            let mut v = base.clone();
            v["optimizer"][key] = bad;
            assert!(!valid(&v), "accepted {key}");
        }
        let mut v = base.clone();
        v["optimizer"].as_object_mut().unwrap().remove(key);
        assert!(!valid(&v));
    }
    for key in ["beta1", "beta2"] {
        let mut v = base.clone();
        v["optimizer"][key] = json!(1.);
        assert!(!valid(&v));
        v["optimizer"][key] = json!(0.);
        assert!(valid(&v));
    }
    let mut v = base.clone();
    v["optimizer"]["epsilon"] = json!(0.);
    assert!(!valid(&v));
    v = base.clone();
    v["optimizer"]["unexpected"] = json!(true);
    assert!(!valid(&v));
    for bad in [
        REQUEST_SCHEMA,
        "unknown",
        "spiraltorch.byte_corpus.request.v2",
    ] {
        let mut v = base.clone();
        v["design_schema"] = json!(bad);
        assert!(!valid(&v));
    }
}

#[test]
fn optimizer_resume_binds_configuration_clocks_and_frozen_history() {
    let value = wrap(designs().pop().unwrap(), recipe());
    let study = ByteCorpusStudy::from_json(value.to_string().as_bytes()).unwrap();
    let baseline = synthetic(&study, 1);
    for (index, case) in study.request.cases.iter().enumerate() {
        let original: Value = serde_json::from_str(&baseline.cases[index].model_json).unwrap();
        for (slot, parameter) in case.parameters.iter().enumerate() {
            let mut cp = baseline.clone();
            let mut model = original.clone();
            model["optimizer"]["slots"][slot]["steps"] =
                json!(if case.parameter_frozen(&parameter.name) {
                    1
                } else {
                    0
                });
            cp.cases[index].model_json = model.to_string();
            assert!(
                study.checked_models(&cp).is_err(),
                "clock {}",
                parameter.name
            );
            if case.parameter_frozen(&parameter.name) {
                let mut model = original.clone();
                model["optimizer"]["slots"][slot]["first"][0] = json!(-0.0);
                cp.cases[index].model_json = model.to_string();
                assert!(
                    study.checked_models(&cp).is_err(),
                    "frozen moment {}",
                    parameter.name
                );
            }
        }
        let mut cp = baseline.clone();
        let mut model = original.clone();
        model["optimizer"]["beta1"] = json!(0.8);
        cp.cases[index].model_json = model.to_string();
        assert!(study.checked_models(&cp).is_err());
        cp.cases[index].model_json = plan(&study.request.config, case)
            .unwrap()
            .initial_checkpoint()
            .to_json()
            .unwrap();
        assert!(study.checked_models(&cp).is_err());
    }
    for field in ["weight_decay", "geometry_weight_decay", "beta1", "epsilon"] {
        let mut changed = value.clone();
        changed["optimizer"][field] = json!(0.2);
        let other = ByteCorpusStudy::from_json(changed.to_string().as_bytes()).unwrap();
        assert!(other
            .checkpoint_from_json(baseline.to_json().unwrap().as_bytes())
            .is_err());
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn adamw_study_resume_is_exact_and_frozen_history_stays_zero() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte_optimizer.test").unwrap();
    let value = wrap(designs()[1].clone(), recipe());
    let study = ByteCorpusStudy::from_json(value.to_string().as_bytes()).unwrap();
    let full = pollster::block_on(study.advance(runtime.clone(), None, 2)).unwrap();
    let partial = pollster::block_on(study.advance(runtime.clone(), None, 1)).unwrap();
    let checkpoint = study
        .checkpoint_from_json(partial.checkpoint.to_json().unwrap().as_bytes())
        .unwrap();
    let resumed = pollster::block_on(study.advance(runtime, Some(&checkpoint), 2)).unwrap();
    assert_eq!(
        full.checkpoint.to_json().unwrap(),
        resumed.checkpoint.to_json().unwrap()
    );
    assert_eq!(full.report, resumed.report);
    for (case, model) in study
        .request
        .cases
        .iter()
        .zip(study.checked_models(&full.checkpoint).unwrap())
    {
        let state = model.adamw_state().unwrap();
        for (parameter, slot) in case.parameters.iter().zip(&state.slots) {
            assert_eq!(
                slot.steps,
                if case.parameter_frozen(&parameter.name) {
                    0
                } else {
                    2
                }
            );
            if case.parameter_frozen(&parameter.name) {
                assert!(slot
                    .first
                    .iter()
                    .chain(&slot.second)
                    .all(|x| x.to_bits() == 0));
            }
        }
    }
}
