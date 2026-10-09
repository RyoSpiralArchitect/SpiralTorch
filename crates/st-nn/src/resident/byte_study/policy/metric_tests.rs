use super::super::tests::{fixture, saved, valid};
use super::*;

fn controls() -> Value {
    let mut value = fixture();
    value["schema"] = json!("spiraltorch.byte_corpus.request.v3");
    value["cases"][0]["geometry_update"] = json!("train");
    let geometric = value["cases"][1].clone();
    value["cases"].as_array_mut().unwrap().truncate(1);
    for metric in ["poincare_squared.v1", "euclidean_chord_squared.v1"] {
        for update in ["train", "frozen"] {
            let mut case = geometric.clone();
            case["name"] = json!(format!("{metric}-{update}"));
            case["geometry_update"] = json!(update);
            case["pair_metric"] = json!(metric);
            value["cases"].as_array_mut().unwrap().push(case);
        }
    }
    value
}

#[test]
fn five_arm_coverage_is_order_independent_and_explicit() {
    let mut value = controls();
    for _ in 0..5 {
        assert!(valid(&value));
        value["cases"].as_array_mut().unwrap().rotate_left(1);
    }
    for (index, field, bad) in [
        (0, "pair_metric", json!("poincare_squared.v1")),
        (1, "pair_metric", json!("euclidean_chord_squared.v1")),
        (3, "pair_metric", json!("poincare_squared.v1")),
        (4, "pair_metric", json!("unknown.v1")),
        (4, "pair_metric", json!("categorical_fisher_rao_squared.v1")),
        (0, "pair_metric", Value::Null),
        (3, "pair_metric", Value::Null),
        (4, "geometry_update", json!("train")),
        (0, "geometry_update", json!("frozen")),
    ] {
        let mut value = controls();
        value["cases"][index][field] = bad;
        assert!(!valid(&value));
    }
    for field in ["pair_metric", "geometry_update"] {
        let mut value = controls();
        value["cases"][3].as_object_mut().unwrap().remove(field);
        assert!(!valid(&value));
    }
    let mut value = controls();
    value["cases"].as_array_mut().unwrap().pop();
    assert!(!valid(&value));
    for schema in [
        "spiraltorch.byte_corpus.request.v1",
        "spiraltorch.byte_corpus.request.v2",
    ] {
        let mut value = controls();
        value["schema"] = json!(schema);
        assert!(!valid(&value));
    }
    let mut legacy = fixture();
    legacy["cases"][1]["pair_metric"] = json!("poincare_squared.v1");
    assert!(!valid(&legacy));
}

#[test]
fn both_metrics_and_policies_share_all_initial_bits() {
    for case in 1..5 {
        for slot in [0, 2, 3, 4, 5, 6] {
            let mut value = controls();
            value["cases"][case]["parameters"][slot]["values"][0] = json!(0.2);
            assert!(!valid(&value));
        }
    }
    let mut value = controls();
    for case in 1..5 {
        value["cases"][case]["parameters"][2]["values"][0] = json!(0.0);
    }
    value["cases"][3]["parameters"][2]["values"][0] = json!(-0.0);
    assert!(!valid(&value));
}

#[test]
fn checkpoints_bind_metric_identity_and_only_freeze_selected_arms() {
    let mut request = controls();
    let batches = request["train_batches"].as_array().unwrap().clone();
    request["train_batches"] = json!(batches.into_iter().cycle().take(128).collect::<Vec<_>>());
    request["checkpoint_every"] = json!(64);
    let study = ByteCorpusStudy::from_json(request.to_string().as_bytes()).unwrap();
    assert_eq!(
        study.request.report_schema(false),
        "spiraltorch.byte_corpus.partial.v3"
    );
    assert_eq!(
        study.request.report_schema(true),
        "spiraltorch.byte_corpus.result.v3"
    );
    for cursor in [0, 37, 64, 128] {
        let checkpoint = saved(&study, cursor);
        assert_eq!(checkpoint.schema, "spiraltorch.byte_corpus.checkpoint.v3");
        study
            .checkpoint_from_json(checkpoint.to_json().unwrap().as_bytes())
            .unwrap();
        assert!(study.checkpoint_size_bound().unwrap() >= checkpoint.to_json().unwrap().len());
    }
    let checkpoint = saved(&study, 1);
    for i in 0..5 {
        let model: Value = serde_json::from_str(&checkpoint.cases[i].model_json).unwrap();
        assert_eq!(
            model["schema"],
            if i < 3 {
                "spiraltorch.nn.byte_decoder_checkpoint.v1"
            } else {
                "spiraltorch.nn.byte_decoder_checkpoint.v2"
            }
        );
        if i >= 3 {
            assert_eq!(
                model["model"]["geometry"]["pair_metric"],
                "euclidean_chord_squared.v1"
            );
        }
    }
    for i in 1..5 {
        let mut changed = checkpoint.clone();
        let mut model: Value = serde_json::from_str(&changed.cases[i].model_json).unwrap();
        model["model"]["geometry"]["raw_decay"][0] = json!(0.2);
        changed.cases[i].model_json = model.to_string();
        assert_eq!(study.checked_models(&changed).is_ok(), i == 1 || i == 3);

        let mut changed = checkpoint.clone();
        let mut model: Value = serde_json::from_str(&changed.cases[i].model_json).unwrap();
        if i < 3 {
            model["schema"] = json!("spiraltorch.nn.byte_decoder_checkpoint.v2");
            model["model"]["geometry"]["pair_metric"] = json!("euclidean_chord_squared.v1");
        } else {
            model["schema"] = json!("spiraltorch.nn.byte_decoder_checkpoint.v1");
            model["model"]["geometry"]
                .as_object_mut()
                .unwrap()
                .remove("pair_metric");
        }
        changed.cases[i].model_json = model.to_string();
        assert!(study.checked_models(&changed).is_err());
    }
    for schema in [CHECKPOINT_SCHEMA, "spiraltorch.byte_corpus.checkpoint.v2"] {
        let mut changed = checkpoint.clone();
        changed.schema = schema.into();
        assert!(study.checked_models(&changed).is_err());
    }
}

#[test]
fn metric_studies_keep_the_bounded_case_budget() {
    let mut request = controls();
    let group = request["cases"].as_array().unwrap().clone();
    for seed in [8, 9] {
        for case in &group {
            let mut copy = case.clone();
            copy["name"] = json!(format!("{}-{seed}", case["name"].as_str().unwrap()));
            copy["seed"] = json!(seed);
            request["cases"].as_array_mut().unwrap().push(copy);
        }
    }
    assert_eq!(request["cases"].as_array().unwrap().len(), 15);
    assert!(valid(&request));
    let mut sixteen = request.clone();
    let mut extra = group[0].clone();
    extra["name"] = json!("extra");
    extra["seed"] = json!(10);
    sixteen["cases"].as_array_mut().unwrap().push(extra);
    assert!(!valid(&sixteen));
    let mut fourth = controls();
    for case in fourth["cases"].as_array_mut().unwrap() {
        case["name"] = json!(format!("{}-10", case["name"].as_str().unwrap()));
        case["seed"] = json!(10);
    }
    assert!(valid(&fourth));
    request["cases"]
        .as_array_mut()
        .unwrap()
        .extend(fourth["cases"].as_array().unwrap().iter().cloned());
    assert_eq!(request["cases"].as_array().unwrap().len(), 20);
    assert!(!valid(&request));
    let raw = controls().to_string().replacen(
        "\"pair_metric\":",
        "\"pair_metric\":null,\"pair_metric\":",
        1,
    );
    assert!(ByteCorpusStudy::from_json(raw.as_bytes()).is_err());
}
