use super::super::mixture::tests::prepared as source;
use super::super::tests::{saved, valid};
use super::*;

fn prepared() -> Value {
    serde_json::from_str(
        &ByteCorpusQueryMixtureControls::from_json(source().to_string().as_bytes())
            .unwrap()
            .prepare()
            .request_json,
    )
    .unwrap()
}

#[test]
fn contextual_controls_preserve_source_and_use_one_shared_intercept() {
    let original = source();
    let mut v = prepared();
    assert_eq!(
        &v["cases"].as_array().unwrap()[..17],
        original["cases"].as_array().unwrap()
    );
    assert_eq!(v["cases"].as_array().unwrap().len(), 25);
    for _ in 0..25 {
        assert!(valid(&v));
        v["cases"].as_array_mut().unwrap().rotate_left(1);
    }
    let r = parse(prepared().to_string().as_bytes()).unwrap();
    for i in [17, 18] {
        let c = &r.cases[i];
        assert!(c.parameter_frozen("geometry.query_projection.0.bias"));
        assert_eq!(
            c.parameter_frozen("geometry.query_projection.0.weight"),
            i == 18
        );
        assert!(!c.parameter_frozen("geometry.mixture_logits.0"));
        assert!(!c.parameter_frozen("geometry.raw_gain.0"));
        let p = plan(&r.config, c).unwrap();
        assert!(p
            .initial_checkpoint()
            .to_json()
            .unwrap()
            .contains("checkpoint.v5"));
    }
}

#[test]
fn contextual_controls_reject_policy_initialization_and_schema_ambiguities() {
    for version in 1..=7 {
        let mut v = prepared();
        v["schema"] = json!(format!("spiraltorch.byte_corpus.request.v{version}"));
        assert!(!valid(&v));
    }
    for value in [
        Value::Null,
        json!({}),
        json!({"source_request_sha256":"bad"}),
    ] {
        let mut v = prepared();
        v["query_controls"] = value;
        assert!(!valid(&v));
    }
    for value in [
        Value::Null,
        json!({}),
        json!({"weight_update":"train","bias":"train"}),
        json!({"weight_update":"detach","bias":"frozen_zero"}),
    ] {
        let mut v = prepared();
        v["cases"][17]["mixture"]["query_projection"] = value;
        assert!(!valid(&v));
    }
    for slot in [0, 6, 7, 8, 9] {
        let mut v = prepared();
        v["cases"][17]["parameters"][slot]["values"][0] = json!(0.01);
        assert!(!valid(&v));
    }
    let mut v = prepared();
    v["cases"][17]["parameters"][8]["values"][0] = json!(-0.0);
    assert!(!valid(&v));
    let mut v = prepared();
    v["cases"][17]["mixture"]["scales"][0][0] = json!(123.);
    assert!(!valid(&v));
    let mut v = prepared();
    v["cases"][17]["mixture"]["router_update"] = json!("frozen");
    assert!(!valid(&v));
    let mut v = prepared();
    v["cases"].as_array_mut().unwrap().pop();
    assert!(!valid(&v));
}

#[test]
fn resumed_query_bias_is_always_frozen_and_weight_policy_is_enforced() {
    let study = ByteCorpusStudy::from_json(prepared().to_string().as_bytes()).unwrap();
    let initial = saved(&study, 1);
    assert_eq!(initial.schema, "spiraltorch.byte_corpus.checkpoint.v8");
    study.checked_models(&initial).unwrap();
    for i in [17, 18] {
        for slot in [0, 1] {
            let mut cp = initial.clone();
            let mut model: Value = serde_json::from_str(&cp.cases[i].model_json).unwrap();
            model["model"]["geometry"]["mixture"]["query_projections"][0]["parameters"][slot]
                ["values"][0] = json!(0.1);
            cp.cases[i].model_json = model.to_string();
            assert_eq!(study.checked_models(&cp).is_ok(), i == 17 && slot == 0);
        }
    }
}
