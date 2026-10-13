use super::super::headwise::tests::source as fixture;
use super::super::tests::{saved, valid};
use super::*;

pub(in crate::resident::byte_study) fn prepared() -> Value {
    serde_json::from_str(
        &ByteCorpusMixtureControls::from_json(fixture().to_string().as_bytes())
            .unwrap()
            .materialize(&BTreeMap::new())
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn mixtures_preserve_v5_and_admit_every_subset_and_only_router_freezing() {
    let original = fixture();
    let mut v = prepared();
    assert_eq!(
        &v["cases"].as_array().unwrap()[..9],
        original["cases"].as_array().unwrap()
    );
    assert_eq!(v["cases"].as_array().unwrap().len(), 17);
    for _ in 0..17 {
        assert!(valid(&v));
        v["cases"].as_array_mut().unwrap().rotate_left(1);
    }
    let r = parse(prepared().to_string().as_bytes()).unwrap();
    let c = &r.cases[10];
    assert!(!c.frozen());
    assert!(c.router_frozen());
    assert!(c.parameter_frozen("geometry.mixture_logits.0"));
    assert!(!c.parameter_frozen("geometry.raw_gain.0"));
    assert!(!c.parameter_frozen("token_embedding"));
    assert_eq!(r.arms_per_seed(), 17);
}

#[test]
fn mixture_metadata_cannot_be_relabelled_or_silently_ignored() {
    for version in 1..=6 {
        let mut v = prepared();
        v["schema"] = json!(format!("spiraltorch.byte_corpus.request.v{version}"));
        assert!(!valid(&v));
    }
    for value in [
        Value::Null,
        json!({}),
        json!({"source_request_sha256":"Z".repeat(64)}),
    ] {
        let mut v = prepared();
        v["mixture_controls"] = value;
        assert!(!valid(&v));
    }
    for value in [
        Value::Null,
        json!({}),
        json!({"metrics":[P,E],"scales":[[1.,1.,1.,1.]],"router_update":"unknown"}),
    ] {
        let mut v = prepared();
        v["cases"][9]["mixture"] = value;
        assert!(!valid(&v));
    }
    for field in ["pair_metric", "head_pair_metrics"] {
        let mut v = prepared();
        v["cases"][9][field] = if field == "pair_metric" {
            json!(P)
        } else {
            json!([[P, E]])
        };
        assert!(!valid(&v));
    }
    let mut v = prepared();
    v["cases"][9]["geometry_update"] = json!("frozen");
    assert!(!valid(&v));
    let mut v = prepared();
    v["cases"].as_array_mut().unwrap().pop();
    assert!(!valid(&v));
}

#[test]
fn mixture_units_uniform_logits_and_twin_parameters_are_bound() {
    for index in 9..17 {
        for slot in [0, 2, 4, 6, 7] {
            let mut v = prepared();
            v["cases"][index]["parameters"][slot]["values"][0] = json!(0.3);
            assert!(!valid(&v));
        }
        for scale in [0., -1., 99.] {
            let mut v = prepared();
            v["cases"][index]["mixture"]["scales"][0][0] = json!(scale);
            assert!(!valid(&v));
        }
        let mut v = prepared();
        v["cases"][index]["parameters"][7]["values"][0] = json!(-0.0);
        assert!(!valid(&v));
    }
    let mut v = prepared();
    for index in [9, 10] {
        v["cases"][index]["parameters"][6]["values"] = json!([1.2, 1.3]);
    }
    assert!(valid(&v)); // Preparation may fit the shared gain, then runtime checks its RMS.
}

#[test]
fn resumed_fixed_router_rejects_mutation_but_not_encoder_learning() {
    let study = ByteCorpusStudy::from_json(prepared().to_string().as_bytes()).unwrap();
    let baseline = saved(&study, 1);
    assert_eq!(baseline.schema, "spiraltorch.byte_corpus.checkpoint.v7");
    study.checked_models(&baseline).unwrap();
    for case in [9, 10] {
        let mut cp = baseline.clone();
        let mut model: Value = serde_json::from_str(&cp.cases[case].model_json).unwrap();
        model["model"]["geometry"]["raw_decay"][0] = json!(0.2);
        cp.cases[case].model_json = model.to_string();
        study.checked_models(&cp).unwrap();
        model["model"]["geometry"]["mixture"]["raw_logits"][0][0] = json!(0.01);
        cp.cases[case].model_json = model.to_string();
        assert_eq!(study.checked_models(&cp).is_ok(), case == 9);
    }
    let mut cp = baseline.clone();
    let mut model: Value = serde_json::from_str(&cp.cases[10].model_json).unwrap();
    model["model"]["geometry"]["mixture"]["scales"][0][0] = json!(2.);
    cp.cases[10].model_json = model.to_string();
    assert!(study.checked_models(&cp).is_err());
}

#[test]
fn expanded_checkpoint_budget_is_checked_before_requesting_gpu() {
    let mut source = fixture();
    let row = source["train_batches"][0].clone();
    source["train_batches"] = json!(vec![row; 1024]);
    source["validation_batches"] = json!(vec![source["validation_batches"][0].clone(); 128]);
    source["checkpoint_every"] = json!(1);
    let input = source.to_string();
    ByteCorpusStudy::from_json(input.as_bytes())
        .unwrap()
        .validate_segment(None, 0)
        .unwrap();
    assert!(ByteCorpusMixtureControls::from_json(input.as_bytes()).is_err());
}
