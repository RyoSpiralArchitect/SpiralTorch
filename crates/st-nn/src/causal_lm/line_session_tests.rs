use super::*;
use sha2::{Digest, Sha256};
type Session = NativeCausalLmLineTrainingSession;

fn wire() -> serde_json::Value {
    json!({"schema":"spiraltorch.native_causal_lm_line_plan.v1","epochs":2,"learning_rate":0.125,"max_tile_values":11,"examples":[
        {"prompt_ids":[1,3],"candidates":[[1],[1,2],[3,4]],"stop_id":0,"objective":{"kind":"set"}},
        {"prompt_ids":[1,3],"candidates":[[1],[1,2],[3,4]],"stop_id":5,"objective":{"kind":"menu","gold_index":1,"mix":0.5}},
        {"prompt_ids":[1,3],"candidates":[[1],[1,2],[3,4]],"stop_id":0,"objective":{"kind":"subset","accepted_indices":[0,2]}}
    ]})
}
fn pin(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[test]
fn all_cuts_match_manual_gradients_sgd_receipts_and_both_owners() {
    let plan = serde_json::to_vec(&wire()).unwrap();
    let p = [1, 3];
    let lines = vec![vec![1], vec![1, 2], vec![3, 4]];
    for acc in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F32WideBackward,
        CausalLmReadoutAccumulator::F64,
    ] {
        let origin = model(1, acc);
        let mut m = origin.clone();
        let mut live = Session::new(origin.clone(), &plan).unwrap();
        assert_eq!(live.plan_id(), pin(&plan));
        let mut snapshots = vec![live.checkpoint_with_id().unwrap()];
        let mut reports = vec![];
        for i in 0..6 {
            let g = match i % 3 {
                0 => m.geometry_line_set_gradients(&p, &lines, 0),
                1 => m.geometry_line_menu_gradients(&p, &lines, 1, 5, 0.5),
                _ => m.geometry_line_subset_gradients(&p, &lines, &[0, 2], 0),
            }
            .unwrap();
            let r = live.step().unwrap().unwrap();
            let next = m
                .step_adaptation_sgd(&g, CausalLmAdaptationMode::Geometry, 0.125)
                .unwrap();
            assert_eq!(
                (
                    r.update as usize,
                    r.epoch as usize,
                    r.example_index,
                    r.loss,
                    r.target_tokens
                ),
                (i + 1, i / 3, i % 3, g.loss(), g.target_tokens())
            );
            assert_eq!(r.pre_geometry_id, m.geometry_artifact_id().unwrap());
            assert_eq!(r.post_geometry_id, next.geometry_artifact_id().unwrap());
            assert_eq!(
                live.model().geometry_artifact().unwrap().to_json(),
                next.geometry_artifact().unwrap().to_json()
            );
            assert_eq!(
                live.model().lora_to_json().unwrap(),
                origin.lora_to_json().unwrap()
            );
            assert_eq!(r.work.peak_logit_values, 11);
            reports.push(serde_json::to_vec(&r).unwrap());
            snapshots.push(live.checkpoint_with_id().unwrap());
            m = next;
        }
        for cut in 0..=6 {
            let (raw, id) = &snapshots[cut];
            let mut s = Session::restore(origin.clone(), &plan, raw.as_bytes(), id).unwrap();
            assert_eq!(&s.checkpoint_with_id().unwrap(), &snapshots[cut]);
            for i in cut..6 {
                assert_eq!(
                    serde_json::to_vec(&s.step().unwrap().unwrap()).unwrap(),
                    reports[i]
                );
                assert_eq!(s.checkpoint_with_id().unwrap(), snapshots[i + 1]);
            }
            assert!(s.exhausted() && s.step().unwrap().is_none());
            assert_eq!(s.checkpoint_with_id().unwrap(), snapshots[6]);
        }
    }
}

#[test]
fn admission_checks_last_case_and_exact_work_without_running_a_decoder() {
    let m = model(1, CausalLmReadoutAccumulator::F32);
    decoder::work_counts::take();
    let mut cases = Vec::new();
    for (field, value) in [
        ("schema", json!("other")),
        ("epochs", json!(0)),
        ("epochs", json!(Session::MAX_UPDATES)),
        ("learning_rate", json!(0)),
        ("max_tile_values", json!(10)),
        ("mystery", json!(3)),
        ("examples", json!([])),
    ] {
        let mut v = wire();
        v[field] = value;
        cases.push(v);
    }
    for (field, value) in [
        ("prompt_ids", json!([11])),
        ("candidates", json!([[2], [1]])),
        ("candidates", json!([[1], [1]])),
        ("stop_id", json!(2)),
        (
            "objective",
            json!({"kind":"subset","accepted_indices":[2,0]}),
        ),
        ("objective", json!({"kind":"menu","gold_index":3,"mix":0.5})),
        ("objective", json!({"kind":"set","junk":0})),
    ] {
        let mut v = wire();
        v["examples"][2][field] = value;
        cases.push(v);
    }
    for v in cases {
        assert!(
            Session::new(m.clone(), v.to_string().as_bytes()).is_err(),
            "{v}"
        );
    }
    let mut v = wire();
    let ex = v["examples"][0].clone();
    v["examples"] = json!(vec![ex; Session::MAX_EXAMPLES + 1]);
    assert!(Session::new(m.clone(), v.to_string().as_bytes()).is_err());
    assert!(Session::new(m.clone(), &vec![b' '; Session::MAX_PLAN_BYTES + 1]).is_err());
    let v = wire().to_string();
    assert!(Session::new(load(), v.as_bytes()).is_err());
    let mut limited = m;
    limited.limits.max_logit_elements = 54;
    assert!(Session::new(limited, v.as_bytes()).is_err());
    assert_eq!(decoder::work_counts::take(), [0; 3]);
}

#[test]
fn restore_rejects_changed_plan_cursor_layout_frozen_values_and_owners() {
    let mut origin = model(1, CausalLmReadoutAccumulator::F32);
    let mut art = origin.geometry_artifact().unwrap().record().clone();
    art.placements[0].tensors[0].requires_grad = false;
    origin = origin
        .with_geometry(GeometryAdapterArtifact::create(art).unwrap())
        .unwrap();
    let plan = wire().to_string();
    let mut live = Session::new(origin.clone(), plan.as_bytes()).unwrap();
    live.step().unwrap();
    let (raw, id) = live.checkpoint_with_id().unwrap();
    let restore = |m: NativeCausalLm, p: &str, r: &str, h: &str| {
        Session::restore(m, p.as_bytes(), r.as_bytes(), h)
    };
    assert!(restore(origin.clone(), &plan, &raw, "sha256:bad").is_err());
    assert!(restore(origin.clone(), &(plan.clone() + " "), &raw, &id).is_err());
    assert!(restore(load(), &plan, &raw, &id).is_err());
    assert!(restore(live.model().clone(), &plan, &raw, &id).is_err());
    assert!(restore(
        origin.with_readout_accumulator(CausalLmReadoutAccumulator::F64),
        &plan,
        &raw,
        &id
    )
    .is_err());
    let changed = load()
        .with_geometry(origin.geometry_artifact().unwrap().clone())
        .unwrap()
        .with_lora(1, 1, 1.)
        .unwrap();
    assert!(restore(changed, &plan, &raw, &id).is_err());
    let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let reject = |v: serde_json::Value| {
        let x = v.to_string();
        assert!(restore(origin.clone(), &plan, &x, &pin(x.as_bytes())).is_err());
    };
    for (key, val) in [
        ("updates", json!(0)),
        ("updates", json!(7)),
        ("schema", json!("other")),
        ("geometry_id", json!("x")),
        ("lora_id", json!(null)),
        ("extra", json!(true)),
    ] {
        let mut v = value.clone();
        v[key] = val;
        reject(v);
    }
    let mut missing = value.clone();
    missing.as_object_mut().unwrap().remove("lora_id");
    reject(missing);
    for layout in [false, true] {
        let mut rec = live.model().geometry_artifact().unwrap().record().clone();
        if layout {
            rec.placements[0].training = !rec.placements[0].training;
        } else {
            rec.placements[0].tensors[0].values[0] += 0.25;
        }
        let replacement = GeometryAdapterArtifact::create(rec).unwrap();
        let mut v = value.clone();
        v["geometry_json"] = json!(replacement.to_json());
        v["geometry_id"] = json!(replacement.artifact_id());
        reject(v);
    }
    assert_eq!(live.checkpoint_with_id().unwrap(), (raw, id));
}

#[test]
fn failing_second_gradient_cannot_commit_a_partial_segment() {
    let mut m = model(1, CausalLmReadoutAccumulator::F32);
    // Inject a post-load fault at just the second example's input embedding.
    // The frozen vocabulary head remains pristine, including its column 10.
    let width = m.config.width;
    m.token.data_mut()[10 * width] = f32::NAN;
    let mut plan = wire();
    plan["examples"][1]["prompt_ids"] = json!([10, 3]);
    let mut session = Session::new(m, plan.to_string().as_bytes()).unwrap();
    let mut direct = session.clone();
    direct.step().unwrap().unwrap();
    let one = direct.checkpoint_with_id().unwrap();
    assert!(direct.step().is_err());
    assert_eq!(direct.checkpoint_with_id().unwrap(), one);
    let initial = session.checkpoint_with_id().unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert!(session
        .advance_to_checkpoint_new(2, dir.path().join("two"))
        .is_err());
    assert_eq!(session.checkpoint_with_id().unwrap(), initial);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn checkpoint_publication_never_clobbers_or_advances_on_error() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("step.json");
    let origin = model(1, CausalLmReadoutAccumulator::F32);
    let plan = wire().to_string();
    let mut live = Session::new(origin.clone(), plan.as_bytes()).unwrap();
    let zero = live.checkpoint_with_id().unwrap();
    let id = live.save_checkpoint_new(&p).unwrap();
    assert_eq!(id, zero.1);
    assert!(live.advance_to_checkpoint_new(2, &p).is_err());
    assert!(live
        .advance_to_checkpoint_new(2, dir.path().join("absent/s.json"))
        .is_err());
    assert!(live
        .advance_to_checkpoint_new(0, dir.path().join("zero"))
        .is_err());
    assert!(live
        .advance_to_checkpoint_new(7, dir.path().join("too-many"))
        .is_err());
    assert_eq!(live.checkpoint_with_id().unwrap(), zero);
    assert_eq!(std::fs::read(&p).unwrap(), zero.0.as_bytes());
    let new = dir.path().join("two");
    let (reports, id) = live.advance_to_checkpoint_new(2, &new).unwrap();
    assert_eq!(reports.len(), 2);
    let resumed =
        Session::restore(origin, plan.as_bytes(), &std::fs::read(&new).unwrap(), &id).unwrap();
    assert_eq!(
        resumed.checkpoint_with_id().unwrap(),
        live.checkpoint_with_id().unwrap()
    );
    assert_eq!(resumed.completed_updates(), 2);
}
