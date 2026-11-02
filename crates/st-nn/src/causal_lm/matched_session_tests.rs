use super::*;
use CausalLmAdaptationMode::{Geometry, Joint, Lora};
use CausalLmSessionWindowPolicy::{CoveringTargets, Disjoint};
use NativeCausalLmTrainingSession as Session;

fn geometry_at(base: &NativeCausalLm, layer: usize) -> GeometryAdapterArtifact {
    let mut r = geometry(base).record().clone();
    r.placements[0].path = format!("transformer.h.{layer}.attn");
    GeometryAdapterArtifact::create(r).unwrap()
}

fn step_all(s: &mut [Session]) -> CausalLmResult<Option<CausalLmMatchedTrainingStep>> {
    Session::step_matched(&mut s.iter_mut().collect::<Vec<_>>())
}

#[test]
fn lockstep_matches_independent_updates_and_pinned_resume_for_every_owner() {
    let docs = vec![
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![9, 0],
        vec![4, 3, 2],
        vec![3, 4, 5, 6],
    ];
    let labels = vec![
        vec![-100, -100, -100, -100, -100, -100, -100, 8],
        vec![-100, 3],
        vec![-100, -100, 2],
        vec![1, -100, -100, -100],
    ];
    for readout in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F64,
    ] {
        let base = load().with_readout_accumulator(readout);
        for layer in [0, 1] {
            let lora = base.with_lora(layer, 2, 3.).unwrap();
            let geom = geometry_at(&base, layer);
            let joint = lora.with_geometry(geom.clone()).unwrap();
            // Geometry-only can have more than one placement and need not have LoRA.
            let mut record = geom.record().clone();
            let mut mlp = record.placements[0].clone();
            mlp.path = format!("transformer.h.{layer}.mlp");
            record.placements.push(mlp);
            let multi = base
                .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
                .unwrap();
            let control = Session::new_with_policy(
                lora,
                docs.clone(),
                Some(labels.clone()),
                session_config(Lora),
                2,
                CoveringTargets,
            )
            .unwrap();
            let mut matched = vec![
                control.clone(),
                control.fork_variant(joint.clone(), Lora).unwrap(),
                control.fork_variant(joint.clone(), Joint).unwrap(),
                control.fork_variant(multi, Geometry).unwrap(),
                control.fork_variant(joint, Geometry).unwrap(),
            ];
            assert!(std::ptr::eq(matched[0].plan(), matched[2].plan()));
            let mut refs = matched.clone();
            for k in 0..control.total_updates() {
                let result = step_all(&mut matched).unwrap().unwrap();
                assert_eq!(result.shared_prefix_layers, layer);
                assert_eq!(result.steps.len(), refs.len());
                for ((s, r), step) in matched.iter_mut().zip(&mut refs).zip(result.steps) {
                    assert_eq!(
                        serde_json::to_string(&step).unwrap(),
                        serde_json::to_string(&r.step().unwrap().unwrap()).unwrap()
                    );
                    let (raw, pin) = s.checkpoint_with_id().unwrap();
                    assert_eq!(
                        (&raw, &pin),
                        (&r.to_json().unwrap(), &r.checkpoint_id().unwrap())
                    );
                    if k == 1 {
                        *s = Session::prepare_restore(base.clone(), raw.as_bytes(), &pin)
                            .unwrap()
                            .resume_with_labels(docs.clone(), Some(labels.clone()))
                            .unwrap();
                    }
                }
            }
            assert!(step_all(&mut matched).unwrap().is_none());
        }
    }
}

#[test]
fn last_arm_can_limit_shared_depth_even_at_the_roster_limit() {
    let base = load();
    let late = base.with_lora(1, 2, 3.).unwrap();
    let early = base
        .with_lora(0, 2, 3.)
        .unwrap()
        .with_geometry(geometry_at(&base, 0))
        .unwrap();
    let first = Session::new(late, vec![vec![1, 2, 3, 4]], session_config(Lora)).unwrap();
    let mut arms = vec![first.clone(); Session::MAX_MATCHED_VARIANTS - 1];
    arms.push(first.fork_variant(early, Lora).unwrap());
    let mut ordinary = arms.clone();
    let result = step_all(&mut arms).unwrap().unwrap();
    assert_eq!(result.shared_prefix_layers, 0);
    assert_eq!(result.steps.len(), Session::MAX_MATCHED_VARIANTS);
    for ((s, reference), report) in arms.iter().zip(&mut ordinary).zip(result.steps) {
        assert_eq!(
            serde_json::to_string(&report).unwrap(),
            serde_json::to_string(&reference.step().unwrap().unwrap()).unwrap()
        );
        assert_eq!(
            s.checkpoint_id().unwrap(),
            reference.checkpoint_id().unwrap()
        );
    }
}

#[test]
fn actual_qkv_projections_skip_only_the_duplicate_frozen_forward_blocks() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    fn measure(run: impl FnOnce()) -> usize {
        let count = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&count);
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |ev| {
            if ev.op_name == "matmul_prepacked_bias"
                && ev.data["inner"] == 4
                && ev.data["cols"] == 12
            {
                seen.fetch_add(1, Ordering::Relaxed);
            }
        })));
        run();
        st_tensor::set_thread_meta_observer(previous);
        count.load(Ordering::Relaxed)
    }
    let base = load();
    let a = base.with_lora(1, 2, 3.).unwrap();
    let j = a.with_geometry(geometry_at(&base, 1)).unwrap();
    let c = Session::new(a, vec![vec![1, 2, 3, 4]], session_config(Lora)).unwrap();
    let mut paired = vec![
        c.clone(),
        c.fork_variant(j.clone(), Lora).unwrap(),
        c.fork_variant(j, Joint).unwrap(),
    ];
    let mut ordinary = paired.clone();
    let normal = measure(|| {
        for s in &mut ordinary {
            s.step().unwrap();
        }
    });
    let shared = measure(|| {
        assert_eq!(
            step_all(&mut paired).unwrap().unwrap().shared_prefix_layers,
            1
        );
    });
    // Each arm replays its own last block in backward. Only two duplicate primals disappear.
    assert_eq!((normal, shared), (9, 7));
}

#[test]
fn mismatch_and_failure_in_the_last_suffix_never_advance_earlier_arms() {
    let base = load();
    let m = base.with_lora(1, 2, 3.).unwrap();
    let docs = vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5]];
    let cfg = session_config(Lora);
    let c = Session::new(m.clone(), docs.clone(), cfg).unwrap();
    let new = |cfg| Session::new(m.clone(), docs.clone(), cfg).unwrap();
    let labels = docs
        .iter()
        .map(|d| d.iter().map(|&id| id as i64).collect())
        .collect();
    let mut different_weights = parameters();
    different_weights.get_mut("wte.weight").unwrap().1[0] += 1.;
    let other = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&different_weights),
        Default::default(),
    )
    .unwrap()
    .with_lora(1, 2, 3.)
    .unwrap();
    let mut ahead = c.clone();
    ahead.step().unwrap();
    assert!(ahead.fork_variant(m.clone(), Lora).is_err());
    assert!(c.fork_variant(other.clone(), Lora).is_err());
    let mut overflowing: serde_json::Value =
        serde_json::from_str(&m.lora_to_json().unwrap()).unwrap();
    for name in ["a", "b"] {
        overflowing[name] = json!(vec![f32::MAX; 8]);
    }
    let bad = c
        .fork_variant(base.with_lora_json(&overflowing.to_string()).unwrap(), Lora)
        .unwrap();
    for mut incompatible in [
        new(CausalLmTrainingSessionConfig {
            learning_rate: cfg.learning_rate / 2.,
            ..cfg
        }),
        new(CausalLmTrainingSessionConfig { epochs: 1, ..cfg }),
        new(CausalLmTrainingSessionConfig {
            context_length: 2,
            ..cfg
        }),
        Session::new(m.clone(), vec![vec![4, 3, 2, 1], vec![2, 3, 4, 5]], cfg).unwrap(),
        Session::new_batched(m.clone(), docs.clone(), cfg, 2).unwrap(),
        Session::new_with_policy(m.clone(), docs.clone(), Some(labels), cfg, 1, Disjoint).unwrap(),
        Session::new_with_policy(m.clone(), docs.clone(), None, cfg, 1, CoveringTargets).unwrap(),
        Session::new(
            m.clone()
                .with_readout_accumulator(CausalLmReadoutAccumulator::F64),
            docs.clone(),
            cfg,
        )
        .unwrap(),
        Session::new(other, docs.clone(), cfg).unwrap(),
        ahead,
        bad,
    ] {
        let mut first = c.clone();
        let mut second = c.clone();
        let before = incompatible.to_json().unwrap();
        assert!(Session::step_matched(&mut [&mut first, &mut second, &mut incompatible]).is_err());
        assert_eq!(incompatible.to_json().unwrap(), before);
        for fresh in [first, second] {
            assert_eq!(fresh.to_json().unwrap(), c.to_json().unwrap());
        }
    }
    assert!(step_all(&mut []).is_err());
    assert!(step_all(&mut vec![c; Session::MAX_MATCHED_VARIANTS + 1]).is_err());
}

#[test]
fn fork_rechecks_every_variant_budget_without_requiring_dense_logits() {
    let base = load();
    let m = base.with_lora(1, 2, 3.).unwrap();
    let docs = vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1], vec![2, 3]];
    let labels = vec![
        vec![-100, -100, -100, 4],
        vec![-100, 3, -100, -100],
        vec![-100, 3],
    ];
    let c = Session::new_with_policy(
        m.clone(),
        docs,
        Some(labels),
        session_config(Lora),
        2,
        CoveringTargets,
    )
    .unwrap();
    let mut other = m.clone();
    other.limits.max_logit_elements = 22;
    let mut a = c.clone();
    let mut b = c.fork_variant(other.clone(), Lora).unwrap();
    assert_eq!(
        Session::step_matched(&mut [&mut a, &mut b])
            .unwrap()
            .unwrap()
            .steps[1]
            .target_tokens,
        2
    );
    other.limits.max_logit_elements = 11;
    assert!(c.fork_variant(other, Lora).is_err());
    let mut limited = m;
    limited.limits.max_training_checkpoint_elements = 1;
    assert!(c.fork_variant(limited, Lora).is_err());
    assert!(c.fork_variant(base, Joint).is_err());
}
