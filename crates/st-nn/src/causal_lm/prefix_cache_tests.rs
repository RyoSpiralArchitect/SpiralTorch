use super::*;
use CausalLmAdaptationMode::{Joint, Lora};
use CausalLmPrefixCacheAction::{CapacityBypass, Hit, Inserted, NoFrozenBlock};
use NativeCausalLmTrainingPrefixCache as Cache;
use NativeCausalLmTrainingSession as Session;

fn session(
    m: NativeCausalLm,
    docs: Vec<Vec<usize>>,
    labels: Option<Vec<Vec<i64>>>,
    ctx: u32,
    batch: usize,
) -> Session {
    Session::new_with_policy(
        m,
        docs,
        labels,
        CausalLmTrainingSessionConfig {
            context_length: ctx,
            ..session_config(Lora)
        },
        batch,
        CausalLmSessionWindowPolicy::CoveringTargets,
    )
    .unwrap()
}

fn check_one(s: &mut Session, c: &mut Cache, action: CausalLmPrefixCacheAction) {
    let mut ordinary = s.clone();
    let non_cached = ordinary.step_with_observation(true).unwrap().unwrap();
    let actual = s.step_with_prefix_cache(c, true).unwrap().unwrap();
    assert_eq!(c.stats().last_action, Some(action));
    assert_eq!(
        serde_json::to_vec(&actual).unwrap(),
        serde_json::to_vec(&non_cached).unwrap()
    );
    assert_eq!(
        serde_json::to_vec(&actual.observation).unwrap(),
        serde_json::to_vec(&non_cached.observation).unwrap()
    );
    assert_eq!(
        ordinary.checkpoint_with_id().unwrap(),
        s.checkpoint_with_id().unwrap()
    );
}

#[test]
fn labels_and_unused_last_id_are_not_hidden_inputs_but_prompt_shape_and_position_are() {
    let base = load();
    let model = base.with_lora(1, 2, 3.).unwrap();
    let mut c = Cache::new(&base, 256, 16).unwrap();
    let mut one = session(
        model.clone(),
        vec![vec![1, 2, 3, 4, 5]],
        Some(vec![vec![-100, -100, -100, 9, 3]]),
        5,
        1,
    );
    check_one(&mut one, &mut c, Inserted);
    // Same actual forward tokens, different masks, replacement targets and final ID.
    check_one(
        &mut session(
            model.clone(),
            vec![vec![1, 2, 3, 4, 8]],
            Some(vec![vec![-100, 7, 9, -100, 6]]),
            5,
            1,
        ),
        &mut c,
        Hit,
    );
    // Flattened IDs match, but the second row resets positions and attention.
    check_one(
        &mut session(
            model.clone(),
            vec![vec![1, 2, 7], vec![3, 4, 8]],
            None,
            3,
            2,
        ),
        &mut c,
        Inserted,
    );
    check_one(
        &mut session(
            model,
            vec![vec![9, 2, 3, 4, 5]],
            Some(vec![vec![-100, -100, -100, 9, 3]]),
            5,
            1,
        ),
        &mut c,
        Inserted,
    );
    assert_eq!(
        (c.stats().entries, c.stats().retained_values, c.stats().hits),
        (3, 48, 1)
    );
}

#[test]
fn capped_working_set_keeps_hot_entries_and_clear_drops_data_and_counters() {
    let base = load();
    let model = base.with_lora(1, 2, 3.).unwrap();
    for (values, entries) in [(12, 4), (120, 1)] {
        let mut c = Cache::new(&base, values, entries).unwrap();
        let mut s = session(
            model.clone(),
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
            None,
            4,
            1,
        );
        for action in [Inserted, CapacityBypass, Hit, CapacityBypass] {
            check_one(&mut s, &mut c, action);
        }
        let snapshot = c.stats();
        assert_eq!(
            (
                snapshot.entries,
                snapshot.retained_values,
                snapshot.capacity_bypasses,
                snapshot.skipped_block_calls
            ),
            (1, 12, 2, 1)
        );
        assert!(s.step_with_prefix_cache(&mut c, false).unwrap().is_none());
        assert_eq!(snapshot, c.stats());
        c.clear();
        assert_eq!(
            (
                c.stats().entries,
                c.stats().retained_values,
                c.stats().hits,
                c.stats().last_action
            ),
            (0, 0, 0, None)
        );
    }
    let mut too_small = Cache::new(&base, 1, 1).unwrap();
    check_one(
        &mut session(model, vec![vec![1, 2, 3, 4]], None, 4, 1),
        &mut too_small,
        CapacityBypass,
    );
    check_one(
        &mut session(
            base.with_lora(0, 2, 3.).unwrap(),
            vec![vec![1, 2, 3, 4]],
            None,
            4,
            1,
        ),
        &mut too_small,
        NoFrozenBlock,
    );
    assert_eq!(too_small.stats().retained_values, 0);
    for (v, e) in [
        (0, 1),
        (1, 0),
        (Cache::MAX_VALUES + 1, 1),
        (1, Cache::MAX_ENTRIES + 1),
    ] {
        assert!(Cache::new(&base, v, e).is_err());
    }
}

#[test]
fn earliest_attached_site_keys_the_depth_even_when_another_arm_trains_later() {
    let mut config = config();
    config["n_layer"] = json!(3);
    let mut weights = parameters();
    let extra: Vec<_> = weights
        .iter()
        .filter(|(k, _)| k.starts_with("h.1."))
        .map(|(k, v)| (k.replacen("h.1.", "h.2.", 1), v.clone()))
        .collect();
    weights.extend(extra);
    let base = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config).unwrap(),
        &archive(&weights),
        Default::default(),
    )
    .unwrap();
    let late = base.with_lora(2, 2, 3.).unwrap();
    let mut geo = geometry(&base).record().clone();
    geo.placements[0].path = "transformer.h.1.mlp".into();
    let earlier = base
        .with_geometry(GeometryAdapterArtifact::create(geo).unwrap())
        .unwrap();
    let template = session(late.clone(), vec![vec![1, 2, 3, 4]], None, 4, 1);
    let mut cache = Cache::new(&base, 256, 8).unwrap();
    for (m, mode, depth, expected) in [
        (late.clone(), Lora, 2, Inserted),
        (earlier, CausalLmAdaptationMode::Geometry, 1, Inserted),
        (late, Lora, 2, Hit),
    ] {
        let mut a = template.clone();
        let mut b = a.fork_variant(m, mode).unwrap();
        let mut refs = [a.clone(), b.clone()];
        let r = Session::step_matched_with_prefix_cache(&mut [&mut a, &mut b], &mut cache, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            (r.shared_prefix_layers, r.prefix_cache_action),
            (depth, Some(expected))
        );
        for ((s, reference), step) in [a, b].into_iter().zip(&mut refs).zip(r.steps) {
            let normal = reference.step_with_observation(true).unwrap().unwrap();
            assert_eq!(
                serde_json::to_vec(&step).unwrap(),
                serde_json::to_vec(&normal).unwrap()
            );
            assert_eq!(
                serde_json::to_vec(&step.observation).unwrap(),
                serde_json::to_vec(&normal.observation).unwrap()
            );
            assert_eq!(
                s.checkpoint_with_id().unwrap(),
                reference.checkpoint_with_id().unwrap()
            );
        }
    }
    assert_eq!(cache.stats().skipped_block_calls, 2);
}

#[test]
fn a_late_failure_rolls_back_every_cursor_and_cache_even_after_a_hit() {
    let base = load();
    let lora = base.with_lora(1, 2, 3.).unwrap();
    let a = session(lora.clone(), vec![vec![1, 2, 3, 4]], None, 4, 1);
    let mut values: serde_json::Value =
        serde_json::from_str(&lora.lora_to_json().unwrap()).unwrap();
    for name in ["a", "b"] {
        values[name] = json!(vec![f32::MAX; 8]);
    }
    let bad = a
        .fork_variant(base.with_lora_json(&values.to_string()).unwrap(), Lora)
        .unwrap();
    let mut cache = Cache::new(&base, 128, 4).unwrap();
    for hot in [false, true] {
        if hot {
            check_one(&mut a.clone(), &mut cache, Inserted);
        }
        let (mut x, mut y) = (a.clone(), bad.clone());
        let before = cache.stats();
        assert!(
            Session::step_matched_with_prefix_cache(&mut [&mut x, &mut y], &mut cache, false)
                .is_err()
        );
        assert_eq!(cache.stats(), before);
        assert_eq!(
            x.checkpoint_with_id().unwrap(),
            a.checkpoint_with_id().unwrap()
        );
        assert_eq!(
            y.checkpoint_with_id().unwrap(),
            bad.checkpoint_with_id().unwrap()
        );
    }
    let mut w = parameters();
    w.get_mut("wte.weight").unwrap().1[0] += 1.;
    let other = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&w),
        Default::default(),
    )
    .unwrap();
    for model in [
        other,
        base.with_readout_accumulator(CausalLmReadoutAccumulator::F64),
    ] {
        let before = cache.stats();
        let mut wrong = session(
            model.with_lora(1, 2, 3.).unwrap(),
            vec![vec![1, 2, 3, 4]],
            None,
            4,
            1,
        );
        assert!(wrong.step_with_prefix_cache(&mut cache, false).is_err());
        assert_eq!(wrong.completed_updates(), 0);
        assert_eq!(cache.stats(), before);
    }
}

#[test]
fn hit_removes_real_prefix_qkv_calls_not_suffix_or_backward_replay() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let base = load();
    let lora = base.with_lora(1, 2, 3.).unwrap();
    let mut record = geometry(&base).record().clone();
    record.placements[0].path = "transformer.h.1.attn".into();
    let both = lora
        .with_geometry(GeometryAdapterArtifact::create(record).unwrap())
        .unwrap();
    let mut a = session(lora, vec![vec![1, 2, 3, 4]], None, 4, 1);
    let mut b = a.fork_variant(both, Joint).unwrap();
    let mut cache = Cache::new(&base, 128, 8).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    let old = st_tensor::set_thread_meta_observer(Some(Arc::new(move |e| {
        if e.op_name == "matmul_prepacked_bias" && e.data["inner"] == 4 && e.data["cols"] == 12 {
            seen.fetch_add(1, Ordering::Relaxed);
        }
    })));
    let mut calls = vec![];
    for _ in 0..2 {
        Session::step_matched_with_prefix_cache(&mut [&mut a, &mut b], &mut cache, true)
            .unwrap()
            .unwrap();
        calls.push(count.swap(0, Ordering::Relaxed));
    }
    st_tensor::set_thread_meta_observer(old);
    assert_eq!(calls, vec![5, 4]);
    assert_eq!(
        (cache.stats().hits, cache.stats().skipped_block_calls),
        (1, 1)
    );
}
