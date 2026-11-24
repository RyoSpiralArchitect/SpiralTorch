use super::*;

#[test]
fn declared_set_components_share_decoder_work_and_reconstruct_ordinary_geometry_gradient() {
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    let labels = [-100, -100, -100, 4, -100, -100, 7, -100];
    let plan = [DeclaredRivalRow {
        position: 6,
        target: 7,
        rivals: vec![1, 4],
    }];
    let shape = [2, 4];
    for policy in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F32WideBackward,
        CausalLmReadoutAccumulator::F64,
    ] {
        let m = model(1, policy);
        let scores = m
            .declared_set_nll(&ids, shape, Some(&labels), &plan)
            .unwrap();
        decoder::work_counts::take();
        let (a, b, s) = m
            .geometry_declared_set_gradients(&ids, shape, Some(&labels), &plan)
            .unwrap();
        assert_eq!(decoder::work_counts::take(), [2, 1, 1]);
        assert_eq!(scores, s);
        assert_eq!((s.target_tokens, s.contrasted_tokens), (2, 1));
        let (f, c) = m
            .geometry_full_and_declared_gradients(&ids, shape, Some(&labels), &plan, 1.)
            .unwrap();
        assert_eq!(s.full_nll as f32, f.loss());
        assert_eq!(
            (a.loss(), b.loss()),
            (s.within_nll as f32, s.coverage_nll as f32)
        );
        assert_eq!(a.parameters(), c.parameters());
        for g in [&a, &b] {
            assert_eq!(g.geometry_artifact_id(), m.geometry_artifact_id());
            assert_eq!(g.lora_artifact_id(), m.lora_artifact_id());
            assert_eq!(g.readout_accumulator(), policy);
        }
        let mut diff = 0.;
        let mut norm = 0.;
        for (p, tensors) in f.parameters() {
            for (name, v) in tensors {
                for ((&full, &within), &mass) in v
                    .iter()
                    .zip(&a.parameters()[p][name])
                    .zip(&b.parameters()[p][name])
                {
                    diff += (f64::from(full) - f64::from(within) - f64::from(mass)).powi(2);
                    norm += f64::from(full).powi(2);
                }
            }
        }
        assert!(
            norm > 0. && (diff / norm).sqrt() < 1e-4,
            "gradient closure: {diff} / {norm}"
        );
        let (new, _) = m.step_geometry_gradient_at_distance(&b, 1. / 256.).unwrap();
        assert_eq!(new.lora_artifact_id(), m.lora_artifact_id());
        assert!(new
            .step_geometry_gradient_at_distance(&a, 1. / 256.)
            .is_err());
    }
    let bare = load();
    assert!(bare
        .declared_set_nll(&ids, shape, Some(&labels), &plan)
        .is_ok());
    assert!(bare
        .geometry_declared_set_gradients(&ids, shape, Some(&labels), &plan)
        .is_err());
    let m = model(1, CausalLmReadoutAccumulator::F32);
    decoder::work_counts::take();
    for (position, target) in [(0, 1), (1, 2), (6, 8)] {
        let bad = [DeclaredRivalRow {
            position,
            target,
            rivals: vec![0],
        }];
        assert!(m
            .declared_set_nll(&ids, shape, Some(&labels), &bad)
            .is_err());
        assert!(m
            .geometry_declared_set_gradients(&ids, shape, Some(&labels), &bad)
            .is_err());
    }
    assert_eq!(decoder::work_counts::take(), [0; 3]);
}

#[test]
fn finite_declared_set_step_checks_one_candidate_and_never_advances_another_owner() {
    let m = model(1, CausalLmReadoutAccumulator::F32WideBackward);
    let (ids, shape) = ([1, 2, 3, 4], [1, 4]);
    let labels = [-100, -100, -100, 4];
    let plan = [DeclaredRivalRow {
        position: 3,
        target: 4,
        rivals: vec![2],
    }];
    decoder::work_counts::take();
    assert!(m
        .step_declared_set_geometry_at_distance(&ids, shape, Some(&labels), &plan, 0.)
        .is_err());
    assert!(m
        .step_declared_set_geometry_at_distance(&ids, shape, Some(&labels), &[], 0.01)
        .is_err());
    assert_eq!(decoder::work_counts::take(), [0; 3]);
    let (next, r) = m
        .step_declared_set_geometry_at_distance(&ids, shape, Some(&labels), &plan, 1. / 256.)
        .unwrap();
    assert_eq!(decoder::work_counts::take(), [3, 1, 2]);
    assert_eq!(
        r.accepted,
        r.proposed.within_nll < r.before.within_nll
            && r.proposed.coverage_nll < r.before.coverage_nll
    );
    assert_eq!(r.result_geometry_id, next.geometry_artifact_id().unwrap());
    assert_eq!(m.lora_artifact_id(), next.lora_artifact_id());
    assert_eq!(
        m.declared_set_nll(&ids, shape, Some(&labels), &plan)
            .unwrap(),
        r.before
    );
    assert_eq!(
        next.declared_set_nll(&ids, shape, Some(&labels), &plan)
            .unwrap(),
        if r.accepted { r.proposed } else { r.before }
    );
}

#[test]
fn checked_suffix_matches_full_candidate_redecode_across_sites_and_sparse_labels() {
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    let shape = [2, 4];
    let labels = [-100, -100, 3, -100, -100, -100, -100, 8];
    let plan = [
        DeclaredRivalRow {
            position: 2,
            target: 3,
            rivals: vec![1, 7],
        },
        DeclaredRivalRow {
            position: 7,
            target: 8,
            rivals: vec![4],
        },
    ];
    use CausalLmReadoutAccumulator::{F32WideBackward, F32, F64};
    for p in [F32, F32WideBackward, F64] {
        let lagged = load()
            .with_key_lag(CausalLmKeyLag { layer: 0, lag: 1 })
            .unwrap()
            .with_readout_accumulator(p);
        let late = geometry(&lagged, 1, 0.25);
        let mut multi = late.record().clone();
        let mut early = multi.placements[0].clone();
        early.path = "transformer.h.0.mlp".into();
        multi.placements.push(early);
        for (m, skipped) in [
            (model(0, p), 0),
            (model(1, p), 1),
            (lagged.with_geometry(late).unwrap(), 1),
            (
                lagged
                    .with_geometry(GeometryAdapterArtifact::create(multi).unwrap())
                    .unwrap(),
                0,
            ),
        ] {
            for (storage, mask) in [
                (
                    CausalLmForwardStorage::OriginalPacked,
                    Some(labels.as_slice()),
                ),
                (CausalLmForwardStorage::PreferCachedTranspose, None),
            ] {
                let m = m.with_forward_readout_storage(storage);
                let (w, c, s) = m
                    .geometry_declared_set_gradients(&ids, shape, mask, &plan)
                    .unwrap();
                let (candidate, _) = m.step_geometry_gradient_at_distance(&w, 1. / 256.).unwrap();
                let (safe, g) = m
                    .guard_geometry_step(&candidate, &[&[&w], &[&c]], 1e-5)
                    .unwrap();
                let expected = safe.declared_set_nll(&ids, shape, mask, &plan).unwrap();
                decoder::work_counts::take();
                let (next, r) = m
                    .step_declared_set_geometry_at_distance(&ids, shape, mask, &plan, 1. / 256.)
                    .unwrap();
                assert_eq!(decoder::work_counts::take(), [4 - skipped, 2 - skipped, 2]);
                assert_eq!(
                    serde_json::to_vec(&r.proposal_guard).unwrap(),
                    serde_json::to_vec(&g).unwrap()
                );
                for (a, b) in [(r.before, s), (r.proposed, expected)] {
                    assert_eq!(
                        (a.target_tokens, a.contrasted_tokens),
                        (b.target_tokens, b.contrasted_tokens)
                    );
                    assert_eq!(a.within_nll.to_bits(), b.within_nll.to_bits());
                    assert_eq!(a.coverage_nll.to_bits(), b.coverage_nll.to_bits());
                    assert_eq!(a.full_nll.to_bits(), b.full_nll.to_bits());
                }
                let accepted =
                    expected.within_nll < s.within_nll && expected.coverage_nll < s.coverage_nll;
                assert_eq!(r.accepted, accepted);
                assert_eq!(
                    next.geometry_artifact_id(),
                    if accepted {
                        safe.geometry_artifact_id()
                    } else {
                        m.geometry_artifact_id()
                    }
                );
            }
        }
    }
}

#[test]
fn recheck_refuses_a_foreign_prefix_before_scoring() {
    let base = load();
    let late = base.with_geometry(geometry(&base, 1, 0.2)).unwrap();
    let different_base = base
        .with_key_lag(CausalLmKeyLag { layer: 0, lag: 1 })
        .unwrap();
    for foreign in [
        base.with_geometry(geometry(&base, 0, 0.2)).unwrap(),
        different_base,
        late.with_lora(1, 2, 3.).unwrap(),
        late.with_readout_accumulator(CausalLmReadoutAccumulator::F64),
        late.with_forward_readout_storage(CausalLmForwardStorage::OriginalPacked),
    ] {
        let plan = [DeclaredRivalRow {
            position: 2,
            target: 3,
            rivals: vec![1],
        }];
        let batch = late
            .declared_set_training(&[1, 2, 3], [1, 3], None, &plan)
            .unwrap();
        decoder::work_counts::take();
        assert!(batch.score_candidate(&foreign).is_err());
        assert_eq!(decoder::work_counts::take(), [0; 3]);
    }
}

#[test]
fn twelve_block_overshoot_has_two_negative_slopes_but_returns_the_origin() {
    let mut cfg = config();
    cfg["n_layer"] = json!(12);
    cfg["vocab_size"] = json!(4);
    cfg["tie_word_embeddings"] = json!(false);
    let mut weights = parameters();
    for (_, values) in weights.values_mut() {
        values.fill(0.);
    }
    let block: Vec<_> = weights
        .iter()
        .filter_map(|(k, v)| Some((k.strip_prefix("h.0.")?.to_owned(), v.clone())))
        .collect();
    for i in 2..12 {
        for (k, v) in &block {
            weights.insert(format!("h.{i}.{k}"), v.clone());
        }
    }
    weights.insert("wte.weight".into(), (vec![4, 4], vec![0.; 16]));
    weights.get_mut("ln_f.weight").unwrap().1.fill(1.);
    // At small distances the mean outside logit falls. At a large distance
    // the -4 outsider wins, despite two negative initial directional derivatives.
    weights.insert(
        "lm_head.weight".into(),
        (
            vec![4, 4],
            vec![
                0., 0., 0., 0., 1., -1., 0., 0., -4., 4., 0., 0., 6., -6., 0., 0.,
            ],
        ),
    );
    let base = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&cfg).unwrap(),
        &archive(&weights),
        Default::default(),
    )
    .unwrap();
    let (ids, shape) = ([2, 0], [1, 2]);
    let plan = [DeclaredRivalRow {
        position: 1,
        target: 0,
        rivals: vec![1],
    }];
    for site in [0, 11] {
        let mut g = geometry(&base, site, 0.).record().clone();
        g.placements[0].path = format!("transformer.h.{site}");
        for t in &mut g.placements[0].tensors {
            t.requires_grad = t.name == "bias";
        }
        let m = base
            .with_geometry(GeometryAdapterArtifact::create(g).unwrap())
            .unwrap();
        for (distance, should_accept) in [(1. / 65536., true), (1. / 128., false)] {
            decoder::work_counts::take();
            let (a, b, start) = m
                .geometry_declared_set_gradients(&ids, shape, None, &plan)
                .unwrap();
            let (candidate, _) = m.step_geometry_gradient_at_distance(&a, distance).unwrap();
            let (safe, _) = m
                .guard_geometry_step(&candidate, &[&[&a], &[&b]], 1e-5)
                .unwrap();
            let full = safe.declared_set_nll(&ids, shape, None, &plan).unwrap();
            assert_eq!(decoder::work_counts::take(), [24, 12 - site, 2]);
            let (returned, r) = m
                .step_declared_set_geometry_at_distance(&ids, shape, None, &plan, distance)
                .unwrap();
            assert_eq!(decoder::work_counts::take(), [24 - site, 12 - site, 2]);
            assert_eq!(r.before, start);
            assert_eq!(r.proposed, full);
            assert!(r
                .proposal_guard
                .applied_loss_dots
                .iter()
                .all(|slope| *slope < 0.));
            assert!(r.proposed.within_nll < start.within_nll);
            assert_eq!(r.proposed.coverage_nll < start.coverage_nll, should_accept);
            assert_eq!(r.accepted, should_accept);
            assert_eq!(
                returned.geometry_artifact_id(),
                if should_accept {
                    safe.geometry_artifact_id()
                } else {
                    m.geometry_artifact_id()
                }
            );
        }
    }
}
