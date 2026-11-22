use super::*;
use CausalLmAdaptationMode::{Geometry, Joint, Lora};

fn objectives() -> [CausalLmLineObjective; 3] {
    use CausalLmLineObjective::*;
    [
        Set {},
        Menu {
            gold_index: 1,
            mix: 0.5,
        },
        Subset {
            accepted_indices: vec![0, 2],
        },
    ]
}

fn lines() -> Vec<Vec<usize>> {
    vec![vec![2], vec![2, 4], vec![2, 4, 7], vec![2, 5]]
}

#[test]
fn joint_is_one_decoder_traversal_and_the_exact_union_with_and_without_head_tiling() {
    for precision in [
        CausalLmReadoutAccumulator::F32,
        CausalLmReadoutAccumulator::F32WideBackward,
        CausalLmReadoutAccumulator::F64,
    ] {
        let m = model(1, precision);
        for spec in objectives() {
            let lines = lines();
            let calc = |mode, cap| {
                decoder::work_counts::take();
                let g = spec
                    .gradients_for(&m, &[1, 3], &lines, 0, cap, mode)
                    .unwrap();
                (g, decoder::work_counts::take())
            };
            let (full, full_work) = calc(Joint, None);
            for cap in [None, Some(11), Some(22)] {
                let (joint, jw) = calc(Joint, cap);
                let (l, lw) = calc(Lora, cap);
                let (g, gw) = calc(Geometry, cap);
                assert_eq!(jw, lw);
                assert_eq!(jw, gw);
                assert_eq!(jw[..2], full_work[..2]);
                assert!(jw[0] > 0 && jw[1] > 0);
                assert_eq!(joint.loss(), full.loss());
                assert_eq!((joint.loss(), joint.loss()), (l.loss(), g.loss()));
                assert_eq!(joint.parameters(), full.parameters());
                assert_eq!(joint.parameters().len(), 2);
                assert_eq!((l.parameters().len(), g.parameters().len()), (1, 1));
                for (key, values) in l.parameters().iter().chain(g.parameters()) {
                    assert_eq!(joint.parameters().get(key), Some(values));
                }
                assert_ne!(l.parameters().keys().next(), g.parameters().keys().next());
                let stepped = m.step_adaptation_sgd(&joint, Joint, 0.25).unwrap();
                let sl = m.step_adaptation_sgd(&l, Lora, 0.25).unwrap();
                let sg = m.step_adaptation_sgd(&g, Geometry, 0.25).unwrap();
                assert_eq!(sl.geometry_artifact_id(), m.geometry_artifact_id());
                assert_eq!(sg.lora_artifact_id(), m.lora_artifact_id());
                assert_eq!(sl.lora_artifact_id(), stepped.lora_artifact_id());
                assert_eq!(sg.geometry_artifact_id(), stepped.geometry_artifact_id());
                assert_ne!(stepped.geometry_artifact_id(), m.geometry_artifact_id());
                assert_ne!(stepped.lora_artifact_id(), m.lora_artifact_id());
                assert!(stepped.step_adaptation_sgd(&joint, Joint, 0.25).is_err());
            }
        }
    }
}

// Independent scalar oracle: walk each full vocabulary prefix to the terminal,
// rather than using the line trie, its partition, or the autodiff head seed.
fn reference(m: &NativeCausalLm, lines: &[Vec<usize>], spec: &CausalLmLineObjective) -> f64 {
    let nll: Vec<_> = lines
        .iter()
        .map(|line| {
            let mut prompt = vec![1, 3];
            line.iter()
                .copied()
                .chain([0])
                .map(|t| {
                    let score = m.next_token_score(&prompt, t).unwrap().nll;
                    prompt.push(t);
                    score
                })
                .sum::<f64>()
        })
        .collect();
    let mass: f64 = nll.iter().map(|x| (-x).exp()).sum();
    match spec {
        CausalLmLineObjective::Set {} => -mass.ln(),
        CausalLmLineObjective::Menu { gold_index: i, mix } => {
            (1. - mix) * nll[*i] / (lines[*i].len() + 1) as f64 + mix * (nll[*i] + mass.ln())
        }
        CausalLmLineObjective::Subset { accepted_indices } => {
            mass.ln()
                - accepted_indices
                    .iter()
                    .map(|&i| (-nll[i]).exp())
                    .sum::<f64>()
                    .ln()
        }
    }
}

#[test]
fn joint_lora_a_b_and_geometry_coordinates_match_independent_terminated_sequence_differences() {
    let m = model(1, CausalLmReadoutAccumulator::F64);
    let lines = lines();
    for spec in objectives() {
        let g = spec
            .gradients_for(&m, &[1, 3], &lines, 0, Some(11), Joint)
            .unwrap();
        assert!((f64::from(g.loss()) - reference(&m, &lines, &spec)).abs() < 1e-6);
        let mut blocks = 0;
        for (path, params) in g.parameters() {
            for (name, values) in params {
                let (i, &expected) = values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                    .unwrap();
                assert!(
                    expected.abs() > 1e-7,
                    "{spec:?} {path} {name}: uninformative derivative"
                );
                let perturb = |d: f32| {
                    if path.ends_with("c_proj") {
                        let mut r: serde_json::Value =
                            serde_json::from_str(&m.lora_to_json().unwrap()).unwrap();
                        r[name][i] = json!(r[name][i].as_f64().unwrap() as f32 + d);
                        m.with_lora_json(&r.to_string()).unwrap()
                    } else {
                        let mut r = m.geometry_artifact().unwrap().record().clone();
                        r.placements[0]
                            .tensors
                            .iter_mut()
                            .find(|t| t.name == *name)
                            .unwrap()
                            .values[i] += d;
                        m.with_geometry(GeometryAdapterArtifact::create(r).unwrap())
                            .unwrap()
                    }
                };
                let eps = 0.002f32;
                let fd = (reference(&perturb(eps), &lines, &spec)
                    - reference(&perturb(-eps), &lines, &spec))
                    / (2. * f64::from(eps));
                assert!(
                    (fd - f64::from(expected)).abs() < 1.5e-4,
                    "{spec:?} {path}/{name}[{i}]: {expected} != {fd}"
                );
                blocks += 1;
            }
        }
        assert_eq!(blocks, 4);
    }
}

#[test]
fn eligibility_is_checked_without_decoding_and_a_frozen_geometry_still_transports_lora_gradients() {
    let both = model(1, CausalLmReadoutAccumulator::F32);
    let plain = load()
        .with_lora_json(&both.lora_to_json().unwrap())
        .unwrap();
    let mut r = both.geometry_artifact().unwrap().record().clone();
    for t in &mut r.placements[0].tensors {
        t.requires_grad = false;
    }
    let frozen = plain
        .with_geometry(GeometryAdapterArtifact::create(r).unwrap())
        .unwrap();
    for spec in objectives() {
        decoder::work_counts::take();
        for mode in [Geometry, Joint] {
            for m in [&plain, &frozen] {
                assert!(spec
                    .gradients_for(m, &[1, 3], &lines(), 0, Some(11), mode)
                    .is_err());
            }
        }
        assert_eq!(decoder::work_counts::take(), [0; 3]);
        let calc = |m| {
            spec.gradients_for(m, &[1, 3], &lines(), 0, Some(11), Lora)
                .unwrap()
        };
        let g = calc(&frozen);
        assert_eq!(g.parameters(), calc(&both).parameters());
        assert_ne!(g.parameters(), calc(&plain).parameters());
        let n = frozen.step_adaptation_sgd(&g, Lora, 0.25).unwrap();
        assert_eq!(n.geometry_artifact_id(), frozen.geometry_artifact_id());
        assert_ne!(n.lora_artifact_id(), frozen.lora_artifact_id());
    }
}
