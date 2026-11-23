use super::*;
use CausalLmAdaptationMode::{Geometry, Joint, Lora};

const PROMPT: [usize; 3] = [2, 5, 1];
const SITES: [&str; 6] = [
    "transformer.h.0.attn",
    "transformer.h.0.mlp",
    "transformer.h.0",
    "transformer.h.1.attn",
    "transformer.h.1.mlp",
    "transformer.h.1",
];

fn source() -> NativeCausalLm {
    use crate::geometry_artifact::{GeometryAdapterRecipe, WaveRecipe};
    let m = model(1, CausalLmReadoutAccumulator::F64);
    let mut r = m.geometry_artifact().unwrap().record().clone();
    r.placements[0].recipe = GeometryAdapterRecipe::Wave {
        features: 4,
        strength: 0.5,
        kernel: WaveRecipe {
            curvature: -1.,
            saturation: 1.,
            porosity: 0.05,
            max_values: 1024,
        },
    };
    let mut l: serde_json::Value = serde_json::from_str(&m.lora_to_json().unwrap()).unwrap();
    l["alpha"] = json!(32.);
    m.with_geometry(GeometryAdapterArtifact::create(r).unwrap())
        .unwrap()
        .with_lora_json(&l.to_string())
        .unwrap()
}

fn logits(m: &NativeCausalLm) -> Vec<f32> {
    m.logits(&PROMPT, [1, 3], false).unwrap().data().to_vec()
}

#[test]
fn frozen_lora_preserves_both_forward_routes_and_changes_only_the_owner_contract() {
    let m = source();
    let base = load().with_readout_accumulator(m.readout_accumulator());
    let weights = m.lora_to_json().unwrap();
    let raw = base
        .with_lora_json_pinned(&weights, m.lora_artifact_id().unwrap())
        .unwrap();
    let f = m.freeze_lora().unwrap();
    assert_ne!(logits(&base), logits(&raw), "zero-effect fixture");
    assert_eq!(logits(&m), logits(&f));
    assert_eq!(logits(&raw), logits(&raw.freeze_lora().unwrap()));
    let cached = |model: &NativeCausalLm| {
        NativeCausalLmContext::new(model.clone(), PROMPT[..2].to_vec(), 1024)
            .unwrap()
            .with_suffix(&PROMPT[2..])
            .unwrap()
            .logit_values()
            .to_vec()
    };
    assert_eq!(cached(&m), cached(&f));
    assert_ne!(f.base_model_identity(), base.base_model_identity());
    assert_eq!(
        f.base_model_identity(),
        raw.freeze_lora().unwrap().base_model_identity()
    );
    assert_ne!(f.geometry_artifact_id(), m.geometry_artifact_id());
    assert_eq!(f.frozen_lora_to_json().unwrap(), weights);
    assert_eq!(f.frozen_lora_artifact_id(), m.lora_artifact_id());
    assert!(f.lora_artifact_id().is_none());
    assert!(m.frozen_lora_artifact_id().is_none());
    assert!(f.lora_to_json().is_err());
    let mut record = f.geometry_artifact().unwrap().record().clone();
    record.base_model_identity = m.base_model_identity().into();
    assert_eq!(
        GeometryAdapterArtifact::create(record)
            .unwrap()
            .artifact_id(),
        m.geometry_artifact_id().unwrap()
    );
    assert!(base.freeze_lora().is_err());
    assert!(f.freeze_lora().is_err());
    assert!(f.with_lora(0, 1, 1.).is_err());
    assert!(f.with_lora_json(&weights).is_err());
    assert!(raw
        .freeze_lora()
        .unwrap()
        .with_key_lag(CausalLmKeyLag { layer: 0, lag: 1 })
        .is_err());
    assert!(f
        .with_geometry(m.geometry_artifact().unwrap().clone())
        .is_err());
    for mode in [Lora, Joint] {
        assert!(f.validate_adaptation_mode(mode).is_err());
    }
    let grad = |m: &NativeCausalLm| {
        CausalLmLineObjective::Set {}
            .gradients_for(m, &PROMPT, &[vec![3], vec![3, 4]], 0, Some(11), Geometry)
            .unwrap()
    };
    let original = grad(&m);
    let pinned = grad(&f);
    assert_eq!(original.loss(), pinned.loss());
    assert_eq!(original.parameters(), pinned.parameters());
    assert!(f.step_adaptation_sgd(&original, Geometry, 0.1).is_err());
    assert!(m.step_adaptation_sgd(&pinned, Geometry, 0.1).is_err());
}

#[test]
fn six_wave_sites_have_real_parameter_vjps_with_one_fixed_nonzero_lora() {
    let m = source();
    let f = m.freeze_lora().unwrap();
    let objective = CausalLmLineObjective::Set {};
    let lines = vec![vec![3], vec![3, 4]];
    // Independent full-vocabulary teacher forcing of each complete line, including stop.
    let loss = |m: &NativeCausalLm| {
        let mut mass = 0.;
        for line in &lines {
            let mut prefix = PROMPT.to_vec();
            let mut nll = 0.;
            for token in line.iter().copied().chain([0]) {
                nll += m.next_token_score(&prefix, token).unwrap().nll;
                prefix.push(token);
            }
            mass += (-nll).exp();
        }
        -mass.ln()
    };
    for path in SITES {
        let placed = f.with_single_geometry_at(path).unwrap();
        assert_eq!(placed.frozen_lora_artifact_id(), m.lora_artifact_id());
        assert_eq!(
            placed
                .with_single_geometry_at(SITES[3])
                .unwrap()
                .geometry_artifact_id(),
            f.geometry_artifact_id()
        );
        let g = objective
            .gradients_for(&placed, &PROMPT, &lines, 0, Some(11), Geometry)
            .unwrap();
        assert!((loss(&placed) - f64::from(g.loss())).abs() < 1e-6);
        assert_eq!(g.parameters().len(), 1);
        assert_eq!(g.parameters()[path].len(), 2);
        for (name, values) in &g.parameters()[path] {
            let (i, &expected) = values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap();
            assert!(expected.abs() > 1e-7, "{path}/{name} is not observable");
            let variant = |dx| {
                let mut r = placed.geometry_artifact().unwrap().record().clone();
                r.placements[0]
                    .tensors
                    .iter_mut()
                    .find(|t| t.name == *name)
                    .unwrap()
                    .values[i] += dx;
                placed
                    .with_geometry(GeometryAdapterArtifact::create(r).unwrap())
                    .unwrap()
            };
            let h = 0.001_f32;
            let fd = (loss(&variant(h)) - loss(&variant(-h))) / (2. * f64::from(h));
            assert!(
                (fd - f64::from(expected)).abs() < 1.5e-4 + 0.005 * fd.abs(),
                "{path}/{name}[{i}]: {expected} vs {fd}"
            );
        }
        let next = placed.step_adaptation_sgd(&g, Geometry, 0.02).unwrap();
        assert_ne!(next.geometry_artifact_id(), placed.geometry_artifact_id());
        assert_eq!(
            next.frozen_lora_to_json().unwrap(),
            f.frozen_lora_to_json().unwrap()
        );
    }
}

#[test]
fn relocation_rejects_ambiguous_and_foreign_sites_without_unlocking_joint_learning() {
    let active = source();
    let m = active.freeze_lora().unwrap();
    for path in [
        "",
        "mlp",
        "transformer.h.01",
        "transformer.h.10.attn",
        "transformer.h.2",
        "transformer.h.1.attn.extra",
    ] {
        assert!(m.with_single_geometry_at(path).is_err());
    }
    assert!(m.with_single_geometry_at(&"a".repeat(65)).is_err());
    assert!(active.with_single_geometry_at(SITES[0]).is_err());
    assert!(load().with_single_geometry_at(SITES[0]).is_err());
    let mut r = m.geometry_artifact().unwrap().record().clone();
    let mut second = r.placements[0].clone();
    second.path = SITES[0].into();
    r.placements.push(second);
    let multi = m
        .with_geometry(GeometryAdapterArtifact::create(r).unwrap())
        .unwrap();
    assert!(multi.with_single_geometry_at(SITES[3]).is_err());
    assert!(m.with_single_geometry_at(SITES[3]).is_ok());
}

#[test]
fn upstream_wave_gradient_agrees_with_a_dense_oracle_not_a_missing_low_rank_input_vjp() {
    let m = source();
    let split = m
        .freeze_lora()
        .unwrap()
        .with_single_geometry_at(SITES[2])
        .unwrap();
    let r: serde_json::Value = serde_json::from_str(&m.lora_to_json().unwrap()).unwrap();
    let mut w = parameters();
    let (shape, values) = w.get_mut("h.1.attn.c_proj.weight").unwrap();
    assert_eq!(shape.as_slice(), [4, 4]);
    for i in 0..4 {
        for j in 0..4 {
            let ab: f64 = (0..2)
                .map(|k| r["a"][2 * i + k].as_f64().unwrap() * r["b"][4 * k + j].as_f64().unwrap())
                .sum();
            values[i * 4 + j] =
                (f64::from(values[i * 4 + j]) + ab * r["alpha"].as_f64().unwrap() / 2.) as f32;
        }
    }
    // This independently collapsed dense projection is an oracle, not the deployed arithmetic.
    let dense = NativeCausalLm::from_gpt2_json(
        &serde_json::to_vec(&config()).unwrap(),
        &archive(&w),
        Default::default(),
    )
    .unwrap()
    .with_readout_accumulator(m.readout_accumulator());
    let mut recipe = split.geometry_artifact().unwrap().record().clone();
    recipe.base_model_identity = dense.base_id.clone();
    let dense = dense
        .with_geometry(GeometryAdapterArtifact::create(recipe).unwrap())
        .unwrap();
    let max_error = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f32::max)
    };
    assert!(max_error(&logits(&split), &logits(&dense)) < 2e-6);
    // Negative control: same forward, but only the original (unadapted) c_proj transpose on reverse.
    let mut missing = dense.clone();
    missing.blocks[1].output.backward_weight = Arc::new(OnceLock::from(Ok(load().blocks[1]
        .output
        .weight
        .transposed()
        .unwrap())));
    assert_eq!(logits(&missing), logits(&dense));
    let grads = |m: &NativeCausalLm| {
        m.geometry_loss_gradients(&[2, 5, 1, 3], [1, 4])
            .unwrap()
            .parameters()[SITES[2]]
            .values()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
    };
    let actual = grads(&split);
    let correct = max_error(&actual, &grads(&dense));
    let omitted = max_error(&actual, &grads(&missing));
    assert!(
        correct < 2e-6 && omitted > 1e-4,
        "dense oracle: {correct}, omitted input VJP: {omitted}"
    );
}

#[test]
fn line_checkpoint_reconstructs_the_exact_frozen_base_and_refuses_other_lora_or_site() {
    type Session = NativeCausalLmLineTrainingSession;
    let m = source();
    let f = m.freeze_lora().unwrap();
    let placed = f.with_single_geometry_at(SITES[0]).unwrap();
    let plan = br#"{"schema":"spiraltorch.native_causal_lm_line_plan.v2","mode":"geometry","learning_rate":0.02,"epochs":2,"max_tile_values":11,"examples":[{"prompt_ids":[2,5,1],"candidates":[[3],[3,4]],"stop_id":0,"objective":{"kind":"set"}}]}"#;
    let mut s = Session::new(placed.clone(), plan).unwrap();
    s.step_adaptation().unwrap().unwrap();
    let (checkpoint, pin) = s.checkpoint_with_id().unwrap();
    let payload: serde_json::Value = serde_json::from_str(&checkpoint).unwrap();
    assert_eq!(payload["base_id"], placed.base_id);
    assert_eq!(
        payload["lora"],
        json!({"origin_id":null,"id":null,"weights":null})
    );
    let restore = |source| Session::restore(source, plan, checkpoint.as_bytes(), &pin);
    let base = load().with_readout_accumulator(m.readout_accumulator());
    let rebuilt = base
        .with_lora_json_pinned(
            &f.frozen_lora_to_json().unwrap(),
            f.frozen_lora_artifact_id().unwrap(),
        )
        .unwrap()
        .freeze_lora()
        .unwrap()
        .with_geometry(placed.geometry_artifact().unwrap().clone())
        .unwrap();
    let mut other = restore(rebuilt).unwrap();
    assert_eq!(
        s.checkpoint_with_id().unwrap(),
        other.checkpoint_with_id().unwrap()
    );
    assert!(
        restore(f.clone()).is_err(),
        "same derived base, wrong geometry"
    );
    assert!(restore(m).is_err());
    let mut raw: serde_json::Value =
        serde_json::from_str(&f.frozen_lora_to_json().unwrap()).unwrap();
    raw["alpha"] = json!(16.);
    let other_base = base
        .with_lora_json(&raw.to_string())
        .unwrap()
        .freeze_lora()
        .unwrap();
    assert_ne!(other_base.base_id, placed.base_id);
    assert!(other_base
        .with_geometry(placed.geometry_artifact().unwrap().clone())
        .is_err());
    assert!(restore(
        other_base
            .with_geometry(geometry(&other_base, 0, 0.2))
            .unwrap()
    )
    .is_err());
    let a = serde_json::to_vec(&s.step_adaptation().unwrap().unwrap()).unwrap();
    let b = serde_json::to_vec(&other.step_adaptation().unwrap().unwrap()).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        s.checkpoint_with_id().unwrap(),
        other.checkpoint_with_id().unwrap()
    );
    assert_eq!(
        s.model().frozen_lora_artifact_id(),
        f.frozen_lora_artifact_id()
    );
    assert_eq!(logits(s.model()), logits(other.model()));
}
