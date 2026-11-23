use super::*;

#[test]
fn ranking_separates_cap_from_ceiling_and_never_promotes_a_zero_row() {
    let cfg = Settings {
        target: 1.,
        cap: 2.,
        h: 1.,
        ceiling: 4.,
        basis: CausalLmSensitivityBasis::Coordinate,
        select: CausalLmSensitivitySelect::Envelope,
    };
    let axes: [_; 4] = [
        (20., [0., 0., 0.]),
        (10., [0., 0., 1.]),
        (1., [0.01, 0.01, 0.01]),
        (2., [0.1, 0.1, 0.1]),
    ]
    .map(|(norm, kl)| CausalLmSensitivityAxis {
        path: "a".into(),
        tensor: "b".into(),
        coordinate: None,
        gradient_l2: norm,
        probe_per_anchor_kl: Some(kl.to_vec()),
        prediction: cfg.predict(&kl, norm),
    });
    assert!(axes[0].prediction.is_none());
    assert!((axes[1].prediction.unwrap().target_fraction - 2. / 3.).abs() < 1e-12);
    assert!((axes[2].prediction.unwrap().target_fraction - 0.16).abs() < 1e-12);
    assert!((axes[3].prediction.unwrap().radius - 10f64.sqrt()).abs() < 1e-12);
    assert_eq!(select(&axes, CausalLmSensitivitySelect::Envelope), Some(3));
    assert_eq!(
        select(&axes[..3], CausalLmSensitivitySelect::Envelope),
        Some(1)
    );
    assert_eq!(
        select(&axes[..1], CausalLmSensitivitySelect::Envelope),
        None
    );
    assert_eq!(
        select(&axes, CausalLmSensitivitySelect::GradientNorm),
        Some(0)
    );
    assert_eq!(
        select(
            &[axes[3].clone(), axes[3].clone()],
            CausalLmSensitivitySelect::Envelope
        ),
        Some(0)
    );
}

#[test]
fn bounds_apply_to_the_entire_probe_family_before_decoding() {
    assert!(within_budget(32, 1 << 20, 1 << 17, 2));
    for (axes, scratch, vocab, bank) in [
        (33, 1, 11, 2),
        (0, 1, 11, 2),
        (2, (1 << 20) + 1, 11, 2),
        (32, 1, (1 << 17) + 1, 2),
        (4, 1, 11, 0),
    ] {
        assert!(!within_budget(axes, scratch, vocab, bank));
    }
}
