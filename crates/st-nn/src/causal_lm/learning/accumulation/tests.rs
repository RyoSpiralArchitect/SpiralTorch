use super::*;

fn accumulator() -> NativeCausalLmAccumulator {
    NativeCausalLmAccumulator {
        base_id: "base".into(),
        geometry_id: "geometry".into(),
        loss: 0.,
        target_tokens: 0,
        microbatches: 0,
        means: BTreeMap::from([(
            "site".into(),
            BTreeMap::from([("gate".into(), vec![0.; 2])]),
        )]),
    }
}

fn batch(shape: [usize; 2], values: [f32; 2], loss: f32) -> NativeCausalLmGradients {
    NativeCausalLmGradients {
        base_id: "base".into(),
        geometry_id: "geometry".into(),
        shape,
        loss,
        parameters: BTreeMap::from([(
            "site".into(),
            BTreeMap::from([("gate".into(), values.to_vec())]),
        )]),
    }
}

#[test]
fn unequal_windows_use_target_weight_not_microbatch_mean() {
    let mut state = accumulator();
    assert_eq!(state.loss(), None);
    assert!(state.parameter_gradients().is_err());
    state.add(&batch([1, 3], [1., -2.], 3.)).unwrap(); // 2 targets.
    state.add(&batch([2, 4], [5., 6.], 7.)).unwrap(); // 6 targets.
    assert_eq!(state.target_tokens(), 8);
    assert_eq!(state.microbatches(), 2);
    assert_eq!(state.loss(), Some(6.));
    assert_eq!(
        state.parameter_gradients().unwrap()["site"]["gate"],
        [4., 4.]
    );
    let mut detached = state.parameter_gradients().unwrap();
    detached.get_mut("site").unwrap().get_mut("gate").unwrap()[0] = 99.;
    assert_eq!(
        state.parameter_gradients().unwrap()["site"]["gate"],
        [4., 4.]
    );
}

#[test]
fn invalid_late_values_keys_identity_and_shapes_never_partially_commit() {
    let mut state = accumulator();
    state.add(&batch([1, 3], [1., -2.], 3.)).unwrap();
    let before = format!("{state:?}");
    let mut invalid = vec![
        batch([1, 3], [5., f32::NAN], 7.),
        batch([1, 3], [5., 6.], f32::INFINITY),
        batch([0, 3], [5., 6.], 7.),
        batch([1, 1], [5., 6.], 7.),
        batch([usize::MAX, 3], [5., 6.], 7.),
    ];
    let mut foreign = batch([1, 3], [5., 6.], 7.);
    foreign.geometry_id = "other".into();
    invalid.push(foreign);
    let mut foreign = batch([1, 3], [5., 6.], 7.);
    foreign.base_id = "other".into();
    invalid.push(foreign);
    let mut short = batch([1, 3], [5., 6.], 7.);
    short
        .parameters
        .get_mut("site")
        .unwrap()
        .get_mut("gate")
        .unwrap()
        .pop();
    invalid.push(short);
    let mut extra = batch([1, 3], [5., 6.], 7.);
    extra
        .parameters
        .get_mut("site")
        .unwrap()
        .insert("extra".into(), vec![0.]);
    invalid.push(extra);
    let mut missing = batch([1, 3], [5., 6.], 7.);
    missing.parameters.clear();
    invalid.push(missing);
    for value in invalid {
        assert!(state.add(&value).is_err());
        assert_eq!(format!("{state:?}"), before);
    }
}

#[test]
fn finite_extremes_do_not_overflow_f32_sums_and_counters_fail_closed() {
    let mut state = accumulator();
    state
        .add(&batch([1, 2], [f32::MAX, f32::from_bits(1)], f32::MAX))
        .unwrap();
    state
        .add(&batch([1, 2], [f32::MAX, f32::from_bits(1)], f32::MAX))
        .unwrap();
    assert_eq!(
        state.parameter_gradients().unwrap()["site"]["gate"],
        [f32::MAX, f32::from_bits(1)]
    );
    assert_eq!(state.loss(), Some(f64::from(f32::MAX)));
    let mut opposite = accumulator();
    opposite
        .add(&batch([1, 2], [f32::MAX, -f32::MAX], 1.))
        .unwrap();
    opposite
        .add(&batch([1, 2], [-f32::MAX, f32::MAX], 1.))
        .unwrap();
    assert_eq!(
        opposite.parameter_gradients().unwrap()["site"]["gate"],
        [0.; 2]
    );
    state.target_tokens = MAX_TARGET_TOKENS;
    let before = format!("{state:?}");
    assert!(state.add(&batch([1, 2], [1.; 2], 1.)).is_err());
    assert_eq!(format!("{state:?}"), before);
    state.target_tokens = 1;
    state.microbatches = u64::MAX;
    let before = format!("{state:?}");
    assert!(state.add(&batch([1, 2], [1.; 2], 1.)).is_err());
    assert_eq!(format!("{state:?}"), before);
}
