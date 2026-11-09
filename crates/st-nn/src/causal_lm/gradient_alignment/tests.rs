use super::*;

#[test]
fn opposition_is_not_always_mutual_interference() {
    // Dot < 0 but both losses decrease along the negative average.
    let s = Moments::measure(&[2., 0.], &[-1., 2.]).unwrap().finish();
    assert_eq!(
        (s.dot, s.left_loss_slope, s.right_loss_slope),
        (-2., -1., -1.5)
    );
    let s = Moments::measure(&[4.], &[-1.]).unwrap().finish();
    assert_eq!(
        (s.cosine, s.retained_fraction, s.mean_l2),
        (Some(-1.), Some(0.6), 1.5)
    );
    assert_eq!((s.left_loss_slope, s.right_loss_slope), (-6., 1.5));
    let s = Moments::measure(&[4.], &[-4.]).unwrap().finish();
    assert_eq!(
        (
            s.retained_fraction,
            s.mean_l2,
            s.left_loss_slope,
            s.right_loss_slope
        ),
        (Some(0.), 0., 0., 0.)
    );
}

#[test]
fn undefined_norms_and_invalid_inputs_are_not_alignment() {
    let s = Moments::measure(&[0.], &[0.]).unwrap().finish();
    assert_eq!((s.cosine, s.retained_fraction), (None, None));
    let s = Moments::measure(&[0.], &[1.]).unwrap().finish();
    assert_eq!(
        (s.cosine, s.retained_fraction, s.left_loss_slope),
        (None, Some(1.), 0.)
    );
    assert!(Moments::measure(&[], &[]).is_err());
    assert!(Moments::measure(&[0.], &[1., 0.]).is_err());
    assert!(Moments::measure(&[f32::NAN], &[0.]).is_err());
}

#[test]
fn component_merging_preserves_cancelling_dot_and_tiny_average() {
    let mut s = Moments::measure(&[1e20, 1.], &[1., 1.]).unwrap();
    s.merge(&Moments::measure(&[1e20], &[-1.]).unwrap())
        .unwrap();
    assert_eq!(s.finish().dot, 1.);
    let s = Moments::measure(&[1e20, 1.], &[-1e20, 0.])
        .unwrap()
        .finish();
    assert_eq!(s.mean_l2, 0.5);
    assert_eq!(s.left_loss_slope, -0.5);
    assert!(s.retained_fraction.unwrap() > 0.);
}
