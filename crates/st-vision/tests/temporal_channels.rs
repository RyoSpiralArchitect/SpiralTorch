use st_logic::temporal_dynamics::{
    integrate_volume, interpolate_temporal_sequence, TemporalPropagationConfig,
};
use st_vision::ZSpaceVolume;

fn volume(channels: usize, values: &[f32]) -> ZSpaceVolume {
    let mut volume = ZSpaceVolume::zeros_with_temporal(1, 1, 3, channels).unwrap();
    volume.temporal_harmonics_mut().copy_from_slice(values);
    volume
}

#[test]
fn harmonic_resize_preserves_each_voxel() {
    let mut v = volume(1, &[1.0, 2.0, 3.0]);
    v.ensure_harmonic_channels(3);
    assert_eq!(
        v.temporal_harmonics(),
        &[1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]
    );
    v.temporal_harmonics_mut()
        .copy_from_slice(&[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    v.ensure_harmonic_channels(2);
    assert_eq!(v.temporal_harmonics(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    v.ensure_harmonic_channels(1);
    assert_eq!(v.temporal_harmonics(), &[1.0, 2.0, 3.0]);
    v.ensure_harmonic_channels(0);
    assert!(v.temporal_harmonics().is_empty());
    v.ensure_harmonic_channels(2);
    assert_eq!(v.temporal_harmonics(), &[0.0; 6]);
}

#[test]
fn interpolation_zero_pads_channels_not_adjacent_voxels() {
    let narrow = volume(1, &[1.0, 2.0, 3.0]);
    let wide = volume(2, &[10.0, 100.0, 20.0, 200.0, 30.0, 300.0]);
    for (start, end) in [(&narrow, &wide), (&wide, &narrow)] {
        let frames = interpolate_temporal_sequence(
            start,
            end,
            TemporalPropagationConfig {
                steps: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].temporal_harmonics(), start.temporal_harmonics());
        assert_eq!(frames[2].temporal_harmonics(), end.temporal_harmonics());
        assert_eq!(
            frames[1].temporal_harmonics(),
            &[5.5, 50.0, 11.0, 100.0, 16.5, 150.0]
        );
    }
}

#[test]
fn integration_keeps_voxel_identity_when_drive_adds_channels() {
    let mut state = volume(1, &[1.0, 2.0, 3.0]);
    let drive = volume(2, &[10.0, 100.0, 20.0, 200.0, 30.0, 300.0]);
    integrate_volume(
        &mut state,
        &drive,
        TemporalPropagationConfig {
            steps: 1,
            harmonic_blend: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        state.temporal_harmonics(),
        &[5.5, 50.0, 11.0, 100.0, 16.5, 150.0]
    );
}
