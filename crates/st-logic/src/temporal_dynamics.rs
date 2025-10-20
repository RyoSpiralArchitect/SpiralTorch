use st_tensor::{PureResult, TensorError};

/// Abstraction over volumes that store per-voxel temporal harmonics and decay.
pub trait TemporalVolume: Clone {
    fn depth(&self) -> usize;
    fn height(&self) -> usize;
    fn width(&self) -> usize;
    fn voxel_count(&self) -> usize;
    fn harmonic_channels(&self) -> usize;
    fn voxels(&self) -> &[f32];
    fn voxels_mut(&mut self) -> &mut [f32];
    fn harmonics(&self) -> &[f32];
    fn harmonics_mut(&mut self) -> &mut [f32];
    fn resonance_decay(&self) -> &[f32];
    fn resonance_decay_mut(&mut self) -> &mut [f32];
    fn ensure_harmonic_channels(&mut self, channels: usize);
    fn blank_like(&self, harmonic_channels: usize) -> PureResult<Self>;
}

/// Configuration for simulating Z-space temporal evolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalPropagationConfig {
    /// Number of integration steps to perform.
    pub steps: usize,
    /// Time delta between steps (seconds).
    pub delta_time: f32,
    /// Blend factor applied to harmonic channels when mixing states.
    pub harmonic_blend: f32,
    /// Scalar applied to resonance decay to control damping strength.
    pub decay_gain: f32,
}

impl Default for TemporalPropagationConfig {
    fn default() -> Self {
        Self {
            steps: 4,
            delta_time: 1.0 / 24.0,
            harmonic_blend: 0.5,
            decay_gain: 1.0,
        }
    }
}

fn validate_pair<T: TemporalVolume>(left: &T, right: &T) -> PureResult<()> {
    if left.depth() != right.depth()
        || left.height() != right.height()
        || left.width() != right.width()
    {
        return Err(TensorError::ShapeMismatch {
            left: (left.depth(), left.height() * left.width()),
            right: (right.depth(), right.height() * right.width()),
        });
    }
    Ok(())
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Generates intermediary volumes between two keyframes using linear interpolation for
/// voxels and harmonics while preserving resonance decay trends.
pub fn interpolate_temporal_sequence<T: TemporalVolume>(
    start: &T,
    end: &T,
    config: TemporalPropagationConfig,
) -> PureResult<Vec<T>> {
    validate_pair(start, end)?;
    let steps = config.steps.max(1);
    let mut frames = Vec::with_capacity(steps + 2);
    frames.push(start.clone());
    let harmonic_channels = start.harmonic_channels().max(end.harmonic_channels());
    let voxel_count = start.voxel_count();
    for step in 1..=steps {
        let t = step as f32 / (steps as f32 + 1.0);
        let mut frame = start.blank_like(harmonic_channels)?;
        frame
            .voxels_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(idx, value)| {
                let s = start.voxels()[idx];
                let e = end.voxels()[idx];
                *value = lerp(s, e, t);
            });
        frame.ensure_harmonic_channels(harmonic_channels);
        for idx in 0..voxel_count {
            let decay_start = start.resonance_decay()[idx];
            let decay_end = end.resonance_decay()[idx];
            frame.resonance_decay_mut()[idx] = lerp(decay_start, decay_end, t);
            for channel in 0..harmonic_channels {
                let start_val = start
                    .harmonics()
                    .get(idx * start.harmonic_channels() + channel)
                    .copied()
                    .unwrap_or(0.0);
                let end_val = end
                    .harmonics()
                    .get(idx * end.harmonic_channels() + channel)
                    .copied()
                    .unwrap_or(0.0);
                frame.harmonics_mut()[idx * harmonic_channels + channel] =
                    lerp(start_val, end_val, t);
            }
        }
        frames.push(frame);
    }
    frames.push(end.clone());
    Ok(frames)
}

/// Applies a simple exponential integration using the drive volume to update the state.
pub fn integrate_volume<T: TemporalVolume>(
    state: &mut T,
    drive: &T,
    config: TemporalPropagationConfig,
) -> PureResult<()> {
    validate_pair(state, drive)?;
    if !config.delta_time.is_finite() || config.delta_time <= 0.0 {
        return Err(TensorError::InvalidValue {
            label: "temporal_delta_time",
        });
    }
    let steps = config.steps.max(1);
    let dt = config.delta_time / steps as f32;
    let harmonic_blend = config.harmonic_blend.clamp(0.0, 1.0);
    let decay_gain = config.decay_gain.max(0.0);
    state.ensure_harmonic_channels(drive.harmonic_channels());
    let voxel_count = state.voxel_count();
    for _ in 0..steps {
        for idx in 0..voxel_count {
            let drive_decay = drive.resonance_decay()[idx].max(0.0) * decay_gain;
            let decay = (drive_decay * dt).min(1.0);
            let retain = 1.0 - decay;
            let target = drive.voxels()[idx];
            let current = state.voxels()[idx];
            state.voxels_mut()[idx] = current * retain + target * (1.0 - retain);
            let baseline_decay = state.resonance_decay()[idx];
            state.resonance_decay_mut()[idx] =
                baseline_decay * retain + drive_decay * (1.0 - retain);
            for channel in 0..state.harmonic_channels() {
                let src = if channel < drive.harmonic_channels() {
                    drive.harmonics()[idx * drive.harmonic_channels() + channel]
                } else {
                    0.0
                };
                let dst_idx = idx * state.harmonic_channels() + channel;
                let dst = state.harmonics()[dst_idx];
                state.harmonics_mut()[dst_idx] =
                    dst * (1.0 - harmonic_blend) + src * harmonic_blend;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct MockVolume {
        depth: usize,
        height: usize,
        width: usize,
        harmonic_channels: usize,
        voxels: Vec<f32>,
        harmonics: Vec<f32>,
        decay: Vec<f32>,
    }

    impl MockVolume {
        fn new(depth: usize, height: usize, width: usize, harmonic_channels: usize) -> Self {
            let voxel_count = depth * height * width;
            Self {
                depth,
                height,
                width,
                harmonic_channels,
                voxels: vec![0.0; voxel_count],
                harmonics: vec![0.0; voxel_count * harmonic_channels],
                decay: vec![1.0; voxel_count],
            }
        }
    }

    impl TemporalVolume for MockVolume {
        fn depth(&self) -> usize {
            self.depth
        }

        fn height(&self) -> usize {
            self.height
        }

        fn width(&self) -> usize {
            self.width
        }

        fn voxel_count(&self) -> usize {
            self.depth * self.height * self.width
        }

        fn harmonic_channels(&self) -> usize {
            self.harmonic_channels
        }

        fn voxels(&self) -> &[f32] {
            &self.voxels
        }

        fn voxels_mut(&mut self) -> &mut [f32] {
            &mut self.voxels
        }

        fn harmonics(&self) -> &[f32] {
            &self.harmonics
        }

        fn harmonics_mut(&mut self) -> &mut [f32] {
            &mut self.harmonics
        }

        fn resonance_decay(&self) -> &[f32] {
            &self.decay
        }

        fn resonance_decay_mut(&mut self) -> &mut [f32] {
            &mut self.decay
        }

        fn ensure_harmonic_channels(&mut self, channels: usize) {
            if self.harmonic_channels == channels {
                return;
            }
            self.harmonic_channels = channels;
            self.harmonics.resize(self.voxel_count() * channels, 0.0);
        }

        fn blank_like(&self, harmonic_channels: usize) -> PureResult<Self> {
            Ok(Self::new(
                self.depth,
                self.height,
                self.width,
                harmonic_channels,
            ))
        }
    }

    #[test]
    fn interpolates_temporal_sequence() {
        let mut start = MockVolume::new(1, 1, 1, 2);
        start.voxels_mut()[0] = 0.0;
        start.harmonics_mut().copy_from_slice(&[0.0, 1.0]);
        start.resonance_decay_mut()[0] = 0.2;
        let mut end = MockVolume::new(1, 1, 1, 2);
        end.voxels_mut()[0] = 1.0;
        end.harmonics_mut().copy_from_slice(&[1.0, 0.0]);
        end.resonance_decay_mut()[0] = 0.8;
        let frames = interpolate_temporal_sequence(
            &start,
            &end,
            TemporalPropagationConfig {
                steps: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(frames.len(), 4);
        let expected_step_one = 1.0 / 3.0;
        let expected_step_two = 2.0 / 3.0;
        assert!((frames[1].voxels()[0] - expected_step_one).abs() < 1e-6);
        assert!((frames[2].voxels()[0] - expected_step_two).abs() < 1e-6);
        assert!((frames[2].harmonics()[0] - expected_step_two).abs() < 1e-6);
    }

    #[test]
    fn integrates_drive_volume() {
        let mut state = MockVolume::new(1, 1, 1, 1);
        state.voxels_mut()[0] = 0.0;
        state.resonance_decay_mut()[0] = 0.1;
        state.harmonics_mut()[0] = 0.0;
        let mut drive = MockVolume::new(1, 1, 1, 1);
        drive.voxels_mut()[0] = 1.0;
        drive.resonance_decay_mut()[0] = 1.0;
        drive.harmonics_mut()[0] = 1.0;
        integrate_volume(
            &mut state,
            &drive,
            TemporalPropagationConfig {
                steps: 4,
                delta_time: 0.5,
                harmonic_blend: 1.0,
                decay_gain: 1.0,
            },
        )
        .unwrap();
        assert!(state.voxels()[0] > 0.0);
        assert!(state.resonance_decay()[0] > 0.1);
        assert!((state.harmonics()[0] - 1.0).abs() < 1e-3);
    }
}
