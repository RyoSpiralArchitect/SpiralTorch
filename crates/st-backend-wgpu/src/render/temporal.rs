use std::f32::consts::TAU;

use thiserror::Error;

pub type TemporalRenderResult<T> = Result<T, TemporalRenderError>;

/// Shape and configuration failures emitted by temporal rendering.
#[non_exhaustive]
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TemporalRenderError {
    #[error("temporal volume dimensions must be positive, got {depth}x{height}x{width}")]
    InvalidDimensions {
        depth: usize,
        height: usize,
        width: usize,
    },
    #[error("temporal volume element count overflowed for {depth}x{height}x{width}")]
    VoxelCountOverflow {
        depth: usize,
        height: usize,
        width: usize,
    },
    #[error("temporal volume {field} length mismatch: expected {expected}, got {actual}")]
    BufferLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "temporal harmonic element count overflowed: voxels={voxel_count} channels={channels}"
    )]
    HarmonicCountOverflow { voxel_count: usize, channels: usize },
    #[error("temporal renderer requires at least one frame")]
    ZeroFrames,
    #[error("temporal sample rate must be finite and positive, got {sample_rate_hz}")]
    InvalidSampleRate { sample_rate_hz: f32 },
}

/// Trait describing the data required to render a temporal Z-space volume.
pub trait TemporalVolumeLike {
    /// Depth of the volume in Z.
    fn depth(&self) -> usize;
    /// Height of each slice.
    fn height(&self) -> usize;
    /// Width of each slice.
    fn width(&self) -> usize;
    /// Number of harmonic channels encoded per voxel.
    fn harmonic_channels(&self) -> usize;
    /// Flat voxel buffer arranged as `depth * height * width`.
    fn voxels(&self) -> &[f32];
    /// Harmonic coefficients stored alongside voxels.
    fn temporal_harmonics(&self) -> &[f32];
    /// Per-voxel resonance decay coefficients.
    fn resonance_decay(&self) -> &[f32];

    /// Total number of voxels contained in the volume.
    fn voxel_count(&self) -> TemporalRenderResult<usize> {
        if self.depth() == 0 || self.height() == 0 || self.width() == 0 {
            return Err(TemporalRenderError::InvalidDimensions {
                depth: self.depth(),
                height: self.height(),
                width: self.width(),
            });
        }
        self.depth()
            .checked_mul(self.height())
            .and_then(|count| count.checked_mul(self.width()))
            .ok_or(TemporalRenderError::VoxelCountOverflow {
                depth: self.depth(),
                height: self.height(),
                width: self.width(),
            })
    }
}

/// Immutable view over the tensors required by [`TemporalVolumeLike`].
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalVolumeView<'a> {
    depth: usize,
    height: usize,
    width: usize,
    harmonic_channels: usize,
    voxels: &'a [f32],
    temporal_harmonics: &'a [f32],
    resonance_decay: &'a [f32],
}

#[cfg_attr(not(test), allow(dead_code))]
impl<'a> TemporalVolumeView<'a> {
    /// Builds a temporal volume view validating the provided buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        depth: usize,
        height: usize,
        width: usize,
        harmonic_channels: usize,
        voxels: &'a [f32],
        temporal_harmonics: &'a [f32],
        resonance_decay: &'a [f32],
    ) -> TemporalRenderResult<Self> {
        if depth == 0 || height == 0 || width == 0 {
            return Err(TemporalRenderError::InvalidDimensions {
                depth,
                height,
                width,
            });
        }
        let voxel_count = depth
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .ok_or(TemporalRenderError::VoxelCountOverflow {
                depth,
                height,
                width,
            })?;
        if voxels.len() != voxel_count {
            return Err(TemporalRenderError::BufferLength {
                field: "voxels",
                expected: voxel_count,
                actual: voxels.len(),
            });
        }
        if resonance_decay.len() != voxel_count {
            return Err(TemporalRenderError::BufferLength {
                field: "resonance_decay",
                expected: voxel_count,
                actual: resonance_decay.len(),
            });
        }
        let harmonic_expected = voxel_count.checked_mul(harmonic_channels).ok_or(
            TemporalRenderError::HarmonicCountOverflow {
                voxel_count,
                channels: harmonic_channels,
            },
        )?;
        if harmonic_channels == 0 {
            if !temporal_harmonics.is_empty() {
                return Err(TemporalRenderError::BufferLength {
                    field: "temporal_harmonics",
                    expected: 0,
                    actual: temporal_harmonics.len(),
                });
            }
        } else if temporal_harmonics.len() != harmonic_expected {
            return Err(TemporalRenderError::BufferLength {
                field: "temporal_harmonics",
                expected: harmonic_expected,
                actual: temporal_harmonics.len(),
            });
        }
        Ok(Self {
            depth,
            height,
            width,
            harmonic_channels,
            voxels,
            temporal_harmonics,
            resonance_decay,
        })
    }
}

impl<'a> TemporalVolumeLike for TemporalVolumeView<'a> {
    fn depth(&self) -> usize {
        self.depth
    }

    fn height(&self) -> usize {
        self.height
    }

    fn width(&self) -> usize {
        self.width
    }

    fn harmonic_channels(&self) -> usize {
        self.harmonic_channels
    }

    fn voxels(&self) -> &[f32] {
        self.voxels
    }

    fn temporal_harmonics(&self) -> &[f32] {
        self.temporal_harmonics
    }

    fn resonance_decay(&self) -> &[f32] {
        self.resonance_decay
    }
}

/// Configuration for rendering a temporal volume into time-aware slices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalRendererConfig {
    /// Number of frames to emit for the animation.
    pub frames: usize,
    /// Sampling rate used to evaluate temporal harmonics.
    pub sample_rate_hz: f32,
}

impl Default for TemporalRendererConfig {
    fn default() -> Self {
        Self {
            frames: 16,
            sample_rate_hz: 24.0,
        }
    }
}

/// A single time-aware slice emitted by the renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalRenderSlice {
    pub frame_index: usize,
    pub time_seconds: f32,
    pub data: Vec<f32>,
}

impl TemporalRenderSlice {
    fn new(frame_index: usize, time_seconds: f32, data: Vec<f32>) -> Self {
        Self {
            frame_index,
            time_seconds,
            data,
        }
    }
}

/// Collection of slices arranged as a 4D `[t, z, y, x]` tensor layout.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalRenderOutput {
    pub frames: usize,
    pub depth: usize,
    pub height: usize,
    pub width: usize,
    pub slices: Vec<TemporalRenderSlice>,
}

impl TemporalRenderOutput {
    /// Returns the flat buffer for a given frame if it exists.
    pub fn frame(&self, index: usize) -> Option<&[f32]> {
        self.slices
            .get(index)
            .filter(|slice| slice.frame_index == index)
            .map(|slice| slice.data.as_slice())
    }
}

/// CPU-backed renderer that mirrors the GPU pipeline for tests and tooling.
#[derive(Clone, Debug)]
pub struct TemporalRenderer {
    config: TemporalRendererConfig,
}

impl TemporalRenderer {
    pub fn new(config: TemporalRendererConfig) -> Self {
        Self { config }
    }

    pub fn render<V>(&self, volume: &V) -> TemporalRenderResult<TemporalRenderOutput>
    where
        V: TemporalVolumeLike + ?Sized,
    {
        if self.config.frames == 0 {
            return Err(TemporalRenderError::ZeroFrames);
        }
        if !self.config.sample_rate_hz.is_finite() || self.config.sample_rate_hz <= 0.0 {
            return Err(TemporalRenderError::InvalidSampleRate {
                sample_rate_hz: self.config.sample_rate_hz,
            });
        }

        let depth = volume.depth();
        let height = volume.height();
        let width = volume.width();
        let voxel_count = volume.voxel_count()?;

        let voxels = volume.voxels();
        if voxels.len() != voxel_count {
            return Err(TemporalRenderError::BufferLength {
                field: "voxels",
                expected: voxel_count,
                actual: voxels.len(),
            });
        }

        let decay = volume.resonance_decay();
        if decay.len() != voxel_count {
            return Err(TemporalRenderError::BufferLength {
                field: "resonance_decay",
                expected: voxel_count,
                actual: decay.len(),
            });
        }

        let harmonic_channels = volume.harmonic_channels();
        let harmonics = volume.temporal_harmonics();
        let harmonic_expected = voxel_count.checked_mul(harmonic_channels).ok_or(
            TemporalRenderError::HarmonicCountOverflow {
                voxel_count,
                channels: harmonic_channels,
            },
        )?;
        if harmonic_channels == 0 {
            if !harmonics.is_empty() {
                return Err(TemporalRenderError::BufferLength {
                    field: "temporal_harmonics",
                    expected: 0,
                    actual: harmonics.len(),
                });
            }
        } else if harmonics.len() != harmonic_expected {
            return Err(TemporalRenderError::BufferLength {
                field: "temporal_harmonics",
                expected: harmonic_expected,
                actual: harmonics.len(),
            });
        }

        let mut slices = Vec::with_capacity(self.config.frames);
        let time_delta = 1.0 / self.config.sample_rate_hz;

        for frame in 0..self.config.frames {
            let t = frame as f32 * time_delta;
            let mut buffer = vec![0.0f32; voxel_count];
            for (voxel_index, value) in buffer.iter_mut().enumerate() {
                let mut base = voxels[voxel_index];
                let decay_coeff = decay
                    .get(voxel_index)
                    .copied()
                    .filter(|v| v.is_finite())
                    .unwrap_or(1.0)
                    .max(0.0);

                if harmonic_channels > 0 {
                    let start = voxel_index * harmonic_channels;
                    let end = start + harmonic_channels;
                    let harmonics_slice = &harmonics[start..end];
                    for (channel, amplitude) in harmonics_slice.iter().enumerate() {
                        if amplitude.abs() <= f32::EPSILON {
                            continue;
                        }
                        let freq = (channel + 1) as f32;
                        base += amplitude * (freq * TAU * t).sin();
                    }
                }

                let damped = if decay_coeff <= f32::EPSILON {
                    0.0
                } else {
                    base * (-decay_coeff * t).exp()
                };
                *value = damped;
            }
            slices.push(TemporalRenderSlice::new(frame, t, buffer));
        }

        Ok(TemporalRenderOutput {
            frames: self.config.frames,
            depth,
            height,
            width,
            slices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_decay_and_harmonics() {
        use std::f32::consts::TAU;

        let voxels = [0.5f32, 1.0];
        let harmonics = [0.25, 0.0, -0.25, 0.5];
        let decay = [0.5, 0.5];
        let volume = TemporalVolumeView::new(1, 1, 2, 2, &voxels, &harmonics, &decay).unwrap();
        let renderer = TemporalRenderer::new(TemporalRendererConfig {
            frames: 3,
            sample_rate_hz: 10.0,
        });
        let output = renderer.render(&volume).unwrap();
        assert_eq!(output.frames, 3);
        assert_eq!(output.slices.len(), 3);
        assert_eq!(output.depth, 1);
        assert_eq!(output.width, 2);
        let first = output.frame(0).unwrap();
        assert_eq!(first.len(), 2);
        // first frame should match base values (no time evolution yet)
        assert!((first[0] - voxels[0]).abs() < 1e-4);
        assert!((first[1] - voxels[1]).abs() < 1e-4);
        let second = output.frame(1).unwrap();
        assert_eq!(second.len(), 2);

        let t = 1.0 / renderer.config.sample_rate_hz;
        let harmonic_channels = 2;
        let expected_voxel0 = {
            let mut base = voxels[0];
            for (channel, amplitude) in harmonics[..harmonic_channels].iter().enumerate() {
                if amplitude.abs() <= f32::EPSILON {
                    continue;
                }
                let freq = (channel + 1) as f32;
                base += amplitude * (freq * TAU * t).sin();
            }
            base * (-decay[0].max(0.0) * t).exp()
        };
        let expected_voxel1 = {
            let mut base = voxels[1];
            for (channel, amplitude) in harmonics[harmonic_channels..].iter().enumerate() {
                if amplitude.abs() <= f32::EPSILON {
                    continue;
                }
                let freq = (channel + 1) as f32;
                base += amplitude * (freq * TAU * t).sin();
            }
            base * (-decay[1].max(0.0) * t).exp()
        };

        assert!((second[0] - expected_voxel0).abs() < 1e-4);
        assert!((second[1] - expected_voxel1).abs() < 1e-4);
    }

    #[test]
    fn rejects_mismatched_buffers() {
        let voxels = [0.0f32, 1.0];
        let harmonics = [0.0f32; 3];
        let decay = [1.0f32, 1.0];
        let result = TemporalVolumeView::new(1, 1, 2, 2, &voxels, &harmonics, &decay);
        assert!(matches!(
            result,
            Err(TemporalRenderError::BufferLength {
                field: "temporal_harmonics",
                ..
            })
        ));
    }

    #[test]
    fn rejects_overflowing_harmonic_shape_without_saturation() {
        let voxels = [0.0f32; 2];
        let decay = [1.0f32; 2];
        let result = TemporalVolumeView::new(1, 1, 2, usize::MAX, &voxels, &[], &decay);
        assert!(matches!(
            result,
            Err(TemporalRenderError::HarmonicCountOverflow { .. })
        ));
    }
}
