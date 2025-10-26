// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// =============================================================================
//  SpiralReality Proprietary
// =============================================================================

//! Video decoding and temporal Z-dynamics orchestration for SpiralTorch Vision.
//! The pipeline bridges FFmpeg-bound frame streams with [`ZSpaceVolume`] fusion,
//! layering motion-aware resonance transforms on top of chrono telemetry.

use crate::{
    ChronoSnapshot, SpectralWindow, StreamedVolume, TemporalResonanceBuffer, ZSpaceStreamFrame,
    ZSpaceVolume,
};
use st_core::telemetry::atlas::{AtlasFrame, AtlasMetric};
use st_tensor::{PureResult, Tensor};
use std::collections::VecDeque;

/// Frame emitted by the decoder prior to Z-space fusion.
#[derive(Clone, Debug)]
pub struct DecodedFrame {
    /// Absolute timestamp for the decoded frame (seconds).
    pub timestamp: f32,
    /// Image plane stored as a tensor (rows = height, cols = width).
    pub tensor: Tensor,
}

/// Decoder abstraction wrapping FFmpeg bindings exposed in `bindings/`.
pub trait VideoDecoder {
    /// Returns the next decoded frame or `None` when the stream is exhausted.
    fn next_frame(&mut self) -> PureResult<Option<DecodedFrame>>;
}

/// Minimal trait that FFmpeg bindings can implement to feed the pipeline.
pub trait FfmpegBinding: Send {
    /// Advances the underlying FFmpeg decoder, returning a timestamped tensor.
    fn decode_next(&mut self) -> PureResult<Option<(f32, Tensor)>>;
}

/// Decoder wrapper that adapts FFmpeg bindings to [`VideoDecoder`].
pub struct FfmpegDecoder<B: FfmpegBinding> {
    binding: B,
}

impl<B: FfmpegBinding> FfmpegDecoder<B> {
    /// Creates a decoder from an FFmpeg binding implementation.
    pub fn new(binding: B) -> Self {
        Self { binding }
    }
}

impl<B: FfmpegBinding> VideoDecoder for FfmpegDecoder<B> {
    fn next_frame(&mut self) -> PureResult<Option<DecodedFrame>> {
        Ok(self
            .binding
            .decode_next()?
            .map(|(timestamp, tensor)| DecodedFrame { timestamp, tensor }))
    }
}

/// Configuration used to construct a [`VideoPipeline`].
#[derive(Clone, Debug)]
pub struct VideoPipelineConfig {
    /// Temporal EMA coefficient applied when fusing consecutive Z volumes.
    pub temporal_alpha: f32,
    /// Decay applied to the resonance buffer when smoothing depth weights.
    pub resonance_decay: f32,
    /// Gain applied to motion embeddings extracted from frame deltas.
    pub motion_gain: f32,
    /// Size of the rolling window used when computing temporal digests.
    pub digest_window: usize,
    /// Energy threshold used to classify frames as quiescent.
    pub quiescence_threshold: f32,
}

impl Default for VideoPipelineConfig {
    fn default() -> Self {
        Self {
            temporal_alpha: 0.35,
            resonance_decay: 0.25,
            motion_gain: 1.0,
            digest_window: 16,
            quiescence_threshold: 0.05,
        }
    }
}

/// Aggregated temporal annotation describing Z-dynamics for a frame.
#[derive(Clone, Debug, Default)]
pub struct ZDynamicsAnnotation {
    /// Temporal buffer of depth attention weights after smoothing.
    pub smoothed_weights: Vec<f32>,
    /// Energy computed for each depth slice prior to smoothing.
    pub per_depth_energy: Vec<f32>,
    /// Spectral response sampled with a Hann window across the depth axis.
    pub spectral_response: Vec<f32>,
}

/// Output emitted by [`VideoPipeline::next`].
#[derive(Clone, Debug)]
pub struct VideoPipelineOutput {
    /// Sequential index of the emitted frame.
    pub frame_index: usize,
    /// Atlas frame carrying temporal metrics and Z-dynamics annotations.
    pub atlas_frame: AtlasFrame,
    /// Streamed volume that fed the annotations.
    pub stream: StreamedVolume,
    /// Motion embedding highlighting inter-frame deltas.
    pub motion_embedding: Tensor,
    /// Resonance envelope accumulated across the stream.
    pub resonance_envelope: Tensor,
    /// Derived Z-dynamics metadata (weights, per-depth energy, spectrum).
    pub z_dynamics: ZDynamicsAnnotation,
    /// Digest capturing temporal statistics collected so far.
    pub temporal_digest: TemporalDigest,
    /// Rolling digest computed over the configured window length.
    pub window_digest: TemporalDigest,
}

/// Video processing pipeline that decodes frames, fuses Z volumes, and emits atlas telemetry.
pub struct VideoPipeline<D: VideoDecoder> {
    decoder: D,
    config: VideoPipelineConfig,
    frame_index: usize,
    temporal_buffer: TemporalResonanceBuffer,
    motion_filter: MotionEmbeddingFilter,
    resonance_envelope: ResonanceEnvelope,
    stats: TemporalStats,
    volume: Option<ZSpaceVolume>,
    timeline: Vec<AtlasFrame>,
}

impl<D: VideoDecoder> VideoPipeline<D> {
    /// Creates a new pipeline from a decoder and configuration.
    pub fn new(decoder: D, mut config: VideoPipelineConfig) -> Self {
        config.temporal_alpha = config.temporal_alpha.clamp(0.0, 1.0);
        let resonance_decay = config.resonance_decay.clamp(0.0, 1.0);
        Self {
            decoder,
            frame_index: 0,
            temporal_buffer: TemporalResonanceBuffer::new(resonance_decay),
            motion_filter: MotionEmbeddingFilter::new(config.motion_gain.max(0.0)),
            resonance_envelope: ResonanceEnvelope::new(resonance_decay),
            stats: TemporalStats::with_threshold(config.quiescence_threshold.max(0.0)),
            volume: None,
            timeline: Vec::new(),
            config,
        }
    }

    /// Decodes the next frame, updates temporal buffers, and emits Z-dynamics annotations.
    pub fn next(&mut self) -> PureResult<Option<VideoPipelineOutput>> {
        let Some(decoded) = self.decoder.next_frame()? else {
            return Ok(None);
        };
        let motion = self.motion_filter.process(&decoded.tensor)?;
        let stream_frame = ZSpaceStreamFrame::new(vec![decoded.tensor.clone(), motion.clone()])?;

        let streamed = if let Some(volume) = self.volume.as_mut() {
            volume.ingest_stream_frame(stream_frame.clone(), self.config.temporal_alpha)?
        } else {
            let streamed = ZSpaceVolume::from_stream_frame(stream_frame.clone())?;
            self.volume = Some(streamed.volume.clone());
            streamed
        };

        let mut streamed = streamed;
        if let Some(volume) = self.volume.as_mut() {
            *volume = streamed.volume.clone();
        } else {
            self.volume = Some(streamed.volume.clone());
        }

        // Compute per-depth energy using tensor ops.
        let mut per_depth = Vec::with_capacity(streamed.volume.depth());
        for index in 0..streamed.volume.depth() {
            let slice = streamed.volume.slice(index)?;
            per_depth.push(tensor_energy(&slice));
        }

        if let Some(history) = self.temporal_buffer.history() {
            if history.len() != per_depth.len() {
                self.temporal_buffer.clear();
            }
        }
        let smoothed_weights = self.temporal_buffer.apply(&per_depth)?;
        let spectral_response = streamed.volume.spectral_response(&SpectralWindow::hann());

        let resonance_envelope = self.resonance_envelope.update(&streamed.volume)?;
        let resonance_energy = tensor_energy(&resonance_envelope);
        let motion_energy = tensor_energy(&motion);
        let snapshot = self
            .stats
            .ingest(decoded.timestamp, motion_energy, resonance_energy);

        streamed.chrono_snapshot = Some(snapshot.clone());

        let mut atlas = streamed
            .atlas_frame
            .clone()
            .unwrap_or_else(|| AtlasFrame::new(snapshot.timestamp()));
        atlas.timestamp = snapshot.timestamp();
        atlas.chrono_summary = Some(snapshot.summary().clone());
        atlas.collapse_total = Some(resonance_energy);
        if let Some((index, weight)) = smoothed_weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            if smoothed_weights.len() > 1 {
                atlas.z_signal = Some(index as f32 / (smoothed_weights.len() - 1) as f32);
            } else {
                atlas.z_signal = Some(*weight);
            }
        }
        atlas.loop_support = motion_energy.max(0.0);
        push_metric(&mut atlas, "z.motion_energy", motion_energy);
        push_metric(&mut atlas, "z.resonance_energy", resonance_energy);
        let entropy = weight_entropy(&smoothed_weights);
        push_metric(&mut atlas, "z.weight_entropy", entropy);
        atlas.notes.push("video.pipeline.z_dynamics".to_string());

        streamed.atlas_frame = Some(atlas.clone());
        self.timeline.push(atlas.clone());

        let z_dynamics = ZDynamicsAnnotation {
            smoothed_weights: smoothed_weights.clone(),
            per_depth_energy: per_depth,
            spectral_response,
        };

        let digest = self.stats.digest();
        let window_digest = self.stats.digest_window(self.config.digest_window.max(1));

        self.frame_index += 1;

        Ok(Some(VideoPipelineOutput {
            frame_index: self.frame_index - 1,
            atlas_frame: atlas,
            stream: streamed,
            motion_embedding: motion,
            resonance_envelope,
            z_dynamics,
            temporal_digest: digest,
            window_digest,
        }))
    }

    /// Returns a reference to the latest fused Z-space volume, if available.
    pub fn last_volume(&self) -> Option<&ZSpaceVolume> {
        self.volume.as_ref()
    }

    /// Provides access to the accumulated atlas timeline for downstream reporting.
    pub fn atlas_timeline(&self) -> &[AtlasFrame] {
        &self.timeline
    }

    /// Exposes the most recent [`TemporalDigest`] without advancing the stream.
    pub fn temporal_digest(&self) -> TemporalDigest {
        self.stats.digest()
    }

    /// Builds a digest limited to the most recent `window` frames.
    pub fn temporal_digest_window(&self, window: usize) -> TemporalDigest {
        self.stats.digest_window(window)
    }
}

/// Aggregated telemetry summarising the temporal behaviour of the pipeline so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemporalDigest {
    /// Number of frames processed by the pipeline.
    pub frames: usize,
    /// Total stream duration in seconds.
    pub duration: f32,
    /// Mean motion energy accumulated from frame deltas.
    pub mean_motion_energy: f32,
    /// Standard deviation of the motion energy signal.
    pub motion_std: f32,
    /// Mean resonance energy collected from the Z volume envelope.
    pub mean_resonance_energy: f32,
    /// Standard deviation of the resonance energy signal.
    pub resonance_std: f32,
    /// Mean fractional decay between consecutive resonance measurements.
    pub mean_resonance_decay: f32,
    /// Minimum resonance energy observed so far.
    pub min_resonance_energy: f32,
    /// Maximum resonance energy observed so far.
    pub max_resonance_energy: f32,
    /// Maximum motion energy observed in the aggregation window.
    pub peak_motion_energy: f32,
    /// Timestamp associated with the motion peak.
    pub peak_motion_timestamp: f32,
    /// Maximum resonance energy observed in the aggregation window.
    pub peak_resonance_energy: f32,
    /// Timestamp associated with the resonance peak.
    pub peak_resonance_timestamp: f32,
    /// Fraction of frames within the digest that were considered quiescent.
    pub quiescence_ratio: f32,
}

struct MotionEmbeddingFilter {
    previous: Option<Tensor>,
    gain: f32,
}

impl MotionEmbeddingFilter {
    fn new(gain: f32) -> Self {
        Self {
            previous: None,
            gain: if gain.is_finite() { gain } else { 1.0 },
        }
    }

    fn process(&mut self, frame: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = frame.shape();
        let output = if let Some(previous) = &self.previous {
            let delta = frame.sub(previous)?;
            delta.scale(self.gain.max(0.0).max(f32::EPSILON))?
        } else {
            Tensor::zeros(rows, cols)?
        };
        self.previous = Some(frame.clone());
        Ok(output)
    }
}

struct ResonanceEnvelope {
    momentum: f32,
    accumulator: Option<Tensor>,
}

impl ResonanceEnvelope {
    fn new(momentum: f32) -> Self {
        Self {
            momentum: momentum.clamp(0.0, 1.0),
            accumulator: None,
        }
    }

    fn update(&mut self, volume: &ZSpaceVolume) -> PureResult<Tensor> {
        let flattened = Tensor::from_vec(
            volume.depth(),
            volume.height() * volume.width(),
            volume.voxels().to_vec(),
        )?;
        let updated = if let Some(previous) = &self.accumulator {
            let retained = previous.scale(1.0 - self.momentum)?;
            let injected = flattened.scale(self.momentum)?;
            retained.add(&injected)?
        } else {
            flattened.clone()
        };
        self.accumulator = Some(updated.clone());
        Ok(updated)
    }
}

struct TemporalStats {
    frames: usize,
    total_duration: f32,
    sum_drift: f32,
    sum_abs_drift: f32,
    sum_drift_sq: f32,
    sum_motion: f32,
    sum_motion_sq: f32,
    sum_energy: f32,
    sum_energy_sq: f32,
    sum_decay: f32,
    min_energy: f32,
    max_energy: f32,
    last_timestamp: Option<f32>,
    last_motion: Option<f32>,
    last_energy: Option<f32>,
    peak_motion: f32,
    peak_motion_timestamp: f32,
    peak_resonance: f32,
    peak_resonance_timestamp: f32,
    quiescent_frames: usize,
    quiescence_threshold: f32,
    history: VecDeque<TemporalSample>,
}

impl TemporalStats {
    fn with_threshold(threshold: f32) -> Self {
        Self {
            frames: 0,
            total_duration: 0.0,
            sum_drift: 0.0,
            sum_abs_drift: 0.0,
            sum_drift_sq: 0.0,
            sum_motion: 0.0,
            sum_motion_sq: 0.0,
            sum_energy: 0.0,
            sum_energy_sq: 0.0,
            sum_decay: 0.0,
            min_energy: 0.0,
            max_energy: 0.0,
            last_timestamp: None,
            last_motion: None,
            last_energy: None,
            peak_motion: 0.0,
            peak_motion_timestamp: 0.0,
            peak_resonance: 0.0,
            peak_resonance_timestamp: 0.0,
            quiescent_frames: 0,
            quiescence_threshold: threshold,
            history: VecDeque::new(),
        }
    }

    fn ingest(
        &mut self,
        timestamp: f32,
        motion_energy: f32,
        resonance_energy: f32,
    ) -> ChronoSnapshot {
        let dt = if let Some(previous) = self.last_timestamp {
            (timestamp - previous).max(0.0)
        } else {
            0.0
        };
        self.total_duration += dt;

        let drift = if let Some(previous) = self.last_motion {
            motion_energy - previous
        } else {
            0.0
        };
        self.sum_drift += drift;
        self.sum_abs_drift += drift.abs();
        self.sum_drift_sq += drift * drift;

        self.sum_motion += motion_energy;
        self.sum_motion_sq += motion_energy * motion_energy;

        self.sum_energy += resonance_energy;
        self.sum_energy_sq += resonance_energy * resonance_energy;

        let decay = if let Some(previous) = self.last_energy {
            if previous.abs() > f32::EPSILON {
                (previous - resonance_energy) / previous.abs()
            } else {
                0.0
            }
        } else {
            0.0
        };
        self.sum_decay += decay;

        if self.frames == 0 {
            self.min_energy = resonance_energy;
            self.max_energy = resonance_energy;
            self.peak_motion = motion_energy;
            self.peak_motion_timestamp = timestamp;
            self.peak_resonance = resonance_energy;
            self.peak_resonance_timestamp = timestamp;
        } else {
            self.min_energy = self.min_energy.min(resonance_energy);
            self.max_energy = self.max_energy.max(resonance_energy);
            if motion_energy > self.peak_motion {
                self.peak_motion = motion_energy;
                self.peak_motion_timestamp = timestamp;
            }
            if resonance_energy > self.peak_resonance {
                self.peak_resonance = resonance_energy;
                self.peak_resonance_timestamp = timestamp;
            }
        }

        if motion_energy <= self.quiescence_threshold {
            self.quiescent_frames = self.quiescent_frames.saturating_add(1);
        }

        self.frames = self.frames.saturating_add(1);
        self.last_timestamp = Some(timestamp);
        self.last_motion = Some(motion_energy);
        self.last_energy = Some(resonance_energy);

        self.history.push_back(TemporalSample {
            timestamp,
            motion: motion_energy,
            resonance: resonance_energy,
        });

        let frames_f32 = self.frames.max(1) as f32;
        let mean_energy = self.sum_energy / frames_f32;
        let mean_drift = self.sum_drift / frames_f32;
        let mean_abs_drift = self.sum_abs_drift / frames_f32;
        let drift_var = (self.sum_drift_sq / frames_f32) - mean_drift.powi(2);
        let energy_var = (self.sum_energy_sq / frames_f32) - mean_energy.powi(2);

        let summary = st_core::telemetry::chrono::ChronoSummary {
            frames: self.frames,
            duration: self.total_duration,
            latest_timestamp: timestamp,
            mean_drift,
            mean_abs_drift,
            drift_std: drift_var.max(0.0).sqrt(),
            mean_energy,
            energy_std: energy_var.max(0.0).sqrt(),
            mean_decay: self.sum_decay / frames_f32,
            min_energy: self.min_energy,
            max_energy: self.max_energy,
        };
        ChronoSnapshot::new(summary, dt)
    }

    fn digest(&self) -> TemporalDigest {
        if self.frames == 0 {
            return TemporalDigest::default();
        }
        let frames = self.frames as f32;
        let mean_motion = self.sum_motion / frames;
        let motion_var = (self.sum_motion_sq / frames) - mean_motion.powi(2);
        let mean_resonance = self.sum_energy / frames;
        let resonance_var = (self.sum_energy_sq / frames) - mean_resonance.powi(2);
        TemporalDigest {
            frames: self.frames,
            duration: self.total_duration,
            mean_motion_energy: mean_motion,
            motion_std: motion_var.max(0.0).sqrt(),
            mean_resonance_energy: mean_resonance,
            resonance_std: resonance_var.max(0.0).sqrt(),
            mean_resonance_decay: self.sum_decay / frames,
            min_resonance_energy: self.min_energy,
            max_resonance_energy: self.max_energy,
            peak_motion_energy: self.peak_motion,
            peak_motion_timestamp: self.peak_motion_timestamp,
            peak_resonance_energy: self.peak_resonance,
            peak_resonance_timestamp: self.peak_resonance_timestamp,
            quiescence_ratio: if self.frames > 0 {
                self.quiescent_frames as f32 / self.frames as f32
            } else {
                0.0
            },
        }
    }

    fn digest_window(&self, window: usize) -> TemporalDigest {
        if window == 0 || self.history.is_empty() {
            return TemporalDigest::default();
        }

        let limit = window.min(self.history.len());
        let mut frames = 0usize;
        let mut duration = 0.0f32;
        let mut sum_motion = 0.0f32;
        let mut sum_motion_sq = 0.0f32;
        let mut sum_resonance = 0.0f32;
        let mut sum_resonance_sq = 0.0f32;
        let mut sum_decay = 0.0f32;
        let mut min_resonance = f32::INFINITY;
        let mut max_resonance = f32::NEG_INFINITY;
        let mut peak_motion = f32::NEG_INFINITY;
        let mut peak_motion_timestamp = 0.0f32;
        let mut peak_resonance = f32::NEG_INFINITY;
        let mut peak_resonance_timestamp = 0.0f32;
        let mut quiescent = 0usize;

        let mut previous: Option<&TemporalSample> = None;

        for sample in self
            .history
            .iter()
            .rev()
            .take(limit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            frames += 1;
            sum_motion += sample.motion;
            sum_motion_sq += sample.motion * sample.motion;
            sum_resonance += sample.resonance;
            sum_resonance_sq += sample.resonance * sample.resonance;
            if sample.resonance < min_resonance {
                min_resonance = sample.resonance;
            }
            if sample.resonance > max_resonance {
                max_resonance = sample.resonance;
            }
            if sample.motion > peak_motion {
                peak_motion = sample.motion;
                peak_motion_timestamp = sample.timestamp;
            }
            if sample.resonance > peak_resonance {
                peak_resonance = sample.resonance;
                peak_resonance_timestamp = sample.timestamp;
            }
            if sample.motion <= self.quiescence_threshold {
                quiescent += 1;
            }
            if let Some(prev) = previous {
                let decay = if prev.resonance.abs() > f32::EPSILON {
                    (prev.resonance - sample.resonance) / prev.resonance.abs()
                } else {
                    0.0
                };
                sum_decay += decay;
                duration += (sample.timestamp - prev.timestamp).max(0.0);
            }
            previous = Some(sample);
        }

        if frames == 0 {
            return TemporalDigest::default();
        }

        let frames_f32 = frames as f32;
        let mean_motion = sum_motion / frames_f32;
        let motion_var = (sum_motion_sq / frames_f32) - mean_motion.powi(2);
        let mean_resonance = sum_resonance / frames_f32;
        let resonance_var = (sum_resonance_sq / frames_f32) - mean_resonance.powi(2);

        TemporalDigest {
            frames,
            duration,
            mean_motion_energy: mean_motion,
            motion_std: motion_var.max(0.0).sqrt(),
            mean_resonance_energy: mean_resonance,
            resonance_std: resonance_var.max(0.0).sqrt(),
            mean_resonance_decay: if frames > 1 {
                sum_decay / (frames_f32 - 1.0)
            } else {
                0.0
            },
            min_resonance_energy: min_resonance,
            max_resonance_energy: max_resonance,
            peak_motion_energy: peak_motion,
            peak_motion_timestamp,
            peak_resonance_energy: peak_resonance,
            peak_resonance_timestamp,
            quiescence_ratio: quiescent as f32 / frames_f32,
        }
    }
}

#[derive(Clone)]
struct TemporalSample {
    timestamp: f32,
    motion: f32,
    resonance: f32,
}

fn push_metric(atlas: &mut AtlasFrame, name: &str, value: f32) {
    if let Some(metric) = AtlasMetric::with_district(name, value, "temporal") {
        atlas.metrics.push(metric);
    }
}

fn tensor_energy(tensor: &Tensor) -> f32 {
    let data = tensor.data();
    if data.is_empty() {
        0.0
    } else {
        data.iter().map(|value| value.abs()).sum::<f32>() / data.len() as f32
    }
}

fn weight_entropy(weights: &[f32]) -> f32 {
    weights
        .iter()
        .filter(|weight| weight.is_finite() && **weight > f32::EPSILON)
        .map(|weight| -weight * weight.ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SequenceDecoder {
        frames: std::vec::IntoIter<DecodedFrame>,
    }

    impl SequenceDecoder {
        fn new(frames: Vec<DecodedFrame>) -> Self {
            Self {
                frames: frames.into_iter(),
            }
        }
    }

    impl VideoDecoder for SequenceDecoder {
        fn next_frame(&mut self) -> PureResult<Option<DecodedFrame>> {
            Ok(self.frames.next())
        }
    }

    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(2, values.len() / 2, values.to_vec()).unwrap()
    }

    #[test]
    fn pipeline_emits_annotations() {
        let frames = vec![
            DecodedFrame {
                timestamp: 0.0,
                tensor: tensor(&[0.0, 0.1, 0.2, 0.3]),
            },
            DecodedFrame {
                timestamp: 1.0 / 30.0,
                tensor: tensor(&[0.1, 0.2, 0.3, 0.4]),
            },
        ];
        let decoder = SequenceDecoder::new(frames);
        let mut pipeline = VideoPipeline::new(decoder, VideoPipelineConfig::default());
        let first = pipeline.next().unwrap().unwrap();
        assert_eq!(first.frame_index, 0);
        assert_eq!(first.stream.volume.depth(), 2);
        assert!(first
            .atlas_frame
            .metrics
            .iter()
            .any(|metric| metric.name == "z.motion_energy"));
        assert_eq!(first.temporal_digest.frames, 1);
        assert_eq!(first.window_digest.frames, 1);
        assert!(pipeline.last_volume().is_some());
        assert_eq!(pipeline.atlas_timeline().len(), 1);
        let second = pipeline.next().unwrap().unwrap();
        assert_eq!(second.frame_index, 1);
        assert!(second
            .atlas_frame
            .metrics
            .iter()
            .any(|metric| metric.name == "z.weight_entropy"));
        assert!(second.temporal_digest.frames >= 2);
        assert!(second.window_digest.frames >= 1);
        assert!(pipeline.temporal_digest().frames >= 2);
        assert!(pipeline.temporal_digest_window(1).frames >= 1);
        assert_eq!(pipeline.atlas_timeline().len(), 2);
        assert!(pipeline.next().unwrap().is_none());
    }
}
