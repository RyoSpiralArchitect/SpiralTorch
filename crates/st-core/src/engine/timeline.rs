// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

/// Identifier describing a scheduled kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSlot {
    pub label: String,
    pub stream: u32,
    pub start: f32,
    pub end: f32,
}

impl KernelSlot {
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }
}

/// Error emitted by [`TimelineScheduler`] when a kernel cannot be registered.
#[derive(Debug, Clone, PartialEq)]
pub enum TimelineError {
    /// Duration must be finite and strictly positive.
    InvalidDuration(f32),
    /// Preferred start, duration, or resulting span contained a non-finite number.
    NonFinite,
    /// A warp/transformation attempted an invalid manipulation of the timeline.
    InvalidWarp(&'static str),
}

impl fmt::Display for TimelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDuration(dur) => {
                write!(f, "kernel duration must be > 0, got {dur}")
            }
            Self::NonFinite => write!(f, "kernel span must contain only finite values"),
            Self::InvalidWarp(msg) => write!(f, "invalid timeline warp: {msg}"),
        }
    }
}

impl std::error::Error for TimelineError {}

/// Deterministic scheduler that packs kernels onto a time axis per stream.
#[derive(Default, Debug, Clone)]
pub struct TimelineScheduler {
    streams: BTreeMap<u32, Vec<KernelSlot>>,
    timeline: Vec<KernelSlot>,
    epsilon: f32,
}

/// Cursor supplied to timeline warp functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeCursor {
    /// Absolute time on the original axis.
    pub time: f32,
    /// Normalised progress over the original makespan (0 at the first start).
    pub progress: f32,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    a + (b - a) * t
}

fn warp_progress(progress: f32, curvature: f32) -> f32 {
    let progress = progress.clamp(0.0, 1.0);
    if curvature.abs() < 1e-6 {
        return progress;
    }

    let intensity = curvature.abs().clamp(0.0, 0.999);
    if curvature > 0.0 {
        let exponent = 1.0 + 3.0 * intensity;
        1.0 - (1.0 - progress).powf(exponent)
    } else {
        let exponent = 1.0 + 3.0 * intensity;
        progress.powf(exponent)
    }
}

impl TimelineScheduler {
    /// Construct a scheduler using the provided gap tolerance.
    pub fn new(epsilon: f32) -> Self {
        Self {
            streams: BTreeMap::new(),
            timeline: Vec::new(),
            epsilon: epsilon.max(0.0),
        }
    }

    fn next_start(slots: &[KernelSlot], preferred_start: f32, duration: f32, epsilon: f32) -> f32 {
        let mut start = preferred_start.max(0.0);
        for slot in slots {
            if start + duration <= slot.start + epsilon {
                break;
            }
            if start < slot.end {
                start = slot.end;
            }
        }
        start
    }

    fn push_sorted(vec: &mut Vec<KernelSlot>, slot: KernelSlot) {
        let idx = vec
            .binary_search_by(|existing| match existing.start.total_cmp(&slot.start) {
                Ordering::Equal => existing.stream.cmp(&slot.stream),
                order => order,
            })
            .unwrap_or_else(|idx| idx);
        vec.insert(idx, slot);
    }

    /// Schedule a new kernel span on the requested stream.
    pub fn schedule_kernel<S: Into<String>>(
        &mut self,
        label: S,
        stream: u32,
        preferred_start: f32,
        duration: f32,
    ) -> Result<KernelSlot, TimelineError> {
        if !duration.is_finite() || duration <= 0.0 {
            return Err(TimelineError::InvalidDuration(duration));
        }
        if !preferred_start.is_finite() {
            return Err(TimelineError::NonFinite);
        }

        let slots = self.streams.entry(stream).or_default();
        let start = Self::next_start(slots, preferred_start, duration, self.epsilon);
        let end = start + duration;
        if !start.is_finite() || !end.is_finite() {
            return Err(TimelineError::NonFinite);
        }

        let slot = KernelSlot {
            label: label.into(),
            stream,
            start,
            end,
        };

        Self::push_sorted(slots, slot.clone());
        Self::push_sorted(&mut self.timeline, slot.clone());

        Ok(slot)
    }

    /// Returns the scheduled kernels ordered by time.
    pub fn iter(&self) -> impl Iterator<Item = &KernelSlot> {
        self.timeline.iter()
    }

    /// Returns the kernels scheduled on the given stream.
    pub fn stream(&self, stream: u32) -> Option<&[KernelSlot]> {
        self.streams.get(&stream).map(|slots| slots.as_slice())
    }

    /// Returns the makespan of the timeline, if at least one kernel exists.
    pub fn makespan(&self) -> Option<f32> {
        let first = self.timeline.first()?;
        let last = self.timeline.last()?;
        Some((last.end - first.start).max(0.0))
    }

    /// Computes the occupancy ratio for a stream.
    pub fn utilization(&self, stream: u32) -> Option<f32> {
        let slots = self.streams.get(&stream)?;
        if slots.is_empty() {
            return Some(0.0);
        }
        let total: f32 = slots.iter().map(KernelSlot::duration).sum();
        let span = (slots.last()?.end - slots.first()?.start).max(f32::EPSILON);
        Some((total / span).min(1.0))
    }

    fn warp_internal<F>(&mut self, mut mapper: F) -> Result<(), TimelineError>
    where
        F: FnMut(&KernelSlot, TimeCursor) -> Result<f32, TimelineError>,
    {
        let original = if self.timeline.is_empty() {
            return Ok(());
        } else {
            self.timeline.clone()
        };

        let origin = original.first().map(|slot| slot.start).unwrap_or(0.0);
        let span = (original.last().map(|slot| slot.end - origin).unwrap_or(0.0)).max(f32::EPSILON);

        let mut warped_slots = Vec::with_capacity(original.len());

        for slot in &original {
            let start_cursor = TimeCursor {
                time: slot.start,
                progress: (slot.start - origin) / span,
            };
            let end_cursor = TimeCursor {
                time: slot.end,
                progress: (slot.end - origin) / span,
            };

            let new_start = mapper(slot, start_cursor)?;
            let new_end = mapper(slot, end_cursor)?;

            if !new_start.is_finite() || !new_end.is_finite() {
                return Err(TimelineError::NonFinite);
            }

            if new_end < new_start - self.epsilon {
                return Err(TimelineError::InvalidWarp(
                    "warp produced a negative duration span",
                ));
            }

            warped_slots.push(KernelSlot {
                label: slot.label.clone(),
                stream: slot.stream,
                start: new_start,
                end: new_end,
            });
        }

        self.streams.clear();
        self.timeline.clear();

        for slot in warped_slots {
            let stream_slots = self.streams.entry(slot.stream).or_default();
            Self::push_sorted(stream_slots, slot.clone());
            Self::push_sorted(&mut self.timeline, slot);
        }

        Ok(())
    }

    /// Applies a time warp to the existing timeline using a mapper that receives the
    /// current slot and its cursor on the original axis, returning the warped absolute
    /// time. The mapper is invoked for both the start and end of every slot.
    pub fn warp_with<F>(&mut self, mut mapper: F) -> Result<(), TimelineError>
    where
        F: FnMut(&KernelSlot, TimeCursor) -> f32,
    {
        self.warp_internal(|slot, cursor| Ok(mapper(slot, cursor)))
    }

    /// Remaps the timeline by modifying the normalised progress (0..1) of each cursor.
    /// The provided closure returns the desired progress which is then projected back
    /// onto the original axis. Values outside 0..1 are allowed and will stretch or
    /// compress the timeline accordingly.
    pub fn remap_progress<F>(&mut self, mut remap: F) -> Result<(), TimelineError>
    where
        F: FnMut(&KernelSlot, f32) -> f32,
    {
        let (origin, span) = match (self.timeline.first(), self.timeline.last()) {
            (Some(first), Some(last)) => {
                let span = (last.end - first.start).max(f32::EPSILON);
                (first.start, span)
            }
            _ => return Ok(()),
        };

        self.warp_internal(|slot, cursor| {
            let new_progress = remap(slot, cursor.progress);
            if !new_progress.is_finite() {
                return Err(TimelineError::NonFinite);
            }
            Ok(origin + span * new_progress)
        })
    }

    /// Uniformly shifts all slots by `delta` along the axis. Negative values rewind the
    /// schedule while positive values fast-forward it.
    pub fn translate(&mut self, delta: f32) -> Result<(), TimelineError> {
        if !delta.is_finite() {
            return Err(TimelineError::NonFinite);
        }
        self.warp_with(|_, cursor| cursor.time + delta)
    }

    /// Scales the timeline about the first scheduled start by `factor`. Values greater
    /// than 1.0 lengthen the perceived duration, values between 0 and 1.0 compress it.
    pub fn stretch(&mut self, factor: f32) -> Result<(), TimelineError> {
        let origin = match self.timeline.first() {
            Some(slot) => slot.start,
            None => return Ok(()),
        };

        self.scale_about(origin, factor)
    }

    /// Scales the timeline about the provided anchor point.
    pub fn scale_about(&mut self, anchor: f32, factor: f32) -> Result<(), TimelineError> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(TimelineError::InvalidWarp(
                "scale factor must be finite and greater than zero",
            ));
        }

        self.warp_with(|_, cursor| anchor + (cursor.time - anchor) * factor)
    }
}

/// Signals describing the training dynamics that should influence timeline warps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineSignal {
    /// Current learning rate used by the optimiser.
    pub learning_rate: f32,
    /// Ratio of the most recent loss against the baseline (1.0 means unchanged).
    pub loss_ratio: Option<f32>,
    /// Normalised stability estimate (1.0 perfectly stable, 0.0 chaotic).
    pub stability: Option<f32>,
    /// Signed velocity of the optimisation signal (positive when improving).
    pub velocity: Option<f32>,
    /// Normalised training progress (0.0 start, 1.0 complete).
    pub progress: Option<f32>,
    /// Normalised GPU memory pressure (0.0 idle, 1.0 at the limit).
    pub memory_pressure: Option<f32>,
    /// Ratio of realised throughput vs the desired reference (1.0 ideal).
    pub throughput_ratio: Option<f32>,
}

impl TimelineSignal {
    pub fn with_learning_rate(lr: f32) -> Self {
        Self {
            learning_rate: lr,
            loss_ratio: None,
            stability: None,
            velocity: None,
            progress: None,
            memory_pressure: None,
            throughput_ratio: None,
        }
    }
}

/// Manual adjustments applied on top of the automatic warp field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManualWarp {
    /// Desired tempo multiplier for the timeline (1.0 keeps the current tempo).
    pub tempo: f32,
    /// Desired translation in time units (positive fast-forwards, negative rewinds).
    pub offset: f32,
    /// Blend ratio between the automatic warp (0.0) and manual override (1.0).
    pub blend: f32,
    /// Desired curvature of the Z-space timeline (-1.0 compresses the future, +1.0 stretches it).
    pub curvature: f32,
}

impl ManualWarp {
    pub fn new(tempo: f32, offset: f32, blend: f32) -> Self {
        Self {
            tempo,
            offset,
            blend,
            curvature: 0.0,
        }
    }

    pub fn with_curvature(tempo: f32, offset: f32, blend: f32, curvature: f32) -> Self {
        Self {
            tempo,
            offset,
            blend,
            curvature,
        }
    }
}

impl Default for ManualWarp {
    fn default() -> Self {
        Self {
            tempo: 1.0,
            offset: 0.0,
            blend: 0.0,
            curvature: 0.0,
        }
    }
}

/// Configuration governing how the automatic warp reacts to runtime signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineAutoConfig {
    pub lr_floor: f32,
    pub lr_ceiling: f32,
    pub lr_sensitivity: f32,
    pub stability_sensitivity: f32,
    pub loss_drag: f32,
    pub velocity_push: f32,
    pub progress_weight: f32,
    pub smoothing: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub max_shift: f32,
    pub memory_sensitivity: f32,
    pub throughput_sensitivity: f32,
    pub curvature_velocity: f32,
    pub curvature_progress: f32,
    pub curvature_smoothing: f32,
    pub max_curvature: f32,
}

impl Default for TimelineAutoConfig {
    fn default() -> Self {
        Self {
            lr_floor: 1e-6,
            lr_ceiling: 10.0,
            lr_sensitivity: 0.75,
            stability_sensitivity: 0.5,
            loss_drag: 0.4,
            velocity_push: 0.5,
            progress_weight: 0.3,
            smoothing: 0.35,
            min_scale: 0.25,
            max_scale: 4.0,
            max_shift: 8.0,
            memory_sensitivity: 0.6,
            throughput_sensitivity: 0.5,
            curvature_velocity: 0.35,
            curvature_progress: 0.25,
            curvature_smoothing: 0.25,
            max_curvature: 0.85,
        }
    }
}

/// Controller that modulates the scheduler timeline according to training signals.
#[derive(Debug, Clone)]
pub struct TimelineWarpController {
    config: TimelineAutoConfig,
    baseline_lr: Option<f32>,
    lr_state: f32,
    stability_state: f32,
    memory_state: f32,
    throughput_state: f32,
    desired_scale: f32,
    desired_offset: f32,
    desired_curvature: f32,
    current_scale: f32,
    current_offset: f32,
    current_curvature: f32,
    manual: ManualWarp,
    initialised: bool,
}

impl TimelineWarpController {
    pub fn new(config: TimelineAutoConfig) -> Self {
        Self {
            config,
            baseline_lr: None,
            lr_state: 1.0,
            stability_state: 1.0,
            memory_state: 0.3,
            throughput_state: 1.0,
            desired_scale: 1.0,
            desired_offset: 0.0,
            desired_curvature: 0.0,
            current_scale: 1.0,
            current_offset: 0.0,
            current_curvature: 0.0,
            manual: ManualWarp::default(),
            initialised: false,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(TimelineAutoConfig::default())
    }

    pub fn config(&self) -> &TimelineAutoConfig {
        &self.config
    }

    pub fn manual_control(&self) -> ManualWarp {
        self.manual
    }

    pub fn set_manual_control(&mut self, manual: ManualWarp) {
        self.manual = manual;
    }

    pub fn reset(&mut self) {
        self.baseline_lr = None;
        self.lr_state = 1.0;
        self.stability_state = 1.0;
        self.memory_state = 0.3;
        self.throughput_state = 1.0;
        self.desired_scale = 1.0;
        self.desired_offset = 0.0;
        self.desired_curvature = 0.0;
        self.current_scale = 1.0;
        self.current_offset = 0.0;
        self.current_curvature = 0.0;
        self.initialised = false;
    }

    fn update_states(&mut self, signal: &TimelineSignal) {
        let lr = signal
            .learning_rate
            .clamp(self.config.lr_floor, self.config.lr_ceiling);
        let baseline = self.baseline_lr.get_or_insert(lr);
        let ratio = (lr / *baseline).clamp(0.1, 10.0);
        self.lr_state += self.config.smoothing * (ratio - self.lr_state);

        let stability = signal.stability.unwrap_or(0.7).clamp(0.0, 1.0);
        self.stability_state += self.config.smoothing * (stability - self.stability_state);

        let memory = signal
            .memory_pressure
            .unwrap_or(self.memory_state)
            .clamp(0.0, 1.0);
        self.memory_state += self.config.smoothing * (memory - self.memory_state);

        let throughput = signal
            .throughput_ratio
            .unwrap_or(self.throughput_state)
            .clamp(0.1, 10.0);
        self.throughput_state += self.config.smoothing * (throughput - self.throughput_state);
    }

    fn compute_auto_scale(&self) -> f32 {
        let mut scale = if self.lr_state >= 1.0 {
            1.0 / (1.0 + self.config.lr_sensitivity * (self.lr_state - 1.0))
        } else {
            1.0 + self.config.lr_sensitivity * (1.0 - self.lr_state)
        };
        scale *= 1.0 + self.config.stability_sensitivity * (1.0 - self.stability_state);

        let memory_term = 1.0 + self.config.memory_sensitivity * (self.memory_state - 0.5);
        let throughput_term = if self.throughput_state >= 1.0 {
            1.0 / (1.0 + self.config.throughput_sensitivity * (self.throughput_state - 1.0))
        } else {
            1.0 + self.config.throughput_sensitivity * (1.0 - self.throughput_state)
        };

        scale *= memory_term.clamp(0.25, 2.5);
        scale *= throughput_term.clamp(0.25, 2.5);
        scale.clamp(self.config.min_scale, self.config.max_scale)
    }

    fn compute_auto_offset(&self, signal: &TimelineSignal) -> f32 {
        let loss_ratio = signal.loss_ratio.unwrap_or(1.0).clamp(0.1, 10.0);
        let velocity = signal.velocity.unwrap_or(0.0).clamp(-1.0, 1.0);
        let progress = signal.progress.unwrap_or(0.5).clamp(0.0, 1.0);
        let memory = signal
            .memory_pressure
            .unwrap_or(self.memory_state)
            .clamp(0.0, 1.0);

        let mut offset = if loss_ratio >= 1.0 {
            -self.config.loss_drag * (loss_ratio - 1.0)
        } else {
            self.config.loss_drag * (1.0 - loss_ratio)
        };

        offset += self.config.velocity_push * velocity;
        offset -= self.config.memory_sensitivity * (memory - 0.5);

        let mid_weight = 1.0 + self.config.progress_weight * (0.5 - (progress - 0.5).abs());
        offset *= mid_weight;
        offset.clamp(-self.config.max_shift, self.config.max_shift)
    }

    fn compute_auto_curvature(&self, signal: &TimelineSignal) -> f32 {
        let memory = signal
            .memory_pressure
            .unwrap_or(self.memory_state)
            .clamp(0.0, 1.0);
        let throughput = signal
            .throughput_ratio
            .unwrap_or(self.throughput_state)
            .clamp(0.1, 10.0);
        let velocity = signal.velocity.unwrap_or(0.0).clamp(-1.0, 1.0);
        let progress = signal.progress.unwrap_or(0.5).clamp(0.0, 1.0);
        let stability = signal
            .stability
            .unwrap_or(self.stability_state)
            .clamp(0.0, 1.0);

        let memory_term = (memory - 0.5) * self.config.memory_sensitivity;
        let throughput_term = (1.0 - throughput).tanh() * self.config.throughput_sensitivity;
        let velocity_term = velocity * self.config.curvature_velocity;
        let progress_term = (progress - 0.5) * self.config.curvature_progress;
        let stability_term = (0.5 - stability) * 0.5 * self.config.curvature_velocity;

        let curvature =
            memory_term + throughput_term + velocity_term + progress_term + stability_term;
        curvature.clamp(-self.config.max_curvature, self.config.max_curvature)
    }

    fn update_targets(&mut self, signal: &TimelineSignal) {
        let auto_scale = self.compute_auto_scale();
        let auto_offset = self.compute_auto_offset(signal);
        let auto_curvature = self.compute_auto_curvature(signal);
        let manual_scale = self
            .manual
            .tempo
            .clamp(self.config.min_scale, self.config.max_scale);
        let manual_offset = self
            .manual
            .offset
            .clamp(-self.config.max_shift, self.config.max_shift);
        let manual_curvature = self
            .manual
            .curvature
            .clamp(-self.config.max_curvature, self.config.max_curvature);
        let blend = self.manual.blend.clamp(0.0, 1.0);

        let blended_scale = lerp(auto_scale, manual_scale, blend);
        let blended_offset = lerp(auto_offset, manual_offset, blend);
        let blended_curvature = lerp(auto_curvature, manual_curvature, blend);

        if !self.initialised {
            self.desired_scale = blended_scale;
            self.desired_offset = blended_offset;
            self.desired_curvature = blended_curvature;
            self.initialised = true;
        } else {
            self.desired_scale += self.config.smoothing * (blended_scale - self.desired_scale);
            self.desired_offset += self.config.smoothing * (blended_offset - self.desired_offset);
            self.desired_curvature +=
                self.config.curvature_smoothing * (blended_curvature - self.desired_curvature);
        }
    }

    fn apply_delta(&mut self, scheduler: &mut TimelineScheduler) -> Result<(), TimelineError> {
        let scale_ratio = (self.desired_scale / self.current_scale).clamp(
            self.config.min_scale / self.current_scale,
            self.config.max_scale / self.current_scale,
        );
        if (scale_ratio - 1.0).abs() > 1e-6 {
            scheduler.stretch(scale_ratio)?;
            self.current_scale *= scale_ratio;
        }

        let translation = self.desired_offset - self.current_offset;
        if translation.abs() > 1e-6 {
            scheduler.translate(translation)?;
            self.current_offset += translation;
        }

        let curvature_delta = self.desired_curvature - self.current_curvature;
        if curvature_delta.abs() > 1e-4 {
            let step = (self.current_curvature + curvature_delta)
                .clamp(-self.config.max_curvature, self.config.max_curvature);
            scheduler.remap_progress(|_, progress| warp_progress(progress, step))?;
            self.current_curvature = step;
        }

        Ok(())
    }

    /// Applies the automatic + manual warp to the provided scheduler timeline.
    pub fn apply(
        &mut self,
        scheduler: &mut TimelineScheduler,
        signal: &TimelineSignal,
    ) -> Result<(), TimelineError> {
        self.update_states(signal);
        self.update_targets(signal);
        self.apply_delta(scheduler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_without_overlap_on_same_stream() {
        let mut scheduler = TimelineScheduler::new(0.0);
        let first = scheduler
            .schedule_kernel("fft", 0, 0.0, 2.0)
            .expect("first kernel");
        let second = scheduler
            .schedule_kernel("conv", 0, 1.0, 1.5)
            .expect("second kernel");
        assert_eq!(first.start, 0.0);
        assert_eq!(first.end, 2.0);
        assert_eq!(second.start, 2.0);
        assert_eq!(second.end, 3.5);
        let timeline: Vec<_> = scheduler.iter().map(|slot| slot.label.clone()).collect();
        assert_eq!(timeline, vec!["fft", "conv"]);
    }

    #[test]
    fn concurrent_streams_do_not_interfere() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 2.0)
            .expect("stream 0");
        let slot = scheduler
            .schedule_kernel("conv", 1, 0.5, 1.0)
            .expect("stream 1");
        assert!((slot.start - 0.5).abs() < 1e-6);
        assert!(scheduler.stream(0).unwrap().len() == 1);
        assert!(scheduler.stream(1).unwrap().len() == 1);
    }

    #[test]
    fn computes_utilization_per_stream() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 2.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 2.0, 2.0)
            .expect("second");
        let util = scheduler.utilization(0).unwrap();
        assert!((util - 1.0).abs() < 1e-3);
        let makespan = scheduler.makespan().unwrap();
        assert!((makespan - 4.0).abs() < 1e-3);
    }

    #[test]
    fn stretching_lengthens_timeline() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 1.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 1.0, 1.0)
            .expect("second");

        scheduler.stretch(2.0).expect("stretched");

        let slots = scheduler.stream(0).unwrap();
        assert!((slots[0].start - 0.0).abs() < 1e-6);
        assert!((slots[0].end - 2.0).abs() < 1e-6);
        assert!((slots[1].start - 2.0).abs() < 1e-6);
        assert!((slots[1].end - 4.0).abs() < 1e-6);
    }

    #[test]
    fn translate_allows_rewind_and_fast_forward() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 1.0, 1.0)
            .expect("first");

        scheduler.translate(-0.5).expect("rewind");
        let slot = scheduler.stream(0).unwrap()[0].clone();
        assert!((slot.start - 0.5).abs() < 1e-6);
        assert!((slot.end - 1.5).abs() < 1e-6);

        scheduler.translate(1.0).expect("fast forward");
        let slot = scheduler.stream(0).unwrap()[0].clone();
        assert!((slot.start - 1.5).abs() < 1e-6);
        assert!((slot.end - 2.5).abs() < 1e-6);
    }

    #[test]
    fn remap_progress_follows_learning_curve() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 1.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 1.0, 1.0)
            .expect("second");

        scheduler
            .remap_progress(|_, progress| {
                if progress < 0.5 {
                    progress * 1.5
                } else {
                    0.75 + (progress - 0.5) * 0.5
                }
            })
            .expect("remap");

        let slots = scheduler.stream(0).unwrap();
        assert!((slots[0].start - 0.0).abs() < 1e-6);
        assert!((slots[0].end - 1.5).abs() < 1e-6);
        assert!((slots[1].start - 1.5).abs() < 1e-6);
        assert!((slots[1].end - 2.0).abs() < 1e-6);
    }

    #[test]
    fn auto_controller_compresses_with_high_learning_rate() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 1.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 1.0, 1.0)
            .expect("second");

        let mut config = TimelineAutoConfig::default();
        config.smoothing = 1.0;
        config.stability_sensitivity = 0.0;
        let mut controller = TimelineWarpController::new(config);

        let baseline = TimelineSignal {
            learning_rate: 0.01,
            loss_ratio: Some(1.0),
            stability: Some(1.0),
            velocity: Some(0.0),
            progress: Some(0.5),
            memory_pressure: Some(0.2),
            throughput_ratio: Some(1.0),
        };
        controller
            .apply(&mut scheduler, &baseline)
            .expect("baseline");
        let initial_span = scheduler.makespan().unwrap();

        let mut fast = baseline;
        fast.learning_rate = 0.05;
        controller.apply(&mut scheduler, &fast).expect("fast warp");

        let compressed_span = scheduler.makespan().unwrap();
        assert!(compressed_span < initial_span);
    }

    #[test]
    fn auto_controller_rewinds_on_loss_spike() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.5, 1.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 1.5, 1.0)
            .expect("second");

        let mut config = TimelineAutoConfig::default();
        config.smoothing = 1.0;
        config.lr_sensitivity = 0.0;
        config.stability_sensitivity = 0.0;
        let mut controller = TimelineWarpController::new(config);

        let baseline = TimelineSignal {
            learning_rate: 0.01,
            loss_ratio: Some(1.0),
            stability: Some(1.0),
            velocity: Some(0.0),
            progress: Some(0.5),
            memory_pressure: Some(0.2),
            throughput_ratio: Some(1.0),
        };
        controller
            .apply(&mut scheduler, &baseline)
            .expect("baseline");
        let before = scheduler.stream(0).unwrap()[0].start;

        let mut spiking = baseline;
        spiking.loss_ratio = Some(1.5);
        controller.apply(&mut scheduler, &spiking).expect("rewind");

        let after = scheduler.stream(0).unwrap()[0].start;
        assert!(after < before);
    }

    #[test]
    fn manual_control_overrides_automatic_field() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 1.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 1.0, 1.0)
            .expect("second");

        let mut config = TimelineAutoConfig::default();
        config.smoothing = 1.0;
        let mut controller = TimelineWarpController::new(config);
        controller.set_manual_control(ManualWarp::new(1.2, 0.5, 1.0));

        let signal = TimelineSignal {
            learning_rate: 0.01,
            loss_ratio: Some(1.0),
            stability: Some(1.0),
            velocity: Some(0.0),
            progress: Some(0.5),
            memory_pressure: Some(0.2),
            throughput_ratio: Some(1.0),
        };
        controller.apply(&mut scheduler, &signal).expect("manual");

        let slots = scheduler.stream(0).unwrap();
        assert!((slots[0].start - 0.5).abs() < 1e-4);
        assert!((slots[0].duration() - 1.2).abs() < 1e-4);
        assert!((slots[1].start - 1.7).abs() < 1e-4);
    }

    #[test]
    fn manual_curvature_bends_timeline() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 1.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 1.0, 1.0)
            .expect("second");

        let mut config = TimelineAutoConfig::default();
        config.smoothing = 1.0;
        config.memory_sensitivity = 0.0;
        config.throughput_sensitivity = 0.0;
        let mut controller = TimelineWarpController::new(config);
        controller.set_manual_control(ManualWarp::with_curvature(1.0, 0.0, 1.0, 0.7));

        let signal = TimelineSignal {
            learning_rate: 0.01,
            loss_ratio: Some(1.0),
            stability: Some(1.0),
            velocity: Some(0.0),
            progress: Some(0.5),
            memory_pressure: Some(0.2),
            throughput_ratio: Some(1.0),
        };

        controller.apply(&mut scheduler, &signal).expect("manual");

        let slots = scheduler.stream(0).unwrap();
        assert!(slots[0].duration() > 1.0);
        assert!(slots[1].duration() < 1.0);
    }

    #[test]
    fn auto_controller_relaxes_under_memory_pressure() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 1.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 1.0, 1.0)
            .expect("second");

        let mut config = TimelineAutoConfig::default();
        config.smoothing = 1.0;
        config.throughput_sensitivity = 0.0;
        let mut controller = TimelineWarpController::new(config);

        let baseline = TimelineSignal {
            learning_rate: 0.01,
            loss_ratio: Some(1.0),
            stability: Some(1.0),
            velocity: Some(0.0),
            progress: Some(0.5),
            memory_pressure: Some(0.2),
            throughput_ratio: Some(1.0),
        };

        controller
            .apply(&mut scheduler, &baseline)
            .expect("baseline");
        let before = scheduler.makespan().unwrap();

        let mut pressured = baseline;
        pressured.memory_pressure = Some(0.95);
        controller
            .apply(&mut scheduler, &pressured)
            .expect("pressure warp");

        let after = scheduler.makespan().unwrap();
        assert!(after > before);
    }

    #[test]
    fn auto_controller_curves_progress_for_future_boost() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 1.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 1.0, 1.0)
            .expect("second");

        let mut config = TimelineAutoConfig::default();
        config.smoothing = 1.0;
        config.memory_sensitivity = 0.0;
        config.throughput_sensitivity = 0.0;
        let mut controller = TimelineWarpController::new(config);

        let mut signal = TimelineSignal {
            learning_rate: 0.01,
            loss_ratio: Some(1.0),
            stability: Some(0.5),
            velocity: Some(-0.8),
            progress: Some(0.2),
            memory_pressure: Some(0.2),
            throughput_ratio: Some(0.6),
        };

        controller
            .apply(&mut scheduler, &signal)
            .expect("first warp");
        let slots = scheduler.stream(0).unwrap();
        assert!(slots[0].start < slots[1].start);
        let delta = slots[1].start - slots[0].end;
        assert!(delta < 1.0);
        let first_duration = slots[0].duration();
        let second_duration = slots[1].duration();

        signal.velocity = Some(0.9);
        signal.progress = Some(0.8);
        signal.throughput_ratio = Some(1.5);
        controller
            .apply(&mut scheduler, &signal)
            .expect("second warp");

        let slots = scheduler.stream(0).unwrap();
        let new_delta = slots[1].start - slots[0].end;
        let new_first = slots[0].duration();
        let new_second = slots[1].duration();
        assert!(new_delta >= delta);
        assert!(new_first < first_duration);
        assert!(new_second > second_duration);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let mut scheduler = TimelineScheduler::new(0.0);
        let err = scheduler.schedule_kernel("bad", 0, 0.0, 0.0).unwrap_err();
        assert!(matches!(err, TimelineError::InvalidDuration(_)));
        let err = scheduler
            .schedule_kernel("nan", 0, f32::NAN, 1.0)
            .unwrap_err();
        assert!(matches!(err, TimelineError::NonFinite));
    }
}
