// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use crate::util::timewarp::{TemporalWarp, TemporalWarpError};

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
}

impl fmt::Display for TimelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDuration(dur) => {
                write!(f, "kernel duration must be > 0, got {dur}")
            }
            Self::NonFinite => write!(f, "kernel span must contain only finite values"),
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

    fn resync_streams(&mut self) {
        self.timeline
            .sort_by(|a, b| match a.start.total_cmp(&b.start) {
                Ordering::Equal => a.stream.cmp(&b.stream),
                order => order,
            });
        let mut rebuilt: BTreeMap<u32, Vec<KernelSlot>> = BTreeMap::new();
        for slot in &self.timeline {
            rebuilt.entry(slot.stream).or_default().push(slot.clone());
        }
        self.streams = rebuilt;
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

    /// Applies an affine warp to the scheduler's time axis.
    pub fn warp_time_axis(&mut self, warp: TemporalWarp) -> Result<(), TemporalWarpError> {
        if self.timeline.is_empty() {
            return Err(TemporalWarpError::Empty);
        }
        warp.validate()?;
        for slot in &mut self.timeline {
            let start = warp.apply(slot.start);
            let end = warp.apply(slot.end);
            if !start.is_finite() || !end.is_finite() {
                return Err(TemporalWarpError::NonFinite);
            }
            if end <= start {
                return Err(TemporalWarpError::Degenerate);
            }
            slot.start = start;
            slot.end = end;
        }
        self.resync_streams();
        Ok(())
    }

    /// Multiplies the timeline span by the provided factor.
    pub fn dilate(&mut self, factor: f32) -> Result<(), TemporalWarpError> {
        self.warp_time_axis(TemporalWarp::dilation(factor))
    }

    /// Shifts the entire timeline by the provided offset.
    pub fn shift(&mut self, offset: f32) -> Result<(), TemporalWarpError> {
        if self.timeline.is_empty() {
            return Err(TemporalWarpError::Empty);
        }
        if !offset.is_finite() {
            return Err(TemporalWarpError::NonFinite);
        }
        for slot in &mut self.timeline {
            slot.start += offset;
            slot.end += offset;
        }
        self.resync_streams();
        Ok(())
    }

    /// Rescales the timeline so the latest kernel sits at `progress * span`.
    pub fn align_to_progress(&mut self, progress: f32, span: f32) -> Result<(), TemporalWarpError> {
        if self.timeline.is_empty() {
            return Err(TemporalWarpError::Empty);
        }
        if !progress.is_finite() || !span.is_finite() {
            return Err(TemporalWarpError::NonFinite);
        }
        let span = span.max(f32::EPSILON);
        let progress = progress.max(0.0);
        let first_start = self.timeline.first().map(|slot| slot.start).unwrap_or(0.0);
        let last_end = self.timeline.last().map(|slot| slot.end).unwrap_or(0.0);
        let makespan = (last_end - first_start).max(f32::EPSILON);
        let target_span = (span * progress).max(f32::EPSILON);
        let factor = target_span / makespan;
        self.dilate(factor)?;
        let new_start = self.timeline.first().map(|slot| slot.start).unwrap_or(0.0);
        if new_start.abs() > f32::EPSILON {
            self.shift(-new_start)?;
        }
        Ok(())
    }

    /// Drops any kernels that extend beyond the provided timestamp.
    pub fn rewind_to(&mut self, time: f32) -> usize {
        if self.timeline.is_empty() {
            return 0;
        }
        let mut removed = 0;
        while let Some(last) = self.timeline.last() {
            if last.end <= time + self.epsilon {
                break;
            }
            self.timeline.pop();
            removed += 1;
        }
        if removed > 0 {
            self.resync_streams();
        }
        removed
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
    fn rejects_invalid_inputs() {
        let mut scheduler = TimelineScheduler::new(0.0);
        let err = scheduler.schedule_kernel("bad", 0, 0.0, 0.0).unwrap_err();
        assert!(matches!(err, TimelineError::InvalidDuration(_)));
        let err = scheduler
            .schedule_kernel("nan", 0, f32::NAN, 1.0)
            .unwrap_err();
        assert!(matches!(err, TimelineError::NonFinite));
    }

    #[test]
    fn dilates_and_aligns_to_progress() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler
            .schedule_kernel("fft", 0, 0.0, 2.0)
            .expect("first");
        scheduler
            .schedule_kernel("conv", 0, 2.0, 2.0)
            .expect("second");
        scheduler.align_to_progress(0.5, 10.0).expect("align");
        let first = scheduler.timeline.first().unwrap();
        let last = scheduler.timeline.last().unwrap();
        assert!((first.start - 0.0).abs() < 1e-6);
        assert!((last.end - 5.0).abs() < 1e-5);
    }

    #[test]
    fn rewinds_trailing_kernels() {
        let mut scheduler = TimelineScheduler::new(0.0);
        scheduler.schedule_kernel("a", 0, 0.0, 1.0).expect("a");
        scheduler.schedule_kernel("b", 0, 1.0, 1.0).expect("b");
        scheduler.schedule_kernel("c", 0, 2.0, 1.0).expect("c");
        let removed = scheduler.rewind_to(2.5);
        assert_eq!(removed, 1);
        assert_eq!(scheduler.timeline.len(), 2);
        let last = scheduler.timeline.last().unwrap();
        assert!((last.end - 2.0).abs() < 1e-6);
    }
}
