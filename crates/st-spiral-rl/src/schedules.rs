// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

/// Linear epsilon schedule used by epsilon-greedy exploration.
#[derive(Clone, Debug)]
pub struct EpsilonGreedySchedule {
    start: f32,
    end: f32,
    steps: u32,
    step: u32,
}

impl EpsilonGreedySchedule {
    pub fn new(start: f32, end: f32, steps: u32) -> Self {
        Self {
            start,
            end,
            steps: steps.max(1),
            step: 0,
        }
    }

    pub fn value(&self) -> f32 {
        let progress = (self.step as f32 / self.steps as f32).clamp(0.0, 1.0);
        self.start + (self.end - self.start) * progress
    }

    pub fn advance(&mut self) -> f32 {
        self.step = self.step.saturating_add(1);
        self.value()
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn parameters(&self) -> (f32, f32, u32) {
        (self.start, self.end, self.steps)
    }

    pub fn set_step(&mut self, step: u32) {
        self.step = step.min(self.steps);
    }
}
