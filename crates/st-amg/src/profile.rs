// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-amg/src/profile.rs
//! Lightweight statistics derived from a sparse system prior to AMG selection.

/// Coarse density buckets used by the heuristic rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityClass {
    UltraSparse,
    Sparse,
    Moderate,
    Dense,
}

/// Coarse aspect buckets that help distinguish between tall, square, and wide systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AspectClass {
    Square,
    Tall,
    Wide,
}

/// Aggregated characteristics of the linear system passed to AMG.
#[derive(Clone, Debug, PartialEq)]
pub struct ProblemProfile {
    rows: usize,
    cols: usize,
    nnz: usize,
    subgroup: bool,
    density: f32,
    aspect: f32,
    curvature: f32,
    diag_ratio: f32,
    mean_row_nnz: f32,
    max_row_nnz: u32,
    bandwidth_hint: u32,
    bandwidth_peak: u32,
    bandwidth_stddev: f32,
    row_nnz_stddev: f32,
}

impl ProblemProfile {
    /// Create a profile and compute derived values eagerly so that the caller can reuse
    /// them across heuristics without re-performing the arithmetic.
    pub fn new(rows: usize, cols: usize, nnz: usize, subgroup: bool) -> Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        let nnz = nnz.max(1);
        let volume = (rows * cols) as f32;
        let density = (nnz as f32 / volume).clamp(1e-6, 1.0);
        let aspect = if rows >= cols {
            rows as f32 / cols as f32
        } else {
            cols as f32 / rows as f32
        };
        let curvature = (rows.max(cols) as f32).log2().max(1.0);
        let mean_row_nnz = (nnz as f32 / rows as f32).max(1.0);
        let row_nnz_stddev = mean_row_nnz.sqrt().max(0.5);
        let max_row_nnz = mean_row_nnz.ceil() as u32;
        let bandwidth_hint = cols.min(rows).max(1) as u32;
        let bandwidth_peak = bandwidth_hint;
        let bandwidth_stddev = (bandwidth_hint as f32 * 0.35).max(1.0);

        Self {
            rows,
            cols,
            nnz,
            subgroup,
            density,
            aspect,
            curvature,
            diag_ratio: 1.0,
            mean_row_nnz,
            max_row_nnz,
            bandwidth_hint,
            bandwidth_peak,
            bandwidth_stddev,
            row_nnz_stddev,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn nnz(&self) -> usize {
        self.nnz
    }
    pub fn subgroup(&self) -> bool {
        self.subgroup
    }
    pub fn density(&self) -> f32 {
        self.density
    }
    pub fn aspect_ratio(&self) -> f32 {
        self.aspect
    }
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Complement of the density — useful for gauging sparsity at a glance.
    pub fn sparsity_index(&self) -> f32 {
        1.0 - self.density
    }

    /// Fraction of rows that contained a diagonal non-zero during profiling.
    pub fn diag_ratio(&self) -> f32 {
        self.diag_ratio
    }

    /// Arithmetic mean of the sampled non-zeros per row.
    pub fn mean_row_nnz(&self) -> f32 {
        self.mean_row_nnz
    }

    /// Largest observed row density expressed as non-zeros per row.
    pub fn max_row_nnz(&self) -> u32 {
        self.max_row_nnz
    }

    /// Estimated half-bandwidth used to bias tile/workgroup selection.
    pub fn bandwidth_hint(&self) -> u32 {
        self.bandwidth_hint
    }

    /// Maximum observed (or assumed) half-bandwidth across sampled rows.
    pub fn bandwidth_peak(&self) -> u32 {
        self.bandwidth_peak
    }

    /// Standard deviation of bandwidth observations captured during profiling.
    pub fn bandwidth_stddev(&self) -> f32 {
        self.bandwidth_stddev
    }

    /// Relative spread of the bandwidth distribution.
    pub fn bandwidth_spread(&self) -> f32 {
        let denom = self.bandwidth_peak.max(1) as f32;
        if denom <= f32::EPSILON {
            0.0
        } else {
            (self.bandwidth_stddev / denom).clamp(0.0, 2.5)
        }
    }

    /// Standard deviation of row densities captured during profiling.
    pub fn row_nnz_stddev(&self) -> f32 {
        self.row_nnz_stddev
    }

    /// Relative spread (σ / μ) of the row non-zero distribution.
    pub fn row_nnz_spread(&self) -> f32 {
        if self.mean_row_nnz <= f32::EPSILON {
            0.0
        } else {
            (self.row_nnz_stddev / self.mean_row_nnz).clamp(0.0, 2.0)
        }
    }

    /// Map the continuous density value into discrete bands which are easier to reason
    /// about when assigning categorical rules.
    pub fn density_class(&self) -> DensityClass {
        match self.density {
            d if d < 0.03 => DensityClass::UltraSparse,
            d if d < 0.10 => DensityClass::Sparse,
            d if d < 0.30 => DensityClass::Moderate,
            _ => DensityClass::Dense,
        }
    }

    /// Categorize the matrix footprint for directional heuristics.
    pub fn aspect_class(&self) -> AspectClass {
        if self.rows == 0 || self.cols == 0 {
            AspectClass::Square
        } else if self.rows == self.cols || self.aspect < 1.2 {
            AspectClass::Square
        } else if self.cols > self.rows {
            AspectClass::Wide
        } else {
            AspectClass::Tall
        }
    }

    /// Derive the number of Jacobi pre-smoothing passes we should seed the solver with.
    pub fn jacobi_hint(&self) -> u32 {
        match self.density_class() {
            DensityClass::UltraSparse => 0,
            DensityClass::Sparse => 1,
            DensityClass::Moderate => 2,
            DensityClass::Dense => 3,
        }
    }

    /// Preferred WGPU workgroup lanes given the input footprint.
    pub fn lane_hint(&self) -> u32 {
        let bandwidth_spread = self.bandwidth_spread();
        let bandwidth_peak = self.bandwidth_peak();
        if bandwidth_peak >= 14_000 {
            512
        } else if self.bandwidth_hint >= 8_000 {
            512
        } else if self.bandwidth_hint <= 2_000 && self.mean_row_nnz < 12.0 {
            128
        } else if bandwidth_spread > 0.4 {
            512
        } else if self.subgroup {
            if self.cols >= 8192 {
                512
            } else {
                256
            }
        } else if self.rows <= 16 {
            128
        } else if self.cols <= 4096 {
            256
        } else {
            512
        }
    }

    /// Preferred tile width heuristic matching the SpiralK kernels.
    pub fn tile_hint(&self) -> u32 {
        let bandwidth = self.bandwidth_hint;
        let bandwidth_spread = self.bandwidth_spread();
        let bandwidth_peak = self.bandwidth_peak();
        let mut tile = match (self.cols, self.subgroup, bandwidth) {
            (..=4096, true, ..=4096) => 4_096,
            (..=4096, false, ..=4096) => 2_048,
            (4097..=16384, true, ..=8192) => 8_192,
            (4097..=16384, false, ..=8192) => 4_096,
            (_, _, 0..=4096) => 8_192,
            (16385..=65536, _, ..=12_000) => 8_192,
            _ => 16_384,
        };
        if bandwidth_peak >= 14_000 {
            tile = tile.max(8_192);
        }
        if bandwidth_spread > 0.4 {
            tile = tile.max(8_192);
        } else if bandwidth_spread < 0.25 && bandwidth_peak < 4_000 {
            tile = tile.min(8_192).max(4_096);
        }
        tile
    }

    /// Wilson-style alpha for blending fractional energy metrics.
    pub fn fractional_alpha(&self) -> f32 {
        let alpha = (self.density.sqrt() * 0.65 + 0.25) * (1.0 - 0.05 / self.curvature);
        alpha.clamp(0.15, 0.95)
    }
}

/// Incrementally build a [`ProblemProfile`] from sparse row samples.
#[derive(Clone, Debug, Default)]
pub struct ProfileBuilder {
    rows: usize,
    cols: usize,
    subgroup: bool,
    row_samples: usize,
    nnz_acc: usize,
    diag_hits: usize,
    row_nnz_sum: f64,
    row_nnz_sq_sum: f64,
    max_row_nnz: usize,
    bandwidth_sum: f64,
    bandwidth_max: usize,
    bandwidth_samples: usize,
    bandwidth_sq_sum: f64,
}

impl ProfileBuilder {
    /// Create a builder for the given system dimensions.
    pub fn new(rows: usize, cols: usize, subgroup: bool) -> Self {
        Self {
            rows,
            cols,
            subgroup,
            ..Self::default()
        }
    }

    /// Push statistics for a single row.
    ///
    /// * `nnz_in_row` — number of structural non-zeros observed in the row.
    /// * `diag_hit` — whether the diagonal entry was present/non-zero.
    /// * `bandwidth` — optional half-bandwidth (max |col-row|) observed for the row.
    pub fn observe_row(&mut self, nnz_in_row: usize, diag_hit: bool, bandwidth: Option<usize>) {
        self.row_samples += 1;
        self.nnz_acc += nnz_in_row;
        self.row_nnz_sum += nnz_in_row as f64;
        self.row_nnz_sq_sum += (nnz_in_row * nnz_in_row) as f64;
        self.max_row_nnz = self.max_row_nnz.max(nnz_in_row.max(1));
        if diag_hit {
            self.diag_hits += 1;
        }
        if let Some(bw) = bandwidth {
            self.bandwidth_sum += bw as f64;
            self.bandwidth_samples += 1;
            self.bandwidth_max = self.bandwidth_max.max(bw);
            self.bandwidth_sq_sum += (bw * bw) as f64;
        }
    }

    fn diag_ratio(&self) -> f32 {
        if self.row_samples == 0 {
            1.0
        } else {
            (self.diag_hits as f32 / self.row_samples as f32).clamp(0.0, 1.0)
        }
    }

    fn mean_row_nnz(&self) -> f32 {
        if self.row_samples == 0 {
            1.0
        } else {
            (self.row_nnz_sum / self.row_samples as f64) as f32
        }
    }

    fn row_stddev(&self) -> f32 {
        if self.row_samples <= 1 {
            self.mean_row_nnz().sqrt().max(0.5)
        } else {
            let mean = self.row_nnz_sum / self.row_samples as f64;
            let mean_sq = self.row_nnz_sq_sum / self.row_samples as f64;
            let variance = (mean_sq - mean * mean).max(0.0);
            variance.sqrt() as f32
        }
    }

    fn bandwidth_hint(&self) -> u32 {
        if self.bandwidth_samples == 0 {
            (self.cols.min(self.rows).max(1)) as u32
        } else {
            let avg = (self.bandwidth_sum / self.bandwidth_samples as f64) as f32;
            let boosted = avg.max(self.bandwidth_max as f32 * 0.9);
            boosted.max(1.0) as u32
        }
    }

    fn bandwidth_peak(&self) -> u32 {
        if self.bandwidth_samples == 0 {
            self.bandwidth_hint()
        } else {
            self.bandwidth_max.max(1) as u32
        }
    }

    fn bandwidth_stddev(&self) -> f32 {
        if self.bandwidth_samples <= 1 {
            (self.bandwidth_peak() as f32 * 0.35).max(1.0)
        } else {
            let mean = self.bandwidth_sum / self.bandwidth_samples as f64;
            let mean_sq = self.bandwidth_sq_sum / self.bandwidth_samples as f64;
            let variance = (mean_sq - mean * mean).max(0.0);
            variance.sqrt() as f32
        }
    }

    /// Finalize the builder into a [`ProblemProfile`].
    pub fn build(self) -> ProblemProfile {
        let nnz = if self.nnz_acc == 0 { 1 } else { self.nnz_acc };
        let mut profile = ProblemProfile::new(self.rows, self.cols, nnz, self.subgroup);
        profile.diag_ratio = self.diag_ratio();
        profile.mean_row_nnz = self.mean_row_nnz().max(profile.mean_row_nnz);
        profile.max_row_nnz = self.max_row_nnz.max(profile.max_row_nnz as usize) as u32;
        profile.bandwidth_hint = self.bandwidth_hint().max(profile.bandwidth_hint);
        profile.bandwidth_peak = self.bandwidth_peak().max(profile.bandwidth_peak);
        profile.bandwidth_stddev = self.bandwidth_stddev().max(profile.bandwidth_stddev);
        profile.row_nnz_stddev = self.row_stddev().max(profile.row_nnz_stddev);
        profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_buckets_progress() {
        let ultra = ProblemProfile::new(1024, 1024, 20_000, false);
        assert_eq!(ultra.density_class(), DensityClass::UltraSparse);
        let sparse = ProblemProfile::new(1024, 1024, 80_000, false);
        assert_eq!(sparse.density_class(), DensityClass::Sparse);
        let moderate = ProblemProfile::new(1024, 1024, 150_000, false);
        assert_eq!(moderate.density_class(), DensityClass::Moderate);
        let dense = ProblemProfile::new(1024, 1024, 800_000, false);
        assert_eq!(dense.density_class(), DensityClass::Dense);
    }

    #[test]
    fn lane_hint_tracks_subgroup() {
        let subgroup = ProblemProfile::new(2048, 2048, 200_000, true);
        assert_eq!(subgroup.lane_hint(), 256);
        let wide = ProblemProfile::new(64, 32_768, 400_000, true);
        assert_eq!(wide.lane_hint(), 512);
        let classic = ProblemProfile::new(1024, 1024, 200_000, false);
        assert_eq!(classic.lane_hint(), 256);
    }

    #[test]
    fn tile_hint_tracks_spans() {
        let small = ProblemProfile::new(1024, 1024, 200_000, false);
        assert_eq!(small.tile_hint(), 2_048);
        let medium = ProblemProfile::new(2048, 8192, 400_000, true);
        assert_eq!(medium.tile_hint(), 8_192);
    }

    #[test]
    fn aspect_class_matches_shape() {
        let square = ProblemProfile::new(1024, 1024, 120_000, false);
        assert_eq!(square.aspect_class(), AspectClass::Square);
        let tall = ProblemProfile::new(16_384, 2048, 600_000, false);
        assert_eq!(tall.aspect_class(), AspectClass::Tall);
        let wide = ProblemProfile::new(1024, 16_384, 600_000, false);
        assert_eq!(wide.aspect_class(), AspectClass::Wide);
    }

    #[test]
    fn builder_enriches_profile() {
        let mut builder = ProfileBuilder::new(4, 8, false);
        builder.observe_row(3, true, Some(2));
        builder.observe_row(5, false, Some(4));
        builder.observe_row(4, true, Some(6));
        builder.observe_row(6, true, Some(6));
        let profile = builder.build();
        assert!((profile.diag_ratio() - 0.75).abs() < 1e-6);
        assert!(profile.mean_row_nnz() > 4.0);
        assert_eq!(profile.max_row_nnz(), 6);
        assert!(profile.bandwidth_hint() >= 5);
        assert!(profile.row_nnz_stddev() >= 1.0);
        assert!(profile.bandwidth_peak() >= 6);
        assert!(profile.bandwidth_stddev() >= 1.0);
    }

    #[test]
    fn builder_variance_tracks_spread() {
        let mut builder = ProfileBuilder::new(8, 16, false);
        for idx in 0..8 {
            builder.observe_row(idx + 1, idx % 2 == 0, Some(4 + idx));
        }
        let profile = builder.build();
        let spread = profile.row_nnz_spread();
        assert!(spread > 0.25);
    }

    #[test]
    fn builder_bandwidth_stats() {
        let mut builder = ProfileBuilder::new(64, 256, false);
        for bw in [8, 12, 30, 20, 40, 16, 24, 32] {
            builder.observe_row(6, true, Some(bw));
        }
        let profile = builder.build();
        assert!(profile.bandwidth_peak() >= 40);
        assert!(profile.bandwidth_stddev() > 8.0);
        assert!(profile.bandwidth_spread() > 0.2);
    }
}
