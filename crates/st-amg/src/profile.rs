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
        Self {
            rows,
            cols,
            nnz,
            subgroup,
            density,
            aspect,
            curvature,
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
        if self.subgroup {
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
        match (self.cols, self.subgroup) {
            (..=4096, true) => 4_096,
            (..=4096, false) => 2_048,
            (4097..=16384, true) => 8_192,
            (4097..=16384, false) => 4_096,
            (16385..=65536, _) => 8_192,
            _ => 16_384,
        }
    }

    /// Wilson-style alpha for blending fractional energy metrics.
    pub fn fractional_alpha(&self) -> f32 {
        let alpha = (self.density.sqrt() * 0.65 + 0.25) * (1.0 - 0.05 / self.curvature);
        alpha.clamp(0.15, 0.95)
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
}
