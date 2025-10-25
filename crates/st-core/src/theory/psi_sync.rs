// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! SynchroMonolith implementation for PSI synchronisation across Z-space branches.
//!
//! This module transliterates the in-process MetaMEMB sampler and circle-map
//! identification pipeline into a Rust-centric engine that can coordinate
//! multiple branches embedded in the Z space. The design keeps the numerical
//! flow of the original prototype (ticker → sampler → κ(γ) identification →
//! Arnold tongue heat-map) while exposing an idiomatic, deterministic API that
//! can be slotted into higher-level PSI experiments.

use std::f64::consts::PI;
use std::fmt;
use std::path::{Path, PathBuf};

use ndarray::Array2;
use plotters::prelude::*;
use thiserror::Error;

/// Shared synchronisation state propagated between the in-proc components.
#[derive(Clone, Debug)]
pub struct SyncState {
    pub tau: f64,
    pub step: f64,
    pub seed: u64,
    pub epoch: u64,
    pub gamma: f64,
    pub lam: f64,
    pub wd: f64,
    pub omega0: f64,
    pub drift_coupled: f64,
    pub poincare: bool,
}

impl Default for SyncState {
    fn default() -> Self {
        Self {
            tau: 0.0,
            step: 0.01,
            seed: 42,
            epoch: 0,
            gamma: 1.3,
            lam: 1.0,
            wd: 0.7,
            omega0: 0.72,
            drift_coupled: 1.05,
            poincare: true,
        }
    }
}

/// Minimal MetaMEMB sampler expressed with vector-friendly primitives.
#[derive(Clone, Debug)]
pub struct MetaMembModel {
    delta: [f64; 3],
    omega: [f64; 3],
    theta: [f64; 3],
}

impl Default for MetaMembModel {
    fn default() -> Self {
        Self {
            delta: [0.03, 0.05, 0.02],
            omega: [1.2, 0.9, 1.6],
            theta: [0.0, 1.0, 2.5],
        }
    }
}

impl MetaMembModel {
    fn bloom_scalar(&self, tau: f64) -> f64 {
        let mut acc = 0.0;
        for ((&delta, &omega), &theta) in self
            .delta
            .iter()
            .zip(self.omega.iter())
            .zip(self.theta.iter())
        {
            acc += delta * (omega * tau + theta).sin();
        }
        acc
    }

    /// Computes the Poincaré phase sample φ for a given configuration at time `t`.
    pub fn phi_sample(&self, t: f64, branch: &PsiBranchConfig) -> f64 {
        let bloom = self.bloom_scalar(t);
        let psi = branch.omega0 * t
            + branch.lam * branch.drift_coupled.powf(branch.gamma) * (5.0 * bloom).tanh();
        let theta = branch.wd * t + branch.phase0;
        let mut phi = (psi - theta) / (2.0 * PI);
        phi = ((phi % 1.0) + 1.0) % 1.0;
        phi
    }
}

/// Configuration for an individual PSI branch embedded in Z space.
#[derive(Clone, Debug)]
pub struct PsiBranchConfig {
    pub branch_id: String,
    pub gamma: f64,
    pub lam: f64,
    pub wd: f64,
    pub omega0: f64,
    pub drift_coupled: f64,
    pub phase0: f64,
    pub samples: usize,
    pub step: f64,
    pub z_coordinate: f32,
}

impl PsiBranchConfig {
    pub fn with_state(branch_id: impl Into<String>, state: &SyncState, samples: usize) -> Self {
        Self {
            branch_id: branch_id.into(),
            gamma: state.gamma,
            lam: state.lam,
            wd: state.wd,
            omega0: state.omega0,
            drift_coupled: state.drift_coupled,
            phase0: 0.0,
            samples,
            step: state.step,
            z_coordinate: 0.0,
        }
    }
}

/// Aggregated report returned for every synchronised branch.
#[derive(Clone, Debug)]
pub struct PsiBranchReport {
    pub branch: PsiBranchConfig,
    pub phi_samples: Vec<f64>,
    pub omega_hat: f64,
    pub kappa_hat: f64,
    pub rmse: f64,
    pub lam_grid: Vec<f64>,
    pub wd_grid: Vec<f64>,
    pub heatmap: Array2<f64>,
    pub best_lock: RationalLock,
    pub heatmap_path: Option<PathBuf>,
}

/// A rational approximation recovered from the rotation number sweep.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RationalLock {
    pub p: i32,
    pub q: i32,
    pub error: f64,
    pub lambda: f64,
    pub wd: f64,
}

impl Default for RationalLock {
    fn default() -> Self {
        Self {
            p: 0,
            q: 1,
            error: f64::MAX,
            lambda: 0.0,
            wd: 0.0,
        }
    }
}

impl fmt::Display for RationalLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} @ λ={:.4} ω_d={:.4} (ε={:.3e})",
            self.p, self.q, self.lambda, self.wd, self.error
        )
    }
}

/// Error conditions surfaced by the synchronisation monolith.
#[derive(Debug, Error)]
pub enum PsiSyncError {
    #[error("branch '{branch_id}' requires at least two samples (requested {samples})")]
    InsufficientSamples { branch_id: String, samples: usize },

    #[error("unable to render heatmap for branch '{branch_id}': {reason}")]
    Plot { branch_id: String, reason: String },
}

/// Configuration shared across the PSI monolith.
#[derive(Clone, Debug)]
pub struct PsiSynchroConfig {
    pub lam_range: (f64, f64, usize),
    pub wd_range: (f64, f64, usize),
    pub burn: usize,
    pub sample: usize,
    pub qmax: usize,
    pub out_dir: Option<PathBuf>,
}

impl Default for PsiSynchroConfig {
    fn default() -> Self {
        Self {
            lam_range: (0.0, 2.0, 60),
            wd_range: (0.3, 1.2, 80),
            burn: 200,
            sample: 300,
            qmax: 8,
            out_dir: None,
        }
    }
}

/// Deterministic translation of the MetaMEMB synchro-monolith.
#[derive(Clone, Debug)]
pub struct PsiSynchroMonolith {
    model: MetaMembModel,
    base_state: SyncState,
    config: PsiSynchroConfig,
}

impl PsiSynchroMonolith {
    pub fn new(base_state: SyncState, config: PsiSynchroConfig) -> Self {
        Self {
            model: MetaMembModel::default(),
            base_state,
            config,
        }
    }

    pub fn with_model(
        base_state: SyncState,
        config: PsiSynchroConfig,
        model: MetaMembModel,
    ) -> Self {
        Self {
            model,
            base_state,
            config,
        }
    }

    /// Runs the synchronisation pipeline for all requested branches.
    pub fn synchronise_branches(
        &self,
        branches: &[PsiBranchConfig],
    ) -> Result<Vec<PsiBranchReport>, PsiSyncError> {
        branches
            .iter()
            .map(|b| self.run_branch(b.clone()))
            .collect()
    }

    fn run_branch(&self, mut branch: PsiBranchConfig) -> Result<PsiBranchReport, PsiSyncError> {
        if branch.samples < 2 {
            return Err(PsiSyncError::InsufficientSamples {
                branch_id: branch.branch_id.clone(),
                samples: branch.samples,
            });
        }

        branch.gamma = branch.gamma.max(1.0e-9);
        branch.lam = branch.lam.max(1.0e-9);
        branch.wd = branch.wd.max(1.0e-9);
        branch.step = branch.step.max(1.0e-6);

        let phi_samples = self.sample_branch(&branch);
        let estimate = identify_kappa(&phi_samples, branch.lam);
        let heatmap = self.heatmap_for_branch(&branch, estimate.kappa_hat);

        let heatmap_path = if let Some(dir) = &self.config.out_dir {
            Some(
                self.render_heatmap(dir, &branch, &heatmap, estimate.kappa_hat)
                    .map_err(|err| PsiSyncError::Plot {
                        branch_id: branch.branch_id.clone(),
                        reason: err,
                    })?,
            )
        } else {
            None
        };

        Ok(PsiBranchReport {
            branch,
            phi_samples,
            omega_hat: estimate.omega_hat,
            kappa_hat: estimate.kappa_hat,
            rmse: estimate.rmse,
            lam_grid: heatmap.lam_grid.clone(),
            wd_grid: heatmap.wd_grid.clone(),
            heatmap: heatmap.matrix,
            best_lock: heatmap.best_lock,
            heatmap_path,
        })
    }

    fn sample_branch(&self, branch: &PsiBranchConfig) -> Vec<f64> {
        let mut phi = Vec::with_capacity(branch.samples);
        let td = 2.0 * PI / branch.wd;
        let mut next_sample = self.base_state.tau;
        let mut t = self.base_state.tau;
        while phi.len() < branch.samples {
            if t + 1.0e-12 >= next_sample {
                let value = self.model.phi_sample(t, branch);
                phi.push(value);
                next_sample += td;
            }
            t += branch.step;
        }
        phi
    }

    fn heatmap_for_branch(&self, branch: &PsiBranchConfig, kappa_hat: f64) -> HeatmapResult {
        let (lam_min, lam_max, lam_bins) = self.config.lam_range;
        let (wd_min, wd_max, wd_bins) = self.config.wd_range;

        let lam_grid = linspace(lam_min, lam_max, lam_bins);
        let wd_grid = linspace(wd_min, wd_max, wd_bins);
        let mut matrix = Array2::<f64>::zeros((lam_bins, wd_bins));
        let mut best_lock = RationalLock::default();

        for (i, &lam) in lam_grid.iter().enumerate() {
            let kappa = kappa_hat * lam;
            for (j, &wd) in wd_grid.iter().enumerate() {
                let omega = branch.omega0 / wd - 1.0;
                let rho = rotation_number(omega, kappa, self.config.burn, self.config.sample);
                let (p, q, err) = nearest_rational(rho, self.config.qmax);
                let strength = 1.0 / (err + 1.0e-6);
                matrix[[i, j]] = strength;
                if err < best_lock.error {
                    best_lock = RationalLock {
                        p,
                        q,
                        error: err,
                        lambda: lam,
                        wd,
                    };
                }
            }
        }

        HeatmapResult {
            lam_grid,
            wd_grid,
            matrix,
            best_lock,
        }
    }

    fn render_heatmap(
        &self,
        out_dir: &Path,
        branch: &PsiBranchConfig,
        heatmap: &HeatmapResult,
        kappa_hat: f64,
    ) -> Result<PathBuf, String> {
        std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
        let filename = format!(
            "heatmap_{}_gamma_{:.2}.png",
            branch.branch_id.replace(':', "-"),
            branch.gamma
        );
        let path = out_dir.join(filename);
        let root = BitMapBackend::new(&path, (720, 560)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| format!("backend error: {e}"))?;

        {
            let mut chart = ChartBuilder::on(&root)
                .margin(10)
                .caption(
                    format!(
                        "Arnold Tongues — γ={:.3} (κ̂={:.3e})",
                        branch.gamma, kappa_hat
                    ),
                    ("sans-serif", 24.0),
                )
                .set_label_area_size(LabelAreaPosition::Left, 60)
                .set_label_area_size(LabelAreaPosition::Bottom, 60)
                .build_cartesian_2d(
                    *heatmap.wd_grid.first().unwrap()..*heatmap.wd_grid.last().unwrap(),
                    *heatmap.lam_grid.first().unwrap()..*heatmap.lam_grid.last().unwrap(),
                )
                .map_err(|e| format!("chart build error: {e}"))?;

            chart
                .configure_mesh()
                .x_desc("drive frequency ω_d")
                .y_desc("λ (forcing strength)")
                .disable_mesh()
                .draw()
                .map_err(|e| format!("mesh error: {e}"))?;

            let lam_step = if heatmap.lam_grid.len() > 1 {
                heatmap.lam_grid[1] - heatmap.lam_grid[0]
            } else {
                1.0
            };
            let wd_step = if heatmap.wd_grid.len() > 1 {
                heatmap.wd_grid[1] - heatmap.wd_grid[0]
            } else {
                1.0
            };
            let max_strength = heatmap
                .matrix
                .iter()
                .cloned()
                .fold(f64::MIN, f64::max)
                .max(1.0);

            for (i, &lam) in heatmap.lam_grid.iter().enumerate() {
                for (j, &wd) in heatmap.wd_grid.iter().enumerate() {
                    let strength = heatmap.matrix[[i, j]] / max_strength;
                    let color = gradient_color(strength);
                    let rect = Rectangle::new(
                        [
                            (wd - wd_step / 2.0, lam - lam_step / 2.0),
                            (wd + wd_step / 2.0, lam + lam_step / 2.0),
                        ],
                        color.filled(),
                    );
                    chart
                        .draw_series(std::iter::once(rect))
                        .map_err(|e| format!("draw error: {e}"))?;
                }
            }
        }

        let output_path = path.clone();
        root.present().map_err(|e| format!("render error: {e}"))?;

        Ok(output_path)
    }
}

struct HeatmapResult {
    lam_grid: Vec<f64>,
    wd_grid: Vec<f64>,
    matrix: Array2<f64>,
    best_lock: RationalLock,
}

struct KappaEstimate {
    omega_hat: f64,
    kappa_hat: f64,
    rmse: f64,
}

fn identify_kappa(phi: &[f64], lam: f64) -> KappaEstimate {
    let mut s11 = 0.0;
    let mut s12 = 0.0;
    let mut s22 = 0.0;
    let mut y1 = 0.0;
    let mut y2 = 0.0;
    let mut mse = 0.0;
    let n = phi.len().saturating_sub(1);

    for i in 0..n {
        let x = phi[i];
        let d = wrap_unit(phi[i + 1] - phi[i]);
        let s = (2.0 * PI * x).sin();
        s11 += 1.0;
        s12 += s;
        s22 += s * s;
        y1 += d;
        y2 += s * d;
    }

    let det = s11 * s22 - s12 * s12;
    let (omega_hat, a_hat) = if det.abs() < 1.0e-12 {
        (0.0, 0.0)
    } else {
        let omega_hat = (s22 * y1 - s12 * y2) / det;
        let a_hat = (s11 * y2 - s12 * y1) / det;
        (omega_hat, a_hat)
    };

    for i in 0..n {
        let x = phi[i];
        let d = wrap_unit(phi[i + 1] - phi[i]);
        let s = (2.0 * PI * x).sin();
        let pred = omega_hat + a_hat * s;
        let err = d - pred;
        mse += err * err;
    }

    let rmse = if n > 0 { (mse / n as f64).sqrt() } else { 0.0 };
    let k_hat = -2.0 * PI * a_hat;
    let kappa_hat = k_hat / lam.max(1.0e-9);

    KappaEstimate {
        omega_hat,
        kappa_hat,
        rmse,
    }
}

fn rotation_number(omega: f64, kappa: f64, burn: usize, sample: usize) -> f64 {
    let mut phi = 0.0;
    let scale = kappa / (2.0 * PI);
    for _ in 0..burn {
        phi = (phi + omega - scale * (2.0 * PI * phi).sin()).rem_euclid(1.0);
    }
    let mut sum = 0.0;
    for _ in 0..sample {
        let prev = phi;
        phi = (phi + omega - scale * (2.0 * PI * phi).sin()).rem_euclid(1.0);
        sum += (phi - prev).rem_euclid(1.0);
    }
    if sample == 0 {
        0.0
    } else {
        sum / sample as f64
    }
}

fn nearest_rational(x: f64, qmax: usize) -> (i32, i32, f64) {
    let mut best = (0, 1, f64::MAX);
    for q in 1..=qmax.max(1) {
        let p = (x * q as f64).round() as i32;
        let err = (x - p as f64 / q as f64).abs();
        if err < best.2 {
            best = (p, q as i32, err);
        }
    }
    best
}

fn linspace(min: f64, max: f64, bins: usize) -> Vec<f64> {
    if bins <= 1 {
        return vec![min];
    }
    let step = (max - min) / (bins as f64 - 1.0);
    (0..bins).map(|i| min + step * i as f64).collect()
}

fn wrap_unit(x: f64) -> f64 {
    ((x + 0.5).rem_euclid(1.0)) - 0.5
}

fn gradient_color(norm: f64) -> RGBColor {
    let clamped = norm.clamp(0.0, 1.0);
    let r = (255.0 * clamped) as u8;
    let g = (255.0 * (1.0 - (clamped - 0.5).abs() * 2.0).max(0.0)) as u8;
    let b = (255.0 * (1.0 - clamped)) as u8;
    RGBColor(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kappa_identification_is_stable() {
        let model = MetaMembModel::default();
        let state = SyncState::default();
        let monolith =
            PsiSynchroMonolith::with_model(state.clone(), PsiSynchroConfig::default(), model);
        let branch = PsiBranchConfig {
            branch_id: "test".into(),
            gamma: state.gamma,
            lam: state.lam,
            wd: state.wd,
            omega0: state.omega0,
            drift_coupled: state.drift_coupled,
            phase0: 0.0,
            samples: 256,
            step: state.step,
            z_coordinate: 0.5,
        };

        let report = monolith
            .synchronise_branches(&[branch])
            .expect("branch report")
            .pop()
            .unwrap();

        assert_eq!(report.phi_samples.len(), 256);
        assert!(report.kappa_hat.is_finite());
        assert!(report.rmse.is_finite());
        assert_eq!(
            report.heatmap.shape(),
            &[monolith.config.lam_range.2, monolith.config.wd_range.2]
        );
        assert!(report.best_lock.error >= 0.0);
    }
}
