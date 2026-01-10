// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Telemetry assembly for Z-RBA modules.

use super::attention::{AttentionTelemetry, ZIndex};
use super::beta_residual::BetaGateSample;
use super::cov_head::CovHeadTelemetry;
use crate::{PureResult, Tensor, TensorError};
use serde::Serialize;
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f32::consts::{PI, SQRT_2};

/// Reliability bin aggregated by band.
#[derive(Clone, Debug, Serialize)]
pub struct ReliabilityBin {
    pub band: usize,
    pub expected: f32,
    pub observed: f32,
}

/// Unified telemetry bundle combining attention, gate, and covariance stats.
#[derive(Clone, Debug)]
pub struct ZRBATelemetry {
    pub attention: AttentionTelemetry,
    pub gate: BetaGateSample,
    pub covariance: CovHeadTelemetry,
}

impl ZRBATelemetry {
    pub fn new(
        attention: AttentionTelemetry,
        gate: BetaGateSample,
        covariance: CovHeadTelemetry,
    ) -> Self {
        Self {
            attention,
            gate,
            covariance,
        }
    }

    pub fn bundle_metrics(&self, metrics: &ZRBAMetrics) -> ZTelemetryBundle {
        ZTelemetryBundle {
            metrics: metrics.clone(),
            gate_mean: self.gate.expected,
            gate_variance: self.gate.variance,
            min_eigenvalue: self.covariance.min_eigenvalue,
            condition_number: self.covariance.condition_number,
        }
    }

    pub fn metrics(
        &self,
        predictions: &Tensor,
        variances: &Tensor,
        targets: &[f32],
        pin: f32,
        indices: &[ZIndex],
    ) -> PureResult<ZRBAMetrics> {
        ZRBAMetrics::compute(predictions, variances, targets, pin, indices)
    }
}

/// Statistical summary computed from predictions and variances.
#[derive(Clone, Debug, Serialize)]
pub struct ZRBAMetrics {
    pub pin: f32,
    pub picp: f32,
    pub pinaw: f32,
    pub nll: f32,
    pub crps: f32,
    pub reliability_by_band: Vec<ReliabilityBin>,
    pub ood_spearman: f32,
}

impl ZRBAMetrics {
    pub fn compute(
        predictions: &Tensor,
        variances: &Tensor,
        targets: &[f32],
        pin: f32,
        indices: &[ZIndex],
    ) -> PureResult<Self> {
        let (rows, cols) = predictions.shape();
        if variances.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: variances.shape(),
                right: (rows, cols),
            });
        }
        if targets.len() != rows {
            return Err(TensorError::DataLength {
                expected: rows,
                got: targets.len(),
            });
        }
        if rows == 0 {
            return Err(TensorError::EmptyInput("zrba::metrics"));
        }
        let quantile = approx_standard_normal_quantile(0.5 + pin / 2.0);
        let preds = predictions.data();
        let vars = variances.data();
        let mut coverage = 0.0f32;
        let mut width_acc = 0.0f32;
        let mut nll_acc = 0.0f32;
        let mut crps_acc = 0.0f32;
        let mut errors = Vec::with_capacity(rows);
        let mut diag_vars = Vec::with_capacity(rows);
        let mut row_means = Vec::with_capacity(rows);
        let mut row_variances = Vec::with_capacity(rows);
        for (i, &target) in targets.iter().enumerate() {
            let row_start = i * cols;
            let row_end = row_start + cols;
            let row_slice = &preds[row_start..row_end];
            let var_slice = &vars[row_start..row_end];
            let mean = row_slice.iter().sum::<f32>() / cols as f32;
            let variance = var_slice.iter().map(|value| value.max(1e-6)).sum::<f32>() / cols as f32;
            let std = variance.sqrt();
            let lower = mean - quantile * std;
            let upper = mean + quantile * std;
            if target >= lower && target <= upper {
                coverage += 1.0;
            }
            width_acc += upper - lower;
            let diff = target - mean;
            nll_acc += 0.5 * ((2.0 * PI * variance).ln() + diff * diff / variance);
            let z = diff / std;
            crps_acc += normal_crps(z, std);
            errors.push(diff.abs());
            diag_vars.push(variance);
            row_means.push(mean);
            row_variances.push(variance);
        }
        let picp = coverage / rows as f32;
        let pinaw = width_acc / rows as f32;
        let nll = nll_acc / rows as f32;
        let crps = crps_acc / rows as f32;
        let reliability =
            reliability_by_band(indices, targets, &row_means, &row_variances, quantile, pin);
        let ood = spearman_rank_correlation(&errors, &diag_vars);
        Ok(Self {
            pin,
            picp,
            pinaw,
            nll,
            crps,
            reliability_by_band: reliability,
            ood_spearman: ood,
        })
    }
}

/// Structured payload ready to be attached to Z-space telemetry streams.
#[derive(Clone, Debug, Serialize)]
pub struct ZTelemetryBundle {
    pub metrics: ZRBAMetrics,
    pub gate_mean: f32,
    pub gate_variance: f32,
    pub min_eigenvalue: f32,
    pub condition_number: f32,
}

impl ZTelemetryBundle {
    pub fn to_json(&self) -> Value {
        json!({
            "metrics": self.metrics,
            "gate": {
                "mean": self.gate_mean,
                "variance": self.gate_variance,
            },
            "covariance": {
                "min_eigenvalue": self.min_eigenvalue,
                "condition_number": self.condition_number,
            }
        })
    }
}

fn reliability_by_band(
    indices: &[ZIndex],
    targets: &[f32],
    means: &[f32],
    variances: &[f32],
    quantile: f32,
    pin: f32,
) -> Vec<ReliabilityBin> {
    let mut per_band: HashMap<usize, (f32, usize)> = HashMap::new();
    for ((idx, &target), (&mean, &variance)) in indices
        .iter()
        .zip(targets.iter())
        .zip(means.iter().zip(variances.iter()))
    {
        let std = variance.max(1e-6).sqrt();
        let lower = mean - quantile * std;
        let upper = mean + quantile * std;
        let entry = per_band.entry(idx.band).or_insert((0.0, 0));
        if target >= lower && target <= upper {
            entry.0 += 1.0;
        }
        entry.1 += 1;
    }
    let mut bins = Vec::new();
    for (band, (covered, count)) in per_band.into_iter() {
        if count == 0 {
            continue;
        }
        bins.push(ReliabilityBin {
            band,
            expected: pin.clamp(0.0, 1.0),
            observed: (covered / count as f32).clamp(0.0, 1.0),
        });
    }
    bins.sort_by(|a, b| a.band.cmp(&b.band));
    bins
}

fn approx_standard_normal_quantile(p: f32) -> f32 {
    // Acklam's approximation with coefficients tuned for f32.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p = p.clamp(1e-6, 1.0 - 1e-6) as f64;
    let result = if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > 1.0 - 0.02425 {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -((((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0))
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    result as f32
}

fn normal_crps(z: f32, sigma: f32) -> f32 {
    let sigma = sigma.max(1e-6);
    let pdf = (-0.5 * z * z).exp() / (SQRT_2 * PI.sqrt());
    let cdf = 0.5 * (1.0 + approx_erf(z / SQRT_2));
    sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / PI.sqrt())
}

fn approx_erf(x: f32) -> f32 {
    // Abramowitz and Stegun approximation (computed in f64 for stability).
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.5 * x);
    let tau = t
        * (-x * x - 1.26551223
            + 1.00002368 * t
            + 0.37409196 * t * t
            + 0.09678418 * t.powi(3)
            - 0.18628806 * t.powi(4)
            + 0.27886807 * t.powi(5)
            - 1.13520398 * t.powi(6)
            + 1.48851587 * t.powi(7)
            - 0.82215223 * t.powi(8)
            + 0.17087277 * t.powi(9))
        .exp();
    (sign * (1.0 - tau)) as f32
}

fn spearman_rank_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let rank_x = rank(x);
    let rank_y = rank(y);
    pearson(&rank_x, &rank_y)
}

fn rank(values: &[f32]) -> Vec<f32> {
    let mut pairs: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let mut ranks = vec![0.0f32; values.len()];
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && (pairs[j].1 - pairs[i].1).abs() < 1e-6 {
            j += 1;
        }
        let rank_value = ((i + j - 1) as f32 / 2.0) + 1.0;
        for k in i..j {
            ranks[pairs[k].0] = rank_value;
        }
        i = j;
    }
    ranks
}

fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f32>() / n as f32;
    let mean_y = y.iter().sum::<f32>() / n as f32;
    let mut num = 0.0f32;
    let mut denom_x = 0.0f32;
    let mut denom_y = 0.0f32;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }
    let denom = (denom_x * denom_y).sqrt();
    if denom <= 1e-6 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_compute_basic_stats() {
        let preds = Tensor::from_vec(3, 1, vec![0.1, 0.2, 0.3]).unwrap();
        let vars = Tensor::from_vec(3, 1, vec![0.05, 0.04, 0.06]).unwrap();
        let targets = vec![0.0, 0.25, 0.28];
        let indices = vec![
            ZIndex {
                band: 0,
                sheet: 0,
                echo: 0,
            },
            ZIndex {
                band: 1,
                sheet: 1,
                echo: 2,
            },
            ZIndex {
                band: 1,
                sheet: 1,
                echo: 3,
            },
        ];
        let metrics = ZRBAMetrics::compute(&preds, &vars, &targets, 0.95, &indices).unwrap();
        assert!(metrics.picp >= 0.0 && metrics.picp <= 1.0);
        assert!((metrics.pin - 0.95).abs() < 1e-3);
        assert!(!metrics.reliability_by_band.is_empty());
        for bin in metrics.reliability_by_band.iter() {
            assert!((bin.expected - 0.95).abs() < 1e-6);
            assert!(bin.observed >= 0.0 && bin.observed <= 1.0);
        }
    }

    #[test]
    fn telemetry_bundle_to_json() {
        let telemetry = ZRBATelemetry::new(
            AttentionTelemetry::new(2),
            BetaGateSample {
                sample: 0.5,
                alpha: 2.0,
                beta: 2.0,
                expected: 0.5,
                variance: 0.05,
                features: [0.0; 7],
                applied: 0.5,
            },
            CovHeadTelemetry {
                min_eigenvalue: 0.1,
                max_eigenvalue: 2.0,
                condition_number: 20.0,
                stabiliser: 1e-4,
            },
        );
        let metrics = ZRBAMetrics {
            pin: 0.95,
            picp: 0.9,
            pinaw: 0.4,
            nll: 1.2,
            crps: 0.3,
            reliability_by_band: vec![ReliabilityBin {
                band: 0,
                expected: 0.95,
                observed: 0.9,
            }],
            ood_spearman: 0.4,
        };
        let bundle = telemetry.bundle_metrics(&metrics);
        let json = bundle.to_json();
        assert!(json.get("metrics").is_some());
    }
}
