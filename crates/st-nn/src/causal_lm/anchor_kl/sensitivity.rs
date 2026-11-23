//! Rank a bounded set of descent directions using finite, same-state KL responses.
//! The local quadratic estimate can choose a ray, never authorize its acceptance.
use super::*;
use crate::geometry_runtime::ParameterGradients;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmSensitivityBasis {
    ParameterBlock,
    Coordinate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmSensitivitySelect {
    Envelope,
    GradientNorm,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CausalLmSensitivityPrediction {
    pub radius: f64,
    pub target_fraction: f64,
    pub linear_loss_decrease: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmSensitivityAxis {
    pub path: String,
    pub tensor: String,
    /// None means a whole trainable tensor; coordinate indices use its stored order.
    pub coordinate: Option<usize>,
    pub gradient_l2: f64,
    /// None means zero source gradient, not a measured invariant distribution.
    pub probe_per_anchor_kl: Option<Vec<f64>>,
    pub prediction: Option<CausalLmSensitivityPrediction>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmSensitivityReport {
    pub probe_l2: f64,
    pub basis: CausalLmSensitivityBasis,
    pub select: CausalLmSensitivitySelect,
    pub axes: Vec<CausalLmSensitivityAxis>,
    pub selected_axis: Option<usize>,
    /// Additional probe readouts only; already included in the parent receipt.
    pub probe_readouts: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_loss_dot_step: Option<f64>,
}

const MAX_AXES: usize = 32;
const MAX_SCRATCH: usize = 1 << 20;
const MAX_PROBE_LOGITS: usize = 1 << 23;

fn within_budget(axes: usize, scratch: usize, vocabulary: usize, anchors: usize) -> bool {
    anchors >= 2
        && (1..=MAX_AXES).contains(&axes)
        && scratch <= MAX_SCRATCH
        && vocabulary <= MAX_PROBE_LOGITS / axes / anchors
}

pub(super) fn admit(
    m: &NativeCausalLm,
    mode: CausalLmAdaptationMode,
    anchors: usize,
    ceiling: f64,
    criterion: CausalLmAnchorKlCriterion,
    h: f64,
    basis: CausalLmSensitivityBasis,
) -> CausalLmResult<()> {
    if !matches!(criterion, CausalLmAnchorKlCriterion::MeanWithCap { .. })
        || anchors < 2
        || !h.is_finite()
        || h <= 0.
        || h > ceiling
    {
        return Err("sensitivity rescue needs a capped mean, two anchors and a positive probe within the ceiling".into());
    }
    let (mut scratch, mut axes) = (0usize, 0usize);
    direction::visit_owner_blocks(m, mode, |_, _, v, trainable| {
        scratch = scratch.saturating_add(v.len());
        if trainable {
            axes = axes.saturating_add(match basis {
                CausalLmSensitivityBasis::ParameterBlock => 1,
                CausalLmSensitivityBasis::Coordinate => v.len(),
            });
        }
    });
    if !within_budget(axes, scratch, m.config.vocabulary, anchors) {
        return Err("sensitivity basis exceeds axis, parameter or probe-readout budget".into());
    }
    m.admit_adaptation_distance(mode, h)
}

pub(super) struct Settings {
    target: f64,
    cap: f64,
    ceiling: f64,
    h: f64,
    basis: CausalLmSensitivityBasis,
    select: CausalLmSensitivitySelect,
}

impl Settings {
    pub(super) fn new(
        policy: CausalLmAnchorKlPolicy,
        criterion: CausalLmAnchorKlCriterion,
        h: f64,
        basis: CausalLmSensitivityBasis,
        select: CausalLmSensitivitySelect,
    ) -> CausalLmResult<Self> {
        let CausalLmAnchorKlCriterion::MeanWithCap { max_anchor_kl: cap } = criterion else {
            return Err("sensitivity prediction requires a capped mean".into());
        };
        Ok(Self {
            target: policy.target_kl,
            cap,
            ceiling: policy.max_parameter_l2,
            h,
            basis,
            select,
        })
    }

    fn predict(&self, values: &[f64], norm: f64) -> Option<CausalLmSensitivityPrediction> {
        let mut sum = sum::Sum::default();
        for &v in values {
            sum.add(v);
        }
        let mean = sum.mean(values.len() as u64);
        let peak = values.iter().copied().fold(0., f64::max);
        if mean == 0. {
            return None;
        }
        // Log scale avoids overflowing cap/peak for very quiet probe banks.
        let h = self.h.ln();
        let log_r = self
            .ceiling
            .ln()
            .min(h + 0.5 * (self.target.ln() - mean.ln()))
            .min(h + 0.5 * (self.cap.ln() - peak.ln()));
        let radius = log_r.exp().min(self.ceiling);
        let fraction = (mean.ln() - self.target.ln() + 2. * (log_r - h))
            .min(0.)
            .exp();
        Some(CausalLmSensitivityPrediction {
            radius,
            target_fraction: fraction,
            linear_loss_decrease: norm * radius,
        })
    }
}

#[cfg(test)]
#[path = "sensitivity_tests.rs"]
mod tests;

fn select(axes: &[CausalLmSensitivityAxis], how: CausalLmSensitivitySelect) -> Option<usize> {
    let mut best = None;
    for (i, a) in axes.iter().enumerate() {
        if a.gradient_l2 == 0. {
            continue;
        }
        let rank = match how {
            CausalLmSensitivitySelect::GradientNorm => (false, a.gradient_l2, 0.),
            CausalLmSensitivitySelect::Envelope => {
                let Some(p) = a.prediction else { continue };
                let feasible = p.target_fraction >= CausalLmAnchorKlPolicy::MIN_TARGET_RATIO;
                (
                    feasible,
                    if feasible { 0. } else { p.target_fraction },
                    p.linear_loss_decrease,
                )
            }
        };
        if best.as_ref().is_none_or(|(_, previous)| rank > *previous) {
            best = Some((i, rank));
        }
    }
    best.map(|(i, _)| i)
}

fn source_block<'a>(
    source: &'a ParameterGradients,
    a: &CausalLmSensitivityAxis,
) -> CausalLmResult<&'a [f32]> {
    source
        .get(&a.path)
        .and_then(|p| p.get(&a.tensor))
        .map(Vec::as_slice)
        .ok_or_else(|| "missing sealed sensitivity gradient block".into())
}

fn mask(
    scratch: &mut ParameterGradients,
    source: &ParameterGradients,
    a: &CausalLmSensitivityAxis,
    enable: bool,
) -> CausalLmResult<()> {
    let src = source_block(source, a)?;
    let dst = scratch
        .get_mut(&a.path)
        .and_then(|p| p.get_mut(&a.tensor))
        .filter(|v| v.len() == src.len())
        .ok_or("mismatched sensitivity gradient shape")?;
    match a.coordinate {
        Some(i) => dst[i] = if enable { src[i] } else { 0. },
        None if enable => dst.copy_from_slice(src),
        None => dst.fill(0.),
    }
    Ok(())
}

impl NativeCausalLm {
    pub(super) fn sensitivity_ray(
        &self,
        source: &ParameterGradients,
        mode: CausalLmAdaptationMode,
        bank: &mut AnchorBank<'_>,
        cfg: Settings,
    ) -> CausalLmResult<(Option<ParameterGradients>, CausalLmSensitivityReport)> {
        let mut scratch = ParameterGradients::new();
        if mode != CausalLmAdaptationMode::Lora {
            if let Some(g) = self.geometry_artifact() {
                for p in &g.record().placements {
                    scratch.entry(p.path.clone()).or_default();
                }
            }
        }
        let mut axes = Vec::new();
        direction::visit_owner_blocks(self, mode, |path, name, values, trainable| {
            scratch
                .entry(path.to_owned())
                .or_default()
                .insert(name.to_owned(), vec![0.; values.len()]);
            if trainable {
                let mut push = |coordinate| {
                    axes.push(CausalLmSensitivityAxis {
                        path: path.to_owned(),
                        tensor: name.to_owned(),
                        coordinate,
                        gradient_l2: 0.,
                        probe_per_anchor_kl: None,
                        prediction: None,
                    })
                };
                match cfg.basis {
                    CausalLmSensitivityBasis::Coordinate => {
                        (0..values.len()).for_each(|i| push(Some(i)))
                    }
                    CausalLmSensitivityBasis::ParameterBlock => push(None),
                }
            }
        });
        axes.sort_by(|a, b| {
            (&a.path, &a.tensor, a.coordinate).cmp(&(&b.path, &b.tensor, b.coordinate))
        });
        let start = bank.readouts;
        for a in &mut axes {
            let block = source_block(source, a)?;
            let norm = match a.coordinate {
                Some(i) => f64::from(*block.get(i).ok_or("missing sensitivity coordinate")?).abs(),
                None => {
                    let mut sum = sum::Sum::default();
                    for &v in block {
                        sum.add(f64::from(v).powi(2));
                    }
                    sum.value().sqrt()
                }
            };
            a.gradient_l2 = norm;
            if norm == 0. {
                continue;
            }
            mask(&mut scratch, source, a, true)?;
            let (probe, _) = self.step_parameter_ray(&scratch, mode, cfg.h)?;
            let mut values = Vec::with_capacity(bank.prompts.len());
            bank.observe(self, &probe, |_, kl, _| values.push(kl))?;
            a.prediction = cfg.predict(&values, norm);
            a.probe_per_anchor_kl = Some(values);
            mask(&mut scratch, source, a, false)?;
        }
        let selected_axis = select(&axes, cfg.select);
        if let Some(i) = selected_axis {
            mask(&mut scratch, source, &axes[i], true)?;
        }
        Ok((
            selected_axis.map(|_| scratch),
            CausalLmSensitivityReport {
                probe_l2: cfg.h,
                basis: cfg.basis,
                select: cfg.select,
                axes,
                selected_axis,
                probe_readouts: bank.readouts - start,
                source_loss_dot_step: None,
            },
        ))
    }
}
