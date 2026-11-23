use super::*;
use crate::geometry_artifact::GeometryAdapterTensorRole;
use crate::geometry_runtime::ParameterGradients;
use std::collections::BTreeMap;

mod probe;
use probe::{Choice, ProbeRows};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CausalLmAnchorDirection {
    LossGradient {},
    /// Estimate one anchor normal using a finite probe, then remove its component.
    SecantProjection {
        probe_l2: f64,
        anchor: CausalLmSecantAnchor,
    },
    /// Try one angular midpoint only after the raw ray has a measured KL miss.
    RawThenSecantMidpoint {
        probe_l2: f64,
        anchor: CausalLmSecantAnchor,
    },
    /// Measure restricted descent rays on the same bank, then search exactly one.
    RawThenSensitivityRay {
        probe_l2: f64,
        basis: CausalLmSensitivityBasis,
        select: CausalLmSensitivitySelect,
    },
}

impl Default for CausalLmAnchorDirection {
    fn default() -> Self {
        Self::LossGradient {}
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmSecantAnchor {
    Largest,
    Successor,
    /// The complete-line gradient's own prefix, independent of bank order.
    LossPrompt,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmSecantMidpoint {
    pub gradient_l2: f64,
    pub source_cosine: f64,
    pub projected_cosine: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmSecantReport {
    pub policy: CausalLmAnchorDirection,
    pub probe_per_anchor_kl: Vec<f64>,
    pub peak_anchor: usize,
    pub selected_anchor: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss_prompt_id: Option<String>,
    pub source_gradient_l2: f64,
    pub normal_l2: f64,
    pub projected_l2: f64,
    pub retained_fraction: f64,
    /// After projecting and rounding the new ray coordinates to f32.
    pub projected_normal_cosine: f64,
    /// The fields above describe the full orthogonal projection, even when
    /// this midpoint rather than the projection is used for the actual ray.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub midpoint: Option<Box<CausalLmSecantMidpoint>>,
    /// The original loss gradient dotted with the actual f32 parameter displacement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_loss_dot_step: Option<f64>,
}

impl CausalLmAnchorDirection {
    fn secant(self) -> Option<(f64, CausalLmSecantAnchor)> {
        match self {
            Self::LossGradient {} | Self::RawThenSensitivityRay { .. } => None,
            Self::SecantProjection { probe_l2, anchor }
            | Self::RawThenSecantMidpoint { probe_l2, anchor } => Some((probe_l2, anchor)),
        }
    }

    pub(super) fn is_rescue(self) -> bool {
        matches!(
            self,
            Self::RawThenSecantMidpoint { .. } | Self::RawThenSensitivityRay { .. }
        )
    }

    pub(in crate::causal_lm) fn uses_loss_prompt(self) -> bool {
        matches!(self.secant(), Some((_, CausalLmSecantAnchor::LossPrompt)))
    }

    pub(super) fn resolve(
        self,
        g: &NativeCausalLmAdaptationGradients,
        prompts: &[&[usize]],
    ) -> CausalLmResult<Option<ResolvedSecant>> {
        let Some((probe_l2, anchor)) = self.secant() else {
            return Ok(None);
        };
        let choice = match anchor {
            CausalLmSecantAnchor::Largest => Choice::Peak,
            CausalLmSecantAnchor::Successor => Choice::Next,
            CausalLmSecantAnchor::LossPrompt => {
                let id = g
                    .line_prompt
                    .as_ref()
                    .ok_or("loss_prompt requires a complete-line gradient")?;
                let i = prompts
                    .iter()
                    .position(|p| line_session::LinePromptId::new(p) == *id)
                    .ok_or("sealed loss prompt is absent from the anchor bank")?;
                Choice::At(i)
            }
        };
        Ok(Some(ResolvedSecant {
            policy: self,
            probe_l2,
            choice,
        }))
    }

    pub(in crate::causal_lm) fn admit(
        self,
        m: &NativeCausalLm,
        mode: CausalLmAdaptationMode,
        prompts: &[&[usize]],
        ceiling: f64,
        criterion: CausalLmAnchorKlCriterion,
    ) -> CausalLmResult<()> {
        if let Self::RawThenSensitivityRay {
            probe_l2, basis, ..
        } = self
        {
            return sensitivity::admit(m, mode, prompts.len(), ceiling, criterion, probe_l2, basis);
        }
        if let Some((probe_l2, _)) = self.secant() {
            if prompts.len() < 2 || !probe_l2.is_finite() || probe_l2 <= 0. || probe_l2 > ceiling {
                return Err(
                    "secant direction needs two anchors and a positive probe within the ceiling"
                        .into(),
                );
            }
            m.admit_adaptation_distance(mode, probe_l2)?;
            if selected_weights(m, mode)
                .values()
                .map(|v| v.len())
                .sum::<usize>()
                > 1 << 20
            {
                return Err("secant direction exceeds selected parameter budget".into());
            }
            for &p in prompts {
                // Use the decoder's existing backward/checkpoint bounds for exactly p.len() inputs.
                m.adaptation_labeled_input_count([1, p.len() + 1], true)?;
            }
        }
        Ok(())
    }
}

type Blocks<'a> = BTreeMap<(String, String), &'a [f32]>;

pub(super) struct ResolvedSecant {
    policy: CausalLmAnchorDirection,
    probe_l2: f64,
    choice: Choice,
}

pub(super) fn visit_owner_blocks<'a>(
    m: &'a NativeCausalLm,
    mode: CausalLmAdaptationMode,
    mut visit: impl FnMut(&str, &str, &'a [f32], bool),
) {
    if let Some(l) = m
        .lora
        .as_ref()
        .filter(|_| mode != CausalLmAdaptationMode::Geometry)
    {
        let path = l.path();
        for (name, values) in l.factor_values() {
            visit(&path, name, values, true);
        }
    }
    if let Some(g) = m
        .geometry_artifact()
        .filter(|_| mode != CausalLmAdaptationMode::Lora)
    {
        for p in &g.record().placements {
            for t in &p.tensors {
                if t.role == GeometryAdapterTensorRole::Parameter {
                    visit(&p.path, &t.name, &t.values, t.requires_grad);
                }
            }
        }
    }
}

fn selected_weights(m: &NativeCausalLm, mode: CausalLmAdaptationMode) -> Blocks<'_> {
    let mut b = BTreeMap::new();
    visit_owner_blocks(m, mode, |path, name, v, trainable| {
        if trainable {
            b.insert((path.to_owned(), name.to_owned()), v);
        }
    });
    b
}

impl NativeCausalLm {
    fn anchor_normal(
        &self,
        p: &[usize],
        delta: Vec<f32>,
        mode: CausalLmAdaptationMode,
    ) -> CausalLmResult<ParameterGradients> {
        let (first, site) = self.first_adaptation_site()?;
        let shape = [1, p.len()];
        let last = [p.len() - 1];
        let (hidden, checkpoints) =
            self.checkpointed_hidden(self.embed(p, shape)?, shape, 0, first, None)?;
        let selected = hidden.gather_rows(&last)?;
        drop(hidden);
        let seed = Tensor::from_vec(1, self.config.vocabulary, delta)?;
        let adjoint = self.vocabulary_head_pullback(&seed)?;
        self.decoder_parameter_pullback(
            checkpoints,
            p.len(),
            selected,
            shape,
            Some(&last),
            adjoint,
            first,
            mode.decoder_target(&site),
            None,
        )
    }

    pub(super) fn secant_direction(
        &self,
        g: &ParameterGradients,
        mode: CausalLmAdaptationMode,
        spec: ResolvedSecant,
        bank: &mut AnchorBank<'_>,
    ) -> CausalLmResult<(ParameterGradients, CausalLmSecantReport)> {
        let ResolvedSecant {
            policy,
            probe_l2,
            choice,
        } = spec;
        let (probe, _) = self.step_parameter_ray(g, mode, probe_l2)?;
        let mut kls = Vec::with_capacity(bank.prompts.len());
        let mut rows = ProbeRows::new(choice);
        bank.observe(self, &probe, |i, kl, row| {
            kls.push(kl);
            rows.observe(i, kl, row);
        })?;
        drop(probe);
        let (peak, selected, row) = rows.finish(kls.len())?;
        let delta = bank.refs[selected].probability_delta(row.data())?;
        drop(row);
        let mut u = self.anchor_normal(bank.prompts[selected], delta, mode)?;
        let weights = selected_weights(self, mode);
        let (source_gradient_l2, normal_l2, projected_l2, projected_normal_cosine) =
            project(&weights, g, &mut u)?;
        let midpoint = if policy.is_rescue() {
            Some(Box::new(angular_midpoint(
                &weights,
                g,
                &mut u,
                source_gradient_l2,
                projected_l2,
            )?))
        } else {
            None
        };
        Ok((
            u,
            CausalLmSecantReport {
                policy,
                probe_per_anchor_kl: kls,
                peak_anchor: peak,
                selected_anchor: selected,
                loss_prompt_id: policy
                    .uses_loss_prompt()
                    .then(|| line_session::LinePromptId::new(bank.prompts[selected]).to_string()),
                source_gradient_l2,
                normal_l2,
                projected_l2,
                retained_fraction: projected_l2 / source_gradient_l2,
                projected_normal_cosine,
                midpoint,
                source_loss_dot_step: None,
            },
        ))
    }

    pub(super) fn source_loss_motion(
        &self,
        end: &Self,
        mode: CausalLmAdaptationMode,
        g: &ParameterGradients,
    ) -> CausalLmResult<f64> {
        let after = selected_weights(end, mode);
        let mut dot = sum::Sum::default();
        for (key, a) in selected_weights(self, mode) {
            let b = after.get(&key).ok_or("missing candidate secant block")?;
            let grad = gradient(g, &key, a.len())?;
            if a.len() != b.len() {
                return Err("secant candidate shape changed".into());
            }
            for ((&old, &new), &v) in a.iter().zip(*b).zip(grad) {
                dot.add((f64::from(new) - f64::from(old)) * f64::from(v));
            }
        }
        let dot = dot.value();
        if !dot.is_finite() || dot >= 0. {
            return Err("secant step is not a first-order descent step".into());
        }
        Ok(dot)
    }
}

fn gradient<'a>(
    g: &'a ParameterGradients,
    (path, name): &(String, String),
    n: usize,
) -> CausalLmResult<&'a [f32]> {
    g.get(path)
        .and_then(|b| b.get(name))
        .filter(|v| v.len() == n)
        .map(Vec::as_slice)
        .ok_or_else(|| "missing or malformed secant gradient block".into())
}

#[derive(Debug)]
pub(super) struct UnresolvedSecant(pub(super) &'static str);

impl std::fmt::Display for UnresolvedSecant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for UnresolvedSecant {}

// u is recycled as the shaped parameter map. Nontrainable entries are not coordinates.
fn project(
    w: &Blocks<'_>,
    g: &ParameterGradients,
    u: &mut ParameterGradients,
) -> CausalLmResult<(f64, f64, f64, f64)> {
    let (mut gg, mut uu, mut gu) = (
        sum::Sum::default(),
        sum::Sum::default(),
        sum::Sum::default(),
    );
    for (key, values) in w {
        for (&a, &b) in gradient(g, key, values.len())?
            .iter()
            .zip(gradient(u, key, values.len())?)
        {
            let (a, b) = (f64::from(a), f64::from(b));
            gg.add(a * a);
            uu.add(b * b);
            gu.add(a * b);
        }
    }
    let (gn, un, scale) = (
        gg.value().sqrt(),
        uu.value().sqrt(),
        gu.value() / uu.value(),
    );
    if !gn.is_finite() || !un.is_finite() {
        return Err("nonfinite secant normal or source gradient".into());
    }
    if gn == 0. || un == 0. {
        return Err(UnresolvedSecant("unresolved secant normal or source gradient").into());
    }
    if !scale.is_finite() {
        return Err("nonfinite secant projection coefficient".into());
    }
    let (mut hh, mut hu) = (sum::Sum::default(), sum::Sum::default());
    for (key, values) in w {
        let normal = u
            .get_mut(&key.0)
            .and_then(|b| b.get_mut(&key.1))
            .ok_or("missing secant block")?;
        for (&a, b) in gradient(g, key, values.len())?.iter().zip(normal) {
            let old = f64::from(*b);
            *b = (f64::from(a) - scale * old) as f32;
            hh.add(f64::from(*b).powi(2));
            hu.add(f64::from(*b) * old);
        }
    }
    let hn = hh.value().sqrt();
    if !hn.is_finite() {
        return Err("nonfinite projected gradient norm".into());
    }
    if hn < 0.001 * gn {
        return Err(UnresolvedSecant(
            "secant projection retains less than 0.001 of the source norm",
        )
        .into());
    }
    Ok((gn, un, hn, hu.value() / (hn * un)))
}

fn angular_midpoint(
    w: &Blocks<'_>,
    g: &ParameterGradients,
    h: &mut ParameterGradients,
    gn: f64,
    hn: f64,
) -> CausalLmResult<CausalLmSecantMidpoint> {
    // Equal-length rays, not equal coefficients of two differently sized gradients.
    let ratio = gn / hn;
    let (mut qq, mut qg, mut qh) = (
        sum::Sum::default(),
        sum::Sum::default(),
        sum::Sum::default(),
    );
    for (key, values) in w {
        let projected = h
            .get_mut(&key.0)
            .and_then(|b| b.get_mut(&key.1))
            .ok_or("missing midpoint block")?;
        for (&a, b) in gradient(g, key, values.len())?.iter().zip(projected) {
            let a = f64::from(a);
            let old = f64::from(*b);
            *b = (0.5 * (a + ratio * old)) as f32;
            let q = f64::from(*b);
            qq.add(q * q);
            qg.add(q * a);
            qh.add(q * old);
        }
    }
    let gradient_l2 = qq.value().sqrt();
    let source_cosine = qg.value() / (gradient_l2 * gn);
    let projected_cosine = qh.value() / (gradient_l2 * hn);
    if !gradient_l2.is_finite()
        || !source_cosine.is_finite()
        || !projected_cosine.is_finite()
        || source_cosine <= 0.
        || projected_cosine <= 0.
    {
        return Err("unrepresentable or non-descent angular midpoint".into());
    }
    Ok(CausalLmSecantMidpoint {
        gradient_l2,
        source_cosine,
        projected_cosine,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midpoint_bisects_unit_rays_not_unnormalized_gradients() {
        let w = Blocks::from([(("a".into(), "b".into()), &[0., 0.][..])]);
        let map = |v| ParameterGradients::from([("a".into(), BTreeMap::from([("b".into(), v)]))]);
        let g = map(vec![3., 4.]);
        let mut u = map(vec![2., 0.]);
        let (gn, _, hn, cos) = project(&w, &g, &mut u).unwrap();
        assert_eq!((gn, hn, cos), (5., 4., 0.));
        let mid = angular_midpoint(&w, &g, &mut u, gn, hn).unwrap();
        assert_eq!(u["a"]["b"], vec![1.5, 4.5]);
        assert!((mid.source_cosine - 0.9f64.sqrt()).abs() < 1e-15);
        assert!((mid.source_cosine - mid.projected_cosine).abs() < 1e-15);
        let zero = project(&w, &g, &mut map(vec![0., 0.])).unwrap_err();
        assert!(zero.downcast_ref::<UnresolvedSecant>().is_some());
        let malformed = project(&w, &g, &mut map(vec![f32::NAN, 0.])).unwrap_err();
        assert!(malformed.downcast_ref::<UnresolvedSecant>().is_none());
    }

    #[test]
    fn no_silent_fallback_on_degenerate_normals_or_misspelled_policy() {
        for json in [
            r#"{"kind":"loss_gradient","probe_l2":0.1}"#,
            r#"{"kind":"secant_projection","probe_l2":0.1,"anchor":"second"}"#,
            r#"{"kind":"secant_projection","probe_l2":0.1,"anchor":"largest","strength":0.5}"#,
            r#"{"kind":"secant_projection","anchor":"largest"}"#,
            r#"{"kind":"raw_then_secant_midpoint","probe_l2":0.1,"anchor":"largest","fraction":0.25}"#,
            r#"{"kind":"raw_then_secant_midpoint","anchor":"largest"}"#,
        ] {
            assert!(serde_json::from_str::<CausalLmAnchorDirection>(json).is_err());
        }
        assert!(
            serde_json::from_str::<CausalLmAnchorDirection>(r#"{"kind":"loss_gradient"}"#).is_ok()
        );
        let w = Blocks::from([(("adapter".into(), "gate".into()), &[1., 1.][..])]);
        let map = |v| {
            ParameterGradients::from([("adapter".into(), BTreeMap::from([("gate".into(), v)]))])
        };
        let g = map(vec![1., 0.]);
        for mut u in [
            map(vec![0., 0.]),
            map(vec![1., 0.0001]),
            map(vec![f32::NAN, 1.]),
        ] {
            assert!(project(&w, &g, &mut u).is_err());
        }
    }
}
