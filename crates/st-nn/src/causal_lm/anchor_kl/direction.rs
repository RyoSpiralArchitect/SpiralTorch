use super::*;
use crate::geometry_artifact::GeometryAdapterTensorRole;
use crate::geometry_runtime::ParameterGradients;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CausalLmAnchorDirection {
    LossGradient {},
    /// Estimate one anchor normal using a finite probe, then remove its component.
    SecantProjection {
        probe_l2: f64,
        anchor: CausalLmSecantAnchor,
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
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmSecantReport {
    pub policy: CausalLmAnchorDirection,
    pub probe_per_anchor_kl: Vec<f64>,
    pub peak_anchor: usize,
    pub selected_anchor: usize,
    pub source_gradient_l2: f64,
    pub normal_l2: f64,
    pub projected_l2: f64,
    pub retained_fraction: f64,
    /// After projecting and rounding the new ray coordinates to f32.
    pub projected_normal_cosine: f64,
    /// The original loss gradient dotted with the actual f32 parameter displacement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_loss_dot_step: Option<f64>,
}

impl CausalLmAnchorDirection {
    pub(in crate::causal_lm) fn admit(
        self,
        m: &NativeCausalLm,
        mode: CausalLmAdaptationMode,
        prompts: &[&[usize]],
        ceiling: f64,
    ) -> CausalLmResult<()> {
        if let Self::SecantProjection { probe_l2, .. } = self {
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

fn selected_weights(m: &NativeCausalLm, mode: CausalLmAdaptationMode) -> Blocks<'_> {
    let mut b = BTreeMap::new();
    if let Some(l) = m
        .lora
        .as_ref()
        .filter(|_| mode != CausalLmAdaptationMode::Geometry)
    {
        for (name, values) in l.factor_values() {
            b.insert((l.path(), name.into()), values);
        }
    }
    if let Some(g) = m
        .geometry_artifact()
        .filter(|_| mode != CausalLmAdaptationMode::Lora)
    {
        for p in &g.record().placements {
            for t in &p.tensors {
                if t.role == GeometryAdapterTensorRole::Parameter && t.requires_grad {
                    b.insert((p.path.clone(), t.name.clone()), t.values.as_slice());
                }
            }
        }
    }
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
        prompts: &[&[usize]],
        probe_l2: f64,
        anchor: CausalLmSecantAnchor,
        refs: &mut Vec<Reference>,
    ) -> CausalLmResult<(ParameterGradients, CausalLmSecantReport)> {
        let (probe, _) = self.step_parameter_ray(g, mode, probe_l2)?;
        let mut kls = Vec::with_capacity(prompts.len());
        let mut rows = Vec::with_capacity(prompts.len());
        let mut peak = 0;
        for (i, p) in prompts.iter().enumerate() {
            let r = Reference::new(self.next_token_logits(p)?.data())?;
            let row = probe.next_token_logits(p)?;
            kls.push(r.forward_kl(row.data())?);
            if kls[i] > kls[peak] {
                peak = i;
            }
            refs.push(r);
            rows.push(row);
        }
        drop(probe);
        let selected = match anchor {
            CausalLmSecantAnchor::Largest => peak,
            CausalLmSecantAnchor::Successor => (peak + 1) % prompts.len(),
        };
        let delta = refs[selected].probability_delta(rows[selected].data())?;
        drop(rows);
        let mut u = self.anchor_normal(prompts[selected], delta, mode)?;
        let (source_gradient_l2, normal_l2, projected_l2, projected_normal_cosine) =
            project(&selected_weights(self, mode), g, &mut u)?;
        Ok((
            u,
            CausalLmSecantReport {
                policy: CausalLmAnchorDirection::SecantProjection { probe_l2, anchor },
                probe_per_anchor_kl: kls,
                peak_anchor: peak,
                selected_anchor: selected,
                source_gradient_l2,
                normal_l2,
                projected_l2,
                retained_fraction: projected_l2 / source_gradient_l2,
                projected_normal_cosine,
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
    if !scale.is_finite() || !gn.is_finite() || gn == 0. || !un.is_finite() || un == 0. {
        return Err("unresolved secant normal or source gradient".into());
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
    if !hn.is_finite() || hn < 0.001 * gn {
        return Err(
            "secant projection retains less than 0.001 of the source norm or is nonfinite".into(),
        );
    }
    Ok((gn, un, hn, hu.value() / (hn * un)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_silent_fallback_on_degenerate_normals_or_misspelled_policy() {
        for json in [
            r#"{"kind":"loss_gradient","probe_l2":0.1}"#,
            r#"{"kind":"secant_projection","probe_l2":0.1,"anchor":"second"}"#,
            r#"{"kind":"secant_projection","probe_l2":0.1,"anchor":"largest","strength":0.5}"#,
            r#"{"kind":"secant_projection","anchor":"largest"}"#,
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
