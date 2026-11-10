//! Fit an attention-output direction that improves matched contrasts in multiple groups.
use super::common_geometry::hull;
use super::*;

pub(super) const MAX_BANK_VALUES: usize = 1 << 22;
pub(super) const MAX_CAPTURE_TERMS: usize = 1 << 27;
pub const MAX_OUTPUT_CONTRAST_ROWS: usize = 2 * MAX_SCALAR_OUTPUT_PAIRS;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLmOutputMarginRule {
    EqualMean,
    CommonCosine,
}

#[derive(Debug, Serialize)]
pub struct CausalLmOutputMarginDirection {
    base: String,
    lora: Option<String>,
    geometry: Option<String>,
    readout: CausalLmReadoutAccumulator,
    layer: usize,
    width: usize,
    pub target_id: usize,
    pub alternative_id: usize,
    /// If present, coordinates are the right factor; otherwise they are an input-major weight matrix.
    left: Option<Vec<f64>>,
    coordinates: Option<Vec<f64>>,
    pub group_sizes: Vec<usize>,
    pub group_norms: Vec<f64>,
    pub slopes: Vec<f64>,
    pub rule: CausalLmOutputMarginRule,
    pub refusal: Option<CommonDirectionRefusal>,
    pub hull_norm: Option<f64>,
    pub unit_group_weights: Option<Vec<f64>>,
}

impl CausalLmOutputMarginDirection {
    /// Small metadata only: coordinates can be retrieved separately without expanding a rank-one axis.
    pub fn certificate(&self) -> serde_json::Value {
        serde_json::json!({"base": self.base, "lora": self.lora, "geometry": self.geometry, "readout": self.readout,
            "layer":self.layer,"width":self.width,"fixed_left":self.left.is_some(),"target_id":self.target_id,
            "alternative_id":self.alternative_id,"group_sizes":self.group_sizes,"group_norms":self.group_norms,
            "slopes":self.slopes,"rule":self.rule,"refusal":self.refusal,"hull_norm":self.hull_norm,
            "unit_group_weights":self.unit_group_weights})
    }
    pub fn coordinates(&self) -> Option<&[f64]> {
        self.coordinates.as_deref()
    }
    pub fn left(&self) -> Option<&[f64]> {
        self.left.as_deref()
    }

    /// Open an existing native scalar learner at zero, preserving both frozen adapter owners.
    pub fn scalar_adapter(
        &self,
        model: &NativeCausalLm,
    ) -> CausalLmResult<NativeCausalLmScalarOutput> {
        if model.base_model_identity() != self.base
            || model.lora_artifact_id() != self.lora.as_deref()
            || model.geometry_artifact_id() != self.geometry.as_deref()
            || model.readout_accumulator() != self.readout
        {
            return Err("output margin direction belongs to another decoder state".into());
        }
        NativeCausalLmScalarOutput::new(
            model,
            self.layer,
            self.left().ok_or("dense direction has no scalar factor")?,
            self.coordinates().ok_or("unresolved output direction")?,
            0.,
        )
    }
}

struct MarginOwner {
    base: String,
    lora: Option<String>,
    geometry: Option<String>,
    readout: CausalLmReadoutAccumulator,
    layer: usize,
    width: usize,
    target_id: usize,
    alternative_id: usize,
    left: Option<Vec<f64>>,
}

pub struct CausalLmOutputMarginBank {
    owner: MarginOwner,
    diffs: Vec<Vec<f64>>,
    gaps: Vec<[f64; 2]>,
}

/// A bounded immutable roster of last-token logits and projection adjoints. Indexed
/// counterfactual pairs can reuse either prompt without replaying the decoder.
pub struct CausalLmOutputContrastBank {
    owner: MarginOwner,
    gradients: Vec<Vec<f64>>,
    contrasts: Vec<CausalLmTokenContrast>,
}

struct FixedLeft<'a> {
    site: CausalLmProjectionSite,
    left: &'a [f64],
    right: Option<Vec<f64>>,
}

impl projection_probe::ProjectionCapture for FixedLeft<'_> {
    fn site(&self) -> CausalLmProjectionSite {
        self.site
    }
    fn record(&mut self, source: &Tensor, adjoint: &Tensor) -> CausalLmResult<()> {
        let (_, w) = source.shape();
        if source.shape() != adjoint.shape() || w != self.left.len() || self.right.is_some() {
            return Err("invalid fixed-left output capture".into());
        }
        let mut sums: Vec<_> = (0..w).map(|_| sum::Sum::default()).collect();
        // u^T X^T dZ = (Xu)^T dZ, without materializing a width-squared adjoint.
        for (x, d) in source
            .data()
            .chunks_exact(w)
            .zip(adjoint.data().chunks_exact(w))
        {
            let mut a = sum::Sum::default();
            for (&xi, &u) in x.iter().zip(self.left) {
                a.add(f64::from(xi) * u);
            }
            let projected = a.value();
            for (s, &v) in sums.iter_mut().zip(d) {
                s.add(projected * f64::from(v));
            }
        }
        self.right = Some(sums.iter().map(sum::Sum::value).collect());
        Ok(())
    }
}

impl NativeCausalLm {
    pub fn output_contrast_dimensions(
        &self,
        layer: usize,
        rows: usize,
        target: usize,
        alternative: usize,
        fixed_left: bool,
    ) -> CausalLmResult<usize> {
        if !(1..=MAX_OUTPUT_CONTRAST_ROWS).contains(&rows) || layer >= self.config.layers {
            return Err("invalid output contrast roster or layer".into());
        }
        self.next_token_contrast_input_count(1, target, alternative)?;
        let w = self.config.width;
        let dim = if fixed_left {
            w
        } else {
            product(&[w, w], MAX_BANK_VALUES, "output margin matrix")?
        };
        product(&[rows, dim], MAX_BANK_VALUES, "output margin bank")?;
        Ok(dim)
    }

    pub fn validate_output_roster_steps(
        &self,
        layer: usize,
        steps: &[usize],
        target: usize,
        alternative: usize,
        fixed_left: bool,
    ) -> CausalLmResult<()> {
        let dim =
            self.output_contrast_dimensions(layer, steps.len(), target, alternative, fixed_left)?;
        let spec = output_spec(layer, fixed_left);
        let mut total = 0usize;
        for &n in steps {
            self.projection_probe_dimensions(n, target, alternative, spec, true)?;
            total = total.checked_add(n).ok_or("output margin token overflow")?;
        }
        product(
            &[total, dim],
            MAX_CAPTURE_TERMS,
            "output margin capture work",
        )?;
        Ok(())
    }

    pub fn validate_output_left(&self, left: Option<&[f64]>) -> CausalLmResult<()> {
        if let Some(u) = left {
            if u.len() != self.config.width {
                return Err("expected a left factor of model width".into());
            }
            let norm = hull::norm(u);
            if !norm.is_finite() || (norm - 1.).abs() > 1e-9 {
                return Err("expected a finite unit left factor".into());
            }
        }
        Ok(())
    }

    fn output_margin_owner(
        &self,
        layer: usize,
        target: usize,
        alternative: usize,
        left: Option<&[f64]>,
    ) -> CausalLmResult<MarginOwner> {
        self.validate_output_left(left)?;
        Ok(MarginOwner {
            base: self.base_id.clone(),
            lora: self.lora_artifact_id().map(str::to_owned),
            geometry: self.geometry_artifact_id().map(str::to_owned),
            readout: self.readout_accumulator,
            layer,
            width: self.config.width,
            target_id: target,
            alternative_id: alternative,
            left: left.map(<[f64]>::to_vec),
        })
    }

    fn observe_output(
        &self,
        ids: &[usize],
        owner: &MarginOwner,
    ) -> CausalLmResult<(CausalLmTokenContrast, Vec<f64>)> {
        let spec = output_spec(owner.layer, owner.left.is_some());
        let (c, g) = if let Some(left) = owner.left.as_deref() {
            self.fixed_left_contrast(ids, owner.target_id, owner.alternative_id, spec, left)?
        } else {
            let r = self.projection_contrast_pullback(
                ids,
                owner.target_id,
                owner.alternative_id,
                spec,
            )?;
            (r.contrast, r.weight)
        };
        if !c.target_minus_alternative.is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite output margin observation".into());
        }
        Ok((c, g))
    }

    pub(super) fn fixed_left_contrast(
        &self,
        ids: &[usize],
        target: usize,
        alternative: usize,
        spec: CausalLmProjectionSpec,
        left: &[f64],
    ) -> CausalLmResult<(CausalLmTokenContrast, Vec<f64>)> {
        let mut capture = FixedLeft {
            site: spec.site,
            left,
            right: None,
        };
        let c = self.capture_projection_contrast(ids, target, alternative, spec, &mut capture)?;
        let g = capture.right.ok_or("unreached projection adjoint")?;
        if !c.target_minus_alternative.is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite fixed-left projection observation".into());
        }
        Ok((c, g))
    }

    pub(super) fn fixed_left_seed(
        &self,
        row: projection_probe::ProjectionLastRow,
        seed: &Tensor,
        spec: CausalLmProjectionSpec,
        left: &[f64],
    ) -> CausalLmResult<Vec<f64>> {
        let mut capture = FixedLeft {
            site: spec.site,
            left,
            right: None,
        };
        self.pull_projection_seed(row, spec, seed, &mut capture)?;
        let g = capture.right.ok_or("unreached seeded projection")?;
        if g.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite seeded projection adjoint".into());
        }
        Ok(g)
    }

    pub fn output_contrast_bank(
        &self,
        layer: usize,
        prompts: &[Vec<usize>],
        target: usize,
        alternative: usize,
        left: Option<&[f64]>,
    ) -> CausalLmResult<CausalLmOutputContrastBank> {
        self.output_contrast_dimensions(layer, prompts.len(), target, alternative, left.is_some())?;
        self.validate_output_roster_steps(
            layer,
            &prompts.iter().map(Vec::len).collect::<Vec<_>>(),
            target,
            alternative,
            left.is_some(),
        )?;
        let owner = self.output_margin_owner(layer, target, alternative, left)?;
        let mut seen = std::collections::HashSet::new();
        if prompts
            .iter()
            .any(|p| p.iter().any(|&id| id >= self.config.vocabulary) || !seen.insert(p.as_slice()))
        {
            return Err("output contrast roster needs unique prompts with valid token IDs".into());
        }
        let (contrasts, gradients) = prompts
            .iter()
            .map(|p| self.observe_output(p, &owner))
            .collect::<CausalLmResult<Vec<_>>>()?
            .into_iter()
            .unzip();
        Ok(CausalLmOutputContrastBank {
            owner,
            gradients,
            contrasts,
        })
    }

    /// Cache d[(target - alt)_positive - (target - alt)_negative] per pair. No bias or base update.
    /// A fixed left factor admits O(width) capture work per token rather than a dense parameter VJP.
    pub fn output_margin_bank(
        &self,
        layer: usize,
        positive: &[Vec<usize>],
        negative: &[Vec<usize>],
        target: usize,
        alternative: usize,
        left: Option<&[f64]>,
    ) -> CausalLmResult<CausalLmOutputMarginBank> {
        let n = positive.len();
        if n == 0
            || n != negative.len()
            || n > MAX_SCALAR_OUTPUT_PAIRS
            || layer >= self.config.layers
        {
            return Err("invalid output margin pair batch or layer".into());
        }
        let dim = self.output_contrast_dimensions(layer, n, target, alternative, left.is_some())?;
        let owner = self.output_margin_owner(layer, target, alternative, left)?;
        let spec = output_spec(layer, left.is_some());
        let mut tokens = 0usize;
        for row in positive.iter().chain(negative) {
            self.projection_probe_dimensions(row.len(), target, alternative, spec, true)?;
            tokens = tokens
                .checked_add(row.len())
                .ok_or("output margin token overflow")?;
            if row.iter().any(|&id| id >= self.config.vocabulary) {
                return Err("invalid output margin token ID".into());
            }
        }
        product(
            &[tokens, dim],
            MAX_CAPTURE_TERMS,
            "output margin capture work",
        )?;
        let mut diffs = Vec::with_capacity(n);
        let mut gaps = Vec::with_capacity(n);
        for (pos, neg) in positive.iter().zip(negative) {
            let (p, mut grad) = self.observe_output(pos, &owner)?;
            let (q, other) = self.observe_output(neg, &owner)?;
            for (x, y) in grad.iter_mut().zip(other) {
                *x -= y;
            }
            if grad.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite output margin observation".into());
            }
            diffs.push(grad);
            gaps.push([p.target_minus_alternative, q.target_minus_alternative]);
        }
        Ok(CausalLmOutputMarginBank { owner, diffs, gaps })
    }
}

fn output_spec(layer: usize, fixed_left: bool) -> CausalLmProjectionSpec {
    CausalLmProjectionSpec {
        layer,
        site: CausalLmProjectionSite::Output,
        basis: if fixed_left {
            CausalLmProjectionBasis::Diagonal
        } else {
            CausalLmProjectionBasis::Dense
        },
    }
}

impl CausalLmOutputContrastBank {
    pub fn contrasts(&self) -> &[CausalLmTokenContrast] {
        &self.contrasts
    }
    pub fn gradient(&self, index: usize) -> CausalLmResult<&[f64]> {
        self.gradients
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| "output roster index out of bounds".into())
    }

    pub fn direction(
        &self,
        pairs: &[[usize; 2]],
        groups: &[usize],
        rule: CausalLmOutputMarginRule,
    ) -> CausalLmResult<CausalLmOutputMarginDirection> {
        if pairs.is_empty()
            || pairs.len() > MAX_SCALAR_OUTPUT_PAIRS
            || pairs.iter().flatten().any(|&i| i >= self.gradients.len())
        {
            return Err("invalid indexed output margin pairs".into());
        }
        self.owner.fit(
            pairs.len(),
            self.gradients[0].len(),
            groups,
            rule,
            |i, sums| {
                let [p, n] = pairs[i];
                for ((s, &a), &b) in sums
                    .iter_mut()
                    .zip(&self.gradients[p])
                    .zip(&self.gradients[n])
                {
                    s.add(a - b);
                }
            },
        )
    }
}

impl CausalLmOutputMarginBank {
    pub fn pair_contrasts(&self) -> &[[f64; 2]] {
        &self.gaps
    }
    pub fn pair_gradients(&self) -> &[Vec<f64>] {
        &self.diffs
    }

    /// Equal example weights within each group. CommonCosine maximizes the worst *normalized*
    /// group slope in the actual admitted coordinate space; report raw slopes separately.
    pub fn direction(
        &self,
        groups: &[usize],
        rule: CausalLmOutputMarginRule,
    ) -> CausalLmResult<CausalLmOutputMarginDirection> {
        self.owner.fit(
            self.diffs.len(),
            self.diffs[0].len(),
            groups,
            rule,
            |i, sums| {
                for (s, &x) in sums.iter_mut().zip(&self.diffs[i]) {
                    s.add(x);
                }
            },
        )
    }
}

impl MarginOwner {
    fn fit(
        &self,
        count: usize,
        width: usize,
        groups: &[usize],
        rule: CausalLmOutputMarginRule,
        add: impl Fn(usize, &mut [sum::Sum]),
    ) -> CausalLmResult<CausalLmOutputMarginDirection> {
        if groups.len() != count || groups.iter().any(|g| *g >= MAX_COMMON_GEOMETRY_GROUPS) {
            return Err("invalid output margin group assignment".into());
        }
        let n = groups.iter().max().unwrap() + 1;
        let mut sizes = vec![0; n];
        for &g in groups {
            sizes[g] += 1;
        }
        if n < 2 || sizes.contains(&0) {
            return Err("expected 2 to 8 contiguous nonempty output margin groups".into());
        }
        product(&[n, width], MAX_BANK_VALUES, "output group means")?;
        let means: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let mut sums: Vec<_> = (0..width).map(|_| sum::Sum::default()).collect();
                for (i, &g) in groups.iter().enumerate() {
                    if g == k {
                        add(i, &mut sums);
                    }
                }
                sums.into_iter().map(|s| s.mean(sizes[k] as u64)).collect()
            })
            .collect();
        let norms: Vec<_> = means.iter().map(|v| hull::norm(v)).collect();
        if norms.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite group gradient norm".into());
        }
        let mut r = CausalLmOutputMarginDirection {
            base: self.base.clone(),
            lora: self.lora.clone(),
            geometry: self.geometry.clone(),
            readout: self.readout,
            layer: self.layer,
            width: self.width,
            target_id: self.target_id,
            alternative_id: self.alternative_id,
            left: self.left.clone(),
            coordinates: None,
            group_sizes: sizes,
            group_norms: norms,
            rule,
            slopes: Vec::new(),
            refusal: None,
            hull_norm: None,
            unit_group_weights: None,
        };
        let axis = match rule {
            CausalLmOutputMarginRule::EqualMean => (0..width)
                .map(|i| {
                    let mut s = sum::Sum::default();
                    for m in &means {
                        s.add(m[i]);
                    }
                    s.mean(n as u64)
                })
                .collect(),
            CausalLmOutputMarginRule::CommonCosine => match hull::solve(&means) {
                Ok(h) => {
                    r.hull_norm = Some(h.hull_norm);
                    r.unit_group_weights = Some(h.alpha);
                    h.vector
                }
                Err(e) => {
                    if let Some(reason) = e.downcast_ref::<CommonDirectionRefusal>() {
                        r.refusal = Some(*reason);
                        return Ok(r);
                    } else {
                        return Err(e);
                    }
                }
            },
        };
        let norm = hull::norm(&axis);
        if !norm.is_finite() {
            return Err("nonfinite output direction".into());
        }
        if norm == 0. {
            r.refusal = Some(CommonDirectionRefusal::Unresolved);
            return Ok(r);
        }
        let unit: Vec<_> = axis.iter().map(|v| v / norm).collect();
        r.slopes = means.iter().map(|v| hull::dot(v, &unit)).collect();
        r.coordinates = Some(unit);
        Ok(r)
    }
}
