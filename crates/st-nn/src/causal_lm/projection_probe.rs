//! Local coordinates at one affine projection, without editing the frozen model or advancing a cache.
use super::*;

const MAX_COORDINATES: usize = 1 << 20;
const MAX_REDUCTION_TERMS: usize = 1 << 26;
const PROJECTION_ADJOINT_HEADROOM: f64 = (1u128 << 64) as f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalLmProjectionSite {
    Query,
    Key,
    Value,
    Output,
}

impl CausalLmProjectionSite {
    pub(super) fn qkv_part(self) -> Option<usize> {
        match self {
            Self::Query => Some(0),
            Self::Key => Some(1),
            Self::Value => Some(2),
            Self::Output => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
            Self::Output => "output",
        }
    }
}

impl std::str::FromStr for CausalLmProjectionSite {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "query" => Ok(Self::Query),
            "key" => Ok(Self::Key),
            "value" => Ok(Self::Value),
            "output" => Ok(Self::Output),
            _ => Err("projection site must be query, key, value or output"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalLmProjectionBasis {
    Diagonal,
    Dense,
}

impl CausalLmProjectionBasis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Diagonal => "diagonal",
            Self::Dense => "dense",
        }
    }
}

impl std::str::FromStr for CausalLmProjectionBasis {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "diagonal" => Ok(Self::Diagonal),
            "dense" => Ok(Self::Dense),
            _ => Err("projection basis must be diagonal or dense"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalLmProjectionSpec {
    pub layer: usize,
    pub site: CausalLmProjectionSite,
    pub basis: CausalLmProjectionBasis,
}

#[derive(Debug)]
pub struct CausalLmProjectionPullback {
    pub base_model_identity: String,
    pub lora_artifact_id: Option<String>,
    pub geometry_artifact_id: Option<String>,
    pub readout_accumulator: CausalLmReadoutAccumulator,
    pub spec: CausalLmProjectionSpec,
    pub contrast: CausalLmTokenContrast,
    pub bias: Vec<f64>,
    /// Diagonal channels, or a row-major [input feature, output feature] matrix.
    pub weight: Vec<f64>,
}

pub(super) trait ProjectionCapture {
    fn site(&self) -> CausalLmProjectionSite;
    fn record(&mut self, source: &Tensor, adjoint: &Tensor) -> CausalLmResult<()>;
}

/// One checkpointed prefix forward. Clones share the immutable tensor buffers,
/// so independent last-logit seeds need not rerun the factual decoder.
#[derive(Clone)]
pub(super) struct ProjectionLastRow {
    steps: usize,
    checkpoints: Vec<Tensor>,
    selected: Tensor,
    pub(super) logits: Tensor,
}

fn binary_exponent(value: f64) -> i32 {
    let bits = value.to_bits();
    let biased = (bits >> 52) as i32;
    if biased == 0 {
        (63 - bits.leading_zeros()) as i32 - 1074
    } else {
        biased - 1023
    }
}

fn binary_scale(shift: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&shift));
    f64::from_bits(((shift + 1023) as u64) << 52)
}

fn projection_adjoint_scale(wide: &[f64]) -> CausalLmResult<f64> {
    let (mut peak, mut smallest) = (0.0_f64, f64::INFINITY);
    for magnitude in wide.iter().map(|x| x.abs()) {
        if !magnitude.is_finite() {
            return Err("nonfinite projection adjoint".into());
        }
        peak = peak.max(magnitude);
        if magnitude > 0. {
            smallest = smallest.min(magnitude);
        }
    }
    if peak == 0. || (peak < PROJECTION_ADJOINT_HEADROOM && smallest as f32 != 0.) {
        return Ok(1.);
    }
    let peak_exp = binary_exponent(peak);
    let small_exp = binary_exponent(smallest);
    // The largest component bounds overflow; the smallest nonzero component
    // bounds underflow. Check the endpoint casts to handle rounding ties.
    let mut lower = (peak_exp - 127).max(-1022);
    if !((peak / binary_scale(lower)) as f32).is_finite() {
        lower += 1;
    }
    let mut upper = (small_exp + 150).min(1023);
    if (smallest / binary_scale(upper)) as f32 == 0. {
        upper -= 1;
    }
    if lower > upper {
        return Err("no common scale preserves the f32 decoder adjoint range".into());
    }
    // Give the decoder exponent headroom, preferring a normal smallest
    // component when the range permits it.
    let normal_upper = small_exp + 126;
    if normal_upper >= lower {
        upper = upper.min(normal_upper);
    }
    let preferred = if peak >= PROJECTION_ADJOINT_HEADROOM {
        peak_exp - 32
    } else {
        0
    };
    Ok(binary_scale(preferred.clamp(lower, upper)))
}

fn pack_projection_adjoint(shape: (usize, usize), wide: &[f64]) -> CausalLmResult<(Tensor, f64)> {
    let scale = projection_adjoint_scale(wide)?;
    let narrowed = wide
        .iter()
        .copied()
        .map(|value| {
            let narrow = (value / scale) as f32;
            if !narrow.is_finite() || (value != 0. && narrow == 0.) {
                return Err("wide layernorm adjoint exceeds the f32 decoder range");
            }
            Ok(narrow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((Tensor::from_vec(shape.0, shape.1, narrowed)?, scale))
}

#[cfg(test)]
mod projection_adjoint_range_tests {
    use super::*;

    #[test]
    fn a_common_scale_can_keep_a_wide_layernorm_cotangent_and_its_small_components() {
        let large = 2f64.powi(178);
        let small = 2f64.powi(-15);
        let values = [large, -large, small, -small];
        let (packed, scale) = pack_projection_adjoint((1, 4), &values).unwrap();
        for (&component, value) in packed.data().iter().zip(values) {
            assert!(component.is_normal());
            assert_eq!(f64::from(component) * scale, value);
        }
    }

    #[test]
    fn scales_preserve_subnormal_endpoints_or_refuse_an_impossible_range() {
        for values in [
            [2f64.powi(177), 2f64.powi(-96)],
            [f64::from_bits(2), f64::from_bits(1)],
        ] {
            let (packed, scale) = pack_projection_adjoint((1, 2), &values).unwrap();
            for (&component, value) in packed.data().iter().zip(values) {
                assert_ne!(component, 0.);
                assert_eq!(f64::from(component) * scale, value);
            }
        }
        let error =
            pack_projection_adjoint((1, 2), &[2f64.powi(200), 2f64.powi(-100)]).unwrap_err();
        assert!(error.to_string().contains("no common scale"));
    }
}

struct Capture {
    site: CausalLmProjectionSite,
    basis: CausalLmProjectionBasis,
    values: Option<(Vec<f64>, Vec<f64>)>,
}

impl ProjectionCapture for Capture {
    fn site(&self) -> CausalLmProjectionSite {
        self.site
    }

    fn record(&mut self, source: &Tensor, adjoint: &Tensor) -> CausalLmResult<()> {
        let (_, w) = source.shape();
        if source.shape() != adjoint.shape() || self.values.is_some() {
            return Err("invalid projection capture".into());
        }
        let nw = if self.basis == CausalLmProjectionBasis::Diagonal {
            w
        } else {
            w * w
        };
        let mut b: Vec<_> = (0..w).map(|_| sum::Sum::default()).collect();
        let mut dw: Vec<_> = (0..nw).map(|_| sum::Sum::default()).collect();
        for (x, d) in source
            .data()
            .chunks_exact(w)
            .zip(adjoint.data().chunks_exact(w))
        {
            for (s, &v) in b.iter_mut().zip(d) {
                s.add(f64::from(v));
            }
            match self.basis {
                CausalLmProjectionBasis::Diagonal => {
                    for j in 0..w {
                        dw[j].add(f64::from(x[j]) * f64::from(d[j]));
                    }
                }
                CausalLmProjectionBasis::Dense => {
                    for (&xi, row) in x.iter().zip(dw.chunks_exact_mut(w)) {
                        for (s, &dj) in row.iter_mut().zip(d) {
                            s.add(f64::from(xi) * f64::from(dj));
                        }
                    }
                }
            }
        }
        let finish = |v: Vec<sum::Sum>| v.into_iter().map(|x| x.value()).collect::<Vec<_>>();
        let (b, dw) = (finish(b), finish(dw));
        if b.iter().chain(&dw).any(|v| !v.is_finite()) {
            return Err("nonfinite projection pullback".into());
        }
        self.values = Some((b, dw));
        Ok(())
    }
}

pub(super) struct ProjectionDelta<'a> {
    pub(super) spec: CausalLmProjectionSpec,
    bias: &'a [f32],
    weight: &'a [f32],
}

impl ProjectionDelta<'_> {
    pub(super) fn apply(
        &self,
        source: &Tensor,
        projection: &Tensor,
        part: usize,
    ) -> CausalLmResult<Tensor> {
        let (rows, w) = source.shape();
        let (n, stride) = projection.shape();
        if n != rows
            || stride
                != (if self.spec.site == CausalLmProjectionSite::Output {
                    1
                } else {
                    3
                }) * w
            || part >= stride / w
        {
            return Err("invalid projection intervention layout".into());
        }
        let mut values = projection.data().to_vec();
        for (x, row) in source
            .data()
            .chunks_exact(w)
            .zip(values.chunks_exact_mut(stride))
        {
            for (j, old) in row[part * w..(part + 1) * w].iter_mut().enumerate() {
                let mut d = sum::Sum::default();
                d.add(f64::from(self.bias[j]));
                match self.spec.basis {
                    CausalLmProjectionBasis::Diagonal => {
                        d.add(f64::from(x[j]) * f64::from(self.weight[j]))
                    }
                    CausalLmProjectionBasis::Dense => {
                        for (&xi, weights) in x.iter().zip(self.weight.chunks_exact(w)) {
                            d.add(f64::from(xi) * f64::from(weights[j]));
                        }
                    }
                }
                let d = d.value();
                if d != 0. {
                    *old = (f64::from(*old) + d) as f32;
                }
                if !old.is_finite() {
                    return Err("nonfinite perturbed projection".into());
                }
            }
        }
        Ok(Tensor::from_vec(rows, stride, values)?)
    }
}

impl NativeCausalLm {
    /// Admit prompt, workspace, and returned parameter dimensions before a binding copies buffers.
    pub fn projection_probe_dimensions(
        &self,
        steps: usize,
        target: usize,
        alternative: usize,
        spec: CausalLmProjectionSpec,
        backward: bool,
    ) -> CausalLmResult<(usize, usize)> {
        self.check_contrast_candidates(target, alternative)?;
        self.projection_work_dimensions(steps, spec, backward)
    }

    pub(super) fn projection_work_dimensions(
        &self,
        steps: usize,
        spec: CausalLmProjectionSpec,
        backward: bool,
    ) -> CausalLmResult<(usize, usize)> {
        self.input_count([1, steps], true)?;
        let c = &self.config;
        if spec.layer >= c.layers {
            return Err("projection probe layer is outside the model".into());
        }
        let w = c.width;
        let count = match spec.basis {
            CausalLmProjectionBasis::Diagonal => w,
            CausalLmProjectionBasis::Dense => {
                product(&[w, w], MAX_COORDINATES, "dense projection coordinates")?
            }
        };
        if count.checked_add(w).is_none_or(|n| n > MAX_COORDINATES) {
            return Err("projection probe exceeds coordinate budget".into());
        }
        product(
            &[steps, count],
            MAX_REDUCTION_TERMS,
            "projection reduction terms",
        )?;
        if backward {
            product(
                &[c.layers - spec.layer, steps, w],
                self.limits.max_training_checkpoint_elements,
                "projection checkpoints",
            )?;
        }
        Ok((w, count))
    }

    /// Signed last-logit VJP at zero additive virtual projection parameters.
    /// Q/K/V use the input to the frozen QKV map; output uses merged values before c_proj.
    pub fn projection_contrast_pullback(
        &self,
        prompt: &[usize],
        target: usize,
        alternative: usize,
        spec: CausalLmProjectionSpec,
    ) -> CausalLmResult<CausalLmProjectionPullback> {
        let mut capture = Capture {
            site: spec.site,
            basis: spec.basis,
            values: None,
        };
        let contrast =
            self.capture_projection_contrast(prompt, target, alternative, spec, &mut capture)?;
        let (bias, weight) = capture
            .values
            .ok_or("projection adjoint did not reach its site")?;
        Ok(CausalLmProjectionPullback {
            base_model_identity: self.base_id.clone(),
            lora_artifact_id: self.lora_artifact_id().map(str::to_owned),
            geometry_artifact_id: self.geometry_artifact_id().map(str::to_owned),
            readout_accumulator: self.readout_accumulator,
            spec,
            contrast,
            bias,
            weight,
        })
    }

    pub(super) fn capture_projection_contrast(
        &self,
        prompt: &[usize],
        target: usize,
        alternative: usize,
        spec: CausalLmProjectionSpec,
        capture: &mut dyn ProjectionCapture,
    ) -> CausalLmResult<CausalLmTokenContrast> {
        let steps = prompt.len();
        self.check_contrast_candidates(target, alternative)?;
        let row = self.projection_last_row(prompt, spec)?;
        let contrast = Self::contrast_row(row.logits.data(), steps, target, alternative);
        let mut seed = vec![0.; self.config.vocabulary];
        seed[target] = 1.;
        seed[alternative] = -1.;
        self.pull_projection_seed(
            row,
            spec,
            &Tensor::from_vec(1, self.config.vocabulary, seed)?,
            capture,
        )?;
        Ok(contrast)
    }

    pub(super) fn projection_last_row(
        &self,
        prompt: &[usize],
        spec: CausalLmProjectionSpec,
    ) -> CausalLmResult<ProjectionLastRow> {
        let steps = prompt.len();
        self.projection_work_dimensions(steps, spec, true)?;
        self.projection_last_row_from_hidden(self.embed(prompt, [1, steps])?, steps, 0, spec)
    }

    pub(super) fn projection_last_row_from_hidden(
        &self,
        input: Tensor,
        steps: usize,
        start: usize,
        spec: CausalLmProjectionSpec,
    ) -> CausalLmResult<ProjectionLastRow> {
        let shape = [1, steps];
        let last = [steps - 1];
        let (hidden, checkpoints) =
            self.checkpointed_hidden(input, shape, start, spec.layer, None)?;
        let selected = hidden.gather_rows(&last)?;
        let normalized = self
            .final_norm
            .forward(&selected, self.config.norm_epsilon)?;
        let logits = self.project_readout(&normalized)?;
        Ok(ProjectionLastRow {
            steps,
            checkpoints,
            selected,
            logits,
        })
    }

    pub(super) fn pull_projection_seed(
        &self,
        row: ProjectionLastRow,
        spec: CausalLmProjectionSpec,
        seed: &Tensor,
        capture: &mut dyn ProjectionCapture,
    ) -> CausalLmResult<()> {
        let upstream = self.vocabulary_head_pullback(seed)?;
        self.pull_projection_upstream(row, spec, upstream, capture)
    }

    pub(super) fn pull_projection_upstream(
        &self,
        row: ProjectionLastRow,
        spec: CausalLmProjectionSpec,
        upstream: Tensor,
        capture: &mut dyn ProjectionCapture,
    ) -> CausalLmResult<()> {
        let ProjectionLastRow {
            steps,
            checkpoints,
            selected,
            ..
        } = row;
        self.decoder_parameter_pullback(
            checkpoints,
            steps,
            selected,
            [1, steps],
            Some(&[steps - 1]),
            upstream,
            spec.layer,
            learning::DecoderTarget::Projection(capture),
            None,
        )?;
        Ok(())
    }

    pub(super) fn pull_projection_wide_upstream(
        &self,
        row: ProjectionLastRow,
        spec: CausalLmProjectionSpec,
        upstream: &[f64],
        capture: &mut dyn ProjectionCapture,
    ) -> CausalLmResult<f64> {
        let ProjectionLastRow {
            steps,
            checkpoints,
            selected,
            ..
        } = row;
        let narrow: Vec<_> = upstream.iter().map(|&value| value as f32).collect();
        let gain = &self.final_norm.gamma;
        let epsilon = self.config.norm_epsilon;
        let normal = if upstream
            .iter()
            .zip(&narrow)
            .all(|(&wide, &value)| value.is_finite() && (wide == 0. || value != 0.))
        {
            let seed = Tensor::from_vec(selected.shape().0, selected.shape().1, narrow)?;
            match selected.layer_norm_input_backward(gain, &seed, epsilon) {
                Ok(value) => Some(value),
                Err(st_tensor::TensorError::NonFiniteValue {
                    label: "layernorm_backward_input_grad",
                    ..
                }) => None,
                Err(error) => return Err(error.into()),
            }
        } else {
            None
        };
        let (d_selected, scale) = if let Some(normal) = normal.filter(|tensor| {
            tensor
                .data()
                .iter()
                .all(|value| f64::from(value.abs()) < PROJECTION_ADJOINT_HEADROOM)
        }) {
            (normal, 1.)
        } else {
            let wide = selected.layer_norm_input_backward_wide(gain, upstream, epsilon)?;
            pack_projection_adjoint(selected.shape(), &wide)?
        };
        drop(selected);
        self.decoder_parameter_pullback_from_selected(
            checkpoints,
            steps,
            [1, steps],
            Some(&[steps - 1]),
            d_selected,
            spec.layer,
            learning::DecoderTarget::Projection(capture),
            None,
        )?;
        Ok(scale)
    }

    /// Re-run the true decoder with z <- f32(z + bias + input * delta_weight) at exactly one
    /// site's output, before the attention softmax or existing output geometry. No base export
    /// or persistent adaptation: ordinary inference and cache identities are unchanged.
    pub fn projection_perturbed_contrast(
        &self,
        prompt: &[usize],
        target: usize,
        alternative: usize,
        spec: CausalLmProjectionSpec,
        bias: &[f32],
        weight: &[f32],
    ) -> CausalLmResult<CausalLmTokenContrast> {
        let steps = prompt.len();
        let dims = self.projection_probe_dimensions(steps, target, alternative, spec, false)?;
        if (bias.len(), weight.len()) != dims || bias.iter().chain(weight).any(|v| !v.is_finite()) {
            return Err("projection deltas have wrong dimensions or nonfinite values".into());
        }
        let shape = [1, steps];
        let hidden = self.forward_blocks(self.embed(prompt, shape)?, shape, 0..spec.layer)?;
        let perturbed = self.run_block(
            spec.layer,
            hidden,
            shape,
            decoder::BlockPass::Projection(ProjectionDelta { spec, bias, weight }),
        )?;
        let end =
            self.forward_blocks(perturbed.output, shape, spec.layer + 1..self.config.layers)?;
        let last = self
            .final_norm
            .forward(&end.gather_rows(&[steps - 1])?, self.config.norm_epsilon)?;
        Ok(Self::contrast_row(
            self.project_readout(&last)?.data(),
            steps,
            target,
            alternative,
        ))
    }
}
