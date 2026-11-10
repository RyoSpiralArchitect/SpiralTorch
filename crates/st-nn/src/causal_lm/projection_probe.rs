//! Local coordinates at one affine projection, without editing the frozen model or advancing a cache.
use super::*;

const MAX_COORDINATES: usize = 1 << 20;
const MAX_REDUCTION_TERMS: usize = 1 << 26;

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
        let shape = [1, steps];
        let last = [steps - 1];
        let (hidden, checkpoints) =
            self.checkpointed_hidden(self.embed(prompt, shape)?, shape, 0, spec.layer, None)?;
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
        let ProjectionLastRow {
            steps,
            checkpoints,
            selected,
            ..
        } = row;
        let upstream = self.vocabulary_head_pullback(seed)?;
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
