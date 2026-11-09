//! One learned coefficient along a fixed rank-one attention output direction.
use super::*;
use projection_probe::ProjectionCapture;
use sha2::{Digest, Sha256};
use st_tensor::PackedLayout;

mod search;
pub use search::{
    CausalLmScalarPairLoss, CausalLmScalarSearch, CausalLmScalarSearchReport,
    CausalLmScalarSearchStatus, CausalLmScalarTrial, MAX_SCALAR_SEARCH_TRIALS,
};

pub const MAX_SCALAR_OUTPUT_PAIRS: usize = 32;
const MAX_MATRIX: usize = 1 << 22;
const MAX_JSON: usize = 1 << 20;
const SCHEMA: &str = "spiraltorch.causal_lm.scalar_output.v1";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    base: String,
    lora: Option<String>,
    geometry: Option<String>,
    readout: CausalLmReadoutAccumulator,
    layer: usize,
    steps: u64,
    coefficient_bits: u64,
    left_bits: Vec<u64>,
    right_bits: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct NativeCausalLmScalarOutput {
    source: Arc<NativeCausalLm>,
    // Never expose this as the original model: its packed output differs at nonzero scale.
    current: NativeCausalLm,
    layer: usize,
    left: Arc<[f64]>,
    right: Arc<[f64]>,
    coefficient: f64,
    steps: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmScalarPairStep {
    pub step: u64,
    pub before: f64,
    pub after: f64,
    pub loss: f64,
    pub gradient: f64,
    pub gradient_prompt_evaluations: usize,
    /// Per pair: positive logit contrast, negative contrast, and their unsigned-axis derivatives.
    pub pairs: Vec<[f64; 4]>,
}

struct PairGradient {
    prompt_evaluations: usize,
    loss: f64,
    gradient: f64,
    pairs: Vec<[f64; 4]>,
}

/// Cache only within one exact model/coefficient and one target pair. Repeated pairs keep their
/// full objective weight; only the identical prompt computation is shared.
fn observe_pairs<'a, T: Copy>(
    positive: &'a [Vec<usize>],
    negative: &'a [Vec<usize>],
    mut observe: impl FnMut(&[usize]) -> CausalLmResult<T>,
) -> CausalLmResult<(usize, Vec<[T; 2]>)> {
    use std::collections::{hash_map::Entry, HashMap};
    let mut cache: HashMap<&'a [usize], T> = HashMap::new();
    let mut score = |ids: &'a [usize]| -> CausalLmResult<T> {
        Ok(match cache.entry(ids) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => *e.insert(observe(ids)?),
        })
    };
    let mut rows = Vec::with_capacity(positive.len());
    for (p, n) in positive.iter().zip(negative) {
        rows.push([score(p)?, score(n)?]);
    }
    Ok((cache.len(), rows))
}

pub(super) fn logistic(m: f64) -> (f64, f64) {
    let e = (-m.abs()).exp();
    (
        (-m).max(0.) + e.ln_1p(),
        -(if m >= 0. { e } else { 1. }) / (1. + e),
    )
}

struct Along<'a> {
    left: &'a [f64],
    right: &'a [f64],
    value: Option<f64>,
}

fn dot(row: &[f32], axis: &[f64]) -> f64 {
    let mut sum = sum::Sum::default();
    for (&x, &v) in row.iter().zip(axis) {
        sum.add(f64::from(x) * v);
    }
    sum.value()
}

impl ProjectionCapture for Along<'_> {
    fn site(&self) -> CausalLmProjectionSite {
        CausalLmProjectionSite::Output
    }
    fn record(&mut self, source: &Tensor, adjoint: &Tensor) -> CausalLmResult<()> {
        let (_, w) = source.shape();
        if source.shape() != adjoint.shape()
            || self.left.len() != w
            || self.right.len() != w
            || self.value.is_some()
        {
            return Err("invalid rank-one projection capture".into());
        }
        // u^T (X^T dZ) v = sum_t (x_t . u) (dZ_t . v); no dense parameter VJP.
        let mut sum = sum::Sum::default();
        for (x, d) in source
            .data()
            .chunks_exact(w)
            .zip(adjoint.data().chunks_exact(w))
        {
            sum.add(dot(x, self.left) * dot(d, self.right));
        }
        if !sum.value().is_finite() {
            return Err("nonfinite scalar adjoint".into());
        }
        self.value = Some(sum.value());
        Ok(())
    }
}

impl NativeCausalLmScalarOutput {
    pub fn new(
        source: &NativeCausalLm,
        layer: usize,
        left: &[f64],
        right: &[f64],
        coefficient: f64,
    ) -> CausalLmResult<Self> {
        let w = source.config.width;
        product(
            &[w, w],
            MAX_MATRIX.min(source.limits.max_parameter_elements),
            "scalar output matrix",
        )?;
        let valid = |v: &[f64]| {
            v.len() == w
                && v.iter().all(|x| x.is_finite())
                && (v.iter().fold(0.0_f64, |a, x| a.hypot(*x)) - 1.).abs() <= 1e-6
        };
        if layer >= source.config.layers || !valid(left) || !valid(right) {
            return Err("scalar output requires a valid layer and two unit width-vectors".into());
        }
        let mut state = Self {
            source: Arc::new(source.clone()),
            current: source.clone(),
            layer,
            left: left.into(),
            right: right.into(),
            coefficient: 0.,
            steps: 0,
        };
        state.set_coefficient(coefficient)?;
        Ok(state)
    }

    fn set_coefficient(&mut self, c: f64) -> CausalLmResult<()> {
        if !c.is_finite() {
            return Err("nonfinite scalar coefficient".into());
        }
        if c == self.coefficient {
            return Ok(());
        }
        let mut next = (*self.source).clone();
        if c != 0. {
            let out = &mut next.blocks[self.layer].output;
            let w = self.left.len();
            if out.weight.layout() != PackedLayout::ColMajor
                || (out.weight.inner(), out.weight.cols()) != (w, w)
            {
                return Err("unexpected packed output layout".into());
            }
            let mut transposed = out.weight.as_slice().to_vec();
            for (row, &v) in transposed.chunks_exact_mut(w).zip(self.right.iter()) {
                for (x, &u) in row.iter_mut().zip(self.left.iter()) {
                    let delta = (c * (u * v)) as f32;
                    if delta != 0. {
                        *x += delta;
                    }
                    if !x.is_finite() {
                        return Err("nonfinite merged output weight".into());
                    }
                }
            }
            out.weight = PackedB::from_tensor_transpose(
                &Tensor::from_vec(w, w, transposed)?,
                out.weight.tile(),
            )?;
            out.backward_weight = Default::default();
        }
        self.current = next;
        self.coefficient = c;
        Ok(())
    }

    pub fn coefficient(&self) -> f64 {
        self.coefficient
    }
    pub fn steps(&self) -> u64 {
        self.steps
    }
    pub fn prompt_count(&self, n: usize) -> CausalLmResult<usize> {
        self.current.input_count([1, n], true)
    }
    pub fn pair_prompt_count(
        &self,
        n: usize,
        target: usize,
        alternative: usize,
    ) -> CausalLmResult<usize> {
        self.current
            .next_token_contrast_input_count(n, target, alternative)
    }
    pub fn last_logits(&self, prompt: &[usize]) -> CausalLmResult<Vec<f32>> {
        Ok(self
            .current
            .logits(prompt, [1, prompt.len()], true)?
            .data()
            .to_vec())
    }
    pub fn contrast_slope(
        &self,
        prompt: &[usize],
        target: usize,
        alternative: usize,
    ) -> CausalLmResult<(CausalLmTokenContrast, f64)> {
        let spec = CausalLmProjectionSpec {
            layer: self.layer,
            site: CausalLmProjectionSite::Output,
            // Admission accounts for O(steps * width) terms, the same size as a diagonal capture.
            basis: CausalLmProjectionBasis::Diagonal,
        };
        let mut along = Along {
            left: &self.left,
            right: &self.right,
            value: None,
        };
        let scores = self.current.capture_projection_contrast(
            prompt,
            target,
            alternative,
            spec,
            &mut along,
        )?;
        Ok((
            scores,
            along.value.ok_or("scalar adjoint did not reach output")?,
        ))
    }

    fn admit_pairs(
        &self,
        positive: &[Vec<usize>],
        negative: &[Vec<usize>],
        target: usize,
        alternative: usize,
    ) -> CausalLmResult<()> {
        if positive.is_empty()
            || positive.len() != negative.len()
            || positive.len() > MAX_SCALAR_OUTPUT_PAIRS
        {
            return Err("invalid scalar pair batch".into());
        }
        for ids in positive.iter().chain(negative) {
            self.current
                .next_token_contrast_input_count(ids.len(), target, alternative)?;
            if ids.iter().any(|i| *i >= self.current.config.vocabulary) {
                return Err("invalid scalar prompt ID".into());
            }
        }
        Ok(())
    }

    fn batch_gradient(
        &self,
        positive: &[Vec<usize>],
        negative: &[Vec<usize>],
        target: usize,
        alternative: usize,
    ) -> CausalLmResult<PairGradient> {
        self.admit_pairs(positive, negative, target, alternative)?;
        let mut loss = sum::Sum::default();
        let mut gradient = sum::Sum::default();
        let mut pairs = Vec::with_capacity(positive.len());
        let (prompt_evaluations, observations) = observe_pairs(positive, negative, |ids| {
            let (r, d) = self.contrast_slope(ids, target, alternative)?;
            Ok((r.target_minus_alternative, d))
        })?;
        for [(x, da), (y, db)] in observations {
            let (l, slope) = logistic(x - y);
            loss.add(l);
            gradient.add(slope * (da - db));
            pairs.push([x, y, da, db]);
        }
        let gradient = gradient.mean(pairs.len() as u64);
        let loss = loss.mean(pairs.len() as u64);
        if !loss.is_finite() || !gradient.is_finite() {
            return Err("nonfinite scalar objective or derivative".into());
        }
        Ok(PairGradient {
            prompt_evaluations,
            loss,
            gradient,
            pairs,
        })
    }

    pub fn validate_sgd(&self, rate: f64, max_step: f64, max_abs: f64) -> CausalLmResult<()> {
        if [rate, max_step, max_abs]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.)
            || self.coefficient.abs() > max_abs
        {
            return Err("invalid scalar SGD bounds".into());
        }
        Ok(())
    }

    fn proposal(
        &self,
        gradient: f64,
        rate: f64,
        max_step: f64,
        max_abs: f64,
    ) -> CausalLmResult<f64> {
        let update = -rate * gradient;
        if !update.is_finite() {
            return Err("nonfinite scalar update".into());
        }
        Ok((self.coefficient + update.clamp(-max_step, max_step)).clamp(-max_abs, max_abs))
    }

    fn advance(&mut self) -> CausalLmResult<()> {
        self.steps = self
            .steps
            .checked_add(1)
            .ok_or("scalar step counter overflow")?;
        Ok(())
    }

    /// Mean softplus(-(contrast(positive)-contrast(negative))); plain clipped scalar SGD.
    /// The derivative is evaluated at the actual rounded merged weights, not at step zero.
    #[allow(clippy::too_many_arguments)]
    pub fn step_pairs(
        &self,
        positive: &[Vec<usize>],
        negative: &[Vec<usize>],
        target: usize,
        alternative: usize,
        rate: f64,
        max_step: f64,
        max_abs: f64,
    ) -> CausalLmResult<(Self, CausalLmScalarPairStep)> {
        self.validate_sgd(rate, max_step, max_abs)?;
        let PairGradient {
            prompt_evaluations,
            loss,
            gradient,
            pairs,
        } = self.batch_gradient(positive, negative, target, alternative)?;
        let after = self.proposal(gradient, rate, max_step, max_abs)?;
        let mut next = self.clone();
        next.set_coefficient(after)?;
        next.advance()?;
        let info = CausalLmScalarPairStep {
            gradient_prompt_evaluations: prompt_evaluations,
            step: next.steps,
            before: self.coefficient,
            after: next.coefficient,
            gradient,
            loss,
            pairs,
        };
        Ok((next, info))
    }

    pub fn checkpoint_json(&self) -> CausalLmResult<String> {
        let m = &self.source;
        Ok(serde_json::to_string(&Checkpoint {
            schema: SCHEMA.into(),
            base: m.base_id.clone(),
            lora: m.lora_artifact_id().map(str::to_owned),
            geometry: m.geometry_artifact_id().map(str::to_owned),
            readout: m.readout_accumulator,
            layer: self.layer,
            steps: self.steps,
            coefficient_bits: self.coefficient.to_bits(),
            left_bits: self.left.iter().map(|v| v.to_bits()).collect(),
            right_bits: self.right.iter().map(|v| v.to_bits()).collect(),
        })?)
    }
    pub fn restore(
        source: &NativeCausalLm,
        payload: &str,
        expected_sha256: &str,
    ) -> CausalLmResult<Self> {
        if payload.len() > MAX_JSON
            || expected_sha256.len() != 64
            || format!("{:x}", Sha256::digest(payload.as_bytes())) != expected_sha256
        {
            return Err("scalar checkpoint length or digest mismatch".into());
        }
        let r: Checkpoint = serde_json::from_str(payload)?;
        if r.schema != SCHEMA
            || r.base != source.base_id
            || r.lora.as_deref() != source.lora_artifact_id()
            || r.geometry.as_deref() != source.geometry_artifact_id()
            || r.readout != source.readout_accumulator
        {
            return Err("scalar checkpoint belongs to a different decoder state".into());
        }
        let decode = |v: Vec<u64>| v.into_iter().map(f64::from_bits).collect::<Vec<_>>();
        let mut s = Self::new(
            source,
            r.layer,
            &decode(r.left_bits),
            &decode(r.right_bits),
            f64::from_bits(r.coefficient_bits),
        )?;
        s.steps = r.steps;
        Ok(s)
    }
}
