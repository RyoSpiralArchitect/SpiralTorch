//! Prepare explicit initial weights once; verify, never refit, during learning.
use super::*;
use st_kernel_contracts::causal_bias::{CausalBiasMoments, CausalBiasScaleMatch};
use std::collections::BTreeMap;

const TOLERANCE: f64 = 1e-5;
const METRICS: [ByteDecoderPairMetric; 2] = [
    ByteDecoderPairMetric::EuclideanChordSquared,
    ByteDecoderPairMetric::CategoricalFisherRaoSquared,
];

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CalibrationSpec {
    source_request_sha256: String,
    pub(super) train_batch_indices: Vec<usize>,
    relative_tolerance: f64,
}

pub(super) fn explicit_spec<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<CalibrationSpec>, D::Error> {
    CalibrationSpec::deserialize(deserializer).map(Some)
}

fn check_selection(indices: &[usize], total: usize) -> Result<()> {
    if !(1..=16).contains(&indices.len())
        || indices.iter().any(|&i| i >= total)
        || indices.iter().collect::<BTreeSet<_>>().len() != indices.len()
    {
        return Err("calibration needs 1..16 distinct preselected training batch indices".into());
    }
    Ok(())
}

pub(super) fn validate_spec(r: &Request) -> Result<()> {
    if r.calibration_controls() != r.bias_calibration.is_some() {
        return Err("v4/v5 require a prepared bias calibration specification".into());
    }
    if let Some(spec) = &r.bias_calibration {
        if spec.relative_tolerance != TOLERANCE
            || spec.source_request_sha256.len() != 64
            || !spec
                .source_request_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("invalid calibration source identity or frozen RMS gate".into());
        }
        check_selection(&spec.train_batch_indices, r.train_batches.len())?;
    }
    Ok(())
}

fn case_for(r: &Request, seed: u64, metric: ByteDecoderPairMetric, matched: bool) -> Result<&Case> {
    r.cases
        .iter()
        .find(|c| {
            c.seed == seed
                && c.geometry
                && !c.frozen()
                && c.pair_metric == Some(metric)
                && c.matched() == matched
        })
        .ok_or_else(|| "missing calibration control case".into())
}

fn candidate_plan(
    r: &Request,
    seed: u64,
    metric: ByteDecoderPairMetric,
) -> Result<ByteDecoderPlan> {
    let mut case = case_for(r, seed, ByteDecoderPairMetric::EuclideanChordSquared, false)?.clone();
    case.pair_metric = Some(metric);
    plan(&r.config, &case)
}

pub(super) async fn measure(
    r: &Request,
    plan: &ByteDecoderPlan,
    runtime: &WgpuRuntime,
    indices: &[usize],
) -> Result<Vec<CausalBiasMoments>> {
    let mut model = plan.compile_training_wgpu(runtime.clone())?;
    let captured = model.checkpoint_snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let initial = captured.read()?;
    #[cfg(target_arch = "wasm32")]
    let initial = captured.read_async().await?;
    if initial.to_json()? != plan.initial_checkpoint().to_json()? {
        return Err("calibration device initialization differs from the frozen plan".into());
    }
    let documents: Vec<_> = r.train_documents.iter().map(Vec::as_slice).collect();
    let mut moments: Vec<CausalBiasMoments> = Vec::new();
    for &index in indices {
        let host =
            ByteLmBatch::from_documents(&documents, &r.train_batches[index], r.config.steps)?;
        let batch = model.prepare_batch(&host)?;
        let tape = model.forward(&batch)?;
        if tape.parameter_revision() != 0 {
            return Err("calibration consumed an update".into());
        }
        for block in 0..plan.block_count() {
            let scores = tape
                .geometry_pair_bias(block)
                .ok_or("missing calibration scores")?;
            let captured = scores.snapshot()?;
            #[cfg(not(target_arch = "wasm32"))]
            let values = captured.read()?;
            #[cfg(target_arch = "wasm32")]
            let values = captured.read_async().await?;
            let observed =
                CausalBiasMoments::from_scores(scores.layout().shape().try_into()?, &values)?;
            if let Some(total) = moments.get_mut(block) {
                total.merge(&observed)?;
            } else {
                moments.push(observed);
            }
        }
    }
    Ok(moments)
}

pub(super) fn checked_errors(
    reference: &[CausalBiasMoments],
    candidate: &[CausalBiasMoments],
) -> Result<Vec<Vec<f64>>> {
    if reference.len() != candidate.len() || reference.is_empty() {
        return Err("calibration block coverage differs".into());
    }
    reference.iter().zip(candidate).map(|(a, b)| {
        if (a.heads(), a.steps(), a.valid_pairs_per_head()) != (b.heads(), b.steps(), b.valid_pairs_per_head()) {
            return Err("calibration coverage differs".into());
        }
        a.rms().iter().zip(b.rms()).map(|(target, actual)| {
            let error = (actual / target - 1.).abs();
            if *target <= 0. || actual <= 0. || !error.is_finite() || error > TOLERANCE {
                return Err("initial bias RMS gate failed; fitting or tolerance retuning is not allowed during learning".into());
            }
            Ok(error)
        }).collect()
    }).collect()
}

/// Verify the frozen initial models even when resuming on a different device.
/// This allocates temporary revision-zero models; it never touches resumed weights.
pub(super) async fn verify_initial(r: &Request, runtime: &WgpuRuntime) -> Result<()> {
    let Some(spec) = &r.bias_calibration else {
        return Ok(());
    };
    for seed in r.cases.iter().map(|c| c.seed).collect::<BTreeSet<_>>() {
        let reference = plan(
            &r.config,
            case_for(r, seed, ByteDecoderPairMetric::PoincareSquared, false)?,
        )?;
        let target = measure(r, &reference, runtime, &spec.train_batch_indices).await?;
        for metric in &METRICS[..if r.fisher_controls() { 2 } else { 1 }] {
            let fitted = plan(&r.config, case_for(r, seed, *metric, true)?)?;
            let actual = measure(r, &fitted, runtime, &spec.train_batch_indices).await?;
            checked_errors(&target, &actual)?;
        }
        for case in r.cases.iter().filter(|c| {
            c.seed == seed
                && !c.frozen()
                && (c.head_pair_metrics.is_some() || (c.mixture.is_some() && !c.router_frozen()))
        }) {
            let actual = measure(
                r,
                &plan(&r.config, case)?,
                runtime,
                &spec.train_batch_indices,
            )
            .await?;
            checked_errors(&target, &actual)?;
        }
    }
    Ok(())
}

/// A bounded CPU-preflighted v3 source and training-only calibration selection.
/// The v4/v5 request contains fitted weights, shared unchanged by all learners.
pub struct ByteCorpusBiasCalibration {
    source: ByteCorpusStudy,
    input: Vec<u8>,
    spec: CalibrationSpec,
    fisher: bool,
}

/// Keep the request's original JSON bytes when saving, learning or resuming.
pub struct ByteCorpusPreparedBiasStudy {
    pub request_json: String,
    pub report: Value,
}

impl ByteCorpusBiasCalibration {
    pub fn from_json(input: &[u8], train_batch_indices: &[usize]) -> Result<Self> {
        let source = ByteCorpusStudy::from_json(input)?;
        if source.request.schema != "spiraltorch.byte_corpus.request.v3" {
            return Err("bias preparation requires an unchanged five-arm v3 source".into());
        }
        check_selection(train_batch_indices, source.total_updates())?;
        let spec = CalibrationSpec {
            source_request_sha256: source.request_sha256.clone(),
            train_batch_indices: train_batch_indices.to_vec(),
            relative_tolerance: TOLERANCE,
        };
        let this = Self {
            source,
            input: input.to_vec(),
            spec,
            fisher: false,
        };
        // Admit the expanded recipe and its worst-case checkpoint before a GPU
        // is requested. Placeholder gains are original values, not a fitted claim.
        this.materialize(&BTreeMap::new())?;
        Ok(this)
    }

    /// Extend the unchanged v3 source to nine v5 arms, matching both flat and
    /// categorical Fisher-Rao to the original Poincare training-window RMS.
    pub fn with_fisher_rao_control(mut self) -> Result<Self> {
        self.fisher = true;
        self.materialize(&BTreeMap::new())?;
        Ok(self)
    }

    fn materialize(&self, fitted: &BTreeMap<(u64, usize), Vec<Vec<f32>>>) -> Result<String> {
        let mut value: Value = serde_json::from_slice(&self.input)?;
        value["schema"] = json!(if self.fisher {
            "spiraltorch.byte_corpus.request.v5"
        } else {
            "spiraltorch.byte_corpus.request.v4"
        });
        value["bias_calibration"] = json!(self.spec);
        let cases = value["cases"]
            .as_array_mut()
            .ok_or("missing source cases")?;
        for c in cases.iter_mut() {
            c["bias_initialization"] = json!(BiasInitialization::Original);
        }
        let mut extras = Vec::new();
        for (metric_index, metric) in
            METRICS
                .iter()
                .enumerate()
                .take(if self.fisher { 2 } else { 1 })
        {
            for c in cases
                .iter()
                .filter(|c| c["pair_metric"] == "euclidean_chord_squared.v1")
            {
                let mut extra = c.clone();
                extra["name"] = json!(format!(
                    "{}{}_rms_matched",
                    c["name"].as_str().ok_or("case name")?,
                    if metric_index == 0 { "" } else { "_fisher_rao" }
                ));
                extra["pair_metric"] = json!(metric);
                extra["bias_initialization"] = json!(BiasInitialization::MatchedPoincareRms);
                if let Some(gains) = fitted.get(&(c["seed"].as_u64().ok_or("seed")?, metric_index))
                {
                    for (block, values) in gains.iter().enumerate() {
                        extra["parameters"][6 + block]["values"] = json!(values);
                    }
                }
                extras.push(extra);
            }
        }
        cases.extend(extras);
        let text = value.to_string();
        let prepared = ByteCorpusStudy::from_json(text.as_bytes())?;
        prepared.validate_segment(None, 0)?;
        for (old, new) in self
            .source
            .request
            .cases
            .iter()
            .zip(&prepared.request.cases)
        {
            if old.name != new.name
                || old.parameters.len() != new.parameters.len()
                || old.parameters.iter().zip(&new.parameters).any(|(a, b)| {
                    a.name != b.name
                        || a.shape != b.shape
                        || !policy::same_bits(&a.values, &b.values)
                })
            {
                return Err("JSON materialization changed a source model's initial bits".into());
            }
        }
        Ok(text)
    }

    pub async fn prepare(&self, runtime: WgpuRuntime) -> Result<ByteCorpusPreparedBiasStudy> {
        if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
            return Err("real GPU required for resident calibration".into());
        }
        let r = &self.source.request;
        let mut gains = BTreeMap::new();
        let mut reports = Vec::new();
        for seed in r.cases.iter().map(|c| c.seed).collect::<BTreeSet<_>>() {
            let reference = plan(
                &r.config,
                case_for(r, seed, ByteDecoderPairMetric::PoincareSquared, false)?,
            )?;
            let target = measure(r, &reference, &runtime, &self.spec.train_batch_indices).await?;
            for (metric_index, metric) in
                METRICS
                    .iter()
                    .enumerate()
                    .take(if self.fisher { 2 } else { 1 })
            {
                let candidate = candidate_plan(r, seed, *metric)?;
                let before =
                    measure(r, &candidate, &runtime, &self.spec.train_batch_indices).await?;
                let fits = target
                    .iter()
                    .zip(&before)
                    .zip(candidate.causal_geometry().ok_or("geometry")?.raw_gains())
                    .map(|((a, b), g)| CausalBiasScaleMatch::new(a, b, g, TOLERANCE))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let raw: Vec<_> = fits.iter().map(|f| f.raw_gains().to_vec()).collect();
                let fitted = candidate.clone().with_initial_geometry_raw_gains(&raw)?;
                let actual = measure(r, &fitted, &runtime, &self.spec.train_batch_indices).await?;
                let errors = checked_errors(&target, &actual)?;
                reports.push(
                    json!({"seed":seed, "valid_pairs_per_head":target[0].valid_pairs_per_head(),
                "reference_rms":target.iter().map(CausalBiasMoments::rms).collect::<Vec<_>>(),
                "candidate_rms":before.iter().map(CausalBiasMoments::rms).collect::<Vec<_>>(),
                "fitted_rms":actual.iter().map(CausalBiasMoments::rms).collect::<Vec<_>>(),
                "raw_gains":raw, "realized_relative_errors":errors,
                "before_checkpoint_json":candidate.initial_checkpoint().to_json()?,
                "fitted_checkpoint_json":fitted.initial_checkpoint().to_json()?,
                "no_updates_consumed":true}),
                );
                if self.fisher {
                    reports.last_mut().unwrap()["pair_metric"] = json!(metric);
                }
                gains.insert((seed, metric_index), raw);
            }
        }
        let request_json = self.materialize(&gains)?;
        let prepared = ByteCorpusStudy::from_json(request_json.as_bytes())?;
        for record in &reports {
            let case = case_for(
                &prepared.request,
                record["seed"].as_u64().unwrap(),
                if self.fisher {
                    serde_json::from_value(record["pair_metric"].clone())?
                } else {
                    ByteDecoderPairMetric::EuclideanChordSquared
                },
                true,
            )?;
            if plan(&prepared.request.config, case)?
                .initial_checkpoint()
                .to_json()?
                != record["fitted_checkpoint_json"].as_str().unwrap()
            {
                return Err("materialized request differs from the fitted initial model".into());
            }
        }
        Ok(ByteCorpusPreparedBiasStudy {
            request_json,
            report: json!({
            "schema":if self.fisher { "spiraltorch.byte_corpus.bias_preparation.v2" } else { "spiraltorch.byte_corpus.bias_preparation.v1" }, "bias_calibration":self.spec,
            "request_sha256":prepared.request_sha256, "adapter":format!("{:?}",runtime.adapter_info()),
            "cases":reports, "scope":"One-time initialization; all later learners consume the same frozen request; not quality, speed or matched dynamics"}),
        })
    }
}

#[cfg(test)]
mod tests;
