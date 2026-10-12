//! Frozen component units, train-only strength calibration and router-only controls.
use super::*;
use st_kernel_contracts::causal_bias::CausalBiasScaleMatch;
use std::collections::BTreeMap;

use ByteDecoderPairMetric::{
    CategoricalFisherRaoSquared as F, EuclideanChordSquared as E, PoincareSquared as P,
};
const COMBINATIONS: [&[ByteDecoderPairMetric]; 4] = [&[P, E], &[P, F], &[E, F], &[P, E, F]];

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MixtureSpec {
    pub(super) metrics: Vec<ByteDecoderPairMetric>,
    pub(super) scales: Vec<Vec<f32>>,
    pub(super) router_update: GeometryUpdate,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "query::explicit_projection"
    )]
    pub(super) query_projection: Option<query::QueryProjectionSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MixtureControlsSpec {
    source_request_sha256: String,
}

pub(super) fn explicit_mixture<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<MixtureSpec>, D::Error> {
    MixtureSpec::deserialize(d).map(Some)
}
pub(super) fn explicit_spec<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<MixtureControlsSpec>, D::Error> {
    MixtureControlsSpec::deserialize(d).map(Some)
}

pub(super) fn validate_spec(r: &Request) -> Result<()> {
    if r.mixture_controls() != r.mixture_controls.is_some()
        || (!r.mixture_controls() && r.cases.iter().any(|c| c.mixture.is_some()))
    {
        return Err("mixture controls belong to v7/v8 only".into());
    }
    if let Some(spec) = &r.mixture_controls {
        if r.config.heads != 2
            || r.config.blocks.len() != 1
            || spec.source_request_sha256.len() != 64
            || !spec
                .source_request_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("v7 requires one block, two heads and a canonical v5 source hash".into());
        }
    }
    Ok(())
}

fn source<'a>(cases: &[&'a Case], metric: ByteDecoderPairMetric) -> Result<&'a Case> {
    cases
        .iter()
        .copied()
        .find(|c| {
            c.geometry
                && !c.frozen()
                && c.mixture.is_none()
                && c.pair_metric == Some(metric)
                && c.matched() == (metric != P)
        })
        .ok_or_else(|| "missing homogeneous mixture unit source".into())
}
fn gains(case: &Case) -> Result<&[f32]> {
    case.parameters
        .iter()
        .find(|p| p.name == "geometry.raw_gain.0")
        .map(|p| p.values.as_slice())
        .filter(|v| v.len() == 2)
        .ok_or_else(|| "missing two-head gains".into())
}
fn positive_gain(raw: f32) -> f64 {
    let x = f64::from(raw);
    x.max(0.) + (-x.abs()).exp().ln_1p()
}
fn scales(cases: &[&Case], metrics: &[ByteDecoderPairMetric]) -> Result<Vec<f32>> {
    let base = gains(source(cases, P)?)?;
    let mut values = Vec::new();
    for (h, &gain) in base.iter().enumerate() {
        for &metric in metrics {
            let ratio =
                (positive_gain(gains(source(cases, metric)?)?[h]) / positive_gain(gain)) as f32;
            if !ratio.is_finite() || ratio <= 0. {
                return Err("invalid inherited mixture unit scale".into());
            }
            values.push(ratio);
        }
    }
    Ok(values)
}
fn parameter_bits(a: &[Parameter], b: &[Parameter]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.name == b.name && a.shape == b.shape && policy::same_bits(&a.values, &b.values)
        })
}

pub(super) fn validate_modes(cases: &[&Case]) -> Result<()> {
    let original = source(cases, P)?;
    for metrics in COMBINATIONS {
        let expected_scales = scales(cases, metrics)?;
        let mut pair = Vec::new();
        for update in [GeometryUpdate::Train, GeometryUpdate::Frozen] {
            let found: Vec<_> = cases
                .iter()
                .copied()
                .filter(|c| {
                    c.mixture.as_ref().is_some_and(|m| {
                        m.query_projection.is_none()
                            && m.metrics == metrics
                            && m.router_update == update
                    })
                })
                .collect();
            if found.len() != 1 {
                return Err("missing or repeated mixture/router control".into());
            }
            let c = found[0];
            let m = c.mixture.as_ref().unwrap();
            if !c.geometry
                || c.geometry_update != Some(GeometryUpdate::Train)
                || !c.matched()
                || c.pair_metric.is_some()
                || c.head_pair_metrics.is_some()
                || m.scales.len() != 1
                || m.scales[0].len() != expected_scales.len()
                || m.scales[0].iter().zip(&expected_scales).any(|(&a, &e)| {
                    !a.is_finite() || a <= 0. || (f64::from(a) / f64::from(e) - 1.).abs() > 1e-6
                })
            {
                return Err("mixture policies or frozen component units differ".into());
            }
            let filtered: Vec<_> = c
                .parameters
                .iter()
                .filter(|p| !p.name.starts_with("geometry.mixture_logits."))
                .cloned()
                .collect();
            if filtered.len() != original.parameters.len()
                || filtered.iter().zip(&original.parameters).any(|(a, b)| {
                    a.name != b.name
                        || a.shape != b.shape
                        || (a.name != "geometry.raw_gain.0"
                            && !policy::same_bits(&a.values, &b.values))
                })
            {
                return Err(
                    "mixture changed source parameters outside its calibrated shared gain".into(),
                );
            }
            let routers: Vec<_> = c
                .parameters
                .iter()
                .filter(|p| p.name.starts_with("geometry.mixture_logits."))
                .collect();
            if routers.len() != 1
                || routers[0].name != "geometry.mixture_logits.0"
                || routers[0].shape != [2, metrics.len()]
                || routers[0].values.len() != 2 * metrics.len()
                || routers[0].values.iter().any(|v| v.to_bits() != 0)
            {
                return Err("mixture routers must start at canonical uniform logits".into());
            }
            pair.push(c);
        }
        if !parameter_bits(&pair[0].parameters, &pair[1].parameters)
            || pair[0]
                .mixture
                .as_ref()
                .unwrap()
                .scales
                .iter()
                .zip(&pair[1].mixture.as_ref().unwrap().scales)
                .any(|(a, b)| !policy::same_bits(a, b))
        {
            return Err(
                "learned/fixed routers must start with identical model and scale bits".into(),
            );
        }
    }
    Ok(())
}

pub(super) fn routers_match(a: &ByteDecoderCheckpoint, b: &ByteDecoderCheckpoint) -> bool {
    let routers = |c: &ByteDecoderCheckpoint| {
        c.plan()
            .causal_geometry()
            .and_then(|g| g.mixture())
            .map(|m| m.raw_logits().to_vec())
    };
    match (routers(a), routers(b)) {
        (Some(a), Some(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| policy::same_bits(a, &b))
        }
        _ => false,
    }
}

/// Add four component subsets with learned/fixed routers to an unchanged v5
/// source. Both arms train the coordinate encoder and share the full pullback.
pub struct ByteCorpusMixtureControls {
    source: ByteCorpusStudy,
    input: Vec<u8>,
}
impl ByteCorpusMixtureControls {
    pub fn from_json(input: &[u8]) -> Result<Self> {
        let source = ByteCorpusStudy::from_json(input)?;
        if source.request.schema != "spiraltorch.byte_corpus.request.v5"
            || source.request.config.heads != 2
            || source.request.config.blocks.len() != 1
        {
            return Err("mixture preparation requires a one-block/two-head v5 source".into());
        }
        let this = Self {
            source,
            input: input.to_vec(),
        };
        let provisional = this.materialize(&BTreeMap::new())?;
        ByteCorpusStudy::from_json(provisional.as_bytes())?.validate_segment(None, 0)?;
        Ok(this)
    }
    fn materialize(&self, fitted: &BTreeMap<(u64, usize), Vec<Vec<f32>>>) -> Result<String> {
        let mut value: Value = serde_json::from_slice(&self.input)?;
        value["schema"] = json!("spiraltorch.byte_corpus.request.v7");
        value["mixture_controls"] = json!({"source_request_sha256":self.source.request_sha256});
        let templates = value["cases"].as_array().ok_or("missing cases")?.clone();
        let r = &self.source.request;
        for seed in r.cases.iter().map(|c| c.seed).collect::<BTreeSet<_>>() {
            let cases: Vec<_> = r.cases.iter().filter(|c| c.seed == seed).collect();
            let original = source(&cases, P)?;
            let template = templates
                .iter()
                .find(|c| c["name"] == original.name)
                .ok_or("missing template")?;
            for (index, metrics) in COMBINATIONS.iter().enumerate() {
                let units = scales(&cases, metrics)?;
                for update in [GeometryUpdate::Train, GeometryUpdate::Frozen] {
                    let mut c = template.clone();
                    c["name"] = json!(format!(
                        "mixture-{seed}-{index}-{}",
                        if update == GeometryUpdate::Train {
                            "train"
                        } else {
                            "frozen_router"
                        }
                    ));
                    c.as_object_mut().unwrap().remove("pair_metric");
                    c["bias_initialization"] = json!(BiasInitialization::MatchedPoincareRms);
                    c["mixture"] = json!(MixtureSpec {
                        metrics: metrics.to_vec(),
                        scales: vec![units.clone()],
                        router_update: update,
                        query_projection: None
                    });
                    let parameters = c["parameters"].as_array_mut().ok_or("missing parameters")?;
                    let gain_index = parameters
                        .iter()
                        .position(|p| p["name"] == "geometry.raw_gain.0")
                        .ok_or("missing gain")?;
                    if let Some(raw) = fitted.get(&(seed, index)) {
                        parameters[gain_index]["values"] = json!(raw[0]);
                    }
                    parameters.insert(gain_index+1,json!({"name":"geometry.mixture_logits.0","shape":[2,metrics.len()],"values":vec![0f32;2*metrics.len()]}));
                    value["cases"].as_array_mut().unwrap().push(c);
                }
            }
        }
        let output = value.to_string();
        ByteCorpusStudy::from_json(output.as_bytes())?;
        Ok(output)
    }
    /// Fit only a shared initial gain per head to the original Poincare RMS on
    /// predeclared training windows. Fixed component units and uniform routers
    /// never fit held-out text. Learning/resume verifies but never refits.
    pub async fn prepare(&self, runtime: WgpuRuntime) -> Result<ByteCorpusPreparedBiasStudy> {
        if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
            return Err("mixture preparation requires a GPU".into());
        }
        calibration::verify_initial(&self.source.request, &runtime).await?;
        let provisional = self.materialize(&BTreeMap::new())?;
        let study = ByteCorpusStudy::from_json(provisional.as_bytes())?;
        let r = &study.request;
        let indices = &r
            .bias_calibration
            .as_ref()
            .ok_or("missing calibration")?
            .train_batch_indices;
        let mut fitted = BTreeMap::new();
        let mut records = Vec::new();
        for seed in r.cases.iter().map(|c| c.seed).collect::<BTreeSet<_>>() {
            let cases: Vec<_> = r.cases.iter().filter(|c| c.seed == seed).collect();
            let target =
                calibration::measure(r, &plan(&r.config, source(&cases, P)?)?, &runtime, indices)
                    .await?;
            for (index, metrics) in COMBINATIONS.iter().enumerate() {
                let case = cases
                    .iter()
                    .find(|c| {
                        c.mixture.as_ref().is_some_and(|m| m.metrics == *metrics)
                            && !c.router_frozen()
                    })
                    .ok_or("missing candidate")?;
                let candidate = plan(&r.config, case)?;
                let before = calibration::measure(r, &candidate, &runtime, indices).await?;
                let raw = target
                    .iter()
                    .zip(&before)
                    .zip(candidate.causal_geometry().unwrap().raw_gains())
                    .map(|((a, b), g)| {
                        CausalBiasScaleMatch::new(a, b, g, 1e-5).map(|fit| fit.raw_gains().to_vec())
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let adjusted = candidate.clone().with_initial_geometry_raw_gains(&raw)?;
                let after = calibration::measure(r, &adjusted, &runtime, indices).await?;
                let errors = calibration::checked_errors(&target, &after)?;
                records.push(json!({"name":case.name,"seed":seed,"mixture":case.mixture,
                    "reference_rms":target.iter().map(|x| x.rms()).collect::<Vec<_>>(),
                    "candidate_rms":before.iter().map(|x| x.rms()).collect::<Vec<_>>(),
                    "fitted_rms":after.iter().map(|x| x.rms()).collect::<Vec<_>>(),
                    "realized_relative_errors":errors,"valid_pairs_per_head":target[0].valid_pairs_per_head(),
                    "raw_gains":raw,"before_checkpoint_json":candidate.initial_checkpoint().to_json()?,
                    "fitted_checkpoint_json":adjusted.initial_checkpoint().to_json()?,"no_updates_consumed":true}));
                fitted.insert((seed, index), raw);
            }
        }
        let request_json = self.materialize(&fitted)?;
        let prepared = ByteCorpusStudy::from_json(request_json.as_bytes())?;
        for record in &records {
            let c = prepared
                .request
                .cases
                .iter()
                .find(|c| c.name == record["name"].as_str().unwrap())
                .ok_or("missing fitted case")?;
            if plan(&prepared.request.config, c)?
                .initial_checkpoint()
                .to_json()?
                != record["fitted_checkpoint_json"].as_str().unwrap()
            {
                return Err("materialized mixture differs from fitted checkpoint".into());
            }
        }
        Ok(ByteCorpusPreparedBiasStudy {
            request_json,
            report: json!({
            "schema":"spiraltorch.byte_corpus.mixture_preparation.v1","source_request_sha256":self.source.request_sha256,
            "request_sha256":prepared.request_sha256,"train_batch_indices":indices,"relative_tolerance":1e-5,
            "adapter":format!("{:?}",runtime.adapter_info()),"cases":records,"no_updates_consumed":true,
            "scope":"Training-only initial RMS matching, not matched patterns/dynamics or evidence of quality"}),
        })
    }
}

#[cfg(test)]
pub(super) mod tests;
