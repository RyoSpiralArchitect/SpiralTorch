//! Fixed heterogeneous assignments inherit calibrated gains, never refit them.
use super::*;

const METRICS: [ByteDecoderPairMetric; 3] = [
    ByteDecoderPairMetric::PoincareSquared,
    ByteDecoderPairMetric::EuclideanChordSquared,
    ByteDecoderPairMetric::CategoricalFisherRaoSquared,
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HeadwiseSpec {
    source_request_sha256: String,
}

pub(super) fn explicit_spec<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<HeadwiseSpec>, D::Error> {
    HeadwiseSpec::deserialize(d).map(Some)
}

pub(super) fn explicit_metrics<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<Vec<Vec<ByteDecoderPairMetric>>>, D::Error> {
    Vec::<Vec<ByteDecoderPairMetric>>::deserialize(d).map(Some)
}

pub(super) fn validate_spec(r: &Request) -> Result<()> {
    if r.headwise_controls() != r.headwise_controls.is_some()
        || (!r.headwise_controls() && r.cases.iter().any(|c| c.head_pair_metrics.is_some()))
    {
        return Err("headwise study metadata belongs to v6 only".into());
    }
    if let Some(spec) = &r.headwise_controls {
        if r.config.heads != 2
            || r.config.blocks.len() != 1
            || spec.source_request_sha256.len() != 64
            || !spec
                .source_request_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("v6 requires one block, two heads and a canonical source hash".into());
        }
    }
    Ok(())
}

fn source_case<'a>(cases: &[&'a Case], metric: ByteDecoderPairMetric) -> Result<&'a Case> {
    cases
        .iter()
        .copied()
        .find(|c| {
            c.geometry
                && !c.frozen()
                && c.head_pair_metrics.is_none()
                && c.pair_metric == Some(metric)
                && c.matched() == (metric != ByteDecoderPairMetric::PoincareSquared)
        })
        .ok_or_else(|| "missing homogeneous gain source".into())
}

fn gains(case: &Case) -> Result<&[f32]> {
    case.parameters
        .iter()
        .find(|p| p.name == "geometry.raw_gain.0")
        .map(|p| p.values.as_slice())
        .filter(|v| v.len() == 2)
        .ok_or_else(|| "missing two-head gains".into())
}

pub(super) fn validate_modes(cases: &[&Case]) -> Result<()> {
    for left in METRICS {
        for right in METRICS.into_iter().filter(|&right| right != left) {
            let grid = vec![vec![left, right]];
            let expected = [
                gains(source_case(cases, left)?)?[0],
                gains(source_case(cases, right)?)?[1],
            ];
            for update in [GeometryUpdate::Train, GeometryUpdate::Frozen] {
                let matches: Vec<_> = cases
                    .iter()
                    .filter(|c| {
                        c.geometry
                            && c.geometry_update == Some(update)
                            && c.pair_metric.is_none()
                            && c.head_pair_metrics.as_ref() == Some(&grid)
                            && c.matched()
                    })
                    .collect();
                if matches.len() != 1 || !policy::same_bits(gains(matches[0])?, &expected) {
                    return Err(
                        "missing/repeated headwise arm or gains not inherited from matching heads"
                            .into(),
                    );
                }
            }
        }
    }
    Ok(())
}

/// Expand a frozen v5 study without fitting, reinitializing or changing its arms.
/// v6 is deliberately bounded to the six ordered heterogeneous two-head routes.
pub struct ByteCorpusHeadwiseControls {
    source: ByteCorpusStudy,
    input: Vec<u8>,
}

impl ByteCorpusHeadwiseControls {
    pub fn from_json(input: &[u8]) -> Result<Self> {
        let source = ByteCorpusStudy::from_json(input)?;
        if source.request.schema != "spiraltorch.byte_corpus.request.v5"
            || source.request.config.heads != 2
            || source.request.config.blocks.len() != 1
        {
            return Err("headwise preparation requires a one-block, two-head v5 source".into());
        }
        let preparation = Self {
            source,
            input: input.to_vec(),
        };
        // Adding twelve arms can exceed the resume-history budget even when
        // the source is small. Reject before a client requests its GPU runtime.
        let expanded = preparation.materialize()?;
        ByteCorpusStudy::from_json(expanded.as_bytes())?.validate_segment(None, 0)?;
        Ok(preparation)
    }

    fn materialize(&self) -> Result<String> {
        let mut value: Value = serde_json::from_slice(&self.input)?;
        value["schema"] = json!("spiraltorch.byte_corpus.request.v6");
        value["headwise_controls"] = json!({"source_request_sha256":self.source.request_sha256});
        let source_json = value["cases"]
            .as_array()
            .ok_or("missing source cases")?
            .clone();
        let r = &self.source.request;
        for seed in r.cases.iter().map(|c| c.seed).collect::<BTreeSet<_>>() {
            let cases: Vec<_> = r.cases.iter().filter(|c| c.seed == seed).collect();
            let original = source_case(&cases, ByteDecoderPairMetric::PoincareSquared)?;
            let template = source_json
                .iter()
                .find(|c| c["name"] == original.name)
                .ok_or("missing source JSON")?;
            for (left_index, left) in METRICS.into_iter().enumerate() {
                for (right_index, right) in METRICS
                    .into_iter()
                    .enumerate()
                    .filter(|(_, right)| *right != left)
                {
                    let selected = [
                        gains(source_case(&cases, left)?)?[0],
                        gains(source_case(&cases, right)?)?[1],
                    ];
                    for update in ["train", "frozen"] {
                        let mut case = template.clone();
                        case["name"] = json!(format!(
                            "headwise-{seed}-{left_index}-{right_index}-{update}"
                        ));
                        case["geometry_update"] = json!(update);
                        case.as_object_mut()
                            .ok_or("invalid case")?
                            .remove("pair_metric");
                        case["head_pair_metrics"] = json!([[left, right]]);
                        case["bias_initialization"] = json!(BiasInitialization::MatchedPoincareRms);
                        let gain = case["parameters"]
                            .as_array_mut()
                            .ok_or("missing parameters")?
                            .iter_mut()
                            .find(|p| p["name"] == "geometry.raw_gain.0")
                            .ok_or("missing gain")?;
                        gain["values"] = json!(selected);
                        value["cases"]
                            .as_array_mut()
                            .ok_or("missing cases")?
                            .push(case);
                    }
                }
            }
        }
        let output = value.to_string();
        ByteCorpusStudy::from_json(output.as_bytes())?;
        Ok(output)
    }

    /// Verify every initial routed bias against the unchanged Poincare target.
    /// These temporary revision-zero models consume no learning updates.
    #[cfg(feature = "wgpu")]
    pub async fn prepare(&self, runtime: WgpuRuntime) -> Result<ByteCorpusPreparedBiasStudy> {
        if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
            return Err("headwise preparation requires a GPU adapter".into());
        }
        let request_json = self.materialize()?;
        let prepared = ByteCorpusStudy::from_json(request_json.as_bytes())?;
        let r = &prepared.request;
        let indices = &r
            .bias_calibration
            .as_ref()
            .ok_or("missing calibration")?
            .train_batch_indices;
        let mut records = Vec::new();
        for seed in r.cases.iter().map(|c| c.seed).collect::<BTreeSet<_>>() {
            let cases: Vec<_> = r.cases.iter().filter(|c| c.seed == seed).collect();
            let target = calibration::measure(
                r,
                &plan(&r.config, source_case(&cases, METRICS[0])?)?,
                &runtime,
                indices,
            )
            .await?;
            for metric in &METRICS[1..] {
                let homogeneous = plan(&r.config, source_case(&cases, *metric)?)?;
                let actual = calibration::measure(r, &homogeneous, &runtime, indices).await?;
                calibration::checked_errors(&target, &actual)?;
            }
            for case in cases
                .iter()
                .filter(|c| c.head_pair_metrics.is_some() && !c.frozen())
            {
                let candidate = plan(&r.config, case)?;
                let actual = calibration::measure(r, &candidate, &runtime, indices).await?;
                let errors = calibration::checked_errors(&target, &actual)?;
                records.push(json!({"name":case.name,"seed":seed,"head_pair_metrics":case.head_pair_metrics,
                    "raw_gains":gains(case)?,"target_rms":target.iter().map(|m| m.rms()).collect::<Vec<_>>(),
                    "actual_rms":actual.iter().map(|m| m.rms()).collect::<Vec<_>>(),
                    "relative_errors":errors,"valid_pairs_per_head":actual.iter().map(|m| m.valid_pairs_per_head()).collect::<Vec<_>>(),
                    "model_json":candidate.initial_checkpoint().to_json()?}));
            }
        }
        let report = json!({"schema":"spiraltorch.byte_corpus.headwise_preparation.v1",
            "source_request_sha256":self.source.request_sha256,"request_sha256":prepared.request_sha256,
            "adapter":format!("{:?}",runtime.adapter_info()),"train_batch_indices":indices,
            "relative_tolerance":1e-5,"no_fitting":true,"no_updates_consumed":true,"cases":records,"passed":true});
        Ok(ByteCorpusPreparedBiasStudy {
            request_json,
            report,
        })
    }
}

#[cfg(test)]
pub(super) mod tests;
