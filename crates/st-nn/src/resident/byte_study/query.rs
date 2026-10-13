//! Context-conditioned mixtures with one trainable shared intercept, not two.
use super::*;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct QueryProjectionSpec {
    pub(super) weight_update: GeometryUpdate,
    bias: FrozenZero,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FrozenZero {
    FrozenZero,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct QueryControlsSpec {
    source_request_sha256: String,
}

pub(super) fn explicit_projection<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<QueryProjectionSpec>, D::Error> {
    QueryProjectionSpec::deserialize(d).map(Some)
}

pub(super) fn explicit_spec<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<QueryControlsSpec>, D::Error> {
    QueryControlsSpec::deserialize(d).map(Some)
}

pub(super) fn validate_spec(r: &Request) -> Result<()> {
    if r.query_controls() != r.query_controls.is_some()
        || (!r.query_controls()
            && r.cases.iter().any(|c| {
                c.mixture
                    .as_ref()
                    .is_some_and(|m| m.query_projection.is_some())
            }))
    {
        return Err("contextual projection controls belong to v8 only".into());
    }
    if let Some(spec) = &r.query_controls {
        if spec.source_request_sha256.len() != 64
            || !spec
                .source_request_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("contextual controls require a canonical v7 source hash".into());
        }
    }
    Ok(())
}

pub(super) fn validate_modes(cases: &[&Case], config: &Config) -> Result<()> {
    for source in cases.iter().filter(|c| {
        c.mixture.as_ref().is_some_and(|m| {
            m.query_projection.is_none() && m.router_update == GeometryUpdate::Train
        })
    }) {
        let m = source.mixture.as_ref().unwrap();
        for update in [GeometryUpdate::Train, GeometryUpdate::Frozen] {
            let found: Vec<_> = cases
                .iter()
                .filter(|c| {
                    c.mixture.as_ref().is_some_and(|q| {
                        q.metrics == m.metrics
                            && q.query_projection
                                .as_ref()
                                .is_some_and(|q| q.weight_update == update)
                    })
                })
                .collect();
            if found.len() != 1 {
                return Err("missing or repeated contextual projection control".into());
            }
            let c = found[0];
            let q = c.mixture.as_ref().unwrap();
            if !c.geometry
                || c.geometry_update != Some(GeometryUpdate::Train)
                || c.bias_initialization != source.bias_initialization
                || c.pair_metric.is_some()
                || c.head_pair_metrics.is_some()
                || q.router_update != GeometryUpdate::Train
                || q.scales.len() != m.scales.len()
                || q.scales
                    .iter()
                    .zip(&m.scales)
                    .any(|(a, b)| !policy::same_bits(a, b))
            {
                return Err("contextual policy or inherited mixture units differ".into());
            }
            let filtered: Vec<_> = c
                .parameters
                .iter()
                .filter(|p| !p.name.starts_with("geometry.query_projection."))
                .collect();
            if filtered.len() != source.parameters.len()
                || filtered.iter().zip(&source.parameters).any(|(a, b)| {
                    a.name != b.name
                        || a.shape != b.shape
                        || !policy::same_bits(&a.values, &b.values)
                })
            {
                return Err(
                    "contextual arm must preserve every shared source parameter bit".into(),
                );
            }
            let projections: Vec<_> = c
                .parameters
                .iter()
                .filter(|p| p.name.starts_with("geometry.query_projection."))
                .collect();
            let output = config.heads * m.metrics.len();
            if projections.len() != 2
                || projections[0].name != "geometry.query_projection.0.weight"
                || projections[0].shape != [config.geometry_cols, output]
                || projections[1].name != "geometry.query_projection.0.bias"
                || projections[1].shape != [output]
                || projections
                    .iter()
                    .any(|p| p.values.iter().any(|v| v.to_bits() != 0))
            {
                return Err(
                    "contextual projections must start at canonical zero weight/bias".into(),
                );
            }
        }
    }
    Ok(())
}

pub(super) fn frozen_projections_match(
    case: &Case,
    actual: &ByteDecoderCheckpoint,
    initial: &ByteDecoderCheckpoint,
) -> Result<bool> {
    let Some(spec) = case
        .mixture
        .as_ref()
        .and_then(|m| m.query_projection.as_ref())
    else {
        return Ok(true);
    };
    let projections = |c: &ByteDecoderCheckpoint| -> Result<Vec<_>> {
        c.plan()
            .causal_geometry()
            .and_then(|g| g.mixture())
            .ok_or("missing contextual checkpoint")?
            .query_projections()
            .iter()
            .map(|p| Ok(p.graph_definition()?))
            .collect()
    };
    let (a, b) = (projections(actual)?, projections(initial)?);
    if a.len() != b.len() || a.is_empty() {
        return Ok(false);
    }
    for (a, b) in a.iter().zip(&b) {
        let (a, b) = (a.parameters(), b.parameters());
        if a.len() != 2 || b.len() != 2 {
            return Ok(false);
        }
        for slot in 0..2 {
            if (slot == 1 || spec.weight_update == GeometryUpdate::Frozen)
                && !policy::same_bits(&a[slot].values, &b[slot].values)
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Extend a complete v7 request without fitting, drawing randomness or consuming
/// data. Existing seventeen arms, gains, units and data order remain unchanged.
pub struct ByteCorpusQueryMixtureControls {
    prepared: ByteCorpusPreparedBiasStudy,
}

impl ByteCorpusQueryMixtureControls {
    pub fn from_json(input: &[u8]) -> Result<Self> {
        let source = ByteCorpusStudy::from_json(input)?;
        if source.request.schema != "spiraltorch.byte_corpus.request.v7" {
            return Err("contextual preparation requires a complete v7 source".into());
        }
        let mut value: Value = serde_json::from_slice(input)?;
        value["schema"] = json!("spiraltorch.byte_corpus.request.v8");
        value["query_controls"] = json!({"source_request_sha256":source.request_sha256});
        let templates = value["cases"].as_array().ok_or("missing cases")?.clone();
        for template in templates
            .iter()
            .filter(|c| c["mixture"]["router_update"] == "train")
        {
            let output = source.request.config.heads
                * template["mixture"]["metrics"]
                    .as_array()
                    .ok_or("missing metrics")?
                    .len();
            for update in [GeometryUpdate::Train, GeometryUpdate::Frozen] {
                let mut c = template.clone();
                c["name"] = json!(format!(
                    "{}-query-{}",
                    template["name"].as_str().ok_or("missing name")?,
                    if update == GeometryUpdate::Train {
                        "train"
                    } else {
                        "frozen"
                    }
                ));
                c["mixture"]["query_projection"] = json!(QueryProjectionSpec {
                    weight_update: update,
                    bias: FrozenZero::FrozenZero,
                });
                let parameters = c["parameters"].as_array_mut().ok_or("missing parameters")?;
                let index = parameters
                    .iter()
                    .position(|p| p["name"] == "geometry.mixture_logits.0")
                    .ok_or("missing shared router")?
                    + 1;
                parameters.insert(
                    index,
                    json!({"name":"geometry.query_projection.0.weight",
                    "shape":[source.request.config.geometry_cols,output],
                    "values":vec![0f32;source.request.config.geometry_cols * output]}),
                );
                parameters.insert(
                    index + 1,
                    json!({"name":"geometry.query_projection.0.bias",
                    "shape":[output],"values":vec![0f32;output]}),
                );
                value["cases"].as_array_mut().unwrap().push(c);
            }
        }
        let request_json = value.to_string();
        let study = ByteCorpusStudy::from_json(request_json.as_bytes())?;
        study.validate_segment(None, 0)?;
        Ok(Self {
            prepared: ByteCorpusPreparedBiasStudy {
                request_json,
                report: json!({"schema":"spiraltorch.byte_corpus.query_preparation.v1",
                "source_request_sha256":source.request_sha256,"request_sha256":study.request_sha256,
                "no_fitting":true,"no_updates_consumed":true,
                "scope":"Zero-start contextual projections with frozen zero bias; not equal parameter count or compute."}),
            },
        })
    }

    /// CPU-only structural extension; learning validates inherited initial RMS
    /// on its actual execution device, without recalibration.
    pub fn prepare(self) -> ByteCorpusPreparedBiasStudy {
        self.prepared
    }
}

#[cfg(test)]
mod tests;
