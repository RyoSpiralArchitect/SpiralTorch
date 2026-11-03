//! Trace one uniquely occurring input byte through a real, unclipped SGD step.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ByteEmbeddingSelection {
    pub request_sha256: String,
    pub case: String,
    /// One-based update in the original uninterrupted study.
    pub update: usize,
    pub token: u8,
    pub feature: usize,
    /// Zero-based [batch row, input time], excluding shifted targets.
    pub input_position: [usize; 2],
    pub rate_f32_bits: u32,
}

/// A CPU-validated selection bound to its original study (not a replay recipe).
pub struct ByteEmbeddingProbe<'a> {
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
    study: &'a ByteCorpusStudy,
    selection: ByteEmbeddingSelection,
}

impl ByteCorpusStudy {
    pub fn embedding_update_probe(
        &self,
        case: &str,
        update: usize,
        token: u8,
        feature: usize,
    ) -> Result<ByteEmbeddingProbe<'_>> {
        let r = &self.request;
        if !r.cases.iter().any(|c| c.name == case) {
            return Err("embedding probe case does not belong to this study".into());
        }
        if update == 0 || update > self.total_updates() || feature >= r.config.width {
            return Err("embedding probe update or feature outside this study".into());
        }
        if r.optimizer().config()?.is_some() || r.gradient_clip_recipe().clip()?.is_some() {
            return Err("embedding probe requires unclipped stateless SGD".into());
        }
        let documents: Vec<_> = r.train_documents.iter().map(Vec::as_slice).collect();
        let batch =
            ByteLmBatch::from_documents(&documents, &r.train_batches[update - 1], r.config.steps)?;
        let mut slots = batch
            .input_bytes()
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == token);
        let slot = match (slots.next(), slots.next()) {
            (Some((slot, _)), None) => slot,
            _ => {
                return Err(
                    "embedding probe token must occur exactly once in the selected input batch"
                        .into(),
                )
            }
        };
        Ok(ByteEmbeddingProbe {
            study: self,
            selection: ByteEmbeddingSelection {
                request_sha256: self.request_sha256().into(),
                case: case.into(),
                update,
                token,
                feature,
                input_position: [slot / r.config.steps, slot % r.config.steps],
                rate_f32_bits: r.rate.to_bits(),
            },
        })
    }
}

impl ByteEmbeddingProbe<'_> {
    pub fn selection(&self) -> &ByteEmbeddingSelection {
        &self.selection
    }

    /// Runs the normal case/update schedule once. The observer only retains
    /// views; holding their backing allocations can change GPU resource reuse.
    #[cfg(feature = "wgpu")]
    pub async fn run(self, runtime: WgpuRuntime) -> Result<ByteEmbeddingObservation> {
        let selection = self.selection;
        let mut capture = None;
        let result = self
            .study
            .execute(
                runtime,
                None,
                self.study.total_updates(),
                false,
                Some(|case: &str, update, step: SubmittedStep<'_>| {
                    if case == selection.case && update == selection.update {
                        if capture.is_some() {
                            return Err("embedding probe boundary invoked more than once".into());
                        }
                        capture = Some(ByteEmbeddingViews::retain(&selection, step));
                    }
                    Ok(())
                }),
                None::<ForwardObserverFn>,
            )
            .await;
        let mut run_outcome = json!({
            "schema": "spiraltorch.byte_embedding_update.observation.v1",
            "selection": selection,
            "capture_status": "not_reached",
            "release_ready": false
        });
        let views = match capture {
            Some(Ok(views)) => {
                run_outcome["capture_status"] = json!("views_retained");
                Some(views)
            }
            Some(Err(e)) => {
                run_outcome["capture_status"] = json!("error");
                run_outcome["capture_error"] = json!(e.to_string());
                None
            }
            None => None,
        };
        match result {
            Ok((report, _)) => {
                run_outcome["status"] = json!("completed");
                run_outcome["report"] = report;
            }
            Err(e) => {
                run_outcome["status"] = json!("error");
                run_outcome["error"] = json!(e.to_string());
            }
        }
        Ok(ByteEmbeddingObservation { run_outcome, views })
    }
}

#[cfg(feature = "wgpu")]
pub struct ByteEmbeddingObservation {
    /// Save separately: retaining views is neither update acceptance nor a successful read.
    pub run_outcome: Value,
    pub views: Option<ByteEmbeddingViews>,
}

#[cfg(feature = "wgpu")]
pub struct ByteEmbeddingViews {
    selection: ByteEmbeddingSelection,
    attempted_revisions: [u64; 2],
    scalars: [ResidentTensor; 4],
}

#[cfg(feature = "wgpu")]
impl ByteEmbeddingViews {
    fn retain(s: &ByteEmbeddingSelection, step: SubmittedStep<'_>) -> Result<Self> {
        fn scalar(t: &ResidentTensor, coordinate: &[usize]) -> Result<ResidentTensor> {
            if t.layout().shape().len() != coordinate.len() {
                return Err("unexpected embedding probe tensor rank".into());
            }
            coordinate
                .iter()
                .try_fold(t.clone(), |v, &i| Ok(v.select(0, i)?))
        }
        let before = step.tape.parameter_snapshot();
        let after = step.model.parameter_snapshot();
        let attempted_revisions = [before.revision(), after.revision()];
        if attempted_revisions != [u64::try_from(s.update - 1)?, u64::try_from(s.update)?] {
            return Err("embedding probe attempted revisions differ from selection".into());
        }
        let table = step.model.parameter_layout().token_embedding();
        let coord = [usize::from(s.token), s.feature];
        let [row, time] = s.input_position;
        Ok(Self {
            selection: s.clone(),
            attempted_revisions,
            scalars: [
                scalar(&before.values()[table], &coord)?,
                scalar(
                    step.gradients.embedding_output_gradient(),
                    &[row, time, s.feature],
                )?,
                scalar(&step.gradients.parameter_gradients()[table], &coord)?,
                scalar(&after.values()[table], &coord)?,
            ],
        })
    }

    fn snapshot(&self) -> Result<st_backend_wgpu::resident_tensor::TensorReadbackBatch> {
        Ok(self.scalars[0]
            .device()
            .snapshot_many(&self.scalars.each_ref())?)
    }

    /// One explicit guarded copy batch after the run; no shader recomputation.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_bits(self) -> Result<Value> {
        encode_bits(
            &self.selection,
            self.attempted_revisions,
            self.snapshot()?.read()?,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_bits_async(self) -> Result<Value> {
        encode_bits(
            &self.selection,
            self.attempted_revisions,
            self.snapshot()?.read_async().await?,
        )
    }
}

#[cfg(any(feature = "wgpu", test))]
fn encode_bits(
    selection: &ByteEmbeddingSelection,
    revisions: [u64; 2],
    values: Vec<Vec<f32>>,
) -> Result<Value> {
    let [before, input, table, after] =
        <[Vec<f32>; 4]>::try_from(values).map_err(|_| "expected four embedding probe scalars")?;
    let bits = |row: &[f32]| match row {
        [x] if x.is_finite() => Ok(x.to_bits()),
        _ => Err("expected one finite embedding probe scalar per stage"),
    };
    Ok(json!({
        "schema": "spiraltorch.byte_embedding_update.bits.v1",
        "selection": selection,
        "attempted_revisions": revisions,
        "f32_bits": {
            "weight_before": bits(&before)?, "embedding_output_gradient": bits(&input)?,
            "token_parameter_gradient": bits(&table)?, "weight_after": bits(&after)?,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resident::byte_study::{
        optimizer::tests::{recipe, wrap},
        tests::fixture,
    };

    fn study(request: Value) -> ByteCorpusStudy {
        ByteCorpusStudy::from_json(request.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn embedding_probe_counts_only_inputs_across_the_whole_batch() {
        let mut r = fixture();
        r["config"]["batch"] = json!(2);
        r["train_batches"] = json!([[[0, 2], [0, 0]], [[0, 0], [0, 0]]]);
        r["validation_batches"] = json!([[[0, 0], [0, 0]]]);
        let study = study(r);
        let probe = study.embedding_update_probe("geometry", 1, 2, 0).unwrap();
        let s = probe.selection();
        assert_eq!(s.request_sha256, study.request_sha256());
        assert_eq!(s.input_position, [1, 1]);
        assert_eq!(s.rate_f32_bits, 0.05f32.to_bits());
        for (case, step, token, feature) in [
            ("geometry", 1, 5, 0), // shifted terminal target, not an input
            ("geometry", 2, 2, 0), // occurs in both input windows
            ("absent", 1, 2, 0),
            ("plain", 0, 2, 0),
            ("plain", 3, 2, 0),
            ("plain", 1, 2, 2),
        ] {
            assert!(study
                .embedding_update_probe(case, step, token, feature)
                .is_err());
        }
    }

    #[test]
    fn embedding_probe_is_only_for_an_unclipped_sgd_update() {
        let mut r = wrap(fixture(), json!({"kind":"stateless_sgd.v1"}));
        assert!(study(r.clone())
            .embedding_update_probe("plain", 1, 2, 1)
            .is_ok());
        r["schema"] = json!("spiraltorch.byte_corpus.request.v10");
        r["gradient_clip"] = json!({"kind":"active_global_l2.v1","max_norm":0.1});
        for request in [r, wrap(fixture(), recipe())] {
            assert!(study(request)
                .embedding_update_probe("plain", 1, 2, 1)
                .is_err());
        }
    }

    #[test]
    fn embedding_probe_bits_preserve_stage_order_and_signed_zero() {
        let study = study(fixture());
        let probe = study.embedding_update_probe("geometry", 1, 2, 0).unwrap();
        let s = probe.selection();
        let values = [0.0, -0.0, f32::from_bits(1), 1.0]
            .map(|x| vec![x])
            .to_vec();
        let record = encode_bits(s, [0, 1], values.clone()).unwrap();
        assert_eq!(record["attempted_revisions"], json!([0, 1]));
        assert_eq!(record["selection"], json!(s));
        assert_eq!(
            record["f32_bits"],
            json!({"weight_before":0, "embedding_output_gradient":2147483648u32, "token_parameter_gradient":1, "weight_after":1065353216})
        );
        for v in [
            vec![],
            vec![vec![0.]; 3],
            vec![vec![0., 0.]; 4],
            vec![vec![f32::NAN]; 4],
        ] {
            assert!(encode_bits(s, [0, 1], v).is_err());
        }
    }
}
