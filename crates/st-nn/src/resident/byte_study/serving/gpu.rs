use super::*;
use crate::resident::{ResidentByteBatch, ResidentByteInference, ResidentByteInferenceForward};
use st_kernel_contracts::causal_bias::CausalBiasMoments;
use std::sync::Arc;

#[derive(Serialize)]
struct RouterHealth {
    rows: usize,
    max_mass_error: f64,
    min_normalized_entropy: f64,
}

fn router_health(components: usize, values: &[f32]) -> Result<RouterHealth> {
    if components < 2
        || values.is_empty()
        || !values.len().is_multiple_of(components)
        || values
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
    {
        return Err("invalid observed router masses".into());
    }
    let mut health = RouterHealth {
        rows: values.len() / components,
        max_mass_error: 0.,
        min_normalized_entropy: f64::INFINITY,
    };
    for row in values.chunks_exact(components) {
        let mass: f64 = row.iter().map(|&v| f64::from(v)).sum();
        if mass <= 0. {
            return Err("observed router has zero mass".into());
        }
        let entropy = -row
            .iter()
            .filter(|&&v| v > 0.)
            .map(|&v| {
                let p = f64::from(v) / mass;
                p * p.ln()
            })
            .sum::<f64>()
            / (components as f64).ln();
        health.max_mass_error = health.max_mass_error.max((mass - 1.).abs());
        health.min_normalized_entropy = health.min_normalized_entropy.min(entropy);
    }
    Ok(health)
}

#[derive(Serialize)]
struct Observation {
    artifact_id: String,
    revision: u64,
    batch_ce: Vec<f64>,
    mean_ce: f64,
    target_bytes: usize,
    logits_f32le_sha256: Vec<String>,
    geometry_centered_rms: Vec<Vec<f64>>,
    router: Vec<RouterHealth>,
}

impl Observation {
    fn assess(&self, reference: &Self, limits: &Limits) -> (Vec<&'static str>, f64, f64) {
        let mut failures = self.failures(limits);
        let mean_delta = self.mean_ce - reference.mean_ce;
        let max_batch_delta = self
            .batch_ce
            .iter()
            .zip(&reference.batch_ce)
            .map(|(a, b)| a - b)
            .fold(f64::NEG_INFINITY, f64::max);
        if mean_delta > limits.max_mean_ce_regression {
            failures.push("mean_ce_regression");
        }
        if max_batch_delta > limits.max_batch_ce_regression {
            failures.push("batch_ce_regression");
        }
        (failures, mean_delta, max_batch_delta)
    }

    fn failures(&self, limits: &Limits) -> Vec<&'static str> {
        let mut failures = Vec::new();
        if self.mean_ce > limits.max_mean_ce {
            failures.push("mean_ce");
        }
        if self
            .geometry_centered_rms
            .iter()
            .flatten()
            .any(|v| *v > limits.max_geometry_centered_rms)
        {
            failures.push("geometry_centered_rms");
        }
        if self
            .router
            .iter()
            .any(|r| r.max_mass_error > limits.max_router_mass_error)
        {
            failures.push("router_mass_error");
        }
        if self
            .router
            .iter()
            .any(|r| r.min_normalized_entropy < limits.min_router_normalized_entropy)
        {
            failures.push("router_entropy");
        }
        failures
    }
}

struct Active {
    artifact: ByteCorpusModelArtifact,
    model: ResidentByteInference,
}

async fn measure_cohort(cohort: &Cohort, active: &mut Active) -> Result<Observation> {
    let plan = active.artifact.checkpoint().plan();
    let blocks = plan.block_count();
    let geometry = plan.causal_geometry();
    let mixture = geometry.and_then(|g| g.mixture());
    let mut moments: Vec<CausalBiasMoments> = Vec::new();
    let mut router: Vec<RouterHealth> = Vec::new();
    let mut batch_ce = Vec::new();
    let mut hashes = Vec::new();
    for host in &cohort.batches {
        let batch = active.model.prepare_batch(host)?;
        let tagged = active.model.forward(&batch)?;
        let forward = tagged.forward();
        if tagged.artifact_id() != active.artifact.artifact_id()
            || forward.parameter_revision() != active.artifact.checkpoint().attempted_revision()
        {
            return Err("serving measurement lost artifact/revision ownership".into());
        }
        let loss =
            forward.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
        let mut tensors = vec![forward.prediction(), loss.value()];
        if geometry.is_some() {
            for block in 0..blocks {
                tensors.push(
                    forward
                        .geometry_pair_bias(block)
                        .ok_or("missing geometry observation")?,
                );
            }
        }
        if mixture.is_some() {
            for block in 0..blocks {
                tensors.push(
                    forward
                        .geometry_mixture_weights(block)
                        .ok_or("missing router observation")?,
                );
            }
        }
        let snapshot = active.model.tensor_device().snapshot_many(&tensors)?;
        #[cfg(not(target_arch = "wasm32"))]
        let values = snapshot.read()?;
        #[cfg(target_arch = "wasm32")]
        let values = snapshot.read_async().await?;
        if values[0].is_empty()
            || values[0].iter().any(|v| !v.is_finite())
            || values[1].len() != 1
            || !values[1][0].is_finite()
            || values[1][0] < 0.
        {
            return Err("non-finite or invalid serving logits/loss".into());
        }
        let mut hash = Sha256::new();
        for v in &values[0] {
            hash.update(v.to_le_bytes());
        }
        hashes.push(format!("{:x}", hash.finalize()));
        batch_ce.push(f64::from(values[1][0]));
        if geometry.is_some() {
            for block in 0..blocks {
                let observed = CausalBiasMoments::from_scores(
                    tensors[2 + block].layout().shape().try_into()?,
                    &values[2 + block],
                )?;
                if let Some(total) = moments.get_mut(block) {
                    total.merge(&observed)?;
                } else {
                    moments.push(observed);
                }
            }
        }
        if let Some(mixture) = mixture {
            for block in 0..blocks {
                let index = 2 + blocks + block;
                let observed = router_health(mixture.metrics().len(), &values[index])?;
                if let Some(total) = router.get_mut(block) {
                    total.rows += observed.rows;
                    total.max_mass_error = total.max_mass_error.max(observed.max_mass_error);
                    total.min_normalized_entropy = total
                        .min_normalized_entropy
                        .min(observed.min_normalized_entropy);
                } else {
                    router.push(observed);
                }
            }
        }
    }
    let shape = cohort.batches[0].shape();
    Ok(Observation {
        artifact_id: active.artifact.artifact_id().to_owned(),
        revision: active.artifact.checkpoint().attempted_revision(),
        mean_ce: batch_ce.iter().sum::<f64>() / batch_ce.len() as f64,
        batch_ce,
        target_bytes: shape[0] * shape[1] * cohort.batches.len(),
        logits_f32le_sha256: hashes,
        geometry_centered_rms: moments.iter().map(CausalBiasMoments::rms).collect(),
        router,
    })
}

async fn measure(gate: &ByteCorpusServingGate, active: &mut Active) -> Result<Vec<Observation>> {
    let mut observations = Vec::with_capacity(gate.cohorts.len());
    for cohort in &gate.cohorts {
        observations.push(
            measure_cohort(cohort, active)
                .await
                .map_err(|e| format!("cohort {:?}: {e}", cohort.name))?,
        );
    }
    Ok(observations)
}

fn absolute_failures(gate: &ByteCorpusServingGate, observed: &[Observation]) -> Result<Vec<Value>> {
    if observed.len() != gate.cohorts.len() {
        return Err("incomplete serving cohort observations".into());
    }
    Ok(gate
        .cohorts
        .iter()
        .zip(observed)
        .filter_map(|(c, o)| {
            let checks = o.failures(&c.limits);
            (!checks.is_empty()).then(|| json!({"cohort":c.name,"checks":checks}))
        })
        .collect())
}

#[derive(Serialize)]
struct CohortAssessment<'a> {
    name: &'a str,
    limits: &'a Limits,
    incumbent: &'a Observation,
    candidate: &'a Observation,
    accepted: bool,
    failed_checks: Vec<&'static str>,
    mean_ce_delta: f64,
    max_batch_ce_delta: f64,
}

fn assess_cohorts<'a>(
    gate: &'a ByteCorpusServingGate,
    reference: &'a [Observation],
    observed: &'a [Observation],
) -> Result<Vec<CohortAssessment<'a>>> {
    if reference.len() != gate.cohorts.len() || observed.len() != gate.cohorts.len() {
        return Err("incomplete paired serving cohort observations".into());
    }
    gate.cohorts
        .iter()
        .zip(reference)
        .zip(observed)
        .map(|((c, incumbent), candidate)| {
            if incumbent.batch_ce.len() != c.batches.len()
                || candidate.batch_ce.len() != c.batches.len()
            {
                return Err("incomplete paired cohort batches".into());
            }
            let (failed_checks, mean_ce_delta, max_batch_ce_delta) =
                candidate.assess(incumbent, &c.limits);
            Ok(CohortAssessment {
                name: &c.name,
                limits: &c.limits,
                incumbent,
                candidate,
                accepted: failed_checks.is_empty(),
                failed_checks,
                mean_ce_delta,
                max_batch_ce_delta,
            })
        })
        .collect()
}

fn restore(runtime: WgpuRuntime, artifact: ByteCorpusModelArtifact) -> Result<Active> {
    Ok(Active {
        model: artifact.restore_wgpu(runtime)?,
        artifact,
    })
}

/// A measured, process-local permit. It cannot be imported from report JSON,
/// reused on another slot, or activated after that slot changes generation.
pub struct ResidentByteCandidate {
    owner: Arc<()>,
    generation: u64,
    incumbent_id: String,
    candidate: Active,
    accepted: bool,
    report: Value,
}

impl ResidentByteCandidate {
    pub fn accepted(&self) -> bool {
        self.accepted
    }
    pub fn artifact_id(&self) -> &str {
        self.candidate.artifact.artifact_id()
    }
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.report)?)
    }
}

/// Runtime selection below orchestration, not a registry, scheduler or durable
/// deployment service. All owners use one device and one immutable gate. A
/// failed preparation/activation leaves the active model and history unchanged.
pub struct ResidentByteServing {
    owner: Arc<()>,
    generation: u64,
    runtime: WgpuRuntime,
    gate: ByteCorpusServingGate,
    active: Active,
    previous: Option<Active>,
    bootstrap: Value,
}

impl ResidentByteServing {
    /// Measure the initial model before making it active. Absolute language and
    /// geometry limits apply; relative limits apply only to later candidates.
    pub async fn bootstrap(
        runtime: WgpuRuntime,
        artifact: ByteCorpusModelArtifact,
        gate: ByteCorpusServingGate,
    ) -> Result<Self> {
        gate.validate_artifact(&artifact)?;
        let mut active = restore(runtime.clone(), artifact)?;
        let observed = measure(&gate, &mut active).await?;
        let failed = absolute_failures(&gate, &observed)?;
        if !failed.is_empty() {
            return Err(format!("bootstrap gate rejected: {failed:?}").into());
        }
        let bootstrap = if gate.version == 1 {
            json!({"schema":"spiraltorch.byte_corpus.serving_bootstrap.v1",
                "gate_sha256":gate.request_sha256(), "observation":observed[0], "limits":gate.cohorts[0].limits})
        } else {
            let cohorts: Vec<_> = gate
                .cohorts
                .iter()
                .zip(&observed)
                .map(|(c, o)| json!({"name":c.name,"limits":c.limits,"observation":o}))
                .collect();
            json!({"schema":"spiraltorch.byte_corpus.serving_bootstrap.v2",
                "gate_sha256":gate.request_sha256(), "cohorts":cohorts})
        };
        Ok(Self {
            owner: Arc::new(()),
            generation: 0,
            runtime,
            gate,
            active,
            previous: None,
            bootstrap,
        })
    }

    pub fn artifact_id(&self) -> &str {
        self.active.artifact.artifact_id()
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn gate_sha256(&self) -> &str {
        self.gate.request_sha256()
    }
    pub fn previous_artifact_id(&self) -> Option<&str> {
        self.previous.as_ref().map(|a| a.artifact.artifact_id())
    }
    pub fn bootstrap_json(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.bootstrap)?)
    }
    pub fn tensor_device(&self) -> &st_backend_wgpu::resident_tensor::TensorDevice {
        self.active.model.tensor_device()
    }
    pub fn prepare_batch(&self, batch: &ByteLmBatch) -> Result<ResidentByteBatch> {
        Ok(self.active.model.prepare_batch(batch)?)
    }
    pub fn forward(&mut self, batch: &ResidentByteBatch) -> Result<ResidentByteInferenceForward> {
        Ok(self.active.model.forward(batch)?)
    }

    /// Remeasure both artifacts on identical pinned windows, without touching
    /// the live owner. JSON output is evidence, never a deserializable permit.
    pub async fn prepare_candidate(
        &self,
        artifact: ByteCorpusModelArtifact,
    ) -> Result<ResidentByteCandidate> {
        self.gate.validate_artifact(&artifact)?;
        if artifact.artifact_id() == self.artifact_id() {
            return Err("candidate is already active".into());
        }
        let mut incumbent = restore(self.runtime.clone(), self.active.artifact.clone())?;
        let reference = measure(&self.gate, &mut incumbent).await?;
        if !absolute_failures(&self.gate, &reference)?.is_empty() {
            return Err("incumbent no longer meets the pinned absolute gate".into());
        }
        drop(incumbent);
        let mut candidate = restore(self.runtime.clone(), artifact)?;
        let observed = measure(&self.gate, &mut candidate).await?;
        let assessments = assess_cohorts(&self.gate, &reference, &observed)?;
        let adapter = self.runtime.adapter_info();
        let accepted = assessments.iter().all(|c| c.accepted);
        let mut report = json!({"schema":format!("spiraltorch.byte_corpus.candidate_assessment.v{}", self.gate.version),
            "gate_sha256":self.gate_sha256(), "generation":self.generation.to_string(),
            "policy":self.active.artifact.policy(), "accepted":accepted,
            "adapter":{"name":adapter.name, "backend":format!("{:?}",adapter.backend),
                "device_type":format!("{:?}",adapter.device_type)}});
        if self.gate.version == 1 {
            let mut legacy = serde_json::to_value(&assessments[0])?;
            legacy
                .as_object_mut()
                .expect("assessment object")
                .remove("name");
            report
                .as_object_mut()
                .expect("report object")
                .extend(legacy.as_object().expect("assessment object").clone());
        } else {
            let failures: Vec<_> = assessments
                .iter()
                .filter(|c| !c.accepted)
                .map(|c| json!({"cohort":c.name,"checks":c.failed_checks}))
                .collect();
            report["failed_checks"] = json!(failures);
            report["cohorts"] = serde_json::to_value(&assessments)?;
        }
        Ok(ResidentByteCandidate {
            owner: self.owner.clone(),
            generation: self.generation,
            incumbent_id: self.artifact_id().to_owned(),
            candidate,
            accepted,
            report,
        })
    }

    /// Consume a fresh permit from this exact slot. Validation precedes the
    /// infallible owner swap, so rejection cannot partially replace the model.
    pub fn activate(&mut self, candidate: ResidentByteCandidate) -> Result<Value> {
        if !Arc::ptr_eq(&self.owner, &candidate.owner)
            || self.generation != candidate.generation
            || self.artifact_id() != candidate.incumbent_id
            || !candidate.accepted
        {
            return Err("foreign, stale or rejected serving candidate".into());
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or("serving generation overflow")?;
        let event = json!({"schema":"spiraltorch.byte_corpus.serving_transition.v1", "action":"activate",
            "from":self.artifact_id(), "to":candidate.artifact_id(),
            "generation":generation.to_string(), "gate_sha256":self.gate_sha256()});
        self.previous = Some(std::mem::replace(&mut self.active, candidate.candidate));
        self.generation = generation;
        Ok(event)
    }

    /// One-step rollback to the exact prior accepted owner; no retraining,
    /// reloading or policy reinterpretation. This consumes the rollback slot.
    pub fn rollback(&mut self, expected_generation: u64) -> Result<Value> {
        if self.generation != expected_generation {
            return Err("stale rollback generation".into());
        }
        let previous = self
            .previous
            .as_ref()
            .ok_or("no previous accepted serving owner")?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or("serving generation overflow")?;
        let event = json!({"schema":"spiraltorch.byte_corpus.serving_transition.v1", "action":"rollback",
            "from":self.artifact_id(), "to":previous.artifact.artifact_id(),
            "generation":generation.to_string(), "gate_sha256":self.gate_sha256()});
        self.active = self.previous.take().expect("checked previous owner");
        self.generation = generation;
        Ok(event)
    }
}

#[cfg(test)]
mod tests;
