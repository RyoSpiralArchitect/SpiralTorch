use super::*;
use std::path::Path;

const SCHEMA: &str = "spiraltorch.native_causal_lm_line_checkpoint.v1";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema: String,
    plan_id: String,
    base_id: String,
    origin_geometry_id: String,
    #[serde(deserialize_with = "nullable_owner")]
    lora_id: Option<String>,
    readout_accumulator: CausalLmReadoutAccumulator,
    updates: u32,
    geometry_id: String,
    geometry_json: String,
}

fn nullable_owner<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    Option::deserialize(d)
}

impl NativeCausalLmLineTrainingSession {
    pub const MAX_CHECKPOINT_BYTES: usize = NativeCausalLmTrainingSession::MAX_PAYLOAD_BYTES;

    /// No prompts, candidate text, base weights or frozen LoRA weights are stored.
    pub fn checkpoint_with_id(&self) -> CausalLmResult<(String, String)> {
        let art = self
            .model
            .geometry_artifact()
            .ok_or("missing line session geometry")?;
        let saved = Snapshot {
            schema: SCHEMA.into(),
            plan_id: self.plan_id.clone(),
            base_id: self.model.base_model_identity().into(),
            origin_geometry_id: self.origin_geometry_id.clone(),
            lora_id: self.model.lora_artifact_id().map(str::to_owned),
            readout_accumulator: self.model.readout_accumulator(),
            updates: self.updates,
            geometry_id: art.artifact_id().into(),
            geometry_json: art.to_json().into(),
        };
        let raw = serde_json::to_string(&saved)?;
        if raw.len() > Self::MAX_CHECKPOINT_BYTES {
            return Err("line checkpoint exceeds byte budget".into());
        }
        let id = pin(raw.as_bytes());
        Ok((raw, id))
    }

    /// Requires the *initial* adapted model, original plan bytes and a separately
    /// retained checkpoint pin. A content hash is not a signature or proof of training.
    pub fn restore(
        origin: NativeCausalLm,
        plan: &[u8],
        bytes: &[u8],
        expected_id: &str,
    ) -> CausalLmResult<Self> {
        if bytes.len() > Self::MAX_CHECKPOINT_BYTES
            || plan.len() > Self::MAX_PLAN_BYTES
            || !crate::geometry_artifact::is_sha256_id(expected_id)
            || pin(bytes) != expected_id
        {
            return Err("line checkpoint byte limit or pin mismatch".into());
        }
        let s: Snapshot = serde_json::from_slice(bytes)?;
        if s.schema != SCHEMA
            || s.plan_id != pin(plan)
            || s.base_id != origin.base_model_identity()
            || Some(s.origin_geometry_id.as_str()) != origin.geometry_artifact_id()
            || s.lora_id.as_deref() != origin.lora_artifact_id()
            || s.readout_accumulator != origin.readout_accumulator()
        {
            return Err(
                "line checkpoint schema, plan, initial model or arithmetic mismatch".into(),
            );
        }
        let mut session = Self::new(origin, plan)?;
        if s.updates > session.total || (s.updates == 0 && s.geometry_id != s.origin_geometry_id) {
            return Err("impossible line session cursor or initial geometry".into());
        }
        let g = GeometryAdapterArtifact::from_json(
            s.geometry_json.as_bytes(),
            &s.geometry_id,
            &s.base_id,
        )?;
        // Same recipes, modes, shape, trainability and bit-identical frozen values.
        session
            .model
            .geometry_artifact()
            .expect("admitted origin")
            .trainable_delta_l2(&g)?;
        session.model = session.model.with_geometry(g)?;
        session.updates = s.updates;
        Ok(session)
    }

    pub fn save_checkpoint_new(&self, path: impl AsRef<Path>) -> CausalLmResult<String> {
        checkpoint::CausalLmPreparedCheckpointFile::stage(path.as_ref(), || {
            self.checkpoint_with_id()
        })?
        .publish()
    }

    /// Absolute target; publish at a fresh path before committing the live state.
    pub fn advance_to_checkpoint_new(
        &mut self,
        stop: u32,
        path: impl AsRef<Path>,
    ) -> CausalLmResult<(Vec<CausalLmLineSessionStep>, String)> {
        let n = checkpoint::count(self.updates.into(), self.total.into(), stop.into())?;
        checkpoint::PreparedCheckpoint::stage(self, path.as_ref(), |origin| {
            let mut next = origin.clone();
            let reports = checkpoint::steps(&mut next, n, Self::step)?;
            let saved = next.checkpoint_with_id()?;
            Ok((Some(next), reports, saved))
        })?
        .publish()
    }
}
