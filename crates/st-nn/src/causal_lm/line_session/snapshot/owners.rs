use super::*;

const V2: &str = "spiraltorch.native_causal_lm_line_checkpoint.v2";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Owner {
    #[serde(deserialize_with = "nullable_owner")]
    origin_id: Option<String>,
    #[serde(deserialize_with = "nullable_owner")]
    id: Option<String>,
    #[serde(deserialize_with = "nullable_owner")]
    weights: Option<String>,
}

impl Owner {
    fn admit(&self, origin: Option<&str>, selected: bool, updates: u32) -> CausalLmResult<()> {
        if self.origin_id.as_deref() != origin
            || self.id.is_some() != origin.is_some()
            || (selected && origin.is_none())
            || self.weights.is_some() != selected
            || ((!selected || updates == 0) && self.id != self.origin_id)
        {
            return Err("line checkpoint adapter lineage, payload or cursor mismatch".into());
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    schema: String,
    plan_id: String,
    base_id: String,
    mode: CausalLmAdaptationMode,
    readout_accumulator: CausalLmReadoutAccumulator,
    updates: u32,
    lora: Owner,
    geometry: Owner,
}

impl NativeCausalLmLineTrainingSession {
    pub(super) fn owners_checkpoint(&self) -> CausalLmResult<(String, String)> {
        let m = &self.model;
        let lora = Owner {
            origin_id: self.origin_lora_id.clone(),
            id: m.lora_artifact_id().map(str::to_owned),
            weights: (self.mode != CausalLmAdaptationMode::Geometry)
                .then(|| m.lora_to_json())
                .transpose()?,
        };
        let geometry = Owner {
            origin_id: self.origin_geometry_id.clone(),
            id: m.geometry_artifact_id().map(str::to_owned),
            weights: if self.mode == CausalLmAdaptationMode::Lora {
                None
            } else {
                Some(
                    m.geometry_artifact()
                        .ok_or("missing selected geometry")?
                        .to_json()
                        .to_owned(),
                )
            },
        };
        if lora.weights.as_ref().map_or(0, String::len)
            + geometry.weights.as_ref().map_or(0, String::len)
            > Self::MAX_CHECKPOINT_BYTES
        {
            return Err("line checkpoint exceeds byte budget".into());
        }
        Self::finish_checkpoint(&State {
            schema: V2.into(),
            plan_id: self.plan_id.clone(),
            base_id: m.base_model_identity().into(),
            mode: self.mode,
            readout_accumulator: m.readout_accumulator(),
            updates: self.updates,
            lora,
            geometry,
        })
    }

    pub(super) fn restore_owners(mut self, bytes: &[u8]) -> CausalLmResult<Self> {
        let s: State = serde_json::from_slice(bytes)?;
        if s.schema != V2
            || s.plan_id != self.plan_id
            || s.base_id != self.model.base_model_identity()
            || s.mode != self.mode
            || s.readout_accumulator != self.model.readout_accumulator()
            || s.updates > self.total
        {
            return Err(
                "line checkpoint schema, plan, model, arithmetic or cursor mismatch".into(),
            );
        }
        s.lora.admit(
            self.model.lora_artifact_id(),
            s.mode != CausalLmAdaptationMode::Geometry,
            s.updates,
        )?;
        s.geometry.admit(
            self.model.geometry_artifact_id(),
            s.mode != CausalLmAdaptationMode::Lora,
            s.updates,
        )?;
        if let Some(weights) = s.lora.weights {
            let next = self
                .model
                .with_lora_json_pinned(&weights, s.lora.id.as_deref().expect("admitted LoRA ID"))?;
            if !self
                .model
                .lora
                .as_ref()
                .expect("admitted LoRA")
                .same_training_layout(next.lora.as_ref().expect("restored LoRA"))
            {
                return Err("line checkpoint LoRA identity or layout mismatch".into());
            }
            self.model = next;
        }
        if let (Some(raw), Some(id)) = (s.geometry.weights, s.geometry.id) {
            let g = GeometryAdapterArtifact::from_json(raw.as_bytes(), &id, &s.base_id)?;
            self.model
                .geometry_artifact()
                .expect("admitted geometry")
                .trainable_delta_l2(&g)?;
            self.model = self.model.with_geometry(g)?;
        }
        self.updates = s.updates;
        Ok(self)
    }
}
