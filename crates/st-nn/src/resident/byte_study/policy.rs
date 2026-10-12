use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GeometryUpdate {
    Train,
    Frozen,
}

// Absence belongs to v1 only; explicit null must not silently select training.
pub(super) fn explicit_update<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<GeometryUpdate>, D::Error> {
    GeometryUpdate::deserialize(deserializer).map(Some)
}

pub(super) fn same_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

impl Case {
    pub(super) fn frozen(&self) -> bool {
        self.geometry_update == Some(GeometryUpdate::Frozen)
    }
}

impl Request {
    pub(super) fn controls(&self) -> bool {
        self.schema == "spiraltorch.byte_corpus.request.v2"
    }

    pub(super) fn checkpoint_schema(&self) -> &'static str {
        if self.controls() {
            "spiraltorch.byte_corpus.checkpoint.v2"
        } else {
            CHECKPOINT_SCHEMA
        }
    }

    pub(super) fn report_schema(&self, complete: bool) -> &'static str {
        match (self.controls(), complete) {
            (false, true) => "spiraltorch.byte_corpus.result.v1",
            (false, false) => "spiraltorch.byte_corpus.partial.v1",
            (true, true) => "spiraltorch.byte_corpus.result.v2",
            (true, false) => "spiraltorch.byte_corpus.partial.v2",
        }
    }

    pub(super) fn validate_modes(&self, cases: &[&Case]) -> Result<()> {
        if !self.controls() {
            if cases.len() != 2
                || cases[0].geometry == cases[1].geometry
                || cases.iter().any(|c| c.geometry_update.is_some())
            {
                return Err("v1 requires ordinary/geometry pairs without an update policy".into());
            }
            return Ok(());
        }
        if cases.len() != 3 || cases.iter().any(|c| c.geometry_update.is_none()) {
            return Err("v2 requires three explicit update policies per seed".into());
        }
        for (geometry, update) in [
            (false, GeometryUpdate::Train),
            (true, GeometryUpdate::Train),
            (true, GeometryUpdate::Frozen),
        ] {
            if cases
                .iter()
                .filter(|c| c.geometry == geometry && c.geometry_update == Some(update))
                .count()
                != 1
            {
                return Err(
                    "v2 needs ordinary, learned geometry and frozen geometry once per seed".into(),
                );
            }
        }
        let geometric = cases.iter().filter(|c| c.geometry).collect::<Vec<_>>();
        let (a, b) = (&geometric[0].parameters, &geometric[1].parameters);
        if a.len() != b.len()
            || a.iter().zip(b).any(|(a, b)| {
                a.name != b.name || a.shape != b.shape || !same_bits(&a.values, &b.values)
            })
        {
            return Err("learned and frozen geometry must start bitwise identical".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{fixture, saved, valid};
    use super::*;

    fn controls() -> Value {
        let mut value = fixture();
        value["schema"] = json!("spiraltorch.byte_corpus.request.v2");
        for case in value["cases"].as_array_mut().unwrap() {
            case["geometry_update"] = json!("train");
        }
        let mut frozen = value["cases"][1].clone();
        frozen["name"] = json!("frozen");
        frozen["geometry_update"] = json!("frozen");
        value["cases"].as_array_mut().unwrap().push(frozen);
        value
    }

    #[test]
    fn controls_preflight_all_orders_without_changing_v1() {
        assert!(valid(&fixture()));
        let mut value = controls();
        for _ in 0..3 {
            assert!(valid(&value));
            value["cases"].as_array_mut().unwrap().rotate_left(1);
        }
        for (index, policy) in [
            (0, json!("frozen")),
            (1, json!("frozen")),
            (2, json!("train")),
            (2, Value::Null),
            (2, json!("detach")),
        ] {
            let mut value = controls();
            value["cases"][index]["geometry_update"] = policy;
            assert!(!valid(&value));
        }
        let mut value = controls();
        value["cases"][2]
            .as_object_mut()
            .unwrap()
            .remove("geometry_update");
        assert!(!valid(&value));
        for policy in [json!("train"), json!("frozen"), Value::Null] {
            let mut value = fixture();
            value["cases"][1]["geometry_update"] = policy;
            assert!(!valid(&value));
        }
    }

    #[test]
    fn controls_require_identical_geometry_and_backbone_bits() {
        for slot in [0, 2, 3, 4, 5, 6] {
            let mut value = controls();
            value["cases"][2]["parameters"][slot]["values"][0] = json!(0.2);
            assert!(!valid(&value));
        }
        let mut value = controls();
        value["cases"][1]["parameters"][2]["values"][0] = json!(0.0);
        value["cases"][2]["parameters"][2]["values"][0] = json!(-0.0);
        assert!(!valid(&value));
    }

    #[test]
    fn controls_checkpoint_binds_policy_and_frozen_bits() {
        let study = ByteCorpusStudy::from_json(controls().to_string().as_bytes()).unwrap();
        for cursor in 0..=2 {
            let checkpoint = saved(&study, cursor);
            assert_eq!(checkpoint.schema, "spiraltorch.byte_corpus.checkpoint.v2");
            study
                .checkpoint_from_json(checkpoint.to_json().unwrap().as_bytes())
                .unwrap();
        }
        let checkpoint = saved(&study, 1);
        for field in [
            "/model/geometry/projection/parameters/0/values/0",
            "/model/geometry/projection/parameters/1/values/0",
            "/model/geometry/raw_decay/0",
            "/model/geometry/raw_phase/0",
            "/model/geometry/raw_gains/0/0",
        ] {
            let mut altered = checkpoint.clone();
            let mut model: Value = serde_json::from_str(&altered.cases[2].model_json).unwrap();
            *model.pointer_mut(field).unwrap() = json!(0.2);
            altered.cases[2].model_json = model.to_string();
            assert!(study.checked_models(&altered).is_err());
        }
        let mut changed = controls();
        changed["cases"][1]["geometry_update"] = json!("frozen");
        changed["cases"][2]["geometry_update"] = json!("train");
        let other = ByteCorpusStudy::from_json(changed.to_string().as_bytes()).unwrap();
        assert!(other
            .checkpoint_from_json(checkpoint.to_json().unwrap().as_bytes())
            .is_err());
        let mut changed_schema = checkpoint.clone();
        changed_schema.schema = CHECKPOINT_SCHEMA.into();
        assert!(study.checked_models(&changed_schema).is_err());
        assert!(study.checkpoint_size_bound().unwrap() >= checkpoint.to_json().unwrap().len());
    }

    #[test]
    fn resumed_freeze_preserves_signed_zero_without_freezing_learned_arm() {
        let mut request = controls();
        for i in [1, 2] {
            request["cases"][i]["parameters"][4]["values"][0] = json!(0.0);
        }
        let study = ByteCorpusStudy::from_json(request.to_string().as_bytes()).unwrap();
        let mut checkpoint = saved(&study, 1);
        let mut model: Value = serde_json::from_str(&checkpoint.cases[1].model_json).unwrap();
        model["model"]["geometry"]["raw_decay"][0] = json!(0.2);
        checkpoint.cases[1].model_json = model.to_string();
        study.checked_models(&checkpoint).unwrap();
        let mut model: Value = serde_json::from_str(&checkpoint.cases[2].model_json).unwrap();
        model["model"]["geometry"]["raw_decay"][0] = json!(-0.0);
        checkpoint.cases[2].model_json = model.to_string();
        assert!(study.checked_models(&checkpoint).is_err());
    }
}
