use super::*;
use serde_json::{json, Value};

#[test]
fn shared_client_fixture_has_all_thirteen_variants() {
    let input = include_bytes!("../../tests/fixtures/geometry_adapter_artifact.json");
    let expected: Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/geometry_adapter_artifact.expected.json"
    ))
    .unwrap();
    let id = format!("sha256:{:x}", Sha256::digest(input));
    assert_eq!(id, expected["fixture_id"]);
    let artifact =
        GeometryAdapterArtifact::from_json(input, &id, &format!("sha256:{}", "a".repeat(64)))
            .unwrap();
    assert_eq!(artifact.record().placements.len(), 13);
    assert_eq!(artifact.to_json().as_bytes(), input);
    let canonical = GeometryAdapterArtifact::create_json(input).unwrap();
    assert_eq!(canonical.artifact_id(), expected["canonical_artifact_id"]);
    assert_eq!(canonical.state_json().unwrap(), canonical.to_json());
}

fn record() -> Value {
    json!({
        "schema": GEOMETRY_ADAPTER_ARTIFACT_SCHEMA,
        "base_model_identity": format!("sha256:{}", "a".repeat(64)),
        "policy": "full_prefix_f32_no_cache.v1", "training": true,
        "placements": [{"path": "transformer.h.0.mlp", "training": true,
            "recipe": {"schema": "spiraltorch.wave_gate_adapter.v1", "features": 2, "strength": 0.1,
                "kernel": {"curvature": -0.75, "saturation": 1.0, "porosity": 0.05, "max_values": 64}},
            "tensors": [tensor("gate", &[2], 0.2), tensor("bias", &[2], -0.0)]}]
    })
}

fn tensor(name: &str, shape: &[usize], value: f32) -> Value {
    json!({"name": name, "shape": shape, "values": vec![value; shape.iter().product()], "requires_grad": true, "role": "parameter"})
}

fn create(value: &Value) -> Result<GeometryAdapterArtifact> {
    GeometryAdapterArtifact::create_json(value.to_string().as_bytes())
}

#[test]
fn exact_byte_and_base_id_required_and_no_mutable_record_escape() {
    let artifact = create(&record()).unwrap();
    let base = &artifact.record().base_model_identity;
    let restored = GeometryAdapterArtifact::from_json(
        artifact.to_json().as_bytes(),
        artifact.artifact_id(),
        base,
    )
    .unwrap();
    assert_eq!(restored.to_json(), artifact.to_json());
    assert_eq!(
        restored.record().placements[0].tensor("bias")[0].to_bits(),
        (-0.0f32).to_bits()
    );
    let modified = format!("{} ", artifact.to_json());
    assert!(
        GeometryAdapterArtifact::from_json(modified.as_bytes(), artifact.artifact_id(), base)
            .is_err()
    );
    assert!(GeometryAdapterArtifact::from_json(
        artifact.to_json().as_bytes(),
        artifact.artifact_id(),
        &format!("sha256:{}", "b".repeat(64))
    )
    .is_err());
}

#[test]
fn rejects_schema_routing_tensor_and_policy_changes() {
    for pointer in [
        "/schema",
        "/policy",
        "/base_model_identity",
        "/placements/0/recipe/schema",
    ] {
        let mut value = record();
        *value.pointer_mut(pointer).unwrap() = json!("invalid");
        assert!(create(&value).is_err(), "{pointer}");
    }
    for path in ["", "a..b", "a\nb"] {
        let mut value = record();
        value["placements"][0]["path"] = json!(path);
        assert!(create(&value).is_err());
    }
    for path in [
        "transformer.h.0.mlp",
        "transformer.h.0",
        "transformer.h.0.mlp.child",
    ] {
        let mut value = record();
        let mut next = value["placements"][0].clone();
        next["path"] = json!(path);
        value["placements"].as_array_mut().unwrap().push(next);
        assert!(create(&value).is_err());
    }
    for field in ["name", "shape", "values"] {
        let mut value = record();
        value["placements"][0]["tensors"][0][field] = match field {
            "name" => json!("bias"),
            "shape" => json!([1, 2]),
            _ => json!([1.0]),
        };
        assert!(create(&value).is_err());
    }
    let mut value = record();
    value["extra"] = json!(false);
    assert!(create(&value).is_err());
    let mut value = record();
    value["placements"][0]["recipe"]["extra"] = json!(false);
    assert!(create(&value).is_err());
    let mut value = record();
    value["placements"][0]["tensors"][0]["values"][0] = json!(1e100);
    assert!(create(&value).is_err());
    let duplicate = record()
        .to_string()
        .replace("\"training\":true", "\"training\":true,\"training\":false");
    assert!(GeometryAdapterArtifact::create_json(duplicate.as_bytes()).is_err());
}

#[test]
fn all_builtin_recipes_validate_and_native_domains_reject_invalid_parameters() {
    let mut wave = record();
    wave["placements"][0]["recipe"]["schema"] = json!("spiraltorch.wave_gate_adapter.v2");
    wave["placements"][0]["recipe"]["learnable_radius"] = json!(false);
    let mut radius = tensor("log_radius", &[], 0.0);
    radius["requires_grad"] = json!(false);
    radius["role"] = json!("buffer");
    wave["placements"][0]["tensors"]
        .as_array_mut()
        .unwrap()
        .push(radius);
    assert!(create(&wave).is_ok());
    wave["placements"][0]["tensors"][2]["requires_grad"] = json!(true);
    assert!(create(&wave).is_err());
    wave["placements"][0]["recipe"]["learnable_radius"] = json!(true);
    wave["placements"][0]["tensors"][2]["role"] = json!("parameter");
    wave["placements"][0]["tensors"][2]["values"][0] = json!(100.0);
    assert!(create(&wave).is_err());
    for family in [
        "elliptic_residual",
        "elliptic_causal_residual",
        "elliptic_gated_causal_residual",
        "elliptic_anchored_residual",
    ] {
        let mut value = record();
        let p = &mut value["placements"][0];
        p["recipe"] = json!({"schema": format!("spiraltorch.{family}_adapter.v1"), "features": 2, "strength": 0.1,
            "warp": {"curvature_radius": 1.0, "sheet_count": 2, "spin_harmonics": 1}});
        p["tensors"] = json!([
            tensor("orientation.weight", &[2, 2], 0.3),
            tensor("orientation.bias", &[2], 0.0),
            tensor("readout.weight", &[2, 9], 0.1)
        ]);
        if family.contains("gated") || family.contains("anchored") {
            p["tensors"]
                .as_array_mut()
                .unwrap()
                .push(tensor("raw_mix", &[], 0.2));
        }
        assert!(create(&value).is_ok(), "{family}");
        value["placements"][0]["recipe"]["warp"]["curvature_radius"] = json!(0.0);
        assert!(create(&value).is_err());
    }
    let mut topos = record();
    topos["placements"][0]["recipe"] = json!({"schema": "spiraltorch.topos_resonator_adapter.v1", "features": 2, "strength": 0.1,
        "kernel": {"coupling": 0.25, "iterations": 4, "saturation": 1.0, "porosity": 0.0, "max_values": 64}});
    topos["placements"][0]["tensors"] = json!([tensor("gate", &[2], 0.1)]);
    assert!(create(&topos).is_ok());
    for family in [
        "memory",
        "history",
        "l2_history",
        "gain_history",
        "angle_gain_history",
    ] {
        let mut value = record();
        let p = &mut value["placements"][0];
        p["recipe"] = json!({"schema": format!("spiraltorch.fractional_{family}_adapter.v1"), "features": 2, "strength": 0.1,
            "kernel": {"kernel_len": 4, "step": 1.0, "max_values": 64, "max_products": 256}});
        let angle = family.contains("angle");
        let gain = family.contains("gain");
        p["tensors"] = json!([
            tensor("gate", &[2], 0.1),
            tensor(if angle { "history_angle" } else { "log_alpha" }, &[], 0.0)
        ]);
        if family != "memory" {
            p["tensors"]
                .as_array_mut()
                .unwrap()
                .push(tensor("local_gate", &[2], 0.1));
        }
        if gain {
            p["tensors"]
                .as_array_mut()
                .unwrap()
                .push(tensor("log_gain", &[], 0.0));
            p["recipe"]["lag_window"] = json!([1, 3]);
        }
        if family == "l2_history" {
            p["recipe"]["gain"] = json!(1.0);
        }
        assert!(create(&value).is_ok(), "{family}");
        value["placements"][0]["tensors"][1]["values"][0] = json!(1000.0);
        assert!(create(&value).is_err());
    }
}

#[test]
fn rejects_role_changes_and_unbounded_or_intrinsically_invalid_fractional_recipes() {
    let mut value = record();
    value["placements"][0]["tensors"][0]["role"] = json!("buffer");
    assert!(create(&value).unwrap_err().to_string().contains("role"));
    let mut value: Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/geometry_adapter_artifact.json"
    ))
    .unwrap();
    let mut p = value["placements"][8].clone();
    p["recipe"]["kernel"]["step"] = json!(1e-30);
    p["tensors"][1]["values"][0] = json!(2f32.ln());
    value["placements"] = json!([p]);
    assert!(create(&value).is_err());
    value["placements"][0]["recipe"]["kernel"]["step"] = json!(1.0);
    value["placements"][0]["recipe"]["kernel"]["kernel_len"] = json!(65_537);
    assert!(create(&value)
        .unwrap_err()
        .to_string()
        .contains("tap budget"));
}
