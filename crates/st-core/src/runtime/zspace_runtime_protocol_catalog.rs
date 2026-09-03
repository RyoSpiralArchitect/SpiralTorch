// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rust-owned catalog for the cross-client Z-space runtime protocol surface.
//!
//! The catalog does not claim that every SpiralTorch API is available in every
//! client. It records the exact admission-certified operations that share one
//! Rust semantic owner across Rust, Python, and WebAssembly. Trusted-local
//! convenience transports remain outside the catalog rather than weakening its
//! hostile-input boundary.

use super::canonical_json::values_equivalent;
use super::zspace_generation_evidence::{
    ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION, ZSPACE_GENERATION_EVIDENCE_KIND,
    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_BYTES, ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH,
    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_NODES, ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND,
    ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER,
};
use super::zspace_periodicity::{
    ZSPACE_PERIODICITY_CONTRACT_VERSION, ZSPACE_PERIODICITY_KIND,
    ZSPACE_PERIODICITY_MAX_INGRESS_BYTES, ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH,
    ZSPACE_PERIODICITY_MAX_INGRESS_NODES, ZSPACE_PERIODICITY_SEMANTIC_BACKEND,
    ZSPACE_PERIODICITY_SEMANTIC_OWNER,
};
use super::zspace_repetition_unlikelihood::{
    ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION, ZSPACE_REPETITION_UNLIKELIHOOD_KIND,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES,
    ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND, ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
};
use super::zspace_semantic_review::{
    ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION, ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND,
    ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION, ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA,
    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_BYTES, ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH,
    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_NODES, ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION,
    ZSPACE_SEMANTIC_REVIEW_PACKET_KIND, ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND,
    ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER, ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION,
    ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND,
};
use super::zspace_stochastic_schrodinger::{
    ZSPACE_SCHRODINGER_COMPLEX_CONTRACT_VERSION, ZSPACE_SCHRODINGER_COMPLEX_KIND,
    ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES, ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_NODES,
    ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION,
    ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND, ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES,
    ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND, ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
    ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION, ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_runtime_protocol_catalog.v4";
pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND: &str =
    "spiraltorch.zspace_runtime_protocol_catalog";
pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER: &str =
    "st-core::runtime::zspace_runtime_protocol_catalog";
pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS: &str = "ready";
/// Maximum JSON-compatible bytes admitted when replaying the catalog itself.
pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_BYTES: u64 = 1_024 * 1_024;
/// Maximum JSON-compatible values admitted when replaying the catalog itself.
pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_NODES: u64 = 20_000;
/// Maximum JSON-compatible nesting admitted when replaying the catalog itself.
pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH: u32 = 16;
pub const ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE: &str =
    "sha256(contract_version UTF-8 || NUL || compact catalog JSON with catalog_id empty)";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRuntimeProtocolArtifact {
    pub name: String,
    pub contract_version: String,
    pub discriminator_field: String,
    pub discriminator_value: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRuntimeProtocolAdmissionLimits {
    pub maximum_bytes: u64,
    pub maximum_nodes: u64,
    pub maximum_depth: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRuntimeProtocolAdmission {
    pub profile: String,
    pub guarantee: String,
    pub limits: Option<ZSpaceRuntimeProtocolAdmissionLimits>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRuntimeProtocolClientSurface {
    pub client: String,
    pub package: String,
    pub transport: String,
    pub normal_admission: ZSpaceRuntimeProtocolAdmission,
    pub operations: Vec<String>,
    pub trusted_legacy_replay: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRuntimeProtocolDescriptor {
    pub name: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub admission_owner: String,
    pub artifacts: Vec<ZSpaceRuntimeProtocolArtifact>,
    pub clients: Vec<ZSpaceRuntimeProtocolClientSurface>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceRuntimeProtocolCatalog {
    pub contract_version: String,
    pub kind: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub catalog_validated: bool,
    pub catalog_id: String,
    pub catalog_id_rule: String,
    pub status: String,
    pub protocol_count: usize,
    pub protocol_order_rule: String,
    pub client_order_rule: String,
    pub legacy_replay_policy: String,
    pub protocols: Vec<ZSpaceRuntimeProtocolDescriptor>,
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceRuntimeProtocolCatalogError {
    #[error("runtime protocol catalog encoding failed: {message}")]
    Encoding { message: String },
    #[error("malformed runtime protocol catalog: {message}")]
    MalformedCatalog { message: String },
    #[error("runtime protocol catalog does not match the current Rust-owned surface")]
    CatalogMismatch,
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn artifact(
    name: &str,
    contract_version: &str,
    discriminator_field: &str,
    discriminator_value: &str,
) -> ZSpaceRuntimeProtocolArtifact {
    ZSpaceRuntimeProtocolArtifact {
        name: name.to_owned(),
        contract_version: contract_version.to_owned(),
        discriminator_field: discriminator_field.to_owned(),
        discriminator_value: discriminator_value.to_owned(),
    }
}

fn ingress_limits(
    maximum_bytes: u64,
    maximum_nodes: u64,
    maximum_depth: u32,
) -> ZSpaceRuntimeProtocolAdmissionLimits {
    ZSpaceRuntimeProtocolAdmissionLimits {
        maximum_bytes,
        maximum_nodes,
        maximum_depth,
    }
}

fn typed_native_admission() -> ZSpaceRuntimeProtocolAdmission {
    ZSpaceRuntimeProtocolAdmission {
        profile: "typed_native".to_owned(),
        guarantee: "Rust receives typed values and applies protocol admission before protocol work; no serialized client-ingress budget applies".to_owned(),
        limits: None,
    }
}

fn passive_json_container_admission(
    maximum_bytes: u64,
    maximum_nodes: u64,
    maximum_depth: u32,
) -> ZSpaceRuntimeProtocolAdmission {
    ZSpaceRuntimeProtocolAdmission {
        profile: "passive_json_containers".to_owned(),
        guarantee: "only concrete dict/list/tuple-backed JSON containers are traversed; Rust charges byte, node, and depth budgets during conversion before protocol deserialization".to_owned(),
        limits: Some(ingress_limits(
            maximum_bytes,
            maximum_nodes,
            maximum_depth,
        )),
    }
}

fn bounded_json_string_admission(
    maximum_bytes: u64,
    maximum_nodes: u64,
    maximum_depth: u32,
) -> ZSpaceRuntimeProtocolAdmission {
    ZSpaceRuntimeProtocolAdmission {
        profile: "bounded_json_string".to_owned(),
        guarantee: "UTF-8 byte preflight precedes Rust String materialization; the duplicate-key-rejecting parser charges node and depth budgets before protocol deserialization".to_owned(),
        limits: Some(ingress_limits(
            maximum_bytes,
            maximum_nodes,
            maximum_depth,
        )),
    }
}

fn client(
    client: &str,
    package: &str,
    transport: &str,
    normal_admission: ZSpaceRuntimeProtocolAdmission,
    operations: &[&str],
    trusted_legacy_replay: bool,
) -> ZSpaceRuntimeProtocolClientSurface {
    ZSpaceRuntimeProtocolClientSurface {
        client: client.to_owned(),
        package: package.to_owned(),
        transport: transport.to_owned(),
        normal_admission,
        operations: strings(operations),
        trusted_legacy_replay,
    }
}

fn generation_evidence_descriptor() -> ZSpaceRuntimeProtocolDescriptor {
    ZSpaceRuntimeProtocolDescriptor {
        name: "generation_evidence".to_owned(),
        semantic_owner: ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER.to_owned(),
        semantic_backend: ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND.to_owned(),
        admission_owner: "rust".to_owned(),
        artifacts: vec![artifact(
            "evidence_report",
            ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION,
            "kind",
            ZSPACE_GENERATION_EVIDENCE_KIND,
        )],
        clients: vec![
            client(
                "rust",
                "st-core",
                "native",
                typed_native_admission(),
                &[
                    "summarize_zspace_generation_evidence",
                    "validate_zspace_generation_evidence_value",
                ],
                false,
            ),
            client(
                "python",
                "spiraltorch",
                "bounded_mapping",
                passive_json_container_admission(
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_BYTES,
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_NODES,
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspace_generation_evidence",
                    "validate_zspace_generation_evidence",
                ],
                false,
            ),
            client(
                "wasm",
                "spiraltorch-wasm",
                "bounded_json",
                bounded_json_string_admission(
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_BYTES,
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_NODES,
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspaceGenerationEvidenceJson",
                    "validateZspaceGenerationEvidenceJson",
                ],
                false,
            ),
        ],
    }
}

fn periodicity_descriptor() -> ZSpaceRuntimeProtocolDescriptor {
    ZSpaceRuntimeProtocolDescriptor {
        name: "periodicity".to_owned(),
        semantic_owner: ZSPACE_PERIODICITY_SEMANTIC_OWNER.to_owned(),
        semantic_backend: ZSPACE_PERIODICITY_SEMANTIC_BACKEND.to_owned(),
        admission_owner: "rust".to_owned(),
        artifacts: vec![artifact(
            "analysis_report",
            ZSPACE_PERIODICITY_CONTRACT_VERSION,
            "kind",
            ZSPACE_PERIODICITY_KIND,
        )],
        clients: vec![
            client(
                "rust",
                "st-core",
                "native",
                typed_native_admission(),
                &[
                    "analyze_zspace_periodicity",
                    "validate_zspace_periodicity_value",
                ],
                false,
            ),
            client(
                "python",
                "spiraltorch",
                "bounded_mapping",
                passive_json_container_admission(
                    ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
                    ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
                    ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH,
                ),
                &["zspace_periodicity", "validate_zspace_periodicity"],
                false,
            ),
            client(
                "wasm",
                "spiraltorch-wasm",
                "bounded_json",
                bounded_json_string_admission(
                    ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
                    ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
                    ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH,
                ),
                &["zspacePeriodicityJson", "validateZspacePeriodicityJson"],
                false,
            ),
        ],
    }
}

fn stochastic_schrodinger_descriptor() -> ZSpaceRuntimeProtocolDescriptor {
    ZSpaceRuntimeProtocolDescriptor {
        name: "stochastic_schrodinger".to_owned(),
        semantic_owner: ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER.to_owned(),
        semantic_backend: ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND.to_owned(),
        admission_owner: "rust".to_owned(),
        artifacts: vec![
            artifact(
                "forward_receipt",
                ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION,
                "kind",
                ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND,
            ),
            artifact(
                "vjp_receipt",
                ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION,
                "kind",
                ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND,
            ),
        ],
        clients: vec![
            client(
                "rust",
                "st-core",
                "native",
                typed_native_admission(),
                &[
                    "run_zspace_stochastic_schrodinger_forward",
                    "validate_zspace_stochastic_schrodinger_forward_value",
                    "run_zspace_stochastic_schrodinger_vjp",
                    "validate_zspace_stochastic_schrodinger_vjp_value",
                ],
                false,
            ),
            client(
                "python",
                "spiraltorch",
                "bounded_mapping",
                passive_json_container_admission(
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspace_stochastic_schrodinger_forward",
                    "validate_zspace_stochastic_schrodinger_forward",
                    "zspace_stochastic_schrodinger_vjp",
                    "validate_zspace_stochastic_schrodinger_vjp",
                ],
                false,
            ),
            client(
                "wasm",
                "spiraltorch-wasm",
                "bounded_json",
                bounded_json_string_admission(
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspaceStochasticSchrodingerForwardJson",
                    "validateZspaceStochasticSchrodingerForwardJson",
                    "zspaceStochasticSchrodingerVjpJson",
                    "validateZspaceStochasticSchrodingerVjpJson",
                ],
                false,
            ),
        ],
    }
}

fn complex_schrodinger_descriptor() -> ZSpaceRuntimeProtocolDescriptor {
    ZSpaceRuntimeProtocolDescriptor {
        name: "stochastic_schrodinger_complex".to_owned(),
        semantic_owner: ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER.to_owned(),
        semantic_backend: ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND.to_owned(),
        admission_owner: "rust".to_owned(),
        artifacts: vec![artifact(
            "complex_step_receipt",
            ZSPACE_SCHRODINGER_COMPLEX_CONTRACT_VERSION,
            "kind",
            ZSPACE_SCHRODINGER_COMPLEX_KIND,
        )],
        clients: vec![
            client(
                "rust",
                "st-core",
                "native",
                typed_native_admission(),
                &[
                    "run_zspace_stochastic_schrodinger_complex_step",
                    "validate_zspace_stochastic_schrodinger_complex_value",
                ],
                false,
            ),
            client(
                "python",
                "spiraltorch",
                "bounded_mapping",
                passive_json_container_admission(
                    ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES,
                    ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_NODES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspace_stochastic_schrodinger_complex_step",
                    "validate_zspace_stochastic_schrodinger_complex",
                ],
                false,
            ),
            client(
                "wasm",
                "spiraltorch-wasm",
                "bounded_json",
                bounded_json_string_admission(
                    ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES,
                    ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_NODES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspaceStochasticSchrodingerComplexStepJson",
                    "validateZspaceStochasticSchrodingerComplexJson",
                ],
                false,
            ),
        ],
    }
}

fn repetition_unlikelihood_descriptor() -> ZSpaceRuntimeProtocolDescriptor {
    ZSpaceRuntimeProtocolDescriptor {
        name: "repetition_unlikelihood".to_owned(),
        semantic_owner: ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER.to_owned(),
        semantic_backend: ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND.to_owned(),
        admission_owner: "rust".to_owned(),
        artifacts: vec![artifact(
            "plan",
            ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION,
            "kind",
            ZSPACE_REPETITION_UNLIKELIHOOD_KIND,
        )],
        clients: vec![
            client(
                "rust",
                "st-core",
                "native",
                typed_native_admission(),
                &[
                    "plan_zspace_repetition_unlikelihood",
                    "validate_zspace_repetition_unlikelihood_value",
                    "validate_zspace_repetition_unlikelihood_value_trusted_legacy_replay",
                ],
                true,
            ),
            client(
                "python",
                "spiraltorch",
                "bounded_mapping",
                passive_json_container_admission(
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES,
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspace_repetition_unlikelihood_plan",
                    "validate_zspace_repetition_unlikelihood_plan",
                    "validate_zspace_repetition_unlikelihood_plan_trusted_legacy_replay",
                ],
                true,
            ),
            client(
                "wasm",
                "spiraltorch-wasm",
                "bounded_json",
                bounded_json_string_admission(
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES,
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspaceRepetitionUnlikelihoodPlanJson",
                    "validateZspaceRepetitionUnlikelihoodPlanJson",
                ],
                false,
            ),
        ],
    }
}

fn semantic_review_descriptor() -> ZSpaceRuntimeProtocolDescriptor {
    ZSpaceRuntimeProtocolDescriptor {
        name: "semantic_review".to_owned(),
        semantic_owner: ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER.to_owned(),
        semantic_backend: ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND.to_owned(),
        admission_owner: "rust".to_owned(),
        artifacts: vec![
            artifact(
                "map_commitment",
                ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION,
                "schema",
                ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA,
            ),
            artifact(
                "packet_receipt",
                ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION,
                "kind",
                ZSPACE_SEMANTIC_REVIEW_PACKET_KIND,
            ),
            artifact(
                "draft_receipt",
                ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION,
                "kind",
                ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND,
            ),
            artifact(
                "unblind_report",
                ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION,
                "kind",
                ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND,
            ),
        ],
        clients: vec![
            client(
                "rust",
                "st-core",
                "native",
                typed_native_admission(),
                &[
                    "zspace_semantic_review_map_id",
                    "seal_zspace_semantic_review_packet",
                    "validate_zspace_semantic_review_packet",
                    "validate_zspace_semantic_review_packet_receipt_value",
                    "summarize_zspace_semantic_review_draft",
                    "validate_zspace_semantic_review_draft_receipt_value",
                    "unblind_zspace_semantic_review",
                    "validate_zspace_semantic_review_unblind_value",
                    "zspace_semantic_review_map_id_trusted_legacy_replay",
                    "validate_zspace_semantic_review_packet_trusted_legacy_replay",
                    "validate_zspace_semantic_review_packet_receipt_value_trusted_legacy_replay",
                    "validate_zspace_semantic_review_draft_receipt_value_trusted_legacy_replay",
                    "validate_zspace_semantic_review_unblind_value_trusted_legacy_replay",
                ],
                true,
            ),
            client(
                "python",
                "spiraltorch",
                "bounded_mapping",
                passive_json_container_admission(
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_BYTES,
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_NODES,
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspace_semantic_review_map_id",
                    "seal_zspace_semantic_review_packet",
                    "validate_zspace_semantic_review_packet",
                    "validate_zspace_semantic_review_packet_receipt",
                    "summarize_zspace_semantic_review_draft",
                    "validate_zspace_semantic_review_draft_receipt",
                    "unblind_zspace_semantic_review",
                    "validate_zspace_semantic_review_unblind",
                    "zspace_semantic_review_map_id_trusted_legacy_replay",
                    "validate_zspace_semantic_review_packet_trusted_legacy_replay",
                    "validate_zspace_semantic_review_packet_receipt_trusted_legacy_replay",
                    "validate_zspace_semantic_review_draft_receipt_trusted_legacy_replay",
                    "validate_zspace_semantic_review_unblind_trusted_legacy_replay",
                ],
                true,
            ),
            client(
                "wasm",
                "spiraltorch-wasm",
                "bounded_json",
                bounded_json_string_admission(
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_BYTES,
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_NODES,
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH,
                ),
                &[
                    "zspaceSemanticReviewMapIdJson",
                    "sealZspaceSemanticReviewPacketJson",
                    "validateZspaceSemanticReviewPacketJson",
                    "validateZspaceSemanticReviewPacketReceiptJson",
                    "summarizeZspaceSemanticReviewDraftJson",
                    "validateZspaceSemanticReviewDraftReceiptJson",
                    "unblindZspaceSemanticReviewJson",
                    "validateZspaceSemanticReviewUnblindJson",
                ],
                false,
            ),
        ],
    }
}

fn catalog_id(
    catalog: &ZSpaceRuntimeProtocolCatalog,
) -> Result<String, ZSpaceRuntimeProtocolCatalogError> {
    let mut content = catalog.clone();
    content.catalog_id.clear();
    let encoded = serde_json::to_vec(&content).map_err(|error| {
        ZSpaceRuntimeProtocolCatalogError::Encoding {
            message: error.to_string(),
        }
    })?;
    let mut hasher = Sha256::new();
    hasher.update(ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION.as_bytes());
    hasher.update([0]);
    hasher.update(encoded);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

pub fn zspace_runtime_protocol_catalog(
) -> Result<ZSpaceRuntimeProtocolCatalog, ZSpaceRuntimeProtocolCatalogError> {
    let protocols = vec![
        generation_evidence_descriptor(),
        periodicity_descriptor(),
        stochastic_schrodinger_descriptor(),
        complex_schrodinger_descriptor(),
        repetition_unlikelihood_descriptor(),
        semantic_review_descriptor(),
    ];
    let mut catalog = ZSpaceRuntimeProtocolCatalog {
        contract_version: ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION.to_owned(),
        kind: ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND.to_owned(),
        semantic_owner: ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER.to_owned(),
        semantic_backend: ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND.to_owned(),
        catalog_validated: true,
        catalog_id: String::new(),
        catalog_id_rule: ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE.to_owned(),
        status: ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS.to_owned(),
        protocol_count: protocols.len(),
        protocol_order_rule:
            "generation_evidence,periodicity,stochastic_schrodinger,stochastic_schrodinger_complex,repetition_unlikelihood,semantic_review"
                .to_owned(),
        client_order_rule: "rust,python,wasm".to_owned(),
        legacy_replay_policy:
            "trusted legacy replay is explicit and Rust/Python-only; WASM never exposes it"
                .to_owned(),
        protocols,
    };
    catalog.catalog_id = catalog_id(&catalog)?;
    Ok(catalog)
}

pub fn validate_zspace_runtime_protocol_catalog_value(
    value: serde_json::Value,
) -> Result<ZSpaceRuntimeProtocolCatalog, ZSpaceRuntimeProtocolCatalogError> {
    if !value.is_object() {
        return Err(ZSpaceRuntimeProtocolCatalogError::MalformedCatalog {
            message: "catalog must be an object".to_owned(),
        });
    }
    let submitted: ZSpaceRuntimeProtocolCatalog =
        serde_json::from_value(value.clone()).map_err(|error| {
            ZSpaceRuntimeProtocolCatalogError::MalformedCatalog {
                message: error.to_string(),
            }
        })?;
    let canonical = zspace_runtime_protocol_catalog()?;
    let canonical_value = serde_json::to_value(&canonical).map_err(|error| {
        ZSpaceRuntimeProtocolCatalogError::Encoding {
            message: error.to_string(),
        }
    })?;
    if submitted.catalog_id != canonical.catalog_id || !values_equivalent(&value, &canonical_value)
    {
        return Err(ZSpaceRuntimeProtocolCatalogError::CatalogMismatch);
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn catalog_is_deterministic_and_records_all_clients() {
        let catalog = zspace_runtime_protocol_catalog().expect("catalog");
        let replay = zspace_runtime_protocol_catalog().expect("catalog replay");

        assert_eq!(catalog, replay);
        assert_eq!(catalog.protocol_count, 6);
        assert!(catalog.catalog_id.starts_with("sha256:"));
        assert_eq!(catalog.catalog_id.len(), 71);
        assert_eq!(
            catalog.catalog_id_rule,
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE
        );
        assert_eq!(
            catalog
                .protocols
                .iter()
                .map(|protocol| protocol.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "generation_evidence",
                "periodicity",
                "stochastic_schrodinger",
                "stochastic_schrodinger_complex",
                "repetition_unlikelihood",
                "semantic_review"
            ]
        );
        for protocol in &catalog.protocols {
            assert_eq!(protocol.semantic_backend, "rust");
            assert_eq!(protocol.admission_owner, "rust");
            assert_eq!(
                protocol
                    .clients
                    .iter()
                    .map(|surface| surface.client.as_str())
                    .collect::<Vec<_>>(),
                vec!["rust", "python", "wasm"]
            );
            assert!(protocol
                .clients
                .iter()
                .all(|surface| !surface.operations.is_empty()));
            assert_eq!(protocol.clients[0].normal_admission.profile, "typed_native");
            assert!(protocol.clients[0].normal_admission.limits.is_none());
            assert_eq!(
                protocol.clients[1].normal_admission.profile,
                "passive_json_containers"
            );
            assert_eq!(
                protocol.clients[2].normal_admission.profile,
                "bounded_json_string"
            );
            assert_eq!(
                protocol.clients[1].normal_admission.limits,
                protocol.clients[2].normal_admission.limits
            );
            assert!(protocol
                .clients
                .iter()
                .all(|surface| !surface.normal_admission.guarantee.is_empty()));
        }
    }

    #[test]
    fn normal_admission_limits_are_protocol_specific_and_rust_owned() {
        let catalog = zspace_runtime_protocol_catalog().expect("catalog");
        for protocol in &catalog.protocols {
            let expected = match protocol.name.as_str() {
                "generation_evidence" => ingress_limits(
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_BYTES,
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_NODES,
                    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH,
                ),
                "periodicity" => ingress_limits(
                    ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
                    ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
                    ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH,
                ),
                "stochastic_schrodinger" => ingress_limits(
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
                ),
                "stochastic_schrodinger_complex" => ingress_limits(
                    ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES,
                    ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_NODES,
                    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
                ),
                "repetition_unlikelihood" => ingress_limits(
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES,
                    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH,
                ),
                "semantic_review" => ingress_limits(
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_BYTES,
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_NODES,
                    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH,
                ),
                other => panic!("unexpected protocol {other}"),
            };
            assert_eq!(
                protocol.clients[1].normal_admission.limits.as_ref(),
                Some(&expected)
            );
            assert_eq!(
                protocol.clients[2].normal_admission.limits.as_ref(),
                Some(&expected)
            );
        }
    }

    #[test]
    fn legacy_replay_is_never_advertised_to_wasm() {
        let catalog = zspace_runtime_protocol_catalog().expect("catalog");
        for protocol in &catalog.protocols {
            let wasm = protocol
                .clients
                .iter()
                .find(|surface| surface.client == "wasm")
                .expect("WASM surface");
            assert!(!wasm.trusted_legacy_replay);
            assert!(wasm
                .operations
                .iter()
                .all(|operation| !operation.contains("legacy")));
            assert_eq!(wasm.transport, "bounded_json");
            assert!(wasm
                .operations
                .iter()
                .all(|operation| operation.ends_with("Json")));
        }
        for protocol_name in ["repetition_unlikelihood", "semantic_review"] {
            let protocol = catalog
                .protocols
                .iter()
                .find(|protocol| protocol.name == protocol_name)
                .expect("protocol");
            assert!(protocol.clients[0].trusted_legacy_replay);
            assert!(protocol.clients[1].trusted_legacy_replay);
        }
    }

    #[test]
    fn catalog_artifacts_and_operations_do_not_overstate_rust_ownership() {
        let catalog = zspace_runtime_protocol_catalog().expect("catalog");
        let semantic_review = catalog
            .protocols
            .iter()
            .find(|protocol| protocol.name == "semantic_review")
            .expect("semantic review protocol");
        assert_eq!(
            semantic_review
                .artifacts
                .iter()
                .map(|artifact| (
                    artifact.name.as_str(),
                    artifact.discriminator_field.as_str(),
                ))
                .collect::<Vec<_>>(),
            vec![
                ("map_commitment", "schema"),
                ("packet_receipt", "kind"),
                ("draft_receipt", "kind"),
                ("unblind_report", "kind"),
            ]
        );
        let python = semantic_review
            .clients
            .iter()
            .find(|surface| surface.client == "python")
            .expect("Python surface");
        assert!(!python
            .operations
            .iter()
            .any(|operation| operation == "new_zspace_semantic_review_draft"));
    }

    #[test]
    fn catalogued_rust_operations_exist() {
        use crate::runtime::zspace_generation_evidence::{
            summarize_zspace_generation_evidence, validate_zspace_generation_evidence_value,
        };
        use crate::runtime::zspace_periodicity::{
            analyze_zspace_periodicity, validate_zspace_periodicity_value,
        };
        use crate::runtime::zspace_repetition_unlikelihood::{
            plan_zspace_repetition_unlikelihood, validate_zspace_repetition_unlikelihood_value,
            validate_zspace_repetition_unlikelihood_value_trusted_legacy_replay,
        };
        use crate::runtime::zspace_semantic_review::{
            seal_zspace_semantic_review_packet, summarize_zspace_semantic_review_draft,
            unblind_zspace_semantic_review, validate_zspace_semantic_review_draft_receipt_value,
            validate_zspace_semantic_review_draft_receipt_value_trusted_legacy_replay,
            validate_zspace_semantic_review_packet,
            validate_zspace_semantic_review_packet_receipt_value,
            validate_zspace_semantic_review_packet_receipt_value_trusted_legacy_replay,
            validate_zspace_semantic_review_packet_trusted_legacy_replay,
            validate_zspace_semantic_review_unblind_value,
            validate_zspace_semantic_review_unblind_value_trusted_legacy_replay,
            zspace_semantic_review_map_id, zspace_semantic_review_map_id_trusted_legacy_replay,
        };
        use crate::runtime::zspace_stochastic_schrodinger::{
            run_zspace_stochastic_schrodinger_forward, run_zspace_stochastic_schrodinger_vjp,
            validate_zspace_stochastic_schrodinger_forward_value,
            validate_zspace_stochastic_schrodinger_vjp_value,
        };

        let _ = summarize_zspace_generation_evidence;
        let _ = validate_zspace_generation_evidence_value;
        let _ = analyze_zspace_periodicity;
        let _ = validate_zspace_periodicity_value;
        let _ = run_zspace_stochastic_schrodinger_forward;
        let _ = validate_zspace_stochastic_schrodinger_forward_value;
        let _ = run_zspace_stochastic_schrodinger_vjp;
        let _ = validate_zspace_stochastic_schrodinger_vjp_value;
        let _ = plan_zspace_repetition_unlikelihood;
        let _ = validate_zspace_repetition_unlikelihood_value;
        let _ = validate_zspace_repetition_unlikelihood_value_trusted_legacy_replay;
        let _ = zspace_semantic_review_map_id;
        let _ = seal_zspace_semantic_review_packet;
        let _ = validate_zspace_semantic_review_packet;
        let _ = validate_zspace_semantic_review_packet_receipt_value;
        let _ = summarize_zspace_semantic_review_draft;
        let _ = validate_zspace_semantic_review_draft_receipt_value;
        let _ = unblind_zspace_semantic_review;
        let _ = validate_zspace_semantic_review_unblind_value;
        let _ = zspace_semantic_review_map_id_trusted_legacy_replay;
        let _ = validate_zspace_semantic_review_packet_trusted_legacy_replay;
        let _ = validate_zspace_semantic_review_packet_receipt_value_trusted_legacy_replay;
        let _ = validate_zspace_semantic_review_draft_receipt_value_trusted_legacy_replay;
        let _ = validate_zspace_semantic_review_unblind_value_trusted_legacy_replay;
    }

    #[test]
    fn validator_replays_only_the_current_exact_catalog() {
        let catalog = zspace_runtime_protocol_catalog().expect("catalog");
        let value = serde_json::to_value(&catalog).expect("catalog value");
        assert_eq!(
            validate_zspace_runtime_protocol_catalog_value(value.clone()).expect("valid catalog"),
            catalog
        );

        let mut tampered = value.clone();
        tampered["protocols"][0]["clients"][2]["operations"][0] = json!("browserOwnedSemantics");
        assert_eq!(
            validate_zspace_runtime_protocol_catalog_value(tampered),
            Err(ZSpaceRuntimeProtocolCatalogError::CatalogMismatch)
        );

        let mut extended = value;
        extended["browser_policy"] = json!("local");
        assert!(matches!(
            validate_zspace_runtime_protocol_catalog_value(extended),
            Err(ZSpaceRuntimeProtocolCatalogError::MalformedCatalog { .. })
        ));
    }
}
