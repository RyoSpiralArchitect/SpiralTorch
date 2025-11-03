//! Intermediate representation for the SpiralK soft-logic DSL.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layout {
    NHWC,
    NCHW,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    WGPU,
    MPS,
    CUDA,
    CPU,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefractOpPolicy {
    pub op: String,
    pub flags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetSpec {
    Graph(String),
    Prsn(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefractBlock {
    pub name: String,
    pub target: TargetSpec,
    pub precision: Option<Precision>,
    pub layout: Option<Layout>,
    pub schedule: Option<String>,
    pub backend: Option<Backend>,
    pub policies: Vec<RefractOpPolicy>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncBlock {
    pub name: String,
    pub pairs: Vec<String>,
    pub tolerance: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeedbackBlock {
    pub name: String,
    pub export_path: String,
    pub metrics: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Document {
    pub refracts: Vec<RefractBlock>,
    pub syncs: Vec<SyncBlock>,
    pub feedbacks: Vec<FeedbackBlock>,
}
