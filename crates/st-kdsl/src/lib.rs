// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use thiserror::Error;

pub mod auto;
pub mod autotune_store;
pub mod cache_key;
pub mod ir;
mod program;
pub mod query;
pub mod registry;
pub mod self_mod;
pub mod tile;

pub use auto::{
    rewrite_with_ai, rewrite_with_wilson, should_rewrite, synthesize_program, wilson_lower_bound,
    AiHintGenerator, AiRewriteConfig, AiRewriteError, AiRewritePrompt, HeuristicHint,
    TemplateAiGenerator, WilsonMetrics,
};
pub use ir::{Expr, ScalarType, SubgroupModule, SubgroupOp, SubgroupStmt};
pub use program::Program;
pub use query::{compile as compile_query, Filter, OrderDirection, QueryPlan};
pub use registry::{
    AutotuneKey, AutotuneRegistry, DeviceProfile, KernelProfile, TelemetryLog, TelemetrySample,
    TelemetrySummary,
};
pub use self_mod::{
    ContextClusterSnapshot, DiversitySnapshot, HintQualitySnapshot, HintStatSnapshot,
    HintTransitionSnapshot, SelfRewriteEngine, SelfRewriteEvent,
};
pub use tile::{TileConfig, TileIter, TileKnowledge, TileTemplate, WeightedMetric};

#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx {
    pub r: u32,
    pub c: u32,
    pub k: u32,
    pub sg: bool,
    pub sgc: u32,
    pub kc: u32,
    /// Estimated number of FFT tiles along the column dimension.
    pub tile_cols: u32,
    /// Preferred radix picked by the tuner.
    pub radix: u32,
    /// Preferred number of segmented dispatches for ND kernels.
    pub segments: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Hard {
    pub use_2ce: Option<bool>,
    pub wg: Option<u32>,
    pub kl: Option<u32>,
    pub ch: Option<u32>,
    pub algo: Option<u8>,
    pub midk: Option<u8>,
    pub bottomk: Option<u8>,
    pub ctile: Option<u32>,
    pub tile_cols: Option<u32>,
    pub radix: Option<u32>,
    pub segments: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum SoftRule {
    U2 { val: bool, w: f32 },
    Wg { val: u32, w: f32 },
    Kl { val: u32, w: f32 },
    Ch { val: u32, w: f32 },
    Algo { val: u8, w: f32 },
    Midk { val: u8, w: f32 },
    Bottomk { val: u8, w: f32 },
    Ctile { val: u32, w: f32 },
    TileCols { val: u32, w: f32 },
    Radix { val: u32, w: f32 },
    Segments { val: u32, w: f32 },
}

pub struct Out {
    pub hard: Hard,
    pub soft: Vec<SoftRule>,
}

#[derive(Error, Debug)]
pub enum Err {
    #[error("parse error at pos {0}")]
    Parse(usize),
    #[error("token")]
    Tok,
}

/// Compiles a program string into a reusable [`Program`].
pub fn compile_program(src: &str) -> Result<Program, Err> {
    Program::parse(src)
}

/// Parses and evaluates a program for the provided context.
pub fn eval_program(src: &str, ctx: &Ctx) -> Result<Out, Err> {
    compile_program(src).map(|program| program.evaluate(ctx))
}
