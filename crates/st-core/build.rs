// build.rs: ensure wgpu_heuristics_generated.rs exists (stub if not).

use std::{path::Path, fs};

fn main() {
    let gen = Path::new("src/backend/wgpu_heuristics_generated.rs");
    if !gen.exists() {
        let stub = r#"
// auto-generated stub (build.rs)
// return None => use SpiralK or fallback in wgpu_heuristics.rs
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg:  u32,
    pub kl:  u32,
    pub ch:  u32,
    pub mk:  u32,
    pub tile:u32,
    pub mkd: Option<u32>,
    pub ctile: Option<u32>,
    pub two_ce_hint: bool,
}

pub(super) fn choose_generated(
    _rows: usize, _cols: usize, _k: usize, _subgroup: bool
) -> Option<Choice> { None }
"#;
        fs::write(gen, stub).unwrap();
        println!("cargo:warning=st-core: wrote stub backend/wgpu_heuristics_generated.rs");
    }
}
