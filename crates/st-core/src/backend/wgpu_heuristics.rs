#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
}

#[allow(unused)]
pub fn choose(rows: usize, cols: usize, k: usize, subgroup: bool) -> Option<Choice> {
    // 1) SpiralK env (highest priority)
    #[cfg(feature="kdsl")]
    if let Ok(src) = std::env::var("SPIRAL_HEUR_K") {
        if let Ok(ch) = st_kdsl::eval_program(&src, st_kdsl::Ctx{ r: rows as u32, c: cols as u32, k: k as u32, sg: subgroup }) {
            return Some(Choice{ use_2ce: ch.use_2ce, wg: ch.wg, kl: ch.kl, ch: ch.ch });
        }
    }
    // 2) WASM tuner generated table
    if let Some(ch) = choose_generated(rows, cols, k, subgroup) { return Some(ch); }
    // 3) Fallback (signal None; caller applies safe defaults)
    None
}

include!("wgpu_heuristics_generated.rs");
