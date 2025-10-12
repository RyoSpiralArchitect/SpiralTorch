use std::{fs, path::Path};

fn main() {
    println!("cargo:warning=");
    println!("cargo:warning=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
    println!("cargo:warning=");
    println!("cargo:warning=                    d8,                    d8b                                      d8b      ");
    println!("cargo:warning=                   `8P                     88P        d8P                           ?88      ");
    println!("cargo:warning=                                          d88      d888888P                          88b     ");
    println!("cargo:warning= .d888b,?88,.d88b,  88b  88bd88b d888b8b  888        ?88'   d8888b   88bd88b d8888b  888888b ");
    println!("cargo:warning= ?8b,   `?88'  ?88  88P  88P'  `d8P' ?88  ?88        88P   d8P' ?88  88P'  `d8P' `P  88P `?8b");
    println!("cargo:warning=   `?8b   88b  d8P d88  d88     88b  ,88b  88b       88b   88b  d88 d88     88b     d88   88P");
    println!("cargo:warning= `?888P'   888888P'd88' d88'     `?88P'`88b  88b      `?8b  `?8888P'd88'     `?888P'd88'   88b");
    println!("cargo:warning=          88P'                                                                               ");
    println!("cargo:warning=         d88                                                                                 ");
    println!("cargo:warning=         ?8P                                                                                 ");
    println!("cargo:warning=");
    println!("cargo:warning=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

    // Ensure subgroup-aware heuristics stub exists (can be overwritten by WASM tuner output)
    let heur = Path::new("src/backend/wgpu_heuristics.rs");
    if !heur.exists() {
        fs::write(heur, r#"
pub fn choose(_rows:u32, _cols:u32, _k:u32, _subgroup: bool) -> Option<(bool, u32, u32, u32)> {
    // (use_2ce, wg_size, k_lane, chunk_cols)
    None
}
"#).expect("write wgpu_heuristics.rs");
    }
}
