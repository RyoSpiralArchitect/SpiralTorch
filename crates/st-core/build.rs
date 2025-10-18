// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    ensure_wgpu_stub();
    build_hip_rankk();
}

fn ensure_wgpu_stub() {
    let gen = Path::new("src/backend/wgpu_heuristics_generated.rs");
    if !gen.exists() {
        let stub = r#"
#[allow(unused)]
pub fn choose(_rows: usize, _cols: usize, _k: usize, _subgroup: bool) -> Option<super::Choice> {
    None
}
"#;
        fs::create_dir_all("src/backend").ok();
        fs::write(gen, stub).expect("write stub heuristics_generated");
        println!("cargo:warning=st-core: wrote stub backend/wgpu_heuristics_generated.rs");
    }
}

fn build_hip_rankk() {
    if env::var("CARGO_FEATURE_HIP_REAL").is_err() {
        return;
    }

    let src = Path::new("src/backend/hip_topk_rankk.hip.cpp");
    if !src.exists() {
        return;
    }

    println!("cargo:rerun-if-changed={}", src.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
    let obj = out_dir.join("hip_topk_rankk.o");

    let status = Command::new(&hipcc)
        .args([
            "-O3",
            "--std=c++17",
            "-ffast-math",
            "-fPIC",
            "-DNDEBUG",
            "-c",
        ])
        .arg(src)
        .arg("-o")
        .arg(&obj)
        .status();

    let status = match status {
        Ok(status) => status,
        Err(err) => {
            println!("cargo:warning=hipcc invocation failed: {err}");
            return;
        }
    };

    if !status.success() {
        println!("cargo:warning=hipcc failed to compile hip_topk_rankk.hip.cpp");
        return;
    }

    let lib = out_dir.join("libsthiprankk.a");
    let ar_status = Command::new("ar").arg("crus").arg(&lib).arg(&obj).status();

    match ar_status {
        Ok(status) if status.success() => {
            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=sthiprankk");

            if let Ok(rocm) = env::var("ROCM_PATH").or_else(|_| env::var("HIP_PATH")) {
                println!("cargo:rustc-link-search=native={}/lib", rocm);
                println!("cargo:rustc-link-search=native={}/lib64", rocm);
            }

            println!("cargo:rustc-link-lib=dylib=amdhip64");
        }
        Ok(_) => {
            println!("cargo:warning=ar failed while creating libsthiprankk.a");
        }
        Err(err) => {
            println!("cargo:warning=failed to invoke ar: {err}");
        }
    }
}
