// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    if env::var("CARGO_FEATURE_HIP_REAL").is_ok() {
        if let Ok(rocm) = env::var("ROCM_PATH").or_else(|_| env::var("HIP_PATH")) {
            println!("cargo:rustc-link-search=native={}/lib", rocm);
            println!("cargo:rustc-link-search=native={}/lib64", rocm);
        }
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rccl");

        let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
        let kernels = [
            // existing (must already exist in tree)
            "src/hip_kernels/topk_pass1.cu",
            "src/hip_kernels/hip_kway_merge_bitonic_f32.cu",
            "src/hip_kernels/pack_vals_idx_u64.cu",
            "src/hip_kernels/hip_kway_merge_bitonic_u64.cu",
            "src/hip_kernels/hip_topk_tile_bitonic_u64.cu",
            // new keep‑k variants
            "src/hip_kernels/hip_kway_merge_shared_heap_keepk_u64.cu",
            "src/hip_kernels/hip_kway_merge_warp_heap_keepk_u64.cu",
            "src/hip_kernels/hip_kway_merge_shared_heap_real_keepk_u64.cu",
            "src/hip_kernels/hip_kway_merge_warp_coop_keepk_u64.cu",
        ];
        let mut objs = Vec::new();
        for src in kernels {
            let stem = std::path::Path::new(src)
                .file_stem()
                .unwrap()
                .to_string_lossy();
            let obj = out.join(format!("{}.o", stem));
            let st = Command::new(&hipcc)
                .args([
                    "-O3",
                    "--std=c++17",
                    "-ffast-math",
                    "-fPIC",
                    "-DNDEBUG",
                    "-c",
                    src,
                    "-o",
                ])
                .arg(&obj)
                .status()
                .expect("failed to run hipcc");
            if st.success() {
                objs.push(obj);
            } else {
                println!("cargo:warning=hipcc failed for {}", src);
            }
        }
        if !objs.is_empty() {
            let lib = out.join("libsthipkernels.a");
            let _ = std::process::Command::new("ar")
                .args(["crus", lib.to_str().unwrap()])
                .args(objs.iter().map(|p| p.to_str().unwrap()))
                .status();
            println!("cargo:rustc-link-lib=static=sthipkernels");
            println!("cargo:rustc-link-search=native={}", out.display());
        }
    }
}
