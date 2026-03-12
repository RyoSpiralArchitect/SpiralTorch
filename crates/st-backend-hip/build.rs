// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{env, path::PathBuf, process::Command};

fn main() {
    if env::var("CARGO_FEATURE_HIP_REAL").is_ok() {
        let out = match env::var("OUT_DIR") {
            Ok(out) => PathBuf::from(out),
            Err(err) => {
                println!("cargo:warning=missing OUT_DIR for HIP build: {err}");
                return;
            }
        };
        if let Ok(rocm) = env::var("ROCM_PATH").or_else(|_| env::var("HIP_PATH")) {
            println!("cargo:rustc-link-search=native={}/lib", rocm);
            println!("cargo:rustc-link-search=native={}/lib64", rocm);
        }
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rccl");

        let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
        let kernels = [
            "src/hip_kernels/topk_pass1.cu",
            "src/hip_kernels/hip_kway_merge_bitonic_f32.cu",
            "src/hip_kernels/hip_kway_merge_bitonic_u64.cu",
            "src/hip_kernels/hip_kway_merge_shared_heap_real_keepk_u64.cu",
            "src/hip_kernels/hip_kway_merge_warp_coop_keepk_u64.cu",
            "src/hip_kernels/hip_topk_tile_bitonic_u64.cu",
            "src/hip_kernels/pack_vals_idx_u64.cu",
            "src/hip_kernels/hip_compaction_1ce.cu",
            "src/hip_kernels/hip_compaction_scan.cu",
            "src/hip_kernels/hip_compaction_scan_pass.cu",
            "src/hip_kernels/hip_compaction_apply.cu",
            "src/hip_kernels/hip_compaction_apply_pass.cu",
        ];
        let mut objs = Vec::new();
        for src in kernels {
            let Some(stem) = std::path::Path::new(src).file_stem() else {
                println!("cargo:warning=unable to derive kernel stem from {}", src);
                continue;
            };
            let obj = out.join(format!("{}.o", stem.to_string_lossy()));
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
                .status();
            let st = match st {
                Ok(status) => status,
                Err(err) => {
                    println!("cargo:warning=failed to run hipcc for {}: {}", src, err);
                    continue;
                }
            };
            if st.success() {
                objs.push(obj);
            } else {
                println!("cargo:warning=hipcc failed for {}", src);
            }
        }
        if !objs.is_empty() {
            let lib = out.join("libsthipkernels.a");
            let ar_status = std::process::Command::new("ar")
                .arg("crus")
                .arg(&lib)
                .args(&objs)
                .status();
            match ar_status {
                Ok(status) if status.success() => {
                    println!("cargo:rustc-link-lib=static=sthipkernels");
                    println!("cargo:rustc-link-search=native={}", out.display());
                }
                Ok(_) => {
                    println!("cargo:warning=ar failed while archiving HIP kernels");
                }
                Err(err) => {
                    println!("cargo:warning=failed to run ar for HIP kernels: {}", err);
                }
            }
        }
    }
}
