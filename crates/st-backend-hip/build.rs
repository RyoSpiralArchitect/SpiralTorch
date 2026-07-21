// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[path = "build_support.rs"]
mod build_support;

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::Command;

use build_support::{archive_kernels_with, compile_kernels_with, CommandReport, HIP_KERNELS};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support.rs");
    for kernel in HIP_KERNELS {
        println!("cargo:rerun-if-changed={kernel}");
    }
    for variable in [
        "HIPCC",
        "AR",
        "ROCM_PATH",
        "HIP_PATH",
        "SPIRALTORCH_HIP_TYPECHECK_ONLY",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    if std::env::var_os("CARGO_FEATURE_HIP_REAL").is_none() {
        return;
    }
    if typecheck_only() {
        println!(
            "cargo:warning=hip-real type-check mode: skipping native kernel compilation and ROCm link directives"
        );
        return;
    }
    if let Err(err) = build_real_backend() {
        panic!("hip-real native build failed: {err}");
    }
}

fn typecheck_only() -> bool {
    std::env::var("SPIRALTORCH_HIP_TYPECHECK_ONLY")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn build_real_backend() -> Result<(), String> {
    let out_dir = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| "Cargo did not provide OUT_DIR for the hip-real build".to_string())?;
    let hipcc = std::env::var_os("HIPCC").unwrap_or_else(|| OsString::from("hipcc"));
    let objects = compile_kernels_with(&hipcc, &out_dir, HIP_KERNELS, execute_command)?;

    let archive = out_dir.join("libsthipkernels.a");
    let archiver = std::env::var_os("AR").unwrap_or_else(|| OsString::from("ar"));
    archive_kernels_with(&archiver, &archive, &objects, execute_command)?;

    if let Some(rocm) = std::env::var_os("ROCM_PATH").or_else(|| std::env::var_os("HIP_PATH")) {
        let rocm = PathBuf::from(rocm);
        println!(
            "cargo:rustc-link-search=native={}",
            rocm.join("lib").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            rocm.join("lib64").display()
        );
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=sthipkernels");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rccl");
    Ok(())
}

fn execute_command(program: &OsStr, args: &[OsString]) -> Result<CommandReport, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| err.to_string())?;
    Ok(CommandReport {
        success: output.status.success(),
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}
