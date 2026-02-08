// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use chrono::{SecondsFormat, Utc};
use serde_json::json;
use sha2::{Digest, Sha256};
use uuid::Uuid;

fn main() {
    emit_build_info();
    verify_wgpu_heuristics_generated();
    build_hip_rankk();
}

fn emit_build_info() {
    let user = env::var("USER")
        .or_else(|_| env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".into());
    let host = env::var("HOSTNAME")
        .or_else(|_| env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown-host".into());
    let timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
    let build_seed = Uuid::new_v4().simple().to_string();

    let git_commit = git_capture(["rev-parse", "HEAD"]).map(|s| s.to_string());
    let git_describe = git_capture(["describe", "--tags", "--always"]).map(|s| s.to_string());
    let git_dirty = git_status_is_dirty();

    let manifest = json!({
        "id": format!("Ryo-ST-{user}-{timestamp}-{build_seed}"),
        "timestamp": timestamp,
        "user": user,
        "host": host,
        "profile": env::var("PROFILE").ok(),
        "pkg": {
            "name": env::var("CARGO_PKG_NAME").ok(),
            "version": env::var("CARGO_PKG_VERSION").ok(),
        },
        "target": {
            "triple": env::var("TARGET").ok(),
            "arch": env::var("CARGO_CFG_TARGET_ARCH").ok(),
            "os": env::var("CARGO_CFG_TARGET_OS").ok(),
        },
        "rustc": detect_rustc_version(),
        "git": {
            "commit": git_commit,
            "describe": git_describe,
            "dirty": git_dirty,
        },
        "seed": build_seed,
    });

    let build_id = manifest
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("Ryo-ST-unknown");
    let manifest_json = manifest.to_string();

    let mut hasher = Sha256::new();
    hasher.update(manifest_json.as_bytes());
    let fingerprint = format!("sha256:{:x}", hasher.finalize());

    let build_info = format!(
        "pub const BUILD_ID: &str = {build_id:?};\n\
         pub const BUILD_MANIFEST_JSON: &str = r#\"{manifest_json}\"#;\n\
         pub const BUILD_FINGERPRINT: &str = {fingerprint:?};\n"
    );

    let out_dir = match env::var("OUT_DIR") {
        Ok(dir) => dir,
        Err(err) => {
            println!("cargo:warning=st-core: failed to read OUT_DIR for build info: {err}");
            return;
        }
    };

    let dest_path = Path::new(&out_dir).join("build_info.rs");
    if let Err(err) = fs::write(&dest_path, build_info) {
        println!(
            "cargo:warning=st-core: failed to write {}: {err}",
            dest_path.display()
        );
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn git_capture<const N: usize>(args: [&'static str; N]) -> Option<String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let output = Command::new("git")
        .args(args)
        .current_dir(manifest_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn git_status_is_dirty() -> Option<bool> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(manifest_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    Some(stdout.lines().any(|line| !line.trim().is_empty()))
}

fn detect_rustc_version() -> Option<String> {
    let output = Command::new("rustc")
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn verify_wgpu_heuristics_generated() {
    let gen = Path::new("src/backend/wgpu_heuristics_generated.rs");
    println!("cargo:rerun-if-changed={}", gen.display());
    if !gen.exists() {
        panic!(
            "missing {}; commit generated heuristics instead of relying on build-time stubs",
            gen.display()
        );
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
