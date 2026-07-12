// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

pub const HIP_KERNELS: &[&str] = &[
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandReport {
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

pub fn compile_kernels_with<F>(
    hipcc: &OsStr,
    out_dir: &Path,
    kernels: &[&str],
    mut execute: F,
) -> Result<Vec<PathBuf>, String>
where
    F: FnMut(&OsStr, &[OsString]) -> Result<CommandReport, String>,
{
    if kernels.is_empty() {
        return Err("no HIP kernels are configured".to_string());
    }

    let mut objects = Vec::with_capacity(kernels.len());
    for source in kernels {
        let stem = Path::new(source)
            .file_stem()
            .ok_or_else(|| format!("unable to derive an object name from HIP kernel `{source}`"))?;
        let mut object_name = stem.to_os_string();
        object_name.push(".o");
        let object = out_dir.join(object_name);
        let args = vec![
            OsString::from("-O3"),
            OsString::from("--std=c++17"),
            OsString::from("-ffast-math"),
            OsString::from("-fPIC"),
            OsString::from("-DNDEBUG"),
            OsString::from("-c"),
            OsString::from(source),
            OsString::from("-o"),
            object.as_os_str().to_os_string(),
        ];
        let report = execute(hipcc, &args).map_err(|err| {
            format!(
                "failed to run HIP compiler `{}` for `{source}`: {err}",
                hipcc.to_string_lossy()
            )
        })?;
        if !report.success {
            return Err(command_failure("HIP compiler", hipcc, source, &report));
        }
        objects.push(object);
    }
    Ok(objects)
}

pub fn archive_kernels_with<F>(
    archiver: &OsStr,
    archive: &Path,
    objects: &[PathBuf],
    mut execute: F,
) -> Result<(), String>
where
    F: FnMut(&OsStr, &[OsString]) -> Result<CommandReport, String>,
{
    if objects.is_empty() {
        return Err("refusing to create an empty HIP kernel archive".to_string());
    }

    let mut args = Vec::with_capacity(objects.len() + 2);
    args.push(OsString::from("crus"));
    args.push(archive.as_os_str().to_os_string());
    args.extend(
        objects
            .iter()
            .map(|object| object.as_os_str().to_os_string()),
    );
    let report = execute(archiver, &args).map_err(|err| {
        format!(
            "failed to run HIP kernel archiver `{}`: {err}",
            archiver.to_string_lossy()
        )
    })?;
    if !report.success {
        return Err(command_failure(
            "HIP kernel archiver",
            archiver,
            &archive.display().to_string(),
            &report,
        ));
    }
    Ok(())
}

fn command_failure(kind: &str, program: &OsStr, target: &str, report: &CommandReport) -> String {
    let status = report
        .code
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".to_string());
    let mut message = format!(
        "{kind} `{}` failed for `{target}` with exit status {status}",
        program.to_string_lossy()
    );
    let stdout = report.stdout.trim();
    if !stdout.is_empty() {
        message.push_str("\nstdout:\n");
        message.push_str(stdout);
    }
    let stderr = report.stderr.trim();
    if !stderr.is_empty() {
        message.push_str("\nstderr:\n");
        message.push_str(stderr);
    }
    message
}
