// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use st_nn::TensorError;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const TEXT_EXT: &str = "txt";

fn is_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case(TEXT_EXT))
}

fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), TensorError> {
    let entries = std::fs::read_dir(dir).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    for entry in entries {
        let entry = entry.map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir(&path, out)?;
        } else if is_text_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

pub fn collect_text_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>, TensorError> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    for raw in paths {
        if !raw.exists() {
            return Err(TensorError::IoError {
                message: format!("path not found: {}", raw.to_string_lossy()),
            });
        }
        if raw.is_dir() {
            walk_dir(raw, &mut candidates)?;
        } else if is_text_file(raw.as_path()) {
            candidates.push(raw.clone());
        }
    }

    candidates.sort();

    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for candidate in candidates {
        let resolved = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        if seen.contains(&resolved) {
            continue;
        }
        seen.insert(resolved);
        out.push(candidate);
    }

    Ok(out)
}

pub fn read_text_lossy(path: &Path) -> Result<String, TensorError> {
    let bytes = std::fs::read(path).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

pub fn read_text_files_lossy(files: &[PathBuf]) -> Result<String, TensorError> {
    let mut out = String::new();
    for (idx, path) in files.iter().enumerate() {
        let part = read_text_lossy(path)?;
        if part.is_empty() {
            continue;
        }
        if idx > 0 && !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&part);
    }
    Ok(out)
}

pub fn write_data_files_manifest(path: &Path, files: &[PathBuf]) -> Result<(), TensorError> {
    let mut text = String::new();
    for file in files {
        text.push_str(&file.to_string_lossy());
        text.push('\n');
    }
    std::fs::write(path, text).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    Ok(())
}

