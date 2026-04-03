// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Small demo for Z-Space tracing + visualization helpers.

use std::path::{Path, PathBuf};
use std::process::Command;

use st_nn::{
    coherence_relation_tensor, OpenCartesianTopos, PureResult, Tensor, ZSpaceCoherenceSequencer,
    ZSpaceTraceConfig,
};

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn exporter_script() -> PathBuf {
    if let Some(path) = std::env::var_os("SPIRAL_ZSPACE_TRACE_EXPORTER") {
        return PathBuf::from(path);
    }
    repo_root().join("examples/zspace_trace_export_artifacts.py")
}

fn export_trace_artifacts(
    trace_jsonl: &Path,
    trace_html: &Path,
    atlas_html: &Path,
    manifest: &Path,
) {
    if env_flag("SPIRAL_ZSPACE_TRACE_SKIP_EXPORT") {
        println!("artifact_export=skipped");
        return;
    }

    let helper = exporter_script();
    if !helper.is_file() {
        println!("artifact_export=missing_helper helper={}", helper.display());
        return;
    }

    let status = Command::new("python3")
        .arg("-S")
        .arg("-s")
        .arg(&helper)
        .arg(trace_jsonl)
        .arg("--trace-html")
        .arg(trace_html)
        .arg("--atlas-html")
        .arg(atlas_html)
        .arg("--manifest")
        .arg(manifest)
        .arg("--title")
        .arg("SpiralTorch Z-Space Trace")
        .env("PYTHONNOUSERSITE", "1")
        .status();

    match status {
        Ok(status) if status.success() => {
            println!(
                "artifact_export=ok helper={} trace_html={} atlas_noncollapse_html={} artifact_manifest={}",
                helper.display(),
                trace_html.display(),
                atlas_html.display(),
                manifest.display()
            );
        }
        Ok(status) => {
            println!(
                "artifact_export=failed helper={} status={status}",
                helper.display()
            );
        }
        Err(err) => {
            println!(
                "artifact_export=failed helper={} error={err}",
                helper.display()
            );
        }
    }
}

fn main() -> PureResult<()> {
    let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192)?;
    let mut seq = ZSpaceCoherenceSequencer::new(64, 8, -1.0, topos)?;

    let recorder = seq.install_trace_recorder(ZSpaceTraceConfig {
        capacity: 256,
        max_vector_len: 64,
        publish_plugin_events: false,
    });

    let x = Tensor::from_vec(1, 64, vec![0.05; 64])?;
    let (y, coherence, diagnostics) = seq.forward_with_diagnostics(&x)?;

    let relation = coherence_relation_tensor(&coherence)?;
    println!(
        "aggregated={:?} coherence_channels={} label={} relation={:?}",
        y.shape(),
        coherence.len(),
        diagnostics.observation().lift_to_label(),
        relation.shape()
    );

    let records = recorder.records();
    let noncollapse = records
        .iter()
        .find_map(|record| record.noncollapse.clone());
    println!(
        "trace_records={} noncollapse={:?}",
        records.len(),
        noncollapse
    );

    let path = std::env::temp_dir().join("spiraltorch_zspace_trace.jsonl");
    recorder.write_jsonl(&path)?;
    println!("trace_jsonl={}", path.display());

    let trace_html = std::env::temp_dir().join("spiraltorch_zspace_trace.html");
    let atlas_html = std::env::temp_dir().join("spiraltorch_zspace_trace.atlas_noncollapse.html");
    let manifest = std::env::temp_dir().join("spiraltorch_zspace_trace.artifacts.json");
    export_trace_artifacts(&path, &trace_html, &atlas_html, &manifest);
    Ok(())
}

