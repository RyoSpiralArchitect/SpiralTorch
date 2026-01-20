// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Small demo for Z-Space tracing + visualization helpers.

use st_nn::{
    coherence_relation_tensor, OpenCartesianTopos, PureResult, Tensor, ZSpaceCoherenceSequencer,
    ZSpaceTraceConfig,
};

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

    let path = std::env::temp_dir().join("spiraltorch_zspace_trace.jsonl");
    recorder.write_jsonl(&path)?;
    println!("trace_jsonl={}", path.display());
    Ok(())
}

