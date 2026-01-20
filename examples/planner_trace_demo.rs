// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! End-to-end trace demo for the "graph planner + cooperative scheduler" surfaces.
//!
//! Captures:
//! - tensor ops (via the plugin bridge),
//! - kernel/backend decisions (via `TensorOpMeta`),
//! - autopilot + blackcat decisions,
//! - roundtable schedule planning.

use st_core::backend::device_caps::DeviceCaps;
use st_core::plugin::{global_registry, init_plugin_system, PluginEventRecorder, PluginEventRecorderConfig};
use st_core::runtime::autopilot::{AutoConfig, Autopilot};
use st_core::runtime::blackcat::{bandit::SoftBanditMode, zmeta::ZMetaParams, BlackCatRuntime, ChoiceGroups};
use st_nn::{Linear, MeanSquaredError, ModuleTrainer, Relu, RoundtableConfig, Sequential, Tensor};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_plugin_system()?;
    let bus = global_registry().event_bus();
    let recorder = PluginEventRecorder::subscribe(
        bus.clone(),
        PluginEventRecorderConfig { capacity: 8192 },
    );

    // --- Kernel fusion targets (softmax + attention) ---
    let logits = Tensor::random_uniform(4, 16, -1.0, 1.0, Some(7))?;
    let _ = logits.row_softmax()?;
    let _ = logits.row_softmax_hardmax_spiral()?;

    let contexts = 1usize;
    let sequence = 4usize;
    let head_dim = 16usize;
    let scale = (head_dim as f32).sqrt().recip();
    let q = Tensor::random_uniform(contexts * sequence, head_dim, -0.5, 0.5, Some(11))?;
    let k = Tensor::random_uniform(contexts * sequence, head_dim, -0.5, 0.5, Some(13))?;
    let v = Tensor::random_uniform(contexts * sequence, head_dim, -0.5, 0.5, Some(17))?;
    let _ = q.scaled_dot_attention(&k, &v, contexts, sequence, scale)?;

    // --- Cooperative scheduling targets (autopilot + roundtable) ---
    let caps = DeviceCaps::cpu();
    let groups = ChoiceGroups {
        groups: HashMap::from([
            ("wg".to_string(), vec!["128".to_string(), "256".to_string()]),
            ("tile".to_string(), vec!["512".to_string(), "1024".to_string(), "2048".to_string()]),
        ]),
    };
    let runtime = BlackCatRuntime::new(ZMetaParams::default(), groups, 8, SoftBanditMode::TS, None);
    let autopilot = Autopilot::new(caps, AutoConfig::default(), runtime);

    let mut trainer = ModuleTrainer::new(caps, -1.0, 1e-2, 1e-2).with_autopilot(autopilot);
    let schedule = trainer.roundtable(4, 1, RoundtableConfig::default().with_top_k(1).with_mid_k(1).with_bottom_k(1));

    let mut model = Sequential::new();
    model.push(Linear::new("l1", 16, 8)?);
    model.push(Relu::new());
    model.push(Linear::new("l2", 8, 1)?);
    trainer.prepare(&mut model)?;

    let mut loss = MeanSquaredError::new();

    let x = Tensor::random_uniform(4, 16, -1.0, 1.0, Some(23))?;
    let y = Tensor::from_fn(4, 1, |r, _| x.data()[r * 16] * 0.3 - 0.1)?;
    let _ = trainer.train_epoch(&mut model, &mut loss, vec![(x, y)], &schedule)?;

    // --- Export trace ---
    let trace_path = std::env::temp_dir().join("spiraltorch_planner_trace.jsonl");
    recorder.write_jsonl(&trace_path)?;
    println!("trace_jsonl={}", trace_path.display());
    println!("\n--- mermaid (first 80 events) ---\n{}\n", recorder.to_mermaid_flowchart(80));

    Ok(())
}
