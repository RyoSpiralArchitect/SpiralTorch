use st_core::backend::device_caps::DeviceCaps;
use st_core::plugin::{global_registry, PluginEvent};
use st_nn::{
    EpochStats, Linear, MeanSquaredError, Module, ModuleTrainer, Relu, RoundtableConfig, Sequential,
    Tensor,
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional: observe epoch-level progress via the plugin event bus.
    global_registry().event_bus().subscribe(
        "EpochEnd",
        Arc::new(|event| {
            if let PluginEvent::EpochEnd { epoch, loss } = event {
                println!("[epoch={epoch}] avg_loss={loss:.6}");
            }
        }),
    );

    let batch = 8u32;
    let in_dim = 2usize;
    let hidden = 8usize;
    let out_dim = 1u32;

    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 1e-2, 1e-2);
    let schedule = trainer.roundtable(
        batch,
        out_dim,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let mut model = Sequential::new();
    model.push(Linear::new("l1", in_dim, hidden)?);
    model.push(Relu::new());
    model.push(Linear::new("l2", hidden, out_dim as usize)?);
    model.attach_hypergrad(-1.0, 1e-2)?;

    let mut loss = MeanSquaredError::new();

    let x = Tensor::random_uniform(batch as usize, in_dim, -1.0, 1.0, Some(1))?;
    let y = Tensor::from_fn(batch as usize, 1, |r, _| {
        // Deterministic "target" built from the same x() formula.
        let x0 = x.data()[r * in_dim];
        let x1 = x.data()[r * in_dim + 1];
        0.7 * x0 - 0.2 * x1
    })?;

    for _ in 0..3 {
        let EpochStats { batches, average_loss, .. } =
            trainer.train_epoch(&mut model, &mut loss, vec![(x.clone(), y.clone())], &schedule)?;
        println!("stats: batches={batches} avg_loss={average_loss:.6}");
    }

    Ok(())
}
