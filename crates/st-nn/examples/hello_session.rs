use st_core::backend::device_caps::DeviceCaps;
use st_nn::{Linear, MeanSquaredError, RoundtableConfig, Sequential, SpiralSession, Tensor};
use st_tensor::pure::PureResult;

fn main() -> PureResult<()> {
    let caps = DeviceCaps::wgpu(32, true, 256);
    let session = SpiralSession::builder(caps)
        .with_curvature(-1.0)
        .with_hyper_learning_rate(0.05)
        .with_fallback_learning_rate(0.01)
        .build()?;

    let densities = vec![
        Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
        Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
    ];
    let weights = [0.5, 0.5];
    let barycenter = session.barycenter(&weights, &densities)?;

    let mut hypergrad = session.hypergrad(1, 2)?;
    session.align_hypergrad(&mut hypergrad, &barycenter)?;

    let mut model = Sequential::new();
    model.push(Linear::new("layer", 2, 2)?);
    session.prepare_module(&mut model)?;

    let mut trainer = session.trainer();
    let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());

    let dataset = vec![
        (
            Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
            Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
        ),
        (
            Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
            Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
        ),
    ];

    let mut loss = MeanSquaredError::new();
    let stats = trainer.train_epoch(&mut model, &mut loss, dataset, &schedule)?;
    println!(
        "roundtable avg loss {:.6} over {} batches",
        stats.average_loss, stats.batches
    );

    Ok(())
}
