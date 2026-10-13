//! Same large-cotangent regression on native WGPU and actual browser WebGPU.
use super::*;
use st_kernel_contracts::fisher_rao::{FisherRaoBiasForward, FisherRaoBiasSpec};

pub(super) async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let spec = FisherRaoBiasSpec::new([1, 2, 2], 1)?;
    let logits = [0.5, -0.5, -0.5, 0.5];
    let seed = [0., 0., 3e38, 0.];
    let cpu = FisherRaoBiasForward::new(spec, &logits, &[0.])?.backward(&seed)?;
    let device = TensorDevice::new(runtime)?;
    let forward = device
        .upload(&spec.shape(), &logits)?
        .causal_fisher_rao_bias(&device.upload(&[1], &[0.])?)?;
    let gradient = forward.backward(&device.upload(&spec.score_shape(), &seed)?)?;
    let coordinates = read(gradient.coordinates()).await?;
    let raw_gain = read(gradient.raw_gain()).await?;
    close(&coordinates, &cpu.coordinates, "wide Fisher logit VJP")?;
    close(&raw_gain, &cpu.raw_gain, "wide Fisher gain VJP")?;
    Ok(json!({
        "schema": "spiraltorch.fisher_rao_wide_pullback.v1",
        "coordinates": coordinates, "raw_gain": raw_gain
    }))
}
