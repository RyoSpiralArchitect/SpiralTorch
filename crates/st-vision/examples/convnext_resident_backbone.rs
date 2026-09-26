// SPDX-License-Identifier: AGPL-3.0-or-later
// Part of SpiralTorch - Licensed under AGPL-3.0-or-later.

use st_backend_wgpu::resident_tensor::TensorDevice;
use st_backend_wgpu::runtime::ensure_default_runtime_blocking;
use st_backend_wgpu::transform::TransformDispatcher;
use st_nn::module::Module;
use st_vision::models::{ConvNeXtBackbone, ConvNeXtConfig};
use st_vision::{CenterCrop, ImageTensor, TransformOperation, TransformPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (runtime, _) = ensure_default_runtime_blocking("vision.convnext_resident_backbone")?;
    let device = TensorDevice::new(runtime)?;
    let mut pipeline = TransformPipeline::with_seed(17)
        .with_gpu_dispatcher(TransformDispatcher::new_default_gpu()?);
    pipeline.add(TransformOperation::CenterCrop(CenterCrop::new(4, 4)?));
    let images = (0..2)
        .map(|frame| {
            ImageTensor::new(
                3,
                6,
                6,
                (0..108)
                    .map(|index| ((index * 7 + frame * 11) % 53) as f32 / 52.0)
                    .collect(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let features = pipeline.apply_geometry_batch_resident(&images, &device)?;
    let backbone = ConvNeXtBackbone::new(ConvNeXtConfig {
        input_channels: 3,
        input_hw: (4, 4),
        stage_dims: vec![3, 4],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        curvature: -1.0,
        epsilon: 1e-6,
    })?;
    let output = backbone.forward_resident(&features)?;
    let shape = output.layout().shape().to_vec();
    let values = output.snapshot()?.read()?;
    println!("resident ConvNeXt backbone: shape={shape:?}, values={values:?}");
    Ok(())
}
