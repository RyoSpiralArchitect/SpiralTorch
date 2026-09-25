// SPDX-License-Identifier: AGPL-3.0-or-later

use st_nn::layers::linear::Linear;
use st_nn::loss::{CrossEntropyWithLogits, Loss};
use st_nn::module::Module;
use st_tensor::{PureResult, Tensor};
use st_vision::models::{ConvNeXtBackbone, ConvNeXtConfig};

fn loss(prediction: &Tensor, targets: &Tensor) -> PureResult<f32> {
    Ok(CrossEntropyWithLogits::default()
        .forward(prediction, targets)?
        .data()[0])
}

fn predict(backbone: &ConvNeXtBackbone, head: &Linear, images: &Tensor) -> PureResult<Tensor> {
    head.forward(&backbone.forward(images)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConvNeXtConfig {
        input_channels: 1,
        input_hw: (8, 8),
        stage_dims: vec![2, 4],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        ..Default::default()
    };
    let mut backbone = ConvNeXtBackbone::new(config)?;
    let (channels, (height, width)) = backbone.output_shape();
    let mut head = Linear::new("vision.classifier", channels * height * width, 2)?;
    let images = Tensor::from_fn(2, 64, |sample, pixel| {
        let row = pixel / 8;
        let col = pixel % 8;
        if (sample == 0 && col < 4) || (sample == 1 && row < 4) {
            1.0
        } else {
            0.0
        }
    })?;
    let targets = Tensor::from_vec(2, 1, vec![0.0, 1.0])?;

    let initial = loss(&predict(&backbone, &head, &images)?, &targets)?;
    for _ in 0..20 {
        let features = backbone.forward(&images)?;
        let prediction = head.forward(&features)?;
        let grad_prediction = CrossEntropyWithLogits::default().backward(&prediction, &targets)?;
        let grad_features = head.backward(&features, &grad_prediction)?;
        backbone.backward(&images, &grad_features)?;
        backbone.visit_parameters_mut(&mut |parameter| parameter.apply_step(0.01))?;
        head.visit_parameters_mut(&mut |parameter| parameter.apply_step(0.01))?;
    }
    let final_prediction = predict(&backbone, &head, &images)?;
    let final_loss = loss(&final_prediction, &targets)?;
    println!("two-image cross entropy: {initial:.6} -> {final_loss:.6}");
    if final_loss >= initial
        || final_prediction.data()[0] <= final_prediction.data()[1]
        || final_prediction.data()[3] <= final_prediction.data()[2]
    {
        return Err("training did not learn both image labels".into());
    }
    Ok(())
}
