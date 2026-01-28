// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Vision model-zoo: Conv2d + MaxPool2d binary classification.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    load_json, save_json, Conv2d, EpochStats, HyperbolicCrossEntropy, Linear, MaxPool2d, Module,
    ModuleTrainer, Relu, RoundtableConfig, Sequential, Tensor,
};
use std::path::Path;

fn conv_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> (usize, usize) {
    let (h, w) = input_hw;
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    (oh, ow)
}

fn pool_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> (usize, usize) {
    let (h, w) = input_hw;
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    (oh, ow)
}

fn build_batch(
    batch: usize,
    hw: (usize, usize),
    base_seed: u64,
) -> st_nn::PureResult<(Tensor, Tensor)> {
    let pixels = hw.0 * hw.1;
    let mut data = Vec::with_capacity(batch * pixels);
    let mut targets = Vec::with_capacity(batch);

    for idx in 0..batch {
        let seed = base_seed.wrapping_add(idx as u64);
        let mut sample = Tensor::random_uniform(1, pixels, 0.0, 0.20, Some(seed))?
            .data()
            .to_vec();
        let class_one = idx % 2 == 1;
        if class_one {
            // Bright bottom-right quadrant
            for r in hw.0 / 2..hw.0 {
                for c in hw.1 / 2..hw.1 {
                    sample[r * hw.1 + c] += 0.9;
                }
            }
            targets.push(1.0);
        } else {
            // Bright top-left quadrant
            for r in 0..hw.0 / 2 {
                for c in 0..hw.1 / 2 {
                    sample[r * hw.1 + c] += 0.9;
                }
            }
            targets.push(0.0);
        }
        data.extend_from_slice(&sample);
    }

    let x = Tensor::from_vec(batch, pixels, data)?;
    let y = Tensor::from_vec(batch, 1, targets)?;
    Ok((x, y))
}

fn build_model(
    input_hw: (usize, usize),
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    pool_kernel: (usize, usize),
    pool_stride: (usize, usize),
) -> st_nn::PureResult<Sequential> {
    let conv_out_hw = conv_output_hw(input_hw, kernel, stride, padding);
    let pool_out_hw = pool_output_hw(conv_out_hw, pool_kernel, pool_stride, (0, 0));
    let pooled_features = out_channels * pool_out_hw.0 * pool_out_hw.1;

    let mut model = Sequential::new();
    model.push(Conv2d::new(
        "conv1",
        1,
        out_channels,
        kernel,
        stride,
        padding,
        (1, 1),
        input_hw,
    )?);
    model.push(Relu::new());
    model.push(MaxPool2d::new(
        out_channels,
        pool_kernel,
        pool_stride,
        (0, 0),
        conv_out_hw,
    )?);
    model.push(Relu::new());
    model.push(Linear::new("head", pooled_features, 1)?);
    Ok(model)
}

fn main() -> st_nn::PureResult<()> {
    let batch = 8usize;
    let input_hw = (8, 8);
    let kernel = (3, 3);
    let stride = (1, 1);
    let padding = (1, 1);
    let out_channels = 4usize;
    let pool_kernel = (2, 2);
    let pool_stride = (2, 2);

    let mut model = build_model(
        input_hw,
        out_channels,
        kernel,
        stride,
        padding,
        pool_kernel,
        pool_stride,
    )?;
    model.attach_hypergrad(-1.0, 2e-2)?;

    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 2e-2, 2e-2);
    let schedule = trainer.roundtable(
        batch as u32,
        1,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );

    let mut loss = HyperbolicCrossEntropy::new(-1.0)?;
    let (x, y) = build_batch(batch, input_hw, 777)?;
    let batches = vec![(x.clone(), y.clone()); 4];

    for _ in 0..8 {
        let EpochStats {
            batches,
            average_loss,
            ..
        } = trainer.train_epoch(&mut model, &mut loss, batches.clone(), &schedule)?;
        println!("stats: batches={batches} avg_loss={average_loss:.6}");
    }

    let weights_path = Path::new("models/weights/vision_conv_pool_classification.json");
    if let Some(parent) = weights_path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| st_nn::TensorError::IoError {
            message: err.to_string(),
        })?;
    }
    save_json(&model, weights_path)?;

    let mut reloaded = build_model(
        input_hw,
        out_channels,
        kernel,
        stride,
        padding,
        pool_kernel,
        pool_stride,
    )?;
    load_json(&mut reloaded, weights_path)?;
    let _ = reloaded.forward(&x)?;

    Ok(())
}
