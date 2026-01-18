// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    HyperbolicCrossEntropy, Linear, Module, ModuleTrainer, RoundtableConfig, Scaler, Sequential,
    Tensor,
};
use st_nn::layers::{conv::Conv2d, NonLiner};

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

fn build_batch(
    batch: usize,
    input_hw: (usize, usize),
    seed: u64,
) -> st_nn::PureResult<(Tensor, Tensor)> {
    let cols = input_hw.0 * input_hw.1;
    let inputs = Tensor::random_uniform(batch, cols, 0.0, 1.0, Some(seed))?;
    let (rows, cols) = inputs.shape();
    let mut targets = Vec::with_capacity(rows * 2);
    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        let row = &inputs.data()[start..end];
        let mid = cols / 2;
        let left: f32 = row[..mid].iter().sum();
        let right: f32 = row[mid..].iter().sum();
        if left >= right {
            targets.extend_from_slice(&[1.0, 0.0]);
        } else {
            targets.extend_from_slice(&[0.0, 1.0]);
        }
    }
    let labels = Tensor::from_vec(rows, 2, targets)?;
    Ok((inputs, labels))
}

fn main() -> st_nn::PureResult<()> {
    let batch = 6;
    let input_hw = (4, 4);
    let kernel = (3, 3);
    let stride = (1, 1);
    let padding = (0, 0);
    let out_channels = 2;

    let out_hw = conv_output_hw(input_hw, kernel, stride, padding);
    let features = out_channels * out_hw.0 * out_hw.1;

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
    model.push(NonLiner::new("nl1", features)?);
    model.push(Scaler::new("sc1", features)?);
    model.push(Linear::new("head", features, 2)?);
    model.attach_hypergrad(-1.0, 1e-2)?;

    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 1e-2, 1e-2);
    let mut config = RoundtableConfig::default();
    config.top_k = 1;
    config.mid_k = 1;
    config.bottom_k = 1;
    config.here_tolerance = 1e-5;
    let schedule = trainer.roundtable(batch as u32, 2, config);

    let mut loss = HyperbolicCrossEntropy::new(-1.0)?;
    let (x, y) = build_batch(batch, input_hw, 11)?;
    let batches = vec![(x.clone(), y.clone())];

    for _ in 0..3 {
        let stats = trainer.train_epoch(&mut model, &mut loss, batches.clone(), &schedule)?;
        println!("stats: {:?}", stats);
    }

    Ok(())
}
