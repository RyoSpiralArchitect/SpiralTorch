// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Quick smoke demo for the fused WGPU LayerNorm kernels.

use st_tensor::{set_tensor_op_meta_observer, TensorOpMetaEvent};
use st_nn::Tensor;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("wgpu_dense_available={}", st_tensor::wgpu_dense::is_available());

    set_tensor_op_meta_observer(Some(Arc::new(|event: &TensorOpMetaEvent| {
        if event.op_name == "layer_norm" {
            println!("layer_norm_meta={}", event.data);
        }
    })));

    let rows = 1024usize;
    let cols = 768usize;
    let epsilon = 1e-5;

    let x = Tensor::random_uniform(rows, cols, -1.0, 1.0, Some(7))?;
    let residual = Tensor::random_uniform(rows, cols, -0.1, 0.1, Some(11))?;
    let gamma = Tensor::from_fn(1, cols, |_, _| 1.0)?;
    let beta = Tensor::zeros(1, cols)?;

    match st_tensor::wgpu_dense::layer_norm_affine(
        x.data(),
        gamma.data(),
        beta.data(),
        rows,
        cols,
        epsilon,
    ) {
        Ok(_) => println!("wgpu_dense_layer_norm_affine=ok"),
        Err(err) => println!("wgpu_dense_layer_norm_affine=err:{err}"),
    }

    let y = x.layer_norm_affine(&gamma, &beta, epsilon)?;
    let y2 = x.layer_norm_affine_add(&residual, &gamma, &beta, epsilon)?;

    println!("y_shape={:?}", y.shape());
    println!("y2_shape={:?}", y2.shape());
    println!("y_head={:?}", &y.data()[0..8.min(y.data().len())]);
    println!("y2_head={:?}", &y2.data()[0..8.min(y2.data().len())]);

    Ok(())
}
