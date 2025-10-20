// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use st_nn::io;
use st_nn::module::Module;
use st_tensor::Tensor;
use tempfile::tempdir;

use st_vision::models::{
    ConvNeXtBackbone, ConvNeXtConfig, ResNetBackbone, ResNetConfig, ViTBackbone, ViTConfig,
};

fn sample_input(channels: usize, hw: (usize, usize), seed: u64) -> Tensor {
    Tensor::random_normal(1, channels * hw.0 * hw.1, 0.0, 1.0, Some(seed)).unwrap()
}

#[test]
fn resnet_produces_expected_shape_and_state_roundtrip() {
    let mut config = ResNetConfig::default();
    config.input_hw = (32, 32);
    config.stage_channels = vec![32, 64];
    config.block_depths = vec![1, 1];
    config.use_max_pool = false;
    config.global_pool = true;
    let resnet = ResNetBackbone::new(config.clone()).unwrap();
    let input = sample_input(config.input_channels, config.input_hw, 11);
    let output = resnet.forward(&input).unwrap();
    assert_eq!(output.shape(), (1, resnet.output_features()));
    assert_eq!(resnet.stage_shapes().len(), config.stage_channels.len());

    let dir = tempdir().unwrap();
    let path = dir.path().join("resnet.bin");
    io::save_bincode(&resnet, &path).unwrap();

    let mut restored = ResNetBackbone::new(config).unwrap();
    restored.load_weights_bincode(&path).unwrap();
    assert_eq!(resnet.state_dict().unwrap(), restored.state_dict().unwrap());
}

#[test]
fn vit_cls_token_shape_and_json_reload() {
    let mut config = ViTConfig::default();
    config.image_hw = (32, 32);
    config.patch_size = (8, 8);
    config.in_channels = 3;
    config.embed_dim = 64;
    config.depth = 2;
    config.mlp_dim = 128;
    config.curvature = -0.5;
    let vit = ViTBackbone::new(config.clone()).unwrap();
    let input = sample_input(config.in_channels, config.image_hw, 21);
    let output = vit.forward(&input).unwrap();
    assert_eq!(output.shape(), (1, config.embed_dim));
    let expected_patches =
        (config.image_hw.0 / config.patch_size.0) * (config.image_hw.1 / config.patch_size.1);
    assert_eq!(vit.patches(), expected_patches);

    let dir = tempdir().unwrap();
    let path = dir.path().join("vit.json");
    io::save_json(&vit, &path).unwrap();

    let mut restored = ViTBackbone::new(config).unwrap();
    restored.load_weights_json(&path).unwrap();
    assert_eq!(vit.state_dict().unwrap(), restored.state_dict().unwrap());
}

#[test]
fn convnext_stage_shapes_are_consistent() {
    let mut config = ConvNeXtConfig::default();
    config.input_hw = (32, 32);
    config.stage_dims = vec![32, 64];
    config.stage_depths = vec![1, 1];
    config.patch_size = (4, 4);
    let convnext = ConvNeXtBackbone::new(config.clone()).unwrap();
    let input = sample_input(config.input_channels, config.input_hw, 7);
    let output = convnext.forward(&input).unwrap();
    let (channels, hw) = convnext.output_shape();
    assert_eq!(output.shape(), (1, channels * hw.0 * hw.1));
    assert_eq!(convnext.stage_shapes().len(), config.stage_dims.len());
}
