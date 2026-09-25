// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "nn")]

use st_nn::io;
use st_nn::layers::linear::Linear;
use st_nn::loss::{CrossEntropyWithLogits, Loss};
use st_nn::module::{Module, Parameter};
use st_nn::PureResult;
use st_tensor::Tensor;
use tempfile::tempdir;

use st_vision::models::{
    ConvNeXtBackbone, ConvNeXtConfig, ResNetBackbone, ResNetConfig, ResNetPreset, SkipSlipSchedule,
    ViTBackbone, ViTConfig,
};

fn sample_input(channels: usize, hw: (usize, usize), seed: u64) -> Tensor {
    Tensor::random_normal(1, channels * hw.0 * hw.1, 0.0, 1.0, Some(seed)).unwrap()
}

fn visit_skip_gate_parameters<M, F>(module: &M, mut callback: F) -> PureResult<()>
where
    M: Module + ?Sized,
    F: FnMut(&Parameter) -> PureResult<()>,
{
    let mut visitor = |param: &Parameter| -> PureResult<()> {
        if param.name().contains("skip_gate") {
            callback(param)?;
        }
        Ok(())
    };
    module.visit_parameters(&mut visitor)
}

#[test]
fn resnet_produces_expected_shape_and_state_roundtrip() {
    let config = ResNetConfig {
        input_hw: (32, 32),
        stage_channels: vec![32, 64],
        block_depths: vec![1, 1],
        use_max_pool: false,
        global_pool: true,
        ..Default::default()
    };
    let resnet = ResNetBackbone::new(config.clone()).unwrap();
    let input = sample_input(config.input_channels, config.input_hw, 11);
    let output = resnet.forward(&input).unwrap();
    assert_eq!(output.shape(), (1, resnet.output_features()));

    let dir = tempdir().unwrap();
    let path = dir.path().join("resnet.bin");
    io::save_bincode(&resnet, &path).unwrap();

    let mut restored = ResNetBackbone::new(config).unwrap();
    restored.load_weights_bincode(&path).unwrap();
    assert_eq!(resnet.state_dict().unwrap(), restored.state_dict().unwrap());
}

#[test]
fn resnet_config_auto_selects_common_presets() {
    let large = ResNetConfig::from_depth(101).expect("resnet101 preset");
    assert_eq!(large.block_depths, vec![3, 4, 23, 3]);
    assert_eq!(large.stage_channels, vec![64, 128, 256, 512]);
    assert_eq!(large.input_hw, (224, 224));
    assert!(large.use_max_pool);

    let cifar = ResNetConfig::from_depth(56).expect("resnet56 preset");
    assert_eq!(cifar.block_depths, vec![9, 9, 9]);
    assert_eq!(cifar.stage_channels, vec![16, 32, 64]);
    assert_eq!(cifar.input_hw, (32, 32));
    assert!(!cifar.use_max_pool);

    assert!(ResNetConfig::from_depth(42).is_err());
}

#[test]
fn resnet_backbone_builds_from_presets() {
    let resnet =
        ResNetBackbone::from_preset(ResNetPreset::ResNet56Cifar).expect("resnet56 backbone");
    let input = sample_input(3, (32, 32), 13);
    let output = resnet.forward(&input).unwrap();
    assert_eq!(output.shape(), (1, resnet.output_features()));
}

#[test]
fn resnet56_skip_gate_accumulates_gradients() {
    let mut config = ResNetConfig::resnet56_cifar(true);
    config.skip_init = 0.9;
    let mut resnet = ResNetBackbone::new(config.clone()).unwrap();
    let input = sample_input(config.input_channels, config.input_hw, 17);
    let output = resnet.forward(&input).unwrap();
    assert_eq!(output.shape(), (1, resnet.output_features()));

    let grad_output =
        Tensor::random_normal(1, resnet.output_features(), 0.0, 1.0, Some(23)).unwrap();
    let _ = resnet.backward(&input, &grad_output).unwrap();

    let mut skip_params = 0usize;
    visit_skip_gate_parameters(&resnet, |param| {
        skip_params += 1;
        let gradient = param.gradient().expect("skip gate gradient present");
        assert_eq!(gradient.shape(), (1, 1));
        Ok(())
    })
    .unwrap();
    assert!(skip_params > 0, "expected at least one learnable skip gate");
}

#[test]
fn resnet_skip_slip_scales_learnable_gate_gradients() {
    let base = ResNetConfig {
        input_hw: (8, 8),
        stage_channels: vec![8],
        block_depths: vec![1],
        stem_kernel: (3, 3),
        stem_stride: (1, 1),
        stem_padding: (1, 1),
        use_max_pool: false,
        global_pool: true,
        skip_learnable: true,
        skip_slip: None,
        ..Default::default()
    };

    let mut baseline = ResNetBackbone::new(base.clone()).unwrap();
    let mut slipped_cfg = base.clone();
    let slip_value = 0.4;
    let slip_schedule = SkipSlipSchedule::constant(slip_value);
    slipped_cfg.skip_slip = Some(slip_schedule.clone());
    let mut slipped = ResNetBackbone::new(slipped_cfg).unwrap();

    let dir = tempdir().unwrap();
    let path = dir.path().join("resnet_slip.bin");
    io::save_bincode(&baseline, &path).unwrap();
    slipped.load_weights_bincode(&path).unwrap();
    let mut dynamic = ResNetBackbone::new(base.clone()).unwrap();
    dynamic.load_weights_bincode(&path).unwrap();
    dynamic.set_skip_slip(Some(slip_schedule.clone())).unwrap();

    let input = sample_input(base.input_channels, base.input_hw, 31);
    let grad_output =
        Tensor::random_normal(1, baseline.output_features(), 0.0, 1.0, Some(41)).unwrap();

    let _ = baseline.forward(&input).unwrap();
    let _ = slipped.forward(&input).unwrap();
    let _ = dynamic.forward(&input).unwrap();
    let _ = baseline.backward(&input, &grad_output).unwrap();
    let _ = slipped.backward(&input, &grad_output).unwrap();
    let _ = dynamic.backward(&input, &grad_output).unwrap();

    let mut baseline_grads: Vec<f32> = Vec::new();
    visit_skip_gate_parameters(&baseline, |param| {
        let gradient = param.gradient().expect("baseline skip gradient");
        baseline_grads.push(gradient.data()[0]);
        Ok(())
    })
    .unwrap();

    let mut slipped_grads: Vec<f32> = Vec::new();
    visit_skip_gate_parameters(&slipped, |param| {
        let gradient = param.gradient().expect("slip skip gradient");
        slipped_grads.push(gradient.data()[0]);
        Ok(())
    })
    .unwrap();

    assert_eq!(baseline_grads.len(), slipped_grads.len());
    assert!(!baseline_grads.is_empty());
    for (&baseline_grad, &slipped_grad) in baseline_grads.iter().zip(slipped_grads.iter()) {
        assert!(baseline_grad.abs() > 0.0);
        let expected = baseline_grad * slip_value;
        let tolerance = baseline_grad.abs() * 1.0e-4 + 1.0e-6;
        assert!(
            (expected - slipped_grad).abs() <= tolerance,
            "expected slip-scaled gradient {expected}, got {slipped_grad}"
        );
    }

    let mut dynamic_grads: Vec<f32> = Vec::new();
    visit_skip_gate_parameters(&dynamic, |param| {
        let gradient = param.gradient().expect("dynamic skip gradient");
        dynamic_grads.push(gradient.data()[0]);
        Ok(())
    })
    .unwrap();

    assert_eq!(slipped_grads.len(), dynamic_grads.len());
    let slip_factors = dynamic.skip_slip_factors();
    for stage in &slip_factors {
        for &factor in stage {
            let tolerance = slip_value.abs() * 1.0e-6 + 1.0e-6;
            assert!((factor - slip_value).abs() <= tolerance);
        }
    }
    for (&slipped_grad, &dynamic_grad) in slipped_grads.iter().zip(dynamic_grads.iter()) {
        let tolerance = slipped_grad.abs() * 1.0e-5 + 1.0e-6;
        assert!((slipped_grad - dynamic_grad).abs() <= tolerance);
    }

    dynamic.set_skip_slip(None).unwrap();
    let reset_factors = dynamic.skip_slip_factors();
    for stage in reset_factors {
        for factor in stage {
            assert!((factor - 1.0).abs() <= 1.0e-6);
        }
    }

    let mut progressive_identity = ResNetBackbone::new(base.clone()).unwrap();
    progressive_identity.load_weights_bincode(&path).unwrap();
    progressive_identity
        .set_skip_slip_progress(Some(slip_schedule.clone()), 0.0)
        .unwrap();
    let identity_factors = progressive_identity.skip_slip_factors();
    for stage in &identity_factors {
        for &factor in stage {
            assert!((factor - 1.0).abs() <= 1.0e-6);
        }
    }
    let _ = progressive_identity.forward(&input).unwrap();
    let _ = progressive_identity.backward(&input, &grad_output).unwrap();
    let mut identity_grads: Vec<f32> = Vec::new();
    visit_skip_gate_parameters(&progressive_identity, |param| {
        let gradient = param.gradient().expect("identity blend gradient");
        identity_grads.push(gradient.data()[0]);
        Ok(())
    })
    .unwrap();
    for (&baseline_grad, &identity_grad) in baseline_grads.iter().zip(identity_grads.iter()) {
        let tolerance = baseline_grad.abs() * 1.0e-5 + 1.0e-6;
        assert!((baseline_grad - identity_grad).abs() <= tolerance);
    }

    let mut progressive_blend = ResNetBackbone::new(base.clone()).unwrap();
    progressive_blend.load_weights_bincode(&path).unwrap();
    let blend_progress = 0.5f32;
    progressive_blend
        .set_skip_slip_progress(Some(slip_schedule.clone()), blend_progress)
        .unwrap();
    let blended_factors = progressive_blend.skip_slip_factors();
    let expected_blended = 1.0 + (slip_value - 1.0) * blend_progress;
    for stage in &blended_factors {
        for &factor in stage {
            let tolerance = expected_blended.abs() * 1.0e-6 + 1.0e-6;
            assert!((factor - expected_blended).abs() <= tolerance);
        }
    }
    let _ = progressive_blend.forward(&input).unwrap();
    let _ = progressive_blend.backward(&input, &grad_output).unwrap();
    let mut blended_grads: Vec<f32> = Vec::new();
    visit_skip_gate_parameters(&progressive_blend, |param| {
        let gradient = param.gradient().expect("blended gradient");
        blended_grads.push(gradient.data()[0]);
        Ok(())
    })
    .unwrap();
    assert_eq!(baseline_grads.len(), blended_grads.len());
    for (&baseline_grad, &blended_grad) in baseline_grads.iter().zip(blended_grads.iter()) {
        let expected = baseline_grad * expected_blended;
        let tolerance = expected.abs() * 1.0e-4 + 1.0e-6;
        assert!((expected - blended_grad).abs() <= tolerance);
    }
}

#[test]
fn resnet_skip_slip_stage_progress_updates_per_stage() {
    let mut base = ResNetConfig::resnet56_cifar(true);
    base.skip_slip = None;
    let mut backbone = ResNetBackbone::new(base).unwrap();

    let schedule = SkipSlipSchedule::linear(0.2, 0.8)
        .with_power(1.3)
        .per_stage();
    let stage_progress = [0.0f32, 0.65f32];

    backbone
        .set_skip_slip_stage_progress(Some(schedule.clone()), &stage_progress)
        .unwrap();

    let runtime_factors = backbone.skip_slip_factors();
    let preview = schedule
        .preview_with_stage_identity_blend(backbone.block_depths(), &stage_progress)
        .unwrap();

    assert_eq!(runtime_factors.len(), preview.len());
    for (runtime_stage, preview_stage) in runtime_factors.iter().zip(preview.iter()) {
        assert_eq!(runtime_stage.len(), preview_stage.len());
        for (&runtime_value, &preview_value) in runtime_stage.iter().zip(preview_stage.iter()) {
            let tolerance = preview_value.abs() * 1.0e-5 + 1.0e-6;
            assert!((runtime_value - preview_value).abs() <= tolerance);
        }
    }
}

#[test]
fn resnet56_cifar_only_slips_learnable_skips() {
    let learnable = ResNetConfig::resnet56_cifar(true);
    assert!(learnable.skip_slip.is_some());

    let fixed = ResNetConfig::resnet56_cifar(false);
    assert!(fixed.skip_slip.is_none());
}

#[test]
fn skip_slip_schedule_preview_matches_formula() {
    let per_stage_schedule = SkipSlipSchedule::linear(0.25, 1.0)
        .with_power(2.0)
        .per_stage();
    let per_stage_depths = [3usize, 2];
    let preview = per_stage_schedule.preview(&per_stage_depths).unwrap();
    for (&depth, stage_preview) in per_stage_depths.iter().zip(preview.iter()) {
        assert_eq!(stage_preview.len(), depth);
        let denominator = if depth > 1 { (depth - 1) as f32 } else { 1.0 };
        for (block_idx, &value) in stage_preview.iter().enumerate() {
            let progress = if denominator <= f32::EPSILON {
                0.0
            } else {
                (block_idx as f32 / denominator).clamp(0.0, 1.0)
            };
            let target = 0.25 + (1.0 - 0.25) * progress.powf(2.0);
            let tolerance = target.abs() * 1.0e-5 + 1.0e-6;
            assert!((value - target).abs() <= tolerance);
        }
    }

    let global_schedule = SkipSlipSchedule::linear(0.1, 0.9).with_power(1.5);
    let global_depths = [2usize, 1, 2];
    let global_preview = global_schedule.preview(&global_depths).unwrap();
    let total_blocks: usize = global_depths.iter().sum();
    let mut global_index = 0usize;
    for (&depth, stage_preview) in global_depths.iter().zip(global_preview.iter()) {
        assert_eq!(stage_preview.len(), depth);
        for &value in stage_preview {
            let progress = if total_blocks > 1 {
                (global_index as f32) / ((total_blocks - 1) as f32)
            } else {
                0.0
            };
            let target = 0.1 + (0.9 - 0.1) * progress.powf(1.5);
            let tolerance = target.abs() * 1.0e-5 + 1.0e-6;
            assert!((value - target).abs() <= tolerance);
            global_index += 1;
        }
    }

    let identity_preview = per_stage_schedule
        .preview_with_identity_blend(&per_stage_depths, 0.0)
        .unwrap();
    for stage in identity_preview {
        for value in stage {
            assert!((value - 1.0).abs() <= 1.0e-6);
        }
    }

    let halfway = global_schedule
        .preview_with_identity_blend(&global_depths, 0.5)
        .unwrap();
    for (stage_idx, (&depth, stage_preview)) in global_depths.iter().zip(halfway.iter()).enumerate()
    {
        assert_eq!(stage_preview.len(), depth);
        for (block_idx, &value) in stage_preview.iter().enumerate() {
            let base = global_preview[stage_idx][block_idx];
            let expected = 1.0 + (base - 1.0) * 0.5;
            let tolerance = expected.abs() * 1.0e-5 + 1.0e-6;
            assert!((value - expected).abs() <= tolerance);
        }
    }

    let stage_progress = [0.0f32, 0.5f32];
    let per_stage_blend = per_stage_schedule
        .preview_with_stage_identity_blend(&per_stage_depths, &stage_progress)
        .unwrap();
    for (stage_idx, stage_preview) in per_stage_blend.iter().enumerate() {
        let expected_progress = if stage_idx < stage_progress.len() {
            stage_progress[stage_idx]
        } else {
            *stage_progress.last().expect("stage progress is non-empty")
        };
        for (block_idx, &value) in stage_preview.iter().enumerate() {
            let base = preview[stage_idx][block_idx];
            let expected = 1.0 + (base - 1.0) * expected_progress;
            let tolerance = expected.abs() * 1.0e-5 + 1.0e-6;
            assert!((value - expected).abs() <= tolerance);
        }
    }

    assert!(per_stage_schedule
        .preview_with_stage_identity_blend(&per_stage_depths, &[])
        .is_err());
}

#[test]
fn vit_cls_token_shape_and_json_reload() {
    let config = ViTConfig {
        image_hw: (32, 32),
        patch_size: (8, 8),
        in_channels: 3,
        embed_dim: 64,
        depth: 2,
        num_heads: 4,
        mlp_dim: 128,
        curvature: -0.5,
        ..Default::default()
    };
    let vit = ViTBackbone::new(config.clone()).unwrap();
    let input = sample_input(config.in_channels, config.image_hw, 21);
    let output = vit.forward(&input).unwrap();
    assert_eq!(output.shape(), (1, config.embed_dim));

    let dir = tempdir().unwrap();
    let path = dir.path().join("vit.json");
    io::save_json(&vit, &path).unwrap();

    let mut restored = ViTBackbone::new(config).unwrap();
    restored.load_weights_json(&path).unwrap();
    assert_eq!(vit.state_dict().unwrap(), restored.state_dict().unwrap());
}

#[test]
fn convnext_stage_shapes_are_consistent() {
    let config = ConvNeXtConfig {
        input_hw: (32, 32),
        stage_dims: vec![32, 64],
        stage_depths: vec![1, 1],
        patch_size: (4, 4),
        ..Default::default()
    };
    let convnext = ConvNeXtBackbone::new(config.clone()).unwrap();
    let input = sample_input(config.input_channels, config.input_hw, 7);
    let output = convnext.forward(&input).unwrap();
    let (channels, hw) = convnext.output_shape();
    assert_eq!(output.shape(), (1, channels * hw.0 * hw.1));
}

#[test]
fn convnext_rejects_empty_stages() {
    let config = ConvNeXtConfig {
        stage_dims: vec![],
        stage_depths: vec![],
        ..Default::default()
    };
    assert!(ConvNeXtBackbone::new(config).is_err());
}

#[test]
fn convnext_input_gradient_matches_finite_difference_through_downsample() {
    let config = ConvNeXtConfig {
        input_channels: 1,
        input_hw: (8, 8),
        stage_dims: vec![2, 4],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        ..Default::default()
    };
    let mut model = ConvNeXtBackbone::new(config).unwrap();
    let input = Tensor::random_normal(2, 64, 0.0, 0.4, Some(77)).unwrap();
    let output = model.forward(&input).unwrap();
    let grad_output = Tensor::random_normal(2, output.shape().1, 0.0, 0.2, Some(79)).unwrap();
    let analytical = model.backward(&input, &grad_output).unwrap();
    assert_eq!(analytical.shape(), input.shape());

    let epsilon = 1.0e-3;
    for index in [0, 9, 27, 45, 63, 64, 89, 127] {
        let mut plus = input.clone();
        plus.data_mut()[index] += epsilon;
        let mut minus = input.clone();
        minus.data_mut()[index] -= epsilon;
        let plus_out = model.forward(&plus).unwrap();
        let minus_out = model.forward(&minus).unwrap();
        let numerical = plus_out
            .data()
            .iter()
            .zip(minus_out.data())
            .zip(grad_output.data())
            .map(|((&hi, &lo), &grad)| (hi - lo) * grad)
            .sum::<f32>()
            / (2.0 * epsilon);
        let actual = analytical.data()[index];
        let tolerance = 0.01 + 0.05 * numerical.abs();
        assert!(
            (actual - numerical).abs() < tolerance,
            "input gradient {index}: analytical={actual}, numerical={numerical}"
        );
    }

    for (name, index) in [
        ("convnext.stem::weight", 0),
        ("convnext.stage0.block0.dw::weight", 24),
        ("convnext.stage0.block0.ln_gamma", 0),
        ("convnext.stage0.block0.fc1::weight", 0),
        ("convnext.stage0.downsample::weight", 0),
        ("convnext.final_norm_gamma", 0),
    ] {
        let mut analytical_parameter = None;
        model
            .visit_parameters(&mut |parameter| {
                if parameter.name() == name {
                    analytical_parameter = Some(parameter.gradient().unwrap().data()[index]);
                }
                Ok(())
            })
            .unwrap();
        let original = model.state_dict().unwrap()[name].data()[index];
        let mut objective = |value| {
            model
                .visit_parameters_mut(&mut |parameter| {
                    if parameter.name() == name {
                        parameter.value_mut().data_mut()[index] = value;
                    }
                    Ok(())
                })
                .unwrap();
            model
                .forward(&input)
                .unwrap()
                .data()
                .iter()
                .zip(grad_output.data())
                .map(|(&output, &grad)| output * grad)
                .sum::<f32>()
        };
        let plus = objective(original + epsilon);
        let minus = objective(original - epsilon);
        objective(original);
        let numerical = (plus - minus) / (2.0 * epsilon);
        let actual = analytical_parameter.unwrap();
        let tolerance = 0.01 + 0.05 * numerical.abs();
        assert!(
            (actual - numerical).abs() < tolerance,
            "parameter {name}[{index}]: analytical={actual}, numerical={numerical}"
        );
    }
}

#[test]
fn convnext_classifier_learns_from_image_batch() {
    let config = ConvNeXtConfig {
        input_channels: 1,
        input_hw: (8, 8),
        stage_dims: vec![2, 4],
        stage_depths: vec![1, 1],
        patch_size: (2, 2),
        ..Default::default()
    };
    let mut backbone = ConvNeXtBackbone::new(config.clone()).unwrap();
    let (channels, hw) = backbone.output_shape();
    let mut head = Linear::new("vision.classifier", channels * hw.0 * hw.1, 2).unwrap();
    let images = Tensor::from_fn(2, 64, |sample, pixel| {
        let row = pixel / 8;
        let col = pixel % 8;
        if (sample == 0 && col < 4) || (sample == 1 && row < 4) {
            1.0
        } else {
            0.0
        }
    })
    .unwrap();
    let targets = Tensor::from_vec(2, 1, vec![0.0, 1.0]).unwrap();
    let loss = |prediction: &Tensor| {
        CrossEntropyWithLogits::default()
            .forward(prediction, &targets)
            .unwrap()
            .data()[0]
    };
    let initial = loss(&head.forward(&backbone.forward(&images).unwrap()).unwrap());
    let initial_state = backbone.state_dict().unwrap();

    for _ in 0..20 {
        let features = backbone.forward(&images).unwrap();
        let prediction = head.forward(&features).unwrap();
        let grad_prediction = CrossEntropyWithLogits::default()
            .backward(&prediction, &targets)
            .unwrap();
        let grad_features = head.backward(&features, &grad_prediction).unwrap();
        let grad_images = backbone.backward(&images, &grad_features).unwrap();
        assert_eq!(grad_images.shape(), images.shape());
        assert!(grad_images.data().iter().all(|value| value.is_finite()));
        backbone
            .visit_parameters_mut(&mut |parameter| parameter.apply_step(0.01))
            .unwrap();
        head.visit_parameters_mut(&mut |parameter| parameter.apply_step(0.01))
            .unwrap();
    }

    let final_prediction = head.forward(&backbone.forward(&images).unwrap()).unwrap();
    let final_loss = loss(&final_prediction);
    assert!(
        final_loss < initial * 0.8,
        "classification loss did not improve: {initial} -> {final_loss}"
    );
    assert!(final_prediction.data()[0] > final_prediction.data()[1]);
    assert!(final_prediction.data()[3] > final_prediction.data()[2]);
    let final_state = backbone.state_dict().unwrap();
    for name in [
        "convnext.stem::weight",
        "convnext.stage0.block0.dw::weight",
        "convnext.stage0.block0.fc1::weight",
        "convnext.stage0.downsample::weight",
        "convnext.stage1.block0.dw::weight",
        "convnext.final_norm_gamma",
    ] {
        let before = initial_state.get(name).unwrap();
        let after = final_state.get(name).unwrap();
        assert!(
            before
                .data()
                .iter()
                .zip(after.data())
                .any(|(&old, &new)| (old - new).abs() > 1.0e-8),
            "{name} did not update"
        );
    }

    let dir = tempdir().unwrap();
    let path = dir.path().join("trained_convnext.bin");
    io::save_bincode(&backbone, &path).unwrap();
    let mut restored = ConvNeXtBackbone::new(config).unwrap();
    restored.load_weights_bincode(&path).unwrap();
    assert_eq!(
        backbone.forward(&images).unwrap().data(),
        restored.forward(&images).unwrap().data()
    );
}
