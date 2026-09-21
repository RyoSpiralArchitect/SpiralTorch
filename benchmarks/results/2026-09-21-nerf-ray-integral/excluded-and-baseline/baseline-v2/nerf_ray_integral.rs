#![cfg(feature = "nerf")]

use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::Module;
use st_tensor::{Layout, Tensor};
use st_vision::datasets::{MultiViewDatasetAdapter, MultiViewFrame, RayBatch};
use st_vision::nerf::{NerfField, NerfFieldConfig, NerfTrainer, NerfTrainingConfig};

fn field_config(dims: usize, direction_dims: usize) -> NerfFieldConfig {
    NerfFieldConfig {
        position_dims: dims,
        direction_dims,
        position_frequencies: 0,
        direction_frequencies: 0,
        hidden_layers: 0,
        feature_dim: 1,
        color_layers: 0,
        ..NerfFieldConfig::default()
    }
}

fn constant_field(dims: usize, direction_dims: usize, sigma: f32) -> NerfField {
    let mut field = NerfField::new(field_config(dims, direction_dims)).unwrap();
    field
        .visit_parameters_mut(&mut |p| {
            p.value_mut().data_mut().fill(0.0);
            match p.name() {
                "density::bias" => p.value_mut().data_mut()[0] = sigma,
                "color_out::bias" => p.value_mut().data_mut().copy_from_slice(&[0.4, 0.2, 0.1]),
                _ => {}
            }
            Ok(())
        })
        .unwrap();
    field
}

fn config(samples: usize, batch_size: usize, stratified: bool, seed: u64) -> NerfTrainingConfig {
    NerfTrainingConfig {
        samples_per_ray: samples,
        batch_size,
        stratified,
        seed,
        learning_rate: 1e-2,
        ..NerfTrainingConfig::default()
    }
}

fn batch(rows: usize, dims: usize, near: f32, far: f32) -> RayBatch {
    RayBatch {
        origins: Tensor::zeros(rows, dims).unwrap(),
        directions: Tensor::from_vec(rows, dims, vec![1.0; rows * dims]).unwrap(),
        colors: Tensor::zeros(rows, 3).unwrap(),
        bounds: Tensor::from_vec(rows, 2, [near, far].repeat(rows)).unwrap(),
    }
}

fn assert_close(actual: f32, expected: f64, relative: f64) {
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() <= relative * expected.abs(),
        "actual={actual:.12e}, expected={expected:.12e}"
    );
}

#[test]
fn constant_density_integrates_the_entire_interval_independent_of_jitter() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    for samples in [1, 8, 64] {
        for stratified in [false, true] {
            for seed in [0, 7, 13] {
                let mut trainer = NerfTrainer::new(
                    constant_field(3, 3, 2.0),
                    config(samples, 2, stratified, seed),
                )
                .unwrap();
                let rays = batch(2, 3, -0.5, 1.5);
                let rendered = trainer.render_batch(&rays).unwrap();
                for (i, &value) in rendered.data().iter().enumerate() {
                    let rgb = f64::from([0.4f32, 0.2, 0.1][i % 3]);
                    assert_close(value, rgb * -(-4.0f64).exp_m1(), 2e-6);
                }
            }
        }
    }
}

#[test]
fn thin_and_zero_width_intervals_do_not_invent_opacity() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    for (near, far) in [(0.0f32, 0.0f32), (0.0, 1e-8), (1.0, 1.0 + f32::EPSILON)] {
        for samples in [1, 8, 1024] {
            let mut trainer =
                NerfTrainer::new(constant_field(3, 3, 2.0), config(samples, 1, true, 7)).unwrap();
            let rendered = trainer.render_batch(&batch(1, 3, near, far)).unwrap();
            let span = f64::from(far) - f64::from(near);
            for (&value, rgb) in rendered.data().iter().zip([0.4f32, 0.2, 0.1]) {
                assert_close(value, f64::from(rgb) * -(-2.0 * span).exp_m1(), 2e-6);
            }
        }
    }
}

#[test]
fn disabling_direction_conditioning_does_not_disable_ray_motion() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut field = constant_field(1, 0, 2.0);
    field
        .visit_parameters_mut(&mut |p| {
            match p.name() {
                "feature::weight" => p.value_mut().data_mut()[0] = 1.0,
                "color_out::weight" => p.value_mut().data_mut()[0] = 1.0,
                "color_out::bias" => p.value_mut().data_mut().fill(0.0),
                _ => {}
            }
            Ok(())
        })
        .unwrap();
    let mut rays = batch(1, 1, 0.0, 1.0);
    rays.directions.data_mut()[0] = 2.0;
    let mut trainer = NerfTrainer::new(field, config(1, 1, false, 13)).unwrap();
    let rendered = trainer.render_batch(&rays).unwrap();
    assert_close(rendered.data()[0], -(-2.0f64).exp_m1(), 2e-6);
    assert_eq!(&rendered.data()[1..], &[0.0, 0.0]);
}

#[test]
fn rendering_observes_logical_ray_layouts() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut rays = batch(2, 6, 0.25, 1.5);
    for (i, x) in rays.origins.data_mut().iter_mut().enumerate() {
        *x = i as f32 / 10.0;
    }
    for (i, x) in rays.directions.data_mut().iter_mut().enumerate() {
        *x = (i + 1) as f32 / 20.0;
    }
    rays.bounds.data_mut().copy_from_slice(&[0.25, 1.5, 0.5, 1.0]);
    let mut trainer = NerfTrainer::new(
        NerfField::new(field_config(6, 6)).unwrap(),
        config(8, 2, false, 13),
    )
    .unwrap();
    let expected = trainer.render_batch(&rays).unwrap();
    for layout in [Layout::ColMajor, Layout::Chimera { stripes: 2, tile: 3 }] {
        let altered = RayBatch {
            origins: rays.origins.to_layout(layout).unwrap(),
            directions: rays.directions.to_layout(layout).unwrap(),
            colors: rays.colors.to_layout(Layout::ColMajor).unwrap(),
            bounds: rays.bounds.to_layout(Layout::ColMajor).unwrap(),
        };
        assert_eq!(trainer.render_batch(&altered).unwrap().data(), expected.data());
    }
}

#[test]
fn unsupported_ray_layout_and_sample_overflow_are_rejected_at_construction() {
    assert!(NerfTrainer::new(
        NerfField::new(field_config(3, 2)).unwrap(),
        config(8, 1, false, 13)
    )
    .is_err());
    assert!(NerfTrainer::new(
        constant_field(3, 3, 2.0),
        config(usize::MAX, 2, false, 13)
    )
    .is_err());
}

#[test]
fn invalid_render_inputs_leave_the_sampling_rng_unchanged() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let valid = batch(2, 3, 0.0, 1.0);
    let mut invalid = Vec::new();
    for which in 0..4 {
        let mut rays = valid.clone();
        let tensor = match which {
            0 => &mut rays.origins,
            1 => &mut rays.directions,
            2 => &mut rays.colors,
            _ => &mut rays.bounds,
        };
        *tensor = Tensor::zeros(1, tensor.shape().1).unwrap();
        invalid.push(rays);
    }
    for which in 0..4 {
        let mut rays = valid.clone();
        let tensor = match which {
            0 => &mut rays.origins,
            1 => &mut rays.directions,
            2 => &mut rays.colors,
            _ => &mut rays.bounds,
        };
        tensor.data_mut()[0] = f32::NAN;
        invalid.push(rays);
    }
    let mut reversed = valid.clone();
    reversed.bounds.data_mut().copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    invalid.push(reversed);
    let mut oversized = valid.clone();
    oversized.origins.data_mut().fill(f32::MAX);
    oversized.directions.data_mut().fill(f32::MAX);
    oversized.bounds.data_mut().fill(2.0);
    invalid.push(oversized);
    for rays in invalid {
        let make = || {
            NerfTrainer::new(
                NerfField::new(field_config(3, 3)).unwrap(),
                config(8, 2, true, 7),
            )
            .unwrap()
        };
        let mut trainer = make();
        let mut control = make();
        assert!(trainer.render_batch(&rays).is_err());
        assert_eq!(
            trainer.render_batch(&valid).unwrap().data(),
            control.render_batch(&valid).unwrap().data()
        );
    }
}

#[test]
fn training_uses_the_same_full_interval_and_analytic_gradient() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut rays = batch(1, 3, 0.0, 1.0);
    let target = [0.1f32, 0.05, 0.02];
    rays.colors.data_mut().copy_from_slice(&target);
    let dataset = MultiViewDatasetAdapter::new(vec![MultiViewFrame::new(
        rays.origins,
        rays.directions,
        rays.colors,
        rays.bounds,
    )
    .unwrap()])
    .unwrap();
    let opacity = -(-2.0f64).exp_m1();
    let trans = (-2.0f64).exp();
    for samples in [1, 8, 64] {
        let mut trainer =
            NerfTrainer::new(constant_field(3, 3, 2.0), config(samples, 1, true, 13)).unwrap();
        let stats = trainer.train_step(&dataset).unwrap();
        let rgb = [0.4f32, 0.2, 0.1];
        let diff: Vec<f64> = rgb
            .iter()
            .zip(target)
            .map(|(&c, y)| f64::from(c) * opacity - f64::from(y))
            .collect();
        assert_close(stats.loss, diff.iter().map(|d| d * d).sum::<f64>() / 3.0, 2e-6);
        assert_close(stats.avg_transmittance, trans, 2e-6);
        let sigma_grad: f64 = diff.iter().zip(rgb).map(|(d, c)| 2.0 / 3.0 * d * f64::from(c) * trans).sum();
        trainer
            .field()
            .visit_parameters(&mut |p| {
                if p.name() == "density::bias" {
                    assert_close(p.value().data()[0], 2.0 - 0.01 * sigma_grad, 2e-6);
                }
                if p.name() == "color_out::bias" {
                    for (i, &c) in p.value().data().iter().enumerate() {
                        assert_close(c, f64::from(rgb[i]) - 0.01 * (2.0 / 3.0) * diff[i] * opacity, 2e-6);
                    }
                }
                Ok(())
            })
            .unwrap();
    }
}
