use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::Module;
use st_tensor::{Layout, Tensor};
use st_vision::datasets::{MultiViewDatasetAdapter, MultiViewFrame, RayBatch};
use st_vision::nerf::{NerfField, NerfFieldConfig, NerfTrainer, NerfTrainingConfig};
use wasm_bindgen::prelude::*;

fn parameters(field: &NerfField) -> serde_json::Value {
    let mut result = serde_json::Map::new();
    field
        .visit_parameters(&mut |p| {
            result.insert(
                p.name().to_string(),
                serde_json::json!({
                    "shape": p.value().shape(), "values": p.value().data(),
                }),
            );
            Ok(())
        })
        .unwrap();
    result.into()
}

#[wasm_bindgen]
pub struct NerfCase {
    trainer: NerfTrainer,
    rays: RayBatch,
    metadata: String,
}

#[wasm_bindgen]
impl NerfCase {
    #[wasm_bindgen(constructor)]
    pub fn new(batch: usize, samples: usize, varying: bool, column_major: bool) -> Self {
        assert!(batch > 0 && batch <= 256 && samples > 0 && samples <= 1024);
        let mut field = NerfField::new(NerfFieldConfig {
            position_frequencies: 2,
            direction_frequencies: 1,
            hidden_layers: 1,
            hidden_width: 16,
            feature_dim: 8,
            color_layers: 1,
            color_hidden_width: 8,
            ..NerfFieldConfig::default()
        })
        .unwrap();
        field
            .visit_parameters_mut(&mut |p| {
                for (i, x) in p.value_mut().data_mut().iter_mut().enumerate() {
                    *x = if varying {
                        (i as i32 % 11 - 5) as f32 * 0.02
                    } else {
                        0.0
                    };
                }
                match p.name() {
                    "density::bias" => p.value_mut().data_mut()[0] = 2.0,
                    "color_out::bias" => p.value_mut().data_mut().copy_from_slice(&[0.4, 0.2, 0.1]),
                    _ => {}
                }
                Ok(())
            })
            .unwrap();
        let mut rays = RayBatch {
            origins: Tensor::from_vec(
                batch,
                3,
                (0..batch * 3)
                    .map(|i| (i as i32 % 17 - 8) as f32 * 0.025)
                    .collect(),
            )
            .unwrap(),
            directions: Tensor::from_vec(
                batch,
                3,
                (0..batch * 3)
                    .map(|i| (i as i32 % 7 - 3) as f32 * 0.1)
                    .collect(),
            )
            .unwrap(),
            colors: Tensor::from_vec(batch, 3, [0.1, 0.05, 0.02].repeat(batch)).unwrap(),
            bounds: Tensor::from_vec(
                batch,
                2,
                (0..batch)
                    .flat_map(|i| {
                        let near = 0.05 * (i % 3) as f32;
                        [near, near + 0.8 + 0.05 * (i % 5) as f32]
                    })
                    .collect(),
            )
            .unwrap(),
        };
        let metadata = serde_json::json!({"batch": batch, "samples": samples, "varying": varying,
            "origins": rays.origins.data(), "directions": rays.directions.data(),
            "bounds": rays.bounds.data(), "targets": rays.colors.data(),
            "parameters": parameters(&field),
            "position_bands": 2, "direction_bands": 1, "learning_rate": 0.01f32,
        })
        .to_string();
        if column_major {
            for t in [
                &mut rays.origins,
                &mut rays.directions,
                &mut rays.colors,
                &mut rays.bounds,
            ] {
                *t = t.to_layout(Layout::ColMajor).unwrap();
            }
        }
        let trainer = NerfTrainer::new(
            field,
            NerfTrainingConfig {
                samples_per_ray: samples,
                batch_size: batch,
                stratified: false,
                learning_rate: 0.01,
                ..NerfTrainingConfig::default()
            },
        )
        .unwrap();
        Self {
            trainer,
            rays,
            metadata,
        }
    }

    pub fn metadata(&self) -> String {
        self.metadata.clone()
    }

    pub fn run(&mut self) -> Rendered {
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        Rendered {
            tensor: self.trainer.render_batch(&self.rays).unwrap(),
        }
    }
}

#[wasm_bindgen]
pub struct Rendered {
    tensor: Tensor,
}

#[wasm_bindgen]
impl Rendered {
    pub fn values(&self) -> Vec<f32> {
        self.tensor.data().to_vec()
    }
}

#[wasm_bindgen]
pub fn contract_report() -> String {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut cases = Vec::new();
    for samples in [1, 8, 64] {
        for width in [0.0f32, 1e-8, 1.0, 20.0] {
            let mut case = NerfCase::new(1, samples, false, false);
            case.rays.bounds.data_mut().copy_from_slice(&[0.0, width]);
            let actual = case.run().values();
            let expected: Vec<_> = [0.4f32, 0.2, 0.1]
                .iter()
                .map(|&c| f64::from(c) * -(-2.0 * f64::from(width)).exp_m1())
                .collect();
            cases.push(serde_json::json!({"samples": samples, "width": width,
                "actual": actual, "expected": expected}));
        }
    }
    let mut row = NerfCase::new(3, 8, true, false);
    let mut column = NerfCase::new(3, 8, true, true);
    let layout_equal = row.run().values() == column.run().values();
    let mut training = NerfCase::new(1, 8, true, false);
    let before = serde_json::from_str::<serde_json::Value>(&training.metadata()).unwrap();
    let data = MultiViewDatasetAdapter::new(vec![MultiViewFrame::new(
        training.rays.origins.clone(),
        training.rays.directions.clone(),
        training.rays.colors.clone(),
        training.rays.bounds.clone(),
    )
    .unwrap()])
    .unwrap();
    let stats = training.trainer.train_step(&data).unwrap();
    serde_json::json!({"constant_cases": cases, "layout_equal": layout_equal,
        "training": {"before": before, "after": parameters(training.trainer.field()),
            "loss": stats.loss, "avg_transmittance": stats.avg_transmittance}})
    .to_string()
}
