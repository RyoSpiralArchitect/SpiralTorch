//! Torch-free execution of selected portable adapter placements.
//!
//! This is not a whole HF model loader. Selected placements share the native
//! kernels, loss and tensor primitives; updates replace a validated artifact
//! atomically. A snapshot is usable for updates only against its exact source.

use crate::execution::{push_backend_policy, BackendPolicy};
use crate::geometry_artifact::{
    GeometryAdapterArtifact, GeometryAdapterPlacement, GeometryAdapterRecipe,
    GeometryAdapterTensorRole,
};
use crate::loss::{Loss, MeanSquaredError};
use crate::{
    ChannelAffineKernel, ChannelAffineLearningBatch, WaveGateKernel, WaveGateLearningBatch,
};
use st_core::backend::device_caps::DeviceCaps;
use st_tensor::{Tensor, TensorUtilBackend};
use std::collections::BTreeMap;

mod elliptic;
#[cfg(test)]
mod sequence_tests;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug)]
enum Pullback {
    Identity,
    Affine(Box<ChannelAffineLearningBatch>),
    Wave(Box<WaveGateLearningBatch>),
    WaveRadius(Box<WaveGateLearningBatch>),
    Elliptic(Box<elliptic::EllipticRuntimeBatch>),
}

#[derive(Debug)]
pub struct NativeGeometryBatch {
    source_id: String,
    placement: GeometryAdapterPlacement,
    output: Tensor,
    strength: f32,
    pullback: Pullback,
    sequence_shape: Option<[usize; 3]>,
}

#[derive(Debug)]
pub struct NativeGeometryVjp {
    pub input: Tensor,
    pub parameters: BTreeMap<String, Vec<f32>>,
}

#[derive(Debug)]
pub struct NativeGeometryRuntime {
    artifact: GeometryAdapterArtifact,
}

fn tensor<'a>(placement: &'a GeometryAdapterPlacement, name: &str) -> &'a [f32] {
    // The only construction path owns an already validated artifact.
    &placement
        .tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap()
        .values
}

impl NativeGeometryRuntime {
    pub fn new(artifact: GeometryAdapterArtifact) -> Self {
        Self { artifact }
    }

    pub fn artifact(&self) -> &GeometryAdapterArtifact {
        &self.artifact
    }

    fn placement(&self, path: &str) -> Result<&GeometryAdapterPlacement> {
        self.artifact
            .record()
            .placements
            .iter()
            .find(|p| p.path == path)
            .ok_or_else(|| "unknown geometry placement".into())
    }

    /// Return the feature width and allocation budget before binding extraction.
    pub fn dimensions(&self, path: &str) -> Result<(usize, usize)> {
        match &self.placement(path)?.recipe {
            GeometryAdapterRecipe::ChannelAffine { features, max_values, .. } => Ok((*features, *max_values)),
            GeometryAdapterRecipe::Wave { features, kernel, .. }
            | GeometryAdapterRecipe::WaveRadius { features, kernel, .. } => Ok((*features, kernel.max_values)),
            GeometryAdapterRecipe::Elliptic { features, .. }
            | GeometryAdapterRecipe::EllipticAnchored { features, .. } => Ok((*features, elliptic::value_budget(*features))),
            _ => Err("native artifact runtime currently supports pointwise channel-affine, WaveGate and elliptic placements only".into()),
        }
    }

    /// Admit full, nonempty, unpadded contexts before copying foreign values.
    /// Returns (feature width, exact value count); causal pair budgets live in Rust.
    pub fn sequence_dimensions(&self, path: &str, shape: [usize; 2]) -> Result<(usize, usize)> {
        let rows = shape[0]
            .checked_mul(shape[1])
            .filter(|&n| n > 0)
            .ok_or("native sequence requires nonempty non-overflowing batch and sequence axes")?;
        match &self.placement(path)?.recipe {
            GeometryAdapterRecipe::EllipticCausal { features, .. }
            | GeometryAdapterRecipe::EllipticGated { features, .. } => {
                Ok((*features, elliptic::sequence_values(*features, shape)?))
            }
            _ => {
                let (features, budget) = self.dimensions(path)?;
                let count = rows
                    .checked_mul(features)
                    .filter(|&n| n <= budget)
                    .ok_or("native sequence exceeds value budget")?;
                Ok((features, count))
            }
        }
    }

    /// Full-prefix causal execution, or independent pointwise execution with
    /// the same explicit batch/sequence boundary. No padding mask or KV cache.
    pub fn forward_sequence(
        &self,
        path: &str,
        input: &[f32],
        shape: [usize; 2],
    ) -> Result<NativeGeometryBatch> {
        let (features, count) = self.sequence_dimensions(path, shape)?;
        if input.len() != count || input.iter().any(|x| !x.is_finite()) {
            return Err("invalid native sequence input length or values".into());
        }
        let placement = self.placement(path)?;
        let (warp, strength) = match &placement.recipe {
            GeometryAdapterRecipe::EllipticCausal { warp, strength, .. }
            | GeometryAdapterRecipe::EllipticGated { warp, strength, .. } => {
                (warp, *strength as f32)
            }
            _ => {
                let mut batch = self.forward(path, input, shape[0] * shape[1])?;
                batch.sequence_shape = Some([shape[0], shape[1], features]);
                return Ok(batch);
            }
        };
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let input = Tensor::from_vec(shape[0] * shape[1], features, input.to_vec())?;
        let pullback = if strength == 0. {
            Pullback::Identity
        } else {
            Pullback::Elliptic(Box::new(elliptic::EllipticRuntimeBatch::forward(
                placement,
                warp,
                &input,
                Some(shape),
            )?))
        };
        let mut batch = self.finish_forward(placement, input, strength, pullback)?;
        batch.sequence_shape = Some([shape[0], shape[1], features]);
        Ok(batch)
    }

    /// Flatten only leading axes; each row is an independent feature vector.
    pub fn forward(&self, path: &str, input: &[f32], rows: usize) -> Result<NativeGeometryBatch> {
        let (features, budget) = self.dimensions(path)?;
        if rows.checked_mul(features) != Some(input.len())
            || input.len() > budget
            || input.iter().any(|x| !x.is_finite())
        {
            return Err("invalid native geometry input shape, value or budget".into());
        }
        let placement = self.placement(path)?;
        let strength = match placement.recipe {
            GeometryAdapterRecipe::ChannelAffine { strength, .. }
            | GeometryAdapterRecipe::Wave { strength, .. }
            | GeometryAdapterRecipe::WaveRadius { strength, .. }
            | GeometryAdapterRecipe::Elliptic { strength, .. }
            | GeometryAdapterRecipe::EllipticAnchored { strength, .. } => strength as f32,
            _ => unreachable!(),
        };
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let input_tensor = Tensor::from_vec(rows, features, input.to_vec())?;
        let pullback = if strength == 0. {
            Pullback::Identity
        } else {
            match &placement.recipe {
                GeometryAdapterRecipe::ChannelAffine { max_values, .. } => {
                    Pullback::Affine(Box::new(ChannelAffineKernel::new(*max_values)?.forward(
                        input,
                        tensor(placement, "gate"),
                        tensor(placement, "bias"),
                        rows,
                        features,
                    )?))
                }
                GeometryAdapterRecipe::Wave { kernel, .. }
                | GeometryAdapterRecipe::WaveRadius { kernel, .. } => {
                    let native = WaveGateKernel::new(
                        kernel.curvature,
                        kernel.saturation,
                        kernel.porosity,
                        kernel.max_values,
                    )?;
                    if matches!(placement.recipe, GeometryAdapterRecipe::WaveRadius { .. }) {
                        Pullback::WaveRadius(Box::new(native.forward_with_log_radius(
                            input,
                            tensor(placement, "gate"),
                            tensor(placement, "bias"),
                            rows,
                            features,
                            tensor(placement, "log_radius")[0],
                        )?))
                    } else {
                        Pullback::Wave(Box::new(native.forward(
                            input,
                            tensor(placement, "gate"),
                            tensor(placement, "bias"),
                            rows,
                            features,
                        )?))
                    }
                }
                GeometryAdapterRecipe::Elliptic { warp, .. }
                | GeometryAdapterRecipe::EllipticAnchored { warp, .. } => {
                    Pullback::Elliptic(Box::new(elliptic::EllipticRuntimeBatch::forward(
                        placement,
                        warp,
                        &input_tensor,
                        None,
                    )?))
                }
                _ => unreachable!(),
            }
        };
        self.finish_forward(placement, input_tensor, strength, pullback)
    }

    fn finish_forward(
        &self,
        placement: &GeometryAdapterPlacement,
        input: Tensor,
        strength: f32,
        pullback: Pullback,
    ) -> Result<NativeGeometryBatch> {
        let residual = match &pullback {
            Pullback::Identity => None,
            Pullback::Affine(batch) => Some(batch.output()),
            Pullback::Wave(batch) | Pullback::WaveRadius(batch) => Some(batch.output()),
            Pullback::Elliptic(batch) => Some(batch.output()),
        };
        let output = if let Some(residual) = residual {
            input.add_with_backend(
                &residual.scale_with_backend(strength, TensorUtilBackend::Cpu)?,
                TensorUtilBackend::Cpu,
            )?
        } else {
            input
        };
        Ok(NativeGeometryBatch {
            source_id: self.artifact.artifact_id().into(),
            placement: placement.clone(),
            output,
            strength,
            pullback,
            sequence_shape: None,
        })
    }

    /// Plain SGD, no hidden optimizer state. Invalid/stale updates are atomic.
    pub fn step_sgd(
        &mut self,
        batch: &NativeGeometryBatch,
        upstream: &[f32],
        learning_rate: f32,
    ) -> Result<()> {
        if batch.source_id != self.artifact.artifact_id() {
            return Err("geometry snapshot is stale or belongs to another artifact".into());
        }
        if !learning_rate.is_finite() || learning_rate <= 0. {
            return Err("learning rate must be finite and positive".into());
        }
        let gradient = batch.vjp(upstream)?;
        let mut record = self.artifact.record().clone();
        let placement = record
            .placements
            .iter_mut()
            .find(|p| p.path == batch.placement.path)
            .unwrap();
        let mut trainable = false;
        for parameter in &mut placement.tensors {
            if parameter.role != GeometryAdapterTensorRole::Parameter || !parameter.requires_grad {
                continue;
            }
            trainable = true;
            let values = gradient
                .parameters
                .get(&parameter.name)
                .ok_or("missing native parameter gradient")?;
            let current = Tensor::from_vec(1, parameter.values.len(), parameter.values.clone())?;
            let delta = Tensor::from_vec(1, values.len(), values.clone())?
                .scale_with_backend(learning_rate, TensorUtilBackend::Cpu)?;
            parameter.values = current
                .sub_with_backend(&delta, TensorUtilBackend::Cpu)?
                .data()
                .to_vec();
        }
        if !trainable {
            return Err("selected placement has no trainable parameters".into());
        }
        let candidate = GeometryAdapterArtifact::create(record)?;
        self.artifact = candidate;
        Ok(())
    }

    /// One mean-MSE SGD step using the existing Rust loss and its pullback.
    /// Returns the pre-update loss, not a quality or checkpoint-selection claim.
    pub fn train_mse(
        &mut self,
        path: &str,
        input: &[f32],
        target: &[f32],
        rows: usize,
        learning_rate: f32,
    ) -> Result<f32> {
        let batch = self.forward(path, input, rows)?;
        self.apply_mse(&batch, target, learning_rate)
    }

    /// Mean-MSE across every value in a full, nonempty unpadded context.
    pub fn train_sequence_mse(
        &mut self,
        path: &str,
        input: &[f32],
        target: &[f32],
        shape: [usize; 2],
        learning_rate: f32,
    ) -> Result<f32> {
        let batch = self.forward_sequence(path, input, shape)?;
        self.apply_mse(&batch, target, learning_rate)
    }

    fn apply_mse(
        &mut self,
        batch: &NativeGeometryBatch,
        target: &[f32],
        learning_rate: f32,
    ) -> Result<f32> {
        if target.len() != batch.output.data().len() {
            return Err("native geometry target length differs from the validated output".into());
        }
        let (rows, features) = batch.output.shape();
        let target = Tensor::from_vec(rows, features, target.to_vec())?;
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let mut loss = MeanSquaredError::new();
        let value = loss.forward(&batch.output, &target)?.data()[0];
        let upstream = loss.backward(&batch.output, &target)?;
        self.step_sgd(batch, upstream.data(), learning_rate)?;
        Ok(value)
    }
}

impl NativeGeometryBatch {
    pub fn output(&self) -> &Tensor {
        &self.output
    }
    pub fn source_artifact_id(&self) -> &str {
        &self.source_id
    }
    pub fn sequence_shape(&self) -> Option<[usize; 3]> {
        self.sequence_shape
    }

    /// First-order residual pullback. Parameter gradients are row sums; fixed
    /// buffers are omitted, and SGD additionally respects requires_grad flags.
    pub fn vjp(&self, upstream: &[f32]) -> Result<NativeGeometryVjp> {
        let (rows, features) = self.output.shape();
        if upstream.len() != self.output.data().len() || upstream.iter().any(|v| !v.is_finite()) {
            return Err("invalid native geometry upstream shape or values".into());
        }
        let direction = Tensor::from_vec(rows, features, upstream.to_vec())?;
        let scaled = direction.scale_with_backend(self.strength, TensorUtilBackend::Cpu)?;
        let (input, gate, bias, radius) = match &self.pullback {
            Pullback::Identity => {
                return Ok(NativeGeometryVjp {
                    input: direction,
                    parameters: self
                        .placement
                        .tensors
                        .iter()
                        .filter(|t| t.role == GeometryAdapterTensorRole::Parameter)
                        .map(|t| (t.name.clone(), vec![0.; t.values.len()]))
                        .collect(),
                });
            }
            Pullback::Affine(batch) => {
                let g = batch.vjp(scaled.data())?;
                (g.grad_input, g.grad_gate, g.grad_bias, None)
            }
            Pullback::Wave(batch) => {
                let g = batch.vjp(scaled.data())?;
                (g.grad_input, g.grad_gate, g.grad_bias, None)
            }
            Pullback::WaveRadius(batch) => {
                let (g, radius) = batch.vjp_with_log_radius(scaled.data())?;
                (g.grad_input, g.grad_gate, g.grad_bias, Some(radius))
            }
            Pullback::Elliptic(batch) => {
                let gradient = batch.vjp(&scaled)?;
                return Ok(NativeGeometryVjp {
                    input: direction.add_with_backend(&gradient.input, TensorUtilBackend::Cpu)?,
                    parameters: gradient.parameters,
                });
            }
        };
        let mut parameters = BTreeMap::from([
            ("gate".into(), gate.data().to_vec()),
            ("bias".into(), bias.data().to_vec()),
        ]);
        if let Some(radius) = radius {
            if self
                .placement
                .tensors
                .iter()
                .any(|t| t.name == "log_radius" && t.role == GeometryAdapterTensorRole::Parameter)
            {
                parameters.insert("log_radius".into(), vec![radius]);
            }
        }
        Ok(NativeGeometryVjp {
            input: direction.add_with_backend(&input, TensorUtilBackend::Cpu)?,
            parameters,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn elliptic_artifact(anchored: bool, strength: f64) -> GeometryAdapterArtifact {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/channel_affine.fixture.json"
        ))
        .unwrap();
        let mut value = fixture["artifact"].clone();
        let placement = &mut value["placements"][0];
        placement["path"] = serde_json::json!("site");
        placement["recipe"] = serde_json::json!({
            "schema": if anchored { "spiraltorch.elliptic_anchored_residual_adapter.v1" } else { "spiraltorch.elliptic_residual_adapter.v1" },
            "features": 3, "strength": strength,
            "warp": {"curvature_radius": 1.3, "sheet_count": 3, "spin_harmonics": 2}
        });
        placement["tensors"] = serde_json::json!([
            {"name": "orientation.weight", "shape": [2,3], "values": [0.2,-0.1,0.15,0.05,0.3,-0.2], "requires_grad": true, "role": "parameter"},
            {"name": "orientation.bias", "shape": [2], "values": [0.1,-0.2], "requires_grad": true, "role": "parameter"},
            {"name": "readout.weight", "shape": [3,9], "values": (0..27).map(|i| (i as f32 - 13.) * 0.01).collect::<Vec<_>>(), "requires_grad": true, "role": "parameter"}
        ]);
        if anchored {
            placement["tensors"].as_array_mut().unwrap().push(serde_json::json!({
                "name": "raw_mix", "shape": [], "values": [0.3], "requires_grad": true, "role": "parameter"
            }));
        }
        GeometryAdapterArtifact::create_json(value.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn elliptic_composite_pullback_matches_all_parameter_and_input_differences() {
        let x = [0.2, -0.3, 0.4, 0.5, 0.1, -0.2];
        let seed = [0.3, 0.5, -0.4, 0.2, -0.1, 0.6];
        for anchored in [false, true] {
            let runtime = NativeGeometryRuntime::new(elliptic_artifact(anchored, 0.4));
            let batch = runtime.forward("site", &x, 2).unwrap();
            let gradient = batch.vjp(&seed).unwrap();
            let objective = |runtime: &NativeGeometryRuntime, input: &[f32]| {
                runtime
                    .forward("site", input, 2)
                    .unwrap()
                    .output()
                    .data()
                    .iter()
                    .zip(seed)
                    .map(|(v, d)| v * d)
                    .sum::<f32>()
            };
            let epsilon = 0.002;
            for index in 0..x.len() {
                let (mut plus, mut minus) = (x, x);
                plus[index] += epsilon;
                minus[index] -= epsilon;
                let numerical =
                    (objective(&runtime, &plus) - objective(&runtime, &minus)) / (2. * epsilon);
                assert!((numerical - gradient.input.data()[index]).abs() < 3e-4);
            }
            for (name, values) in &gradient.parameters {
                for (index, expected) in values.iter().enumerate() {
                    let perturbed = |sign: f32| {
                        let mut record = runtime.artifact().record().clone();
                        record.placements[0]
                            .tensors
                            .iter_mut()
                            .find(|t| t.name == *name)
                            .unwrap()
                            .values[index] += sign * epsilon;
                        NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap())
                    };
                    let numerical = (objective(&perturbed(1.), &x)
                        - objective(&perturbed(-1.), &x))
                        / (2. * epsilon);
                    assert!(
                        (numerical - expected).abs() < 3e-4,
                        "{name}[{index}]: {numerical} vs {expected}"
                    );
                }
            }
            assert_eq!(gradient.parameters.contains_key("raw_mix"), anchored);
        }
    }

    #[test]
    fn elliptic_identity_start_learns_and_continues_exactly_after_artifact_restore() {
        for anchored in [false, true] {
            let mut record = elliptic_artifact(anchored, 0.5).record().clone();
            record.placements[0]
                .tensors
                .iter_mut()
                .find(|t| t.name == "readout.weight")
                .unwrap()
                .values
                .fill(0.);
            let mut runtime =
                NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap());
            let x = [0.2, -0.3, 0.4, 0.5, 0.1, -0.2];
            let target = [0.1; 6];
            let batch = runtime.forward("site", &x, 2).unwrap();
            assert_eq!(batch.output().data(), &x);
            let gradient = batch.vjp(&[1.; 6]).unwrap();
            assert_eq!(gradient.parameters["orientation.weight"], vec![0.; 6]);
            assert!(gradient.parameters["readout.weight"]
                .iter()
                .any(|v| *v != 0.));
            let first = runtime.train_mse("site", &x, &target, 2, 0.1).unwrap();
            for _ in 0..39 {
                runtime.train_mse("site", &x, &target, 2, 0.1).unwrap();
            }
            let saved = runtime.artifact();
            let restored = GeometryAdapterArtifact::from_json(
                saved.to_json().as_bytes(),
                saved.artifact_id(),
                saved.record().base_model_identity.as_str(),
            )
            .unwrap();
            let mut restored = NativeGeometryRuntime::new(restored);
            let loss = runtime.train_mse("site", &x, &target, 2, 0.1).unwrap();
            assert!(loss < first);
            assert_eq!(
                loss,
                restored.train_mse("site", &x, &target, 2, 0.1).unwrap()
            );
            assert_eq!(
                runtime.artifact().artifact_id(),
                restored.artifact().artifact_id()
            );
        }
    }

    #[test]
    fn elliptic_rows_bypass_empty_and_atomic_failure_boundaries() {
        for anchored in [false, true] {
            let mut runtime = NativeGeometryRuntime::new(elliptic_artifact(anchored, 0.4));
            assert_eq!(runtime.dimensions("site").unwrap(), (3, 3 * 65_536));
            assert!(runtime
                .forward("site", &vec![0.; 3 * 65_537], 65_537)
                .is_err());
            let empty = runtime.forward("site", &[], 0).unwrap();
            assert!(empty.output().data().is_empty());
            assert!(empty
                .vjp(&[])
                .unwrap()
                .parameters
                .values()
                .flatten()
                .all(|v| *v == 0.));
            let id = runtime.artifact().artifact_id().to_string();
            assert!(runtime
                .train_mse("site", &[0.2; 6], &[f32::NAN; 6], 2, 0.1)
                .is_err());
            let batch = runtime.forward("site", &[0.2; 6], 2).unwrap();
            assert!(runtime.step_sgd(&batch, &[f32::MAX; 6], f32::MAX).is_err());
            assert_eq!(runtime.artifact().artifact_id(), id);
            let bypass = NativeGeometryRuntime::new(elliptic_artifact(anchored, 0.));
            let identity = bypass.forward("site", &[f32::MAX; 6], 2).unwrap();
            assert_eq!(identity.output().data(), &[f32::MAX; 6]);
            assert!(identity
                .vjp(&[1.; 6])
                .unwrap()
                .parameters
                .values()
                .flatten()
                .all(|v| *v == 0.));
        }
    }

    fn artifact(wave: bool, radius: Option<bool>, strength: f64) -> GeometryAdapterArtifact {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/channel_affine.fixture.json"
        ))
        .unwrap();
        let mut value = fixture["artifact"].clone();
        let placement = &mut value["placements"][0];
        placement["path"] = serde_json::json!("site");
        placement["recipe"]["strength"] = serde_json::json!(strength);
        if wave {
            placement["recipe"] = serde_json::json!({
                "schema": if radius.is_some() { "spiraltorch.wave_gate_adapter.v2" } else { "spiraltorch.wave_gate_adapter.v1" },
                "features": 3, "strength": strength,
                "kernel": {"curvature": -1., "saturation": 1., "porosity": 0.05, "max_values": 64}
            });
            if let Some(learnable) = radius {
                placement["recipe"]["learnable_radius"] = serde_json::json!(learnable);
                placement["tensors"].as_array_mut().unwrap().push(serde_json::json!({
                    "name": "log_radius", "shape": [], "values": [0.2], "requires_grad": learnable,
                    "role": if learnable { "parameter" } else { "buffer" }
                }));
            }
        }
        for tensor in placement["tensors"].as_array_mut().unwrap() {
            if tensor["name"] == "gate" {
                tensor["values"] = serde_json::json!([0.1, -0.2, 0.3]);
            }
            if tensor["name"] == "bias" {
                tensor["values"] = serde_json::json!([0.05, 0.1, -0.05]);
            }
        }
        GeometryAdapterArtifact::create_json(value.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn residual_pullback_matches_finite_differences_for_all_supported_recipes() {
        let x = [0.2, -0.3, 0.4, 0.5, 0.1, -0.2];
        let upstream = [0.3, 0.5, -0.4, 0.2, -0.1, 0.6];
        for (wave, radius) in [
            (false, None),
            (true, None),
            (true, Some(false)),
            (true, Some(true)),
        ] {
            let runtime = NativeGeometryRuntime::new(artifact(wave, radius, 0.4));
            let batch = runtime.forward("site", &x, 2).unwrap();
            let gradient = batch.vjp(&upstream).unwrap();
            let objective = |runtime: &NativeGeometryRuntime, input: &[f32]| -> f32 {
                runtime
                    .forward("site", input, 2)
                    .unwrap()
                    .output()
                    .data()
                    .iter()
                    .zip(upstream)
                    .map(|(y, d)| y * d)
                    .sum()
            };
            let epsilon = 0.001;
            for index in 0..x.len() {
                let mut plus = x;
                let mut minus = x;
                plus[index] += epsilon;
                minus[index] -= epsilon;
                let numerical =
                    (objective(&runtime, &plus) - objective(&runtime, &minus)) / (2. * epsilon);
                assert!((numerical - gradient.input.data()[index]).abs() < 1e-4);
            }
            for (name, values) in &gradient.parameters {
                for (index, &expected) in values.iter().enumerate() {
                    let mutate = |sign: f32| {
                        let mut record = runtime.artifact().record().clone();
                        record.placements[0]
                            .tensors
                            .iter_mut()
                            .find(|t| t.name == *name)
                            .unwrap()
                            .values[index] += sign * epsilon;
                        NativeGeometryRuntime::new(GeometryAdapterArtifact::create(record).unwrap())
                    };
                    let numerical =
                        (objective(&mutate(1.), &x) - objective(&mutate(-1.), &x)) / (2. * epsilon);
                    assert!(
                        (numerical - expected).abs() < 1e-4,
                        "{name}: {numerical} vs {expected}"
                    );
                }
            }
            assert_eq!(
                gradient.parameters.contains_key("log_radius"),
                radius == Some(true)
            );
        }
    }

    #[test]
    fn mse_training_reloads_and_continues_without_an_external_framework() {
        for wave in [false, true] {
            let mut runtime = NativeGeometryRuntime::new(artifact(wave, None, 0.5));
            let x = [0.2, -0.3, 0.4, 0.5, 0.1, -0.2];
            let target = [0.1; 6];
            let first = runtime.train_mse("site", &x, &target, 2, 0.2).unwrap();
            for _ in 0..39 {
                runtime.train_mse("site", &x, &target, 2, 0.2).unwrap();
            }
            let saved = runtime.artifact();
            let loaded = GeometryAdapterArtifact::from_json(
                saved.to_json().as_bytes(),
                saved.artifact_id(),
                &saved.record().base_model_identity,
            )
            .unwrap();
            let mut restored = NativeGeometryRuntime::new(loaded);
            let final_loss = runtime.train_mse("site", &x, &target, 2, 0.2).unwrap();
            assert!(final_loss < first * 0.7);
            assert_eq!(
                final_loss,
                restored.train_mse("site", &x, &target, 2, 0.2).unwrap()
            );
            assert_eq!(
                runtime.artifact().artifact_id(),
                restored.artifact().artifact_id()
            );
        }
    }

    #[test]
    fn invalid_and_stale_updates_are_atomic_and_frozen_values_stay_fixed() {
        let mut runtime = NativeGeometryRuntime::new(artifact(true, Some(false), 0.5));
        let batch = runtime.forward("site", &[0.2; 6], 2).unwrap();
        let before = runtime.artifact().to_json().to_owned();
        for rate in [0., -1., f32::NAN, f32::INFINITY] {
            assert!(runtime.step_sgd(&batch, &[1.; 6], rate).is_err());
            assert_eq!(runtime.artifact().to_json(), before);
        }
        assert!(runtime.step_sgd(&batch, &[f32::NAN; 6], 0.1).is_err());
        assert_eq!(runtime.artifact().to_json(), before);
        let old_radius = tensor(&runtime.artifact().record().placements[0], "log_radius").to_vec();
        runtime.step_sgd(&batch, &[1.; 6], 0.1).unwrap();
        assert_eq!(
            tensor(&runtime.artifact().record().placements[0], "log_radius"),
            old_radius
        );
        let updated = runtime.artifact().to_json().to_owned();
        assert!(runtime.step_sgd(&batch, &[1.; 6], 0.1).is_err());
        assert_eq!(runtime.artifact().to_json(), updated);
        assert_eq!(batch.vjp(&[1.; 6]).unwrap().input.shape(), (2, 3));
    }

    #[test]
    fn zero_strength_is_identity_and_empty_rows_are_valid() {
        let runtime = NativeGeometryRuntime::new(artifact(true, Some(true), 0.));
        let batch = runtime.forward("site", &[0.2; 6], 2).unwrap();
        assert_eq!(batch.output().data(), &[0.2; 6]);
        let gradient = batch.vjp(&[1.; 6]).unwrap();
        assert_eq!(gradient.input.data(), &[1.; 6]);
        assert!(gradient.parameters.values().flatten().all(|&x| x == 0.));
        let empty = runtime.forward("site", &[], 0).unwrap();
        assert_eq!(empty.vjp(&[]).unwrap().input.shape(), (0, 3));
        assert!(runtime.forward("site", &[f32::NAN; 6], 2).is_err());
        assert!(runtime.forward("missing", &[], 0).is_err());
        assert!(runtime.forward("site", &[1.; 6], usize::MAX).is_err());
    }

    #[test]
    fn direct_rust_target_length_is_bounded_before_copy_and_update() {
        let mut runtime = NativeGeometryRuntime::new(artifact(false, None, 0.5));
        let before = runtime.artifact().artifact_id().to_owned();
        for (input, rows) in [(&[][..], 0), (&[0.2; 6][..], 2)] {
            let error = runtime
                .train_mse("site", input, &[0.; 128], rows, 0.1)
                .unwrap_err();
            assert!(error.to_string().contains("target length"));
            assert_eq!(runtime.artifact().artifact_id(), before);
        }
    }
}
