use std::collections::HashMap;
use std::time::SystemTime;

use st_tensor::fractal::FractalPatch;
use st_tensor::wasm_canvas::{CanvasProjector, ColorVectorField};
use st_tensor::Tensor;

use crate::error::RoboticsError;
use crate::runtime::RuntimeStep;

/// Snapshot linking sensor measurements with the CanvasProjector state.
#[derive(Debug, Clone)]
pub struct VisionFeedbackSnapshot {
    pub channel: String,
    pub timestamp: SystemTime,
    pub sensor: Vec<f32>,
    pub sensor_norm: f32,
    pub canvas_mean_energy: f32,
    pub canvas_rms_energy: f32,
    pub chroma: [f32; 3],
    pub alignment: f32,
}

impl VisionFeedbackSnapshot {
    pub fn metrics(&self) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();
        metrics.insert("vision.energy.mean".to_string(), self.canvas_mean_energy);
        metrics.insert("vision.energy.rms".to_string(), self.canvas_rms_energy);
        metrics.insert("vision.alignment".to_string(), self.alignment);
        metrics.insert("vision.sensor.norm".to_string(), self.sensor_norm);
        metrics.insert("vision.chroma.r".to_string(), self.chroma[0]);
        metrics.insert("vision.chroma.g".to_string(), self.chroma[1]);
        metrics.insert("vision.chroma.b".to_string(), self.chroma[2]);
        metrics
    }

    pub fn gradient_component(&self) -> Vec<f32> {
        vec![self.alignment, self.canvas_rms_energy]
    }
}

/// Synchronises robotics sensor payloads with CanvasProjector vector fields.
#[derive(Debug, Clone)]
pub struct VisionFeedbackSynchronizer {
    channel: String,
    coherence: f32,
    tension: f32,
    depth: u32,
}

impl VisionFeedbackSynchronizer {
    pub fn new(channel: impl Into<String>) -> Self {
        Self {
            channel: channel.into(),
            coherence: 1.0,
            tension: 1.0,
            depth: 1,
        }
    }

    pub fn with_patch(
        channel: impl Into<String>,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> Self {
        Self {
            channel: channel.into(),
            coherence: coherence.max(1e-6),
            tension: tension.max(1e-6),
            depth: depth.max(1),
        }
    }

    pub fn channel(&self) -> &str {
        &self.channel
    }

    pub fn set_patch_parameters(&mut self, coherence: f32, tension: f32, depth: u32) {
        self.coherence = coherence.max(1e-6);
        self.tension = tension.max(1e-6);
        self.depth = depth.max(1);
    }

    pub fn sync_with_projector(
        &self,
        step: &RuntimeStep,
        projector: &mut CanvasProjector,
    ) -> Result<VisionFeedbackSnapshot, RoboticsError> {
        let sensor = self
            .extract_sensor(&step.frame.coordinates)
            .ok_or_else(|| {
                RoboticsError::VisionSync(format!(
                    "channel '{}' missing from fused frame",
                    self.channel
                ))
            })?;
        let relation = Tensor::from_vec(1, sensor.len(), sensor.clone()).map_err(|err| {
            RoboticsError::VisionSync(format!("failed to build sensor tensor: {err}"))
        })?;
        let patch = FractalPatch::new(relation, self.coherence, self.tension, self.depth).map_err(
            |err| RoboticsError::VisionSync(format!("failed to build fractal patch: {err}")),
        )?;
        projector.scheduler().push(patch).map_err(|err| {
            RoboticsError::VisionSync(format!("failed to enqueue canvas patch: {err}"))
        })?;
        let (_, field) = projector.refresh_with_vectors().map_err(|err| {
            RoboticsError::VisionSync(format!("failed to refresh canvas vectors: {err}"))
        })?;
        Ok(self.snapshot_from_field(step, sensor, field))
    }

    pub fn sync_with_field(
        &self,
        step: &RuntimeStep,
        vectors: &ColorVectorField,
    ) -> Result<VisionFeedbackSnapshot, RoboticsError> {
        let sensor = self
            .extract_sensor(&step.frame.coordinates)
            .ok_or_else(|| {
                RoboticsError::VisionSync(format!(
                    "channel '{}' missing from fused frame",
                    self.channel
                ))
            })?;
        Ok(self.snapshot_from_vectors(step, sensor, vectors.vectors()))
    }

    pub fn sync_with_vectors(
        &self,
        step: &RuntimeStep,
        vectors: &[[f32; 4]],
    ) -> Result<VisionFeedbackSnapshot, RoboticsError> {
        let sensor = self
            .extract_sensor(&step.frame.coordinates)
            .ok_or_else(|| {
                RoboticsError::VisionSync(format!(
                    "channel '{}' missing from fused frame",
                    self.channel
                ))
            })?;
        Ok(self.snapshot_from_vectors(step, sensor, vectors))
    }

    fn extract_sensor(&self, coordinates: &HashMap<String, Vec<f32>>) -> Option<Vec<f32>> {
        coordinates.get(&self.channel).map(|values| values.clone())
    }

    fn snapshot_from_field(
        &self,
        step: &RuntimeStep,
        sensor: Vec<f32>,
        field: &ColorVectorField,
    ) -> VisionFeedbackSnapshot {
        self.snapshot_from_vectors(step, sensor, field.vectors())
    }

    fn snapshot_from_vectors(
        &self,
        step: &RuntimeStep,
        sensor: Vec<f32>,
        vectors: &[[f32; 4]],
    ) -> VisionFeedbackSnapshot {
        let sensor_norm = sensor.iter().map(|value| value * value).sum::<f32>().sqrt();
        let (mean_energy, rms_energy, chroma) = summarise_vectors(vectors);
        let alignment = if sensor_norm > 0.0 {
            mean_energy / sensor_norm
        } else {
            0.0
        };
        VisionFeedbackSnapshot {
            channel: self.channel.clone(),
            timestamp: step.frame.timestamp,
            sensor,
            sensor_norm,
            canvas_mean_energy: mean_energy,
            canvas_rms_energy: rms_energy,
            chroma,
            alignment,
        }
    }
}

fn summarise_vectors(vectors: &[[f32; 4]]) -> (f32, f32, [f32; 3]) {
    let mut energy_sum = 0.0;
    let mut energy_sq = 0.0;
    let mut chroma = [0.0; 3];
    let mut count = 0.0;
    for vector in vectors.iter() {
        energy_sum += vector[0];
        energy_sq += vector[0] * vector[0];
        chroma[0] += vector[1];
        chroma[1] += vector[2];
        chroma[2] += vector[3];
        count += 1.0;
    }
    if count > 0.0 {
        chroma[0] /= count;
        chroma[1] /= count;
        chroma[2] /= count;
    }
    let mean_energy = if count > 0.0 { energy_sum / count } else { 0.0 };
    let rms_energy = if count > 0.0 {
        (energy_sq / count).sqrt()
    } else {
        0.0
    };
    (mean_energy, rms_energy, chroma)
}
