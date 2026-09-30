use super::*;
use crate::models::convnext::checkpoint::{ConvNeXtTrainingCheckpoint, StoredParameter};
use st_backend_wgpu::resident_tensor::TensorReadbackBatch;

/// Frozen model metadata and one ordered weight capture; mapping is explicit.
/// This owns the captured version even if learning continues or the model drops.
pub struct ConvNeXtCheckpointSnapshot {
    config: ConvNeXtConfig,
    batch: usize,
    revision: u64,
    names: Vec<String>,
    shapes: Vec<[usize; 2]>,
    raw: TensorReadbackBatch,
}

impl ResidentConvNeXtBackbone {
    pub fn checkpoint_snapshot(&self) -> Result<ConvNeXtCheckpointSnapshot, InferenceError> {
        let current = self.parameters.snapshot();
        let refs: Vec<_> = current.values().iter().collect();
        let raw = self.device.snapshot_many(&refs)?;
        Ok(ConvNeXtCheckpointSnapshot {
            config: self.config.clone(),
            batch: self.input_shape[0],
            revision: current.revision(),
            names: self.names.clone(),
            shapes: self.shapes.clone(),
            raw,
        })
    }
}

impl ConvNeXtCheckpointSnapshot {
    fn finish(
        config: ConvNeXtConfig,
        batch: usize,
        revision: u64,
        names: Vec<String>,
        shapes: Vec<[usize; 2]>,
        values: Vec<Vec<f32>>,
    ) -> Result<ConvNeXtTrainingCheckpoint, InferenceError> {
        if names.len() != shapes.len() || names.len() != values.len() {
            return Err(InferenceError::ModuleUpdate(
                "checkpoint parameter count differs",
            ));
        }
        let parameters = names
            .into_iter()
            .zip(shapes)
            .zip(values)
            .map(|((name, shape), values)| StoredParameter {
                name,
                shape,
                values,
            })
            .collect();
        Ok(ConvNeXtTrainingCheckpoint::new(
            config, batch, revision, parameters,
        )?)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<ConvNeXtTrainingCheckpoint, InferenceError> {
        Self::finish(
            self.config,
            self.batch,
            self.revision,
            self.names,
            self.shapes,
            self.raw.read()?,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<ConvNeXtTrainingCheckpoint, InferenceError> {
        Self::finish(
            self.config,
            self.batch,
            self.revision,
            self.names,
            self.shapes,
            self.raw.read_async().await?,
        )
    }
}

impl ConvNeXtTrainingCheckpoint {
    /// Restore numerical state and the attempted-update clock, with fresh model
    /// and gradient identities. No old forward or derivative token is restored.
    pub fn restore_resident(
        &self,
        device: TensorDevice,
    ) -> Result<ResidentConvNeXtBackbone, InferenceError> {
        self.to_host()?.compile_resident_at_revision(
            device,
            self.batch_size(),
            self.attempted_updates(),
        )
    }
}
