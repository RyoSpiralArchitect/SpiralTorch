use super::*;
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{TensorDevice, TensorReadbackBatch},
    resident_training::parameters::ResidentParameterSnapshot,
    runtime::WgpuRuntime,
};

/// An immutable capture of one attempted revision. Mapping is explicit; later
/// updates or dropping the original model cannot change the captured values.
pub struct ByteDecoderCheckpointReadback {
    model: ModelRecord,
    attempted_revision: u64,
    readback: TensorReadbackBatch,
}

impl CheckpointTemplate {
    pub(in crate::resident::byte_decoder) fn capture(
        &self,
        device: &TensorDevice,
        parameters: ResidentParameterSnapshot,
    ) -> Result<ByteDecoderCheckpointReadback, InferenceError> {
        Ok(ByteDecoderCheckpointReadback {
            model: self.0.clone(),
            attempted_revision: parameters.revision(),
            readback: device.snapshot_many(&parameters.values().iter().collect::<Vec<_>>())?,
        })
    }
}

fn finish(
    mut model: ModelRecord,
    attempted_revision: u64,
    values: Vec<Vec<f32>>,
) -> Result<ByteDecoderCheckpoint, InferenceError> {
    let mut values = values.into_iter();
    for destination in model.values_mut() {
        *destination = values
            .next()
            .ok_or(InferenceError::ByteDecoder("missing checkpoint parameter"))?;
    }
    if values.next().is_some() {
        return Err(InferenceError::ByteDecoder("excess checkpoint parameter"));
    }
    Ok(ByteDecoderCheckpoint {
        plan: model.into_plan()?,
        attempted_revision,
    })
}

impl ByteDecoderCheckpointReadback {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<ByteDecoderCheckpoint, InferenceError> {
        finish(self.model, self.attempted_revision, self.readback.read()?)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<ByteDecoderCheckpoint, InferenceError> {
        finish(
            self.model,
            self.attempted_revision,
            self.readback.read_async().await?,
        )
    }
}

impl ByteDecoderCheckpoint {
    pub fn restore_wgpu(
        &self,
        runtime: WgpuRuntime,
    ) -> Result<ResidentByteDecoder, InferenceError> {
        self.restore_wgpu_with_options(
            runtime,
            Default::default(),
            MatmulKernel::Scalar,
            Default::default(),
        )
    }

    /// Runtime and numerical kernel choices are explicit caller state. Preserve
    /// them when testing uninterrupted/resumed numerical equivalence.
    pub fn restore_wgpu_with_options(
        &self,
        runtime: WgpuRuntime,
        tile: MatmulTile,
        kernel: MatmulKernel,
        accumulation: MatmulAccumulation,
    ) -> Result<ResidentByteDecoder, InferenceError> {
        self.plan.compile_training_wgpu_at_revision(
            runtime,
            tile,
            kernel,
            accumulation,
            self.attempted_revision,
        )
    }
}
