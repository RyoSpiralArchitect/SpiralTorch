use super::*;
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
    resident_tensor::{TensorDevice, TensorReadbackBatch},
    resident_training::parameters::{ResidentAdamWReadback, ResidentParameterSnapshot},
    runtime::WgpuRuntime,
};

/// An immutable capture of one attempted revision. Mapping is explicit; later
/// updates or dropping the original model cannot change the captured values.
pub struct ByteDecoderCheckpointReadback {
    model: ModelRecord,
    attempted_revision: u64,
    readback: TensorReadbackBatch,
    optimizer: Option<ResidentAdamWReadback>,
}

impl CheckpointTemplate {
    pub(in crate::resident::byte_decoder) fn capture(
        &self,
        device: &TensorDevice,
        parameters: ResidentParameterSnapshot,
        optimizer: Option<ResidentAdamWReadback>,
    ) -> Result<ByteDecoderCheckpointReadback, InferenceError> {
        Ok(ByteDecoderCheckpointReadback {
            model: self.0.clone(),
            attempted_revision: parameters.revision(),
            readback: device.snapshot_many(&parameters.values().iter().collect::<Vec<_>>())?,
            optimizer,
        })
    }
}

fn finish(
    mut model: ModelRecord,
    attempted_revision: u64,
    values: Vec<Vec<f32>>,
    optimizer: Option<AdamWState>,
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
    let plan = model.into_plan()?;
    if let Some(state) = &optimizer {
        state.validate(&plan.parameter_shapes()?, attempted_revision)?;
    }
    Ok(ByteDecoderCheckpoint {
        plan,
        attempted_revision,
        optimizer,
    })
}

impl ByteDecoderCheckpointReadback {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<ByteDecoderCheckpoint, InferenceError> {
        let optimizer = self
            .optimizer
            .map(ResidentAdamWReadback::read)
            .transpose()?;
        finish(
            self.model,
            self.attempted_revision,
            self.readback.read()?,
            optimizer,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<ByteDecoderCheckpoint, InferenceError> {
        let optimizer = match self.optimizer {
            Some(state) => Some(state.read_async().await?),
            None => None,
        };
        finish(
            self.model,
            self.attempted_revision,
            self.readback.read_async().await?,
            optimizer,
        )
    }
}

impl ByteDecoderCheckpoint {
    /// Immutable serving does not consume optimizer history. Keep the original
    /// checkpoint intact for provenance and explicit training restoration.
    pub(crate) fn restore_weights_wgpu(
        &self,
        runtime: WgpuRuntime,
    ) -> Result<ResidentByteDecoder, InferenceError> {
        self.plan.compile_training_wgpu_at_revision(
            runtime,
            MatmulTile::default(),
            MatmulKernel::Scalar,
            MatmulAccumulation::default(),
            self.attempted_revision,
        )
    }

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
        let mut model = self.plan.compile_training_wgpu_at_revision(
            runtime,
            tile,
            kernel,
            accumulation,
            self.attempted_revision,
        )?;
        if let Some(state) = &self.optimizer {
            model.restore_adamw(state)?;
        }
        Ok(model)
    }
}
