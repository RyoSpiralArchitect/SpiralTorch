use super::*;

pub(super) struct RawSnapshot {
    lease: runtime::ReadbackLease,
    context: WgpuContext,
}

impl RawSnapshot {
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn read(mut self) -> Result<Vec<u8>, TrainingError> {
        Ok(self.lease.read(
            &self.context,
            std::time::Duration::from_secs(30),
            "training.snapshot",
        )?)
    }
    #[cfg(target_arch = "wasm32")]
    pub(super) async fn read_async(self) -> Result<Vec<u8>, TrainingError> {
        Ok(self
            .lease
            .read_async(self.context, "training.snapshot")
            .await?)
    }
}

pub(super) fn capture(
    context: &WgpuContext,
    pool: &runtime::ReadbackPool,
    buffers: &[&wgpu::Buffer],
) -> Result<RawSnapshot, TrainingError> {
    let lease = pool.checkout("training.snapshot");
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let mut offset = 0u64;
    for buffer in buffers {
        let next = offset
            .checked_add(buffer.size())
            .ok_or(TrainingError::Overflow)?;
        if next > lease.buffer().size() {
            return Err(TrainingError::InvalidReadback);
        }
        encoder.copy_buffer_to_buffer(buffer, 0, lease.buffer(), offset, buffer.size());
        offset = next;
    }
    if offset != lease.buffer().size() {
        return Err(TrainingError::InvalidReadback);
    }
    context.queue().submit(Some(encoder.finish()));
    Ok(RawSnapshot {
        lease,
        context: context.clone(),
    })
}

pub(super) fn capture_new(
    context: &WgpuContext,
    buffers: &[&wgpu::Buffer],
) -> Result<RawSnapshot, TrainingError> {
    let bytes = buffers
        .iter()
        .try_fold(0u64, |n, b| n.checked_add(b.size()))
        .ok_or(TrainingError::Overflow)?;
    let len = usize::try_from(bytes / 4).map_err(|_| TrainingError::Overflow)?;
    let pool = runtime::ReadbackPool::new::<u32>(context.clone(), len)?;
    capture(context, &pool, buffers)
}

pub(super) fn loss_and_flags(bytes: &[u8], stages: usize) -> Result<(f32, usize), TrainingError> {
    let prefix = stages
        .checked_add(3)
        .and_then(|n| n.checked_mul(4))
        .ok_or(TrainingError::InvalidReadback)?;
    if bytes.len() < prefix {
        return Err(TrainingError::InvalidReadback);
    }
    let mut all = 0;
    let mut failure = None;
    for stage in 0..=stages {
        let start = (1 + stage) * 4;
        let flags = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        all |= flags;
        if flags != 0 && failure.is_none() {
            failure = Some(TrainingError::Rejected { stage, flags });
        }
    }
    let committed_flags = u32::from_le_bytes(bytes[prefix - 4..prefix].try_into().unwrap());
    if all != committed_flags {
        return Err(TrainingError::InvalidReadback);
    }
    if let Some(error) = failure {
        return Err(error);
    }
    let loss = f32::from_le_bytes(bytes[..4].try_into().unwrap());
    if !loss.is_finite() || loss < 0. {
        return Err(TrainingError::InvalidReadback);
    }
    Ok((loss, prefix))
}

pub(super) fn values(
    bytes: &[u8],
    offset: &mut usize,
    len: usize,
) -> Result<Vec<f32>, TrainingError> {
    let end = len
        .checked_mul(4)
        .and_then(|n| offset.checked_add(n))
        .ok_or(TrainingError::InvalidReadback)?;
    if end > bytes.len() {
        return Err(TrainingError::InvalidReadback);
    }
    #[allow(clippy::chunks_exact_to_as_chunks)]
    let values: Vec<_> = bytes[*offset..end]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if !values.iter().all(|v| v.is_finite()) {
        return Err(TrainingError::InvalidReadback);
    }
    *offset = end;
    Ok(values)
}

fn layer(bytes: &[u8], offset: &mut usize, spec: Spec) -> Result<DenseLayer, TrainingError> {
    Ok(DenseLayer {
        inner: spec.inner,
        cols: spec.cols,
        activation: spec.activation,
        weights: values(bytes, offset, spec.inner * spec.cols)?,
        bias: values(bytes, offset, spec.cols)?,
    })
}

pub struct StepReadback {
    pub(super) raw: RawSnapshot,
    pub(super) stages: usize,
    pub(super) step: u64,
    pub(super) batch_generation: u64,
}
impl StepReadback {
    pub fn submitted_step(&self) -> u64 {
        self.step
    }
    pub fn batch_generation(&self) -> u64 {
        self.batch_generation
    }
    fn decode(bytes: &[u8], stages: usize) -> Result<f32, TrainingError> {
        let (loss, end) = loss_and_flags(bytes, stages)?;
        if end != bytes.len() {
            return Err(TrainingError::InvalidReadback);
        }
        Ok(loss)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<f32, TrainingError> {
        Self::decode(&self.raw.read()?, self.stages)
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<f32, TrainingError> {
        Self::decode(&self.raw.read_async().await?, self.stages)
    }
}

#[derive(Debug)]
pub struct LayerGradient {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Debug)]
pub struct TrainingState {
    pub loss: f32,
    pub submitted_step: u64,
    pub batch_generation: u64,
    pub input_layout: NdLayout,
    pub output_layout: NdLayout,
    pub prediction: Vec<f32>,
    pub input_gradient: Vec<f32>,
    pub parameters: Vec<DenseLayer>,
    pub parameter_gradients: Vec<LayerGradient>,
}

pub struct TrainingStateReadback {
    pub(super) raw: RawSnapshot,
    pub(super) specs: Vec<Spec>,
    pub(super) input: NdLayout,
    pub(super) output: NdLayout,
    pub(super) step: u64,
    pub(super) batch_generation: u64,
}
impl TrainingStateReadback {
    pub fn input_layout(&self) -> &NdLayout {
        &self.input
    }
    pub fn output_layout(&self) -> &NdLayout {
        &self.output
    }
    pub fn submitted_step(&self) -> u64 {
        self.step
    }
    pub fn batch_generation(&self) -> u64 {
        self.batch_generation
    }

    fn decode(
        bytes: &[u8],
        specs: &[Spec],
        input: NdLayout,
        output: NdLayout,
        step: u64,
        batch_generation: u64,
    ) -> Result<TrainingState, TrainingError> {
        let (loss, mut offset) = loss_and_flags(bytes, specs.len())?;
        let prediction = values(bytes, &mut offset, output.len())?;
        let input_gradient = values(bytes, &mut offset, input.len())?;
        let mut parameters = Vec::new();
        let mut parameter_gradients = Vec::new();
        for &spec in specs {
            parameters.push(layer(bytes, &mut offset, spec)?);
            parameter_gradients.push(LayerGradient {
                weights: values(bytes, &mut offset, spec.inner * spec.cols)?,
                bias: values(bytes, &mut offset, spec.cols)?,
            });
        }
        if offset != bytes.len() {
            return Err(TrainingError::InvalidReadback);
        }
        Ok(TrainingState {
            loss,
            submitted_step: step,
            batch_generation,
            input_layout: input,
            output_layout: output,
            prediction,
            input_gradient,
            parameters,
            parameter_gradients,
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<TrainingState, TrainingError> {
        Self::decode(
            &self.raw.read()?,
            &self.specs,
            self.input,
            self.output,
            self.step,
            self.batch_generation,
        )
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<TrainingState, TrainingError> {
        Self::decode(
            &self.raw.read_async().await?,
            &self.specs,
            self.input,
            self.output,
            self.step,
            self.batch_generation,
        )
    }
}

pub struct ParameterReadback {
    pub(super) raw: RawSnapshot,
    pub(super) specs: Vec<Spec>,
}
impl ParameterReadback {
    fn decode(bytes: &[u8], specs: &[Spec]) -> Result<Vec<DenseLayer>, TrainingError> {
        let mut offset = 0;
        let result = specs
            .iter()
            .map(|&spec| layer(bytes, &mut offset, spec))
            .collect::<Result<_, _>>()?;
        if offset != bytes.len() {
            return Err(TrainingError::InvalidReadback);
        }
        Ok(result)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(self) -> Result<Vec<DenseLayer>, TrainingError> {
        Self::decode(&self.raw.read()?, &self.specs)
    }
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<Vec<DenseLayer>, TrainingError> {
        Self::decode(&self.raw.read_async().await?, &self.specs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_decode_requires_the_global_decision_and_all_finite_flags() {
        let mut words = [0.25f32.to_bits(), 0, 0, 0, 0];
        assert_eq!(
            StepReadback::decode(bytemuck::cast_slice(&words), 2).unwrap(),
            0.25
        );
        words[1] = 8192;
        assert!(matches!(
            StepReadback::decode(bytemuck::cast_slice(&words), 2),
            Err(TrainingError::InvalidReadback)
        ));
        words[4] = 8192;
        assert!(matches!(
            StepReadback::decode(bytemuck::cast_slice(&words), 2),
            Err(TrainingError::Rejected {
                stage: 0,
                flags: 8192
            })
        ));
        assert!(StepReadback::decode(&[], usize::MAX).is_err());
        assert!(StepReadback::decode(
            bytemuck::cast_slice(&[f32::NAN.to_bits(), 0u32, 0, 0, 0]),
            2
        )
        .is_err());
    }
}
