//! Resident image preparation. Submission is not numerical acceptance: failed
//! GPU guards are observed by a downstream loss/update or an explicit snapshot.

use super::*;
use st_backend_wgpu::{resident_tensor::pointwise::PointwisePlan, runtime::Shared};
use st_kernel_contracts::{
    layout::NdLayout,
    pointwise::{PointwiseChain, PointwiseExecution, PointwiseStep},
};

fn gpu_error(error: impl fmt::Display) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: error.to_string(),
    }
}

#[derive(Debug)]
pub(crate) struct PreparedNormalize {
    layout: NdLayout,
    means: ResidentTensor,
    stds: ResidentTensor,
    plan: PointwisePlan,
}

impl PreparedNormalize {
    fn new(op: &Normalize, input: &ResidentTensor) -> PureResult<Self> {
        let device = input.device();
        let shape = [1, op.means.len(), 1, 1];
        let means = device.upload(&shape, &op.means).map_err(gpu_error)?;
        let stds = device.upload(&shape, &op.stds).map_err(gpu_error)?;
        let chain = PointwiseChain::new(
            3,
            vec![
                PointwiseStep::named("subtract", Some(1)).map_err(gpu_error)?,
                PointwiseStep::named("divide", Some(2)).map_err(gpu_error)?,
            ],
        )
        .map_err(gpu_error)?;
        let layout = input.layout().clone();
        let plan = PointwisePlan::new(
            device.clone(),
            chain,
            vec![
                layout.clone(),
                means.layout().clone(),
                stds.layout().clone(),
            ],
        )
        .map_err(gpu_error)?;
        Ok(Self {
            layout,
            means,
            stds,
            plan,
        })
    }

    fn matches(&self, input: &ResidentTensor) -> bool {
        self.layout == *input.layout()
            && self
                .means
                .device()
                .runtime()
                .context()
                .shares_handles_with(input.device().runtime().context())
    }

    fn run(&self, input: &ResidentTensor) -> PureResult<ResidentTensor> {
        self.plan
            .run(&[input, &self.means, &self.stds], PointwiseExecution::Fused)
            .map_err(gpu_error)
    }
}

fn pack_images(images: &[ImageTensor]) -> PureResult<([usize; 4], Vec<f32>)> {
    let first = images
        .first()
        .ok_or(TensorError::EmptyInput("vision_batch"))?;
    let (c, h, w) = first.shape();
    let len =
        first
            .as_slice()
            .len()
            .checked_mul(images.len())
            .ok_or(TensorError::InvalidValue {
                label: "vision_batch_volume",
            })?;
    for image in images {
        if image.shape() != first.shape() {
            return Err(TensorError::InvalidValue {
                label: "vision_batch_shape",
            });
        }
        validate_image_values(image.as_slice())?;
    }
    let mut values = Vec::with_capacity(len);
    for image in images {
        values.extend_from_slice(image.as_slice());
    }
    Ok(([images.len(), c, h, w], values))
}

impl TransformPipeline {
    fn validate_resident_shape(&self, shape: &[usize; 4]) -> PureResult<()> {
        let &[n, c, mut h, mut w] = shape;
        if n == 0 || c == 0 || h == 0 || w == 0 {
            return Err(TensorError::InvalidValue {
                label: "vision_batch_shape",
            });
        }
        for op in &self.ops {
            match op {
                TransformOperation::Normalize(op) => op.validate_channels(c)?,
                TransformOperation::Resize(op) => {
                    h = op.height;
                    w = op.width;
                }
                TransformOperation::CenterCrop(op) => {
                    if op.height > h || op.width > w {
                        return Err(TensorError::InvalidValue {
                            label: "center_crop_size",
                        });
                    }
                    h = op.height;
                    w = op.width;
                }
                TransformOperation::RandomHorizontalFlip(_) => {}
                TransformOperation::ColorJitter(_) => {
                    return Err(gpu_error(
                        "resident image pipeline does not support ColorJitter",
                    ))
                }
            }
        }
        Ok(())
    }

    /// One CHW image, including normalization, with no terminal readback.
    pub fn apply_resident(
        &mut self,
        image: &ImageTensor,
        device: &TensorDevice,
    ) -> PureResult<ResidentTensor> {
        let (c, h, w) = image.shape();
        let output = self.apply_packed_resident_batch(&[1, c, h, w], image.as_slice(), device)?;
        output
            .reshape(&output.layout().shape()[1..])
            .map_err(gpu_error)
    }

    /// Pack a homogeneous batch once. Statistics and compiled normalization
    /// plans are reused while the input layout and device stay unchanged.
    pub fn apply_resident_batch(
        &mut self,
        images: &[ImageTensor],
        device: &TensorDevice,
    ) -> PureResult<ResidentTensor> {
        let (shape, values) = pack_images(images)?;
        self.apply_packed_resident_batch(&shape, &values, device)
    }

    pub fn apply_packed_resident_batch(
        &mut self,
        shape: &[usize; 4],
        values: &[f32],
        device: &TensorDevice,
    ) -> PureResult<ResidentTensor> {
        self.validate_resident_shape(shape)?;
        let input = device.upload(shape, values).map_err(gpu_error)?;
        self.apply_from_resident(&input)
    }

    /// Apply Normalize and geometry in declared order to an NCHW tensor.
    /// Host validation/submission errors preserve the flip RNG. Successful
    /// submission advances it even if a deferred GPU guard later rejects data.
    pub fn apply_from_resident(&mut self, input: &ResidentTensor) -> PureResult<ResidentTensor> {
        let shape: &[usize; 4] =
            input
                .layout()
                .shape()
                .try_into()
                .map_err(|_| TensorError::InvalidValue {
                    label: "vision_batch_shape",
                })?;
        self.validate_resident_shape(shape)?;
        let dispatcher = self
            .dispatcher
            .clone()
            .ok_or_else(|| gpu_error("resident image pipeline requires a GPU dispatcher"))?;
        // Check the context before sampling. Empty geometry does not copy values.
        let mut current = dispatcher
            .run_geometry_batch_from_resident(input, &[])
            .map_err(map_dispatch_error)?;
        let mut rng = self.rng.clone();
        let count = self
            .ops
            .iter()
            .filter(|op| matches!(op, TransformOperation::RandomHorizontalFlip(_)))
            .count();
        let mut masks = vec![Vec::with_capacity(shape[0]); count];
        for _ in 0..shape[0] {
            let mut index = 0;
            for op in &self.ops {
                if let TransformOperation::RandomHorizontalFlip(op) = op {
                    masks[index].push(op.should_apply(&mut rng));
                    index += 1;
                }
            }
        }
        let mut masks = masks.into_iter();
        let mut index = 0;
        while index < self.ops.len() {
            if let TransformOperation::Normalize(op) = &self.ops[index] {
                let cache = &mut self.normalizers[index];
                if !cache
                    .as_ref()
                    .is_some_and(|prepared| prepared.matches(&current))
                {
                    *cache = Some(Shared::new(PreparedNormalize::new(op, &current)?));
                }
                current = cache.as_ref().unwrap().run(&current)?;
                index += 1;
                continue;
            }
            let mut commands = Vec::new();
            while index < self.ops.len() {
                commands.push(match &self.ops[index] {
                    TransformOperation::Resize(op) => BatchGeometryCommand::Resize {
                        height: op.height,
                        width: op.width,
                    },
                    TransformOperation::CenterCrop(op) => BatchGeometryCommand::CenterCrop {
                        height: op.height,
                        width: op.width,
                    },
                    TransformOperation::RandomHorizontalFlip(_) => {
                        BatchGeometryCommand::HorizontalFlip(masks.next().unwrap())
                    }
                    TransformOperation::Normalize(_) => break,
                    TransformOperation::ColorJitter(_) => {
                        unreachable!("preflight rejects unsupported operations")
                    }
                });
                index += 1;
            }
            current = dispatcher
                .run_geometry_batch_from_resident(&current, &commands)
                .map_err(map_dispatch_error)?;
        }
        self.rng = rng;
        Ok(current)
    }

    /// Explicit async host handoff. Unlike a resident submission, this waits
    /// for validity and commits neither the image nor RNG when mapping fails.
    pub async fn apply_gpu_async(
        &mut self,
        image: &mut ImageTensor,
        device: &TensorDevice,
    ) -> PureResult<()> {
        let mut candidate = self.clone();
        let output = candidate.apply_resident(image, device)?;
        let shape = output.layout().shape();
        let snapshot = output.snapshot().map_err(gpu_error)?;
        #[cfg(target_arch = "wasm32")]
        let values = snapshot.read_async().await.map_err(gpu_error)?;
        #[cfg(not(target_arch = "wasm32"))]
        let values = snapshot.read().map_err(gpu_error)?;
        *image = ImageTensor::new(shape[0], shape[1], shape[2], values)?;
        *self = candidate;
        Ok(())
    }
}

/// GPU images plus caller-owned host target/annotation metadata. No source
/// image copy is retained. Geometry currently transforms images, not boxes/masks.
#[derive(Clone, Debug)]
pub struct ResidentVisionBatch {
    pub images: ResidentTensor,
    pub targets: Vec<Option<Tensor>>,
    pub labels: Vec<Option<String>>,
    pub boxes: Vec<Option<Vec<[f32; 4]>>>,
    pub masks: Vec<Option<Vec<ImageTensor>>>,
}

impl ResidentVisionBatch {
    pub fn len(&self) -> usize {
        self.images.layout().shape()[0]
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upload explicit `(1, K)` targets as `(N, K)` on the image device. Labels
    /// are never converted to targets implicitly; missing/ragged targets fail.
    pub fn upload_targets(&self) -> PureResult<ResidentTensor> {
        let first = self
            .targets
            .first()
            .and_then(Option::as_ref)
            .ok_or(TensorError::EmptyInput("vision_targets"))?;
        let (rows, cols) = first.shape();
        if rows != 1 || cols == 0 || self.targets.len() != self.len() {
            return Err(TensorError::InvalidValue {
                label: "vision_target_shape",
            });
        }
        let capacity = self
            .len()
            .checked_mul(cols)
            .ok_or(TensorError::InvalidValue {
                label: "vision_target_volume",
            })?;
        let mut values = Vec::with_capacity(capacity);
        for target in &self.targets {
            let target = target
                .as_ref()
                .ok_or(TensorError::EmptyInput("vision_targets"))?;
            if target.shape() != (1, cols) {
                return Err(TensorError::InvalidValue {
                    label: "vision_target_shape",
                });
            }
            values.extend_from_slice(target.to_layout(st_tensor::Layout::RowMajor)?.data());
        }
        self.images
            .device()
            .upload(&[self.len(), cols], &values)
            .map_err(gpu_error)
    }
}

impl<D: VisionDataset> DataLoader<D> {
    /// Prepare one homogeneous image batch without a GPU-to-host boundary.
    /// Cursor/RNG commit after successful submission, not deferred GPU acceptance.
    pub fn next_resident_batch(
        &mut self,
        device: &TensorDevice,
    ) -> PureResult<Option<ResidentVisionBatch>> {
        let Some((end, batch)) = self.pending_batch()? else {
            return Ok(None);
        };
        let mut candidate = self.pipeline.clone();
        let images = if let Some(pipeline) = &mut candidate {
            pipeline.apply_resident_batch(&batch.images, device)?
        } else {
            let (shape, values) = pack_images(&batch.images)?;
            if shape.contains(&0) {
                return Err(TensorError::InvalidValue {
                    label: "vision_batch_shape",
                });
            }
            device.upload(&shape, &values).map_err(gpu_error)?
        };
        self.pipeline = candidate;
        self.position = end;
        Ok(Some(ResidentVisionBatch {
            images,
            targets: batch.targets,
            labels: batch.labels,
            boxes: batch.boxes,
            masks: batch.masks,
        }))
    }
}

#[cfg(test)]
mod tests;
