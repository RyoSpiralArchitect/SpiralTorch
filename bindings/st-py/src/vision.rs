use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use pyo3::{wrap_pyfunction, Bound, PyRefMut};

use crate::telemetry::PyAtlasFrame;
use crate::tensor::{tensor_err_to_py, PyTensor};
use crate::theory::PyZRelativityModel;
use st_core::telemetry::chrono::ChronoSummary;
use st_tensor::wasm_canvas::{
    CanvasPalette, FractalCanvas as PureFractalCanvas,
    InfiniteZSpacePatch as PureInfiniteZSpacePatch,
};
use st_tensor::{Tensor, TensorError};
use st_vision::{
    create_classification_model, dataset_catalog, find_dataset_descriptor, find_model_descriptor,
    model_catalog, standard_classification_pipeline, CenterCrop,
    ChronoSnapshot as PureChronoSnapshot, ColorJitter, DataLoader as PureVisionDataLoader,
    DatasetSample, FeatureStage, ImageTensor as PureImageTensor, ModelKind, ModelMetadata,
    Normalize, RandomHorizontalFlip, Resize, StreamedVolume as PureStreamedVolume,
    TensorVisionDataset, TransformOperation, TransformPipeline as PureTransformPipeline,
    VisionBatch, VisionDataset, VisionModel, VisionTask, ZSliceProfile as PureZSliceProfile,
    ZSpaceStreamFrame as PureZSpaceStreamFrame,
    ZSpaceStreamFrameAggregator as PureZSpaceStreamFrameAggregator,
    ZSpaceTelemetryReport as PureZSpaceTelemetryReport,
};

const MIN_SMOOTHING: f32 = 0.0;
const MAX_SMOOTHING: f32 = 0.999;

fn push_slice_stats(metrics: &mut HashMap<String, f64>, base: &str, data: &[f32]) {
    let total = data.len() as f64;
    let mut finite_count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for &value in data {
        if !value.is_finite() {
            continue;
        }
        finite_count += 1;
        let v = value as f64;
        sum += v;
        sum_abs += v.abs();
        sum_sq += v * v;
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }

    metrics.insert(
        format!("{base}.finite_fraction"),
        if total > 0.0 {
            finite_count as f64 / total
        } else {
            0.0
        },
    );

    if finite_count == 0 {
        return;
    }

    let denom = finite_count as f64;
    metrics.insert(format!("{base}.mean"), sum / denom);
    metrics.insert(format!("{base}.abs_mean"), sum_abs / denom);
    metrics.insert(format!("{base}.min"), min as f64);
    metrics.insert(format!("{base}.max"), max as f64);
    metrics.insert(format!("{base}.rms"), (sum_sq / denom).sqrt());
    metrics.insert(format!("{base}.l2"), sum_sq.sqrt());
}

fn tensor_to_rows(tensor: &Tensor) -> Vec<Vec<f32>> {
    let (rows, cols) = tensor.shape();
    let data = tensor.data();
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        out.push(data[start..start + cols].to_vec());
    }
    out
}

fn chrono_summary_from_any(summary: &Bound<'_, PyAny>) -> PyResult<ChronoSummary> {
    if let Ok(dict) = summary.downcast::<PyDict>() {
        let frames_obj = dict
            .get_item("frames")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'frames'"))?;
        let duration_obj = dict
            .get_item("duration")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'duration'"))?;
        let latest_timestamp_obj = dict
            .get_item("latest_timestamp")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'latest_timestamp'"))?;
        let mean_drift_obj = dict
            .get_item("mean_drift")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'mean_drift'"))?;
        let mean_abs_drift_obj = dict
            .get_item("mean_abs_drift")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'mean_abs_drift'"))?;
        let drift_std_obj = dict
            .get_item("drift_std")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'drift_std'"))?;
        let mean_energy_obj = dict
            .get_item("mean_energy")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'mean_energy'"))?;
        let energy_std_obj = dict
            .get_item("energy_std")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'energy_std'"))?;
        let mean_decay_obj = dict
            .get_item("mean_decay")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'mean_decay'"))?;
        let min_energy_obj = dict
            .get_item("min_energy")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'min_energy'"))?;
        let max_energy_obj = dict
            .get_item("max_energy")?
            .ok_or_else(|| PyValueError::new_err("summary missing key 'max_energy'"))?;
        return Ok(ChronoSummary {
            frames: frames_obj.extract()?,
            duration: duration_obj.extract()?,
            latest_timestamp: latest_timestamp_obj.extract()?,
            mean_drift: mean_drift_obj.extract()?,
            mean_abs_drift: mean_abs_drift_obj.extract()?,
            drift_std: drift_std_obj.extract()?,
            mean_energy: mean_energy_obj.extract()?,
            energy_std: energy_std_obj.extract()?,
            mean_decay: mean_decay_obj.extract()?,
            min_energy: min_energy_obj.extract()?,
            max_energy: max_energy_obj.extract()?,
        });
    }

    Ok(ChronoSummary {
        frames: summary.getattr("frames")?.extract()?,
        duration: summary.getattr("duration")?.extract()?,
        latest_timestamp: summary.getattr("latest_timestamp")?.extract()?,
        mean_drift: summary.getattr("mean_drift")?.extract()?,
        mean_abs_drift: summary.getattr("mean_abs_drift")?.extract()?,
        drift_std: summary.getattr("drift_std")?.extract()?,
        mean_energy: summary.getattr("mean_energy")?.extract()?,
        energy_std: summary.getattr("energy_std")?.extract()?,
        mean_decay: summary.getattr("mean_decay")?.extract()?,
        min_energy: summary.getattr("min_energy")?.extract()?,
        max_energy: summary.getattr("max_energy")?.extract()?,
    })
}

fn chrono_summary_to_dict(py: Python<'_>, summary: &ChronoSummary) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("frames", summary.frames)?;
    dict.set_item("duration", summary.duration)?;
    dict.set_item("latest_timestamp", summary.latest_timestamp)?;
    dict.set_item("mean_drift", summary.mean_drift)?;
    dict.set_item("mean_abs_drift", summary.mean_abs_drift)?;
    dict.set_item("drift_std", summary.drift_std)?;
    dict.set_item("mean_energy", summary.mean_energy)?;
    dict.set_item("energy_std", summary.energy_std)?;
    dict.set_item("mean_decay", summary.mean_decay)?;
    dict.set_item("min_energy", summary.min_energy)?;
    dict.set_item("max_energy", summary.max_energy)?;
    Ok(dict.into())
}

fn z_slice_profile_to_dict(py: Python<'_>, profile: &PureZSliceProfile) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("depth", profile.depth())?;
    dict.set_item("means", profile.means().to_vec())?;
    dict.set_item("stds", profile.stds().to_vec())?;
    dict.set_item("energies", profile.energies().to_vec())?;
    dict.set_item("average_mean", profile.average_mean())?;
    dict.set_item("average_std", profile.average_std())?;
    dict.set_item("average_energy", profile.average_energy())?;
    dict.set_item("total_energy", profile.total_energy())?;
    dict.set_item("max_energy", profile.max_energy())?;
    dict.set_item("energy_entropy", profile.energy_entropy())?;
    Ok(dict.into())
}

fn zspace_telemetry_to_dict(
    py: Python<'_>,
    report: &PureZSpaceTelemetryReport,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("depth", report.depth())?;
    dict.set_item("mean_intensity", report.mean_intensity())?;
    dict.set_item("mean_std", report.mean_std())?;
    dict.set_item("mean_energy", report.mean_energy())?;
    dict.set_item("total_energy", report.total_energy())?;
    dict.set_item("max_energy", report.max_energy())?;
    dict.set_item("energy_entropy", report.energy_entropy())?;
    Ok(dict.into())
}

fn vision_task_name(task: VisionTask) -> &'static str {
    match task {
        VisionTask::Classification => "classification",
        VisionTask::Segmentation => "segmentation",
        VisionTask::Detection => "detection",
        VisionTask::Keypoint => "keypoint",
        VisionTask::Video => "video",
        VisionTask::OpticalFlow => "optical_flow",
        VisionTask::Depth => "depth",
        VisionTask::MultiLabel => "multi_label",
    }
}

fn transform_category_name(
    category: st_vision::transforms::audit::TransformCategory,
) -> &'static str {
    match category {
        st_vision::transforms::audit::TransformCategory::Photometric => "photometric",
        st_vision::transforms::audit::TransformCategory::Geometric => "geometric",
        st_vision::transforms::audit::TransformCategory::Normalisation => "normalisation",
    }
}

fn dataset_descriptor_to_dict(
    py: Python<'_>,
    descriptor: &st_vision::DatasetDescriptor,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", descriptor.name)?;
    dict.set_item("task", vision_task_name(descriptor.task))?;
    dict.set_item("description", descriptor.description)?;
    dict.set_item("supports_download", descriptor.supports_download)?;
    dict.set_item("homepage", descriptor.homepage)?;
    dict.set_item("paper_url", descriptor.paper_url)?;
    Ok(dict.into())
}

fn model_descriptor_to_dict(
    py: Python<'_>,
    descriptor: &st_vision::ModelDescriptor,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", descriptor.name)?;
    dict.set_item("kind", format!("{:?}", descriptor.kind))?;
    dict.set_item("task", vision_task_name(descriptor.task))?;
    dict.set_item("default_input_channels", descriptor.default_input_channels)?;
    dict.set_item("default_image_size", descriptor.default_image_size)?;
    dict.set_item("has_pretrained", descriptor.has_pretrained)?;
    Ok(dict.into())
}

fn model_metadata_to_dict(py: Python<'_>, metadata: &ModelMetadata) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", metadata.name)?;
    dict.set_item("kind", format!("{:?}", metadata.kind))?;
    dict.set_item("task", vision_task_name(metadata.task))?;
    dict.set_item("input_channels", metadata.input_channels)?;
    dict.set_item("image_size", metadata.image_size)?;
    dict.set_item("num_classes", metadata.num_classes)?;
    dict.set_item("has_pretrained", metadata.has_pretrained)?;
    Ok(dict.into())
}

fn parse_model_kind(name: &str) -> PyResult<ModelKind> {
    find_model_descriptor(name)
        .map(|descriptor| descriptor.kind)
        .ok_or_else(|| PyValueError::new_err(format!("unknown vision model '{name}'")))
}

fn parse_feature_stage(stage: &str) -> PyResult<FeatureStage> {
    match stage.to_ascii_lowercase().as_str() {
        "stem" => Ok(FeatureStage::Stem),
        "head" | "features" | "feature" => Ok(FeatureStage::Head),
        "logits" | "logit" => Ok(FeatureStage::Logits),
        other => Err(PyValueError::new_err(format!(
            "unknown feature stage '{other}' (expected stem, head, or logits)"
        ))),
    }
}

fn box_tuples_to_arrays(boxes: Vec<(f32, f32, f32, f32)>) -> Vec<[f32; 4]> {
    boxes
        .into_iter()
        .map(|(x0, y0, x1, y1)| [x0, y0, x1, y1])
        .collect()
}

fn box_arrays_to_tuples(boxes: &[[f32; 4]]) -> Vec<(f32, f32, f32, f32)> {
    boxes
        .iter()
        .map(|bbox| (bbox[0], bbox[1], bbox[2], bbox[3]))
        .collect()
}

fn transform_audit_entry_to_dict(
    py: Python<'_>,
    entry: &st_vision::transforms::audit::TransformAuditEntry,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", entry.name)?;
    dict.set_item("category", transform_category_name(entry.category))?;
    dict.set_item("gpu_candidate", entry.gpu_candidate)?;
    dict.set_item("gpu_available", entry.gpu_available)?;
    dict.set_item("notes", entry.notes)?;
    Ok(dict.into())
}

fn transform_coverage_summary_to_dict(
    py: Python<'_>,
    summary: &st_vision::transforms::audit::TransformCoverageSummary,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let by_category = PyDict::new(py);
    for (category, count) in &summary.by_category {
        by_category.set_item(transform_category_name(*category), count)?;
    }
    dict.set_item("by_category", by_category)?;
    dict.set_item("gpu_supported", summary.gpu_supported)?;
    dict.set_item("gpu_candidates_missing", summary.gpu_candidates_missing)?;
    dict.set_item("total_stages", summary.total_stages)?;
    dict.set_item("unknown_operations", summary.unknown_operations)?;
    Ok(dict.into())
}

fn transform_pipeline_audit_to_dict(
    py: Python<'_>,
    report: &st_vision::transforms::audit::PipelineAuditReport,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let stages: PyResult<Vec<_>> = report
        .stages
        .iter()
        .map(|entry| transform_audit_entry_to_dict(py, entry))
        .collect();
    dict.set_item("stages", stages?)?;
    dict.set_item("unknown_operations", report.unknown_operations.clone())?;
    dict.set_item("has_gpu_dispatcher", report.has_gpu_dispatcher)?;
    let summary = st_vision::transforms::audit::summarize_pipeline(report);
    dict.set_item("summary", transform_coverage_summary_to_dict(py, &summary)?)?;
    Ok(dict.into())
}

#[pyclass(module = "spiraltorch.vision", name = "ImageTensor")]
#[derive(Clone)]
pub(crate) struct PyImageTensor {
    inner: PureImageTensor,
}

impl PyImageTensor {
    fn from_inner(inner: PureImageTensor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyImageTensor {
    #[new]
    fn new(channels: usize, height: usize, width: usize, data: Vec<f32>) -> PyResult<Self> {
        let inner =
            PureImageTensor::new(channels, height, width, data).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn zeros(channels: usize, height: usize, width: usize) -> PyResult<Self> {
        let inner = PureImageTensor::zeros(channels, height, width).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_tensor(
        tensor: &PyTensor,
        channels: usize,
        height: usize,
        width: usize,
    ) -> PyResult<Self> {
        let inner = PureImageTensor::from_tensor(&tensor.inner, channels, height, width)
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn channels(&self) -> usize {
        self.inner.channels()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn shape(&self) -> (usize, usize, usize) {
        self.inner.shape()
    }

    fn data(&self) -> Vec<f32> {
        self.inner.as_slice().to_vec()
    }

    fn pixel(&self, channel: usize, y: usize, x: usize) -> PyResult<f32> {
        self.inner.pixel(channel, y, x).map_err(tensor_err_to_py)
    }

    fn set_pixel(&mut self, channel: usize, y: usize, x: usize, value: f32) -> PyResult<()> {
        self.inner
            .set_pixel(channel, y, x, value)
            .map_err(tensor_err_to_py)
    }

    fn relu_inplace(&mut self) {
        self.inner.relu_inplace();
    }

    fn flatten(&self) -> Vec<f32> {
        self.inner.flatten()
    }

    fn to_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self.inner.into_tensor().map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("channels", self.inner.channels())?;
        dict.set_item("height", self.inner.height())?;
        dict.set_item("width", self.inner.width())?;
        dict.set_item("shape", self.inner.shape())?;
        dict.set_item("data", self.inner.as_slice().to_vec())?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageTensor(channels={}, height={}, width={})",
            self.inner.channels(),
            self.inner.height(),
            self.inner.width()
        )
    }
}

#[pyclass(module = "spiraltorch.vision", name = "TransformPipeline", unsendable)]
#[derive(Clone)]
pub(crate) struct PyTransformPipeline {
    inner: PureTransformPipeline,
}

impl PyTransformPipeline {
    fn from_inner(inner: PureTransformPipeline) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTransformPipeline {
    #[new]
    #[pyo3(signature = (*, seed=None))]
    fn new(seed: Option<u64>) -> Self {
        let inner = match seed {
            Some(value) => PureTransformPipeline::with_seed(value),
            None => PureTransformPipeline::new(),
        };
        Self { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (image_size=224, *, seed=None))]
    fn standard_classification(image_size: usize, seed: Option<u64>) -> PyResult<Self> {
        let inner = standard_classification_pipeline(image_size, seed).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    fn add_normalize(&mut self, means: Vec<f32>, stds: Vec<f32>) -> PyResult<()> {
        let op = Normalize::new(means, stds).map_err(tensor_err_to_py)?;
        self.inner.add(TransformOperation::Normalize(op));
        Ok(())
    }

    fn add_resize(&mut self, height: usize, width: usize) -> PyResult<()> {
        let op = Resize::new(height, width).map_err(tensor_err_to_py)?;
        self.inner.add(TransformOperation::Resize(op));
        Ok(())
    }

    fn add_center_crop(&mut self, height: usize, width: usize) -> PyResult<()> {
        let op = CenterCrop::new(height, width).map_err(tensor_err_to_py)?;
        self.inner.add(TransformOperation::CenterCrop(op));
        Ok(())
    }

    #[pyo3(signature = (probability=0.5))]
    fn add_horizontal_flip(&mut self, probability: f32) -> PyResult<()> {
        let op = RandomHorizontalFlip::new(probability).map_err(tensor_err_to_py)?;
        self.inner.add(TransformOperation::RandomHorizontalFlip(op));
        Ok(())
    }

    #[pyo3(signature = (*, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))]
    fn add_color_jitter(
        &mut self,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
    ) -> PyResult<()> {
        let op =
            ColorJitter::new(brightness, contrast, saturation, hue).map_err(tensor_err_to_py)?;
        self.inner.add(TransformOperation::ColorJitter(op));
        Ok(())
    }

    fn operations(&self) -> Vec<&'static str> {
        self.inner
            .operations()
            .iter()
            .map(TransformOperation::name)
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn has_gpu_dispatcher(&self) -> bool {
        self.inner.has_gpu_dispatcher()
    }

    fn apply(&mut self, image: &PyImageTensor) -> PyResult<PyImageTensor> {
        let mut inner = image.inner.clone();
        self.inner.apply(&mut inner).map_err(tensor_err_to_py)?;
        Ok(PyImageTensor::from_inner(inner))
    }

    fn apply_inplace(&mut self, mut image: PyRefMut<'_, PyImageTensor>) -> PyResult<()> {
        self.inner.apply(&mut image.inner).map_err(tensor_err_to_py)
    }

    fn audit(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let report = st_vision::transforms::audit::audit_pipeline(&self.inner);
        transform_pipeline_audit_to_dict(py, &report)
    }

    fn __repr__(&self) -> String {
        format!("TransformPipeline(operations={:?})", self.operations())
    }
}

#[pyclass(module = "spiraltorch.vision", name = "VisionSample")]
#[derive(Clone)]
pub(crate) struct PyVisionSample {
    inner: DatasetSample,
}

impl PyVisionSample {
    fn from_inner(inner: DatasetSample) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVisionSample {
    #[new]
    #[pyo3(signature = (image, *, target=None, label=None, boxes=None, masks=None))]
    fn new(
        image: &PyImageTensor,
        target: Option<&PyTensor>,
        label: Option<String>,
        boxes: Option<Vec<(f32, f32, f32, f32)>>,
        masks: Option<Vec<PyImageTensor>>,
    ) -> Self {
        let mut inner = DatasetSample::new(image.inner.clone());
        if let Some(target) = target {
            inner = inner.with_target(target.inner.clone());
        }
        if let Some(label) = label {
            inner = inner.with_label(label);
        }
        if let Some(boxes) = boxes {
            inner = inner.with_boxes(box_tuples_to_arrays(boxes));
        }
        if let Some(masks) = masks {
            inner = inner.with_masks(masks.into_iter().map(|mask| mask.inner).collect());
        }
        Self { inner }
    }

    #[getter]
    fn image(&self) -> PyImageTensor {
        PyImageTensor::from_inner(self.inner.image.clone())
    }

    #[getter]
    fn target(&self) -> Option<PyTensor> {
        self.inner
            .target
            .as_ref()
            .cloned()
            .map(PyTensor::from_tensor)
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }

    #[getter]
    fn boxes(&self) -> Option<Vec<(f32, f32, f32, f32)>> {
        self.inner.boxes.as_deref().map(box_arrays_to_tuples)
    }

    #[getter]
    fn masks(&self) -> Option<Vec<PyImageTensor>> {
        self.inner.masks.as_ref().map(|masks| {
            masks
                .iter()
                .cloned()
                .map(PyImageTensor::from_inner)
                .collect()
        })
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("image", self.image().to_dict(py)?)?;
        dict.set_item("target", self.inner.target.as_ref().map(tensor_to_rows))?;
        dict.set_item("label", self.inner.label.clone())?;
        dict.set_item("boxes", self.boxes())?;
        dict.set_item(
            "masks",
            self.masks()
                .map(|masks| {
                    masks
                        .into_iter()
                        .map(|mask| mask.to_dict(py))
                        .collect::<PyResult<Vec<_>>>()
                })
                .transpose()?,
        )?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "VisionSample(image_shape={:?}, label={:?})",
            self.inner.image.shape(),
            self.inner.label
        )
    }
}

#[pyclass(module = "spiraltorch.vision", name = "TensorVisionDataset")]
#[derive(Clone)]
pub(crate) struct PyTensorVisionDataset {
    inner: TensorVisionDataset,
}

#[pymethods]
impl PyTensorVisionDataset {
    #[new]
    #[pyo3(signature = (descriptor="CIFAR10"))]
    fn new(descriptor: &str) -> PyResult<Self> {
        let descriptor = find_dataset_descriptor(descriptor)
            .cloned()
            .ok_or_else(|| {
                PyValueError::new_err(format!("unknown vision dataset '{descriptor}'"))
            })?;
        Ok(Self {
            inner: TensorVisionDataset::new(descriptor),
        })
    }

    fn descriptor(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        dataset_descriptor_to_dict(py, self.inner.descriptor())
    }

    fn push_sample(&mut self, sample: &PyVisionSample) -> PyResult<()> {
        self.inner
            .push_sample(sample.inner.clone())
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (image, *, target=None, label=None, boxes=None, masks=None))]
    fn push(
        &mut self,
        image: &PyImageTensor,
        target: Option<&PyTensor>,
        label: Option<String>,
        boxes: Option<Vec<(f32, f32, f32, f32)>>,
        masks: Option<Vec<PyImageTensor>>,
    ) -> PyResult<()> {
        let sample = PyVisionSample::new(image, target, label, boxes, masks);
        self.push_sample(&sample)
    }

    fn get(&self, index: usize) -> PyResult<PyVisionSample> {
        self.inner
            .get(index)
            .map(PyVisionSample::from_inner)
            .map_err(tensor_err_to_py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[pyo3(signature = (batch_size, *, seed=None, pipeline=None, shuffle=false))]
    fn dataloader(
        &self,
        batch_size: usize,
        seed: Option<u64>,
        pipeline: Option<&PyTransformPipeline>,
        shuffle: bool,
    ) -> PyResult<PyVisionDataLoader> {
        let dataset = Arc::new(self.inner.clone());
        let mut inner =
            PureVisionDataLoader::new(dataset, batch_size, seed).map_err(tensor_err_to_py)?;
        if let Some(pipeline) = pipeline {
            inner = inner.with_pipeline(pipeline.inner.clone());
        }
        inner.enable_shuffle(shuffle);
        Ok(PyVisionDataLoader { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorVisionDataset(name={}, len={})",
            self.inner.descriptor().name,
            self.inner.len()
        )
    }
}

#[pyclass(module = "spiraltorch.vision", name = "VisionBatch")]
#[derive(Clone)]
pub(crate) struct PyVisionBatch {
    inner: VisionBatch,
}

impl PyVisionBatch {
    fn from_inner(inner: VisionBatch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVisionBatch {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn images(&self) -> Vec<PyImageTensor> {
        self.inner
            .images
            .iter()
            .cloned()
            .map(PyImageTensor::from_inner)
            .collect()
    }

    fn targets(&self) -> Vec<Option<PyTensor>> {
        self.inner
            .targets
            .iter()
            .map(|target| target.as_ref().cloned().map(PyTensor::from_tensor))
            .collect()
    }

    fn labels(&self) -> Vec<Option<String>> {
        self.inner.labels.clone()
    }

    fn boxes(&self) -> Vec<Option<Vec<(f32, f32, f32, f32)>>> {
        self.inner
            .boxes
            .iter()
            .map(|boxes| boxes.as_deref().map(box_arrays_to_tuples))
            .collect()
    }

    fn stack(&self) -> PyResult<PyTensor> {
        self.inner
            .stack()
            .map(PyTensor::from_tensor)
            .map_err(tensor_err_to_py)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("len", self.inner.len())?;
        dict.set_item("labels", self.labels())?;
        dict.set_item("boxes", self.boxes())?;
        dict.set_item(
            "image_shapes",
            self.inner
                .images
                .iter()
                .map(PureImageTensor::shape)
                .collect::<Vec<_>>(),
        )?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!("VisionBatch(len={})", self.inner.len())
    }
}

#[pyclass(module = "spiraltorch.vision", name = "VisionDataLoader", unsendable)]
pub(crate) struct PyVisionDataLoader {
    inner: PureVisionDataLoader<TensorVisionDataset>,
}

#[pymethods]
impl PyVisionDataLoader {
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn enable_shuffle(&mut self, shuffle: bool) {
        self.inner.enable_shuffle(shuffle);
    }

    fn next_batch(&mut self) -> PyResult<Option<PyVisionBatch>> {
        self.inner
            .next_batch()
            .map(|batch| batch.map(PyVisionBatch::from_inner))
            .map_err(tensor_err_to_py)
    }

    fn __repr__(&self) -> String {
        "VisionDataLoader(...)".to_string()
    }
}

#[pyclass(module = "spiraltorch.vision", name = "VisionModel")]
#[derive(Clone)]
pub(crate) struct PyVisionModel {
    inner: Arc<dyn VisionModel>,
}

impl PyVisionModel {
    fn from_inner(inner: Arc<dyn VisionModel>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVisionModel {
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        model_metadata_to_dict(py, self.inner.metadata())
    }

    fn forward(&self, images: Vec<PyImageTensor>) -> PyResult<PyTensor> {
        let images: Vec<_> = images.into_iter().map(|image| image.inner).collect();
        self.inner
            .forward(&images)
            .map(PyTensor::from_tensor)
            .map_err(tensor_err_to_py)
    }

    #[pyo3(signature = (stage, image))]
    fn extract_features(&self, stage: &str, image: &PyImageTensor) -> PyResult<PyTensor> {
        let stage = parse_feature_stage(stage)?;
        self.inner
            .extract_features(stage, &image.inner)
            .map(PyTensor::from_tensor)
            .map_err(tensor_err_to_py)
    }

    fn __repr__(&self) -> String {
        let metadata = self.inner.metadata();
        format!(
            "VisionModel(name={}, num_classes={})",
            metadata.name, metadata.num_classes
        )
    }
}

#[derive(Clone, Copy)]
enum TapeTarget {
    Hypergrad,
    Realgrad,
}

#[pyclass(module = "spiraltorch.vision", name = "ChronoSnapshot")]
#[derive(Clone)]
pub(crate) struct PyChronoSnapshot {
    inner: PureChronoSnapshot,
}

impl PyChronoSnapshot {
    fn from_inner(inner: PureChronoSnapshot) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyChronoSnapshot {
    #[new]
    #[pyo3(signature = (summary, *, dt=0.0))]
    fn new(summary: &Bound<'_, PyAny>, dt: f32) -> PyResult<Self> {
        let parsed = chrono_summary_from_any(summary)?;
        Ok(Self {
            inner: PureChronoSnapshot::new(parsed, dt),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (
        *,
        frames,
        duration,
        latest_timestamp,
        mean_drift,
        mean_abs_drift,
        drift_std,
        mean_energy,
        energy_std,
        mean_decay,
        min_energy,
        max_energy,
        dt=0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_values(
        frames: usize,
        duration: f32,
        latest_timestamp: f32,
        mean_drift: f32,
        mean_abs_drift: f32,
        drift_std: f32,
        mean_energy: f32,
        energy_std: f32,
        mean_decay: f32,
        min_energy: f32,
        max_energy: f32,
        dt: f32,
    ) -> Self {
        let summary = ChronoSummary {
            frames,
            duration,
            latest_timestamp,
            mean_drift,
            mean_abs_drift,
            drift_std,
            mean_energy,
            energy_std,
            mean_decay,
            min_energy,
            max_energy,
        };
        Self {
            inner: PureChronoSnapshot::new(summary, dt),
        }
    }

    #[getter]
    fn timestamp(&self) -> f32 {
        self.inner.timestamp()
    }

    #[getter]
    fn dt(&self) -> f32 {
        self.inner.dt()
    }

    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        chrono_summary_to_dict(py, self.inner.summary())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("timestamp", self.timestamp())?;
        dict.set_item("dt", self.dt())?;
        dict.set_item("summary", chrono_summary_to_dict(py, self.inner.summary())?)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "ChronoSnapshot(timestamp={:.6}, dt={:.6}, frames={})",
            self.inner.timestamp(),
            self.inner.dt(),
            self.inner.summary().frames
        )
    }
}

#[pyclass(module = "spiraltorch.vision", name = "ZSpaceStreamFrame", unsendable)]
#[derive(Clone)]
pub(crate) struct PyZSpaceStreamFrame {
    inner: PureZSpaceStreamFrame,
}

#[pymethods]
impl PyZSpaceStreamFrame {
    #[new]
    #[pyo3(signature = (slices, *, atlas_frame=None, chrono_snapshot=None))]
    fn new(
        py: Python<'_>,
        slices: Vec<Py<PyTensor>>,
        atlas_frame: Option<Py<PyAtlasFrame>>,
        chrono_snapshot: Option<Py<PyChronoSnapshot>>,
    ) -> PyResult<Self> {
        let tensors: Vec<Tensor> = slices
            .into_iter()
            .map(|tensor| tensor.borrow(py).inner.clone())
            .collect();
        let mut inner = PureZSpaceStreamFrame::new(tensors).map_err(tensor_err_to_py)?;
        if let Some(frame) = atlas_frame {
            inner = inner.with_atlas(frame.borrow(py).to_frame());
        }
        if let Some(snapshot) = chrono_snapshot {
            inner = inner.with_snapshot(snapshot.borrow(py).inner.clone());
        }
        Ok(Self { inner })
    }

    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    #[getter]
    fn slice_shape(&self) -> (usize, usize) {
        self.inner.slice_shape()
    }

    fn slices(&self) -> Vec<PyTensor> {
        self.inner
            .slices()
            .iter()
            .cloned()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn with_atlas(&mut self, frame: &PyAtlasFrame) {
        self.inner = self.inner.clone().with_atlas(frame.to_frame());
    }

    fn with_snapshot(&mut self, snapshot: &PyChronoSnapshot) {
        self.inner = self.inner.clone().with_snapshot(snapshot.inner.clone());
    }

    #[getter]
    fn atlas_frame(&self) -> Option<PyAtlasFrame> {
        self.inner
            .atlas_frame()
            .cloned()
            .map(PyAtlasFrame::from_frame)
    }

    #[getter]
    fn chrono_snapshot(&self) -> Option<PyChronoSnapshot> {
        self.inner
            .chrono_snapshot()
            .cloned()
            .map(PyChronoSnapshot::from_inner)
    }

    fn profile(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let profile = self.inner.profile().map_err(tensor_err_to_py)?;
        z_slice_profile_to_dict(py, &profile)
    }

    fn telemetry_report(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let report = self.inner.telemetry_report().map_err(tensor_err_to_py)?;
        zspace_telemetry_to_dict(py, &report)
    }

    fn to_streamed_volume(&self) -> PyResult<PyStreamedVolume> {
        let streamed = self.inner.to_streamed_volume().map_err(tensor_err_to_py)?;
        Ok(PyStreamedVolume::from_inner(streamed))
    }

    #[allow(clippy::wrong_self_convention)]
    fn into_streamed_volume(&self) -> PyResult<PyStreamedVolume> {
        let streamed = self
            .inner
            .clone()
            .into_streamed_volume()
            .map_err(tensor_err_to_py)?;
        Ok(PyStreamedVolume::from_inner(streamed))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("depth", self.inner.depth())?;
        let (rows, cols) = self.inner.slice_shape();
        dict.set_item("slice_shape", (rows, cols))?;
        let slice_rows: Vec<Vec<Vec<f32>>> =
            self.inner.slices().iter().map(tensor_to_rows).collect();
        dict.set_item("slices", slice_rows)?;

        let atlas_payload = if let Some(atlas) = self.inner.atlas_frame() {
            let atlas_dict = PyDict::new(py);
            atlas_dict.set_item("timestamp", atlas.timestamp)?;
            atlas_dict.set_item("metrics", atlas.metrics.len())?;
            atlas_dict.set_item("notes", atlas.notes.len())?;
            Some(atlas_dict.into_py(py))
        } else {
            None
        };
        dict.set_item("atlas_frame", atlas_payload)?;

        let snapshot_payload = if let Some(snapshot) = self.inner.chrono_snapshot() {
            Some(PyChronoSnapshot::from_inner(snapshot.clone()).to_dict(py)?)
        } else {
            None
        };
        dict.set_item("chrono_snapshot", snapshot_payload)?;
        dict.set_item("profile", self.profile(py)?)?;
        dict.set_item("telemetry_report", self.telemetry_report(py)?)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        let (rows, cols) = self.inner.slice_shape();
        format!(
            "ZSpaceStreamFrame(depth={}, slice_shape=({}, {}), atlas={}, snapshot={})",
            self.inner.depth(),
            rows,
            cols,
            self.inner.atlas_frame().is_some(),
            self.inner.chrono_snapshot().is_some()
        )
    }
}

#[pyclass(module = "spiraltorch.vision", name = "StreamedVolume", unsendable)]
#[derive(Clone)]
pub(crate) struct PyStreamedVolume {
    inner: PureStreamedVolume,
}

impl PyStreamedVolume {
    fn from_inner(inner: PureStreamedVolume) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStreamedVolume {
    #[staticmethod]
    fn from_frame(frame: &PyZSpaceStreamFrame) -> PyResult<Self> {
        let streamed = frame.inner.to_streamed_volume().map_err(tensor_err_to_py)?;
        Ok(Self { inner: streamed })
    }

    #[getter]
    fn depth(&self) -> usize {
        self.inner.volume.depth()
    }

    #[getter]
    fn slice_shape(&self) -> (usize, usize) {
        (self.inner.volume.height(), self.inner.volume.width())
    }

    fn slices(&self) -> PyResult<Vec<PyTensor>> {
        let mut slices = Vec::with_capacity(self.inner.volume.depth());
        for index in 0..self.inner.volume.depth() {
            let slice = self.inner.volume.slice(index).map_err(tensor_err_to_py)?;
            slices.push(PyTensor::from_tensor(slice));
        }
        Ok(slices)
    }

    #[getter]
    fn atlas_frame(&self) -> Option<PyAtlasFrame> {
        self.inner
            .atlas_frame
            .as_ref()
            .cloned()
            .map(PyAtlasFrame::from_frame)
    }

    #[getter]
    fn chrono_snapshot(&self) -> Option<PyChronoSnapshot> {
        self.inner
            .chrono_snapshot
            .as_ref()
            .cloned()
            .map(PyChronoSnapshot::from_inner)
    }

    fn profile(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let profile = self.inner.profile().map_err(tensor_err_to_py)?;
        z_slice_profile_to_dict(py, &profile)
    }

    fn telemetry_report(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let report = self.inner.telemetry_report().map_err(tensor_err_to_py)?;
        zspace_telemetry_to_dict(py, &report)
    }

    #[pyo3(signature = (frame, *, alpha=0.5))]
    fn ingest_frame(
        &mut self,
        frame: &PyZSpaceStreamFrame,
        alpha: f32,
    ) -> PyResult<PyStreamedVolume> {
        let streamed = self
            .inner
            .volume
            .ingest_stream_frame(frame.inner.clone(), alpha)
            .map_err(tensor_err_to_py)?;
        self.inner = streamed.clone();
        Ok(PyStreamedVolume::from_inner(streamed))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("depth", self.inner.volume.depth())?;
        dict.set_item(
            "slice_shape",
            (self.inner.volume.height(), self.inner.volume.width()),
        )?;

        let mut slice_rows = Vec::with_capacity(self.inner.volume.depth());
        for index in 0..self.inner.volume.depth() {
            let slice = self.inner.volume.slice(index).map_err(tensor_err_to_py)?;
            slice_rows.push(tensor_to_rows(&slice));
        }
        dict.set_item("slices", slice_rows)?;

        let atlas_payload = if let Some(atlas) = self.inner.atlas_frame.as_ref() {
            let atlas_dict = PyDict::new(py);
            atlas_dict.set_item("timestamp", atlas.timestamp)?;
            atlas_dict.set_item("metrics", atlas.metrics.len())?;
            atlas_dict.set_item("notes", atlas.notes.len())?;
            Some(atlas_dict.into_py(py))
        } else {
            None
        };
        dict.set_item("atlas_frame", atlas_payload)?;

        let snapshot_payload = self
            .inner
            .chrono_snapshot
            .as_ref()
            .cloned()
            .map(PyChronoSnapshot::from_inner)
            .map(|snapshot| snapshot.to_dict(py))
            .transpose()?;
        dict.set_item("chrono_snapshot", snapshot_payload)?;
        dict.set_item("profile", self.profile(py)?)?;
        dict.set_item("telemetry_report", self.telemetry_report(py)?)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamedVolume(depth={}, slice_shape=({}, {}), atlas={}, snapshot={})",
            self.inner.volume.depth(),
            self.inner.volume.height(),
            self.inner.volume.width(),
            self.inner.atlas_frame.is_some(),
            self.inner.chrono_snapshot.is_some()
        )
    }
}

#[pyclass(
    module = "spiraltorch.vision",
    name = "ZSpaceStreamFrameAggregator",
    unsendable
)]
#[derive(Clone)]
pub(crate) struct PyZSpaceStreamFrameAggregator {
    inner: PureZSpaceStreamFrameAggregator,
}

#[pymethods]
impl PyZSpaceStreamFrameAggregator {
    #[new]
    #[pyo3(signature = (*, max_depth=None))]
    fn new(max_depth: Option<usize>) -> PyResult<Self> {
        let inner = match max_depth {
            Some(depth) => {
                PureZSpaceStreamFrameAggregator::with_max_depth(depth).map_err(tensor_err_to_py)?
            }
            None => PureZSpaceStreamFrameAggregator::new(),
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    fn with_max_depth(max_depth: usize) -> PyResult<Self> {
        let inner =
            PureZSpaceStreamFrameAggregator::with_max_depth(max_depth).map_err(tensor_err_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn max_depth(&self) -> Option<usize> {
        self.inner.max_depth()
    }

    #[pyo3(signature = (max_depth=None))]
    fn set_max_depth(&mut self, max_depth: Option<usize>) -> PyResult<()> {
        self.inner
            .set_max_depth(max_depth)
            .map_err(tensor_err_to_py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    fn slice_shape(&self) -> Option<(usize, usize)> {
        self.inner.slice_shape()
    }

    fn extend(&mut self, frame: &PyZSpaceStreamFrame) -> PyResult<()> {
        self.inner
            .extend(frame.inner.clone())
            .map_err(tensor_err_to_py)
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn prune_oldest(&mut self, keep: usize) {
        self.inner.prune_oldest(keep);
    }

    fn as_frame(&self) -> Option<PyZSpaceStreamFrame> {
        self.inner
            .as_frame()
            .map(|inner| PyZSpaceStreamFrame { inner })
    }

    fn drain_frame(&mut self) -> PyResult<Option<PyZSpaceStreamFrame>> {
        let max_depth = self.inner.max_depth();
        let frame = std::mem::take(&mut self.inner)
            .into_frame()
            .map(|inner| PyZSpaceStreamFrame { inner });
        self.inner = match max_depth {
            Some(depth) => {
                PureZSpaceStreamFrameAggregator::with_max_depth(depth).map_err(tensor_err_to_py)?
            }
            None => PureZSpaceStreamFrameAggregator::new(),
        };
        Ok(frame)
    }

    fn to_streamed_volume(&self) -> PyResult<Option<PyStreamedVolume>> {
        self.inner
            .as_frame()
            .map(|frame| {
                frame
                    .to_streamed_volume()
                    .map(PyStreamedVolume::from_inner)
                    .map_err(tensor_err_to_py)
            })
            .transpose()
    }

    #[getter]
    fn atlas_frame(&self) -> Option<PyAtlasFrame> {
        self.inner
            .as_frame()
            .and_then(|frame| frame.atlas_frame().cloned().map(PyAtlasFrame::from_frame))
    }

    #[getter]
    fn chrono_snapshot(&self) -> Option<PyChronoSnapshot> {
        self.inner.as_frame().and_then(|frame| {
            frame
                .chrono_snapshot()
                .cloned()
                .map(PyChronoSnapshot::from_inner)
        })
    }

    fn telemetry_report(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let report = self.inner.telemetry_report().map_err(tensor_err_to_py)?;
        report
            .as_ref()
            .map(|value| zspace_telemetry_to_dict(py, value))
            .transpose()
    }

    fn profile(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .as_frame()
            .map(|frame| frame.profile().map_err(tensor_err_to_py))
            .transpose()?
            .as_ref()
            .map(|value| z_slice_profile_to_dict(py, value))
            .transpose()
    }

    fn __repr__(&self) -> String {
        format!(
            "ZSpaceStreamFrameAggregator(len={}, max_depth={:?}, slice_shape={:?})",
            self.inner.len(),
            self.inner.max_depth(),
            self.inner.slice_shape()
        )
    }
}

#[pyclass(module = "spiraltorch", name = "CanvasSnapshot")]
pub(crate) struct PyCanvasSnapshot {
    canvas: Vec<Vec<f32>>,
    hypergrad: Vec<Vec<f32>>,
    realgrad: Vec<Vec<f32>>,
    summary: HashMap<String, HashMap<String, f32>>,
    patch: Option<Vec<Vec<f32>>>,
}

impl PyCanvasSnapshot {
    fn new(
        canvas: Vec<Vec<f32>>,
        hypergrad: Vec<Vec<f32>>,
        realgrad: Vec<Vec<f32>>,
        summary: HashMap<String, HashMap<String, f32>>,
        patch: Option<Vec<Vec<f32>>>,
    ) -> Self {
        Self {
            canvas,
            hypergrad,
            realgrad,
            summary,
            patch,
        }
    }
}

#[pymethods]
impl PyCanvasSnapshot {
    #[getter]
    fn canvas(&self) -> Vec<Vec<f32>> {
        self.canvas.clone()
    }

    #[getter]
    fn hypergrad(&self) -> Vec<Vec<f32>> {
        self.hypergrad.clone()
    }

    #[getter]
    fn realgrad(&self) -> Vec<Vec<f32>> {
        self.realgrad.clone()
    }

    #[getter]
    fn summary(&self) -> HashMap<String, HashMap<String, f32>> {
        self.summary.clone()
    }

    #[getter]
    fn patch(&self) -> Option<Vec<Vec<f32>>> {
        self.patch.clone()
    }

    #[setter]
    fn set_patch(&mut self, patch: Option<Vec<Vec<f32>>>) {
        self.patch = patch;
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CanvasSnapshot(canvas_shape=({rows}, {cols}))",
            rows = self.canvas.len(),
            cols = self.canvas.first().map(|row| row.len()).unwrap_or(0)
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "CanvasTransformer", unsendable)]
pub(crate) struct PyCanvasTransformer {
    width: usize,
    height: usize,
    smoothing: f32,
    canvas: Vec<f32>,
    hypergrad: Vec<f32>,
    realgrad: Vec<f32>,
}

#[pymethods]
impl PyCanvasTransformer {
    #[new]
    #[pyo3(signature = (width, height, *, smoothing=0.85))]
    fn new(width: usize, height: usize, smoothing: f32) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must be positive"));
        }
        let clamped = smoothing.clamp(MIN_SMOOTHING, MAX_SMOOTHING);
        let size = width * height;
        Ok(Self {
            width,
            height,
            smoothing: clamped,
            canvas: vec![0.0; size],
            hypergrad: vec![0.0; size],
            realgrad: vec![0.0; size],
        })
    }

    #[getter]
    fn smoothing(&self) -> f32 {
        self.smoothing
    }

    fn refresh(
        &mut self,
        py: Python<'_>,
        projection: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<f32>>> {
        self.refresh_from_any(py, projection)
    }

    fn accumulate_hypergrad(
        &mut self,
        py: Python<'_>,
        gradient: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.accumulate_from_any(py, gradient, TapeTarget::Hypergrad)
    }

    fn accumulate_realgrad(&mut self, py: Python<'_>, gradient: &Bound<'_, PyAny>) -> PyResult<()> {
        self.accumulate_from_any(py, gradient, TapeTarget::Realgrad)
    }

    fn reset(&mut self) {
        for store in [&mut self.canvas, &mut self.hypergrad, &mut self.realgrad] {
            store.fill(0.0);
        }
    }

    fn gradient_summary(&self) -> HashMap<String, HashMap<String, f32>> {
        self.gradient_summary_map()
    }

    #[pyo3(signature = (vision, weight=1.0))]
    fn emit_zspace_patch(
        &mut self,
        py: Python<'_>,
        vision: &Bound<'_, PyAny>,
        weight: f32,
    ) -> PyResult<Vec<Vec<f32>>> {
        let projection = vision.call_method0("project").map_err(|err| {
            PyValueError::new_err(format!(
                "failed to obtain projection from vision pipeline: {err}"
            ))
        })?;
        let mut patch = self.refresh_from_any(py, &projection)?;
        if (weight - 1.0).abs() > f32::EPSILON {
            let scale = weight;
            for row in &mut patch {
                for value in row {
                    *value *= scale;
                }
            }
        }
        Ok(patch)
    }

    fn canvas(&self) -> Vec<Vec<f32>> {
        matrix_rows_from_slice(&self.canvas, self.height, self.width)
    }

    fn hypergrad(&self) -> Vec<Vec<f32>> {
        matrix_rows_from_slice(&self.hypergrad, self.height, self.width)
    }

    fn realgrad(&self) -> Vec<Vec<f32>> {
        matrix_rows_from_slice(&self.realgrad, self.height, self.width)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("width", self.width)?;
        dict.set_item("height", self.height)?;
        dict.set_item("smoothing", self.smoothing)?;
        dict.set_item("canvas", self.canvas())?;
        dict.set_item("hypergrad", self.hypergrad())?;
        dict.set_item("realgrad", self.realgrad())?;
        Ok(dict.into())
    }

    #[pyo3(signature = (state, *, strict=true))]
    fn load_state_dict(
        &mut self,
        py: Python<'_>,
        state: &Bound<'_, PyAny>,
        strict: bool,
    ) -> PyResult<()> {
        let mapping = state
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("state must be a mapping with string keys"))?;

        let mut width = self.width;
        let mut height = self.height;

        if let Some(value) = mapping.get_item("width")? {
            width = value.extract()?;
        }
        if let Some(value) = mapping.get_item("height")? {
            height = value.extract()?;
        }
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err(
                "state width and height must be positive",
            ));
        }
        if strict && (width != self.width || height != self.height) {
            return Err(PyValueError::new_err(
                "state dimensions do not match the canvas transformer",
            ));
        }

        if width != self.width || height != self.height {
            self.resize(width, height);
        }

        if let Some(value) = mapping.get_item("smoothing")? {
            self.smoothing = value.extract::<f32>()?.clamp(MIN_SMOOTHING, MAX_SMOOTHING);
        }

        if let Some(matrix) = mapping.get_item("canvas")? {
            let values = coerce_matrix(py, &matrix, self.height, self.width)?;
            self.canvas.copy_from_slice(&values);
        }

        if let Some(matrix) = mapping.get_item("hypergrad")? {
            let values = coerce_matrix(py, &matrix, self.height, self.width)?;
            self.hypergrad.copy_from_slice(&values);
        }

        if let Some(matrix) = mapping.get_item("realgrad")? {
            let values = coerce_matrix(py, &matrix, self.height, self.width)?;
            self.realgrad.copy_from_slice(&values);
        }

        Ok(())
    }

    fn snapshot(&self) -> PyCanvasSnapshot {
        self.snapshot_with_patch(None)
    }
}

impl PyCanvasTransformer {
    fn resize(&mut self, width: usize, height: usize) {
        let size = width * height;
        self.width = width;
        self.height = height;
        self.canvas.resize(size, 0.0);
        self.hypergrad.resize(size, 0.0);
        self.realgrad.resize(size, 0.0);
        self.canvas.fill(0.0);
        self.hypergrad.fill(0.0);
        self.realgrad.fill(0.0);
    }

    fn refresh_from_any(
        &mut self,
        py: Python<'_>,
        projection: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let matrix = coerce_matrix(py, projection, self.height, self.width)?;
        self.apply_smoothing(&matrix);
        let patch = matrix_rows_from_vec(matrix, self.height, self.width);
        Ok(patch)
    }

    fn apply_smoothing(&mut self, matrix: &[f32]) {
        let alpha = 1.0 - self.smoothing;
        for (dst, &src) in self.canvas.iter_mut().zip(matrix.iter()) {
            *dst = self.smoothing * *dst + alpha * src;
        }
    }

    fn accumulate_from_any(
        &mut self,
        py: Python<'_>,
        gradient: &Bound<'_, PyAny>,
        target: TapeTarget,
    ) -> PyResult<()> {
        let values = coerce_matrix(py, gradient, self.height, self.width)?;
        let store = match target {
            TapeTarget::Hypergrad => &mut self.hypergrad,
            TapeTarget::Realgrad => &mut self.realgrad,
        };
        for (dst, src) in store.iter_mut().zip(values.into_iter()) {
            *dst += src;
        }
        Ok(())
    }

    fn gradient_summary_map(&self) -> HashMap<String, HashMap<String, f32>> {
        let mut summary = HashMap::new();
        summary.insert("hypergrad".to_string(), matrix_summary(&self.hypergrad));
        summary.insert("realgrad".to_string(), matrix_summary(&self.realgrad));
        summary
    }

    fn snapshot_with_patch(&self, patch: Option<Vec<Vec<f32>>>) -> PyCanvasSnapshot {
        PyCanvasSnapshot::new(
            self.canvas(),
            self.hypergrad(),
            self.realgrad(),
            self.gradient_summary_map(),
            patch,
        )
    }
}

fn coerce_matrix(
    py: Python<'_>,
    any: &Bound<'_, PyAny>,
    height: usize,
    width: usize,
) -> PyResult<Vec<f32>> {
    if let Ok(tensor) = any.extract::<Py<PyTensor>>() {
        let tensor_ref = tensor.borrow(py);
        let (rows, cols) = tensor_ref.inner.shape();
        if rows != height || cols != width {
            return Err(PyValueError::new_err(format!(
                "expected tensor shape ({height}, {width}), received ({rows}, {cols})"
            )));
        }
        return Ok(tensor_ref.inner.data().to_vec());
    }

    let rows: Vec<Vec<f32>> = any.extract().map_err(|_| {
        PyTypeError::new_err("expected a Tensor or sequence of sequences of floats")
    })?;
    if rows.len() != height {
        return Err(PyValueError::new_err(format!(
            "expected {height} rows, received {}",
            rows.len()
        )));
    }
    let mut out = Vec::with_capacity(height * width);
    for (idx, row) in rows.iter().enumerate() {
        if row.len() != width {
            return Err(PyValueError::new_err(format!(
                "row {idx} expected {width} columns, received {}",
                row.len()
            )));
        }
        out.extend(row.iter().copied());
    }
    Ok(out)
}

fn matrix_rows_from_slice(data: &[f32], height: usize, width: usize) -> Vec<Vec<f32>> {
    data.chunks(width)
        .take(height)
        .map(|row| row.to_vec())
        .collect()
}

fn matrix_rows_from_vec(data: Vec<f32>, height: usize, width: usize) -> Vec<Vec<f32>> {
    data.chunks(width)
        .take(height)
        .map(|row| row.to_vec())
        .collect()
}

fn matrix_summary(data: &[f32]) -> HashMap<String, f32> {
    let mut summary = HashMap::new();
    if data.is_empty() {
        summary.insert("l1".to_string(), 0.0);
        summary.insert("l2".to_string(), 0.0);
        summary.insert("linf".to_string(), 0.0);
        summary.insert("mean".to_string(), 0.0);
        return summary;
    }

    let mut l1 = 0.0f32;
    let mut l2 = 0.0f32;
    let mut linf = 0.0f32;
    let mut sum = 0.0f32;
    for &value in data {
        let abs = value.abs();
        l1 += abs;
        l2 += value * value;
        linf = linf.max(abs);
        sum += value;
    }
    summary.insert("l1".to_string(), l1);
    summary.insert("l2".to_string(), l2.sqrt());
    summary.insert("linf".to_string(), linf);
    summary.insert("mean".to_string(), sum / data.len() as f32);
    summary
}

#[pyclass(module = "spiraltorch.vision", name = "InfiniteZSpacePatch")]
#[derive(Clone)]
pub(crate) struct PyInfiniteZSpacePatch {
    inner: PureInfiniteZSpacePatch,
}

impl From<PureInfiniteZSpacePatch> for PyInfiniteZSpacePatch {
    fn from(inner: PureInfiniteZSpacePatch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyInfiniteZSpacePatch {
    #[getter]
    pub fn dimension(&self) -> f32 {
        self.inner.dimension()
    }

    #[getter]
    pub fn zoom(&self) -> f32 {
        self.inner.zoom()
    }

    #[getter]
    pub fn support(&self) -> (f32, f32) {
        self.inner.support()
    }

    #[getter]
    pub fn mellin_weights(&self) -> Vec<f32> {
        self.inner.mellin_weights().to_vec()
    }

    #[getter]
    pub fn density(&self) -> Vec<f32> {
        self.inner.density().to_vec()
    }

    pub fn eta_bar(&self) -> f32 {
        self.inner.eta_bar()
    }
}

#[pyclass(module = "spiraltorch.vision", name = "FractalCanvas")]
#[derive(Clone)]
pub(crate) struct PyFractalCanvas {
    inner: PureFractalCanvas,
    dimension: f32,
}

#[pymethods]
impl PyFractalCanvas {
    #[new]
    #[pyo3(signature = (dimension=2.0, capacity=32, width=64, height=64))]
    pub fn new(dimension: f32, capacity: usize, width: usize, height: usize) -> PyResult<Self> {
        let canvas = PureFractalCanvas::new(capacity.max(4), width.max(1), height.max(1))
            .map_err(tensor_err_to_py)?;
        Ok(Self {
            inner: canvas,
            dimension,
        })
    }

    #[getter]
    pub fn dimension(&self) -> f32 {
        self.dimension
    }

    pub fn set_dimension(&mut self, dimension: f32) {
        self.dimension = dimension;
    }

    #[pyo3(signature = (dimension=None, zoom=None, steps=None))]
    pub fn emit_zspace_patch(
        &self,
        dimension: Option<f32>,
        zoom: Option<f32>,
        steps: Option<usize>,
    ) -> PyResult<PyInfiniteZSpacePatch> {
        let dim = dimension.unwrap_or(self.dimension);
        let zoom_value = zoom.unwrap_or(f32::INFINITY);
        let step_value = steps.unwrap_or(96);
        let patch = self
            .inner
            .emit_zspace_patch(dim, zoom_value, step_value)
            .map_err(tensor_err_to_py)?;
        Ok(PyInfiniteZSpacePatch::from(patch))
    }

    #[pyo3(signature = (dimension=None))]
    pub fn emit_zspace_infinite(&self, dimension: Option<f32>) -> PyResult<PyInfiniteZSpacePatch> {
        let dim = dimension.unwrap_or(self.dimension);
        let patch = self
            .inner
            .emit_zspace_infinite(dim)
            .map_err(tensor_err_to_py)?;
        Ok(PyInfiniteZSpacePatch::from(patch))
    }

    #[pyo3(signature = (zoom=None, steps=None, dimension=None))]
    pub fn emit_infinite_z(
        &self,
        zoom: Option<f32>,
        steps: Option<usize>,
        dimension: Option<f32>,
    ) -> PyResult<PyInfiniteZSpacePatch> {
        self.emit_zspace_patch(dimension, zoom, steps)
    }
}

#[pyclass(module = "spiraltorch.canvas", name = "CanvasProjector", unsendable)]
pub(crate) struct PyCanvasProjector {
    inner: PureFractalCanvas,
}

#[pymethods]
impl PyCanvasProjector {
    #[new]
    #[pyo3(signature = (width=64, height=64, *, capacity=32, palette="blue-magenta"))]
    fn new(width: usize, height: usize, capacity: usize, palette: &str) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must be positive"));
        }
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be positive"));
        }
        let mut canvas =
            PureFractalCanvas::new(capacity, width, height).map_err(tensor_err_to_py)?;
        let palette = CanvasPalette::parse(palette)
            .ok_or_else(|| PyValueError::new_err(format!("unknown palette '{palette}'")))?;
        canvas.projector_mut().set_palette(palette);
        Ok(Self { inner: canvas })
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    fn queue_len(&self) -> usize {
        self.inner.scheduler().len()
    }

    fn total_weight(&self) -> f32 {
        self.inner.scheduler().total_weight()
    }

    fn palette(&self) -> &'static str {
        self.inner.projector().palette().canonical_name()
    }

    fn set_palette(&mut self, name: &str) -> PyResult<()> {
        let palette = CanvasPalette::parse(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown palette '{name}'")))?;
        self.inner.projector_mut().set_palette(palette);
        Ok(())
    }

    fn reset_normalizer(&mut self) {
        self.inner.projector_mut().normalizer_mut().reset();
    }

    #[pyo3(signature = (relation, *, coherence=1.0, tension=1.0, depth=0))]
    fn push_patch(
        &self,
        py: Python<'_>,
        relation: &Bound<'_, PyAny>,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> PyResult<()> {
        let height = self.inner.height();
        let width = self.inner.width();
        let flat = coerce_matrix(py, relation, height, width)?;
        let tensor = Tensor::from_vec(height, width, flat).map_err(tensor_err_to_py)?;
        self.inner
            .push_patch(tensor, coherence, tension, depth)
            .map_err(tensor_err_to_py)
    }

    /// Emits a Z-space patch derived from the current colour-energy field.
    ///
    /// This is useful for feedback loops: call `emit_zspace_patch(...)` and
    /// feed the returned `relation` back into `push_patch(...)`.
    #[pyo3(signature = (*, coherence=1.0, tension=1.0, depth=0))]
    fn emit_zspace_patch(
        &mut self,
        py: Python<'_>,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> PyResult<Py<PyDict>> {
        let patch = self
            .inner
            .projector_mut()
            .emit_zspace_patch(coherence, tension, depth)
            .map_err(tensor_err_to_py)?;
        let relation = Py::new(py, PyTensor::from_tensor(patch.relation().clone()))?;
        let dict = PyDict::new(py);
        dict.set_item("relation", relation)?;
        dict.set_item("coherence", patch.coherence())?;
        dict.set_item("tension", patch.tension())?;
        dict.set_item("depth", patch.depth())?;
        dict.set_item("weight", patch.weight())?;
        Ok(dict.into())
    }

    /// Emits an AR/WebGPU-friendly trail packet derived from the current vector field.
    #[pyo3(signature = (curvature=1.0))]
    fn emit_wasm_trail(&mut self, py: Python<'_>, curvature: f32) -> PyResult<Py<PyDict>> {
        let trail = self
            .inner
            .projector_mut()
            .emit_wasm_trail(curvature)
            .map_err(tensor_err_to_py)?;
        let samples = trail.samples().len();
        let flat = trail.as_f32_slice();
        let tensor = Tensor::from_vec(samples, 7, flat).map_err(tensor_err_to_py)?;
        let samples_tensor = Py::new(py, PyTensor::from_tensor(tensor))?;

        let dict = PyDict::new(py);
        dict.set_item("curvature", trail.curvature())?;
        dict.set_item("width", trail.width())?;
        dict.set_item("height", trail.height())?;
        dict.set_item("samples", samples_tensor)?;
        Ok(dict.into())
    }

    /// Emits an `AtlasFrame` (telemetry packet) describing the current canvas state.
    #[pyo3(signature = (*, prefix="canvas", refresh=true, timestamp=None))]
    fn emit_atlas_frame(
        &mut self,
        prefix: &str,
        refresh: bool,
        timestamp: Option<f64>,
    ) -> PyResult<PyAtlasFrame> {
        let prefix = prefix.trim();
        if prefix.is_empty() {
            return Err(PyValueError::new_err("prefix must be non-empty"));
        }

        if refresh {
            match self.inner.refresh_tensor() {
                Ok(_) | Err(TensorError::EmptyInput(_)) => {}
                Err(err) => return Err(tensor_err_to_py(err)),
            }
        }

        let projector = self.inner.projector();
        let tensor = projector.tensor();
        let (rows, cols) = tensor.shape();
        let mut metrics: HashMap<String, f64> = HashMap::new();
        metrics.insert(
            format!("{prefix}.queue_len"),
            self.inner.scheduler().len() as f64,
        );
        metrics.insert(
            format!("{prefix}.total_weight"),
            self.inner.scheduler().total_weight() as f64,
        );
        metrics.insert(format!("{prefix}.shape.rows"), rows as f64);
        metrics.insert(format!("{prefix}.shape.cols"), cols as f64);

        push_slice_stats(&mut metrics, &format!("{prefix}.energy"), tensor.data());

        let palette = projector.palette();
        let (palette_id, palette_blue_magenta, palette_turbo, palette_grayscale) = match palette {
            CanvasPalette::BlueMagenta => (0.0, 1.0, 0.0, 0.0),
            CanvasPalette::Turbo => (1.0, 0.0, 1.0, 0.0),
            CanvasPalette::Grayscale => (2.0, 0.0, 0.0, 1.0),
        };
        metrics.insert(format!("{prefix}.palette.id"), palette_id);
        metrics.insert(
            format!("{prefix}.palette.blue_magenta"),
            palette_blue_magenta,
        );
        metrics.insert(format!("{prefix}.palette.turbo"), palette_turbo);
        metrics.insert(format!("{prefix}.palette.grayscale"), palette_grayscale);

        let normalizer = projector.normalizer();
        let normalizer_state = normalizer.state();
        metrics.insert(
            format!("{prefix}.normalizer.alpha"),
            normalizer.alpha() as f64,
        );
        metrics.insert(
            format!("{prefix}.normalizer.epsilon"),
            normalizer.epsilon() as f64,
        );
        metrics.insert(
            format!("{prefix}.normalizer.has_state"),
            if normalizer_state.is_some() { 1.0 } else { 0.0 },
        );
        if let Some((min, max)) = normalizer_state {
            metrics.insert(format!("{prefix}.normalizer.min"), min as f64);
            metrics.insert(format!("{prefix}.normalizer.max"), max as f64);
            metrics.insert(format!("{prefix}.normalizer.span"), (max - min) as f64);
            metrics.insert(
                format!("{prefix}.normalizer.center"),
                ((min + max) * 0.5) as f64,
            );
        }

        let field = projector.vector_field();
        let vectors = field.vectors();
        metrics.insert(format!("{prefix}.trail.samples"), vectors.len() as f64);

        if !vectors.is_empty() {
            let curvature = 1.0f32;
            let curvature_scale = curvature.tanh();
            metrics.insert(format!("{prefix}.trail.curvature"), curvature as f64);
            metrics.insert(
                format!("{prefix}.trail.curvature_scale"),
                curvature_scale as f64,
            );

            let mut energy_total = vectors.len() as f64;
            let mut energy_finite = 0usize;
            let mut energy_sum = 0.0f64;
            let mut energy_sum_abs = 0.0f64;
            let mut energy_sum_sq = 0.0f64;
            let mut energy_min = f32::INFINITY;
            let mut energy_max = f32::NEG_INFINITY;

            let mut z_finite = 0usize;
            let mut z_sum = 0.0f64;
            let mut z_sum_abs = 0.0f64;
            let mut z_sum_sq = 0.0f64;
            let mut z_min = f32::INFINITY;
            let mut z_max = f32::NEG_INFINITY;

            let mut chroma_total = 0usize;
            let mut chroma_finite = 0usize;
            let mut chroma_sum_abs = 0.0f64;
            let mut chroma_sum_sq = 0.0f64;
            let mut chroma_min = f32::INFINITY;
            let mut chroma_max = f32::NEG_INFINITY;

            for vector in vectors {
                let energy = vector[0];
                if energy.is_finite() {
                    energy_finite += 1;
                    let v = energy as f64;
                    energy_sum += v;
                    energy_sum_abs += v.abs();
                    energy_sum_sq += v * v;
                    if energy < energy_min {
                        energy_min = energy;
                    }
                    if energy > energy_max {
                        energy_max = energy;
                    }

                    let z = curvature_scale * energy;
                    if z.is_finite() {
                        z_finite += 1;
                        let z64 = z as f64;
                        z_sum += z64;
                        z_sum_abs += z64.abs();
                        z_sum_sq += z64 * z64;
                        if z < z_min {
                            z_min = z;
                        }
                        if z > z_max {
                            z_max = z;
                        }
                    }
                }

                for channel in vector.iter().skip(1) {
                    chroma_total += 1;
                    let c = *channel;
                    if !c.is_finite() {
                        continue;
                    }
                    chroma_finite += 1;
                    let c64 = c as f64;
                    chroma_sum_abs += c64.abs();
                    chroma_sum_sq += c64 * c64;
                    if c < chroma_min {
                        chroma_min = c;
                    }
                    if c > chroma_max {
                        chroma_max = c;
                    }
                }
            }

            if energy_total <= 0.0 {
                energy_total = 0.0;
            }
            metrics.insert(
                format!("{prefix}.trail.energy.finite_fraction"),
                if energy_total > 0.0 {
                    energy_finite as f64 / energy_total
                } else {
                    0.0
                },
            );
            if energy_finite > 0 {
                let denom = energy_finite as f64;
                metrics.insert(format!("{prefix}.trail.energy.mean"), energy_sum / denom);
                metrics.insert(
                    format!("{prefix}.trail.energy.abs_mean"),
                    energy_sum_abs / denom,
                );
                metrics.insert(format!("{prefix}.trail.energy.min"), energy_min as f64);
                metrics.insert(format!("{prefix}.trail.energy.max"), energy_max as f64);
                metrics.insert(
                    format!("{prefix}.trail.energy.rms"),
                    (energy_sum_sq / denom).sqrt(),
                );
                metrics.insert(format!("{prefix}.trail.energy.l2"), energy_sum_sq.sqrt());
            }

            metrics.insert(
                format!("{prefix}.trail.z.finite_fraction"),
                if energy_total > 0.0 {
                    z_finite as f64 / energy_total
                } else {
                    0.0
                },
            );
            if z_finite > 0 {
                let denom = z_finite as f64;
                metrics.insert(format!("{prefix}.trail.z.mean"), z_sum / denom);
                metrics.insert(format!("{prefix}.trail.z.abs_mean"), z_sum_abs / denom);
                metrics.insert(format!("{prefix}.trail.z.min"), z_min as f64);
                metrics.insert(format!("{prefix}.trail.z.max"), z_max as f64);
                metrics.insert(format!("{prefix}.trail.z.rms"), (z_sum_sq / denom).sqrt());
                metrics.insert(format!("{prefix}.trail.z.l2"), z_sum_sq.sqrt());
            }

            metrics.insert(
                format!("{prefix}.trail.chroma.finite_fraction"),
                if chroma_total > 0 {
                    chroma_finite as f64 / chroma_total as f64
                } else {
                    0.0
                },
            );
            if chroma_finite > 0 {
                let denom = chroma_finite as f64;
                metrics.insert(
                    format!("{prefix}.trail.chroma.abs_mean"),
                    chroma_sum_abs / denom,
                );
                metrics.insert(format!("{prefix}.trail.chroma.min"), chroma_min as f64);
                metrics.insert(format!("{prefix}.trail.chroma.max"), chroma_max as f64);
                metrics.insert(
                    format!("{prefix}.trail.chroma.rms"),
                    (chroma_sum_sq / denom).sqrt(),
                );
                metrics.insert(format!("{prefix}.trail.chroma.l2"), chroma_sum_sq.sqrt());
            }
        }

        if let Ok(fft_db) = field.fft_rows_power_db_tensor(false) {
            let (fft_rows, fft_cols) = fft_db.shape();
            metrics.insert(format!("{prefix}.fft_db.shape.rows"), fft_rows as f64);
            metrics.insert(format!("{prefix}.fft_db.shape.cols"), fft_cols as f64);
            push_slice_stats(&mut metrics, &format!("{prefix}.fft_db"), fft_db.data());

            let mut counts = [0usize; 4];
            let mut sums = [0.0f64; 4];
            let mut mins = [f32::INFINITY; 4];
            let mut maxs = [f32::NEG_INFINITY; 4];

            for (idx, &value) in fft_db.data().iter().enumerate() {
                if !value.is_finite() {
                    continue;
                }
                let ch = idx % 4;
                counts[ch] += 1;
                sums[ch] += value as f64;
                if value < mins[ch] {
                    mins[ch] = value;
                }
                if value > maxs[ch] {
                    maxs[ch] = value;
                }
            }

            let labels = ["energy", "chroma_r", "chroma_g", "chroma_b"];
            for (idx, label) in labels.iter().enumerate() {
                let count = counts[idx];
                if count == 0 {
                    continue;
                }
                let denom = count as f64;
                metrics.insert(format!("{prefix}.fft_db.{label}.mean"), sums[idx] / denom);
                metrics.insert(format!("{prefix}.fft_db.{label}.min"), mins[idx] as f64);
                metrics.insert(format!("{prefix}.fft_db.{label}.max"), maxs[idx] as f64);
            }
        }

        PyAtlasFrame::from_metrics(metrics, timestamp)
    }

    fn clear_queue(&self) {
        while self.inner.scheduler().pop().is_some() {}
    }

    /// Returns the current RGBA surface (without forcing a refresh).
    fn rgba<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.projector().surface().as_rgba())
    }

    /// Refreshes the projector and returns the RGBA surface as bytes.
    fn refresh_rgba<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match self.inner.refresh() {
            Ok(rgba) => Ok(PyBytes::new(py, rgba)),
            Err(TensorError::EmptyInput(_)) => Ok(self.rgba(py)),
            Err(err) => Err(tensor_err_to_py(err)),
        }
    }

    /// Refreshes the projector and returns the latest relation tensor.
    fn refresh_tensor(&mut self, py: Python<'_>) -> PyResult<PyTensor> {
        let _ = py;
        match self.inner.refresh_tensor() {
            Ok(tensor) => Ok(PyTensor::from_tensor(tensor.clone())),
            Err(TensorError::EmptyInput(_)) => Ok(PyTensor::from_tensor(
                self.inner.projector().tensor().clone(),
            )),
            Err(err) => Err(tensor_err_to_py(err)),
        }
    }

    /// Returns the last relation tensor (without forcing a refresh).
    fn tensor(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let _ = py;
        Ok(PyTensor::from_tensor(
            self.inner.projector().tensor().clone(),
        ))
    }

    /// Refreshes and returns the row-wise complex FFT spectrum as a tensor with shape `(height, width * 8)`.
    #[pyo3(signature = (*, inverse=false))]
    fn refresh_vector_fft_tensor(&mut self, inverse: bool) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .projector_mut()
            .refresh_vector_fft_tensor(inverse)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    /// Refreshes and returns the row-wise FFT log-power (dB) tensor with shape `(height, width * 4)`.
    #[pyo3(signature = (*, inverse=false))]
    fn refresh_vector_fft_power_db_tensor(&mut self, inverse: bool) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .projector_mut()
            .refresh_vector_fft_power_db_tensor(inverse)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CanvasProjector(width={width}, height={height}, palette={palette})",
            width = self.inner.width(),
            height = self.inner.height(),
            palette = self.palette(),
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (vision, canvas, *, hypergrad=None, realgrad=None, weight=1.0, include_patch=false))]
fn apply_vision_update(
    py: Python<'_>,
    vision: &Bound<'_, PyAny>,
    canvas: &Bound<'_, PyAny>,
    hypergrad: Option<&Bound<'_, PyAny>>,
    realgrad: Option<&Bound<'_, PyAny>>,
    weight: f32,
    include_patch: bool,
) -> PyResult<PyCanvasSnapshot> {
    let mut canvas_ref: PyRefMut<PyCanvasTransformer> = canvas.extract()?;
    let patch = canvas_ref.emit_zspace_patch(py, vision, weight)?;
    if let Some(hyper) = hypergrad {
        canvas_ref.accumulate_from_any(py, hyper, TapeTarget::Hypergrad)?;
    }
    if let Some(real) = realgrad {
        canvas_ref.accumulate_from_any(py, real, TapeTarget::Realgrad)?;
    }
    let patch_opt = if include_patch { Some(patch) } else { None };
    Ok(canvas_ref.snapshot_with_patch(patch_opt))
}

#[pyfunction]
fn canvas_available_palettes() -> Vec<&'static str> {
    CanvasPalette::ALL
        .iter()
        .map(|palette| palette.canonical_name())
        .collect()
}

#[pyfunction]
fn canvas_canonical_palette(name: &str) -> PyResult<&'static str> {
    CanvasPalette::parse(name)
        .map(|palette| palette.canonical_name())
        .ok_or_else(|| PyValueError::new_err(format!("unknown palette '{name}'")))
}

#[pyfunction]
fn vision_dataset_catalog(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    dataset_catalog()
        .iter()
        .map(|descriptor| dataset_descriptor_to_dict(py, descriptor))
        .collect()
}

#[pyfunction]
fn vision_dataset_descriptor(py: Python<'_>, name: &str) -> PyResult<Option<Py<PyDict>>> {
    find_dataset_descriptor(name)
        .map(|descriptor| dataset_descriptor_to_dict(py, descriptor))
        .transpose()
}

#[pyfunction]
fn vision_model_catalog(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    model_catalog()
        .iter()
        .map(|descriptor| model_descriptor_to_dict(py, descriptor))
        .collect()
}

#[pyfunction]
fn vision_model_descriptor(py: Python<'_>, name: &str) -> PyResult<Option<Py<PyDict>>> {
    find_model_descriptor(name)
        .map(|descriptor| model_descriptor_to_dict(py, descriptor))
        .transpose()
}

#[pyfunction]
fn vision_transform_audit_catalog(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    st_vision::transforms::audit::audit_cpu_transforms()
        .iter()
        .map(|entry| transform_audit_entry_to_dict(py, entry))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (image_size=224, *, seed=None))]
fn vision_standard_classification_pipeline(
    image_size: usize,
    seed: Option<u64>,
) -> PyResult<PyTransformPipeline> {
    let inner = standard_classification_pipeline(image_size, seed).map_err(tensor_err_to_py)?;
    Ok(PyTransformPipeline::from_inner(inner))
}

#[pyfunction]
#[pyo3(signature = (kind, num_classes=10, *, seed=None))]
fn vision_create_classification_model(
    kind: &str,
    num_classes: usize,
    seed: Option<u64>,
) -> PyResult<PyVisionModel> {
    let kind = parse_model_kind(kind)?;
    let model = create_classification_model(kind, num_classes, seed).map_err(tensor_err_to_py)?;
    Ok(PyVisionModel::from_inner(model))
}

#[pyfunction]
#[pyo3(signature = (model, field="block"))]
pub fn zrelativity_heatmap(model: &PyZRelativityModel, field: &str) -> PyResult<Vec<Vec<f32>>> {
    let bundle = model.inner.tensor_bundle().map_err(tensor_err_to_py)?;
    let lower = field.to_ascii_lowercase();
    let tensor = match lower.as_str() {
        "block" => &bundle.block_metric,
        "effective" => &bundle.effective_metric,
        "gauge" => &bundle.gauge_field,
        "moduli" => &bundle.scalar_moduli,
        "field_equation" => &bundle.field_equation,
        "warp" => bundle
            .warp
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("warp factor not present on model"))?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown ZRelativity field '{other}' (expected block/effective/gauge/moduli/field_equation/warp)"
            )))
        }
    };
    Ok(tensor_to_rows(tensor))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<PyImageTensor>()?;
    parent.add_class::<PyTransformPipeline>()?;
    parent.add_class::<PyVisionSample>()?;
    parent.add_class::<PyTensorVisionDataset>()?;
    parent.add_class::<PyVisionBatch>()?;
    parent.add_class::<PyVisionDataLoader>()?;
    parent.add_class::<PyVisionModel>()?;
    parent.add_class::<PyChronoSnapshot>()?;
    parent.add_class::<PyZSpaceStreamFrame>()?;
    parent.add_class::<PyStreamedVolume>()?;
    parent.add_class::<PyZSpaceStreamFrameAggregator>()?;
    parent.add_class::<PyCanvasTransformer>()?;
    parent.add_class::<PyCanvasSnapshot>()?;
    parent.add_class::<PyCanvasProjector>()?;
    parent.add_class::<PyFractalCanvas>()?;
    parent.add_class::<PyInfiniteZSpacePatch>()?;
    parent.add_function(wrap_pyfunction!(apply_vision_update, parent)?)?;
    parent.add_function(wrap_pyfunction!(canvas_available_palettes, parent)?)?;
    parent.add_function(wrap_pyfunction!(canvas_canonical_palette, parent)?)?;
    parent.add_function(wrap_pyfunction!(vision_dataset_catalog, parent)?)?;
    parent.add_function(wrap_pyfunction!(vision_dataset_descriptor, parent)?)?;
    parent.add_function(wrap_pyfunction!(vision_model_catalog, parent)?)?;
    parent.add_function(wrap_pyfunction!(vision_model_descriptor, parent)?)?;
    parent.add_function(wrap_pyfunction!(vision_transform_audit_catalog, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        vision_standard_classification_pipeline,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        vision_create_classification_model,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(zrelativity_heatmap, parent)?)?;
    parent.add("__doc__", "Canvas transformer utilities")?;
    let _ = py;
    Ok(())
}
