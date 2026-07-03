mod sot;

use crate::sot::{PySoT3DPlan, Sot3DParams};

use ndarray::{Array2, ArrayD, Ix2};
use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use pyo3::PyRef;
use pyo3::PyRefMut;
use st_backend_hip::{
    device_info as hip_device_info, hip_available as hip_runtime_available,
    DeviceInfo as HipDeviceInfo,
};
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_core::backend::unison_heuristics::RankKind;
use st_core::ops::compaction::{plan_compaction, CompactionPlan};
use st_core::ops::rank_entry::{plan_rank, RankPlan};
#[cfg(any(feature = "psi", feature = "psychoid"))]
use st_core::telemetry::hub;
use st_frac::fft::{fft_inplace as frac_fft_inplace, Complex32 as FracComplex32, FftError};
use st_frac::{
    fracdiff_gl_nd, fracdiff_gl_nd_backward, gl_coeffs as frac_gl_coeffs, FracErr, Pad as FracPad,
};
use st_nn::dataset::DataLoaderBatches as NnDataLoaderBatches;
use st_nn::dataset_from_vec as nn_dataset_from_vec;
use st_nn::{
    byte_lm_corpus_windows as nn_byte_lm_corpus_windows,
    byte_lm_sample_stats as nn_byte_lm_sample_stats, byte_lm_windows as nn_byte_lm_windows,
    interleave_replay_samples as nn_interleave_replay_samples,
    padded_byte_lm_samples as nn_padded_byte_lm_samples, summarize_epoch_history,
    ByteLmSampleStats, Conv1d as NnConv1d, DataLoader as NnDataLoader, DifferentialTrace,
    DistConfig, DistMode, EarlyStoppingConfig, EpochBestState, EpochHistory,
    EpochSparseRetentionBestState, EpochStats, EpochValidationBestState, Linear as NnLinear,
    LoraLinear as NnLoraLinear, Loss, LrPlateauConfig, MeanSquaredError, Module, ModuleTrainer,
    ParameterMovementReport, ParameterTrainingFingerprint, Relu as NnRelu, RoundtableConfig,
    RoundtableSchedule, Sequential as NnSequential, SoftmaxCrossEntropy, SparseClassificationDelta,
    SparseClassificationMetrics, SparseFineTuneRegressionLimits, SparseFineTuneReport,
    SparseFineTuneReportSummary, SparseRetentionGuardConfig, SpiralSession, SpiralSessionBuilder,
    StateCompatibilityReport, StateFingerprint, StateKeyMapRule, StateLoadReport,
    StateTensorTransform, TrainerStateFingerprint, TrainingResumeFingerprint,
    ValidationTrainingControls, WaveRnn as NnWaveRnn, ZSpaceProjector as NnZSpaceProjector,
    BYTE_LM_VOCAB,
};
use st_tensor::pure::{
    measure::{
        z_space_barycenter as rust_z_space_barycenter, BarycenterIntermediate, ZSpaceBarycenter,
    },
    topos::OpenCartesianTopos,
    AmegaHypergrad, Complex32, ComplexTensor, DifferentialResonance, LanguageWaveEncoder,
    PureResult, Tensor, TensorBiome, TensorError,
};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

fn tensor_err(err: TensorError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn convert<T>(value: PureResult<T>) -> PyResult<T> {
    value.map_err(tensor_err)
}

fn convert_frac<T>(value: Result<T, FracErr>) -> PyResult<T> {
    value.map_err(|err| PyValueError::new_err(err.to_string()))
}

fn convert_fft<T>(value: Result<T, FftError>) -> PyResult<T> {
    value.map_err(|err| {
        PyValueError::new_err(match err {
            FftError::Empty => "signal must not be empty".to_string(),
            FftError::NonPowerOfTwo => {
                "signal length must be a power of two for the radix FFT".to_string()
            }
        })
    })
}

fn intern_label(label: &str) -> &'static str {
    static INTERNER: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let storage = INTERNER.get_or_init(|| Mutex::new(Vec::new()));
    {
        let guard = storage.lock().expect("intern labels lock poisoned");
        if let Some(&existing) = guard.iter().find(|&&item| item == label) {
            return existing;
        }
    }
    let leaked: &'static str = Box::leak(label.to_owned().into_boxed_str());
    let mut guard = storage.lock().expect("intern labels lock poisoned");
    guard.push(leaked);
    leaked
}

fn state_to_pydict(py: Python<'_>, state: HashMap<String, Tensor>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (name, tensor) in state {
        let py_tensor = PyTensor::from_tensor(tensor);
        dict.set_item(name, py_tensor.into_py(py))?;
    }
    Ok(dict.into_py(py))
}

fn pydict_to_state(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Tensor>> {
    let mut state = HashMap::new();
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        let tensor: PyTensor = value.extract()?;
        state.insert(name, tensor.as_tensor().clone());
    }
    Ok(state)
}

fn pydict_to_key_rules(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, StateKeyMapRule>> {
    let mut rules = HashMap::new();
    for (key, value) in dict.iter() {
        let source: String = key.extract()?;
        if let Ok(target) = value.extract::<String>() {
            rules.insert(source, StateKeyMapRule::new(target));
            continue;
        }

        let rule_dict: HashMap<String, String> = value.extract().map_err(|_| {
            PyValueError::new_err(
                "state key rules must map source keys to a target string or ".to_string()
                    + "{'target': str, 'transform': str}",
            )
        })?;
        let target = rule_dict
            .get("target")
            .ok_or_else(|| PyValueError::new_err("state key rule missing 'target'"))?
            .clone();
        let transform = match rule_dict.get("transform") {
            Some(value) => convert(StateTensorTransform::parse(value))?,
            None => StateTensorTransform::Identity,
        };
        rules.insert(source, StateKeyMapRule::with_transform(target, transform));
    }
    Ok(rules)
}

fn set_module_trainable<M: Module>(module: &mut M, trainable: bool) -> PyResult<()> {
    convert(module.set_trainable(trainable))
}

fn set_module_parameter_trainable<M: Module>(
    module: &mut M,
    name: &str,
    trainable: bool,
) -> PyResult<()> {
    convert(module.set_parameter_trainable(name, trainable))
}

fn set_module_parameters_trainable_by_prefix<M: Module>(
    module: &mut M,
    prefix: &str,
    trainable: bool,
) -> PyResult<usize> {
    convert(module.set_parameters_trainable_by_prefix(prefix, trainable))
}

fn set_module_parameters_trainable_by_suffix<M: Module>(
    module: &mut M,
    suffix: &str,
    trainable: bool,
) -> PyResult<usize> {
    convert(module.set_parameters_trainable_by_suffix(suffix, trainable))
}

fn set_module_parameters_trainable_by_contains<M: Module>(
    module: &mut M,
    needle: &str,
    trainable: bool,
) -> PyResult<usize> {
    convert(module.set_parameters_trainable_by_contains(needle, trainable))
}

fn scale_module_parameter_learning_rate<M: Module>(
    module: &mut M,
    name: &str,
    factor: f32,
) -> PyResult<()> {
    convert(module.scale_parameter_learning_rate(name, factor))
}

fn set_module_parameter_learning_rate_scale<M: Module>(
    module: &mut M,
    name: &str,
    scale: f32,
) -> PyResult<()> {
    convert(module.set_parameter_learning_rate_scale(name, scale))
}

fn scale_module_parameters_learning_rate_by_prefix<M: Module>(
    module: &mut M,
    prefix: &str,
    factor: f32,
) -> PyResult<usize> {
    convert(module.scale_parameters_learning_rate_by_prefix(prefix, factor))
}

fn scale_module_parameters_learning_rate_by_suffix<M: Module>(
    module: &mut M,
    suffix: &str,
    factor: f32,
) -> PyResult<usize> {
    convert(module.scale_parameters_learning_rate_by_suffix(suffix, factor))
}

fn scale_module_parameters_learning_rate_by_contains<M: Module>(
    module: &mut M,
    needle: &str,
    factor: f32,
) -> PyResult<usize> {
    convert(module.scale_parameters_learning_rate_by_contains(needle, factor))
}

fn set_module_parameters_learning_rate_scale_by_prefix<M: Module>(
    module: &mut M,
    prefix: &str,
    scale: f32,
) -> PyResult<usize> {
    convert(module.set_parameters_learning_rate_scale_by_prefix(prefix, scale))
}

fn set_module_parameters_learning_rate_scale_by_suffix<M: Module>(
    module: &mut M,
    suffix: &str,
    scale: f32,
) -> PyResult<usize> {
    convert(module.set_parameters_learning_rate_scale_by_suffix(suffix, scale))
}

fn set_module_parameters_learning_rate_scale_by_contains<M: Module>(
    module: &mut M,
    needle: &str,
    scale: f32,
) -> PyResult<usize> {
    convert(module.set_parameters_learning_rate_scale_by_contains(needle, scale))
}

fn set_module_parameter_weight_decay<M: Module>(
    module: &mut M,
    name: &str,
    weight_decay: f32,
) -> PyResult<()> {
    convert(module.set_parameter_weight_decay(name, weight_decay))
}

fn set_module_parameters_weight_decay_by_prefix<M: Module>(
    module: &mut M,
    prefix: &str,
    weight_decay: f32,
) -> PyResult<usize> {
    convert(module.set_parameters_weight_decay_by_prefix(prefix, weight_decay))
}

fn set_module_parameters_weight_decay_by_suffix<M: Module>(
    module: &mut M,
    suffix: &str,
    weight_decay: f32,
) -> PyResult<usize> {
    convert(module.set_parameters_weight_decay_by_suffix(suffix, weight_decay))
}

fn set_module_parameters_weight_decay_by_contains<M: Module>(
    module: &mut M,
    needle: &str,
    weight_decay: f32,
) -> PyResult<usize> {
    convert(module.set_parameters_weight_decay_by_contains(needle, weight_decay))
}

fn validation_controls_from_py(
    patience: Option<usize>,
    min_delta: f32,
    lr_decay_patience: Option<usize>,
    lr_decay_factor: f32,
    lr_decay_min_delta: f32,
) -> PyResult<ValidationTrainingControls> {
    let mut controls = ValidationTrainingControls::default();
    if let Some(patience) = patience {
        controls =
            controls.with_early_stopping(convert(EarlyStoppingConfig::new(patience, min_delta))?);
    }
    if let Some(patience) = lr_decay_patience {
        controls = controls.with_lr_plateau(convert(LrPlateauConfig::new(
            patience,
            lr_decay_factor,
            lr_decay_min_delta,
        ))?);
    }
    Ok(controls)
}

fn sparse_retention_guard_from_py(
    max_loss_increase: f32,
    max_accuracy_drop: f32,
    max_perplexity_increase: Option<f32>,
    target_min_loss_delta: f32,
) -> PyResult<SparseRetentionGuardConfig> {
    let mut guard = convert(SparseRetentionGuardConfig::new(
        max_loss_increase,
        max_accuracy_drop,
    ))?;
    if let Some(max_perplexity_increase) = max_perplexity_increase {
        guard = convert(guard.with_max_perplexity_increase(max_perplexity_increase))?;
    }
    if target_min_loss_delta > 0.0 {
        guard = convert(guard.with_target_min_loss_delta(target_min_loss_delta))?;
    } else if target_min_loss_delta < 0.0 || !target_min_loss_delta.is_finite() {
        guard = convert(guard.with_target_min_loss_delta(target_min_loss_delta))?;
    }
    Ok(guard)
}

fn capture_validation_best_with_controls<M: Module, L: Loss>(
    trainer: &mut ModuleTrainer,
    module: &mut M,
    loss: &mut L,
    train_loader: NnDataLoader,
    validation_loader: NnDataLoader,
    schedule: &RoundtableSchedule,
    epochs: usize,
    controls: ValidationTrainingControls,
) -> PyResult<EpochValidationBestState> {
    if controls.early_stopping.is_some() || controls.lr_plateau.is_some() {
        convert(
            trainer.train_epochs_capture_best_on_validation_with_controls(
                module,
                loss,
                train_loader,
                validation_loader,
                schedule,
                epochs,
                controls,
            ),
        )
    } else {
        convert(trainer.train_epochs_capture_best_on_validation(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
        ))
    }
}

fn restore_validation_best_with_controls<M: Module, L: Loss>(
    trainer: &mut ModuleTrainer,
    module: &mut M,
    loss: &mut L,
    train_loader: NnDataLoader,
    validation_loader: NnDataLoader,
    schedule: &RoundtableSchedule,
    epochs: usize,
    controls: ValidationTrainingControls,
) -> PyResult<EpochValidationBestState> {
    if controls.early_stopping.is_some() || controls.lr_plateau.is_some() {
        convert(
            trainer.train_epochs_restore_best_on_validation_with_controls(
                module,
                loss,
                train_loader,
                validation_loader,
                schedule,
                epochs,
                controls,
            ),
        )
    } else {
        convert(trainer.train_epochs_restore_best_on_validation(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
        ))
    }
}

fn capture_sparse_retention_best_with_controls<M: Module>(
    trainer: &mut ModuleTrainer,
    module: &mut M,
    loss: &mut SoftmaxCrossEntropy,
    train_loader: NnDataLoader,
    validation_loader: NnDataLoader,
    retention_loader: NnDataLoader,
    schedule: &RoundtableSchedule,
    epochs: usize,
    guard: SparseRetentionGuardConfig,
    controls: ValidationTrainingControls,
) -> PyResult<EpochSparseRetentionBestState> {
    if controls.early_stopping.is_some() || controls.lr_plateau.is_some() {
        convert(
            trainer.train_epochs_capture_best_sparse_with_retention_guard_and_controls(
                module,
                loss,
                train_loader,
                validation_loader,
                retention_loader,
                schedule,
                epochs,
                guard,
                controls,
            ),
        )
    } else {
        convert(
            trainer.train_epochs_capture_best_sparse_with_retention_guard(
                module,
                loss,
                train_loader,
                validation_loader,
                retention_loader,
                schedule,
                epochs,
                guard,
            ),
        )
    }
}

fn restore_sparse_retention_best_with_controls<M: Module>(
    trainer: &mut ModuleTrainer,
    module: &mut M,
    loss: &mut SoftmaxCrossEntropy,
    train_loader: NnDataLoader,
    validation_loader: NnDataLoader,
    retention_loader: NnDataLoader,
    schedule: &RoundtableSchedule,
    epochs: usize,
    guard: SparseRetentionGuardConfig,
    controls: ValidationTrainingControls,
) -> PyResult<EpochSparseRetentionBestState> {
    if controls.early_stopping.is_some() || controls.lr_plateau.is_some() {
        convert(
            trainer.train_epochs_restore_best_sparse_with_retention_guard_and_controls(
                module,
                loss,
                train_loader,
                validation_loader,
                retention_loader,
                schedule,
                epochs,
                guard,
                controls,
            ),
        )
    } else {
        convert(
            trainer.train_epochs_restore_best_sparse_with_retention_guard(
                module,
                loss,
                train_loader,
                validation_loader,
                retention_loader,
                schedule,
                epochs,
                guard,
            ),
        )
    }
}

fn restore_sparse_finetune_report_with_controls<M: Module>(
    trainer: &mut ModuleTrainer,
    module: &mut M,
    loss: &mut SoftmaxCrossEntropy,
    train_loader: NnDataLoader,
    validation_loader: NnDataLoader,
    retention_loader: NnDataLoader,
    schedule: &RoundtableSchedule,
    epochs: usize,
    guard: SparseRetentionGuardConfig,
    movement_tolerance: f32,
    controls: ValidationTrainingControls,
) -> PyResult<SparseFineTuneReport> {
    if controls.early_stopping.is_some() || controls.lr_plateau.is_some() {
        convert(
            trainer.train_epochs_restore_best_sparse_with_finetune_report_and_controls(
                module,
                loss,
                train_loader,
                validation_loader,
                retention_loader,
                schedule,
                epochs,
                guard,
                movement_tolerance,
                controls,
            ),
        )
    } else {
        convert(
            trainer.train_epochs_restore_best_sparse_with_finetune_report(
                module,
                loss,
                train_loader,
                validation_loader,
                retention_loader,
                schedule,
                epochs,
                guard,
                movement_tolerance,
            ),
        )
    }
}

fn movement_report_to_pydict<'py>(
    py: Python<'py>,
    report: &ParameterMovementReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("status", report.status())?;
    dict.set_item("frozen_stable", report.frozen_stable())?;
    dict.set_item(
        "trainable_movement_observed",
        report.trainable_movement_observed(),
    )?;
    dict.set_item("trainable_changed", report.trainable_changed)?;
    dict.set_item("trainable_unchanged", report.trainable_unchanged)?;
    dict.set_item("frozen_changed", report.frozen_changed)?;
    dict.set_item("frozen_unchanged", report.frozen_unchanged)?;
    dict.set_item("max_trainable_l2_delta", report.max_trainable_l2_delta)?;
    dict.set_item("max_frozen_l2_delta", report.max_frozen_l2_delta)?;
    dict.set_item("max_frozen_abs_delta", report.max_frozen_abs_delta)?;

    let parameters = PyList::empty_bound(py);
    for movement in &report.parameters {
        let entry = PyDict::new_bound(py);
        entry.set_item("name", movement.name.as_str())?;
        entry.set_item("trainable", movement.trainable)?;
        entry.set_item("changed", movement.changed)?;
        entry.set_item("l2_delta", movement.l2_delta)?;
        entry.set_item("max_abs_delta", movement.max_abs_delta)?;
        parameters.append(entry)?;
    }
    dict.set_item("parameters", parameters)?;
    Ok(dict)
}

fn fingerprint_to_pydict<'py>(
    py: Python<'py>,
    fingerprint: &StateFingerprint,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("hash", fingerprint.hash.as_str())?;
    dict.set_item("parameters", fingerprint.parameters)?;
    dict.set_item("values", fingerprint.values)?;
    Ok(dict)
}

fn load_report_to_pydict<'py>(
    py: Python<'py>,
    report: &StateLoadReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("matched", report.matched)?;
    dict.set_item("source", fingerprint_to_pydict(py, &report.source)?)?;
    dict.set_item("loaded", fingerprint_to_pydict(py, &report.loaded)?)?;
    Ok(dict)
}

fn shape_to_pyobject(py: Python<'_>, shape: Option<(usize, usize)>) -> PyObject {
    match shape {
        Some((rows, cols)) => (rows, cols).into_py(py),
        None => py.None(),
    }
}

fn compatibility_report_to_pydict<'py>(
    py: Python<'py>,
    report: &StateCompatibilityReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("compatible", report.compatible)?;
    dict.set_item("expected_parameters", report.expected_parameters)?;
    dict.set_item("source_parameters", report.source_parameters)?;
    dict.set_item("matched", report.matched)?;
    dict.set_item("missing", report.missing)?;
    dict.set_item("shape_mismatched", report.shape_mismatched)?;
    dict.set_item("extra", report.extra)?;
    dict.set_item("source", fingerprint_to_pydict(py, &report.source)?)?;
    dict.set_item(
        "matched_subset",
        fingerprint_to_pydict(py, &report.matched_subset)?,
    )?;

    let entries = PyList::empty_bound(py);
    for entry in &report.entries {
        let item = PyDict::new_bound(py);
        item.set_item("name", entry.name.as_str())?;
        item.set_item("status", entry.status.as_str())?;
        match entry.source_name.as_ref() {
            Some(source_name) => item.set_item("source_name", source_name.as_str())?,
            None => item.set_item("source_name", py.None())?,
        }
        item.set_item("transform", entry.transform.as_str())?;
        item.set_item(
            "expected_shape",
            shape_to_pyobject(py, entry.expected_shape),
        )?;
        item.set_item("source_shape", shape_to_pyobject(py, entry.source_shape))?;
        item.set_item(
            "original_source_shape",
            shape_to_pyobject(py, entry.original_source_shape),
        )?;
        entries.append(item)?;
    }
    dict.set_item("entries", entries)?;
    Ok(dict)
}

fn parameter_training_fingerprint_to_pydict<'py>(
    py: Python<'py>,
    fingerprint: &ParameterTrainingFingerprint,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("hash", fingerprint.hash.as_str())?;
    dict.set_item("parameters", fingerprint.parameters)?;
    dict.set_item("trainable", fingerprint.trainable)?;
    dict.set_item("frozen", fingerprint.frozen)?;
    dict.set_item("hypergrad_tapes", fingerprint.hypergrad_tapes)?;
    dict.set_item("accumulated_l2", fingerprint.accumulated_l2)?;
    Ok(dict)
}

fn trainer_fingerprint_to_pydict<'py>(
    py: Python<'py>,
    fingerprint: &TrainerStateFingerprint,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("hash", fingerprint.hash.as_str())?;
    dict.set_item(
        "gradient_accumulation_steps",
        fingerprint.gradient_accumulation_steps,
    )?;
    dict.set_item("runtime_hooks", fingerprint.runtime_hooks)?;
    Ok(dict)
}

fn resume_fingerprint_to_pydict<'py>(
    py: Python<'py>,
    fingerprint: &TrainingResumeFingerprint,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("hash", fingerprint.hash.as_str())?;
    dict.set_item(
        "trainer",
        trainer_fingerprint_to_pydict(py, &fingerprint.trainer)?,
    )?;
    dict.set_item(
        "parameters",
        fingerprint_to_pydict(py, &fingerprint.parameters)?,
    )?;
    dict.set_item(
        "parameter_training",
        parameter_training_fingerprint_to_pydict(py, &fingerprint.parameter_training)?,
    )?;
    Ok(dict)
}

fn parameter_movement_py<M: Module>(
    py: Python<'_>,
    module: &M,
    before: &Bound<'_, PyDict>,
    tolerance: f32,
) -> PyResult<PyObject> {
    let before_state = pydict_to_state(before)?;
    let report = convert(module.audit_parameter_movement(&before_state, tolerance))?;
    Ok(movement_report_to_pydict(py, &report)?.into_py(py))
}

fn state_fingerprint_py<M: Module>(py: Python<'_>, module: &M) -> PyResult<PyObject> {
    let fingerprint = convert(module.state_fingerprint())?;
    Ok(fingerprint_to_pydict(py, &fingerprint)?.into_py(py))
}

fn training_state_fingerprint_py<M: Module>(py: Python<'_>, module: &M) -> PyResult<PyObject> {
    let fingerprint = convert(module.training_state_fingerprint())?;
    Ok(parameter_training_fingerprint_to_pydict(py, &fingerprint)?.into_py(py))
}

fn checked_load_state_py<M: Module>(
    py: Python<'_>,
    module: &mut M,
    state: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let state = pydict_to_state(state)?;
    let report = convert(module.load_state_dict_checked(&state))?;
    Ok(load_report_to_pydict(py, &report)?.into_py(py))
}

fn checked_load_state_subset_py<M: Module>(
    py: Python<'_>,
    module: &mut M,
    state: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let state = pydict_to_state(state)?;
    let report = convert(module.load_state_dict_subset_checked(&state))?;
    Ok(load_report_to_pydict(py, &report)?.into_py(py))
}

fn checked_load_state_subset_mapped_py<M: Module>(
    py: Python<'_>,
    module: &mut M,
    state: &Bound<'_, PyDict>,
    key_map: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let state = pydict_to_state(state)?;
    let key_rules = pydict_to_key_rules(key_map)?;
    let report = convert(module.load_state_dict_subset_adapted_checked(&state, &key_rules))?;
    Ok(load_report_to_pydict(py, &report)?.into_py(py))
}

fn state_dict_compatibility_py<M: Module>(
    py: Python<'_>,
    module: &M,
    state: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let state = pydict_to_state(state)?;
    let report = convert(module.state_dict_compatibility(&state))?;
    Ok(compatibility_report_to_pydict(py, &report)?.into_py(py))
}

fn state_dict_compatibility_mapped_py<M: Module>(
    py: Python<'_>,
    module: &M,
    state: &Bound<'_, PyDict>,
    key_map: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let state = pydict_to_state(state)?;
    let key_rules = pydict_to_key_rules(key_map)?;
    let report = convert(module.state_dict_compatibility_with_key_rules(&state, &key_rules))?;
    Ok(compatibility_report_to_pydict(py, &report)?.into_py(py))
}

fn py_device_info<'py>(py: Python<'py>, info: HipDeviceInfo) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", info.id)?;
    dict.set_item("name", info.name.as_ref())?;
    dict.set_item("multi_node", info.multi_node)?;
    Ok(dict)
}

fn backend_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Wgpu => "wgpu",
        BackendKind::Mps => "mps",
        BackendKind::Cuda => "cuda",
        BackendKind::Hip => "hip",
        BackendKind::Cpu => "cpu",
    }
}

fn tensor_to_array(tensor: &Tensor) -> PyResult<ArrayD<f32>> {
    let (rows, cols) = tensor.shape();
    Array2::from_shape_vec((rows, cols), tensor.data().to_vec())
        .map(|array| array.into_dyn())
        .map_err(|_| PyValueError::new_err("failed to view tensor as ndarray"))
}

fn array_to_tensor(array: ArrayD<f32>) -> PyResult<PyTensor> {
    let matrix: Array2<f32> = array
        .into_dimensionality::<Ix2>()
        .map_err(|_| PyValueError::new_err("fractional operators require 2D tensors"))?;
    let (rows, cols) = matrix.dim();
    let data = matrix.into_raw_vec();
    Ok(PyTensor::from_tensor(convert(Tensor::from_vec(
        rows, cols, data,
    ))?))
}

fn device_caps_dict<'py>(py: Python<'py>, caps: DeviceCaps) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("backend", backend_name(caps.backend))?;
    dict.set_item("lane_width", caps.lane_width)?;
    dict.set_item("max_workgroup", caps.max_workgroup)?;
    dict.set_item("subgroup", caps.subgroup)?;
    match caps.shared_mem_per_workgroup {
        Some(value) => dict.set_item("shared_mem_per_workgroup", value)?,
        None => dict.set_item("shared_mem_per_workgroup", py.None())?,
    }
    Ok(dict)
}

#[pyclass(module = "spiraltorch", name = "Tensor")]
#[derive(Clone, Debug)]
struct PyTensor {
    inner: Tensor,
}

impl PyTensor {
    fn from_tensor(tensor: Tensor) -> Self {
        Self { inner: tensor }
    }

    fn as_tensor(&self) -> &Tensor {
        &self.inner
    }

    fn as_tensor_mut(&mut self) -> &mut Tensor {
        &mut self.inner
    }

    fn into_tensor(self) -> Tensor {
        self.inner
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (rows, cols, data=None))]
    fn new(rows: usize, cols: usize, data: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let tensor = match data {
            Some(obj) => {
                let values: Vec<f32> = obj.extract()?;
                convert(Tensor::from_vec(rows, cols, values))?
            }
            None => convert(Tensor::zeros(rows, cols))?,
        };
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(Tensor::zeros(rows, cols))?))
    }

    #[pyo3(name = "clone")]
    fn clone_py(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.shape().0
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.shape().1
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn data(&self) -> Vec<f32> {
        self.inner.data().to_vec()
    }

    fn tolist(&self) -> Vec<Vec<f32>> {
        let (rows, cols) = self.inner.shape();
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            out.push(self.inner.data()[start..end].to_vec());
        }
        out
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.matmul(other.as_tensor()),
        )?))
    }

    fn add(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.add(other.as_tensor()),
        )?))
    }

    fn sub(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.sub(other.as_tensor()),
        )?))
    }

    fn scale(&self, value: f32) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(self.inner.scale(value))?))
    }

    fn hadamard(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.hadamard(other.as_tensor()),
        )?))
    }

    fn add_scaled_inplace(&mut self, other: &PyTensor, scale: f32) -> PyResult<()> {
        convert(self.inner.add_scaled(other.as_tensor(), scale))
    }

    fn add_row_inplace(&mut self, bias: Vec<f32>) -> PyResult<()> {
        convert(self.inner.add_row_inplace(&bias))
    }

    fn transpose(&self) -> Self {
        Self::from_tensor(self.inner.transpose())
    }

    fn sum_axis0(&self) -> Vec<f32> {
        self.inner.sum_axis0()
    }

    fn squared_l2_norm(&self) -> f32 {
        self.inner.squared_l2_norm()
    }

    fn project_to_poincare(&self, curvature: f32) -> PyResult<Self> {
        Ok(Self::from_tensor(convert(
            self.inner.project_to_poincare(curvature),
        )?))
    }

    fn hyperbolic_distance(&self, other: &PyTensor, curvature: f32) -> PyResult<f32> {
        convert(self.inner.hyperbolic_distance(other.as_tensor(), curvature))
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rows, cols) = self.inner.shape();
        Ok(format!("Tensor(rows={rows}, cols={cols})"))
    }
}

#[pyclass(module = "spiraltorch.dataset", name = "DataLoader", unsendable)]
#[derive(Clone)]
struct PyDataLoader {
    inner: NnDataLoader,
}

impl PyDataLoader {
    fn from_loader(inner: NnDataLoader) -> Self {
        Self { inner }
    }

    fn clone_inner(&self) -> NnDataLoader {
        self.inner.clone()
    }
}

#[pymethods]
impl PyDataLoader {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    #[getter]
    fn prefetch_depth(&self) -> usize {
        self.inner.prefetch_depth()
    }

    fn shuffle(&self, seed: u64) -> Self {
        Self::from_loader(self.inner.clone().shuffle(seed))
    }

    fn batched(&self, batch_size: usize) -> Self {
        Self::from_loader(self.inner.clone().batched(batch_size))
    }

    fn prefetch(&self, depth: usize) -> Self {
        Self::from_loader(self.inner.clone().prefetch(depth))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyDataLoaderIter>> {
        Py::new(
            slf.py(),
            PyDataLoaderIter {
                inner: Some(slf.clone_inner().into_iter()),
            },
        )
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DataLoader(len={}, batch_size={})",
            self.inner.len(),
            self.inner.batch_size()
        ))
    }
}

#[pyclass(module = "spiraltorch.dataset", name = "DataLoaderIter", unsendable)]
struct PyDataLoaderIter {
    inner: Option<NnDataLoaderBatches>,
}

#[pymethods]
impl PyDataLoaderIter {
    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<(PyTensor, PyTensor)>> {
        if let Some(iter) = self.inner.as_mut() {
            match iter.next() {
                Some(Ok((input, target))) => {
                    return Ok(Some((
                        PyTensor::from_tensor(input),
                        PyTensor::from_tensor(target),
                    )));
                }
                Some(Err(err)) => return Err(tensor_err(err)),
                None => {
                    self.inner = None;
                    return Ok(None);
                }
            }
        }
        Ok(None)
    }
}

#[pyfunction(name = "from_vec")]
fn dataset_from_vec_py(samples: Vec<(PyTensor, PyTensor)>) -> PyResult<PyDataLoader> {
    let owned: Vec<(Tensor, Tensor)> = samples
        .into_iter()
        .map(|(input, target)| (input.into_tensor(), target.into_tensor()))
        .collect();
    Ok(PyDataLoader::from_loader(nn_dataset_from_vec(owned)))
}

fn samples_to_py(samples: Vec<(Tensor, Tensor)>) -> Vec<(PyTensor, PyTensor)> {
    samples
        .into_iter()
        .map(|(input, target)| (PyTensor::from_tensor(input), PyTensor::from_tensor(target)))
        .collect()
}

fn py_samples_to_rust(samples: Vec<(PyTensor, PyTensor)>) -> Vec<(Tensor, Tensor)> {
    samples
        .into_iter()
        .map(|(input, target)| (input.into_tensor(), target.into_tensor()))
        .collect()
}

fn byte_lm_sample_stats_to_pydict<'py>(
    py: Python<'py>,
    stats: ByteLmSampleStats,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("samples", stats.samples)?;
    dict.set_item("total_rows", stats.total_rows)?;
    dict.set_item("active_rows", stats.active_rows)?;
    Ok(dict)
}

#[pyfunction(name = "byte_lm_windows")]
fn byte_lm_windows_py(text: &str, context: usize) -> PyResult<Vec<(PyTensor, PyTensor)>> {
    Ok(samples_to_py(convert(nn_byte_lm_windows(text, context))?))
}

#[pyfunction(name = "byte_lm_corpus_windows")]
fn byte_lm_corpus_windows_py(
    texts: Vec<String>,
    context: usize,
) -> PyResult<Vec<(PyTensor, PyTensor)>> {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    Ok(samples_to_py(convert(nn_byte_lm_corpus_windows(
        &refs, context,
    ))?))
}

#[pyfunction(name = "padded_byte_lm_samples")]
fn padded_byte_lm_samples_py(
    texts: Vec<String>,
    pad_rows: usize,
    ignore_index: i32,
) -> PyResult<Vec<(PyTensor, PyTensor)>> {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    Ok(samples_to_py(convert(nn_padded_byte_lm_samples(
        &refs,
        pad_rows,
        ignore_index,
    ))?))
}

#[pyfunction(name = "byte_lm_sample_stats")]
#[pyo3(signature = (samples, ignore_index=None))]
fn byte_lm_sample_stats_py<'py>(
    py: Python<'py>,
    samples: Vec<(PyTensor, PyTensor)>,
    ignore_index: Option<i32>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned = py_samples_to_rust(samples);
    byte_lm_sample_stats_to_pydict(py, nn_byte_lm_sample_stats(&owned, ignore_index))
}

#[pyfunction(name = "interleave_replay_samples")]
#[pyo3(signature = (target_samples, replay_samples, target_per_replay=1))]
fn interleave_replay_samples_py(
    target_samples: Vec<(PyTensor, PyTensor)>,
    replay_samples: Vec<(PyTensor, PyTensor)>,
    target_per_replay: usize,
) -> PyResult<Vec<(PyTensor, PyTensor)>> {
    let target = py_samples_to_rust(target_samples);
    let replay = py_samples_to_rust(replay_samples);
    Ok(samples_to_py(convert(nn_interleave_replay_samples(
        &target,
        &replay,
        target_per_replay,
    ))?))
}

#[pyclass(module = "spiraltorch", name = "ComplexTensor")]
#[derive(Clone, Debug)]
struct PyComplexTensor {
    inner: ComplexTensor,
}

impl PyComplexTensor {
    fn from_complex(inner: ComplexTensor) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "spiraltorch", name = "BarycenterIntermediate")]
#[derive(Clone, Debug)]
struct PyBarycenterIntermediate {
    inner: BarycenterIntermediate,
}

impl PyBarycenterIntermediate {
    fn from_stage(stage: BarycenterIntermediate) -> Self {
        Self { inner: stage }
    }
}

#[pyclass(module = "spiraltorch", name = "ZSpaceBarycenter")]
#[derive(Clone, Debug)]
struct PyZSpaceBarycenter {
    inner: ZSpaceBarycenter,
}

impl PyZSpaceBarycenter {
    fn from_result(result: ZSpaceBarycenter) -> Self {
        Self { inner: result }
    }
}

#[pymethods]
impl PyZSpaceBarycenter {
    #[getter]
    fn density(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.density.clone()))
    }

    #[getter]
    fn kl_energy(&self) -> f32 {
        self.inner.kl_energy
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.inner.entropy
    }

    #[getter]
    fn coupling_energy(&self) -> f32 {
        self.inner.coupling_energy
    }

    #[getter]
    fn objective(&self) -> f32 {
        self.inner.objective
    }

    #[getter]
    fn effective_weight(&self) -> f32 {
        self.inner.effective_weight
    }

    fn intermediates(&self, py: Python<'_>) -> PyResult<Vec<Py<PyBarycenterIntermediate>>> {
        let mut out = Vec::with_capacity(self.inner.intermediates.len());
        for stage in &self.inner.intermediates {
            out.push(Py::new(
                py,
                PyBarycenterIntermediate::from_stage(stage.clone()),
            )?);
        }
        Ok(out)
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "density",
            PyTensor::from_tensor(self.inner.density.clone()).into_py(py),
        )?;
        dict.set_item("kl_energy", self.inner.kl_energy)?;
        dict.set_item("entropy", self.inner.entropy)?;
        dict.set_item("coupling_energy", self.inner.coupling_energy)?;
        dict.set_item("objective", self.inner.objective)?;
        dict.set_item("effective_weight", self.inner.effective_weight)?;
        let py_intermediates = PyList::empty_bound(py);
        for stage in &self.inner.intermediates {
            py_intermediates
                .append(PyBarycenterIntermediate::from_stage(stage.clone()).into_py(py))?;
        }
        dict.set_item("intermediates", py_intermediates.into_py(py))?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ZSpaceBarycenter(objective={:.6}, entropy={:.6})",
            self.inner.objective, self.inner.entropy
        ))
    }
}

#[pymethods]
impl PyBarycenterIntermediate {
    #[getter]
    fn interpolation(&self) -> f32 {
        self.inner.interpolation
    }

    #[getter]
    fn density(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.density.clone()))
    }

    #[getter]
    fn kl_energy(&self) -> f32 {
        self.inner.kl_energy
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.inner.entropy
    }

    #[getter]
    fn objective(&self) -> f32 {
        self.inner.objective
    }

    fn as_tuple(&self) -> PyResult<(f32, PyTensor, f32, f32, f32)> {
        Ok((
            self.inner.interpolation,
            PyTensor::from_tensor(self.inner.density.clone()),
            self.inner.kl_energy,
            self.inner.entropy,
            self.inner.objective,
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "BarycenterIntermediate(interpolation={:.2}, objective={:.6})",
            self.inner.interpolation, self.inner.objective
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "DifferentialResonance")]
#[derive(Clone)]
struct PyDifferentialResonance {
    inner: DifferentialResonance,
}

impl PyDifferentialResonance {
    fn from_resonance(inner: DifferentialResonance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDifferentialResonance {
    #[getter]
    fn homotopy_flow(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.homotopy_flow.clone()))
    }

    #[getter]
    fn functor_linearisation(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(
            self.inner.functor_linearisation.clone(),
        ))
    }

    #[getter]
    fn recursive_objective(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(
            self.inner.recursive_objective.clone(),
        ))
    }

    #[getter]
    fn infinity_projection(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(
            self.inner.infinity_projection.clone(),
        ))
    }

    #[getter]
    fn infinity_energy(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(self.inner.infinity_energy.clone()))
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "homotopy_flow",
            PyTensor::from_tensor(self.inner.homotopy_flow.clone()).into_py(py),
        )?;
        dict.set_item(
            "functor_linearisation",
            PyTensor::from_tensor(self.inner.functor_linearisation.clone()).into_py(py),
        )?;
        dict.set_item(
            "recursive_objective",
            PyTensor::from_tensor(self.inner.recursive_objective.clone()).into_py(py),
        )?;
        dict.set_item(
            "infinity_projection",
            PyTensor::from_tensor(self.inner.infinity_projection.clone()).into_py(py),
        )?;
        dict.set_item(
            "infinity_energy",
            PyTensor::from_tensor(self.inner.infinity_energy.clone()).into_py(py),
        )?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("DifferentialResonance(...)".to_string())
    }
}

#[pyclass(module = "spiraltorch", name = "SpiralDifferentialTrace")]
struct PySpiralDifferentialTrace {
    trace: Option<DifferentialTrace>,
    sot_plan: Option<PySoT3DPlan>,
}

impl PySpiralDifferentialTrace {
    fn from_trace_with_plan(trace: DifferentialTrace, plan: Option<PySoT3DPlan>) -> Self {
        Self {
            trace: Some(trace),
            sot_plan: plan,
        }
    }

    fn map_trace<F>(&mut self, f: F) -> PyResult<()>
    where
        F: FnOnce(DifferentialTrace) -> PureResult<DifferentialTrace>,
    {
        let trace = self
            .trace
            .take()
            .ok_or_else(|| PyValueError::new_err("trace has already been consumed"))?;
        let trace = convert(f(trace))?;
        self.trace = Some(trace);
        Ok(())
    }

    fn take_trace(&mut self) -> PyResult<DifferentialTrace> {
        self.trace
            .take()
            .ok_or_else(|| PyValueError::new_err("trace has already been consumed"))
    }
}

#[pymethods]
impl PySpiralDifferentialTrace {
    #[getter]
    fn sot_plan(&self) -> Option<PySoT3DPlan> {
        self.sot_plan.clone()
    }

    fn deform(&mut self, generator: &PyTensor, direction: &PyTensor) -> PyResult<()> {
        let generator = generator.as_tensor().clone();
        let direction = direction.as_tensor().clone();
        self.map_trace(move |trace| trace.deform(generator.clone(), direction.clone()))
    }

    fn across(&mut self, topos: &PyOpenTopos) -> PyResult<()> {
        let guard = topos.inner.clone();
        self.map_trace(move |trace| trace.across(guard.clone()))
    }

    #[pyo3(signature = (kernel, source=None))]
    fn via(&mut self, kernel: &PyTensor, source: Option<&PyTensor>) -> PyResult<()> {
        let kernel_tensor = kernel.as_tensor().clone();
        let source_tensor = source.map(|s| s.as_tensor().clone());
        self.map_trace(move |trace| {
            if let Some(ref src) = source_tensor {
                trace.via_with(kernel_tensor.clone(), src.clone())
            } else {
                trace.via(kernel_tensor.clone())
            }
        })
    }

    fn functor_step(&mut self, epsilon: f32) -> PyResult<()> {
        self.map_trace(move |trace| trace.functor_step(epsilon))
    }

    fn with_barycenter(&mut self, barycenter: &PyZSpaceBarycenter) -> PyResult<()> {
        let barycenter_clone = barycenter.inner.clone();
        self.map_trace(move |trace| trace.with_barycenter(&barycenter_clone))
    }

    fn with_barycenter_from(
        &mut self,
        weights: Vec<f32>,
        densities: Vec<PyTensor>,
    ) -> PyResult<()> {
        let tensors: Vec<Tensor> = densities.into_iter().map(PyTensor::into_tensor).collect();
        self.map_trace(move |trace| trace.with_barycenter_from(&weights, tensors.as_slice()))
    }

    #[pyo3(signature = (weights, densities, coupling=None))]
    fn with_barycenter_with(
        &mut self,
        weights: Vec<f32>,
        densities: Vec<PyTensor>,
        coupling: Option<&PyTensor>,
    ) -> PyResult<()> {
        let tensors: Vec<Tensor> = densities.into_iter().map(PyTensor::into_tensor).collect();
        let coupling_tensor = coupling.map(|tensor| tensor.as_tensor().clone());
        self.map_trace(move |trace| {
            let coupling_ref: Option<&Tensor> = coupling_tensor.as_ref().map(|tensor| tensor);
            trace.with_barycenter_with(&weights, tensors.as_slice(), coupling_ref)
        })
    }

    #[pyo3(signature = (levels, curvatures=None))]
    fn with_infinity(
        &mut self,
        levels: Vec<PyTensor>,
        curvatures: Option<Vec<f32>>,
    ) -> PyResult<()> {
        let tensors: Vec<Tensor> = levels.into_iter().map(PyTensor::into_tensor).collect();
        let curvatures = curvatures.unwrap_or_default();
        self.map_trace(move |trace| trace.with_infinity(tensors.clone(), curvatures.clone()))
    }

    fn resonate(&mut self) -> PyResult<PyDifferentialResonance> {
        let trace = self.take_trace()?;
        let resonance = convert(trace.resonate())?;
        Ok(PyDifferentialResonance::from_resonance(resonance))
    }

    fn resonate_with_hypergrad(
        &mut self,
        hypergrad: &mut PyHypergrad,
    ) -> PyResult<PyDifferentialResonance> {
        let trace = self.take_trace()?;
        let resonance = convert(trace.resonate_with_hypergrad(&mut hypergrad.inner))?;
        Ok(PyDifferentialResonance::from_resonance(resonance))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("SpiralDifferentialTrace(...)".to_string())
    }
}

#[pymethods]
impl PyComplexTensor {
    #[new]
    #[pyo3(signature = (rows, cols, data=None))]
    fn new(rows: usize, cols: usize, data: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let tensor = match data {
            Some(obj) => {
                let raw: Vec<(f32, f32)> = obj.extract()?;
                let values = raw
                    .into_iter()
                    .map(|(re, im)| Complex32::new(re, im))
                    .collect();
                convert(ComplexTensor::from_vec(rows, cols, values))?
            }
            None => convert(ComplexTensor::zeros(rows, cols))?,
        };
        Ok(Self { inner: tensor })
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn data(&self) -> Vec<(f32, f32)> {
        self.inner
            .data()
            .iter()
            .map(|value| (value.re, value.im))
            .collect()
    }

    fn to_tensor(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(self.inner.to_tensor())?))
    }

    fn matmul(&self, other: &PyComplexTensor) -> PyResult<Self> {
        Ok(Self::from_complex(convert(
            self.inner.matmul(&other.inner),
        )?))
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rows, cols) = self.inner.shape();
        Ok(format!("ComplexTensor(rows={rows}, cols={cols})"))
    }
}

#[pyclass(module = "spiraltorch", name = "OpenTopos")]
#[derive(Clone, Debug)]
struct PyOpenTopos {
    inner: OpenCartesianTopos,
}

impl PyOpenTopos {
    fn from_topos(topos: OpenCartesianTopos) -> Self {
        Self { inner: topos }
    }
}

#[pymethods]
impl PyOpenTopos {
    #[new]
    fn new(
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PyResult<Self> {
        Ok(Self::from_topos(convert(OpenCartesianTopos::new(
            curvature, tolerance, saturation, max_depth, max_volume,
        ))?))
    }

    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    fn tolerance(&self) -> f32 {
        self.inner.tolerance()
    }

    fn saturation(&self) -> f32 {
        self.inner.saturation()
    }

    fn max_depth(&self) -> usize {
        self.inner.max_depth()
    }

    fn max_volume(&self) -> usize {
        self.inner.max_volume()
    }

    fn guard_tensor(&self, label: &str, tensor: &PyTensor) -> PyResult<()> {
        convert(
            self.inner
                .guard_tensor(intern_label(label), tensor.as_tensor()),
        )
    }

    fn saturate_scalar(&self, value: f32) -> f32 {
        self.inner.saturate(value)
    }

    fn saturate_tensor(&self, mut tensor: PyRefMut<'_, PyTensor>) -> PyResult<()> {
        let inner = tensor.as_tensor_mut();
        self.inner.saturate_slice(inner.data_mut());
        convert(self.inner.guard_tensor("tensor", inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "OpenTopos(curvature={}, tolerance={}, saturation={}, max_depth={}, max_volume={})",
            self.curvature(),
            self.tolerance(),
            self.saturation(),
            self.max_depth(),
            self.max_volume()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "TensorBiome")]
#[derive(Clone, Debug)]
struct PyTensorBiome {
    inner: TensorBiome,
}

impl PyTensorBiome {
    fn from_biome(biome: TensorBiome) -> Self {
        Self { inner: biome }
    }

    fn total_weight_value(&self) -> f32 {
        self.inner.total_weight()
    }
}

#[pymethods]
impl PyTensorBiome {
    #[new]
    fn new(topos: &PyOpenTopos) -> Self {
        Self {
            inner: TensorBiome::new(topos.inner.clone()),
        }
    }

    fn topos(&self) -> PyOpenTopos {
        PyOpenTopos::from_topos(self.inner.topos().clone())
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn total_weight(&self) -> f32 {
        self.total_weight_value()
    }

    fn weights(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    fn absorb(&mut self, label: &str, tensor: &PyTensor) -> PyResult<()> {
        convert(
            self.inner
                .absorb(intern_label(label), tensor.as_tensor().clone()),
        )
    }

    fn absorb_weighted(&mut self, label: &str, tensor: &PyTensor, weight: f32) -> PyResult<()> {
        convert(
            self.inner
                .absorb_weighted(intern_label(label), tensor.as_tensor().clone(), weight),
        )
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn canopy(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(self.inner.canopy())?))
    }

    fn stack(&self) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(self.inner.stack())?))
    }

    fn shoots(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTensor>>> {
        self.inner
            .shoots()
            .iter()
            .cloned()
            .map(|tensor| Py::new(py, PyTensor::from_tensor(tensor)))
            .collect()
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.len())
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rows, cols) = self
            .inner
            .shoots()
            .first()
            .map(|tensor| tensor.shape())
            .unwrap_or((0, 0));
        Ok(format!(
            "TensorBiome(len={}, shape=({}, {}), total_weight={:.3})",
            self.len(),
            rows,
            cols,
            self.total_weight_value()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "LanguageWaveEncoder")]
#[derive(Clone, Debug)]
struct PyLanguageWaveEncoder {
    inner: LanguageWaveEncoder,
}

impl PyLanguageWaveEncoder {
    fn encode_wave_internal(&self, text: &str) -> PyResult<ComplexTensor> {
        convert(self.inner.encode_wave(text))
    }
}

#[pymethods]
impl PyLanguageWaveEncoder {
    #[new]
    fn new(curvature: f32, temperature: f32) -> PyResult<Self> {
        Ok(Self {
            inner: convert(LanguageWaveEncoder::new(curvature, temperature))?,
        })
    }

    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    fn temperature(&self) -> f32 {
        self.inner.temperature()
    }

    fn encode_wave(&self, text: &str) -> PyResult<PyComplexTensor> {
        Ok(PyComplexTensor::from_complex(
            self.encode_wave_internal(text)?,
        ))
    }

    fn encode_z_space(&self, text: &str) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.inner.encode_z_space(text),
        )?))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "LanguageWaveEncoder(curvature={}, temperature={})",
            self.curvature(),
            self.temperature()
        ))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "ZSpaceProjector")]
struct PyZSpaceProjector {
    inner: Option<NnZSpaceProjector>,
}

impl PyZSpaceProjector {
    fn borrow(&self) -> PyResult<&NnZSpaceProjector> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ZSpaceProjector has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnZSpaceProjector> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("ZSpaceProjector has been moved"))
    }

    fn take(&mut self) -> PyResult<NnZSpaceProjector> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("ZSpaceProjector has been moved"))
    }
}

#[pymethods]
impl PyZSpaceProjector {
    #[new]
    #[pyo3(signature = (topos, encoder, strength=1.0))]
    fn new(topos: &PyOpenTopos, encoder: &PyLanguageWaveEncoder, strength: f32) -> PyResult<Self> {
        let inner = convert(NnZSpaceProjector::with_strength(
            topos.inner.clone(),
            encoder.inner.clone(),
            strength,
        ))?;
        Ok(Self { inner: Some(inner) })
    }

    #[getter]
    fn strength(&self) -> PyResult<f32> {
        Ok(self.borrow()?.strength())
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow_mut()?
                .backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn set_trainable(&mut self, trainable: bool) -> PyResult<()> {
        set_module_trainable(self.borrow_mut()?, trainable)
    }

    fn set_parameter_trainable(&mut self, name: &str, trainable: bool) -> PyResult<()> {
        set_module_parameter_trainable(self.borrow_mut()?, name, trainable)
    }

    fn set_parameters_trainable_by_prefix(
        &mut self,
        prefix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_prefix(self.borrow_mut()?, prefix, trainable)
    }

    fn set_parameters_trainable_by_suffix(
        &mut self,
        suffix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_suffix(self.borrow_mut()?, suffix, trainable)
    }

    fn set_parameters_trainable_by_contains(
        &mut self,
        needle: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_contains(self.borrow_mut()?, needle, trainable)
    }

    fn scale_parameter_learning_rate(&mut self, name: &str, factor: f32) -> PyResult<()> {
        scale_module_parameter_learning_rate(self.borrow_mut()?, name, factor)
    }

    fn scale_parameters_learning_rate_by_prefix(
        &mut self,
        prefix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_prefix(self.borrow_mut()?, prefix, factor)
    }

    fn scale_parameters_learning_rate_by_suffix(
        &mut self,
        suffix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_suffix(self.borrow_mut()?, suffix, factor)
    }

    fn scale_parameters_learning_rate_by_contains(
        &mut self,
        needle: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_contains(self.borrow_mut()?, needle, factor)
    }

    fn set_parameter_learning_rate_scale(&mut self, name: &str, scale: f32) -> PyResult<()> {
        set_module_parameter_learning_rate_scale(self.borrow_mut()?, name, scale)
    }

    fn set_parameters_learning_rate_scale_by_prefix(
        &mut self,
        prefix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_prefix(self.borrow_mut()?, prefix, scale)
    }

    fn set_parameters_learning_rate_scale_by_suffix(
        &mut self,
        suffix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_suffix(self.borrow_mut()?, suffix, scale)
    }

    fn set_parameters_learning_rate_scale_by_contains(
        &mut self,
        needle: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_contains(self.borrow_mut()?, needle, scale)
    }

    fn set_parameter_weight_decay(&mut self, name: &str, weight_decay: f32) -> PyResult<()> {
        set_module_parameter_weight_decay(self.borrow_mut()?, name, weight_decay)
    }

    fn set_parameters_weight_decay_by_prefix(
        &mut self,
        prefix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_prefix(self.borrow_mut()?, prefix, weight_decay)
    }

    fn set_parameters_weight_decay_by_suffix(
        &mut self,
        suffix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_suffix(self.borrow_mut()?, suffix, weight_decay)
    }

    fn set_parameters_weight_decay_by_contains(
        &mut self,
        needle: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_contains(self.borrow_mut()?, needle, weight_decay)
    }

    #[pyo3(signature = (before, tolerance=0.0))]
    fn parameter_movement(
        &self,
        py: Python<'_>,
        before: &Bound<'_, PyDict>,
        tolerance: f32,
    ) -> PyResult<PyObject> {
        parameter_movement_py(py, self.borrow()?, before, tolerance)
    }

    fn state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_fingerprint_py(py, self.borrow()?)
    }

    fn load_state_dict_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_mapped_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_mapped_py(py, self.borrow_mut()?, dict, key_map)
    }

    fn state_dict_compatibility(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_py(py, self.borrow()?, dict)
    }

    fn state_dict_compatibility_with_key_map(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_mapped_py(py, self.borrow()?, dict, key_map)
    }

    fn curvature(&self) -> PyResult<f32> {
        Ok(self.borrow()?.curvature())
    }

    fn topos(&self) -> PyResult<PyOpenTopos> {
        Ok(PyOpenTopos::from_topos(self.borrow()?.topos().clone()))
    }

    fn encode_text(&self, text: &str) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.encode_text(text),
        )?))
    }

    fn project_spiral(&self, plan: &PySoT3DPlan) -> PyResult<PyTensor> {
        let base = plan.positions_tensor().map_err(tensor_err)?;
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(&base),
        )?))
    }

    fn reimport_biome(&self, biome: &PyTensorBiome) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.reimport_biome(&biome.inner),
        )?))
    }
}

#[pyclass(module = "spiraltorch", name = "Hypergrad")]
struct PyHypergrad {
    inner: AmegaHypergrad,
}

impl PyHypergrad {
    fn from_hypergrad(inner: AmegaHypergrad) -> Self {
        Self { inner }
    }

    fn ensure_shape(&self, tensor: &PyTensor) -> PyResult<()> {
        let shape = tensor.shape();
        if shape != self.inner.shape() {
            return Err(PyValueError::new_err(format!(
                "tensor shape {:?} does not match hypergrad {:?}",
                shape,
                self.inner.shape()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyHypergrad {
    #[new]
    #[pyo3(signature = (curvature, learning_rate, rows, cols, topos=None))]
    fn new(
        curvature: f32,
        learning_rate: f32,
        rows: usize,
        cols: usize,
        topos: Option<&PyOpenTopos>,
    ) -> PyResult<Self> {
        let inner = if let Some(topos) = topos {
            convert(AmegaHypergrad::with_topos(
                curvature,
                learning_rate,
                rows,
                cols,
                topos.inner.clone(),
            ))?
        } else {
            convert(AmegaHypergrad::new(curvature, learning_rate, rows, cols))?
        };
        Ok(Self { inner })
    }

    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }

    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn gradient(&self) -> Vec<f32> {
        self.inner.gradient().to_vec()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn accumulate_wave(&mut self, tensor: &PyTensor) -> PyResult<()> {
        self.ensure_shape(tensor)?;
        convert(self.inner.accumulate_wave(tensor.as_tensor()))
    }

    fn accumulate_complex_wave(&mut self, wave: &PyComplexTensor) -> PyResult<()> {
        convert(self.inner.accumulate_complex_wave(&wave.inner))
    }

    fn accumulate_pair(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<()> {
        self.ensure_shape(prediction)?;
        convert(
            self.inner
                .accumulate_pair(prediction.as_tensor(), target.as_tensor()),
        )
    }

    fn absorb_text(&mut self, encoder: &PyLanguageWaveEncoder, text: &str) -> PyResult<()> {
        convert(self.inner.absorb_text(&encoder.inner, text))
    }

    #[pyo3(signature = (intermediates))]
    fn accumulate_barycenter_path(
        &mut self,
        py: Python<'_>,
        intermediates: Vec<Py<PyBarycenterIntermediate>>,
    ) -> PyResult<()> {
        if intermediates.is_empty() {
            return Err(PyValueError::new_err(
                "barycenter intermediates must not be empty",
            ));
        }
        let mut stages = Vec::with_capacity(intermediates.len());
        for stage in intermediates {
            let guard = stage.borrow(py);
            stages.push(guard.inner.clone());
        }
        convert(self.inner.accumulate_barycenter_path(&stages))
    }

    fn apply(&mut self, mut weights: PyRefMut<'_, PyTensor>) -> PyResult<()> {
        self.ensure_shape(&weights)?;
        convert(self.inner.apply(weights.as_tensor_mut()))
    }

    fn topos(&self) -> PyResult<PyOpenTopos> {
        Ok(PyOpenTopos::from_topos(self.inner.topos().clone()))
    }

    fn __repr__(&self) -> PyResult<String> {
        let (rows, cols) = self.shape();
        Ok(format!(
            "Hypergrad(curvature={}, learning_rate={}, rows={}, cols={})",
            self.curvature(),
            self.learning_rate(),
            rows,
            cols
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "DistConfig")]
#[derive(Clone)]
struct PyDistConfig {
    inner: DistConfig,
}

#[pymethods]
impl PyDistConfig {
    #[new]
    #[pyo3(signature = (node_id=None, mode=None, push_interval=None, summary_window=None, meta_endpoints=None))]
    fn new(
        node_id: Option<String>,
        mode: Option<&str>,
        push_interval: Option<f32>,
        summary_window: Option<usize>,
        meta_endpoints: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut config = DistConfig::default();
        if let Some(id) = node_id {
            config.node_id = id;
        }
        if let Some(mode_str) = mode {
            config.mode = match mode_str {
                "local" | "local-only" => DistMode::LocalOnly,
                "periodic" | "periodic-meta" => DistMode::PeriodicMeta,
                "global" | "fully-global" => DistMode::FullyGlobal,
                other => {
                    return Err(PyValueError::new_err(format!(
                    "unknown dist mode '{}': expected local-only, periodic-meta, or fully-global",
                    other
                )))
                }
            };
        }
        if let Some(interval) = push_interval {
            if interval <= 0.0 {
                return Err(PyValueError::new_err(
                    "push_interval must be positive seconds",
                ));
            }
            config.push_interval = Duration::from_secs_f32(interval);
        }
        if let Some(window) = summary_window {
            config.summary_window = window.max(1);
        }
        if let Some(endpoints) = meta_endpoints {
            config.meta_endpoints = endpoints;
        }
        Ok(Self { inner: config })
    }

    #[getter]
    fn node_id(&self) -> &str {
        &self.inner.node_id
    }

    #[getter]
    fn mode(&self) -> &'static str {
        match self.inner.mode {
            DistMode::LocalOnly => "local-only",
            DistMode::PeriodicMeta => "periodic-meta",
            DistMode::FullyGlobal => "fully-global",
        }
    }

    #[getter]
    fn push_interval(&self) -> f32 {
        self.inner.push_interval.as_secs_f32()
    }

    #[getter]
    fn summary_window(&self) -> usize {
        self.inner.summary_window
    }

    #[getter]
    fn meta_endpoints(&self) -> Vec<String> {
        self.inner.meta_endpoints.clone()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DistConfig(node_id='{}', mode='{}', push_interval={:.1}, summary_window={}, endpoints={:?})",
            self.node_id(),
            self.mode(),
            self.push_interval(),
            self.summary_window(),
            self.meta_endpoints(),
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "RoundtableSchedule", unsendable)]
struct PyRoundtableSchedule {
    inner: RoundtableSchedule,
}

impl PyRoundtableSchedule {
    fn from_schedule(inner: RoundtableSchedule) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRoundtableSchedule {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn top_k(&self) -> u32 {
        self.inner.above().k
    }

    #[getter]
    fn mid_k(&self) -> u32 {
        self.inner.here().k
    }

    #[getter]
    fn bottom_k(&self) -> u32 {
        self.inner.beneath().k
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "RoundtableSchedule(top={}, mid={}, bottom={})",
            self.top_k(),
            self.mid_k(),
            self.bottom_k()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "EpochStats")]
#[derive(Clone, Copy)]
struct PyEpochStats {
    inner: EpochStats,
}

impl PyEpochStats {
    fn from_stats(inner: EpochStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEpochStats {
    #[getter]
    fn batches(&self) -> usize {
        self.inner.batches
    }

    #[getter]
    fn optimizer_steps(&self) -> usize {
        self.inner.optimizer_steps
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    #[getter]
    fn total_loss(&self) -> f32 {
        self.inner.total_loss
    }

    #[getter]
    fn total_row_weighted_loss(&self) -> f32 {
        self.inner.total_row_weighted_loss
    }

    #[getter]
    fn average_loss(&self) -> f32 {
        self.inner.average_loss
    }

    #[getter]
    fn average_loss_per_row(&self) -> f32 {
        self.inner.average_loss_per_row
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "EpochStats(batches={}, optimizer_steps={}, rows={}, total_loss={:.6}, average_loss={:.6}, average_loss_per_row={:.6})",
            self.batches(),
            self.optimizer_steps(),
            self.rows(),
            self.total_loss(),
            self.average_loss(),
            self.average_loss_per_row()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "EpochHistory")]
#[derive(Clone)]
struct PyEpochHistory {
    inner: EpochHistory,
}

impl PyEpochHistory {
    fn from_history(inner: EpochHistory) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEpochHistory {
    #[getter]
    fn epochs(&self) -> usize {
        self.inner.epochs
    }

    #[getter]
    fn batches(&self) -> usize {
        self.inner.batches
    }

    #[getter]
    fn optimizer_steps(&self) -> usize {
        self.inner.optimizer_steps
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    #[getter]
    fn initial_loss_per_row(&self) -> f32 {
        self.inner.initial_loss_per_row
    }

    #[getter]
    fn final_loss_per_row(&self) -> f32 {
        self.inner.final_loss_per_row
    }

    #[getter]
    fn best_epoch(&self) -> Option<usize> {
        self.inner.best_epoch
    }

    #[getter]
    fn best_loss_per_row(&self) -> f32 {
        self.inner.best_loss_per_row
    }

    #[getter]
    fn final_improvement(&self) -> f32 {
        self.inner.final_improvement
    }

    #[getter]
    fn best_improvement(&self) -> f32 {
        self.inner.best_improvement
    }

    #[getter]
    fn improved(&self) -> bool {
        self.inner.improved
    }

    #[getter]
    fn best_improved(&self) -> bool {
        self.inner.best_improved
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "EpochHistory(epochs={}, batches={}, optimizer_steps={}, best_epoch={:?}, initial_loss_per_row={:.6}, final_loss_per_row={:.6}, best_loss_per_row={:.6}, final_improvement={:.6}, best_improvement={:.6})",
            self.epochs(),
            self.batches(),
            self.optimizer_steps(),
            self.best_epoch(),
            self.initial_loss_per_row(),
            self.final_loss_per_row(),
            self.best_loss_per_row(),
            self.final_improvement(),
            self.best_improvement()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "EpochBestState")]
#[derive(Clone)]
struct PyEpochBestState {
    inner: EpochBestState,
}

impl PyEpochBestState {
    fn from_best_state(inner: EpochBestState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEpochBestState {
    #[getter]
    fn history(&self, py: Python<'_>) -> PyResult<PyObject> {
        epoch_history_to_pylist(py, self.inner.history.clone())
    }

    #[getter]
    fn summary(&self) -> PyEpochHistory {
        PyEpochHistory::from_history(self.inner.summary.clone())
    }

    #[getter]
    fn best_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_to_pydict(py, self.inner.best_state.clone())
    }

    #[getter]
    fn final_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_to_pydict(py, self.inner.final_state.clone())
    }

    #[getter]
    fn best_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(fingerprint_to_pydict(py, &self.inner.best_fingerprint)?.into_py(py))
    }

    #[getter]
    fn final_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(fingerprint_to_pydict(py, &self.inner.final_fingerprint)?.into_py(py))
    }

    #[getter]
    fn best_differs_from_final(&self) -> bool {
        self.inner.best_differs_from_final
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "EpochBestState(summary={:?}, best_hash={}, final_hash={}, best_differs_from_final={})",
            self.inner.summary,
            self.inner.best_fingerprint.hash,
            self.inner.final_fingerprint.hash,
            self.inner.best_differs_from_final
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "EpochValidationBestState")]
#[derive(Clone)]
struct PyEpochValidationBestState {
    inner: EpochValidationBestState,
}

impl PyEpochValidationBestState {
    fn from_validation_best_state(inner: EpochValidationBestState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEpochValidationBestState {
    #[getter]
    fn train_history(&self, py: Python<'_>) -> PyResult<PyObject> {
        epoch_history_to_pylist(py, self.inner.train_history.clone())
    }

    #[getter]
    fn validation_history(&self, py: Python<'_>) -> PyResult<PyObject> {
        epoch_history_to_pylist(py, self.inner.validation_history.clone())
    }

    #[getter]
    fn train_summary(&self) -> PyEpochHistory {
        PyEpochHistory::from_history(self.inner.train_summary.clone())
    }

    #[getter]
    fn validation_summary(&self) -> PyEpochHistory {
        PyEpochHistory::from_history(self.inner.validation_summary.clone())
    }

    #[getter]
    fn best_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_to_pydict(py, self.inner.best_state.clone())
    }

    #[getter]
    fn final_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_to_pydict(py, self.inner.final_state.clone())
    }

    #[getter]
    fn best_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(fingerprint_to_pydict(py, &self.inner.best_fingerprint)?.into_py(py))
    }

    #[getter]
    fn final_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(fingerprint_to_pydict(py, &self.inner.final_fingerprint)?.into_py(py))
    }

    #[getter]
    fn best_differs_from_final(&self) -> bool {
        self.inner.best_differs_from_final
    }

    #[getter]
    fn epochs_requested(&self) -> usize {
        self.inner.epochs_requested
    }

    #[getter]
    fn early_stopped(&self) -> bool {
        self.inner.early_stopped
    }

    #[getter]
    fn stop_epoch(&self) -> Option<usize> {
        self.inner.stop_epoch
    }

    #[getter]
    fn early_stopping(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let Some(config) = self.inner.early_stopping else {
            return Ok(None);
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("patience", config.patience)?;
        dict.set_item("min_delta", config.min_delta)?;
        Ok(Some(dict.into_py(py)))
    }

    #[getter]
    fn lr_plateau(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let Some(config) = self.inner.lr_plateau else {
            return Ok(None);
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("patience", config.patience)?;
        dict.set_item("factor", config.factor)?;
        dict.set_item("min_delta", config.min_delta)?;
        Ok(Some(dict.into_py(py)))
    }

    #[getter]
    fn lr_decay_steps(&self) -> usize {
        self.inner.lr_decay_steps
    }

    #[getter]
    fn final_hyper_learning_rate(&self) -> f32 {
        self.inner.final_hyper_learning_rate
    }

    #[getter]
    fn final_fallback_learning_rate(&self) -> f32 {
        self.inner.final_fallback_learning_rate
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "EpochValidationBestState(validation_summary={:?}, best_hash={}, final_hash={}, best_differs_from_final={}, early_stopped={}, lr_decay_steps={})",
            self.inner.validation_summary,
            self.inner.best_fingerprint.hash,
            self.inner.final_fingerprint.hash,
            self.inner.best_differs_from_final,
            self.inner.early_stopped,
            self.inner.lr_decay_steps
        ))
    }
}

fn epoch_history_to_pylist(py: Python<'_>, history: Vec<EpochStats>) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for stats in history {
        list.append(Py::new(py, PyEpochStats::from_stats(stats))?)?;
    }
    Ok(list.into_py(py))
}

#[pyclass(module = "spiraltorch", name = "EpochSparseRetentionBestState")]
#[derive(Clone)]
struct PyEpochSparseRetentionBestState {
    inner: EpochSparseRetentionBestState,
}

impl PyEpochSparseRetentionBestState {
    fn from_sparse_retention_best_state(inner: EpochSparseRetentionBestState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEpochSparseRetentionBestState {
    #[getter]
    fn train_history(&self, py: Python<'_>) -> PyResult<PyObject> {
        epoch_history_to_pylist(py, self.inner.train_history.clone())
    }

    #[getter]
    fn validation_history(&self, py: Python<'_>) -> PyResult<PyObject> {
        sparse_metrics_history_to_pylist(py, self.inner.validation_history.clone())
    }

    #[getter]
    fn retention_history(&self, py: Python<'_>) -> PyResult<PyObject> {
        sparse_metrics_history_to_pylist(py, self.inner.retention_history.clone())
    }

    #[getter]
    fn validation_baseline(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_metrics_to_pydict(py, self.inner.validation_baseline)?.into_py(py))
    }

    #[getter]
    fn retention_baseline(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_metrics_to_pydict(py, self.inner.retention_baseline)?.into_py(py))
    }

    #[getter]
    fn train_summary(&self) -> PyEpochHistory {
        PyEpochHistory::from_history(self.inner.train_summary.clone())
    }

    #[getter]
    fn retention_guard(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item(
            "max_loss_increase",
            self.inner.retention_guard.max_loss_increase,
        )?;
        dict.set_item(
            "max_accuracy_drop",
            self.inner.retention_guard.max_accuracy_drop,
        )?;
        dict.set_item(
            "max_perplexity_increase",
            self.inner.retention_guard.max_perplexity_increase,
        )?;
        dict.set_item(
            "target_min_loss_delta",
            self.inner.retention_guard.target_min_loss_delta,
        )?;
        Ok(dict.into_py(py))
    }

    #[getter]
    fn max_allowed_retention_loss(&self) -> f32 {
        self.inner.max_allowed_retention_loss
    }

    #[getter]
    fn min_allowed_retention_accuracy(&self) -> f32 {
        self.inner.min_allowed_retention_accuracy
    }

    #[getter]
    fn max_allowed_retention_perplexity(&self) -> Option<f32> {
        self.inner.max_allowed_retention_perplexity
    }

    #[getter]
    fn guarded_best_epoch(&self) -> Option<usize> {
        self.inner.guarded_best_epoch
    }

    #[getter]
    fn guard_accepted_epochs(&self) -> usize {
        self.inner.guard_accepted_epochs
    }

    #[getter]
    fn guard_retention_rejected_epochs(&self) -> usize {
        self.inner.guard_retention_rejected_epochs
    }

    #[getter]
    fn guard_target_stale_epochs(&self) -> usize {
        self.inner.guard_target_stale_epochs
    }

    #[getter]
    fn best_validation_metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_metrics_to_pydict(py, self.inner.best_validation_metrics)?.into_py(py))
    }

    #[getter]
    fn best_retention_metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_metrics_to_pydict(py, self.inner.best_retention_metrics)?.into_py(py))
    }

    #[getter]
    fn best_retention_loss_increase(&self) -> f32 {
        self.inner.best_retention_loss_increase
    }

    #[getter]
    fn best_retention_accuracy_drop(&self) -> f32 {
        self.inner.best_retention_accuracy_drop
    }

    #[getter]
    fn best_retention_perplexity_increase(&self) -> f32 {
        self.inner.best_retention_perplexity_increase
    }

    #[getter]
    fn best_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_to_pydict(py, self.inner.best_state.clone())
    }

    #[getter]
    fn final_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_to_pydict(py, self.inner.final_state.clone())
    }

    #[getter]
    fn best_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(fingerprint_to_pydict(py, &self.inner.best_fingerprint)?.into_py(py))
    }

    #[getter]
    fn final_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(fingerprint_to_pydict(py, &self.inner.final_fingerprint)?.into_py(py))
    }

    #[getter]
    fn best_differs_from_final(&self) -> bool {
        self.inner.best_differs_from_final
    }

    #[getter]
    fn epochs_requested(&self) -> usize {
        self.inner.epochs_requested
    }

    #[getter]
    fn early_stopped(&self) -> bool {
        self.inner.early_stopped
    }

    #[getter]
    fn stop_epoch(&self) -> Option<usize> {
        self.inner.stop_epoch
    }

    #[getter]
    fn lr_decay_steps(&self) -> usize {
        self.inner.lr_decay_steps
    }

    #[getter]
    fn final_hyper_learning_rate(&self) -> f32 {
        self.inner.final_hyper_learning_rate
    }

    #[getter]
    fn final_fallback_learning_rate(&self) -> f32 {
        self.inner.final_fallback_learning_rate
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "EpochSparseRetentionBestState(guarded_best_epoch={:?}, guard_accepted_epochs={}, guard_retention_rejected_epochs={}, guard_target_stale_epochs={}, best_validation_loss={:.6}, best_retention_loss={:.6}, best_retention_accuracy_drop={:.6}, best_hash={}, final_hash={})",
            self.inner.guarded_best_epoch,
            self.inner.guard_accepted_epochs,
            self.inner.guard_retention_rejected_epochs,
            self.inner.guard_target_stale_epochs,
            self.inner.best_validation_metrics.mean_loss,
            self.inner.best_retention_metrics.mean_loss,
            self.inner.best_retention_accuracy_drop,
            self.inner.best_fingerprint.hash,
            self.inner.final_fingerprint.hash
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "SparseFineTuneReport")]
#[derive(Clone)]
struct PySparseFineTuneReport {
    inner: SparseFineTuneReport,
}

impl PySparseFineTuneReport {
    fn from_sparse_finetune_report(inner: SparseFineTuneReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySparseFineTuneReport {
    #[getter]
    fn captured(&self) -> PyEpochSparseRetentionBestState {
        PyEpochSparseRetentionBestState::from_sparse_retention_best_state(
            self.inner.captured.clone(),
        )
    }

    #[getter]
    fn target_after(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_metrics_to_pydict(py, self.inner.target_after)?.into_py(py))
    }

    #[getter]
    fn retention_after(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_metrics_to_pydict(py, self.inner.retention_after)?.into_py(py))
    }

    #[getter]
    fn target_delta(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_delta_to_pydict(py, self.inner.target_delta)?.into_py(py))
    }

    #[getter]
    fn retention_delta(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_delta_to_pydict(py, self.inner.retention_delta)?.into_py(py))
    }

    #[getter]
    fn movement(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(movement_report_to_pydict(py, &self.inner.movement)?.into_py(py))
    }

    #[getter]
    fn accepted(&self) -> bool {
        self.inner.accepted()
    }

    #[getter]
    fn target_loss_improved(&self) -> bool {
        self.inner.target_loss_improved()
    }

    #[getter]
    fn movement_ok(&self) -> bool {
        self.inner.movement_ok()
    }

    #[getter]
    fn status(&self) -> &'static str {
        self.inner.status()
    }

    fn summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(sparse_finetune_summary_to_pydict(py, self.inner.summary())?.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SparseFineTuneReport(status={}, accepted={}, target_loss_delta={:.6}, retention_loss_delta={:.6}, movement_status={})",
            self.inner.status(),
            self.inner.accepted(),
            self.inner.target_delta.loss_delta,
            self.inner.retention_delta.loss_delta,
            self.inner.movement.status()
        ))
    }
}

fn sparse_metrics_history_to_pylist(
    py: Python<'_>,
    history: Vec<SparseClassificationMetrics>,
) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for metrics in history {
        list.append(sparse_metrics_to_pydict(py, metrics)?)?;
    }
    Ok(list.into_py(py))
}

fn sparse_metrics_to_pydict<'py>(
    py: Python<'py>,
    metrics: SparseClassificationMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("active_rows", metrics.active_rows)?;
    dict.set_item("correct", metrics.correct)?;
    dict.set_item("accuracy", metrics.accuracy)?;
    dict.set_item("mean_loss", metrics.mean_loss)?;
    dict.set_item("perplexity", metrics.perplexity)?;
    Ok(dict)
}

fn sparse_delta_to_pydict<'py>(
    py: Python<'py>,
    delta: SparseClassificationDelta,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("before", sparse_metrics_to_pydict(py, delta.before)?)?;
    dict.set_item("after", sparse_metrics_to_pydict(py, delta.after)?)?;
    dict.set_item("loss_delta", delta.loss_delta)?;
    dict.set_item("accuracy_delta", delta.accuracy_delta)?;
    dict.set_item("perplexity_delta", delta.perplexity_delta)?;
    Ok(dict)
}

fn sparse_finetune_summary_to_pydict<'py>(
    py: Python<'py>,
    summary: SparseFineTuneReportSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("status", summary.status)?;
    dict.set_item("accepted", summary.accepted)?;
    dict.set_item("target_loss_improved", summary.target_loss_improved)?;
    dict.set_item("movement_ok", summary.movement_ok)?;
    dict.set_item("guarded_best_epoch", summary.guarded_best_epoch)?;
    dict.set_item("guard_epochs_run", summary.guard_epochs_run)?;
    dict.set_item("guard_accepted_epochs", summary.guard_accepted_epochs)?;
    dict.set_item(
        "guard_retention_rejected_epochs",
        summary.guard_retention_rejected_epochs,
    )?;
    dict.set_item("guard_target_stale_epochs", summary.guard_target_stale_epochs)?;
    dict.set_item("guard_acceptance_rate", summary.guard_acceptance_rate)?;
    dict.set_item(
        "guard_retention_rejected_rate",
        summary.guard_retention_rejected_rate,
    )?;
    dict.set_item("guard_target_stale_rate", summary.guard_target_stale_rate)?;
    dict.set_item("epochs_run", summary.epochs_run)?;
    dict.set_item("train_rows", summary.train_rows)?;
    dict.set_item("train_batches", summary.train_batches)?;
    dict.set_item("optimizer_steps", summary.optimizer_steps)?;
    dict.set_item("target_loss_delta", summary.target_loss_delta)?;
    dict.set_item("target_accuracy_delta", summary.target_accuracy_delta)?;
    dict.set_item("target_perplexity_delta", summary.target_perplexity_delta)?;
    dict.set_item("retention_loss_delta", summary.retention_loss_delta)?;
    dict.set_item("retention_accuracy_delta", summary.retention_accuracy_delta)?;
    dict.set_item(
        "retention_perplexity_delta",
        summary.retention_perplexity_delta,
    )?;
    dict.set_item("target_retention_gap", summary.target_retention_gap)?;
    dict.set_item("target_retention_ratio", summary.target_retention_ratio)?;
    dict.set_item(
        "best_retention_loss_increase",
        summary.best_retention_loss_increase,
    )?;
    dict.set_item(
        "best_retention_accuracy_drop",
        summary.best_retention_accuracy_drop,
    )?;
    dict.set_item(
        "best_retention_perplexity_increase",
        summary.best_retention_perplexity_increase,
    )?;
    dict.set_item(
        "retention_max_loss_increase",
        summary.retention_max_loss_increase,
    )?;
    dict.set_item(
        "retention_max_accuracy_drop",
        summary.retention_max_accuracy_drop,
    )?;
    dict.set_item(
        "retention_max_perplexity_increase",
        summary.retention_max_perplexity_increase,
    )?;
    dict.set_item("target_min_loss_delta", summary.target_min_loss_delta)?;
    dict.set_item("target_loss_margin", summary.target_loss_margin)?;
    dict.set_item("retention_loss_margin", summary.retention_loss_margin)?;
    dict.set_item(
        "retention_accuracy_margin",
        summary.retention_accuracy_margin,
    )?;
    dict.set_item(
        "retention_perplexity_margin",
        summary.retention_perplexity_margin,
    )?;
    dict.set_item(
        "max_allowed_retention_loss",
        summary.max_allowed_retention_loss,
    )?;
    dict.set_item(
        "min_allowed_retention_accuracy",
        summary.min_allowed_retention_accuracy,
    )?;
    dict.set_item(
        "max_allowed_retention_perplexity",
        summary.max_allowed_retention_perplexity,
    )?;
    dict.set_item("movement_status", summary.movement_status)?;
    dict.set_item("frozen_stable", summary.frozen_stable)?;
    dict.set_item(
        "trainable_movement_observed",
        summary.trainable_movement_observed,
    )?;
    dict.set_item("movement_tolerance", summary.movement_tolerance)?;
    dict.set_item("trainable_changed", summary.trainable_changed)?;
    dict.set_item("frozen_changed", summary.frozen_changed)?;
    dict.set_item("max_trainable_l2_delta", summary.max_trainable_l2_delta)?;
    dict.set_item("max_frozen_l2_delta", summary.max_frozen_l2_delta)?;
    dict.set_item("resume_hash", summary.resume_hash)?;
    dict.set_item("resume_trainer_hash", summary.resume_trainer_hash)?;
    dict.set_item("resume_parameter_hash", summary.resume_parameter_hash)?;
    dict.set_item(
        "resume_parameter_training_hash",
        summary.resume_parameter_training_hash,
    )?;
    dict.set_item("resume_trainable", summary.resume_trainable)?;
    dict.set_item("resume_frozen", summary.resume_frozen)?;
    dict.set_item("resume_hypergrad_tapes", summary.resume_hypergrad_tapes)?;
    dict.set_item(
        "resume_gradient_accumulation_steps",
        summary.resume_gradient_accumulation_steps,
    )?;
    dict.set_item("resume_runtime_hooks", summary.resume_runtime_hooks)?;
    dict.set_item("best_hash", summary.best_hash)?;
    dict.set_item("final_hash", summary.final_hash)?;
    dict.set_item("best_differs_from_final", summary.best_differs_from_final)?;
    Ok(dict)
}

fn required_pydict<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: FromPyObject<'py>,
{
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("dict missing '{key}'")))?
        .extract()
}

fn optional_pydict<'py, T>(dict: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
where
    T: FromPyObject<'py>,
{
    match dict.get_item(key)? {
        Some(value) => value.extract(),
        None => Ok(default),
    }
}

fn sparse_finetune_status_label(value: &str) -> PyResult<&'static str> {
    match value {
        "ok" => Ok("ok"),
        "guard_rejected" => Ok("guard_rejected"),
        "frozen_changed" => Ok("frozen_changed"),
        "no_trainable_movement" => Ok("no_trainable_movement"),
        "target_not_improved" => Ok("target_not_improved"),
        _ => Err(PyValueError::new_err(format!(
            "unknown sparse fine-tune status '{value}'"
        ))),
    }
}

fn sparse_finetune_movement_status_label(value: &str) -> PyResult<&'static str> {
    match value {
        "ok" => Ok("ok"),
        "frozen_changed" => Ok("frozen_changed"),
        "no_trainable_movement" => Ok("no_trainable_movement"),
        _ => Err(PyValueError::new_err(format!(
            "unknown sparse fine-tune movement_status '{value}'"
        ))),
    }
}

fn sparse_finetune_summary_from_pydict(
    dict: &Bound<'_, PyDict>,
) -> PyResult<SparseFineTuneReportSummary> {
    let status: String = required_pydict(dict, "status")?;
    let movement_status: String = required_pydict(dict, "movement_status")?;
    let target_loss_delta: f32 = required_pydict(dict, "target_loss_delta")?;
    let retention_loss_delta: f32 = required_pydict(dict, "retention_loss_delta")?;
    let target_retention_gap: f32 = optional_pydict(
        dict,
        "target_retention_gap",
        target_loss_delta - retention_loss_delta,
    )?;
    let target_retention_ratio: Option<f32> = optional_pydict(
        dict,
        "target_retention_ratio",
        if retention_loss_delta > 0.0 {
            Some(target_loss_delta / retention_loss_delta)
        } else {
            None
        },
    )?;
    let best_retention_loss_increase: f32 = required_pydict(dict, "best_retention_loss_increase")?;
    let best_retention_accuracy_drop: f32 = required_pydict(dict, "best_retention_accuracy_drop")?;
    let best_retention_perplexity_increase: f32 =
        required_pydict(dict, "best_retention_perplexity_increase")?;
    let retention_max_loss_increase: f32 =
        optional_pydict(dict, "retention_max_loss_increase", 0.0f32)?;
    let retention_max_accuracy_drop: f32 =
        optional_pydict(dict, "retention_max_accuracy_drop", 0.0f32)?;
    let retention_max_perplexity_increase: Option<f32> =
        optional_pydict(dict, "retention_max_perplexity_increase", None::<f32>)?;
    let target_min_loss_delta: f32 = optional_pydict(dict, "target_min_loss_delta", 0.0f32)?;
    let target_loss_margin = optional_pydict(
        dict,
        "target_loss_margin",
        target_loss_delta - target_min_loss_delta,
    )?;
    let retention_loss_margin = optional_pydict(
        dict,
        "retention_loss_margin",
        retention_max_loss_increase - best_retention_loss_increase,
    )?;
    let retention_accuracy_margin = optional_pydict(
        dict,
        "retention_accuracy_margin",
        retention_max_accuracy_drop - best_retention_accuracy_drop,
    )?;
    let retention_perplexity_margin = optional_pydict(
        dict,
        "retention_perplexity_margin",
        retention_max_perplexity_increase
            .map(|ceiling| ceiling - best_retention_perplexity_increase),
    )?;
    let accepted: bool = required_pydict(dict, "accepted")?;
    let guarded_best_epoch: Option<usize> = required_pydict(dict, "guarded_best_epoch")?;
    let epochs_run: usize = required_pydict(dict, "epochs_run")?;
    let guard_epochs_run = optional_pydict(dict, "guard_epochs_run", epochs_run)?;
    let default_guard_accepted_epochs = if accepted { 1usize } else { 0usize };
    let guard_accepted_epochs = optional_pydict(
        dict,
        "guard_accepted_epochs",
        default_guard_accepted_epochs,
    )?;
    let guard_retention_rejected_epochs =
        optional_pydict(dict, "guard_retention_rejected_epochs", 0usize)?;
    let default_guard_target_stale_epochs = epochs_run.saturating_sub(
        guard_accepted_epochs.saturating_add(guard_retention_rejected_epochs),
    );
    let guard_target_stale_epochs = optional_pydict(
        dict,
        "guard_target_stale_epochs",
        default_guard_target_stale_epochs,
    )?;
    let guard_rate = |key: &str, count: usize| {
        optional_pydict(
            dict,
            key,
            if guard_epochs_run == 0 {
                0.0f32
            } else {
                count as f32 / guard_epochs_run as f32
            },
        )
    };
    let guard_acceptance_rate = guard_rate("guard_acceptance_rate", guard_accepted_epochs)?;
    let guard_retention_rejected_rate =
        guard_rate("guard_retention_rejected_rate", guard_retention_rejected_epochs)?;
    let guard_target_stale_rate =
        guard_rate("guard_target_stale_rate", guard_target_stale_epochs)?;
    Ok(SparseFineTuneReportSummary {
        status: sparse_finetune_status_label(&status)?,
        accepted,
        target_loss_improved: required_pydict(dict, "target_loss_improved")?,
        movement_ok: required_pydict(dict, "movement_ok")?,
        guarded_best_epoch,
        guard_epochs_run,
        guard_accepted_epochs,
        guard_retention_rejected_epochs,
        guard_target_stale_epochs,
        guard_acceptance_rate,
        guard_retention_rejected_rate,
        guard_target_stale_rate,
        epochs_run,
        train_rows: required_pydict(dict, "train_rows")?,
        train_batches: required_pydict(dict, "train_batches")?,
        optimizer_steps: required_pydict(dict, "optimizer_steps")?,
        target_loss_delta,
        target_accuracy_delta: required_pydict(dict, "target_accuracy_delta")?,
        target_perplexity_delta: required_pydict(dict, "target_perplexity_delta")?,
        retention_loss_delta,
        retention_accuracy_delta: required_pydict(dict, "retention_accuracy_delta")?,
        retention_perplexity_delta: required_pydict(dict, "retention_perplexity_delta")?,
        target_retention_gap,
        target_retention_ratio,
        best_retention_loss_increase,
        best_retention_accuracy_drop,
        best_retention_perplexity_increase,
        retention_max_loss_increase,
        retention_max_accuracy_drop,
        retention_max_perplexity_increase,
        target_min_loss_delta,
        target_loss_margin,
        retention_loss_margin,
        retention_accuracy_margin,
        retention_perplexity_margin,
        max_allowed_retention_loss: required_pydict(dict, "max_allowed_retention_loss")?,
        min_allowed_retention_accuracy: required_pydict(dict, "min_allowed_retention_accuracy")?,
        max_allowed_retention_perplexity: required_pydict(
            dict,
            "max_allowed_retention_perplexity",
        )?,
        movement_status: sparse_finetune_movement_status_label(&movement_status)?,
        frozen_stable: required_pydict(dict, "frozen_stable")?,
        trainable_movement_observed: required_pydict(dict, "trainable_movement_observed")?,
        movement_tolerance: optional_pydict(dict, "movement_tolerance", 1e-8f32)?,
        trainable_changed: required_pydict(dict, "trainable_changed")?,
        frozen_changed: required_pydict(dict, "frozen_changed")?,
        max_trainable_l2_delta: required_pydict(dict, "max_trainable_l2_delta")?,
        max_frozen_l2_delta: required_pydict(dict, "max_frozen_l2_delta")?,
        resume_hash: optional_pydict(dict, "resume_hash", String::new())?,
        resume_trainer_hash: optional_pydict(dict, "resume_trainer_hash", String::new())?,
        resume_parameter_hash: optional_pydict(dict, "resume_parameter_hash", String::new())?,
        resume_parameter_training_hash: optional_pydict(
            dict,
            "resume_parameter_training_hash",
            String::new(),
        )?,
        resume_trainable: optional_pydict(dict, "resume_trainable", 0usize)?,
        resume_frozen: optional_pydict(dict, "resume_frozen", 0usize)?,
        resume_hypergrad_tapes: optional_pydict(dict, "resume_hypergrad_tapes", 0usize)?,
        resume_gradient_accumulation_steps: optional_pydict(
            dict,
            "resume_gradient_accumulation_steps",
            0usize,
        )?,
        resume_runtime_hooks: optional_pydict(dict, "resume_runtime_hooks", 0usize)?,
        best_hash: required_pydict(dict, "best_hash")?,
        final_hash: required_pydict(dict, "final_hash")?,
        best_differs_from_final: required_pydict(dict, "best_differs_from_final")?,
    })
}

fn sparse_finetune_summary_comparison_to_pydict<'py>(
    py: Python<'py>,
    current: &SparseFineTuneReportSummary,
    baseline: &SparseFineTuneReportSummary,
    comparison: st_nn::SparseFineTuneSummaryComparison,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "target_loss_delta_change",
        comparison.target_loss_delta_change,
    )?;
    dict.set_item(
        "retention_loss_delta_change",
        comparison.retention_loss_delta_change,
    )?;
    dict.set_item(
        "target_retention_gap_change",
        comparison.target_retention_gap_change,
    )?;
    dict.set_item(
        "target_retention_ratio_change",
        comparison.target_retention_ratio_change,
    )?;
    dict.set_item("target_loss_regression", comparison.target_loss_regression)?;
    dict.set_item(
        "retention_loss_regression",
        comparison.retention_loss_regression,
    )?;
    dict.set_item(
        "target_retention_gap_regression",
        comparison.target_retention_gap_regression,
    )?;
    dict.set_item(
        "target_retention_ratio_regression",
        comparison.target_retention_ratio_regression,
    )?;
    dict.set_item(
        "target_loss_margin_shortfall",
        comparison.target_loss_margin_shortfall,
    )?;
    dict.set_item(
        "target_retention_ratio_shortfall",
        comparison.target_retention_ratio_shortfall,
    )?;
    dict.set_item(
        "retention_loss_margin_shortfall",
        comparison.retention_loss_margin_shortfall,
    )?;
    dict.set_item(
        "retention_accuracy_margin_shortfall",
        comparison.retention_accuracy_margin_shortfall,
    )?;
    dict.set_item(
        "retention_perplexity_margin_shortfall",
        comparison.retention_perplexity_margin_shortfall,
    )?;
    dict.set_item("status_changed", comparison.status_changed)?;
    dict.set_item("accepted_changed", comparison.accepted_changed)?;
    dict.set_item("guard_changed", comparison.guard_changed)?;
    dict.set_item(
        "movement_tolerance_changed",
        comparison.movement_tolerance_changed,
    )?;
    dict.set_item("resume_changed", comparison.resume_changed)?;
    dict.set_item("passed", comparison.passed)?;
    dict.set_item("current_status", current.status)?;
    dict.set_item("baseline_status", baseline.status)?;
    dict.set_item("current_accepted", current.accepted)?;
    dict.set_item("baseline_accepted", baseline.accepted)?;
    dict.set_item(
        "current_retention_max_loss_increase",
        current.retention_max_loss_increase,
    )?;
    dict.set_item(
        "baseline_retention_max_loss_increase",
        baseline.retention_max_loss_increase,
    )?;
    dict.set_item(
        "current_retention_max_accuracy_drop",
        current.retention_max_accuracy_drop,
    )?;
    dict.set_item(
        "baseline_retention_max_accuracy_drop",
        baseline.retention_max_accuracy_drop,
    )?;
    dict.set_item(
        "current_retention_max_perplexity_increase",
        current.retention_max_perplexity_increase,
    )?;
    dict.set_item(
        "baseline_retention_max_perplexity_increase",
        baseline.retention_max_perplexity_increase,
    )?;
    dict.set_item(
        "current_target_min_loss_delta",
        current.target_min_loss_delta,
    )?;
    dict.set_item(
        "baseline_target_min_loss_delta",
        baseline.target_min_loss_delta,
    )?;
    dict.set_item("current_target_loss_margin", current.target_loss_margin)?;
    dict.set_item("baseline_target_loss_margin", baseline.target_loss_margin)?;
    dict.set_item("current_target_retention_gap", current.target_retention_gap)?;
    dict.set_item(
        "baseline_target_retention_gap",
        baseline.target_retention_gap,
    )?;
    dict.set_item(
        "current_target_retention_ratio",
        current.target_retention_ratio,
    )?;
    dict.set_item(
        "baseline_target_retention_ratio",
        baseline.target_retention_ratio,
    )?;
    dict.set_item(
        "current_retention_loss_margin",
        current.retention_loss_margin,
    )?;
    dict.set_item(
        "baseline_retention_loss_margin",
        baseline.retention_loss_margin,
    )?;
    dict.set_item(
        "current_retention_accuracy_margin",
        current.retention_accuracy_margin,
    )?;
    dict.set_item(
        "baseline_retention_accuracy_margin",
        baseline.retention_accuracy_margin,
    )?;
    dict.set_item(
        "current_retention_perplexity_margin",
        current.retention_perplexity_margin,
    )?;
    dict.set_item(
        "baseline_retention_perplexity_margin",
        baseline.retention_perplexity_margin,
    )?;
    dict.set_item("current_movement_tolerance", current.movement_tolerance)?;
    dict.set_item("baseline_movement_tolerance", baseline.movement_tolerance)?;
    dict.set_item("current_resume_hash", current.resume_hash.as_str())?;
    dict.set_item("baseline_resume_hash", baseline.resume_hash.as_str())?;
    dict.set_item(
        "current_resume_trainer_hash",
        current.resume_trainer_hash.as_str(),
    )?;
    dict.set_item(
        "baseline_resume_trainer_hash",
        baseline.resume_trainer_hash.as_str(),
    )?;
    dict.set_item(
        "current_resume_parameter_training_hash",
        current.resume_parameter_training_hash.as_str(),
    )?;
    dict.set_item(
        "baseline_resume_parameter_training_hash",
        baseline.resume_parameter_training_hash.as_str(),
    )?;
    Ok(dict)
}

#[pyfunction(name = "compare_sparse_finetune_summaries")]
#[pyo3(signature = (current, baseline, max_target_loss_regression=None, max_retention_loss_regression=None, max_target_retention_gap_regression=None, max_target_retention_ratio_regression=None, min_target_loss_margin=None, min_target_retention_ratio=None, min_retention_loss_margin=None, min_retention_accuracy_margin=None, min_retention_perplexity_margin=None, require_status_match=false, require_accepted_match=false, require_guard_match=false, require_movement_tolerance_match=false, require_resume_match=false))]
fn compare_sparse_finetune_summaries_py<'py>(
    py: Python<'py>,
    current: &Bound<'_, PyDict>,
    baseline: &Bound<'_, PyDict>,
    max_target_loss_regression: Option<f32>,
    max_retention_loss_regression: Option<f32>,
    max_target_retention_gap_regression: Option<f32>,
    max_target_retention_ratio_regression: Option<f32>,
    min_target_loss_margin: Option<f32>,
    min_target_retention_ratio: Option<f32>,
    min_retention_loss_margin: Option<f32>,
    min_retention_accuracy_margin: Option<f32>,
    min_retention_perplexity_margin: Option<f32>,
    require_status_match: bool,
    require_accepted_match: bool,
    require_guard_match: bool,
    require_movement_tolerance_match: bool,
    require_resume_match: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let current = sparse_finetune_summary_from_pydict(current)?;
    let baseline = sparse_finetune_summary_from_pydict(baseline)?;
    let mut limits = SparseFineTuneRegressionLimits::new()
        .with_status_match_required(require_status_match)
        .with_accepted_match_required(require_accepted_match)
        .with_guard_match_required(require_guard_match)
        .with_movement_tolerance_match_required(require_movement_tolerance_match)
        .with_resume_match_required(require_resume_match);
    if let Some(value) = max_target_loss_regression {
        limits = convert(limits.with_max_target_loss_regression(value))?;
    }
    if let Some(value) = max_retention_loss_regression {
        limits = convert(limits.with_max_retention_loss_regression(value))?;
    }
    if let Some(value) = max_target_retention_gap_regression {
        limits = convert(limits.with_max_target_retention_gap_regression(value))?;
    }
    if let Some(value) = max_target_retention_ratio_regression {
        limits = convert(limits.with_max_target_retention_ratio_regression(value))?;
    }
    if let Some(value) = min_target_loss_margin {
        limits = convert(limits.with_min_target_loss_margin(value))?;
    }
    if let Some(value) = min_target_retention_ratio {
        limits = convert(limits.with_min_target_retention_ratio(value))?;
    }
    if let Some(value) = min_retention_loss_margin {
        limits = convert(limits.with_min_retention_loss_margin(value))?;
    }
    if let Some(value) = min_retention_accuracy_margin {
        limits = convert(limits.with_min_retention_accuracy_margin(value))?;
    }
    if let Some(value) = min_retention_perplexity_margin {
        limits = convert(limits.with_min_retention_perplexity_margin(value))?;
    }
    let comparison = convert(current.compare_to(&baseline, limits))?;
    sparse_finetune_summary_comparison_to_pydict(py, &current, &baseline, comparison)
}

fn sparse_metrics_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<SparseClassificationMetrics> {
    fn required<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
    where
        T: FromPyObject<'py>,
    {
        dict.get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("sparse metrics missing '{key}'")))?
            .extract()
    }

    let active_rows = required(dict, "active_rows")?;
    let correct = required(dict, "correct")?;
    let accuracy = required(dict, "accuracy")?;
    let mean_loss = required(dict, "mean_loss")?;
    let perplexity = required(dict, "perplexity")?;
    Ok(SparseClassificationMetrics {
        active_rows,
        correct,
        accuracy,
        mean_loss,
        perplexity,
    })
}

#[pyfunction(name = "sparse_classification_delta")]
fn sparse_classification_delta_py<'py>(
    py: Python<'py>,
    before: &Bound<'_, PyDict>,
    after: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let before = sparse_metrics_from_pydict(before)?;
    let after = sparse_metrics_from_pydict(after)?;
    sparse_delta_to_pydict(py, before.delta_to(after))
}

fn epoch_history_from_py(history: &Bound<'_, PyAny>) -> PyResult<Vec<EpochStats>> {
    let stats: Vec<PyRef<'_, PyEpochStats>> = history.extract()?;
    Ok(stats.iter().map(|stat| stat.inner).collect())
}

#[pyfunction(name = "summarize_epoch_history")]
fn summarize_epoch_history_py(history: &Bound<'_, PyAny>) -> PyResult<PyEpochHistory> {
    let stats = epoch_history_from_py(history)?;
    Ok(PyEpochHistory::from_history(summarize_epoch_history(
        &stats,
    )))
}

#[pyclass(module = "spiraltorch", name = "ModuleTrainer", unsendable)]
struct PyModuleTrainer {
    inner: ModuleTrainer,
}

impl PyModuleTrainer {
    fn from_trainer(inner: ModuleTrainer) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyModuleTrainer {
    #[new]
    #[pyo3(signature = (device=None, curvature=-1.0, hyper_learning_rate=0.05, fallback_learning_rate=0.01))]
    fn new(
        device: Option<&str>,
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
    ) -> Self {
        let caps = caps_for(device);
        let inner =
            ModuleTrainer::new(caps, curvature, hyper_learning_rate, fallback_learning_rate);
        Self { inner }
    }

    #[getter]
    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    #[getter]
    fn hyper_learning_rate(&self) -> f32 {
        self.inner.hyper_learning_rate()
    }

    #[getter]
    fn fallback_learning_rate(&self) -> f32 {
        self.inner.fallback_learning_rate()
    }

    #[getter]
    fn max_grad_norm(&self) -> Option<f32> {
        self.inner.max_grad_norm()
    }

    #[getter]
    fn gradient_accumulation_steps(&self) -> usize {
        self.inner.gradient_accumulation_steps()
    }

    #[pyo3(signature = (max_norm=None))]
    fn set_max_grad_norm(&mut self, max_norm: Option<f32>) -> PyResult<()> {
        convert(self.inner.set_max_grad_norm(max_norm))
    }

    fn set_gradient_accumulation_steps(&mut self, steps: usize) -> PyResult<()> {
        convert(self.inner.set_gradient_accumulation_steps(steps))
    }

    fn state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        let fingerprint = self.inner.state_fingerprint();
        Ok(trainer_fingerprint_to_pydict(py, &fingerprint)?.into_py(py))
    }

    fn resume_fingerprint(&self, py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Ok(seq) = module.extract::<PyRef<'_, PySequentialModule>>() {
            let fingerprint = convert(self.inner.resume_fingerprint(seq.borrow()?))?;
            return Ok(resume_fingerprint_to_pydict(py, &fingerprint)?.into_py(py));
        }
        if let Ok(linear) = module.extract::<PyRef<'_, PyLinearModule>>() {
            let fingerprint = convert(self.inner.resume_fingerprint(linear.borrow()?))?;
            return Ok(resume_fingerprint_to_pydict(py, &fingerprint)?.into_py(py));
        }
        if let Ok(lora) = module.extract::<PyRef<'_, PyLoraLinearModule>>() {
            let fingerprint = convert(self.inner.resume_fingerprint(lora.borrow()?))?;
            return Ok(resume_fingerprint_to_pydict(py, &fingerprint)?.into_py(py));
        }
        Err(PyValueError::new_err(
            "ModuleTrainer.resume_fingerprint expects a Sequential, Linear, or LoraLinear module",
        ))
    }

    #[pyo3(signature = (rows, cols, top_k=8, mid_k=8, bottom_k=8, here_tolerance=1e-5, psychoid=false, psychoid_log=false, psi=false, collapse=false, dist=None))]
    fn roundtable(
        &mut self,
        rows: u32,
        cols: u32,
        top_k: u32,
        mid_k: u32,
        bottom_k: u32,
        here_tolerance: f32,
        psychoid: bool,
        psychoid_log: bool,
        psi: bool,
        collapse: bool,
        dist: Option<PyDistConfig>,
    ) -> PyResult<PyRoundtableSchedule> {
        let mut config = RoundtableConfig {
            top_k,
            mid_k,
            bottom_k,
            here_tolerance: here_tolerance.max(0.0),
            ..RoundtableConfig::default()
        };
        #[cfg(feature = "psychoid")]
        {
            if psychoid {
                config = if psychoid_log {
                    config.enable_psychoid_with_log()
                } else {
                    config.enable_psychoid()
                };
            }
        }
        #[cfg(feature = "psi")]
        {
            if psi {
                config = config.enable_psi();
            }
        }
        #[cfg(feature = "collapse")]
        {
            if collapse {
                config = config.enable_collapse();
            }
        }
        if let Some(dist_cfg) = dist {
            self.inner.configure_distribution(dist_cfg.inner.clone());
        } else {
            self.inner.clear_distribution();
        }
        Ok(PyRoundtableSchedule::from_schedule(
            self.inner.roundtable(rows, cols, config),
        ))
    }

    #[pyo3(signature = (threshold, participants=2))]
    fn install_meta_conductor(&mut self, threshold: f32, participants: usize) {
        self.inner.install_meta_conductor(threshold, participants);
    }

    #[pyo3(signature = (threshold, participants=2))]
    fn install_blackcat_moderator(&mut self, threshold: f32, participants: usize) {
        self.inner
            .install_blackcat_moderator(threshold, participants);
    }

    fn blackcat_minutes<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let minutes = self.inner.blackcat_minutes();
        let list = PyList::empty_bound(py);
        for minute in minutes {
            let entry = PyDict::new_bound(py);
            entry.set_item("plan_signature", minute.plan_signature.clone())?;
            entry.set_item("script_hint", minute.script_hint.clone())?;
            entry.set_item("winner", format!("{:?}", minute.winner))?;
            entry.set_item("support", minute.support)?;
            entry.set_item("mean_score", minute.mean_score)?;
            entry.set_item("mean_psi", minute.mean_psi)?;
            entry.set_item("confidence", (minute.confidence.0, minute.confidence.1))?;
            entry.set_item("reward", minute.reward)?;
            entry.set_item("notes", minute.notes.clone())?;
            entry.set_item(
                "issued_at",
                minute
                    .issued_at
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
            )?;
            let picks = PyDict::new_bound(py);
            for (k, v) in minute.picks.iter() {
                picks.set_item(k.clone(), v.clone())?;
            }
            entry.set_item("picks", picks)?;
            list.append(entry)?;
        }
        Ok(list.into_py(py))
    }

    #[pyo3(signature = (module, loss, batches, schedule))]
    fn train_epoch(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        batches: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
    ) -> PyResult<PyEpochStats> {
        let as_loader = batches.extract::<PyRef<PyDataLoader>>();
        if let Ok(loader) = as_loader {
            if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
                if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                    let stats = convert(self.inner.train_epoch(
                        seq.borrow_mut()?,
                        mse.inner_mut(),
                        loader.clone_inner(),
                        &schedule.inner,
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
                if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                    let stats = convert(self.inner.train_epoch(
                        seq.borrow_mut()?,
                        ce.inner_mut(),
                        loader.clone_inner(),
                        &schedule.inner,
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
            }

            if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
                if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                    let stats = convert(self.inner.train_epoch(
                        linear.borrow_mut()?,
                        mse.inner_mut(),
                        loader.clone_inner(),
                        &schedule.inner,
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
                if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                    let stats = convert(self.inner.train_epoch(
                        linear.borrow_mut()?,
                        ce.inner_mut(),
                        loader.clone_inner(),
                        &schedule.inner,
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
            }

            if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
                if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                    let stats = convert(self.inner.train_epoch(
                        lora.borrow_mut()?,
                        mse.inner_mut(),
                        loader.clone_inner(),
                        &schedule.inner,
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
                if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                    let stats = convert(self.inner.train_epoch(
                        lora.borrow_mut()?,
                        ce.inner_mut(),
                        loader.clone_inner(),
                        &schedule.inner,
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
            }
        }

        let dataset: Vec<(Tensor, Tensor)> = batches
            .extract::<Vec<(PyTensor, PyTensor)>>()?
            .into_iter()
            .map(|(input, target)| (input.into_tensor(), target.into_tensor()))
            .collect();

        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(self.inner.train_epoch(
                    seq.borrow_mut()?,
                    mse.inner_mut(),
                    dataset.clone(),
                    &schedule.inner,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let stats = convert(self.inner.train_epoch(
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    dataset.clone(),
                    &schedule.inner,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(self.inner.train_epoch(
                    linear.borrow_mut()?,
                    mse.inner_mut(),
                    dataset,
                    &schedule.inner,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let stats = convert(self.inner.train_epoch(
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    dataset,
                    &schedule.inner,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(self.inner.train_epoch(
                    lora.borrow_mut()?,
                    mse.inner_mut(),
                    dataset.clone(),
                    &schedule.inner,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let stats = convert(self.inner.train_epoch(
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    dataset,
                    &schedule.inner,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epoch expects a Sequential, Linear, or LoraLinear module and a supported loss",
        ))
    }

    #[pyo3(signature = (module, loss, batches))]
    fn evaluate_epoch(
        &self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        batches: &Bound<'_, PyAny>,
    ) -> PyResult<PyEpochStats> {
        let as_loader = batches.extract::<PyRef<PyDataLoader>>();
        if let Ok(loader) = as_loader {
            if let Ok(seq) = module.extract::<PyRef<'_, PySequentialModule>>() {
                if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                    let stats = convert(self.inner.evaluate_epoch(
                        seq.borrow()?,
                        mse.inner_mut(),
                        loader.clone_inner(),
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
                if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                    let stats = convert(self.inner.evaluate_epoch(
                        seq.borrow()?,
                        ce.inner_mut(),
                        loader.clone_inner(),
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
            }

            if let Ok(linear) = module.extract::<PyRef<'_, PyLinearModule>>() {
                if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                    let stats = convert(self.inner.evaluate_epoch(
                        linear.borrow()?,
                        mse.inner_mut(),
                        loader.clone_inner(),
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
                if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                    let stats = convert(self.inner.evaluate_epoch(
                        linear.borrow()?,
                        ce.inner_mut(),
                        loader.clone_inner(),
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
            }

            if let Ok(lora) = module.extract::<PyRef<'_, PyLoraLinearModule>>() {
                if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                    let stats = convert(self.inner.evaluate_epoch(
                        lora.borrow()?,
                        mse.inner_mut(),
                        loader.clone_inner(),
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
                if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                    let stats = convert(self.inner.evaluate_epoch(
                        lora.borrow()?,
                        ce.inner_mut(),
                        loader.clone_inner(),
                    ))?;
                    return Ok(PyEpochStats::from_stats(stats));
                }
            }
        }

        let dataset: Vec<(Tensor, Tensor)> = batches
            .extract::<Vec<(PyTensor, PyTensor)>>()?
            .into_iter()
            .map(|(input, target)| (input.into_tensor(), target.into_tensor()))
            .collect();

        if let Ok(seq) = module.extract::<PyRef<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(self.inner.evaluate_epoch(
                    seq.borrow()?,
                    mse.inner_mut(),
                    dataset.clone(),
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let stats = convert(self.inner.evaluate_epoch(
                    seq.borrow()?,
                    ce.inner_mut(),
                    dataset.clone(),
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
        }

        if let Ok(linear) = module.extract::<PyRef<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(self.inner.evaluate_epoch(
                    linear.borrow()?,
                    mse.inner_mut(),
                    dataset,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let stats = convert(self.inner.evaluate_epoch(
                    linear.borrow()?,
                    ce.inner_mut(),
                    dataset,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
        }

        if let Ok(lora) = module.extract::<PyRef<'_, PyLoraLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let stats = convert(self.inner.evaluate_epoch(
                    lora.borrow()?,
                    mse.inner_mut(),
                    dataset.clone(),
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let stats = convert(self.inner.evaluate_epoch(
                    lora.borrow()?,
                    ce.inner_mut(),
                    dataset,
                ))?;
                return Ok(PyEpochStats::from_stats(stats));
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.evaluate_epoch expects a Sequential, Linear, or LoraLinear module and a supported loss",
        ))
    }

    #[pyo3(signature = (module, loss, batches))]
    fn evaluate_sparse_classification_epoch<'py>(
        &self,
        py: Python<'py>,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        batches: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ce = loss.extract::<PyRef<'_, PySoftmaxCrossEntropy>>()?;
        let as_loader = batches.extract::<PyRef<PyDataLoader>>();
        if let Ok(loader) = as_loader {
            if let Ok(seq) = module.extract::<PyRef<'_, PySequentialModule>>() {
                let metrics = convert(self.inner.evaluate_sparse_classification_epoch(
                    seq.borrow()?,
                    &ce.inner,
                    loader.clone_inner(),
                ))?;
                return sparse_metrics_to_pydict(py, metrics);
            }

            if let Ok(linear) = module.extract::<PyRef<'_, PyLinearModule>>() {
                let metrics = convert(self.inner.evaluate_sparse_classification_epoch(
                    linear.borrow()?,
                    &ce.inner,
                    loader.clone_inner(),
                ))?;
                return sparse_metrics_to_pydict(py, metrics);
            }

            if let Ok(lora) = module.extract::<PyRef<'_, PyLoraLinearModule>>() {
                let metrics = convert(self.inner.evaluate_sparse_classification_epoch(
                    lora.borrow()?,
                    &ce.inner,
                    loader.clone_inner(),
                ))?;
                return sparse_metrics_to_pydict(py, metrics);
            }
        }

        let dataset: Vec<(Tensor, Tensor)> = batches
            .extract::<Vec<(PyTensor, PyTensor)>>()?
            .into_iter()
            .map(|(input, target)| (input.into_tensor(), target.into_tensor()))
            .collect();

        if let Ok(seq) = module.extract::<PyRef<'_, PySequentialModule>>() {
            let metrics = convert(self.inner.evaluate_sparse_classification_epoch(
                seq.borrow()?,
                &ce.inner,
                dataset.clone(),
            ))?;
            return sparse_metrics_to_pydict(py, metrics);
        }

        if let Ok(linear) = module.extract::<PyRef<'_, PyLinearModule>>() {
            let metrics = convert(self.inner.evaluate_sparse_classification_epoch(
                linear.borrow()?,
                &ce.inner,
                dataset,
            ))?;
            return sparse_metrics_to_pydict(py, metrics);
        }

        if let Ok(lora) = module.extract::<PyRef<'_, PyLoraLinearModule>>() {
            let metrics = convert(self.inner.evaluate_sparse_classification_epoch(
                lora.borrow()?,
                &ce.inner,
                dataset,
            ))?;
            return sparse_metrics_to_pydict(py, metrics);
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.evaluate_sparse_classification_epoch expects a Sequential, Linear, or LoraLinear module, SoftmaxCrossEntropy loss, and batches",
        ))
    }

    #[pyo3(signature = (module, loss, loader, schedule, epochs))]
    fn train_epochs(
        &mut self,
        py: Python<'_>,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
    ) -> PyResult<PyObject> {
        let loader = loader.extract::<PyRef<PyDataLoader>>()?;
        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let history = convert(self.inner.train_epochs(
                    seq.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return epoch_history_to_pylist(py, history);
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let history = convert(self.inner.train_epochs(
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return epoch_history_to_pylist(py, history);
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let history = convert(self.inner.train_epochs(
                    linear.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return epoch_history_to_pylist(py, history);
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let history = convert(self.inner.train_epochs(
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return epoch_history_to_pylist(py, history);
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let history = convert(self.inner.train_epochs(
                    lora.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return epoch_history_to_pylist(py, history);
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let history = convert(self.inner.train_epochs(
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return epoch_history_to_pylist(py, history);
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epochs expects a DataLoader, a Sequential, Linear, or LoraLinear module, and a supported loss",
        ))
    }

    #[pyo3(signature = (module, loss, loader, schedule, epochs))]
    fn train_epochs_capture_best(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
    ) -> PyResult<PyEpochBestState> {
        let loader = loader.extract::<PyRef<PyDataLoader>>()?;
        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = convert(self.inner.train_epochs_capture_best(
                    seq.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = convert(self.inner.train_epochs_capture_best(
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = convert(self.inner.train_epochs_capture_best(
                    linear.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = convert(self.inner.train_epochs_capture_best(
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = convert(self.inner.train_epochs_capture_best(
                    lora.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = convert(self.inner.train_epochs_capture_best(
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epochs_capture_best expects a DataLoader, a Sequential, Linear, or LoraLinear module, and a supported loss",
        ))
    }

    #[pyo3(signature = (module, loss, loader, schedule, epochs))]
    fn train_epochs_restore_best(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
    ) -> PyResult<PyEpochBestState> {
        let loader = loader.extract::<PyRef<PyDataLoader>>()?;
        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = convert(self.inner.train_epochs_restore_best(
                    seq.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = convert(self.inner.train_epochs_restore_best(
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = convert(self.inner.train_epochs_restore_best(
                    linear.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = convert(self.inner.train_epochs_restore_best(
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = convert(self.inner.train_epochs_restore_best(
                    lora.borrow_mut()?,
                    mse.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = convert(self.inner.train_epochs_restore_best(
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                ))?;
                return Ok(PyEpochBestState::from_best_state(captured));
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epochs_restore_best expects a DataLoader, a Sequential, Linear, or LoraLinear module, and a supported loss",
        ))
    }

    #[pyo3(signature = (module, loss, train_loader, validation_loader, schedule, epochs, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_capture_best_on_validation(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PyEpochValidationBestState> {
        let train_loader = train_loader.extract::<PyRef<PyDataLoader>>()?;
        let validation_loader = validation_loader.extract::<PyRef<PyDataLoader>>()?;
        let controls = validation_controls_from_py(
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )?;
        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = capture_validation_best_with_controls(
                    &mut self.inner,
                    seq.borrow_mut()?,
                    mse.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = capture_validation_best_with_controls(
                    &mut self.inner,
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = capture_validation_best_with_controls(
                    &mut self.inner,
                    linear.borrow_mut()?,
                    mse.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = capture_validation_best_with_controls(
                    &mut self.inner,
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = capture_validation_best_with_controls(
                    &mut self.inner,
                    lora.borrow_mut()?,
                    mse.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = capture_validation_best_with_controls(
                    &mut self.inner,
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epochs_capture_best_on_validation expects train/validation DataLoaders, a Sequential, Linear, or LoraLinear module, and a supported loss",
        ))
    }

    #[pyo3(signature = (module, loss, train_loader, validation_loader, schedule, epochs, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_restore_best_on_validation(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PyEpochValidationBestState> {
        let train_loader = train_loader.extract::<PyRef<PyDataLoader>>()?;
        let validation_loader = validation_loader.extract::<PyRef<PyDataLoader>>()?;
        let controls = validation_controls_from_py(
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )?;
        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = restore_validation_best_with_controls(
                    &mut self.inner,
                    seq.borrow_mut()?,
                    mse.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = restore_validation_best_with_controls(
                    &mut self.inner,
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = restore_validation_best_with_controls(
                    &mut self.inner,
                    linear.borrow_mut()?,
                    mse.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = restore_validation_best_with_controls(
                    &mut self.inner,
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut mse) = loss.extract::<PyRefMut<'_, PyMeanSquaredError>>() {
                let captured = restore_validation_best_with_controls(
                    &mut self.inner,
                    lora.borrow_mut()?,
                    mse.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = restore_validation_best_with_controls(
                    &mut self.inner,
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    controls,
                )?;
                return Ok(PyEpochValidationBestState::from_validation_best_state(
                    captured,
                ));
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epochs_restore_best_on_validation expects train/validation DataLoaders, a Sequential, Linear, or LoraLinear module, and a supported loss",
        ))
    }

    #[pyo3(signature = (module, loss, train_loader, validation_loader, retention_loader, schedule, epochs, max_loss_increase=0.5, max_accuracy_drop=0.25, max_perplexity_increase=None, target_min_loss_delta=0.0, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_capture_best_sparse_with_retention_guard(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        retention_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        max_loss_increase: f32,
        max_accuracy_drop: f32,
        max_perplexity_increase: Option<f32>,
        target_min_loss_delta: f32,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PyEpochSparseRetentionBestState> {
        let train_loader = train_loader.extract::<PyRef<PyDataLoader>>()?;
        let validation_loader = validation_loader.extract::<PyRef<PyDataLoader>>()?;
        let retention_loader = retention_loader.extract::<PyRef<PyDataLoader>>()?;
        let controls = validation_controls_from_py(
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )?;
        let guard = sparse_retention_guard_from_py(
            max_loss_increase,
            max_accuracy_drop,
            max_perplexity_increase,
            target_min_loss_delta,
        )?;

        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = capture_sparse_retention_best_with_controls(
                    &mut self.inner,
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    controls,
                )?;
                return Ok(
                    PyEpochSparseRetentionBestState::from_sparse_retention_best_state(captured),
                );
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = capture_sparse_retention_best_with_controls(
                    &mut self.inner,
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    controls,
                )?;
                return Ok(
                    PyEpochSparseRetentionBestState::from_sparse_retention_best_state(captured),
                );
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = capture_sparse_retention_best_with_controls(
                    &mut self.inner,
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    controls,
                )?;
                return Ok(
                    PyEpochSparseRetentionBestState::from_sparse_retention_best_state(captured),
                );
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epochs_capture_best_sparse_with_retention_guard expects train/validation/retention DataLoaders, a Sequential, Linear, or LoraLinear module, and SoftmaxCrossEntropy",
        ))
    }

    #[pyo3(signature = (module, loss, train_loader, validation_loader, retention_loader, schedule, epochs, max_loss_increase=0.5, max_accuracy_drop=0.25, max_perplexity_increase=None, target_min_loss_delta=0.0, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_restore_best_sparse_with_retention_guard(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        retention_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        max_loss_increase: f32,
        max_accuracy_drop: f32,
        max_perplexity_increase: Option<f32>,
        target_min_loss_delta: f32,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PyEpochSparseRetentionBestState> {
        let train_loader = train_loader.extract::<PyRef<PyDataLoader>>()?;
        let validation_loader = validation_loader.extract::<PyRef<PyDataLoader>>()?;
        let retention_loader = retention_loader.extract::<PyRef<PyDataLoader>>()?;
        let controls = validation_controls_from_py(
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )?;
        let guard = sparse_retention_guard_from_py(
            max_loss_increase,
            max_accuracy_drop,
            max_perplexity_increase,
            target_min_loss_delta,
        )?;

        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = restore_sparse_retention_best_with_controls(
                    &mut self.inner,
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    controls,
                )?;
                return Ok(
                    PyEpochSparseRetentionBestState::from_sparse_retention_best_state(captured),
                );
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = restore_sparse_retention_best_with_controls(
                    &mut self.inner,
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    controls,
                )?;
                return Ok(
                    PyEpochSparseRetentionBestState::from_sparse_retention_best_state(captured),
                );
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let captured = restore_sparse_retention_best_with_controls(
                    &mut self.inner,
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    controls,
                )?;
                return Ok(
                    PyEpochSparseRetentionBestState::from_sparse_retention_best_state(captured),
                );
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epochs_restore_best_sparse_with_retention_guard expects train/validation/retention DataLoaders, a Sequential, Linear, or LoraLinear module, and SoftmaxCrossEntropy",
        ))
    }

    #[pyo3(signature = (module, loss, train_loader, validation_loader, retention_loader, schedule, epochs, movement_tolerance=1e-8, max_loss_increase=0.5, max_accuracy_drop=0.25, max_perplexity_increase=None, target_min_loss_delta=0.0, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_restore_best_sparse_with_finetune_report(
        &mut self,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        retention_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        movement_tolerance: f32,
        max_loss_increase: f32,
        max_accuracy_drop: f32,
        max_perplexity_increase: Option<f32>,
        target_min_loss_delta: f32,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PySparseFineTuneReport> {
        let train_loader = train_loader.extract::<PyRef<PyDataLoader>>()?;
        let validation_loader = validation_loader.extract::<PyRef<PyDataLoader>>()?;
        let retention_loader = retention_loader.extract::<PyRef<PyDataLoader>>()?;
        let controls = validation_controls_from_py(
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )?;
        let guard = sparse_retention_guard_from_py(
            max_loss_increase,
            max_accuracy_drop,
            max_perplexity_increase,
            target_min_loss_delta,
        )?;

        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let report = restore_sparse_finetune_report_with_controls(
                    &mut self.inner,
                    seq.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    movement_tolerance,
                    controls,
                )?;
                return Ok(PySparseFineTuneReport::from_sparse_finetune_report(report));
            }
        }

        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let report = restore_sparse_finetune_report_with_controls(
                    &mut self.inner,
                    linear.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    movement_tolerance,
                    controls,
                )?;
                return Ok(PySparseFineTuneReport::from_sparse_finetune_report(report));
            }
        }

        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            if let Ok(mut ce) = loss.extract::<PyRefMut<'_, PySoftmaxCrossEntropy>>() {
                let report = restore_sparse_finetune_report_with_controls(
                    &mut self.inner,
                    lora.borrow_mut()?,
                    ce.inner_mut(),
                    train_loader.clone_inner(),
                    validation_loader.clone_inner(),
                    retention_loader.clone_inner(),
                    &schedule.inner,
                    epochs,
                    guard,
                    movement_tolerance,
                    controls,
                )?;
                return Ok(PySparseFineTuneReport::from_sparse_finetune_report(report));
            }
        }

        Err(PyValueError::new_err(
            "ModuleTrainer.train_epochs_restore_best_sparse_with_finetune_report expects train/validation/retention DataLoaders, a Sequential, Linear, or LoraLinear module, and SoftmaxCrossEntropy",
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ModuleTrainer(curvature={}, hyper_lr={:.4}, fallback_lr={:.4}, gradient_accumulation_steps={})",
            self.curvature(),
            self.hyper_learning_rate(),
            self.fallback_learning_rate(),
            self.gradient_accumulation_steps()
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "SpiralSessionBuilder")]
struct PySpiralSessionBuilder {
    builder: Option<SpiralSessionBuilder>,
}

impl PySpiralSessionBuilder {
    fn from_builder(builder: SpiralSessionBuilder) -> Self {
        Self {
            builder: Some(builder),
        }
    }

    fn ensure_builder(&mut self) -> PyResult<&mut SpiralSessionBuilder> {
        self.builder
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("builder has already been consumed"))
    }
}

#[pymethods]
impl PySpiralSessionBuilder {
    #[new]
    #[pyo3(signature = (device=None))]
    fn new(device: Option<&str>) -> Self {
        let caps = caps_for(device);
        Self {
            builder: Some(SpiralSession::builder(caps)),
        }
    }

    fn curvature(&mut self, curvature: f32) -> PyResult<()> {
        self.ensure_builder()?.set_curvature(curvature);
        Ok(())
    }

    fn hyper_learning_rate(&mut self, learning_rate: f32) -> PyResult<()> {
        self.ensure_builder()?
            .set_hyper_learning_rate(learning_rate);
        Ok(())
    }

    fn fallback_learning_rate(&mut self, learning_rate: f32) -> PyResult<()> {
        self.ensure_builder()?
            .set_fallback_learning_rate(learning_rate);
        Ok(())
    }

    fn entropy_weight(&mut self, entropy_weight: f32) -> PyResult<()> {
        self.ensure_builder()?
            .set_barycenter_entropy(entropy_weight);
        Ok(())
    }

    fn beta_j(&mut self, beta_j: f32) -> PyResult<()> {
        self.ensure_builder()?.set_barycenter_beta_j(beta_j);
        Ok(())
    }

    #[pyo3(signature = (coupling=None))]
    fn coupling(&mut self, coupling: Option<PyTensor>) -> PyResult<()> {
        let tensor = coupling.map(PyTensor::into_tensor);
        self.ensure_builder()?.set_barycenter_coupling(tensor);
        Ok(())
    }

    fn topos_guard(&mut self, topos: &PyOpenTopos) -> PyResult<()> {
        self.ensure_builder()?.set_topos(Some(topos.inner.clone()));
        Ok(())
    }

    #[pyo3(signature = (curvature, tolerance, saturation, max_depth, max_volume))]
    fn topos(
        &mut self,
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PyResult<()> {
        self.ensure_builder()?
            .set_topos_from_params(curvature, tolerance, saturation, max_depth, max_volume)
            .map_err(tensor_err)?;
        Ok(())
    }

    fn clear_topos(&mut self) {
        if let Some(builder) = self.builder.as_mut() {
            builder.set_topos(None);
        }
    }

    fn build(&mut self) -> PyResult<PySpiralSession> {
        let builder = self
            .builder
            .take()
            .ok_or_else(|| PyValueError::new_err("builder has already been consumed"))?;
        let session = convert(builder.build())?;
        Ok(PySpiralSession::from_session(session))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("SpiralSessionBuilder(...)".to_string())
    }
}

#[pyclass(module = "spiraltorch", name = "SpiralSession")]
#[derive(Clone)]
struct PySpiralSession {
    inner: SpiralSession,
}

impl PySpiralSession {
    fn from_session(inner: SpiralSession) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySpiralSession {
    #[new]
    #[pyo3(signature = (device=None, curvature=-1.0, hyper_learning_rate=0.05, fallback_learning_rate=0.01, entropy_weight=0.1, beta_j=0.0, topos=None, coupling=None))]
    fn new(
        device: Option<&str>,
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
        entropy_weight: f32,
        beta_j: f32,
        topos: Option<&PyOpenTopos>,
        coupling: Option<&PyTensor>,
    ) -> PyResult<Self> {
        let caps = caps_for(device);
        let mut builder = SpiralSession::builder(caps);
        builder.set_curvature(curvature);
        builder.set_hyper_learning_rate(hyper_learning_rate);
        builder.set_fallback_learning_rate(fallback_learning_rate);
        builder.set_barycenter_entropy(entropy_weight);
        builder.set_barycenter_beta_j(beta_j);
        if let Some(topos) = topos {
            builder.set_topos(Some(topos.inner.clone()));
        }
        if let Some(coupling) = coupling {
            builder.set_barycenter_coupling(Some(coupling.as_tensor().clone()));
        }
        let session = convert(builder.build())?;
        Ok(Self { inner: session })
    }

    fn builder(&self) -> PySpiralSessionBuilder {
        PySpiralSessionBuilder::from_builder(self.inner.to_builder())
    }

    fn trainer(&self) -> PyModuleTrainer {
        PyModuleTrainer::from_trainer(self.inner.trainer())
    }

    #[pyo3(signature = (rows, cols, top_k=8, mid_k=8, bottom_k=8, here_tolerance=1e-5, psychoid=false, psychoid_log=false, psi=false, collapse=false))]
    fn roundtable(
        &self,
        rows: u32,
        cols: u32,
        top_k: u32,
        mid_k: u32,
        bottom_k: u32,
        here_tolerance: f32,
        psychoid: bool,
        psychoid_log: bool,
        psi: bool,
        collapse: bool,
    ) -> PyRoundtableSchedule {
        let mut config = RoundtableConfig {
            top_k,
            mid_k,
            bottom_k,
            here_tolerance: here_tolerance.max(0.0),
            ..RoundtableConfig::default()
        };
        #[cfg(feature = "psychoid")]
        {
            if psychoid {
                config = if psychoid_log {
                    config.enable_psychoid_with_log()
                } else {
                    config.enable_psychoid()
                };
            }
        }
        #[cfg(feature = "psi")]
        {
            if psi {
                config = config.enable_psi();
            }
        }
        #[cfg(feature = "collapse")]
        {
            if collapse {
                config = config.enable_collapse();
            }
        }
        PyRoundtableSchedule::from_schedule(self.inner.roundtable(rows, cols, config))
    }

    #[pyo3(signature = (module))]
    fn prepare_module(&self, module: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(mut seq) = module.extract::<PyRefMut<'_, PySequentialModule>>() {
            return convert(self.inner.prepare_module(seq.borrow_mut()?));
        }
        if let Ok(mut linear) = module.extract::<PyRefMut<'_, PyLinearModule>>() {
            return convert(self.inner.prepare_module(linear.borrow_mut()?));
        }
        if let Ok(mut lora) = module.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
            return convert(self.inner.prepare_module(lora.borrow_mut()?));
        }
        if let Ok(mut conv) = module.extract::<PyRefMut<'_, PyConv1dModule>>() {
            return convert(self.inner.prepare_module(conv.borrow_mut()?));
        }
        if let Ok(mut wave) = module.extract::<PyRefMut<'_, PyWaveRnnModule>>() {
            return convert(self.inner.prepare_module(wave.borrow_mut()?));
        }
        if let Ok(mut projector) = module.extract::<PyRefMut<'_, PyZSpaceProjector>>() {
            return convert(self.inner.prepare_module(projector.borrow_mut()?));
        }

        Err(PyValueError::new_err(
            "prepare_module expects Linear, LoraLinear, Conv1d, WaveRnn, ZSpaceProjector, or Sequential modules",
        ))
    }

    #[pyo3(signature = (trainer, module, loss, batches, schedule))]
    fn train_epoch(
        &self,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        batches: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
    ) -> PyResult<PyEpochStats> {
        trainer.train_epoch(module, loss, batches, schedule)
    }

    #[pyo3(signature = (trainer, module, loss, batches))]
    fn evaluate_epoch(
        &self,
        trainer: &PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        batches: &Bound<'_, PyAny>,
    ) -> PyResult<PyEpochStats> {
        trainer.evaluate_epoch(module, loss, batches)
    }

    #[pyo3(signature = (trainer, module, loss, batches))]
    fn evaluate_sparse_classification_epoch<'py>(
        &self,
        py: Python<'py>,
        trainer: &PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        batches: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        trainer.evaluate_sparse_classification_epoch(py, module, loss, batches)
    }

    #[pyo3(signature = (trainer, module, loss, loader, schedule, epochs))]
    fn train_epochs(
        &self,
        py: Python<'_>,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
    ) -> PyResult<PyObject> {
        trainer.train_epochs(py, module, loss, loader, schedule, epochs)
    }

    #[pyo3(signature = (trainer, module, loss, loader, schedule, epochs))]
    fn train_epochs_capture_best(
        &self,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
    ) -> PyResult<PyEpochBestState> {
        trainer.train_epochs_capture_best(module, loss, loader, schedule, epochs)
    }

    #[pyo3(signature = (trainer, module, loss, loader, schedule, epochs))]
    fn train_epochs_restore_best(
        &self,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
    ) -> PyResult<PyEpochBestState> {
        trainer.train_epochs_restore_best(module, loss, loader, schedule, epochs)
    }

    #[pyo3(signature = (trainer, module, loss, train_loader, validation_loader, schedule, epochs, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_capture_best_on_validation(
        &self,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PyEpochValidationBestState> {
        trainer.train_epochs_capture_best_on_validation(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )
    }

    #[pyo3(signature = (trainer, module, loss, train_loader, validation_loader, schedule, epochs, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_restore_best_on_validation(
        &self,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PyEpochValidationBestState> {
        trainer.train_epochs_restore_best_on_validation(
            module,
            loss,
            train_loader,
            validation_loader,
            schedule,
            epochs,
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )
    }

    #[pyo3(signature = (trainer, module, loss, train_loader, validation_loader, retention_loader, schedule, epochs, max_loss_increase=0.5, max_accuracy_drop=0.25, max_perplexity_increase=None, target_min_loss_delta=0.0, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_capture_best_sparse_with_retention_guard(
        &self,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        retention_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        max_loss_increase: f32,
        max_accuracy_drop: f32,
        max_perplexity_increase: Option<f32>,
        target_min_loss_delta: f32,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PyEpochSparseRetentionBestState> {
        trainer.train_epochs_capture_best_sparse_with_retention_guard(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            max_loss_increase,
            max_accuracy_drop,
            max_perplexity_increase,
            target_min_loss_delta,
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )
    }

    #[pyo3(signature = (trainer, module, loss, train_loader, validation_loader, retention_loader, schedule, epochs, max_loss_increase=0.5, max_accuracy_drop=0.25, max_perplexity_increase=None, target_min_loss_delta=0.0, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_restore_best_sparse_with_retention_guard(
        &self,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        retention_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        max_loss_increase: f32,
        max_accuracy_drop: f32,
        max_perplexity_increase: Option<f32>,
        target_min_loss_delta: f32,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PyEpochSparseRetentionBestState> {
        trainer.train_epochs_restore_best_sparse_with_retention_guard(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            max_loss_increase,
            max_accuracy_drop,
            max_perplexity_increase,
            target_min_loss_delta,
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )
    }

    #[pyo3(signature = (trainer, module, loss, train_loader, validation_loader, retention_loader, schedule, epochs, movement_tolerance=1e-8, max_loss_increase=0.5, max_accuracy_drop=0.25, max_perplexity_increase=None, target_min_loss_delta=0.0, patience=None, min_delta=0.0, lr_decay_patience=None, lr_decay_factor=0.5, lr_decay_min_delta=0.0))]
    fn train_epochs_restore_best_sparse_with_finetune_report(
        &self,
        trainer: &mut PyModuleTrainer,
        module: &Bound<'_, PyAny>,
        loss: &Bound<'_, PyAny>,
        train_loader: &Bound<'_, PyAny>,
        validation_loader: &Bound<'_, PyAny>,
        retention_loader: &Bound<'_, PyAny>,
        schedule: &PyRoundtableSchedule,
        epochs: usize,
        movement_tolerance: f32,
        max_loss_increase: f32,
        max_accuracy_drop: f32,
        max_perplexity_increase: Option<f32>,
        target_min_loss_delta: f32,
        patience: Option<usize>,
        min_delta: f32,
        lr_decay_patience: Option<usize>,
        lr_decay_factor: f32,
        lr_decay_min_delta: f32,
    ) -> PyResult<PySparseFineTuneReport> {
        trainer.train_epochs_restore_best_sparse_with_finetune_report(
            module,
            loss,
            train_loader,
            validation_loader,
            retention_loader,
            schedule,
            epochs,
            movement_tolerance,
            max_loss_increase,
            max_accuracy_drop,
            max_perplexity_increase,
            target_min_loss_delta,
            patience,
            min_delta,
            lr_decay_patience,
            lr_decay_factor,
            lr_decay_min_delta,
        )
    }

    #[pyo3(signature = (seed, sot=None))]
    fn trace(
        &self,
        seed: &PyTensor,
        sot: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PySpiralDifferentialTrace> {
        let (rows, cols) = seed.as_tensor().shape();
        let default_steps = match rows.checked_mul(cols) {
            Some(0) | None => 1,
            Some(value) => value.max(1),
        };
        let mut plan_steps = default_steps;
        let mut params = Sot3DParams {
            base_radius: 1.0,
            radial_growth: 0.05,
            base_height: 1.0,
            meso_gain: 0.2,
            micro_gain: 0.05,
        };
        if let Some(cfg) = sot {
            if let Some(value) = cfg.get_item("steps")? {
                plan_steps = value.extract()?;
            }
            if let Some(value) = cfg.get_item("base_radius")? {
                params.base_radius = value.extract()?;
            }
            if let Some(value) = cfg.get_item("radial_growth")? {
                params.radial_growth = value.extract()?;
            }
            if let Some(value) = cfg.get_item("base_height")? {
                params.base_height = value.extract()?;
            }
            if let Some(value) = cfg.get_item("meso_gain")? {
                params.meso_gain = value.extract()?;
            }
            if let Some(value) = cfg.get_item("micro_gain")? {
                params.micro_gain = value.extract()?;
            }
        }

        let trace = convert(self.inner.trace(seed.as_tensor().clone()))?;
        let plan = if plan_steps == 0 {
            None
        } else {
            Some(crate::sot::generate_plan_with_params(plan_steps, params)?)
        };
        Ok(PySpiralDifferentialTrace::from_trace_with_plan(trace, plan))
    }

    #[getter]
    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }

    #[getter]
    fn hyper_learning_rate(&self) -> f32 {
        self.inner.hyper_learning_rate()
    }

    #[getter]
    fn fallback_learning_rate(&self) -> f32 {
        self.inner.fallback_learning_rate()
    }

    #[getter]
    fn entropy_weight(&self) -> f32 {
        self.inner.barycenter_entropy_weight()
    }

    #[getter]
    fn beta_j(&self) -> f32 {
        self.inner.barycenter_beta_j()
    }

    #[getter]
    fn coupling(&self) -> Option<PyTensor> {
        self.inner
            .barycenter_coupling()
            .cloned()
            .map(PyTensor::from_tensor)
    }

    fn device_caps(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(device_caps_dict(py, self.inner.device_caps())?.into_py(py))
    }

    #[pyo3(signature = (kind, rows, cols, k))]
    fn plan(&self, py: Python<'_>, kind: &str, rows: u32, cols: u32, k: u32) -> PyResult<PyObject> {
        let rank_kind = parse_kind(kind)?;
        let plan = self.inner.plan_rank(rank_kind, rows, cols, k);
        let out = PyDict::new_bound(py);
        out.set_item("kind", kind.to_ascii_lowercase())?;
        out.set_item("rows", rows)?;
        out.set_item("cols", cols)?;
        out.set_item("k", k)?;
        out.set_item("choice", choice_dict(py, &plan)?.into_py(py))?;
        Ok(out.into_py(py))
    }

    fn hypergrad(&self, rows: usize, cols: usize) -> PyResult<PyHypergrad> {
        Ok(PyHypergrad::from_hypergrad(convert(
            self.inner.hypergrad(rows, cols),
        )?))
    }

    #[pyo3(signature = (densities, weights=None, entropy_weight=None, beta_j=None, coupling=None))]
    fn barycenter(
        &self,
        densities: Vec<PyTensor>,
        weights: Option<Vec<f32>>,
        entropy_weight: Option<f32>,
        beta_j: Option<f32>,
        coupling: Option<PyTensor>,
    ) -> PyResult<PyZSpaceBarycenter> {
        if densities.is_empty() {
            return Err(PyValueError::new_err("densities must not be empty"));
        }
        let tensors: Vec<Tensor> = densities.into_iter().map(PyTensor::into_tensor).collect();
        let weight_vec = weights.unwrap_or_else(|| vec![1.0; tensors.len()]);
        if weight_vec.len() != tensors.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} weights, received {}",
                tensors.len(),
                weight_vec.len()
            )));
        }
        let coupling_tensor = coupling.map(PyTensor::into_tensor);
        let coupling_ref = coupling_tensor.as_ref();
        let entropy = entropy_weight.unwrap_or_else(|| self.inner.barycenter_entropy_weight());
        let beta = beta_j.unwrap_or_else(|| self.inner.barycenter_beta_j());
        let result = self.inner.barycenter_with_parameters(
            &weight_vec,
            &tensors,
            entropy,
            beta,
            coupling_ref,
        );
        Ok(PyZSpaceBarycenter::from_result(convert(result)?))
    }

    fn align_hypergrad(
        &self,
        hypergrad: &mut PyHypergrad,
        barycenter: &PyZSpaceBarycenter,
    ) -> PyResult<()> {
        convert(
            self.inner
                .align_hypergrad(&mut hypergrad.inner, &barycenter.inner),
        )
    }

    #[getter]
    fn topos(&self) -> Option<PyOpenTopos> {
        self.inner.topos().cloned().map(PyOpenTopos::from_topos)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SpiralSession(device={}, curvature={}, hyper_lr={}, fallback_lr={})",
            backend_name(self.inner.device_caps().backend),
            self.inner.curvature(),
            self.inner.hyper_learning_rate(),
            self.inner.fallback_learning_rate()
        ))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "MeanSquaredError")]
struct PyMeanSquaredError {
    inner: MeanSquaredError,
}

impl PyMeanSquaredError {
    fn inner_mut(&mut self) -> &mut MeanSquaredError {
        &mut self.inner
    }
}

#[pymethods]
impl PyMeanSquaredError {
    #[new]
    fn new() -> Self {
        Self {
            inner: MeanSquaredError::new(),
        }
    }

    fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.inner
                .forward(prediction.as_tensor(), target.as_tensor()),
        )?))
    }

    fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.inner
                .backward(prediction.as_tensor(), target.as_tensor()),
        )?))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok("MeanSquaredError()".to_string())
    }
}

#[pyclass(module = "spiraltorch.nn", name = "SoftmaxCrossEntropy")]
struct PySoftmaxCrossEntropy {
    inner: SoftmaxCrossEntropy,
}

impl PySoftmaxCrossEntropy {
    fn inner_mut(&mut self) -> &mut SoftmaxCrossEntropy {
        &mut self.inner
    }
}

#[pymethods]
impl PySoftmaxCrossEntropy {
    #[new]
    #[pyo3(signature = (ignore_index=None, label_smoothing=0.0))]
    fn new(ignore_index: Option<i32>, label_smoothing: f32) -> PyResult<Self> {
        let mut inner = SoftmaxCrossEntropy::new();
        inner.set_ignore_index(ignore_index);
        convert(inner.set_label_smoothing(label_smoothing))?;
        Ok(Self { inner })
    }

    #[getter]
    fn ignore_index(&self) -> Option<i32> {
        self.inner.ignore_index()
    }

    #[setter]
    fn set_ignore_index(&mut self, ignore_index: Option<i32>) {
        self.inner.set_ignore_index(ignore_index);
    }

    #[getter]
    fn label_smoothing(&self) -> f32 {
        self.inner.label_smoothing()
    }

    #[setter]
    fn set_label_smoothing(&mut self, label_smoothing: f32) -> PyResult<()> {
        convert(self.inner.set_label_smoothing(label_smoothing))
    }

    fn forward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.inner
                .forward(prediction.as_tensor(), target.as_tensor()),
        )?))
    }

    fn backward(&mut self, prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.inner
                .backward(prediction.as_tensor(), target.as_tensor()),
        )?))
    }

    fn sparse_metrics<'py>(
        &self,
        py: Python<'py>,
        prediction: &PyTensor,
        target: &PyTensor,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = convert(
            self.inner
                .sparse_metrics(prediction.as_tensor(), target.as_tensor()),
        )?;
        sparse_metrics_to_pydict(py, metrics)
    }

    fn __repr__(&self) -> PyResult<String> {
        let ignore_index = match self.inner.ignore_index() {
            Some(ignore_index) => format!("Some({ignore_index})"),
            None => "None".to_string(),
        };
        Ok(format!(
            "SoftmaxCrossEntropy(ignore_index={}, label_smoothing={:.6})",
            ignore_index,
            self.inner.label_smoothing()
        ))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "Linear")]
struct PyLinearModule {
    inner: Option<NnLinear>,
}

impl PyLinearModule {
    fn borrow(&self) -> PyResult<&NnLinear> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Linear module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnLinear> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Linear module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnLinear> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("Linear module has been moved"))
    }
}

#[pymethods]
impl PyLinearModule {
    #[new]
    #[pyo3(signature = (input_dim, output_dim, name=None))]
    fn new(input_dim: usize, output_dim: usize, name: Option<&str>) -> PyResult<Self> {
        let ident = name.unwrap_or("linear");
        let inner = convert(NnLinear::new(ident, input_dim, output_dim))?;
        Ok(Self { inner: Some(inner) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let layer = self.borrow()?;
        Ok(PyTensor::from_tensor(convert(
            layer.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let layer = self.borrow_mut()?;
        Ok(PyTensor::from_tensor(convert(
            layer.backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn set_trainable(&mut self, trainable: bool) -> PyResult<()> {
        set_module_trainable(self.borrow_mut()?, trainable)
    }

    fn set_parameter_trainable(&mut self, name: &str, trainable: bool) -> PyResult<()> {
        set_module_parameter_trainable(self.borrow_mut()?, name, trainable)
    }

    fn set_parameters_trainable_by_prefix(
        &mut self,
        prefix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_prefix(self.borrow_mut()?, prefix, trainable)
    }

    fn set_parameters_trainable_by_suffix(
        &mut self,
        suffix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_suffix(self.borrow_mut()?, suffix, trainable)
    }

    fn set_parameters_trainable_by_contains(
        &mut self,
        needle: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_contains(self.borrow_mut()?, needle, trainable)
    }

    fn scale_parameter_learning_rate(&mut self, name: &str, factor: f32) -> PyResult<()> {
        scale_module_parameter_learning_rate(self.borrow_mut()?, name, factor)
    }

    fn scale_parameters_learning_rate_by_prefix(
        &mut self,
        prefix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_prefix(self.borrow_mut()?, prefix, factor)
    }

    fn scale_parameters_learning_rate_by_suffix(
        &mut self,
        suffix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_suffix(self.borrow_mut()?, suffix, factor)
    }

    fn scale_parameters_learning_rate_by_contains(
        &mut self,
        needle: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_contains(self.borrow_mut()?, needle, factor)
    }

    fn set_parameter_learning_rate_scale(&mut self, name: &str, scale: f32) -> PyResult<()> {
        set_module_parameter_learning_rate_scale(self.borrow_mut()?, name, scale)
    }

    fn set_parameters_learning_rate_scale_by_prefix(
        &mut self,
        prefix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_prefix(self.borrow_mut()?, prefix, scale)
    }

    fn set_parameters_learning_rate_scale_by_suffix(
        &mut self,
        suffix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_suffix(self.borrow_mut()?, suffix, scale)
    }

    fn set_parameters_learning_rate_scale_by_contains(
        &mut self,
        needle: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_contains(self.borrow_mut()?, needle, scale)
    }

    fn set_parameter_weight_decay(&mut self, name: &str, weight_decay: f32) -> PyResult<()> {
        set_module_parameter_weight_decay(self.borrow_mut()?, name, weight_decay)
    }

    fn set_parameters_weight_decay_by_prefix(
        &mut self,
        prefix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_prefix(self.borrow_mut()?, prefix, weight_decay)
    }

    fn set_parameters_weight_decay_by_suffix(
        &mut self,
        suffix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_suffix(self.borrow_mut()?, suffix, weight_decay)
    }

    fn set_parameters_weight_decay_by_contains(
        &mut self,
        needle: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_contains(self.borrow_mut()?, needle, weight_decay)
    }

    #[pyo3(signature = (before, tolerance=0.0))]
    fn parameter_movement(
        &self,
        py: Python<'_>,
        before: &Bound<'_, PyDict>,
        tolerance: f32,
    ) -> PyResult<PyObject> {
        parameter_movement_py(py, self.borrow()?, before, tolerance)
    }

    fn state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_fingerprint_py(py, self.borrow()?)
    }

    fn training_state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        training_state_fingerprint_py(py, self.borrow()?)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }

    fn load_state_dict_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_mapped_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_mapped_py(py, self.borrow_mut()?, dict, key_map)
    }

    fn state_dict_compatibility(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_py(py, self.borrow()?, dict)
    }

    fn state_dict_compatibility_with_key_map(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_mapped_py(py, self.borrow()?, dict, key_map)
    }
}

#[pyclass(module = "spiraltorch.nn", name = "LoraLinear")]
struct PyLoraLinearModule {
    inner: Option<NnLoraLinear>,
}

impl PyLoraLinearModule {
    fn borrow(&self) -> PyResult<&NnLoraLinear> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("LoraLinear module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnLoraLinear> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("LoraLinear module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnLoraLinear> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("LoraLinear module has been moved"))
    }
}

#[pymethods]
impl PyLoraLinearModule {
    #[new]
    #[pyo3(signature = (input_dim, output_dim, rank, alpha=None, name=None))]
    fn new(
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        alpha: Option<f32>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let ident = name.unwrap_or("lora_linear");
        let alpha = alpha.unwrap_or(rank as f32);
        let inner = convert(NnLoraLinear::new(ident, input_dim, output_dim, rank, alpha))?;
        Ok(Self { inner: Some(inner) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let layer = self.borrow()?;
        Ok(PyTensor::from_tensor(convert(
            layer.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        let layer = self.borrow_mut()?;
        Ok(PyTensor::from_tensor(convert(
            layer.backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn rank(&self) -> PyResult<usize> {
        Ok(self.borrow()?.rank())
    }

    fn alpha(&self) -> PyResult<f32> {
        Ok(self.borrow()?.alpha())
    }

    fn adapter_scale(&self) -> PyResult<f32> {
        Ok(self.borrow()?.adapter_scale())
    }

    fn set_base_trainable(&mut self, trainable: bool) -> PyResult<()> {
        self.borrow_mut()?.set_base_trainable(trainable);
        Ok(())
    }

    fn set_adapter_trainable(&mut self, trainable: bool) -> PyResult<()> {
        self.borrow_mut()?.set_adapter_trainable(trainable);
        Ok(())
    }

    fn set_trainable(&mut self, trainable: bool) -> PyResult<()> {
        set_module_trainable(self.borrow_mut()?, trainable)
    }

    fn set_parameter_trainable(&mut self, name: &str, trainable: bool) -> PyResult<()> {
        set_module_parameter_trainable(self.borrow_mut()?, name, trainable)
    }

    fn set_parameters_trainable_by_prefix(
        &mut self,
        prefix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_prefix(self.borrow_mut()?, prefix, trainable)
    }

    fn set_parameters_trainable_by_suffix(
        &mut self,
        suffix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_suffix(self.borrow_mut()?, suffix, trainable)
    }

    fn set_parameters_trainable_by_contains(
        &mut self,
        needle: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_contains(self.borrow_mut()?, needle, trainable)
    }

    fn scale_parameter_learning_rate(&mut self, name: &str, factor: f32) -> PyResult<()> {
        scale_module_parameter_learning_rate(self.borrow_mut()?, name, factor)
    }

    fn scale_parameters_learning_rate_by_prefix(
        &mut self,
        prefix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_prefix(self.borrow_mut()?, prefix, factor)
    }

    fn scale_parameters_learning_rate_by_suffix(
        &mut self,
        suffix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_suffix(self.borrow_mut()?, suffix, factor)
    }

    fn scale_parameters_learning_rate_by_contains(
        &mut self,
        needle: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_contains(self.borrow_mut()?, needle, factor)
    }

    fn set_parameter_learning_rate_scale(&mut self, name: &str, scale: f32) -> PyResult<()> {
        set_module_parameter_learning_rate_scale(self.borrow_mut()?, name, scale)
    }

    fn set_parameters_learning_rate_scale_by_prefix(
        &mut self,
        prefix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_prefix(self.borrow_mut()?, prefix, scale)
    }

    fn set_parameters_learning_rate_scale_by_suffix(
        &mut self,
        suffix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_suffix(self.borrow_mut()?, suffix, scale)
    }

    fn set_parameters_learning_rate_scale_by_contains(
        &mut self,
        needle: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_contains(self.borrow_mut()?, needle, scale)
    }

    fn set_parameter_weight_decay(&mut self, name: &str, weight_decay: f32) -> PyResult<()> {
        set_module_parameter_weight_decay(self.borrow_mut()?, name, weight_decay)
    }

    fn set_parameters_weight_decay_by_prefix(
        &mut self,
        prefix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_prefix(self.borrow_mut()?, prefix, weight_decay)
    }

    fn set_parameters_weight_decay_by_suffix(
        &mut self,
        suffix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_suffix(self.borrow_mut()?, suffix, weight_decay)
    }

    fn set_parameters_weight_decay_by_contains(
        &mut self,
        needle: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_contains(self.borrow_mut()?, needle, weight_decay)
    }

    #[pyo3(signature = (before, tolerance=0.0))]
    fn parameter_movement(
        &self,
        py: Python<'_>,
        before: &Bound<'_, PyDict>,
        tolerance: f32,
    ) -> PyResult<PyObject> {
        parameter_movement_py(py, self.borrow()?, before, tolerance)
    }

    fn state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_fingerprint_py(py, self.borrow()?)
    }

    fn training_state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        training_state_fingerprint_py(py, self.borrow()?)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn base_state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_to_pydict(py, self.borrow()?.base_state_dict())
    }

    fn base_state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        let fingerprint = self.borrow()?.base_state_fingerprint();
        Ok(fingerprint_to_pydict(py, &fingerprint)?.into_py(py))
    }

    fn base_state_dict_compatibility(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let state = pydict_to_state(dict)?;
        let report = self.borrow()?.base_state_dict_compatibility(&state);
        Ok(compatibility_report_to_pydict(py, &report)?.into_py(py))
    }

    fn base_state_dict_compatibility_with_key_map(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let state = pydict_to_state(dict)?;
        let key_rules = pydict_to_key_rules(key_map)?;
        let report = convert(
            self.borrow()?
                .base_state_dict_compatibility_with_key_rules(&state, &key_rules),
        )?;
        Ok(compatibility_report_to_pydict(py, &report)?.into_py(py))
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }

    fn load_state_dict_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_mapped_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_mapped_py(py, self.borrow_mut()?, dict, key_map)
    }

    fn state_dict_compatibility(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_py(py, self.borrow()?, dict)
    }

    fn state_dict_compatibility_with_key_map(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_mapped_py(py, self.borrow()?, dict, key_map)
    }

    fn load_base_from_state_dict(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let state = pydict_to_state(dict)?;
        let report = convert(self.borrow_mut()?.load_base_from_state_dict(&state))?;
        Ok(load_report_to_pydict(py, &report)?.into_py(py))
    }

    fn load_base_from_state_dict_mapped(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let state = pydict_to_state(dict)?;
        let key_rules = pydict_to_key_rules(key_map)?;
        let report = convert(
            self.borrow_mut()?
                .load_base_from_state_dict_adapted(&state, &key_rules),
        )?;
        Ok(load_report_to_pydict(py, &report)?.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        let layer = self.borrow()?;
        Ok(format!(
            "LoraLinear(rank={}, alpha={:.6}, adapter_scale={:.6})",
            layer.rank(),
            layer.alpha(),
            layer.adapter_scale()
        ))
    }
}

#[pyclass(module = "spiraltorch.nn", name = "Relu")]
struct PyReluModule {
    inner: Option<NnRelu>,
}

impl PyReluModule {
    fn borrow(&self) -> PyResult<&NnRelu> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Relu module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnRelu> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Relu module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnRelu> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("Relu module has been moved"))
    }
}

#[pymethods]
impl PyReluModule {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(NnRelu::new()),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow_mut()?
                .backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_fingerprint_py(py, self.borrow()?)
    }

    fn training_state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        training_state_fingerprint_py(py, self.borrow()?)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn __repr__(&self) -> PyResult<String> {
        self.borrow()?;
        Ok("Relu()".to_string())
    }
}

#[pyclass(module = "spiraltorch.nn", name = "Conv1d")]
struct PyConv1dModule {
    inner: Option<NnConv1d>,
}

impl PyConv1dModule {
    fn borrow(&self) -> PyResult<&NnConv1d> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Conv1d module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnConv1d> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Conv1d module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnConv1d> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("Conv1d module has been moved"))
    }
}

#[pymethods]
impl PyConv1dModule {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, name=None))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let ident = name.unwrap_or("conv1d");
        let inner = convert(NnConv1d::new(
            ident,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        ))?;
        Ok(Self { inner: Some(inner) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow_mut()?
                .backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn set_trainable(&mut self, trainable: bool) -> PyResult<()> {
        set_module_trainable(self.borrow_mut()?, trainable)
    }

    fn set_parameter_trainable(&mut self, name: &str, trainable: bool) -> PyResult<()> {
        set_module_parameter_trainable(self.borrow_mut()?, name, trainable)
    }

    fn set_parameters_trainable_by_prefix(
        &mut self,
        prefix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_prefix(self.borrow_mut()?, prefix, trainable)
    }

    fn set_parameters_trainable_by_suffix(
        &mut self,
        suffix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_suffix(self.borrow_mut()?, suffix, trainable)
    }

    fn set_parameters_trainable_by_contains(
        &mut self,
        needle: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_contains(self.borrow_mut()?, needle, trainable)
    }

    fn scale_parameter_learning_rate(&mut self, name: &str, factor: f32) -> PyResult<()> {
        scale_module_parameter_learning_rate(self.borrow_mut()?, name, factor)
    }

    fn scale_parameters_learning_rate_by_prefix(
        &mut self,
        prefix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_prefix(self.borrow_mut()?, prefix, factor)
    }

    fn scale_parameters_learning_rate_by_suffix(
        &mut self,
        suffix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_suffix(self.borrow_mut()?, suffix, factor)
    }

    fn scale_parameters_learning_rate_by_contains(
        &mut self,
        needle: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_contains(self.borrow_mut()?, needle, factor)
    }

    fn set_parameter_learning_rate_scale(&mut self, name: &str, scale: f32) -> PyResult<()> {
        set_module_parameter_learning_rate_scale(self.borrow_mut()?, name, scale)
    }

    fn set_parameters_learning_rate_scale_by_prefix(
        &mut self,
        prefix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_prefix(self.borrow_mut()?, prefix, scale)
    }

    fn set_parameters_learning_rate_scale_by_suffix(
        &mut self,
        suffix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_suffix(self.borrow_mut()?, suffix, scale)
    }

    fn set_parameters_learning_rate_scale_by_contains(
        &mut self,
        needle: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_contains(self.borrow_mut()?, needle, scale)
    }

    fn set_parameter_weight_decay(&mut self, name: &str, weight_decay: f32) -> PyResult<()> {
        set_module_parameter_weight_decay(self.borrow_mut()?, name, weight_decay)
    }

    fn set_parameters_weight_decay_by_prefix(
        &mut self,
        prefix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_prefix(self.borrow_mut()?, prefix, weight_decay)
    }

    fn set_parameters_weight_decay_by_suffix(
        &mut self,
        suffix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_suffix(self.borrow_mut()?, suffix, weight_decay)
    }

    fn set_parameters_weight_decay_by_contains(
        &mut self,
        needle: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_contains(self.borrow_mut()?, needle, weight_decay)
    }

    #[pyo3(signature = (before, tolerance=0.0))]
    fn parameter_movement(
        &self,
        py: Python<'_>,
        before: &Bound<'_, PyDict>,
        tolerance: f32,
    ) -> PyResult<PyObject> {
        parameter_movement_py(py, self.borrow()?, before, tolerance)
    }

    fn state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_fingerprint_py(py, self.borrow()?)
    }

    fn training_state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        training_state_fingerprint_py(py, self.borrow()?)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }

    fn load_state_dict_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_mapped_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_mapped_py(py, self.borrow_mut()?, dict, key_map)
    }

    fn state_dict_compatibility(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_py(py, self.borrow()?, dict)
    }

    fn state_dict_compatibility_with_key_map(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_mapped_py(py, self.borrow()?, dict, key_map)
    }
}

#[pyclass(module = "spiraltorch.nn", name = "WaveRnn")]
struct PyWaveRnnModule {
    inner: Option<NnWaveRnn>,
}

impl PyWaveRnnModule {
    fn borrow(&self) -> PyResult<&NnWaveRnn> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("WaveRnn module has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnWaveRnn> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("WaveRnn module has been moved"))
    }

    fn take(&mut self) -> PyResult<NnWaveRnn> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("WaveRnn module has been moved"))
    }
}

#[pymethods]
impl PyWaveRnnModule {
    #[new]
    #[pyo3(signature = (in_channels, hidden_dim, kernel_size, stride=1, padding=0, curvature=-1.0, temperature=0.5, name=None))]
    fn new(
        in_channels: usize,
        hidden_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        curvature: f32,
        temperature: f32,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let ident = name.unwrap_or("wave_rnn");
        let inner = convert(NnWaveRnn::new(
            ident,
            in_channels,
            hidden_dim,
            kernel_size,
            stride,
            padding,
            curvature,
            temperature,
        ))?;
        Ok(Self { inner: Some(inner) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow_mut()?
                .backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn set_trainable(&mut self, trainable: bool) -> PyResult<()> {
        set_module_trainable(self.borrow_mut()?, trainable)
    }

    fn set_parameter_trainable(&mut self, name: &str, trainable: bool) -> PyResult<()> {
        set_module_parameter_trainable(self.borrow_mut()?, name, trainable)
    }

    fn set_parameters_trainable_by_prefix(
        &mut self,
        prefix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_prefix(self.borrow_mut()?, prefix, trainable)
    }

    fn set_parameters_trainable_by_suffix(
        &mut self,
        suffix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_suffix(self.borrow_mut()?, suffix, trainable)
    }

    fn set_parameters_trainable_by_contains(
        &mut self,
        needle: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_contains(self.borrow_mut()?, needle, trainable)
    }

    fn scale_parameter_learning_rate(&mut self, name: &str, factor: f32) -> PyResult<()> {
        scale_module_parameter_learning_rate(self.borrow_mut()?, name, factor)
    }

    fn scale_parameters_learning_rate_by_prefix(
        &mut self,
        prefix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_prefix(self.borrow_mut()?, prefix, factor)
    }

    fn scale_parameters_learning_rate_by_suffix(
        &mut self,
        suffix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_suffix(self.borrow_mut()?, suffix, factor)
    }

    fn scale_parameters_learning_rate_by_contains(
        &mut self,
        needle: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_contains(self.borrow_mut()?, needle, factor)
    }

    fn set_parameter_learning_rate_scale(&mut self, name: &str, scale: f32) -> PyResult<()> {
        set_module_parameter_learning_rate_scale(self.borrow_mut()?, name, scale)
    }

    fn set_parameters_learning_rate_scale_by_prefix(
        &mut self,
        prefix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_prefix(self.borrow_mut()?, prefix, scale)
    }

    fn set_parameters_learning_rate_scale_by_suffix(
        &mut self,
        suffix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_suffix(self.borrow_mut()?, suffix, scale)
    }

    fn set_parameters_learning_rate_scale_by_contains(
        &mut self,
        needle: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_contains(self.borrow_mut()?, needle, scale)
    }

    fn set_parameter_weight_decay(&mut self, name: &str, weight_decay: f32) -> PyResult<()> {
        set_module_parameter_weight_decay(self.borrow_mut()?, name, weight_decay)
    }

    fn set_parameters_weight_decay_by_prefix(
        &mut self,
        prefix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_prefix(self.borrow_mut()?, prefix, weight_decay)
    }

    fn set_parameters_weight_decay_by_suffix(
        &mut self,
        suffix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_suffix(self.borrow_mut()?, suffix, weight_decay)
    }

    fn set_parameters_weight_decay_by_contains(
        &mut self,
        needle: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_contains(self.borrow_mut()?, needle, weight_decay)
    }

    #[pyo3(signature = (before, tolerance=0.0))]
    fn parameter_movement(
        &self,
        py: Python<'_>,
        before: &Bound<'_, PyDict>,
        tolerance: f32,
    ) -> PyResult<PyObject> {
        parameter_movement_py(py, self.borrow()?, before, tolerance)
    }

    fn state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_fingerprint_py(py, self.borrow()?)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }

    fn load_state_dict_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_mapped_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_mapped_py(py, self.borrow_mut()?, dict, key_map)
    }

    fn state_dict_compatibility(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_py(py, self.borrow()?, dict)
    }

    fn state_dict_compatibility_with_key_map(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_mapped_py(py, self.borrow()?, dict, key_map)
    }
}

#[pyclass(module = "spiraltorch.nn", name = "Sequential", unsendable)]
struct PySequentialModule {
    inner: Option<NnSequential>,
}

impl PySequentialModule {
    fn borrow(&self) -> PyResult<&NnSequential> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Sequential has been moved"))
    }

    fn borrow_mut(&mut self) -> PyResult<&mut NnSequential> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Sequential has been moved"))
    }
}

#[pymethods]
impl PySequentialModule {
    #[new]
    fn new(py_layers: &Bound<'_, PyAny>) -> PyResult<Self> {
        let seq_iter = py_layers.iter()?;
        let mut seq = NnSequential::new();
        for item in seq_iter {
            let obj = item?;
            if let Ok(mut linear) = obj.extract::<PyRefMut<'_, PyLinearModule>>() {
                seq.push(linear.take()?);
            } else if let Ok(mut lora) = obj.extract::<PyRefMut<'_, PyLoraLinearModule>>() {
                seq.push(lora.take()?);
            } else if let Ok(mut relu) = obj.extract::<PyRefMut<'_, PyReluModule>>() {
                seq.push(relu.take()?);
            } else if let Ok(mut conv) = obj.extract::<PyRefMut<'_, PyConv1dModule>>() {
                seq.push(conv.take()?);
            } else if let Ok(mut wave) = obj.extract::<PyRefMut<'_, PyWaveRnnModule>>() {
                seq.push(wave.take()?);
            } else if let Ok(mut projector) = obj.extract::<PyRefMut<'_, PyZSpaceProjector>>() {
                seq.push(projector.take()?);
            } else {
                return Err(PyValueError::new_err(
                    "Sequential expects Linear, LoraLinear, Relu, Conv1d, WaveRnn, or ZSpaceProjector modules",
                ));
            }
        }
        Ok(Self { inner: Some(seq) })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow()?.forward(input.as_tensor()),
        )?))
    }

    fn backward(&mut self, input: &PyTensor, grad_output: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor::from_tensor(convert(
            self.borrow_mut()?
                .backward(input.as_tensor(), grad_output.as_tensor()),
        )?))
    }

    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PyResult<()> {
        convert(
            self.borrow_mut()?
                .attach_hypergrad(curvature, learning_rate),
        )
    }

    fn apply_step(&mut self, fallback_lr: f32) -> PyResult<()> {
        convert(self.borrow_mut()?.apply_step(fallback_lr))
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        convert(self.borrow_mut()?.zero_accumulators())
    }

    fn set_trainable(&mut self, trainable: bool) -> PyResult<()> {
        set_module_trainable(self.borrow_mut()?, trainable)
    }

    fn set_parameter_trainable(&mut self, name: &str, trainable: bool) -> PyResult<()> {
        set_module_parameter_trainable(self.borrow_mut()?, name, trainable)
    }

    fn set_parameters_trainable_by_prefix(
        &mut self,
        prefix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_prefix(self.borrow_mut()?, prefix, trainable)
    }

    fn set_parameters_trainable_by_suffix(
        &mut self,
        suffix: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_suffix(self.borrow_mut()?, suffix, trainable)
    }

    fn set_parameters_trainable_by_contains(
        &mut self,
        needle: &str,
        trainable: bool,
    ) -> PyResult<usize> {
        set_module_parameters_trainable_by_contains(self.borrow_mut()?, needle, trainable)
    }

    fn scale_parameter_learning_rate(&mut self, name: &str, factor: f32) -> PyResult<()> {
        scale_module_parameter_learning_rate(self.borrow_mut()?, name, factor)
    }

    fn scale_parameters_learning_rate_by_prefix(
        &mut self,
        prefix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_prefix(self.borrow_mut()?, prefix, factor)
    }

    fn scale_parameters_learning_rate_by_suffix(
        &mut self,
        suffix: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_suffix(self.borrow_mut()?, suffix, factor)
    }

    fn scale_parameters_learning_rate_by_contains(
        &mut self,
        needle: &str,
        factor: f32,
    ) -> PyResult<usize> {
        scale_module_parameters_learning_rate_by_contains(self.borrow_mut()?, needle, factor)
    }

    fn set_parameter_learning_rate_scale(&mut self, name: &str, scale: f32) -> PyResult<()> {
        set_module_parameter_learning_rate_scale(self.borrow_mut()?, name, scale)
    }

    fn set_parameters_learning_rate_scale_by_prefix(
        &mut self,
        prefix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_prefix(self.borrow_mut()?, prefix, scale)
    }

    fn set_parameters_learning_rate_scale_by_suffix(
        &mut self,
        suffix: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_suffix(self.borrow_mut()?, suffix, scale)
    }

    fn set_parameters_learning_rate_scale_by_contains(
        &mut self,
        needle: &str,
        scale: f32,
    ) -> PyResult<usize> {
        set_module_parameters_learning_rate_scale_by_contains(self.borrow_mut()?, needle, scale)
    }

    fn set_parameter_weight_decay(&mut self, name: &str, weight_decay: f32) -> PyResult<()> {
        set_module_parameter_weight_decay(self.borrow_mut()?, name, weight_decay)
    }

    fn set_parameters_weight_decay_by_prefix(
        &mut self,
        prefix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_prefix(self.borrow_mut()?, prefix, weight_decay)
    }

    fn set_parameters_weight_decay_by_suffix(
        &mut self,
        suffix: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_suffix(self.borrow_mut()?, suffix, weight_decay)
    }

    fn set_parameters_weight_decay_by_contains(
        &mut self,
        needle: &str,
        weight_decay: f32,
    ) -> PyResult<usize> {
        set_module_parameters_weight_decay_by_contains(self.borrow_mut()?, needle, weight_decay)
    }

    #[pyo3(signature = (before, tolerance=0.0))]
    fn parameter_movement(
        &self,
        py: Python<'_>,
        before: &Bound<'_, PyDict>,
        tolerance: f32,
    ) -> PyResult<PyObject> {
        parameter_movement_py(py, self.borrow()?, before, tolerance)
    }

    fn state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_fingerprint_py(py, self.borrow()?)
    }

    fn training_state_fingerprint(&self, py: Python<'_>) -> PyResult<PyObject> {
        training_state_fingerprint_py(py, self.borrow()?)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = convert(self.borrow()?.state_dict())?;
        state_to_pydict(py, state)
    }

    fn load_state_dict(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let state = pydict_to_state(dict)?;
        convert(self.borrow_mut()?.load_state_dict(&state))
    }

    fn load_state_dict_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_py(py, self.borrow_mut()?, dict)
    }

    fn load_state_dict_subset_mapped_checked(
        &mut self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        checked_load_state_subset_mapped_py(py, self.borrow_mut()?, dict, key_map)
    }

    fn state_dict_compatibility(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_py(py, self.borrow()?, dict)
    }

    fn state_dict_compatibility_with_key_map(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key_map: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        state_dict_compatibility_mapped_py(py, self.borrow()?, dict, key_map)
    }
}

fn parse_kind(kind: &str) -> PyResult<RankKind> {
    match kind.to_ascii_lowercase().as_str() {
        "topk" | "top" => Ok(RankKind::TopK),
        "midk" | "mid" => Ok(RankKind::MidK),
        "bottomk" | "bottom" => Ok(RankKind::BottomK),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported rank kind: {}",
            other
        ))),
    }
}

fn caps_for(device: Option<&str>) -> DeviceCaps {
    match device.map(|d| d.to_ascii_lowercase()) {
        Some(ref name) if name == "cuda" => DeviceCaps::cuda(32, 1024, Some(96 * 1024)),
        Some(ref name) if name == "hip" => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
        Some(ref name) if name == "cpu" => DeviceCaps::cpu(),
        Some(ref name) if name == "mps" => DeviceCaps::mps(32, 256, Some(32 * 1024)),
        Some(ref name) if name == "wgpu" => DeviceCaps::wgpu(32, true, 256),
        Some(ref name) if name == "auto" => DeviceCaps::wgpu(32, true, 256),
        Some(ref name) if name == "hip-real" => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
        _ => DeviceCaps::wgpu(32, true, 256),
    }
}

fn choice_dict<'py>(py: Python<'py>, plan: &RankPlan) -> PyResult<Bound<'py, PyDict>> {
    let choice = PyDict::new_bound(py);
    choice.set_item("workgroup", plan.choice.wg)?;
    choice.set_item("kl", plan.choice.kl)?;
    choice.set_item("channel_stride", plan.choice.ch)?;
    choice.set_item("merge_kind", plan.choice.mk)?;
    choice.set_item("merge_detail", plan.choice.mkd)?;
    choice.set_item("tile", plan.choice.tile)?;
    Ok(choice)
}

fn compaction_choice_dict<'py>(
    py: Python<'py>,
    plan: &CompactionPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let choice = PyDict::new_bound(py);
    choice.set_item("use_2ce", plan.choice.use_2ce)?;
    choice.set_item("compaction_tile", plan.choice.ctile)?;
    Ok(choice)
}

/// Inspect the unified heuristics for the requested rank family.
#[pyfunction]
#[pyo3(signature = (kind, rows, cols, k, device=None))]
fn plan(
    py: Python<'_>,
    kind: &str,
    rows: u32,
    cols: u32,
    k: u32,
    device: Option<&str>,
) -> PyResult<PyObject> {
    let rank_kind = parse_kind(kind)?;
    let caps = caps_for(device);
    let plan = plan_rank(rank_kind, rows, cols, k, caps);

    let out = PyDict::new_bound(py);
    out.set_item("kind", kind.to_ascii_lowercase())?;
    out.set_item("rows", rows)?;
    out.set_item("cols", cols)?;
    out.set_item("k", k)?;
    out.set_item("choice", choice_dict(py, &plan)?.into_py(py))?;
    Ok(out.into_py(py))
}

/// Inspect the threshold compaction heuristics for the requested backend.
#[pyfunction(name = "plan_compaction")]
#[pyo3(signature = (rows, cols, device=None))]
fn plan_compaction_py(
    py: Python<'_>,
    rows: u32,
    cols: u32,
    device: Option<&str>,
) -> PyResult<PyObject> {
    let caps = caps_for(device);
    let plan = plan_compaction(rows, cols, caps);

    let out = PyDict::new_bound(py);
    out.set_item("rows", rows)?;
    out.set_item("cols", cols)?;
    out.set_item("choice", compaction_choice_dict(py, &plan)?.into_py(py))?;
    Ok(out.into_py(py))
}

/// Compute the Z-space barycenter described by the weighted KL objective.
#[pyfunction(name = "z_space_barycenter")]
#[pyo3(signature = (densities, weights=None, entropy_weight=0.0, beta_j=0.0, coupling=None))]
fn z_space_barycenter_py(
    py: Python<'_>,
    densities: Vec<PyTensor>,
    weights: Option<Vec<f32>>,
    entropy_weight: f32,
    beta_j: f32,
    coupling: Option<PyTensor>,
) -> PyResult<PyObject> {
    if densities.is_empty() {
        return Err(PyValueError::new_err("densities must not be empty"));
    }
    let tensors: Vec<Tensor> = densities.into_iter().map(PyTensor::into_tensor).collect();
    let weight_vec = weights.unwrap_or_else(|| vec![1.0; tensors.len()]);
    if weight_vec.len() != tensors.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} weights, received {}",
            tensors.len(),
            weight_vec.len()
        )));
    }
    let coupling_tensor = coupling.map(PyTensor::into_tensor);
    let coupling_ref = coupling_tensor.as_ref();
    let barycenter = convert(rust_z_space_barycenter(
        &weight_vec,
        &tensors,
        entropy_weight,
        beta_j,
        coupling_ref,
    ))?;
    PyZSpaceBarycenter::from_result(barycenter).as_dict(py)
}

fn parse_frac_pad(pad: &str) -> PyResult<FracPad> {
    match pad.to_ascii_lowercase().as_str() {
        "zero" => Ok(FracPad::Zero),
        "reflect" => Ok(FracPad::Reflect),
        other => Err(PyValueError::new_err(format!(
            "unsupported padding kind: {other}; expected 'zero' or 'reflect'"
        ))),
    }
}

#[pyfunction(name = "gl_coeffs")]
fn gl_coeffs_py(alpha: f32, len: usize) -> Vec<f32> {
    frac_gl_coeffs(alpha, len)
}

#[pyfunction(name = "fracdiff_gl")]
#[pyo3(signature = (tensor, alpha, axis, kernel_len, pad="zero", scale=None))]
fn fracdiff_gl_py(
    tensor: &PyTensor,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: &str,
    scale: Option<f32>,
) -> PyResult<PyTensor> {
    let array = tensor_to_array(tensor.as_tensor())?;
    let pad = parse_frac_pad(pad)?;
    let result = convert_frac(fracdiff_gl_nd(&array, alpha, axis, kernel_len, pad, scale))?;
    array_to_tensor(result)
}

#[pyfunction(name = "fracdiff_gl_backward")]
#[pyo3(signature = (tensor, alpha, axis, kernel_len, pad="zero", scale=None))]
fn fracdiff_gl_backward_py(
    tensor: &PyTensor,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: &str,
    scale: Option<f32>,
) -> PyResult<PyTensor> {
    let array = tensor_to_array(tensor.as_tensor())?;
    let pad = parse_frac_pad(pad)?;
    let result = convert_frac(fracdiff_gl_nd_backward(
        &array, alpha, axis, kernel_len, pad, scale,
    ))?;
    array_to_tensor(result)
}

#[pyfunction(name = "fft")]
#[pyo3(signature = (signal, inverse=false))]
fn frac_fft_py(signal: Vec<Complex64>, inverse: bool) -> PyResult<Vec<Complex64>> {
    let mut buffer: Vec<FracComplex32> = signal
        .into_iter()
        .map(|c| FracComplex32::new(c.re as f32, c.im as f32))
        .collect();
    convert_fft(frac_fft_inplace(&mut buffer, inverse))?;
    Ok(buffer
        .into_iter()
        .map(|c| Complex64::new(c.re as f64, c.im as f64))
        .collect())
}

#[pymodule]
fn nn(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sparse_classification_delta_py, m)?)?;
    m.add_function(wrap_pyfunction!(compare_sparse_finetune_summaries_py, m)?)?;
    m.add_class::<PyMeanSquaredError>()?;
    m.add_class::<PySoftmaxCrossEntropy>()?;
    m.add_class::<PyLinearModule>()?;
    m.add_class::<PyLoraLinearModule>()?;
    m.add_class::<PyReluModule>()?;
    m.add_class::<PyConv1dModule>()?;
    m.add_class::<PyWaveRnnModule>()?;
    m.add_class::<PyZSpaceProjector>()?;
    m.add_class::<PySequentialModule>()?;
    m.setattr(
        "__all__",
        vec![
            "MeanSquaredError",
            "SoftmaxCrossEntropy",
            "sparse_classification_delta",
            "compare_sparse_finetune_summaries",
            "Linear",
            "LoraLinear",
            "Relu",
            "Conv1d",
            "WaveRnn",
            "ZSpaceProjector",
            "Sequential",
        ],
    )?;
    m.setattr(
        "__doc__",
        "Rust-backed neural network modules and losses: Linear, LoraLinear, Relu, Conv1d, WaveRnn, ZSpaceProjector, Sequential, MeanSquaredError, SoftmaxCrossEntropy.",
    )?;
    Ok(())
}

#[pymodule]
fn frac(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gl_coeffs_py, m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_gl_py, m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_gl_backward_py, m)?)?;
    m.add_function(wrap_pyfunction!(frac_fft_py, m)?)?;
    m.setattr(
        "__all__",
        vec!["gl_coeffs", "fracdiff_gl", "fracdiff_gl_backward", "fft"],
    )?;
    m.setattr(
        "__doc__",
        "Fractional calculus operators and FFT helpers used by SpiralTorch.",
    )?;
    Ok(())
}

#[pymodule]
fn dataset(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dataset_from_vec_py, m)?)?;
    m.add_function(wrap_pyfunction!(byte_lm_windows_py, m)?)?;
    m.add_function(wrap_pyfunction!(byte_lm_corpus_windows_py, m)?)?;
    m.add_function(wrap_pyfunction!(padded_byte_lm_samples_py, m)?)?;
    m.add_function(wrap_pyfunction!(byte_lm_sample_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(interleave_replay_samples_py, m)?)?;
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyDataLoaderIter>()?;
    m.setattr("BYTE_LM_VOCAB", BYTE_LM_VOCAB)?;
    m.setattr(
        "__all__",
        vec![
            "from_vec",
            "byte_lm_windows",
            "byte_lm_corpus_windows",
            "padded_byte_lm_samples",
            "byte_lm_sample_stats",
            "interleave_replay_samples",
            "BYTE_LM_VOCAB",
            "DataLoader",
        ],
    )?;
    m.setattr(
        "__doc__",
        "Dataset helpers for SpiralTorch sessions: tokenizerless byte-LM samples, shuffle, batch, and prefetch in Rust.",
    )?;
    Ok(())
}

/// Convenience helper for the TopK family.
#[pyfunction]
#[pyo3(signature = (rows, cols, k, device=None))]
fn plan_topk(
    py: Python<'_>,
    rows: u32,
    cols: u32,
    k: u32,
    device: Option<&str>,
) -> PyResult<PyObject> {
    plan(py, "topk", rows, cols, k, device)
}

/// Execute TopK/MidK/BottomK selection on CPU for a dense tensor.
#[pyfunction]
#[pyo3(signature = (tensor, kind="topk", k=8))]
fn rank_select_cpu(py: Python<'_>, tensor: &PyTensor, kind: &str, k: u32) -> PyResult<PyObject> {
    if k == 0 {
        return Err(PyValueError::new_err("k must be positive"));
    }

    let rank_kind = parse_kind(kind)?;
    let (rows, cols) = tensor.as_tensor().shape();
    let rows_u32 =
        u32::try_from(rows).map_err(|_| PyValueError::new_err("rows must fit in u32"))?;
    let cols_u32 =
        u32::try_from(cols).map_err(|_| PyValueError::new_err("cols must fit in u32"))?;

    let plan = plan_rank(rank_kind, rows_u32, cols_u32, k, DeviceCaps::cpu());
    let selection =
        st_core::ops::rank_cpu::select_rank_cpu(&plan, tensor.as_tensor().data(), cols_u32)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let st_core::ops::rank_cpu::RankKSelection {
        values, indices, ..
    } = selection;

    let values_tensor = PyTensor::from_tensor(convert(Tensor::from_vec(rows, k as usize, values))?);

    let mut idx_rows: Vec<Vec<u32>> = Vec::with_capacity(rows);
    let k_usize = k as usize;
    for r in 0..rows {
        let base = r * k_usize;
        idx_rows.push(indices[base..base + k_usize].to_vec());
    }

    let out = PyDict::new_bound(py);
    out.set_item("kind", kind.to_ascii_lowercase())?;
    out.set_item("rows", rows_u32)?;
    out.set_item("cols", cols_u32)?;
    out.set_item("k", k)?;
    out.set_item("values", values_tensor.into_py(py))?;
    out.set_item("indices", idx_rows)?;
    Ok(out.into_py(py))
}

/// Surface ROCm probing hints for Python callers.
#[pyfunction]
fn hip_probe(py: Python<'_>) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("available", hip_runtime_available())?;

    let devices = PyList::empty_bound(py);
    for info in hip_device_info() {
        devices.append(py_device_info(py, info)?.into_py(py))?;
    }
    out.set_item("devices", devices.into_py(py))?;

    Ok(out.into_py(py))
}

#[pyfunction]
fn get_psychoid_stats(py: Python<'_>) -> PyResult<Option<PyObject>> {
    #[cfg(feature = "psychoid")]
    {
        if let Some(reading) = hub::get_last_psychoid() {
            let dict = PyDict::new_bound(py);
            dict.set_item("step", reading.step)?;
            dict.set_item("cti", reading.cti)?;
            let raw = PyDict::new_bound(py);
            for (key, value) in reading.raw.iter() {
                raw.set_item(*key, value)?;
            }
            let z = PyDict::new_bound(py);
            for (key, value) in reading.z_scores.iter() {
                z.set_item(*key, value)?;
            }
            dict.set_item("raw", raw)?;
            dict.set_item("z", z)?;
            return Ok(Some(dict.into_py(py)));
        }
        Ok(None)
    }
    #[cfg(not(feature = "psychoid"))]
    {
        let _ = py;
        Ok(None)
    }
}

/// Return a basic capability template for the given device string.
#[pyfunction]
#[pyo3(signature = (device=None))]
fn describe_device(py: Python<'_>, device: Option<&str>) -> PyResult<PyObject> {
    let caps = caps_for(device);
    Ok(device_caps_dict(py, caps)?.into_py(py))
}

/// SpiralTorch Python module.
#[pymodule]
fn spiraltorch(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys_modules = _py.import_bound("sys")?.getattr("modules")?;

    let nn_mod = PyModule::new_bound(_py, "spiraltorch.nn")?;
    nn(_py, &nn_mod)?;
    m.add_submodule(&nn_mod)?;
    m.setattr("nn", &nn_mod)?;
    sys_modules.set_item("spiraltorch.nn", &nn_mod)?;
    sys_modules.set_item("spiraltorch.spiraltorch.nn", &nn_mod)?;
    let frac_mod = PyModule::new_bound(_py, "spiraltorch.frac")?;
    frac(_py, &frac_mod)?;
    m.add_submodule(&frac_mod)?;
    m.setattr("frac", &frac_mod)?;
    sys_modules.set_item("spiraltorch.frac", &frac_mod)?;
    sys_modules.set_item("spiraltorch.spiraltorch.frac", &frac_mod)?;
    let dataset_mod = PyModule::new_bound(_py, "spiraltorch.dataset")?;
    dataset(_py, &dataset_mod)?;
    m.add_submodule(&dataset_mod)?;
    m.setattr("dataset", &dataset_mod)?;
    sys_modules.set_item("spiraltorch.dataset", &dataset_mod)?;
    sys_modules.set_item("spiraltorch.spiraltorch.dataset", &dataset_mod)?;
    let sot_mod = PyModule::new_bound(_py, "spiraltorch.sot")?;
    sot::module(_py, &sot_mod)?;
    m.add_submodule(&sot_mod)?;
    m.setattr("sot", &sot_mod)?;
    sys_modules.set_item("spiraltorch.sot", &sot_mod)?;
    sys_modules.set_item("spiraltorch.spiraltorch.sot", &sot_mod)?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(plan_compaction_py, m)?)?;
    m.add_function(wrap_pyfunction!(plan_topk, m)?)?;
    m.add_function(wrap_pyfunction!(rank_select_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(z_space_barycenter_py, m)?)?;
    m.add_function(wrap_pyfunction!(hip_probe, m)?)?;
    m.add_function(wrap_pyfunction!(describe_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_psychoid_stats, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_epoch_history_py, m)?)?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyComplexTensor>()?;
    m.add_class::<PyBarycenterIntermediate>()?;
    m.add_class::<PyZSpaceBarycenter>()?;
    m.add_class::<PyDifferentialResonance>()?;
    m.add_class::<PySpiralDifferentialTrace>()?;
    m.add_class::<PyOpenTopos>()?;
    m.add_class::<PyTensorBiome>()?;
    m.add_class::<PyLanguageWaveEncoder>()?;
    m.add_class::<PyHypergrad>()?;
    m.add_class::<PyDistConfig>()?;
    m.add_class::<PyRoundtableSchedule>()?;
    m.add_class::<PyEpochStats>()?;
    m.add_class::<PyEpochHistory>()?;
    m.add_class::<PyEpochBestState>()?;
    m.add_class::<PyEpochValidationBestState>()?;
    m.add_class::<PyEpochSparseRetentionBestState>()?;
    m.add_class::<PySparseFineTuneReport>()?;
    m.add_class::<PyModuleTrainer>()?;
    m.add_class::<PySpiralSessionBuilder>()?;
    m.add_class::<PySpiralSession>()?;

    m.setattr(
        "__all__",
        vec![
            "plan",
            "plan_topk",
            "rank_select_cpu",
            "z_space_barycenter",
            "hip_probe",
            "describe_device",
            "get_psychoid_stats",
            "summarize_epoch_history",
            "Tensor",
            "ComplexTensor",
            "BarycenterIntermediate",
            "ZSpaceBarycenter",
            "DifferentialResonance",
            "SpiralDifferentialTrace",
            "OpenTopos",
            "TensorBiome",
            "LanguageWaveEncoder",
            "Hypergrad",
            "DistConfig",
            "RoundtableSchedule",
            "EpochStats",
            "EpochHistory",
            "EpochBestState",
            "EpochValidationBestState",
            "EpochSparseRetentionBestState",
            "SparseFineTuneReport",
            "ModuleTrainer",
            "SpiralSessionBuilder",
            "SpiralSession",
            "nn",
            "frac",
            "dataset",
            "sot",
        ],
    )?;
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;

    // Provide a tiny doc string that highlights the zero-shim approach.
    m.setattr(
        "__doc__",
        "Rust-first training primitives for SpiralTorch: tensors, hypergrads, and unified planners.",
    )?;

    Ok(())
}
