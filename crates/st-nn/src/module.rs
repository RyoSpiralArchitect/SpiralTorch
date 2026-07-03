// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

use crate::schedule::GradientBands;
use st_core::backend::device_caps::DeviceCaps;
#[cfg(feature = "psychoid")]
use st_core::telemetry::psychoid::PsychoidSample;
use st_tensor::pure::{
    topos::OpenCartesianTopos, AmegaHypergrad, ComplexTensor, LanguageWaveEncoder, PureResult,
    Tensor, TensorError,
};
use std::collections::{HashMap, HashSet};

const FNV64_OFFSET: u64 = 0xcbf29ce484222325;
const FNV64_PRIME: u64 = 0x00000100000001b3;

pub(crate) fn fingerprint_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV64_PRIME);
    }
}

pub(crate) fn fingerprint_usize(hash: &mut u64, value: usize) {
    fingerprint_bytes(hash, &(value as u64).to_le_bytes());
}

pub(crate) fn fingerprint_u32(hash: &mut u64, value: u32) {
    fingerprint_bytes(hash, &value.to_le_bytes());
}

pub(crate) fn fingerprint_bool(hash: &mut u64, value: bool) {
    fingerprint_bytes(hash, &[u8::from(value)]);
}

pub(crate) fn fingerprint_f32(hash: &mut u64, value: f32) {
    fingerprint_u32(hash, value.to_bits());
}

/// Trainable parameter that can either rely on the hypergrad tape or fall back
/// to standard Euclidean accumulation.
pub struct Parameter {
    name: String,
    value: Tensor,
    gradient: Option<Tensor>,
    hypergrad: Option<AmegaHypergrad>,
    trainable: bool,
    learning_rate_scale: f32,
    weight_decay: f32,
}

impl core::fmt::Debug for Parameter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (rows, cols) = self.value.shape();
        write!(
            f,
            "Parameter(name={},shape=({},{}),has_grad={},has_hypergrad={},trainable={},lr_scale={},weight_decay={})",
            self.name,
            rows,
            cols,
            self.gradient.is_some(),
            self.hypergrad.is_some(),
            self.trainable,
            self.learning_rate_scale,
            self.weight_decay
        )
    }
}

impl Parameter {
    /// Creates a new parameter with the provided tensor value.
    pub fn new(name: impl Into<String>, value: Tensor) -> Self {
        Self {
            name: name.into(),
            value,
            gradient: None,
            hypergrad: None,
            trainable: true,
            learning_rate_scale: 1.0,
            weight_decay: 0.0,
        }
    }

    /// Returns the identifier assigned to the parameter.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Overrides the parameter name.
    pub fn rename(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Provides an immutable view into the underlying tensor value.
    pub fn value(&self) -> &Tensor {
        &self.value
    }

    /// Provides a mutable view into the underlying tensor value.
    pub fn value_mut(&mut self) -> &mut Tensor {
        &mut self.value
    }

    /// Returns whether the parameter participates in gradient accumulation and updates.
    pub fn is_trainable(&self) -> bool {
        self.trainable
    }

    /// Enables or disables learning for this parameter.
    ///
    /// Frozen parameters still accept checkpoint values and still contribute to
    /// forward/backward signal propagation through their current value, but they
    /// do not accumulate gradients or apply optimizer steps.
    pub fn set_trainable(&mut self, trainable: bool) {
        self.trainable = trainable;
        if !trainable {
            self.zero_gradient();
        }
    }

    /// Returns the multiplier applied to this parameter's learning rate.
    pub fn learning_rate_scale(&self) -> f32 {
        self.learning_rate_scale
    }

    /// Returns the decoupled weight decay applied on optimizer steps.
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Scales this parameter's future learning rate.
    pub fn try_scale_learning_rate(&mut self, factor: f32) -> PureResult<()> {
        if factor <= 0.0 || !factor.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: factor });
        }
        self.learning_rate_scale *= factor;
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.scale_learning_rate(factor);
        }
        Ok(())
    }

    /// Sets this parameter's future learning-rate multiplier to an absolute value.
    pub fn try_set_learning_rate_scale(&mut self, scale: f32) -> PureResult<()> {
        if scale <= 0.0 || !scale.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: scale });
        }
        let factor = scale / self.learning_rate_scale;
        self.try_scale_learning_rate(factor)
    }

    /// Sets this parameter's decoupled weight decay. A value of `0.0` disables it.
    pub fn try_set_weight_decay(&mut self, weight_decay: f32) -> PureResult<()> {
        if weight_decay < 0.0 || !weight_decay.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "parameter_weight_decay",
                value: weight_decay,
            });
        }
        self.weight_decay = weight_decay;
        Ok(())
    }

    /// Returns the currently cached Euclidean gradient when no hypergrad tape is active.
    pub fn gradient(&self) -> Option<&Tensor> {
        self.gradient.as_ref()
    }

    /// Attaches a hypergrad tape to the parameter.
    pub fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PureResult<()> {
        let (rows, cols) = self.value.shape();
        let tape = AmegaHypergrad::new(
            curvature,
            learning_rate * self.learning_rate_scale,
            rows,
            cols,
        )?;
        self.hypergrad = Some(tape);
        self.gradient = None;
        Ok(())
    }

    /// Attaches a hypergrad tape using a caller-supplied topos.
    pub fn attach_hypergrad_with_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        let (rows, cols) = self.value.shape();
        let tape = AmegaHypergrad::with_topos(
            curvature,
            learning_rate * self.learning_rate_scale,
            rows,
            cols,
            topos,
        )?;
        self.hypergrad = Some(tape);
        self.gradient = None;
        Ok(())
    }

    /// Provides direct access to the hypergrad tape when attached.
    pub fn hypergrad(&self) -> Option<&AmegaHypergrad> {
        self.hypergrad.as_ref()
    }

    /// Provides mutable access to the hypergrad tape when attached.
    pub fn hypergrad_mut(&mut self) -> Option<&mut AmegaHypergrad> {
        self.hypergrad.as_mut()
    }

    fn assert_shape(&self, tensor: &Tensor) -> PureResult<()> {
        if self.value.shape() != tensor.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.value.shape(),
                right: tensor.shape(),
            });
        }
        Ok(())
    }

    /// Accumulates a Euclidean gradient update. When a hypergrad tape is
    /// attached the value is streamed through the tape, otherwise a local
    /// gradient buffer is maintained.
    pub fn accumulate_euclidean(&mut self, update: &Tensor) -> PureResult<()> {
        self.assert_shape(update)?;
        if !self.trainable {
            return Ok(());
        }
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.accumulate_wave(update)?;
        } else {
            match self.gradient.as_mut() {
                Some(existing) => existing.add_scaled(update, 1.0)?,
                None => {
                    self.gradient = Some(update.clone());
                }
            }
        }
        Ok(())
    }

    /// Streams a complex wave through the attached hypergrad tape or caches an
    /// Euclidean equivalent when no tape is present.
    pub fn accumulate_complex_wave(&mut self, wave: &ComplexTensor) -> PureResult<()> {
        let tensor = wave.to_tensor()?;
        self.assert_shape(&tensor)?;
        if !self.trainable {
            return Ok(());
        }
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.accumulate_complex_wave(wave)
        } else {
            match self.gradient.as_mut() {
                Some(existing) => existing.add_scaled(&tensor, 1.0)?,
                None => {
                    self.gradient = Some(tensor);
                }
            }
            Ok(())
        }
    }

    /// Absorbs free-form text directly into the parameter's accumulator by
    /// delegating to the hypergrad tape or caching the encoded tensor.
    pub fn absorb_text(&mut self, encoder: &LanguageWaveEncoder, text: &str) -> PureResult<()> {
        let tensor = encoder.encode_z_space(text)?;
        self.assert_shape(&tensor)?;
        if !self.trainable {
            return Ok(());
        }
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.absorb_text(encoder, text)
        } else {
            match self.gradient.as_mut() {
                Some(existing) => existing.add_scaled(&tensor, 1.0)?,
                None => {
                    self.gradient = Some(tensor);
                }
            }
            Ok(())
        }
    }

    /// Clears the cached gradient or resets the hypergrad tape accumulator.
    pub fn zero_gradient(&mut self) {
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.reset();
        }
        if let Some(grad) = self.gradient.as_mut() {
            for value in grad.data_mut() {
                *value = 0.0;
            }
        }
    }

    /// Applies the accumulated update either via the hypergrad tape or by using
    /// the supplied fallback learning rate.
    pub fn apply_step(&mut self, fallback_lr: f32) -> PureResult<()> {
        if !self.trainable {
            return Ok(());
        }
        let effective_lr = self
            .hypergrad
            .as_ref()
            .map(|tape| tape.learning_rate())
            .unwrap_or(fallback_lr * self.learning_rate_scale);
        if self.weight_decay > 0.0 && effective_lr > 0.0 {
            let decay = effective_lr * self.weight_decay;
            if !decay.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "parameter_weight_decay_step",
                    value: decay,
                });
            }
            let before = self.value.clone();
            self.value.add_scaled(&before, -decay)?;
        }
        if let Some(tape) = self.hypergrad.as_mut() {
            tape.apply(&mut self.value)?;
        } else if let Some(grad) = self.gradient.as_mut() {
            self.value
                .add_scaled(grad, -(fallback_lr * self.learning_rate_scale))?;
            for value in grad.data_mut() {
                *value = 0.0;
            }
        }
        Ok(())
    }

    /// Scales any accumulated gradient or hypergradient buffers by the provided factor.
    pub fn scale_accumulators(&mut self, factor: f32) {
        if !self.trainable {
            return;
        }
        if !factor.is_finite() {
            return;
        }
        if let Some(tape) = self.hypergrad.as_mut() {
            for grad in tape.gradient_mut() {
                *grad *= factor;
            }
        }
        if let Some(grad) = self.gradient.as_mut() {
            for value in grad.data_mut() {
                *value *= factor;
            }
        }
    }

    /// Returns the squared L2 norm of any accumulated gradients.
    pub fn accumulators_norm_sq(&self) -> f64 {
        if !self.trainable {
            return 0.0;
        }
        if let Some(tape) = self.hypergrad.as_ref() {
            tape.gradient()
                .iter()
                .map(|&value| {
                    let v = value as f64;
                    v * v
                })
                .sum()
        } else if let Some(grad) = self.gradient.as_ref() {
            grad.data()
                .iter()
                .map(|&value| {
                    let v = value as f64;
                    v * v
                })
                .sum()
        } else {
            0.0
        }
    }

    /// Scales the learning rate inside the attached hypergrad tape, if present.
    pub fn scale_learning_rate(&mut self, factor: f32) {
        let _ = self.try_scale_learning_rate(factor);
    }

    /// Replaces the parameter value with the provided tensor.
    pub fn load_value(&mut self, value: &Tensor) -> PureResult<()> {
        self.assert_shape(value)?;
        *self.value_mut() = value.clone();
        Ok(())
    }
}

/// Per-parameter delta captured by [`Module::audit_parameter_movement`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterMovement {
    pub name: String,
    pub trainable: bool,
    pub changed: bool,
    pub l2_delta: f32,
    pub max_abs_delta: f32,
}

/// Summary of which parameters moved after an optimization window.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterMovementReport {
    pub parameters: Vec<ParameterMovement>,
    pub trainable_changed: usize,
    pub trainable_unchanged: usize,
    pub frozen_changed: usize,
    pub frozen_unchanged: usize,
    pub max_trainable_l2_delta: f32,
    pub max_frozen_l2_delta: f32,
    pub max_frozen_abs_delta: f32,
}

impl ParameterMovementReport {
    /// Returns true when no frozen parameter moved beyond the audit tolerance.
    pub fn frozen_stable(&self) -> bool {
        self.frozen_changed == 0
    }

    /// Returns true when at least one trainable parameter changed beyond tolerance.
    pub fn trainable_movement_observed(&self) -> bool {
        self.trainable_changed > 0
    }

    /// Compact status label for checkpoint/FT gates.
    pub fn status(&self) -> &'static str {
        if !self.frozen_stable() {
            "frozen_changed"
        } else if !self.trainable_movement_observed() {
            "no_trainable_movement"
        } else {
            "ok"
        }
    }
}

/// Stable digest of a module state dictionary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateFingerprint {
    pub hash: String,
    pub parameters: usize,
    pub values: usize,
}

/// Result returned by checked state/checkpoint loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateLoadReport {
    pub source: StateFingerprint,
    pub loaded: StateFingerprint,
    pub matched: bool,
}

/// Per-parameter status returned by [`Module::state_dict_compatibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateCompatibilityStatus {
    Matched,
    Missing,
    ShapeMismatch,
    Extra,
}

impl StateCompatibilityStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Matched => "matched",
            Self::Missing => "missing",
            Self::ShapeMismatch => "shape_mismatch",
            Self::Extra => "extra",
        }
    }
}

/// Single checkpoint key comparison against a module's expected state dict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCompatibilityEntry {
    pub name: String,
    pub status: StateCompatibilityStatus,
    pub source_name: Option<String>,
    pub transform: StateTensorTransform,
    pub expected_shape: Option<(usize, usize)>,
    pub source_shape: Option<(usize, usize)>,
    pub original_source_shape: Option<(usize, usize)>,
}

/// Side-effect-free report for checking whether a checkpoint can seed a module.
///
/// Extra source keys are reported but do not make the report incompatible,
/// because adapter/head fine-tunes often load a subset from a larger checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCompatibilityReport {
    pub compatible: bool,
    pub expected_parameters: usize,
    pub source_parameters: usize,
    pub matched: usize,
    pub missing: usize,
    pub shape_mismatched: usize,
    pub extra: usize,
    pub source: StateFingerprint,
    pub matched_subset: StateFingerprint,
    pub entries: Vec<StateCompatibilityEntry>,
}

/// Tensor layout/shape transform applied while importing external checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateTensorTransform {
    Identity,
    Transpose,
    CopyOverlapZeros,
    TransposeCopyOverlapZeros,
}

impl StateTensorTransform {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Transpose => "transpose",
            Self::CopyOverlapZeros => "copy_overlap_zeros",
            Self::TransposeCopyOverlapZeros => "transpose_copy_overlap_zeros",
        }
    }

    pub fn parse(value: &str) -> PureResult<Self> {
        match value {
            "identity" | "" => Ok(Self::Identity),
            "transpose" => Ok(Self::Transpose),
            "copy_overlap_zeros" | "copy_overlap" => Ok(Self::CopyOverlapZeros),
            "transpose_copy_overlap_zeros" | "transpose_copy_overlap" => {
                Ok(Self::TransposeCopyOverlapZeros)
            }
            other => Err(TensorError::IoError {
                message: format!("unsupported state tensor transform: {other}"),
            }),
        }
    }

    fn transposes(self) -> bool {
        matches!(self, Self::Transpose | Self::TransposeCopyOverlapZeros)
    }

    fn copies_overlap(self) -> bool {
        matches!(
            self,
            Self::CopyOverlapZeros | Self::TransposeCopyOverlapZeros
        )
    }
}

/// External checkpoint key import rule.
///
/// `target` is the SpiralTorch module key and `transform` adapts the source
/// tensor before compatibility checks or loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateKeyMapRule {
    pub target: String,
    pub transform: StateTensorTransform,
}

impl StateKeyMapRule {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            transform: StateTensorTransform::Identity,
        }
    }

    pub fn with_transform(target: impl Into<String>, transform: StateTensorTransform) -> Self {
        Self {
            target: target.into(),
            transform,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StateImportEntryMetadata {
    source_name: String,
    transform: StateTensorTransform,
    original_source_shape: (usize, usize),
}

/// Stable digest of parameter training metadata used by resume audits.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterTrainingFingerprint {
    pub hash: String,
    pub parameters: usize,
    pub trainable: usize,
    pub frozen: usize,
    pub hypergrad_tapes: usize,
    pub accumulated_l2: f32,
}

/// Computes a deterministic fingerprint for a state dictionary.
pub fn fingerprint_state_dict(state: &HashMap<String, Tensor>) -> StateFingerprint {
    let mut hash = FNV64_OFFSET;
    let mut names: Vec<&String> = state.keys().collect();
    names.sort();
    let mut values = 0usize;
    fingerprint_usize(&mut hash, names.len());
    for name in names {
        let tensor = &state[name];
        let (rows, cols) = tensor.shape();
        fingerprint_usize(&mut hash, name.len());
        fingerprint_bytes(&mut hash, name.as_bytes());
        fingerprint_usize(&mut hash, rows);
        fingerprint_usize(&mut hash, cols);
        fingerprint_usize(&mut hash, tensor.data().len());
        values += tensor.data().len();
        for value in tensor.data() {
            fingerprint_u32(&mut hash, value.to_bits());
        }
    }
    StateFingerprint {
        hash: format!("{hash:016x}"),
        parameters: state.len(),
        values,
    }
}

fn copy_overlap_zeros(source: &Tensor, rows: usize, cols: usize) -> PureResult<Tensor> {
    let (source_rows, source_cols) = source.shape();
    Tensor::from_fn(rows, cols, |r, c| {
        if r < source_rows && c < source_cols {
            source.data()[r * source_cols + c]
        } else {
            0.0
        }
    })
}

fn transform_state_tensor(
    tensor: &Tensor,
    transform: StateTensorTransform,
    target_shape: Option<(usize, usize)>,
) -> PureResult<Tensor> {
    let transformed = if transform.transposes() {
        tensor.transpose()
    } else {
        tensor.clone()
    };
    if transform.copies_overlap() {
        let Some((rows, cols)) = target_shape else {
            return Err(TensorError::IoError {
                message: format!(
                    "state tensor transform {} requires an expected target shape",
                    transform.as_str()
                ),
            });
        };
        copy_overlap_zeros(&transformed, rows, cols)
    } else {
        Ok(transformed)
    }
}

/// Remaps source checkpoint keys into module-canonical target keys.
///
/// The map is interpreted as `source_key -> target_key`. Unmapped source keys
/// are preserved so large external checkpoints still appear as `extra` keys in
/// compatibility reports. Duplicate target keys are rejected because they would
/// make checkpoint loading order-dependent.
pub fn remap_state_dict_keys(
    state: &HashMap<String, Tensor>,
    source_to_target: &HashMap<String, String>,
) -> PureResult<HashMap<String, Tensor>> {
    for (source, target) in source_to_target {
        if source.is_empty() || target.is_empty() {
            return Err(TensorError::IoError {
                message: "state dict key map entries must not be empty".to_string(),
            });
        }
    }

    let mut remapped = HashMap::new();
    let mut target_sources = HashMap::new();
    let mut sources: Vec<&String> = state.keys().collect();
    sources.sort();
    for source in sources {
        let target = source_to_target.get(source).unwrap_or(source);
        if let Some(previous) = target_sources.insert(target.clone(), source.clone()) {
            return Err(TensorError::IoError {
                message: format!(
                    "state dict key map sends both {previous} and {source} to {target}"
                ),
            });
        }
        remapped.insert(target.clone(), state[source].clone());
    }
    Ok(remapped)
}

pub fn state_key_rules_from_map(
    source_to_target: &HashMap<String, String>,
) -> HashMap<String, StateKeyMapRule> {
    source_to_target
        .iter()
        .map(|(source, target)| (source.clone(), StateKeyMapRule::new(target.clone())))
        .collect()
}

pub fn adapt_state_dict_keys(
    state: &HashMap<String, Tensor>,
    rules: &HashMap<String, StateKeyMapRule>,
    expected_shapes: &HashMap<String, (usize, usize)>,
) -> PureResult<HashMap<String, Tensor>> {
    let (adapted, _) = adapt_state_dict_keys_with_metadata(state, rules, expected_shapes)?;
    Ok(adapted)
}

fn adapt_state_dict_keys_with_metadata(
    state: &HashMap<String, Tensor>,
    rules: &HashMap<String, StateKeyMapRule>,
    expected_shapes: &HashMap<String, (usize, usize)>,
) -> PureResult<(
    HashMap<String, Tensor>,
    HashMap<String, StateImportEntryMetadata>,
)> {
    for (source, rule) in rules {
        if source.is_empty() || rule.target.is_empty() {
            return Err(TensorError::IoError {
                message: "state dict key rule entries must not be empty".to_string(),
            });
        }
    }

    let mut adapted = HashMap::new();
    let mut metadata = HashMap::new();
    let mut target_sources = HashMap::new();
    let mut sources: Vec<&String> = state.keys().collect();
    sources.sort();
    for source in sources {
        let (target, transform) = rules
            .get(source)
            .map(|rule| (rule.target.as_str(), rule.transform))
            .unwrap_or((source.as_str(), StateTensorTransform::Identity));
        if let Some(previous) = target_sources.insert(target.to_string(), source.clone()) {
            return Err(TensorError::IoError {
                message: format!(
                    "state dict key rule sends both {previous} and {source} to {target}"
                ),
            });
        }
        let target_shape = expected_shapes.get(target).copied();
        let original_source_shape = state[source].shape();
        let tensor = if target_shape.is_none() && transform.copies_overlap() {
            if transform.transposes() {
                state[source].transpose()
            } else {
                state[source].clone()
            }
        } else {
            transform_state_tensor(&state[source], transform, target_shape)?
        };
        adapted.insert(target.to_string(), tensor);
        metadata.insert(
            target.to_string(),
            StateImportEntryMetadata {
                source_name: source.clone(),
                transform,
                original_source_shape,
            },
        );
    }
    Ok((adapted, metadata))
}

pub(crate) fn state_dict_compatibility_for_expected(
    mut expected: Vec<(String, (usize, usize))>,
    state: &HashMap<String, Tensor>,
) -> StateCompatibilityReport {
    state_dict_compatibility_for_expected_with_metadata(&mut expected, state, None)
}

pub(crate) fn state_dict_compatibility_for_expected_with_rules(
    mut expected: Vec<(String, (usize, usize))>,
    state: &HashMap<String, Tensor>,
    rules: &HashMap<String, StateKeyMapRule>,
) -> PureResult<StateCompatibilityReport> {
    let expected_shapes: HashMap<String, (usize, usize)> = expected
        .iter()
        .map(|(name, shape)| (name.clone(), *shape))
        .collect();
    let (adapted, metadata) = adapt_state_dict_keys_with_metadata(state, rules, &expected_shapes)?;
    Ok(state_dict_compatibility_for_expected_with_metadata(
        &mut expected,
        &adapted,
        Some(&metadata),
    ))
}

fn state_dict_compatibility_for_expected_with_metadata(
    expected: &mut Vec<(String, (usize, usize))>,
    state: &HashMap<String, Tensor>,
    metadata: Option<&HashMap<String, StateImportEntryMetadata>>,
) -> StateCompatibilityReport {
    let source = fingerprint_state_dict(state);
    expected.sort_by(|left, right| left.0.cmp(&right.0));

    let expected_names: HashSet<String> = expected.iter().map(|(name, _)| name.clone()).collect();
    let mut entries = Vec::new();
    let mut matched_subset = HashMap::new();
    let mut matched = 0usize;
    let mut missing = 0usize;
    let mut shape_mismatched = 0usize;

    for (name, expected_shape) in expected.iter() {
        match state.get(name) {
            Some(source_tensor) if source_tensor.shape() == *expected_shape => {
                let entry_metadata = compatibility_metadata(name, source_tensor, metadata);
                matched += 1;
                matched_subset.insert(name.clone(), source_tensor.clone());
                entries.push(StateCompatibilityEntry {
                    name: name.clone(),
                    status: StateCompatibilityStatus::Matched,
                    source_name: Some(entry_metadata.source_name),
                    transform: entry_metadata.transform,
                    expected_shape: Some(*expected_shape),
                    source_shape: Some(source_tensor.shape()),
                    original_source_shape: Some(entry_metadata.original_source_shape),
                });
            }
            Some(source_tensor) => {
                let entry_metadata = compatibility_metadata(name, source_tensor, metadata);
                shape_mismatched += 1;
                entries.push(StateCompatibilityEntry {
                    name: name.clone(),
                    status: StateCompatibilityStatus::ShapeMismatch,
                    source_name: Some(entry_metadata.source_name),
                    transform: entry_metadata.transform,
                    expected_shape: Some(*expected_shape),
                    source_shape: Some(source_tensor.shape()),
                    original_source_shape: Some(entry_metadata.original_source_shape),
                });
            }
            None => {
                missing += 1;
                entries.push(StateCompatibilityEntry {
                    name: name.clone(),
                    status: StateCompatibilityStatus::Missing,
                    source_name: None,
                    transform: StateTensorTransform::Identity,
                    expected_shape: Some(*expected_shape),
                    source_shape: None,
                    original_source_shape: None,
                });
            }
        }
    }

    let mut extra_names: Vec<&String> = state
        .keys()
        .filter(|name| !expected_names.contains(*name))
        .collect();
    extra_names.sort();
    let extra = extra_names.len();
    for name in extra_names {
        let tensor = &state[name];
        let entry_metadata = compatibility_metadata(name, tensor, metadata);
        entries.push(StateCompatibilityEntry {
            name: name.clone(),
            status: StateCompatibilityStatus::Extra,
            source_name: Some(entry_metadata.source_name),
            transform: entry_metadata.transform,
            expected_shape: None,
            source_shape: Some(tensor.shape()),
            original_source_shape: Some(entry_metadata.original_source_shape),
        });
    }

    let compatible = missing == 0 && shape_mismatched == 0;
    StateCompatibilityReport {
        compatible,
        expected_parameters: expected_names.len(),
        source_parameters: state.len(),
        matched,
        missing,
        shape_mismatched,
        extra,
        source,
        matched_subset: fingerprint_state_dict(&matched_subset),
        entries,
    }
}

fn compatibility_metadata(
    name: &str,
    tensor: &Tensor,
    metadata: Option<&HashMap<String, StateImportEntryMetadata>>,
) -> StateImportEntryMetadata {
    metadata
        .and_then(|entries| entries.get(name))
        .cloned()
        .unwrap_or_else(|| StateImportEntryMetadata {
            source_name: name.to_string(),
            transform: StateTensorTransform::Identity,
            original_source_shape: tensor.shape(),
        })
}

#[derive(Debug, Clone)]
struct ParameterTrainingEntry {
    name: String,
    rows: usize,
    cols: usize,
    trainable: bool,
    learning_rate_scale: f32,
    weight_decay: f32,
    hypergrad_learning_rate: Option<f32>,
    accumulator_l2: f32,
}

/// High-level module trait inspired by PyTorch's `nn.Module` but expressed in
/// pure Rust so it can be used from WebGPU, HIP, or CPU flows alike.
pub trait Module {
    /// Runs a forward pass.
    fn forward(&self, input: &Tensor) -> PureResult<Tensor>;

    /// Propagates a gradient backwards. Implementations should populate the
    /// relevant parameter accumulators before returning the gradient with
    /// respect to `input`.
    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor>;

    /// Visits immutable parameters.
    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()>;

    /// Visits mutable parameters.
    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()>;

    /// Propagates an Above/Here/Beneath gradient schedule through the module.
    fn backward_bands(&mut self, input: &Tensor, bands: &GradientBands) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let mut total = Tensor::zeros(rows, cols)?;
        for grad in bands.iter() {
            if grad.squared_l2_norm() == 0.0 {
                continue;
            }
            let contribution = self.backward(input, grad)?;
            total.add_scaled(&contribution, 1.0)?;
        }
        Ok(total)
    }

    /// Attaches a hypergrad tape to every parameter.
    fn attach_hypergrad(&mut self, curvature: f32, learning_rate: f32) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| param.attach_hypergrad(curvature, learning_rate))
    }

    /// Attaches a hypergrad tape using a shared topos.
    fn attach_hypergrad_with_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| {
            param.attach_hypergrad_with_topos(curvature, learning_rate, topos.clone())
        })
    }

    /// Applies every parameter update.
    fn apply_step(&mut self, fallback_lr: f32) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| param.apply_step(fallback_lr))
    }

    /// Enables or freezes all parameters in the module.
    fn set_trainable(&mut self, trainable: bool) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| {
            param.set_trainable(trainable);
            Ok(())
        })
    }

    /// Enables or freezes one parameter by its canonical state-dict name.
    fn set_parameter_trainable(&mut self, name: &str, trainable: bool) -> PureResult<()> {
        let mut found = false;
        self.visit_parameters_mut(&mut |param| {
            if param.name() == name {
                param.set_trainable(trainable);
                found = true;
            }
            Ok(())
        })?;
        if found {
            Ok(())
        } else {
            Err(TensorError::MissingParameter {
                name: name.to_string(),
            })
        }
    }

    /// Enables or freezes every parameter whose name satisfies the matcher.
    ///
    /// Returns the number of matched parameters. A zero-match result is treated
    /// as an error so fine-tuning presets do not silently miss their intended
    /// adapter/head group.
    fn set_parameters_trainable_matching(
        &mut self,
        label: &str,
        trainable: bool,
        matcher: &mut dyn FnMut(&str) -> bool,
    ) -> PureResult<usize> {
        if label.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter trainable matcher label must not be empty".to_string(),
            });
        }
        let mut matched = 0usize;
        self.visit_parameters_mut(&mut |param| {
            if matcher(param.name()) {
                param.set_trainable(trainable);
                matched += 1;
            }
            Ok(())
        })?;
        if matched == 0 {
            Err(TensorError::MissingParameter {
                name: label.to_string(),
            })
        } else {
            Ok(matched)
        }
    }

    /// Enables or freezes parameters whose canonical names start with `prefix`.
    fn set_parameters_trainable_by_prefix(
        &mut self,
        prefix: &str,
        trainable: bool,
    ) -> PureResult<usize> {
        if prefix.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter prefix must not be empty".to_string(),
            });
        }
        let label = format!("prefix:{prefix}");
        self.set_parameters_trainable_matching(&label, trainable, &mut |name| {
            name.starts_with(prefix)
        })
    }

    /// Enables or freezes parameters whose canonical names end with `suffix`.
    fn set_parameters_trainable_by_suffix(
        &mut self,
        suffix: &str,
        trainable: bool,
    ) -> PureResult<usize> {
        if suffix.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter suffix must not be empty".to_string(),
            });
        }
        let label = format!("suffix:{suffix}");
        self.set_parameters_trainable_matching(&label, trainable, &mut |name| {
            name.ends_with(suffix)
        })
    }

    /// Enables or freezes parameters whose canonical names contain `needle`.
    fn set_parameters_trainable_by_contains(
        &mut self,
        needle: &str,
        trainable: bool,
    ) -> PureResult<usize> {
        if needle.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter contains pattern must not be empty".to_string(),
            });
        }
        let label = format!("contains:{needle}");
        self.set_parameters_trainable_matching(&label, trainable, &mut |name| name.contains(needle))
    }

    /// Scales one parameter's learning rate by exact canonical name.
    fn scale_parameter_learning_rate(&mut self, name: &str, factor: f32) -> PureResult<()> {
        let mut found = false;
        self.visit_parameters_mut(&mut |param| {
            if param.name() == name {
                param.try_scale_learning_rate(factor)?;
                found = true;
            }
            Ok(())
        })?;
        if found {
            Ok(())
        } else {
            Err(TensorError::MissingParameter {
                name: name.to_string(),
            })
        }
    }

    /// Sets one parameter's learning-rate multiplier by exact canonical name.
    fn set_parameter_learning_rate_scale(&mut self, name: &str, scale: f32) -> PureResult<()> {
        let mut found = false;
        self.visit_parameters_mut(&mut |param| {
            if param.name() == name {
                param.try_set_learning_rate_scale(scale)?;
                found = true;
            }
            Ok(())
        })?;
        if found {
            Ok(())
        } else {
            Err(TensorError::MissingParameter {
                name: name.to_string(),
            })
        }
    }

    /// Scales learning rates for every parameter whose name satisfies the matcher.
    fn scale_parameters_learning_rate_matching(
        &mut self,
        label: &str,
        factor: f32,
        matcher: &mut dyn FnMut(&str) -> bool,
    ) -> PureResult<usize> {
        if label.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter learning-rate matcher label must not be empty".to_string(),
            });
        }
        if factor <= 0.0 || !factor.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: factor });
        }
        let mut matched = 0usize;
        self.visit_parameters_mut(&mut |param| {
            if matcher(param.name()) {
                param.try_scale_learning_rate(factor)?;
                matched += 1;
            }
            Ok(())
        })?;
        if matched == 0 {
            Err(TensorError::MissingParameter {
                name: label.to_string(),
            })
        } else {
            Ok(matched)
        }
    }

    /// Sets learning-rate multipliers for every parameter whose name satisfies the matcher.
    fn set_parameters_learning_rate_scale_matching(
        &mut self,
        label: &str,
        scale: f32,
        matcher: &mut dyn FnMut(&str) -> bool,
    ) -> PureResult<usize> {
        if label.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter learning-rate-scale matcher label must not be empty"
                    .to_string(),
            });
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: scale });
        }
        let mut matched = 0usize;
        self.visit_parameters_mut(&mut |param| {
            if matcher(param.name()) {
                param.try_set_learning_rate_scale(scale)?;
                matched += 1;
            }
            Ok(())
        })?;
        if matched == 0 {
            Err(TensorError::MissingParameter {
                name: label.to_string(),
            })
        } else {
            Ok(matched)
        }
    }

    /// Scales learning rates for parameters whose canonical names start with `prefix`.
    fn scale_parameters_learning_rate_by_prefix(
        &mut self,
        prefix: &str,
        factor: f32,
    ) -> PureResult<usize> {
        if prefix.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter learning-rate prefix must not be empty".to_string(),
            });
        }
        let label = format!("prefix:{prefix}");
        self.scale_parameters_learning_rate_matching(&label, factor, &mut |name| {
            name.starts_with(prefix)
        })
    }

    /// Scales learning rates for parameters whose canonical names end with `suffix`.
    fn scale_parameters_learning_rate_by_suffix(
        &mut self,
        suffix: &str,
        factor: f32,
    ) -> PureResult<usize> {
        if suffix.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter learning-rate suffix must not be empty".to_string(),
            });
        }
        let label = format!("suffix:{suffix}");
        self.scale_parameters_learning_rate_matching(&label, factor, &mut |name| {
            name.ends_with(suffix)
        })
    }

    /// Scales learning rates for parameters whose canonical names contain `needle`.
    fn scale_parameters_learning_rate_by_contains(
        &mut self,
        needle: &str,
        factor: f32,
    ) -> PureResult<usize> {
        if needle.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter learning-rate contains pattern must not be empty".to_string(),
            });
        }
        let label = format!("contains:{needle}");
        self.scale_parameters_learning_rate_matching(&label, factor, &mut |name| {
            name.contains(needle)
        })
    }

    /// Sets learning-rate multipliers for parameters whose canonical names start with `prefix`.
    fn set_parameters_learning_rate_scale_by_prefix(
        &mut self,
        prefix: &str,
        scale: f32,
    ) -> PureResult<usize> {
        if prefix.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter learning-rate-scale prefix must not be empty".to_string(),
            });
        }
        let label = format!("prefix:{prefix}");
        self.set_parameters_learning_rate_scale_matching(&label, scale, &mut |name| {
            name.starts_with(prefix)
        })
    }

    /// Sets learning-rate multipliers for parameters whose canonical names end with `suffix`.
    fn set_parameters_learning_rate_scale_by_suffix(
        &mut self,
        suffix: &str,
        scale: f32,
    ) -> PureResult<usize> {
        if suffix.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter learning-rate-scale suffix must not be empty".to_string(),
            });
        }
        let label = format!("suffix:{suffix}");
        self.set_parameters_learning_rate_scale_matching(&label, scale, &mut |name| {
            name.ends_with(suffix)
        })
    }

    /// Sets learning-rate multipliers for parameters whose canonical names contain `needle`.
    fn set_parameters_learning_rate_scale_by_contains(
        &mut self,
        needle: &str,
        scale: f32,
    ) -> PureResult<usize> {
        if needle.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter learning-rate-scale contains pattern must not be empty"
                    .to_string(),
            });
        }
        let label = format!("contains:{needle}");
        self.set_parameters_learning_rate_scale_matching(&label, scale, &mut |name| {
            name.contains(needle)
        })
    }

    /// Sets decoupled weight decay for one parameter by exact canonical name.
    fn set_parameter_weight_decay(&mut self, name: &str, weight_decay: f32) -> PureResult<()> {
        let mut found = false;
        self.visit_parameters_mut(&mut |param| {
            if param.name() == name {
                param.try_set_weight_decay(weight_decay)?;
                found = true;
            }
            Ok(())
        })?;
        if found {
            Ok(())
        } else {
            Err(TensorError::MissingParameter {
                name: name.to_string(),
            })
        }
    }

    /// Sets decoupled weight decay for every parameter whose name satisfies the matcher.
    fn set_parameters_weight_decay_matching(
        &mut self,
        label: &str,
        weight_decay: f32,
        matcher: &mut dyn FnMut(&str) -> bool,
    ) -> PureResult<usize> {
        if label.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter weight-decay matcher label must not be empty".to_string(),
            });
        }
        if weight_decay < 0.0 || !weight_decay.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "parameter_weight_decay",
                value: weight_decay,
            });
        }
        let mut matched = 0usize;
        self.visit_parameters_mut(&mut |param| {
            if matcher(param.name()) {
                param.try_set_weight_decay(weight_decay)?;
                matched += 1;
            }
            Ok(())
        })?;
        if matched == 0 {
            Err(TensorError::MissingParameter {
                name: label.to_string(),
            })
        } else {
            Ok(matched)
        }
    }

    /// Sets decoupled weight decay for parameters whose canonical names start with `prefix`.
    fn set_parameters_weight_decay_by_prefix(
        &mut self,
        prefix: &str,
        weight_decay: f32,
    ) -> PureResult<usize> {
        if prefix.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter weight-decay prefix must not be empty".to_string(),
            });
        }
        let label = format!("prefix:{prefix}");
        self.set_parameters_weight_decay_matching(&label, weight_decay, &mut |name| {
            name.starts_with(prefix)
        })
    }

    /// Sets decoupled weight decay for parameters whose canonical names end with `suffix`.
    fn set_parameters_weight_decay_by_suffix(
        &mut self,
        suffix: &str,
        weight_decay: f32,
    ) -> PureResult<usize> {
        if suffix.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter weight-decay suffix must not be empty".to_string(),
            });
        }
        let label = format!("suffix:{suffix}");
        self.set_parameters_weight_decay_matching(&label, weight_decay, &mut |name| {
            name.ends_with(suffix)
        })
    }

    /// Sets decoupled weight decay for parameters whose canonical names contain `needle`.
    fn set_parameters_weight_decay_by_contains(
        &mut self,
        needle: &str,
        weight_decay: f32,
    ) -> PureResult<usize> {
        if needle.is_empty() {
            return Err(TensorError::IoError {
                message: "parameter weight-decay contains pattern must not be empty".to_string(),
            });
        }
        let label = format!("contains:{needle}");
        self.set_parameters_weight_decay_matching(&label, weight_decay, &mut |name| {
            name.contains(needle)
        })
    }

    /// Compares current parameters against a previously captured state dict.
    fn audit_parameter_movement(
        &self,
        before: &HashMap<String, Tensor>,
        tolerance: f32,
    ) -> PureResult<ParameterMovementReport> {
        if !tolerance.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "parameter_movement_tolerance",
                value: tolerance,
            });
        }
        let tolerance = tolerance.max(0.0);
        let mut report = ParameterMovementReport {
            parameters: Vec::new(),
            trainable_changed: 0,
            trainable_unchanged: 0,
            frozen_changed: 0,
            frozen_unchanged: 0,
            max_trainable_l2_delta: 0.0,
            max_frozen_l2_delta: 0.0,
            max_frozen_abs_delta: 0.0,
        };
        self.visit_parameters(&mut |param| {
            let Some(previous) = before.get(param.name()) else {
                return Err(TensorError::MissingParameter {
                    name: param.name().to_string(),
                });
            };
            if previous.shape() != param.value().shape() {
                return Err(TensorError::ShapeMismatch {
                    left: previous.shape(),
                    right: param.value().shape(),
                });
            }
            let mut l2_sq = 0.0f64;
            let mut max_abs_delta = 0.0f32;
            for (&old, &new) in previous.data().iter().zip(param.value().data().iter()) {
                let delta = new - old;
                let abs_delta = delta.abs();
                l2_sq += (delta as f64) * (delta as f64);
                max_abs_delta = max_abs_delta.max(abs_delta);
            }
            let l2_delta = l2_sq.sqrt() as f32;
            let changed = l2_delta > tolerance || max_abs_delta > tolerance;
            if param.is_trainable() {
                report.max_trainable_l2_delta = report.max_trainable_l2_delta.max(l2_delta);
                if changed {
                    report.trainable_changed += 1;
                } else {
                    report.trainable_unchanged += 1;
                }
            } else {
                report.max_frozen_l2_delta = report.max_frozen_l2_delta.max(l2_delta);
                report.max_frozen_abs_delta = report.max_frozen_abs_delta.max(max_abs_delta);
                if changed {
                    report.frozen_changed += 1;
                } else {
                    report.frozen_unchanged += 1;
                }
            }
            report.parameters.push(ParameterMovement {
                name: param.name().to_string(),
                trainable: param.is_trainable(),
                changed,
                l2_delta,
                max_abs_delta,
            });
            Ok(())
        })?;
        Ok(report)
    }

    /// Clears accumulators across every parameter.
    fn zero_accumulators(&mut self) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| {
            param.zero_gradient();
            Ok(())
        })
    }

    /// Optional hook that surfaces activation drift telemetry for ψ metering.
    ///
    /// Implementations may override this to provide a smoothed scalar that
    /// captures how far their activations drifted during the most recent step.
    /// Returning `None` indicates that the module does not contribute drift
    /// telemetry, allowing the ψ meter to fall back to zero for that component.
    fn psi_probe(&self) -> Option<f32> {
        None
    }

    #[cfg(feature = "psychoid")]
    fn psychoid_sample(&self, _input: &Tensor, _output: &Tensor) -> Option<PsychoidSample> {
        None
    }

    /// Allows modules to describe the device they expect to run on. The default
    /// implementation simply returns `None` which indicates the module is agnostic.
    fn preferred_device(&self) -> Option<DeviceCaps> {
        None
    }

    /// Captures a copy of every parameter tensor keyed by its canonical name.
    fn state_dict(&self) -> PureResult<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        self.visit_parameters(&mut |param| {
            state.insert(param.name().to_string(), param.value().clone());
            Ok(())
        })?;
        Ok(state)
    }

    /// Computes a stable digest for the current parameter tensors.
    fn state_fingerprint(&self) -> PureResult<StateFingerprint> {
        let state = self.state_dict()?;
        Ok(fingerprint_state_dict(&state))
    }

    /// Checks whether a source state dict can seed this module without loading it.
    ///
    /// Missing or shape-mismatched module parameters make the report
    /// incompatible. Extra source keys are included in the report but allowed,
    /// matching [`Self::load_state_dict_subset_checked`].
    fn state_dict_compatibility(
        &self,
        state: &HashMap<String, Tensor>,
    ) -> PureResult<StateCompatibilityReport> {
        let mut expected = Vec::new();
        self.visit_parameters(&mut |param| {
            expected.push((param.name().to_string(), param.value().shape()));
            Ok(())
        })?;
        Ok(state_dict_compatibility_for_expected(expected, state))
    }

    /// Checks compatibility after remapping external checkpoint keys.
    ///
    /// `source_to_target` is interpreted as `external_checkpoint_key ->
    /// module_state_dict_key`; unmapped keys are preserved as extras.
    fn state_dict_compatibility_with_key_map(
        &self,
        state: &HashMap<String, Tensor>,
        source_to_target: &HashMap<String, String>,
    ) -> PureResult<StateCompatibilityReport> {
        let rules = state_key_rules_from_map(source_to_target);
        self.state_dict_compatibility_with_key_rules(state, &rules)
    }

    /// Checks compatibility after applying external key/layout transform rules.
    fn state_dict_compatibility_with_key_rules(
        &self,
        state: &HashMap<String, Tensor>,
        rules: &HashMap<String, StateKeyMapRule>,
    ) -> PureResult<StateCompatibilityReport> {
        let mut expected = Vec::new();
        self.visit_parameters(&mut |param| {
            expected.push((param.name().to_string(), param.value().shape()));
            Ok(())
        })?;
        let expected_shapes: HashMap<String, (usize, usize)> = expected
            .iter()
            .map(|(name, shape)| (name.clone(), *shape))
            .collect();
        let (adapted, metadata) =
            adapt_state_dict_keys_with_metadata(state, rules, &expected_shapes)?;
        Ok(state_dict_compatibility_for_expected_with_metadata(
            &mut expected,
            &adapted,
            Some(&metadata),
        ))
    }

    /// Computes a stable digest for trainability, LR scale, hypergrad, and
    /// accumulator metadata. This complements [`Self::state_fingerprint`],
    /// which intentionally covers parameter values only.
    fn training_state_fingerprint(&self) -> PureResult<ParameterTrainingFingerprint> {
        let mut entries = Vec::new();
        self.visit_parameters(&mut |param| {
            let (rows, cols) = param.value().shape();
            let accumulator_l2 = param.accumulators_norm_sq().sqrt() as f32;
            if !accumulator_l2.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "parameter_training_accumulator_l2",
                    value: accumulator_l2,
                });
            }
            entries.push(ParameterTrainingEntry {
                name: param.name().to_string(),
                rows,
                cols,
                trainable: param.is_trainable(),
                learning_rate_scale: param.learning_rate_scale(),
                weight_decay: param.weight_decay(),
                hypergrad_learning_rate: param.hypergrad().map(|tape| tape.learning_rate()),
                accumulator_l2,
            });
            Ok(())
        })?;
        entries.sort_by(|left, right| left.name.cmp(&right.name));

        let mut hash = FNV64_OFFSET;
        let mut trainable = 0usize;
        let mut frozen = 0usize;
        let mut hypergrad_tapes = 0usize;
        let mut accumulated_l2_sq = 0.0f64;
        fingerprint_usize(&mut hash, entries.len());
        for entry in entries {
            fingerprint_usize(&mut hash, entry.name.len());
            fingerprint_bytes(&mut hash, entry.name.as_bytes());
            fingerprint_usize(&mut hash, entry.rows);
            fingerprint_usize(&mut hash, entry.cols);
            fingerprint_bool(&mut hash, entry.trainable);
            fingerprint_f32(&mut hash, entry.learning_rate_scale);
            fingerprint_f32(&mut hash, entry.weight_decay);
            match entry.hypergrad_learning_rate {
                Some(rate) => {
                    hypergrad_tapes += 1;
                    fingerprint_bool(&mut hash, true);
                    fingerprint_f32(&mut hash, rate);
                }
                None => fingerprint_bool(&mut hash, false),
            }
            fingerprint_f32(&mut hash, entry.accumulator_l2);
            if entry.trainable {
                trainable += 1;
            } else {
                frozen += 1;
            }
            let l2 = entry.accumulator_l2 as f64;
            accumulated_l2_sq += l2 * l2;
        }
        Ok(ParameterTrainingFingerprint {
            hash: format!("{hash:016x}"),
            parameters: trainable + frozen,
            trainable,
            frozen,
            hypergrad_tapes,
            accumulated_l2: accumulated_l2_sq.sqrt() as f32,
        })
    }

    /// Restores parameters from a state dictionary produced by [`Module::state_dict`].
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| {
            let Some(value) = state.get(param.name()) else {
                return Err(TensorError::MissingParameter {
                    name: param.name().to_string(),
                });
            };
            param.load_value(value)
        })
    }

    /// Restores parameters and verifies the loaded state matches the source digest.
    fn load_state_dict_checked(
        &mut self,
        state: &HashMap<String, Tensor>,
    ) -> PureResult<StateLoadReport> {
        let source = fingerprint_state_dict(state);
        self.load_state_dict(state)?;
        let loaded = self.state_fingerprint()?;
        let matched = source == loaded;
        Ok(StateLoadReport {
            source,
            loaded,
            matched,
        })
    }

    /// Restores this module from the subset of a larger state dictionary whose
    /// keys match the module's canonical parameter names.
    ///
    /// This is useful when adapting part of a larger checkpoint, such as
    /// loading a dense MLP's `embed::*` tensors into a standalone embedding
    /// layer before replacing the head with a low-rank adapter. Extra source
    /// keys are ignored, but every current module parameter must be present.
    fn load_state_dict_subset_checked(
        &mut self,
        state: &HashMap<String, Tensor>,
    ) -> PureResult<StateLoadReport> {
        let mut subset = HashMap::new();
        self.visit_parameters(&mut |param| {
            let Some(value) = state.get(param.name()) else {
                return Err(TensorError::MissingParameter {
                    name: param.name().to_string(),
                });
            };
            subset.insert(param.name().to_string(), value.clone());
            Ok(())
        })?;
        let source = fingerprint_state_dict(&subset);
        self.load_state_dict(&subset)?;
        let loaded = self.state_fingerprint()?;
        let matched = source == loaded;
        Ok(StateLoadReport {
            source,
            loaded,
            matched,
        })
    }

    /// Remaps external checkpoint keys, then loads the matching module subset.
    fn load_state_dict_subset_mapped_checked(
        &mut self,
        state: &HashMap<String, Tensor>,
        source_to_target: &HashMap<String, String>,
    ) -> PureResult<StateLoadReport> {
        let rules = state_key_rules_from_map(source_to_target);
        self.load_state_dict_subset_adapted_checked(state, &rules)
    }

    /// Applies external key/layout transform rules, then loads the module subset.
    fn load_state_dict_subset_adapted_checked(
        &mut self,
        state: &HashMap<String, Tensor>,
        rules: &HashMap<String, StateKeyMapRule>,
    ) -> PureResult<StateLoadReport> {
        let mut expected = Vec::new();
        self.visit_parameters(&mut |param| {
            expected.push((param.name().to_string(), param.value().shape()));
            Ok(())
        })?;
        let expected_shapes: HashMap<String, (usize, usize)> = expected
            .iter()
            .map(|(name, shape)| (name.clone(), *shape))
            .collect();
        let adapted = adapt_state_dict_keys(state, rules, &expected_shapes)?;
        self.load_state_dict_subset_checked(&adapted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_absorbs_waves_without_hypergrad() {
        let encoder = LanguageWaveEncoder::new(-1.1, 0.6).unwrap();
        let wave = encoder.encode_wave("flux").unwrap();
        let tensor = wave.to_tensor().unwrap();
        let mut param = Parameter::new("gate", Tensor::zeros(1, tensor.shape().1).unwrap());
        param.accumulate_complex_wave(&wave).unwrap();
        assert!(param.gradient().is_some());
        param.absorb_text(&encoder, "flux").unwrap();
        assert!(param.gradient().unwrap().squared_l2_norm() > 0.0);
    }

    #[test]
    fn parameter_streams_wave_through_hypergrad() {
        let encoder = LanguageWaveEncoder::new(-0.95, 0.8).unwrap();
        let wave = encoder.encode_wave("spiral").unwrap();
        let tensor = wave.to_tensor().unwrap();
        let mut param = Parameter::new("gate", Tensor::zeros(1, tensor.shape().1).unwrap());
        param.attach_hypergrad(encoder.curvature(), 0.05).unwrap();
        param.accumulate_complex_wave(&wave).unwrap();
        param.absorb_text(&encoder, "spiral").unwrap();
        assert!(param.hypergrad().is_some());
        param.apply_step(0.01).unwrap();
    }

    #[test]
    fn parameter_learning_rate_scale_affects_fallback_step() {
        let mut param = Parameter::new("bias", Tensor::zeros(1, 1).unwrap());
        param.try_scale_learning_rate(2.0).unwrap();
        param
            .accumulate_euclidean(&Tensor::from_vec(1, 1, vec![1.0]).unwrap())
            .unwrap();
        param.apply_step(0.1).unwrap();
        assert!((param.value().data()[0] + 0.2).abs() < 1e-6);
        assert!((param.learning_rate_scale() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn parameter_learning_rate_scale_survives_hypergrad_attach() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 1).unwrap());
        param.try_scale_learning_rate(0.5).unwrap();
        param.attach_hypergrad(-1.0, 0.1).unwrap();
        assert!((param.hypergrad().unwrap().learning_rate() - 0.05).abs() < 1e-6);
        param.try_scale_learning_rate(2.0).unwrap();
        assert!((param.hypergrad().unwrap().learning_rate() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn parameter_learning_rate_scale_can_be_set_absolutely() {
        let mut param = Parameter::new("weight", Tensor::zeros(1, 1).unwrap());
        param.try_scale_learning_rate(4.0).unwrap();
        param.try_set_learning_rate_scale(2.0).unwrap();
        assert!((param.learning_rate_scale() - 2.0).abs() < 1e-6);
        param.attach_hypergrad(-1.0, 0.1).unwrap();
        assert!((param.hypergrad().unwrap().learning_rate() - 0.2).abs() < 1e-6);
        param.try_set_learning_rate_scale(0.5).unwrap();
        assert!((param.learning_rate_scale() - 0.5).abs() < 1e-6);
        assert!((param.hypergrad().unwrap().learning_rate() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn frozen_parameter_ignores_accumulators_but_loads_values() {
        let mut param = Parameter::new("gate", Tensor::zeros(1, 2).unwrap());
        param.set_trainable(false);
        assert!(!param.is_trainable());

        let update = Tensor::from_vec(1, 2, vec![1.0, -1.0]).unwrap();
        param.accumulate_euclidean(&update).unwrap();
        assert!(param.gradient().is_none());
        let before = param.value().clone();
        param.apply_step(0.1).unwrap();
        assert_eq!(param.value(), &before);

        let loaded = Tensor::from_vec(1, 2, vec![0.5, -0.25]).unwrap();
        param.load_value(&loaded).unwrap();
        assert_eq!(param.value(), &loaded);
    }

    #[test]
    fn named_parameter_freeze_stops_update_path() {
        let mut layer = crate::layers::linear::Linear::new("head", 2, 1).unwrap();
        layer
            .set_parameter_trainable("head::weight", false)
            .unwrap();
        let weight_before = layer.weight().value().clone();
        let bias_before = layer.bias().value().clone();

        let input = Tensor::from_vec(1, 2, vec![1.0, -2.0]).unwrap();
        let grad = Tensor::from_vec(1, 1, vec![3.0]).unwrap();
        layer.backward(&input, &grad).unwrap();
        layer.apply_step(0.1).unwrap();

        assert_eq!(layer.weight().value(), &weight_before);
        assert_ne!(layer.bias().value(), &bias_before);
    }

    #[test]
    fn grouped_trainable_matching_controls_parameter_sets() {
        let mut layer = crate::layers::linear::Linear::new("adapter", 2, 1).unwrap();
        assert_eq!(
            layer
                .set_parameters_trainable_by_prefix("adapter::", false)
                .unwrap(),
            2
        );
        assert!(!layer.weight().is_trainable());
        assert!(!layer.bias().is_trainable());

        assert_eq!(
            layer
                .set_parameters_trainable_by_suffix("::bias", true)
                .unwrap(),
            1
        );
        assert!(!layer.weight().is_trainable());
        assert!(layer.bias().is_trainable());

        assert_eq!(
            layer
                .set_parameters_trainable_by_contains("weight", true)
                .unwrap(),
            1
        );
        assert!(layer.weight().is_trainable());
        assert!(layer.bias().is_trainable());
    }

    #[test]
    fn grouped_trainable_matching_rejects_empty_or_missing_patterns() {
        let mut layer = crate::layers::linear::Linear::new("adapter", 2, 1).unwrap();
        let empty = layer
            .set_parameters_trainable_by_prefix("", false)
            .unwrap_err();
        assert!(matches!(empty, TensorError::IoError { .. }));

        let missing = layer
            .set_parameters_trainable_by_contains("does_not_exist", false)
            .unwrap_err();
        assert!(matches!(missing, TensorError::MissingParameter { .. }));
        assert!(layer.weight().is_trainable());
        assert!(layer.bias().is_trainable());
    }

    #[test]
    fn grouped_learning_rate_scaling_controls_parameter_sets() {
        let mut layer = crate::layers::linear::Linear::new("adapter", 2, 1).unwrap();
        assert_eq!(
            layer
                .scale_parameters_learning_rate_by_prefix("adapter::", 0.5)
                .unwrap(),
            2
        );
        assert!((layer.weight().learning_rate_scale() - 0.5).abs() < 1e-6);
        assert!((layer.bias().learning_rate_scale() - 0.5).abs() < 1e-6);

        assert_eq!(
            layer
                .scale_parameters_learning_rate_by_suffix("::bias", 4.0)
                .unwrap(),
            1
        );
        assert!((layer.weight().learning_rate_scale() - 0.5).abs() < 1e-6);
        assert!((layer.bias().learning_rate_scale() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn grouped_learning_rate_scale_setters_are_idempotent() {
        let mut layer = crate::layers::linear::Linear::new("adapter", 2, 1).unwrap();
        assert_eq!(
            layer
                .set_parameters_learning_rate_scale_by_prefix("adapter::", 3.0)
                .unwrap(),
            2
        );
        assert_eq!(
            layer
                .set_parameters_learning_rate_scale_by_suffix("::bias", 1.5)
                .unwrap(),
            1
        );
        assert_eq!(
            layer
                .set_parameters_learning_rate_scale_by_contains("bias", 1.5)
                .unwrap(),
            1
        );
        assert!((layer.weight().learning_rate_scale() - 3.0).abs() < 1e-6);
        assert!((layer.bias().learning_rate_scale() - 1.5).abs() < 1e-6);
        layer.attach_hypergrad(-1.0, 0.1).unwrap();
        layer
            .set_parameter_learning_rate_scale("adapter::bias", 0.75)
            .unwrap();
        assert!((layer.bias().learning_rate_scale() - 0.75).abs() < 1e-6);
        assert!((layer.bias().hypergrad().unwrap().learning_rate() - 0.075).abs() < 1e-6);
    }

    #[test]
    fn parameter_weight_decay_applies_decoupled_fallback_step() {
        let mut param = Parameter::new("weight", Tensor::from_vec(1, 1, vec![2.0]).unwrap());
        param.try_set_weight_decay(0.1).unwrap();
        param.apply_step(0.5).unwrap();

        assert!((param.value().data()[0] - 1.9).abs() < 1e-6);
        assert!((param.weight_decay() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn parameter_weight_decay_tracks_hypergrad_learning_rate() {
        let mut param = Parameter::new("weight", Tensor::from_vec(1, 1, vec![0.2]).unwrap());
        param.try_set_weight_decay(0.1).unwrap();
        param.attach_hypergrad(-1.0, 0.5).unwrap();
        param.apply_step(0.01).unwrap();

        let expected = (0.2_f32 - 0.2_f32 * 0.5 * 0.1).tanh();
        assert!((param.value().data()[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn grouped_weight_decay_controls_parameter_sets_and_fingerprint() {
        let mut layer = crate::layers::linear::Linear::new("adapter", 2, 1).unwrap();
        let before = layer.training_state_fingerprint().unwrap();

        assert_eq!(
            layer
                .set_parameters_weight_decay_by_prefix("adapter::", 0.01)
                .unwrap(),
            2
        );
        assert_eq!(
            layer
                .set_parameters_weight_decay_by_suffix("::bias", 0.0)
                .unwrap(),
            1
        );
        assert_eq!(
            layer
                .set_parameters_weight_decay_by_contains("weight", 0.02)
                .unwrap(),
            1
        );

        assert!((layer.weight().weight_decay() - 0.02).abs() < 1e-6);
        assert_eq!(layer.bias().weight_decay(), 0.0);
        let after = layer.training_state_fingerprint().unwrap();
        assert_ne!(before.hash, after.hash);

        layer
            .set_parameter_weight_decay("adapter::bias", 0.03)
            .unwrap();
        assert!((layer.bias().weight_decay() - 0.03).abs() < 1e-6);
    }

    #[test]
    fn grouped_weight_decay_rejects_invalid_or_missing_patterns() {
        let mut layer = crate::layers::linear::Linear::new("adapter", 2, 1).unwrap();
        let invalid = layer
            .set_parameters_weight_decay_by_contains("weight", -0.01)
            .unwrap_err();
        assert!(matches!(invalid, TensorError::NonFiniteValue { .. }));
        let missing = layer
            .set_parameters_weight_decay_by_contains("does_not_exist", 0.01)
            .unwrap_err();
        assert!(matches!(missing, TensorError::MissingParameter { .. }));
        let empty = layer
            .set_parameters_weight_decay_by_prefix("", 0.01)
            .unwrap_err();
        assert!(matches!(empty, TensorError::IoError { .. }));
        assert_eq!(layer.weight().weight_decay(), 0.0);
        assert_eq!(layer.bias().weight_decay(), 0.0);
    }

    #[test]
    fn grouped_learning_rate_scaling_rejects_invalid_or_missing_patterns() {
        let mut layer = crate::layers::linear::Linear::new("adapter", 2, 1).unwrap();
        let invalid = layer
            .scale_parameters_learning_rate_by_contains("weight", 0.0)
            .unwrap_err();
        assert!(matches!(
            invalid,
            TensorError::NonPositiveLearningRate { .. }
        ));
        let missing = layer
            .scale_parameters_learning_rate_by_contains("does_not_exist", 1.5)
            .unwrap_err();
        assert!(matches!(missing, TensorError::MissingParameter { .. }));
    }

    #[test]
    fn movement_audit_reports_trainable_and_frozen_deltas() {
        let mut layer = crate::layers::linear::Linear::new("audit", 2, 1).unwrap();
        layer
            .set_parameter_trainable("audit::weight", false)
            .unwrap();
        let before = layer.state_dict().unwrap();

        let input = Tensor::from_vec(1, 2, vec![1.0, -2.0]).unwrap();
        let grad = Tensor::from_vec(1, 1, vec![3.0]).unwrap();
        layer.backward(&input, &grad).unwrap();
        layer.apply_step(0.1).unwrap();

        let report = layer.audit_parameter_movement(&before, 0.0).unwrap();
        assert_eq!(report.status(), "ok");
        assert!(report.frozen_stable());
        assert!(report.trainable_movement_observed());
        assert_eq!(report.frozen_unchanged, 1);
        assert_eq!(report.trainable_changed, 1);
        assert_eq!(report.max_frozen_abs_delta, 0.0);
    }

    #[test]
    fn state_fingerprint_is_stable_across_hashmap_order() {
        let layer = crate::layers::linear::Linear::new("fingerprint", 2, 1).unwrap();
        let state = layer.state_dict().unwrap();
        let mut reordered = HashMap::new();
        for name in ["fingerprint::bias", "fingerprint::weight"] {
            reordered.insert(name.to_string(), state[name].clone());
        }
        assert_eq!(
            fingerprint_state_dict(&state),
            fingerprint_state_dict(&reordered)
        );
    }

    #[test]
    fn subset_checked_load_ignores_extra_source_keys() {
        let source_head = crate::layers::linear::Linear::new("head", 2, 1).unwrap();
        let source_embed = crate::layers::linear::Linear::new("embed", 3, 2).unwrap();
        let mut source_state = source_head.state_dict().unwrap();
        source_state.extend(source_embed.state_dict().unwrap());

        let mut target = crate::layers::linear::Linear::new("head", 2, 1).unwrap();
        let load = target
            .load_state_dict_subset_checked(&source_state)
            .unwrap();
        assert!(load.matched);
        assert_eq!(load.source.parameters, 2);
        assert_eq!(load.loaded, target.state_fingerprint().unwrap());

        let mut missing_bias = source_state;
        missing_bias.remove("head::bias");
        let err = target
            .load_state_dict_subset_checked(&missing_bias)
            .unwrap_err();
        assert_eq!(
            err,
            TensorError::MissingParameter {
                name: "head::bias".to_string()
            }
        );
    }

    #[test]
    fn state_dict_compatibility_reports_subset_readiness_without_loading() {
        let source_head = crate::layers::linear::Linear::new("head", 2, 1).unwrap();
        let source_embed = crate::layers::linear::Linear::new("embed", 3, 2).unwrap();
        let mut source_state = source_head.state_dict().unwrap();
        source_state.extend(source_embed.state_dict().unwrap());

        let target = crate::layers::linear::Linear::new("head", 2, 1).unwrap();
        let before = target.state_fingerprint().unwrap();
        let report = target.state_dict_compatibility(&source_state).unwrap();
        let after = target.state_fingerprint().unwrap();

        assert!(report.compatible);
        assert_eq!(before, after);
        assert_eq!(report.expected_parameters, 2);
        assert_eq!(report.source_parameters, 4);
        assert_eq!(report.matched, 2);
        assert_eq!(report.missing, 0);
        assert_eq!(report.shape_mismatched, 0);
        assert_eq!(report.extra, 2);
        assert_eq!(report.matched_subset.parameters, 2);
        assert_eq!(
            report
                .entries
                .iter()
                .filter(|entry| entry.status == StateCompatibilityStatus::Extra)
                .count(),
            2
        );
    }

    #[test]
    fn state_dict_compatibility_reports_missing_and_shape_mismatch() {
        let mut source_state = HashMap::new();
        source_state.insert("head::weight".to_string(), Tensor::zeros(3, 1).unwrap());
        source_state.insert("unrelated::bias".to_string(), Tensor::zeros(1, 1).unwrap());
        let target = crate::layers::linear::Linear::new("head", 2, 1).unwrap();

        let report = target.state_dict_compatibility(&source_state).unwrap();
        assert!(!report.compatible);
        assert_eq!(report.expected_parameters, 2);
        assert_eq!(report.source_parameters, 2);
        assert_eq!(report.matched, 0);
        assert_eq!(report.missing, 1);
        assert_eq!(report.shape_mismatched, 1);
        assert_eq!(report.extra, 1);

        let weight = report
            .entries
            .iter()
            .find(|entry| entry.name == "head::weight")
            .unwrap();
        assert_eq!(weight.status, StateCompatibilityStatus::ShapeMismatch);
        assert_eq!(weight.expected_shape, Some((2, 1)));
        assert_eq!(weight.source_shape, Some((3, 1)));

        let bias = report
            .entries
            .iter()
            .find(|entry| entry.name == "head::bias")
            .unwrap();
        assert_eq!(bias.status, StateCompatibilityStatus::Missing);
        assert_eq!(bias.expected_shape, Some((1, 1)));
        assert_eq!(bias.source_shape, None);
    }

    #[test]
    fn mapped_state_dict_compatibility_bridges_external_keys() {
        let source = crate::layers::linear::Linear::new("source_head", 2, 1).unwrap();
        let source_state = source.state_dict().unwrap();
        let mut external_state = HashMap::new();
        external_state.insert(
            "external.lm_head.weight".to_string(),
            source_state["source_head::weight"].clone(),
        );
        external_state.insert(
            "external.lm_head.bias".to_string(),
            source_state["source_head::bias"].clone(),
        );
        external_state.insert("external.unused".to_string(), Tensor::zeros(1, 1).unwrap());
        let key_map = HashMap::from([
            (
                "external.lm_head.weight".to_string(),
                "head::weight".to_string(),
            ),
            (
                "external.lm_head.bias".to_string(),
                "head::bias".to_string(),
            ),
        ]);

        let mut target = crate::layers::linear::Linear::new("head", 2, 1).unwrap();
        let plain = target.state_dict_compatibility(&external_state).unwrap();
        assert!(!plain.compatible);
        assert_eq!(plain.missing, 2);
        assert_eq!(plain.extra, 3);

        let mapped = target
            .state_dict_compatibility_with_key_map(&external_state, &key_map)
            .unwrap();
        assert!(mapped.compatible);
        assert_eq!(mapped.matched, 2);
        assert_eq!(mapped.missing, 0);
        assert_eq!(mapped.shape_mismatched, 0);
        assert_eq!(mapped.extra, 1);

        let load = target
            .load_state_dict_subset_mapped_checked(&external_state, &key_map)
            .unwrap();
        assert!(load.matched);
        assert_eq!(load.loaded, target.state_fingerprint().unwrap());
    }

    #[test]
    fn state_dict_key_rules_transpose_external_weights() {
        let external_weight = Tensor::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let external_bias = Tensor::from_vec(1, 3, vec![0.5, -0.25, 0.75]).unwrap();
        let external_state = HashMap::from([
            ("external.weight".to_string(), external_weight.clone()),
            ("external.bias".to_string(), external_bias.clone()),
        ]);
        let rules = HashMap::from([
            (
                "external.weight".to_string(),
                StateKeyMapRule::with_transform("head::weight", StateTensorTransform::Transpose),
            ),
            (
                "external.bias".to_string(),
                StateKeyMapRule::new("head::bias"),
            ),
        ]);
        let mut target = crate::layers::linear::Linear::new("head", 2, 3).unwrap();

        let plain = target.state_dict_compatibility(&external_state).unwrap();
        assert!(!plain.compatible);
        let adapted = target
            .state_dict_compatibility_with_key_rules(&external_state, &rules)
            .unwrap();
        assert!(adapted.compatible);
        let bias_entry = adapted
            .entries
            .iter()
            .find(|entry| entry.name == "head::bias")
            .unwrap();
        assert_eq!(bias_entry.source_shape, Some((1, 3)));
        let weight_entry = adapted
            .entries
            .iter()
            .find(|entry| entry.name == "head::weight")
            .unwrap();
        assert_eq!(weight_entry.source_name.as_deref(), Some("external.weight"));
        assert_eq!(weight_entry.transform, StateTensorTransform::Transpose);
        assert_eq!(weight_entry.original_source_shape, Some((3, 2)));
        assert_eq!(weight_entry.source_shape, Some((2, 3)));

        let load = target
            .load_state_dict_subset_adapted_checked(&external_state, &rules)
            .unwrap();
        assert!(load.matched);
        let loaded = target.state_dict().unwrap();
        assert_eq!(loaded["head::weight"], external_weight.transpose());
        assert_eq!(loaded["head::bias"], external_bias);
    }

    #[test]
    fn state_dict_key_rules_copy_overlap_and_zero_pad_shapes() {
        let external_weight = Tensor::from_vec(2, 1, vec![1.0, 2.0]).unwrap();
        let external_bias = Tensor::from_vec(1, 1, vec![0.5]).unwrap();
        let external_state = HashMap::from([
            ("external.weight".to_string(), external_weight),
            ("external.bias".to_string(), external_bias),
        ]);
        let rules = HashMap::from([
            (
                "external.weight".to_string(),
                StateKeyMapRule::with_transform(
                    "head::weight",
                    StateTensorTransform::CopyOverlapZeros,
                ),
            ),
            (
                "external.bias".to_string(),
                StateKeyMapRule::with_transform(
                    "head::bias",
                    StateTensorTransform::CopyOverlapZeros,
                ),
            ),
        ]);
        let mut target = crate::layers::linear::Linear::new("head", 3, 2).unwrap();
        let adapted = target
            .state_dict_compatibility_with_key_rules(&external_state, &rules)
            .unwrap();
        assert!(adapted.compatible);
        let weight_entry = adapted
            .entries
            .iter()
            .find(|entry| entry.name == "head::weight")
            .unwrap();
        assert_eq!(
            weight_entry.transform,
            StateTensorTransform::CopyOverlapZeros
        );
        assert_eq!(weight_entry.original_source_shape, Some((2, 1)));
        assert_eq!(weight_entry.source_shape, Some((3, 2)));
        let bias_entry = adapted
            .entries
            .iter()
            .find(|entry| entry.name == "head::bias")
            .unwrap();
        assert_eq!(bias_entry.original_source_shape, Some((1, 1)));
        assert_eq!(bias_entry.source_shape, Some((1, 2)));

        let load = target
            .load_state_dict_subset_adapted_checked(&external_state, &rules)
            .unwrap();
        assert!(load.matched);
        let loaded = target.state_dict().unwrap();
        assert_eq!(
            loaded["head::weight"],
            Tensor::from_vec(3, 2, vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0]).unwrap()
        );
        assert_eq!(
            loaded["head::bias"],
            Tensor::from_vec(1, 2, vec![0.5, 0.0]).unwrap()
        );
    }

    #[test]
    fn state_dict_key_remap_rejects_duplicate_targets() {
        let state = HashMap::from([
            (
                "external.weight.a".to_string(),
                Tensor::zeros(2, 1).unwrap(),
            ),
            (
                "external.weight.b".to_string(),
                Tensor::zeros(2, 1).unwrap(),
            ),
        ]);
        let key_map = HashMap::from([
            ("external.weight.a".to_string(), "head::weight".to_string()),
            ("external.weight.b".to_string(), "head::weight".to_string()),
        ]);

        let err = remap_state_dict_keys(&state, &key_map).unwrap_err();
        assert!(matches!(err, TensorError::IoError { .. }));
    }

    #[test]
    fn training_state_fingerprint_tracks_freeze_lr_and_hypergrad() {
        let mut layer = crate::layers::linear::Linear::new("resume", 2, 1).unwrap();
        let initial = layer.training_state_fingerprint().unwrap();
        assert_eq!(initial.parameters, 2);
        assert_eq!(initial.trainable, 2);
        assert_eq!(initial.frozen, 0);
        assert_eq!(initial.hypergrad_tapes, 0);

        layer
            .set_parameters_trainable_by_suffix("::weight", false)
            .unwrap();
        layer
            .scale_parameters_learning_rate_by_suffix("::bias", 1.25)
            .unwrap();
        let tuned = layer.training_state_fingerprint().unwrap();
        assert_ne!(initial.hash, tuned.hash);
        assert_eq!(tuned.trainable, 1);
        assert_eq!(tuned.frozen, 1);

        layer.attach_hypergrad(-1.0, 0.05).unwrap();
        let prepared = layer.training_state_fingerprint().unwrap();
        assert_ne!(tuned.hash, prepared.hash);
        assert_eq!(prepared.hypergrad_tapes, 2);
    }

    #[test]
    fn checked_state_load_reports_matching_source() {
        let source = crate::layers::linear::Linear::new("checked", 2, 1).unwrap();
        let state = source.state_dict().unwrap();
        let mut target = crate::layers::linear::Linear::new("checked", 2, 1).unwrap();
        let input = Tensor::from_vec(1, 2, vec![1.0, 1.0]).unwrap();
        let grad = Tensor::from_vec(1, 1, vec![2.0]).unwrap();
        target.backward(&input, &grad).unwrap();
        target.apply_step(0.1).unwrap();

        let report = target.load_state_dict_checked(&state).unwrap();
        assert!(report.matched);
        assert_eq!(report.source, report.loaded);
        assert_eq!(report.loaded.parameters, 2);
    }
}
