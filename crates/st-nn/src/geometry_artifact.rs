//! Portable, typed geometry weights and recipes; not a full trainer checkpoint.
//!
//! This contract shares the existing native kernel validators. Python owns HF
//! placement/transport; a model identity supplied by the caller is not proof of
//! the identity of a live, possibly mutated model. Hashes are not authentication.

use crate::WaveGateKernel;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use st_core::dynamics::topos_resonator::{ToposResonatorConfig, ToposResonatorOperator};
use st_core::theory::microlocal::EllipticWarp;
use st_frac::learning::{FractionalGlAngleChart, FractionalGlKernel, FractionalGlRecipe};
use st_tensor::topos::OpenCartesianTopos;
use std::collections::BTreeMap;

pub const GEOMETRY_ADAPTER_ARTIFACT_SCHEMA: &str = "spiraltorch.geometry_adapter_artifact.v1";
pub const GEOMETRY_ADAPTER_MAX_BYTES: usize = 64 * 1024 * 1024;
pub const GEOMETRY_ADAPTER_MAX_VALUES: usize = 1_048_576;
pub const GEOMETRY_ADAPTER_MAX_KERNEL_TAPS: usize = 65_536;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaveRecipe {
    pub curvature: f32,
    pub saturation: f32,
    pub porosity: f32,
    pub max_values: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToposRecipe {
    pub coupling: f32,
    pub iterations: usize,
    pub saturation: f32,
    pub porosity: f32,
    pub max_values: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EllipticRecipe {
    pub curvature_radius: f32,
    pub sheet_count: usize,
    pub spin_harmonics: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FractionalRecipe {
    pub kernel_len: usize,
    pub step: f32,
    pub max_values: usize,
    pub max_products: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "schema", deny_unknown_fields)]
pub enum GeometryAdapterRecipe {
    #[serde(rename = "spiraltorch.channel_affine_adapter.v1")]
    ChannelAffine {
        features: usize,
        strength: f64,
        max_values: usize,
    },
    #[serde(rename = "spiraltorch.wave_gate_adapter.v1")]
    Wave {
        features: usize,
        strength: f64,
        kernel: WaveRecipe,
    },
    #[serde(rename = "spiraltorch.wave_gate_adapter.v2")]
    WaveRadius {
        features: usize,
        strength: f64,
        kernel: WaveRecipe,
        learnable_radius: bool,
    },
    #[serde(rename = "spiraltorch.topos_resonator_adapter.v1")]
    Topos {
        features: usize,
        strength: f64,
        kernel: ToposRecipe,
    },
    #[serde(rename = "spiraltorch.elliptic_residual_adapter.v1")]
    Elliptic {
        features: usize,
        strength: f64,
        warp: EllipticRecipe,
    },
    #[serde(rename = "spiraltorch.elliptic_causal_residual_adapter.v1")]
    EllipticCausal {
        features: usize,
        strength: f64,
        warp: EllipticRecipe,
    },
    #[serde(rename = "spiraltorch.elliptic_gated_causal_residual_adapter.v1")]
    EllipticGated {
        features: usize,
        strength: f64,
        warp: EllipticRecipe,
    },
    #[serde(rename = "spiraltorch.elliptic_anchored_residual_adapter.v1")]
    EllipticAnchored {
        features: usize,
        strength: f64,
        warp: EllipticRecipe,
    },
    #[serde(rename = "spiraltorch.fractional_memory_adapter.v1")]
    Fractional {
        features: usize,
        strength: f64,
        kernel: FractionalRecipe,
    },
    #[serde(rename = "spiraltorch.fractional_history_adapter.v1")]
    History {
        features: usize,
        strength: f64,
        kernel: FractionalRecipe,
    },
    #[serde(rename = "spiraltorch.fractional_l2_history_adapter.v1")]
    HistoryL2 {
        features: usize,
        strength: f64,
        kernel: FractionalRecipe,
        gain: f32,
    },
    #[serde(rename = "spiraltorch.fractional_gain_history_adapter.v1")]
    HistoryGain {
        features: usize,
        strength: f64,
        kernel: FractionalRecipe,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lag_window: Option<[usize; 2]>,
    },
    #[serde(rename = "spiraltorch.fractional_angle_gain_history_adapter.v1")]
    HistoryAngle {
        features: usize,
        strength: f64,
        kernel: FractionalRecipe,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lag_window: Option<[usize; 2]>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryAdapterTensor {
    pub name: String,
    /// Scalars have shape []. Values are row-major, owned finite f32.
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub requires_grad: bool,
    pub role: GeometryAdapterTensorRole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryAdapterTensorRole {
    Parameter,
    Buffer,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryAdapterPlacement {
    pub path: String,
    pub recipe: GeometryAdapterRecipe,
    pub training: bool,
    pub tensors: Vec<GeometryAdapterTensor>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeometryAdapterPolicy {
    /// Complete unpadded single-document prefixes, f32, no KV cache. Geometry
    /// kernels currently run on the Rust CPU path; no resident-GPU promise.
    #[serde(rename = "full_prefix_f32_no_cache.v1")]
    FullPrefixF32NoCache,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryAdapterRecord {
    pub schema: String,
    pub base_model_identity: String,
    pub policy: GeometryAdapterPolicy,
    pub training: bool,
    /// Order fixes placement and optimizer parameter order.
    pub placements: Vec<GeometryAdapterPlacement>,
}

/// Validated immutable state, with no pickle, code import, device or file access.
#[derive(Clone, Debug)]
pub struct GeometryAdapterArtifact {
    record: GeometryAdapterRecord,
    json: String,
    artifact_id: String,
}

fn identity(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|v| {
        v.len() == 64
            && v.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

impl GeometryAdapterArtifact {
    /// Validate a new locally constructed record, then issue its exact-byte ID.
    pub fn create(mut record: GeometryAdapterRecord) -> Result<Self> {
        record.validate()?;
        record.sort_tensors();
        let json = serde_json::to_string(&record)?;
        Self::decode(json.as_bytes(), &record.base_model_identity)
    }

    /// Binding entry point. This constructs a new artifact, not a trusted load.
    pub fn create_json(input: &[u8]) -> Result<Self> {
        if input.len() > GEOMETRY_ADAPTER_MAX_BYTES {
            return Err("geometry artifact exceeds byte budget".into());
        }
        Self::create(serde_json::from_slice(input)?)
    }

    /// Check a separately pinned exact-byte digest BEFORE deserialization.
    pub fn from_json(
        input: &[u8],
        expected_id: &str,
        expected_base_model_identity: &str,
    ) -> Result<Self> {
        if input.len() > GEOMETRY_ADAPTER_MAX_BYTES {
            return Err("geometry artifact exceeds byte budget".into());
        }
        if !identity(expected_id) || format!("sha256:{:x}", Sha256::digest(input)) != expected_id {
            return Err("geometry artifact differs from the pinned identity".into());
        }
        Self::decode(input, expected_base_model_identity)
    }

    fn decode(input: &[u8], expected_base: &str) -> Result<Self> {
        if input.len() > GEOMETRY_ADAPTER_MAX_BYTES {
            return Err("geometry artifact exceeds byte budget".into());
        }
        let mut record: GeometryAdapterRecord = serde_json::from_slice(input)?;
        record.validate()?;
        record.sort_tensors();
        if record.base_model_identity != expected_base {
            return Err("geometry artifact belongs to a different base model identity".into());
        }
        Ok(Self {
            record,
            json: std::str::from_utf8(input)?.to_owned(),
            artifact_id: format!("sha256:{:x}", Sha256::digest(input)),
        })
    }

    pub fn to_json(&self) -> &str {
        &self.json
    }
    /// Typed normalized state for reconstruction; not the bytes used for the ID.
    pub fn state_json(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.record)?)
    }
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }
    /// Resume compatibility without weights or train/eval mode. Includes exact
    /// ordered placement, recipes, tensor shapes/roles/trainability and base ID.
    pub fn layout_id(&self) -> Result<String> {
        let placements: Vec<_> = self
            .record
            .placements
            .iter()
            .map(|p| {
                let tensors: Vec<_> = p
                    .tensors
                    .iter()
                    .map(|t| (&t.name, &t.shape, t.role, t.requires_grad))
                    .collect();
                (&p.path, &p.recipe, tensors)
            })
            .collect();
        let bytes = serde_json::to_vec(&(
            "spiraltorch.geometry_adapter_layout.v1",
            &self.record.base_model_identity,
            self.record.policy,
            placements,
        ))?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }
    pub fn record(&self) -> &GeometryAdapterRecord {
        &self.record
    }
}

impl GeometryAdapterRecord {
    fn sort_tensors(&mut self) {
        // Names define tensor ownership; only placement order is semantic.
        for placement in &mut self.placements {
            placement.tensors.sort_by(|a, b| a.name.cmp(&b.name));
        }
    }

    fn validate(&self) -> Result<()> {
        if self.schema != GEOMETRY_ADAPTER_ARTIFACT_SCHEMA || !identity(&self.base_model_identity) {
            return Err("invalid geometry artifact schema or base model identity".into());
        }
        if self.placements.is_empty() || self.placements.len() > 128 {
            return Err("geometry artifact needs 1..=128 placements".into());
        }
        let mut total = 0usize;
        let mut taps = 0usize;
        for (i, placement) in self.placements.iter().enumerate() {
            let path = &placement.path;
            if path.len() > 1024
                || path
                    .split('.')
                    .any(|p| p.is_empty() || p.chars().any(char::is_control))
            {
                return Err("invalid geometry placement path".into());
            }
            for previous in &self.placements[..i] {
                if path == &previous.path
                    || path.starts_with(&format!("{}.", previous.path))
                    || previous.path.starts_with(&format!("{path}."))
                {
                    return Err("duplicate or nested geometry placements".into());
                }
            }
            use GeometryAdapterRecipe::*;
            if let Fractional { kernel, .. }
            | History { kernel, .. }
            | HistoryL2 { kernel, .. }
            | HistoryGain { kernel, .. }
            | HistoryAngle { kernel, .. } = &placement.recipe
            {
                if kernel.kernel_len > GEOMETRY_ADAPTER_MAX_KERNEL_TAPS {
                    return Err("fractional artifact kernel tap budget exceeded".into());
                }
                taps += kernel.kernel_len;
                if taps > GEOMETRY_ADAPTER_MAX_VALUES {
                    return Err("fractional artifact total tap budget exceeded".into());
                }
            }
            total = total
                .checked_add(placement.validate()?)
                .ok_or("geometry value budget overflow")?;
            if total > GEOMETRY_ADAPTER_MAX_VALUES {
                return Err("geometry value budget exceeded".into());
            }
        }
        Ok(())
    }
}

impl GeometryAdapterPlacement {
    fn tensor(&self, name: &str) -> &[f32] {
        // Used only after exact tensor names/shapes have been validated.
        &self.tensors.iter().find(|t| t.name == name).unwrap().values
    }

    fn validate(&self) -> Result<usize> {
        use GeometryAdapterRecipe::*;
        let (features, strength) = match &self.recipe {
            ChannelAffine {
                features, strength, ..
            }
            | Wave {
                features, strength, ..
            }
            | WaveRadius {
                features, strength, ..
            }
            | Topos {
                features, strength, ..
            }
            | Elliptic {
                features, strength, ..
            }
            | EllipticCausal {
                features, strength, ..
            }
            | EllipticGated {
                features, strength, ..
            }
            | EllipticAnchored {
                features, strength, ..
            }
            | Fractional {
                features, strength, ..
            }
            | History {
                features, strength, ..
            }
            | HistoryL2 {
                features, strength, ..
            }
            | HistoryGain {
                features, strength, ..
            }
            | HistoryAngle {
                features, strength, ..
            } => (*features, *strength),
        };
        if !(1..=65_536).contains(&features)
            || !strength.is_finite()
            || !(0.0..=1.0).contains(&strength)
        {
            return Err("invalid geometry features or residual strength".into());
        }
        let mut shapes = BTreeMap::new();
        match &self.recipe {
            ChannelAffine { .. } | Wave { .. } | WaveRadius { .. } => {
                shapes.insert("gate", vec![features]);
                shapes.insert("bias", vec![features]);
            }
            Topos { .. } => {
                shapes.insert("gate", vec![features]);
            }
            Elliptic { .. }
            | EllipticCausal { .. }
            | EllipticGated { .. }
            | EllipticAnchored { .. } => {
                shapes.insert("orientation.weight", vec![2, features]);
                shapes.insert("orientation.bias", vec![2]);
                shapes.insert("readout.weight", vec![features, 9]);
            }
            _ => {
                shapes.insert("gate", vec![features]);
                if !matches!(self.recipe, Fractional { .. }) {
                    shapes.insert("local_gate", vec![features]);
                }
                shapes.insert(
                    if matches!(self.recipe, HistoryAngle { .. }) {
                        "history_angle"
                    } else {
                        "log_alpha"
                    },
                    vec![],
                );
            }
        }
        match &self.recipe {
            WaveRadius { .. } => {
                shapes.insert("log_radius", vec![]);
            }
            EllipticGated { .. } | EllipticAnchored { .. } => {
                shapes.insert("raw_mix", vec![]);
            }
            HistoryGain { .. } | HistoryAngle { .. } => {
                shapes.insert("log_gain", vec![]);
            }
            _ => {}
        }
        let mut total = 0;
        for tensor in &self.tensors {
            let fixed_radius = tensor.name == "log_radius"
                && matches!(
                    self.recipe,
                    WaveRadius {
                        learnable_radius: false,
                        ..
                    }
                );
            let expected_role = if fixed_radius {
                GeometryAdapterTensorRole::Buffer
            } else {
                GeometryAdapterTensorRole::Parameter
            };
            if tensor.role != expected_role {
                return Err("geometry tensor role differs from the built-in recipe".into());
            }
            let shape = shapes
                .remove(tensor.name.as_str())
                .ok_or("unknown or duplicate geometry tensor")?;
            if tensor.shape != shape
                || tensor.values.len() != shape.iter().product::<usize>()
                || tensor.values.iter().any(|v| !v.is_finite())
            {
                return Err("invalid geometry tensor shape or non-finite values".into());
            }
            if tensor.name == "log_radius"
                && matches!(
                    self.recipe,
                    WaveRadius {
                        learnable_radius: false,
                        ..
                    }
                )
                && tensor.requires_grad
            {
                return Err("fixed radius buffer cannot require gradients".into());
            }
            total += tensor.values.len();
        }
        if !shapes.is_empty() {
            return Err("missing geometry tensors".into());
        }
        match &self.recipe {
            ChannelAffine { max_values, .. } => {
                crate::ChannelAffineKernel::new(*max_values)?.forward(
                    &[],
                    self.tensor("gate"),
                    self.tensor("bias"),
                    0,
                    features,
                )?;
            }
            Wave { kernel, .. } | WaveRadius { kernel, .. } => {
                u32::try_from(kernel.max_values).map_err(|_| "wave budget exceeds portable u32")?;
                let native = WaveGateKernel::new(
                    kernel.curvature,
                    kernel.saturation,
                    kernel.porosity,
                    kernel.max_values,
                )?;
                if let WaveRadius { .. } = self.recipe {
                    native.forward_with_log_radius(
                        &[],
                        self.tensor("gate"),
                        self.tensor("bias"),
                        0,
                        features,
                        self.tensor("log_radius")[0],
                    )?;
                } else {
                    native.forward(&[], self.tensor("gate"), self.tensor("bias"), 0, features)?;
                }
            }
            Topos { kernel, .. } => {
                u32::try_from(kernel.max_values)
                    .map_err(|_| "topos budget exceeds portable u32")?;
                let config = ToposResonatorConfig::new(kernel.coupling, kernel.iterations)?;
                let depth = kernel
                    .iterations
                    .checked_add(1)
                    .ok_or("topos iteration overflow")?;
                let topos = OpenCartesianTopos::new(
                    -1.0,
                    1e-6,
                    kernel.saturation,
                    depth,
                    kernel.max_values,
                )?
                .with_porosity(kernel.porosity)?;
                ToposResonatorOperator::new(config, topos)?;
                if features > kernel.max_values {
                    return Err("topos feature budget exceeded".into());
                }
            }
            Elliptic { warp, .. }
            | EllipticCausal { warp, .. }
            | EllipticGated { warp, .. }
            | EllipticAnchored { warp, .. } => {
                u32::try_from(warp.spin_harmonics)
                    .map_err(|_| "elliptic harmonics exceed portable u32")?;
                EllipticWarp::for_learning(
                    warp.curvature_radius,
                    warp.sheet_count,
                    warp.spin_harmonics,
                )?;
            }
            Fractional { kernel, .. }
            | History { kernel, .. }
            | HistoryL2 { kernel, .. }
            | HistoryGain { kernel, .. }
            | HistoryAngle { kernel, .. } => {
                for value in [kernel.kernel_len, kernel.max_values, kernel.max_products] {
                    u32::try_from(value).map_err(|_| "fractional budget exceeds portable u32")?;
                }
                let native = FractionalGlKernel::new(
                    kernel.kernel_len,
                    kernel.step,
                    kernel.max_values,
                    kernel.max_products,
                )?;
                if features > kernel.max_values
                    || features
                        .checked_mul(kernel.kernel_len)
                        .is_none_or(|n| n > kernel.max_products)
                {
                    return Err("fractional feature/product budget exceeded".into());
                }
                let alpha = if matches!(self.recipe, HistoryAngle { .. }) {
                    FractionalGlAngleChart::new(self.tensor("history_angle")[0])?.alpha()
                } else {
                    let alpha = self.tensor("log_alpha")[0].exp();
                    if !alpha.is_finite() || alpha <= 0.0 {
                        return Err("invalid fractional log order".into());
                    }
                    alpha
                };
                let mode = match self.recipe {
                    Fractional { .. } => FractionalGlRecipe::Memory,
                    History { .. } => FractionalGlRecipe::History,
                    HistoryL2 { gain, .. } => {
                        FractionalGlRecipe::NormalizedHistory { gain, lags: None }
                    }
                    HistoryGain { lag_window, .. } | HistoryAngle { lag_window, .. } => {
                        let gain =
                            FractionalGlKernel::gain_from_log_gain(self.tensor("log_gain")[0])?;
                        FractionalGlRecipe::NormalizedHistory {
                            gain,
                            lags: lag_window.map(|[start, end]| start..end),
                        }
                    }
                    _ => unreachable!(),
                };
                native.validate_recipe(alpha, mode)?;
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests;
