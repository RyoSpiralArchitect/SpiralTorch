//! Artifact-bound optimizer ownership; arithmetic stays in st-kernel-contracts.
use super::{GeometryAdapterArtifact, GeometryAdapterTensorRole, NativeGeometryBatch};
use super::{NativeGeometryRuntime, ParameterGradients, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};
use st_kernel_contracts::adamw::{AdamWSlot, AdamWState};
pub use st_kernel_contracts::gradient_clip::GlobalNormClip;

const SCHEMA: &str = "spiraltorch.native_geometry_adamw.v1";
const MAX_BYTES: usize = 64 * 1024 * 1024;

/// Immutable optimizer snapshot paired with one exact geometry artifact.
/// Missing placements do not decay or advance their moments/clocks. Frozen
/// parameters and buffers have no slots; zero rate freezes all state.
#[derive(Clone, Debug)]
pub struct NativeGeometryAdamW {
    geometry_id: String,
    keys: Vec<(String, String)>,
    state: AdamWState,
    revision: u64,
    clip: Option<GlobalNormClip>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    geometry_artifact_id: String,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_gradient_norm: Option<f32>,
    revision: u64,
    slots: Vec<Slot>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Slot {
    placement: String,
    parameter: String,
    shape: Vec<usize>,
    first: Vec<f32>,
    second: Vec<f32>,
    steps: u32,
}

fn identity(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn validate_placement_clocks(
    keys: &[(String, String)],
    clocks: &[u32],
    revision: u64,
) -> Result<()> {
    if keys.len() != clocks.len() {
        return Err("native geometry AdamW clock count mismatch".into());
    }
    let mut placements = std::collections::BTreeMap::new();
    for ((path, _), &clock) in keys.iter().zip(clocks) {
        if u64::from(clock) > revision {
            return Err("native geometry AdamW clock exceeds revision".into());
        }
        if placements
            .insert(path, clock)
            .is_some_and(|old| old != clock)
        {
            return Err("native geometry AdamW placement clocks differ".into());
        }
    }
    let accepted: u128 = placements.values().map(|&n| u128::from(n)).sum();
    if u128::from(revision) > accepted {
        return Err("native geometry AdamW revision exceeds accepted placement updates".into());
    }
    Ok(())
}

impl NativeGeometryAdamW {
    pub fn new(
        artifact: &GeometryAdapterArtifact,
        config: AdamWConfig,
        clip: Option<GlobalNormClip>,
    ) -> Result<Self> {
        let mut keys = Vec::new();
        let mut slots = Vec::new();
        for placement in &artifact.record().placements {
            for p in &placement.tensors {
                if p.role == GeometryAdapterTensorRole::Parameter && p.requires_grad {
                    keys.push((placement.path.clone(), p.name.clone()));
                    slots.push(AdamWSlot {
                        shape: p.shape.clone(),
                        first: vec![0.; p.values.len()],
                        second: vec![0.; p.values.len()],
                        steps: 0,
                    });
                }
            }
        }
        if slots.is_empty() {
            return Err("native geometry AdamW requires trainable parameters".into());
        }
        Ok(Self {
            geometry_id: artifact.artifact_id().into(),
            keys,
            state: AdamWState { config, slots },
            revision: 0,
            clip,
        })
    }

    pub fn geometry_artifact_id(&self) -> &str {
        &self.geometry_id
    }
    /// Accepted nonzero-rate updates since optimizer creation, not model lineage.
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn state(&self) -> &AdamWState {
        &self.state
    }
    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.clip.map(GlobalNormClip::max_norm)
    }

    /// Check a caller's uniform-all-placement update history, while permitting
    /// independently aged placements at that caller's initial boundary.
    pub fn validate_uniform_continuation(
        &self,
        initial_revision: u64,
        initial_clocks: &[u32],
        accepted_steps: u64,
    ) -> Result<()> {
        validate_placement_clocks(&self.keys, initial_clocks, initial_revision)?;
        if initial_revision.checked_add(accepted_steps) != Some(self.revision)
            || self
                .state
                .slots
                .iter()
                .zip(initial_clocks)
                .any(|(slot, &initial)| {
                    u64::from(initial).checked_add(accepted_steps) != Some(u64::from(slot.steps))
                })
        {
            return Err("native geometry AdamW uniform continuation clocks differ".into());
        }
        Ok(())
    }

    /// Canonical checkpoint bytes. Save these together with the geometry artifact
    /// and separately pinned IDs. Learning-rate schedules remain caller-owned.
    pub fn to_json(&self) -> Result<String> {
        let c = self.state.config;
        let checkpoint = Checkpoint {
            schema: SCHEMA.into(),
            geometry_artifact_id: self.geometry_id.clone(),
            beta1: c.beta1(),
            beta2: c.beta2(),
            epsilon: c.epsilon(),
            max_gradient_norm: self.max_gradient_norm(),
            revision: self.revision,
            slots: self
                .keys
                .iter()
                .zip(&self.state.slots)
                .map(|((path, name), slot)| Slot {
                    placement: path.clone(),
                    parameter: name.clone(),
                    shape: slot.shape.clone(),
                    first: slot.first.clone(),
                    second: slot.second.clone(),
                    steps: slot.steps,
                })
                .collect(),
        };
        let json = serde_json::to_string(&checkpoint)?;
        if json.len() > MAX_BYTES {
            return Err("native geometry AdamW checkpoint exceeds byte budget".into());
        }
        Ok(json)
    }

    pub fn checkpoint_id(&self) -> Result<String> {
        Ok(identity(self.to_json()?.as_bytes()))
    }

    /// Verify the pinned bytes before parsing, then bind names, shapes and clocks
    /// to the supplied artifact. Hashes identify bytes, not training provenance.
    pub fn from_json(
        bytes: &[u8],
        expected_id: &str,
        artifact: &GeometryAdapterArtifact,
    ) -> Result<Self> {
        if bytes.len() > MAX_BYTES {
            return Err("native geometry AdamW checkpoint exceeds byte budget".into());
        }
        if identity(bytes) != expected_id {
            return Err("native geometry AdamW checkpoint differs from pinned identity".into());
        }
        let record: Checkpoint = serde_json::from_slice(bytes)?;
        if record.schema != SCHEMA || record.geometry_artifact_id != artifact.artifact_id() {
            return Err("native geometry AdamW checkpoint schema or artifact mismatch".into());
        }
        let config = AdamWConfig::new(record.beta1, record.beta2, record.epsilon)?;
        let clip = record
            .max_gradient_norm
            .map(GlobalNormClip::new)
            .transpose()?;
        let mut result = Self::new(artifact, config, clip)?;
        if record.slots.len() != result.keys.len() {
            return Err("native geometry AdamW checkpoint parameter count mismatch".into());
        }
        let shapes: Vec<_> = result.state.slots.iter().map(|s| s.shape.clone()).collect();
        let mut slots = Vec::with_capacity(record.slots.len());
        for (slot, (path, name)) in record.slots.into_iter().zip(&result.keys) {
            if slot.placement != *path || slot.parameter != *name {
                return Err("native geometry AdamW checkpoint parameter order mismatch".into());
            }
            slots.push(AdamWSlot {
                shape: slot.shape,
                first: slot.first,
                second: slot.second,
                steps: slot.steps,
            });
        }
        let state = AdamWState { config, slots };
        state.validate(&shapes, record.revision)?;
        // The shared slot validator permits independently clocked tensors. This
        // owner updates all trainable tensors of a selected placement together.
        validate_placement_clocks(
            &result.keys,
            &state.slots.iter().map(|s| s.steps).collect::<Vec<_>>(),
            record.revision,
        )?;
        result.state = state;
        result.revision = record.revision;
        Ok(result)
    }
}

impl NativeGeometryRuntime {
    /// Stage a parameter-only pullback and AdamW update. Return both new owners
    /// only after all candidates and the resulting geometry domain validate.
    pub fn step_adamw(
        &self,
        batch: &NativeGeometryBatch,
        upstream: &[f32],
        optimizer: &NativeGeometryAdamW,
        step: AdamWStep,
    ) -> Result<(Self, NativeGeometryAdamW)> {
        if batch.source_id != self.artifact.artifact_id() {
            return Err("geometry snapshot is stale or belongs to another artifact".into());
        }
        let gradients = batch.vjp_parameters(upstream)?;
        self.with_gradients_adamw(
            &batch.source_id,
            &ParameterGradients::from([(batch.placement.path.clone(), gradients)]),
            optimizer,
            step,
        )
    }

    pub(crate) fn with_gradients_adamw(
        &self,
        source_id: &str,
        gradients: &ParameterGradients,
        optimizer: &NativeGeometryAdamW,
        step: AdamWStep,
    ) -> Result<(Self, NativeGeometryAdamW)> {
        self.validate_gradients(source_id, gradients)?;
        if optimizer.geometry_id != self.artifact.artifact_id() {
            return Err("native geometry AdamW belongs to a different artifact snapshot".into());
        }
        if step.rate() == 0. {
            return Ok((Self::new(self.artifact.clone()), optimizer.clone()));
        }
        let revision = optimizer
            .revision
            .checked_add(1)
            .ok_or("native geometry AdamW revision exhausted")?;
        let mut squared_norm = 0f64;
        for (path, name) in &optimizer.keys {
            if let Some(values) = gradients.get(path) {
                for &value in &values[name] {
                    squared_norm += f64::from(value) * f64::from(value);
                }
            }
        }
        let factors = optimizer
            .clip
            .map(|c| c.factors(squared_norm))
            .transpose()?;
        let factors = factors.as_ref().map_or(&[][..], |f| f.as_slice());
        let mut record = self.artifact.record().clone();
        let mut next = optimizer.clone();
        let mut slot_index = 0;
        for placement in &mut record.placements {
            for p in &mut placement.tensors {
                if p.role != GeometryAdapterTensorRole::Parameter || !p.requires_grad {
                    continue;
                }
                let slot = &mut next.state.slots[slot_index];
                slot_index += 1;
                let Some(values) = gradients.get(&placement.path) else {
                    continue;
                };
                let clock = slot.steps;
                for (i, &gradient) in values[&p.name].iter().enumerate() {
                    let gradient = factors.iter().fold(gradient, |g, f| g * f);
                    let candidate = step.candidate(
                        next.state.config,
                        p.values[i],
                        gradient,
                        slot.first[i],
                        slot.second[i],
                        clock,
                    )?;
                    p.values[i] = candidate.value;
                    slot.first[i] = candidate.first;
                    slot.second[i] = candidate.second;
                    slot.steps = candidate.steps;
                }
            }
        }
        let artifact = GeometryAdapterArtifact::create(record)?;
        next.geometry_id = artifact.artifact_id().into();
        next.revision = revision;
        Ok((Self::new(artifact), next))
    }
}

#[cfg(test)]
mod tests;
