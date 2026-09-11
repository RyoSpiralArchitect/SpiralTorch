//! Explicit weight handoff; never substitute resident SGD for ModuleTrainer policy.
use super::*;
use crate::module::Parameter;
use std::collections::{HashMap, HashSet};

pub type ResidentParameterBinding<'a> = (ParameterRole, &'a Parameter);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModuleOptimizerStatePolicy {
    /// Reject any attached tape or pending Euclidean gradient, including zeros.
    #[default]
    Reject,
    /// Discard parameter-local gradient/hypergrad/realgrad state explicitly.
    /// Trainer-level state is not reset or migrated by this operation.
    Reset,
}

impl std::str::FromStr for ModuleOptimizerStatePolicy {
    type Err = InferenceError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "reject" => Ok(Self::Reject),
            "reset" => Ok(Self::Reset),
            _ => Err(InferenceError::ModuleUpdate(
                "optimizer_state must be 'reject' or 'reset'",
            )),
        }
    }
}

struct Prepared {
    identity: *const Parameter,
    before: Vec<u32>,
    value: Tensor,
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn same_structure(a: &GraphDefinition, b: &GraphDefinition) -> Result<bool, InferenceError> {
    // Accept only the existing Rust fusion normalization, not arbitrary claimed
    // equivalence. Parameter IDs, roles, shapes and input layout remain fixed.
    let a = a.fuse_pointwise(3)?;
    let b = b.fuse_pointwise(3)?;
    Ok(a.input_layout() == b.input_layout()
        && a.stages() == b.stages()
        && a.parameters().len() == b.parameters().len()
        && a.parameters()
            .iter()
            .zip(b.parameters())
            .all(|(a, b)| a.role == b.role && a.shape == b.shape))
}

impl InferencePlan {
    /// Apply updated plan values to an existing Module only while it still
    /// matches this baseline. No GPU work or readback is hidden here: callers
    /// first read an owning parameter snapshot into `updated`.
    ///
    /// All ownership, topology, value, visitor and optimizer checks/allocation
    /// precede mutation. Commit relies on Module's stable visitor contract.
    /// This is a weight handoff, not object identity, training lineage, optimizer
    /// migration or authorization. A separately loaded matching model is valid.
    pub fn apply_parameters_to<M: Module + ?Sized>(
        &self,
        module: &mut M,
        updated: &InferencePlan,
        optimizer: ModuleOptimizerStatePolicy,
    ) -> Result<usize, InferenceError> {
        let baseline = self.graph_definition()?;
        let current = Self::from_module(module, self.input.clone())?.graph_definition()?;
        let target = updated.graph_definition()?;
        if !same_structure(&baseline, &current)? || !same_structure(&baseline, &target)? {
            return Err(InferenceError::ModuleUpdate("graph structure differs"));
        }
        if baseline
            .parameters()
            .iter()
            .zip(current.parameters())
            .any(|(a, b)| bits(&a.values) != bits(&b.values))
        {
            return Err(InferenceError::ModuleUpdate(
                "module values changed since baseline",
            ));
        }
        let mut prepared = {
            let bindings = module.resident_parameter_bindings()?;
            if bindings.len() != baseline.parameters().len() {
                return Err(InferenceError::ModuleUpdate(
                    "parameter binding count differs",
                ));
            }
            let mut prepared = HashMap::with_capacity(bindings.len());
            for ((role, param), (before, after)) in bindings
                .into_iter()
                .zip(baseline.parameters().iter().zip(target.parameters()))
            {
                let shape = match before.role {
                    ParameterRole::Weight => (before.shape[0], before.shape[1]),
                    ParameterRole::Bias | ParameterRole::Gain => (1, before.shape[0]),
                };
                if role != before.role
                    || param.value().shape() != shape
                    || bits(param.value().to_layout(Layout::RowMajor)?.data())
                        != bits(&before.values)
                {
                    return Err(InferenceError::ModuleUpdate(
                        "parameter binding role, shape or values differ",
                    ));
                }
                let value = Tensor::from_vec(shape.0, shape.1, after.values.clone())?
                    .to_layout(param.value().layout())?
                    .into_snapshot();
                let entry = Prepared {
                    identity: std::ptr::from_ref(param),
                    before: bits(param.value().data()),
                    value,
                };
                if prepared.insert(param.name().to_owned(), entry).is_some() {
                    return Err(InferenceError::ModuleUpdate(
                        "parameter names must be unique",
                    ));
                }
            }
            prepared
        };
        let mut seen = HashSet::new();
        let mut valid = true;
        module.visit_parameters(&mut |param| {
            valid &= prepared
                .get(param.name())
                .is_some_and(|p| p.identity == std::ptr::from_ref(param))
                && seen.insert(param.name().to_owned());
            Ok(())
        })?;
        if !valid || seen.len() != prepared.len() {
            return Err(InferenceError::ModuleUpdate(
                "immutable visitor does not match bindings",
            ));
        }
        seen.clear();
        let mut attached_optimizer = false;
        module.visit_parameters_mut(&mut |param| {
            valid &= prepared.get(param.name()).is_some_and(|p| {
                p.identity == std::ptr::from_ref(param)
                    && p.value.shape() == param.value().shape()
                    && p.value.layout() == param.value().layout()
                    && p.before == bits(param.value().data())
            }) && seen.insert(param.name().to_owned());
            attached_optimizer |= param.gradient().is_some()
                || param.hypergrad().is_some()
                || param.realgrad().is_some();
            Ok(())
        })?;
        if !valid || seen.len() != prepared.len() {
            return Err(InferenceError::ModuleUpdate(
                "mutable visitor does not match bindings",
            ));
        }
        if optimizer == ModuleOptimizerStatePolicy::Reject && attached_optimizer {
            return Err(InferenceError::ModuleUpdate(
                "optimizer state is attached; explicit reset is required",
            ));
        }
        let count = prepared.len();
        module.visit_parameters_mut(&mut |param| {
            let entry = prepared
                .remove(param.name())
                .expect("Module must honor its prevalidated stable visitor contract");
            param
                .commit_resident_value(entry.value, optimizer == ModuleOptimizerStatePolicy::Reset);
            Ok(())
        })?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests;
