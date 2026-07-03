use crate::module::{
    adapt_state_dict_keys, fingerprint_state_dict, state_dict_compatibility_for_expected,
    state_dict_compatibility_for_expected_with_rules, state_key_rules_from_map, Module, Parameter,
    StateCompatibilityReport, StateFingerprint, StateKeyMapRule, StateLoadReport,
};
use crate::{PureResult, Tensor, TensorError};
use std::collections::HashMap;

/// Linear layer with a trainable low-rank adapter on top of a frozen base.
///
/// The base parameters intentionally use the same `name::weight` and
/// `name::bias` keys as [`crate::layers::Linear`], so a checkpoint from an
/// existing dense layer can seed this module while the adapter learns the
/// fine-tune delta.
#[derive(Debug)]
pub struct LoraLinear {
    weight: Parameter,
    bias: Parameter,
    lora_down: Parameter,
    lora_up: Parameter,
    rank: usize,
    alpha: f32,
}

impl LoraLinear {
    /// Creates a LoRA-style linear adapter with frozen base parameters.
    pub fn new(
        name: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        alpha: f32,
    ) -> PureResult<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim,
                cols: output_dim,
            });
        }
        if rank == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim,
                cols: rank,
            });
        }
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lora_linear_alpha",
                value: alpha,
            });
        }

        let name = name.into();
        let mut scale = 0.01f32;
        let weights = Tensor::from_fn(input_dim, output_dim, |_r, _c| {
            let value = scale;
            scale += 0.01;
            value
        })?;
        let bias = Tensor::zeros(1, output_dim)?;
        let lora_down = Tensor::from_fn(input_dim, rank, |r, c| {
            (((r + c + 1) % 11) as f32 - 5.0) * 0.0005
        })?;
        let lora_up = Tensor::from_fn(rank, output_dim, |r, c| {
            (((r * output_dim + c + 3) % 13) as f32 - 6.0) * 0.0005
        })?;

        let mut weight = Parameter::new(format!("{name}::weight"), weights);
        let mut bias = Parameter::new(format!("{name}::bias"), bias);
        weight.set_trainable(false);
        bias.set_trainable(false);

        Ok(Self {
            weight,
            bias,
            lora_down: Parameter::new(format!("{name}::lora_down"), lora_down),
            lora_up: Parameter::new(format!("{name}::lora_up"), lora_up),
            rank,
            alpha,
        })
    }

    /// Returns the frozen base weight parameter.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    /// Returns the frozen base bias parameter.
    pub fn bias(&self) -> &Parameter {
        &self.bias
    }

    /// Returns the trainable down-projection adapter parameter.
    pub fn lora_down(&self) -> &Parameter {
        &self.lora_down
    }

    /// Returns the trainable up-projection adapter parameter.
    pub fn lora_up(&self) -> &Parameter {
        &self.lora_up
    }

    /// Returns the adapter rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the LoRA alpha used to scale the adapter path.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns `alpha / rank`, the multiplier applied to the adapter logits.
    pub fn adapter_scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Enables or freezes the dense base parameters.
    pub fn set_base_trainable(&mut self, trainable: bool) {
        self.weight.set_trainable(trainable);
        self.bias.set_trainable(trainable);
    }

    /// Enables or freezes the low-rank adapter parameters.
    pub fn set_adapter_trainable(&mut self, trainable: bool) {
        self.lora_down.set_trainable(trainable);
        self.lora_up.set_trainable(trainable);
    }

    /// Captures only the dense base state using `Linear`-compatible keys.
    pub fn base_state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::from([
            (self.weight.name().to_string(), self.weight.value().clone()),
            (self.bias.name().to_string(), self.bias.value().clone()),
        ])
    }

    /// Computes a fingerprint for only the dense base parameters.
    pub fn base_state_fingerprint(&self) -> StateFingerprint {
        fingerprint_state_dict(&self.base_state_dict())
    }

    /// Checks whether a dense `Linear` checkpoint can seed this adapter's base.
    pub fn base_state_dict_compatibility(
        &self,
        state: &HashMap<String, Tensor>,
    ) -> StateCompatibilityReport {
        state_dict_compatibility_for_expected(self.base_expected_shapes(), state)
    }

    /// Checks dense-base compatibility after remapping external checkpoint keys.
    pub fn base_state_dict_compatibility_with_key_map(
        &self,
        state: &HashMap<String, Tensor>,
        source_to_target: &HashMap<String, String>,
    ) -> PureResult<StateCompatibilityReport> {
        let rules = state_key_rules_from_map(source_to_target);
        self.base_state_dict_compatibility_with_key_rules(state, &rules)
    }

    /// Checks dense-base compatibility after applying key/layout transform rules.
    pub fn base_state_dict_compatibility_with_key_rules(
        &self,
        state: &HashMap<String, Tensor>,
        rules: &HashMap<String, StateKeyMapRule>,
    ) -> PureResult<StateCompatibilityReport> {
        state_dict_compatibility_for_expected_with_rules(self.base_expected_shapes(), state, rules)
    }

    /// Loads base parameters from a dense `Linear` checkpoint/state dict.
    pub fn load_base_from_state_dict(
        &mut self,
        state: &HashMap<String, Tensor>,
    ) -> PureResult<StateLoadReport> {
        let weight_name = self.weight.name().to_string();
        let bias_name = self.bias.name().to_string();
        let Some(weight) = state.get(&weight_name) else {
            return Err(TensorError::MissingParameter { name: weight_name });
        };
        let Some(bias) = state.get(&bias_name) else {
            return Err(TensorError::MissingParameter { name: bias_name });
        };
        let base_state = HashMap::from([
            (self.weight.name().to_string(), weight.clone()),
            (self.bias.name().to_string(), bias.clone()),
        ]);
        let source = fingerprint_state_dict(&base_state);
        self.weight.load_value(weight)?;
        self.bias.load_value(bias)?;
        let loaded = self.base_state_fingerprint();
        let matched = source == loaded;
        Ok(StateLoadReport {
            source,
            loaded,
            matched,
        })
    }

    /// Remaps external checkpoint keys, then loads this adapter's dense base.
    pub fn load_base_from_state_dict_mapped(
        &mut self,
        state: &HashMap<String, Tensor>,
        source_to_target: &HashMap<String, String>,
    ) -> PureResult<StateLoadReport> {
        let rules = state_key_rules_from_map(source_to_target);
        self.load_base_from_state_dict_adapted(state, &rules)
    }

    /// Applies external key/layout transform rules, then loads this adapter's base.
    pub fn load_base_from_state_dict_adapted(
        &mut self,
        state: &HashMap<String, Tensor>,
        rules: &HashMap<String, StateKeyMapRule>,
    ) -> PureResult<StateLoadReport> {
        let expected = self.base_expected_shapes();
        let expected_shapes: HashMap<String, (usize, usize)> = expected
            .iter()
            .map(|(name, shape)| (name.clone(), *shape))
            .collect();
        let adapted = adapt_state_dict_keys(state, rules, &expected_shapes)?;
        self.load_base_from_state_dict(&adapted)
    }

    fn base_expected_shapes(&self) -> Vec<(String, (usize, usize))> {
        vec![
            (self.weight.name().to_string(), self.weight.value().shape()),
            (self.bias.name().to_string(), self.bias.value().shape()),
        ]
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        if input.shape().1 != self.weight.value().shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.weight.value().shape(),
            });
        }
        let mut base = input.matmul(self.weight.value())?;
        base.add_row_inplace(self.bias.value().data())?;
        let adapter = input
            .matmul(self.lora_down.value())?
            .matmul(self.lora_up.value())?
            .scale(self.adapter_scale())?;
        base.add(&adapter)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape().0 != grad_output.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        if input.shape().1 != self.weight.value().shape().0
            || grad_output.shape().1 != self.weight.value().shape().1
        {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.weight.value().shape(),
            });
        }

        let batch = input.shape().0 as f32;
        if self.weight.is_trainable() {
            let grad_w = input.transpose().matmul(grad_output)?.scale(1.0 / batch)?;
            self.weight.accumulate_euclidean(&grad_w)?;
        }
        if self.bias.is_trainable() {
            let summed = grad_output.sum_axis0();
            let grad_b = Tensor::from_vec(1, summed.len(), summed)?.scale(1.0 / batch)?;
            self.bias.accumulate_euclidean(&grad_b)?;
        }

        let adapter_scale = self.adapter_scale();
        let adapter_hidden = input.matmul(self.lora_down.value())?;
        let grad_up = adapter_hidden
            .transpose()
            .matmul(grad_output)?
            .scale(adapter_scale / batch)?;
        self.lora_up.accumulate_euclidean(&grad_up)?;

        let up_t = self.lora_up.value().transpose();
        let grad_hidden = grad_output.matmul(&up_t)?.scale(adapter_scale)?;
        let grad_down = input.transpose().matmul(&grad_hidden)?.scale(1.0 / batch)?;
        self.lora_down.accumulate_euclidean(&grad_down)?;

        let weight_t = self.weight.value().transpose();
        let grad_input_base = grad_output.matmul(&weight_t)?;
        let down_t = self.lora_down.value().transpose();
        let grad_input_adapter = grad_hidden.matmul(&down_t)?;
        grad_input_base.add(&grad_input_adapter)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight)?;
        visitor(&self.bias)?;
        visitor(&self.lora_down)?;
        visitor(&self.lora_up)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight)?;
        visitor(&mut self.bias)?;
        visitor(&mut self.lora_down)?;
        visitor(&mut self.lora_up)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::module::StateTensorTransform;

    #[test]
    fn lora_linear_forward_matches_base_plus_adapter() {
        let layer = LoraLinear::new("fc", 3, 2, 2, 2.0).unwrap();
        let input = Tensor::from_vec(1, 3, vec![1.0, -2.0, 0.5]).unwrap();
        let output = layer.forward(&input).unwrap();

        let mut base = input.matmul(layer.weight.value()).unwrap();
        base.add_row_inplace(layer.bias.value().data()).unwrap();
        let adapter = input
            .matmul(layer.lora_down.value())
            .unwrap()
            .matmul(layer.lora_up.value())
            .unwrap()
            .scale(layer.adapter_scale())
            .unwrap();
        assert_eq!(output, base.add(&adapter).unwrap());
    }

    #[test]
    fn lora_linear_loads_linear_base_and_moves_adapter_only() {
        let dense = Linear::new("head", 3, 2).unwrap();
        let dense_state = dense.state_dict().unwrap();
        let mut layer = LoraLinear::new("head", 3, 2, 2, 1.0).unwrap();
        let full_compatibility = layer.state_dict_compatibility(&dense_state).unwrap();
        assert!(!full_compatibility.compatible);
        assert_eq!(full_compatibility.missing, 2);

        let base_compatibility = layer.base_state_dict_compatibility(&dense_state);
        assert!(base_compatibility.compatible);
        assert_eq!(base_compatibility.expected_parameters, 2);
        assert_eq!(base_compatibility.matched, 2);
        assert_eq!(base_compatibility.missing, 0);
        assert_eq!(base_compatibility.shape_mismatched, 0);

        let load = layer.load_base_from_state_dict(&dense_state).unwrap();
        assert!(load.matched);
        assert!(!layer.weight().is_trainable());
        assert!(!layer.bias().is_trainable());
        assert!(layer.lora_down().is_trainable());
        assert!(layer.lora_up().is_trainable());

        let before = layer.state_dict().unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.2, -0.1, 0.5, -0.3, 0.7, 0.4]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.1, -0.2, 0.05, 0.3]).unwrap();
        let _ = layer.backward(&input, &grad_output).unwrap();
        layer.apply_step(0.1).unwrap();
        let movement = layer.audit_parameter_movement(&before, 1e-8).unwrap();
        assert!(movement.frozen_stable());
        assert!(movement.trainable_movement_observed());
    }

    #[test]
    fn lora_linear_respects_adapter_freeze_during_backward() {
        let mut layer = LoraLinear::new("head", 3, 2, 2, 1.0).unwrap();
        layer.set_adapter_trainable(false);

        let before = layer.state_dict().unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.2, -0.1, 0.5, -0.3, 0.7, 0.4]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.1, -0.2, 0.05, 0.3]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(layer.lora_down().gradient().is_none());
        assert!(layer.lora_up().gradient().is_none());

        layer.apply_step(0.1).unwrap();
        assert_eq!(before, layer.state_dict().unwrap());
    }

    #[test]
    fn lora_linear_loads_mapped_external_base_keys() {
        let dense = Linear::new("source_head", 3, 2).unwrap();
        let dense_state = dense.state_dict().unwrap();
        let external_state = HashMap::from([
            (
                "model.lm_head.weight".to_string(),
                dense_state["source_head::weight"].clone(),
            ),
            (
                "model.lm_head.bias".to_string(),
                dense_state["source_head::bias"].clone(),
            ),
        ]);
        let key_map = HashMap::from([
            (
                "model.lm_head.weight".to_string(),
                "head::weight".to_string(),
            ),
            ("model.lm_head.bias".to_string(), "head::bias".to_string()),
        ]);
        let mut layer = LoraLinear::new("head", 3, 2, 2, 1.0).unwrap();

        let plain = layer.base_state_dict_compatibility(&external_state);
        assert!(!plain.compatible);
        assert_eq!(plain.missing, 2);
        let mapped = layer
            .base_state_dict_compatibility_with_key_map(&external_state, &key_map)
            .unwrap();
        assert!(mapped.compatible);
        assert_eq!(mapped.matched, 2);

        let load = layer
            .load_base_from_state_dict_mapped(&external_state, &key_map)
            .unwrap();
        assert!(load.matched);
    }

    #[test]
    fn lora_linear_adapts_transposed_external_base_weights() {
        let external_weight = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let external_bias = Tensor::from_vec(1, 2, vec![0.25, -0.5]).unwrap();
        let external_state = HashMap::from([
            ("model.lm_head.weight".to_string(), external_weight.clone()),
            ("model.lm_head.bias".to_string(), external_bias.clone()),
        ]);
        let rules = HashMap::from([
            (
                "model.lm_head.weight".to_string(),
                StateKeyMapRule::with_transform("head::weight", StateTensorTransform::Transpose),
            ),
            (
                "model.lm_head.bias".to_string(),
                StateKeyMapRule::new("head::bias"),
            ),
        ]);
        let mut layer = LoraLinear::new("head", 3, 2, 2, 1.0).unwrap();

        let compatibility = layer
            .base_state_dict_compatibility_with_key_rules(&external_state, &rules)
            .unwrap();
        assert!(compatibility.compatible);
        let weight_entry = compatibility
            .entries
            .iter()
            .find(|entry| entry.name == "head::weight")
            .unwrap();
        assert_eq!(
            weight_entry.source_name.as_deref(),
            Some("model.lm_head.weight")
        );
        assert_eq!(weight_entry.transform, StateTensorTransform::Transpose);
        assert_eq!(weight_entry.original_source_shape, Some((2, 3)));
        assert_eq!(weight_entry.source_shape, Some((3, 2)));
        let load = layer
            .load_base_from_state_dict_adapted(&external_state, &rules)
            .unwrap();
        assert!(load.matched);
        assert_eq!(layer.weight().value(), &external_weight.transpose());
        assert_eq!(layer.bias().value(), &external_bias);
    }
}
