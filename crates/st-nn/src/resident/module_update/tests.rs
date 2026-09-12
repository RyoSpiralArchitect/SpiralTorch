use super::*;
use crate::{Gelu, Linear, ModuleTrainer, Relu, Scaler, Sequential};
use st_core::backend::device_caps::DeviceCaps;

fn cpu() -> crate::execution::BackendPolicyGuard {
    let context = crate::execution::RuntimeExecutionContext::from_device_caps_with_config(
        DeviceCaps::cpu(),
        Default::default(),
    );
    crate::execution::push_backend_policy(context.backend_policy())
}

fn model() -> Sequential {
    let mut model = Sequential::new();
    model.push(Linear::new("up", 2, 3).unwrap());
    model.push(Gelu::new());
    model.push(Scaler::new("hidden", 3).unwrap());
    model.push(Relu::new());
    model.push(Linear::new("down", 3, 2).unwrap());
    model.push(Scaler::new("output", 2).unwrap());
    model
}

fn plan(model: &impl Module) -> InferencePlan {
    InferencePlan::from_module(model, NdLayout::contiguous(&[2, 2, 2]).unwrap()).unwrap()
}

fn changed(base: &InferencePlan) -> InferencePlan {
    base.with_graph_values(
        base.graph_definition()
            .unwrap()
            .parameters()
            .iter()
            .map(|p| p.values.iter().map(|v| v + 0.125).collect())
            .collect(),
    )
    .unwrap()
}

fn state(model: &impl Module) -> Vec<(String, Vec<u32>, serde_json::Value)> {
    let mut out = Vec::new();
    model
        .visit_parameters(&mut |p| {
            out.push((
                p.name().to_owned(),
                bits(p.value().data()),
                serde_json::to_value(p.optimizer_checkpoint_state()).unwrap(),
            ));
            Ok(())
        })
        .unwrap();
    out
}

#[test]
fn handoff_updates_all_roles_and_invalidates_forward_and_transpose_packs() {
    let _guard = cpu();
    let mut model = model();
    model
        .visit_parameters_mut(&mut |p| {
            if p.name() == "up::weight" {
                let value = p.value().to_layout(Layout::ColMajor)?;
                p.load_value(&value)?;
            }
            p.ensure_matmul_pack()?;
            p.ensure_matmul_transpose_pack()?;
            Ok(())
        })
        .unwrap();
    let base = plan(&model);
    let updated = changed(&base).fuse_pointwise().unwrap();
    let x = Tensor::from_vec(4, 2, vec![0.25; 8]).unwrap();
    let before = model.forward(&x).unwrap();
    let mut reference = self::model();
    let values = updated.graph_definition().unwrap();
    let mut slot = 0;
    reference
        .visit_parameters_mut(&mut |p| {
            let shape = p.value().shape();
            p.load_value(&Tensor::from_vec(
                shape.0,
                shape.1,
                values.parameters()[slot].values.clone(),
            )?)?;
            slot += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(
        base.apply_parameters_to(&mut model, &updated, ModuleOptimizerStatePolicy::Reject)
            .unwrap(),
        6
    );
    assert_ne!(model.forward(&x).unwrap(), before);
    assert_eq!(
        model.forward(&x).unwrap().data(),
        reference.forward(&x).unwrap().data()
    );
    let grad = Tensor::from_vec(4, 2, vec![0.5; 8]).unwrap();
    assert_eq!(
        model.backward(&x, &grad).unwrap().data(),
        reference.backward(&x, &grad).unwrap().data()
    );
    model
        .visit_parameters(&mut |p| {
            if p.name() == "up::weight" {
                assert_eq!(p.value().layout(), Layout::ColMajor);
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn drift_and_changed_program_reject_every_parameter_before_reset() {
    let mut model = model();
    let base = plan(&model);
    let updated = changed(&base);
    model.attach_hypergrad(-1., 0.01).unwrap();
    model.attach_realgrad(0.01).unwrap();
    model
        .visit_parameters_mut(&mut |p| {
            if p.name() == "output::gain" {
                p.value_mut().data_mut()[0] += 0.25;
            }
            Ok(())
        })
        .unwrap();
    let before = state(&model);
    assert!(base
        .apply_parameters_to(&mut model, &updated, ModuleOptimizerStatePolicy::Reset)
        .is_err());
    assert_eq!(before, state(&model));
    let base = plan(&model);
    let graph = base.graph_definition().unwrap();
    let mut stages = graph.stages().to_vec();
    if let GraphStage::Linear { gelu, .. } = &mut stages[0] {
        *gelu = false;
    }
    let altered = InferencePlan::from_graph_definition(
        GraphDefinition::new(
            graph.input_layout().clone(),
            stages,
            graph.parameters().to_vec(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(base
        .apply_parameters_to(&mut model, &altered, ModuleOptimizerStatePolicy::Reset)
        .is_err());
    assert_eq!(before, state(&model));
}

#[test]
fn optimizer_reset_is_explicit_and_the_existing_trainer_can_prepare_again() {
    let _guard = cpu();
    let mut model = model();
    let base = plan(&model);
    let updated = changed(&base);
    model.attach_hypergrad(-1., 0.01).unwrap();
    model.attach_realgrad(0.01).unwrap();
    let before = state(&model);
    assert!(base
        .apply_parameters_to(&mut model, &updated, ModuleOptimizerStatePolicy::Reject)
        .is_err());
    assert_eq!(before, state(&model));
    base.apply_parameters_to(&mut model, &updated, ModuleOptimizerStatePolicy::Reset)
        .unwrap();
    model
        .visit_parameters(&mut |p| {
            assert!(p.gradient().is_none() && p.hypergrad().is_none() && p.realgrad().is_none());
            Ok(())
        })
        .unwrap();
    let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1., 0.01, 0.01);
    trainer.prepare(&mut model).unwrap();
    let after_handoff = state(&model);
    let x = Tensor::from_vec(4, 2, vec![0.25; 8]).unwrap();
    model
        .backward(&x, &Tensor::from_vec(4, 2, vec![0.5; 8]).unwrap())
        .unwrap();
    trainer.step(&mut model).unwrap();
    assert_ne!(after_handoff, state(&model));
    assert!(model
        .forward(&x)
        .unwrap()
        .data()
        .iter()
        .all(|v| v.is_finite()));
    for invalid in ["", "auto", "preserve", "RESET"] {
        assert!(invalid.parse::<ModuleOptimizerStatePolicy>().is_err());
    }
}

#[test]
fn pending_zero_gradients_signed_zero_and_duplicate_names_are_not_silent() {
    let mut layer = Linear::new("zero", 2, 2).unwrap();
    let base = plan(&layer);
    layer
        .visit_parameters_mut(&mut |p| {
            let (rows, cols) = p.value().shape();
            p.accumulate_euclidean(&Tensor::zeros(rows, cols)?)
        })
        .unwrap();
    let before = state(&layer);
    assert!(base
        .apply_parameters_to(
            &mut layer,
            &changed(&base),
            ModuleOptimizerStatePolicy::Reject
        )
        .is_err());
    assert_eq!(before, state(&layer));
    layer.zero_accumulators().unwrap();
    layer
        .visit_parameters_mut(&mut |p| {
            if p.name() == "zero::bias" {
                p.value_mut().data_mut()[0] = -0.;
            }
            Ok(())
        })
        .unwrap();
    let before = state(&layer);
    assert!(base
        .apply_parameters_to(&mut layer, &base, ModuleOptimizerStatePolicy::Reset)
        .is_err());
    assert_eq!(before, state(&layer));
    let mut duplicate = Sequential::new();
    duplicate.push(Linear::new("same", 2, 2).unwrap());
    duplicate.push(Linear::new("same", 2, 2).unwrap());
    let base = plan(&duplicate);
    let before = state(&duplicate);
    assert!(base
        .apply_parameters_to(
            &mut duplicate,
            &changed(&base),
            ModuleOptimizerStatePolicy::Reset
        )
        .is_err());
    assert_eq!(before, state(&duplicate));
}

#[test]
fn dense_v1_and_parameter_free_modules_use_the_same_handoff_contract() {
    let mut dense = Linear::new("dense", 2, 3).unwrap();
    let base = plan(&dense);
    assert!(base.graph.is_none());
    let updated = changed(&base);
    assert_eq!(
        base.apply_parameters_to(&mut dense, &updated, ModuleOptimizerStatePolicy::Reject)
            .unwrap(),
        2
    );
    assert_eq!(
        plan(&dense)
            .graph_definition()
            .unwrap()
            .parameters()
            .iter()
            .map(|p| p.values.clone())
            .collect::<Vec<_>>(),
        updated
            .graph_definition()
            .unwrap()
            .parameters()
            .iter()
            .map(|p| p.values.clone())
            .collect::<Vec<_>>()
    );
    let mut activation = Relu::new();
    let base = plan(&activation);
    assert_eq!(
        base.apply_parameters_to(&mut activation, &base, ModuleOptimizerStatePolicy::Reject)
            .unwrap(),
        0
    );
}

struct ForeignBinding {
    actual: Linear,
    decoy: Linear,
    foreign_mutable_visitor: bool,
}
impl Module for ForeignBinding {
    fn inference_ops(&self) -> Result<Vec<InferenceOp>, InferenceError> {
        self.actual.inference_ops()
    }
    fn resident_parameter_bindings(
        &self,
    ) -> Result<Vec<ResidentParameterBinding<'_>>, InferenceError> {
        if self.foreign_mutable_visitor {
            self.actual.resident_parameter_bindings()
        } else {
            self.decoy.resident_parameter_bindings()
        }
    }
    fn forward(&self, x: &Tensor) -> crate::PureResult<Tensor> {
        self.actual.forward(x)
    }
    fn backward(&mut self, x: &Tensor, g: &Tensor) -> crate::PureResult<Tensor> {
        self.actual.backward(x, g)
    }
    fn visit_parameters(
        &self,
        f: &mut dyn FnMut(&Parameter) -> crate::PureResult<()>,
    ) -> crate::PureResult<()> {
        self.actual.visit_parameters(f)
    }
    fn visit_parameters_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Parameter) -> crate::PureResult<()>,
    ) -> crate::PureResult<()> {
        if self.foreign_mutable_visitor {
            self.decoy.visit_parameters_mut(f)
        } else {
            self.actual.visit_parameters_mut(f)
        }
    }
}

#[test]
fn same_named_same_valued_foreign_bindings_are_rejected() {
    let mut model = ForeignBinding {
        actual: Linear::new("same", 2, 2).unwrap(),
        decoy: Linear::new("same", 2, 2).unwrap(),
        foreign_mutable_visitor: false,
    };
    let base = plan(&model);
    let before = state(&model);
    assert!(matches!(
        base.apply_parameters_to(
            &mut model,
            &changed(&base),
            ModuleOptimizerStatePolicy::Reject
        ),
        Err(InferenceError::ModuleUpdate(
            "immutable visitor does not match bindings"
        ))
    ));
    assert_eq!(before, state(&model));
}

#[test]
fn foreign_mutable_visitor_is_rejected_before_any_values_or_tapes_change() {
    let mut model = ForeignBinding {
        actual: Linear::new("same", 2, 2).unwrap(),
        decoy: Linear::new("same", 2, 2).unwrap(),
        foreign_mutable_visitor: true,
    };
    model.actual.attach_hypergrad(-1., 0.01).unwrap();
    model.decoy.attach_hypergrad(-1., 0.01).unwrap();
    let base = plan(&model);
    let actual = state(&model.actual);
    let decoy = state(&model.decoy);
    assert!(matches!(
        base.apply_parameters_to(
            &mut model,
            &changed(&base),
            ModuleOptimizerStatePolicy::Reset
        ),
        Err(InferenceError::ModuleUpdate(
            "mutable visitor does not match bindings"
        ))
    ));
    assert_eq!(actual, state(&model.actual));
    assert_eq!(decoy, state(&model.decoy));
}

#[test]
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn native_gpu_training_returns_to_the_original_module_without_stale_packs() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let _guard = cpu();
    let mut model = model();
    let x = Tensor::from_vec(4, 2, vec![0.25; 8]).unwrap();
    let original = model.forward(&x).unwrap();
    let base = plan(&model);
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("module.handoff.test").unwrap();
    assert_ne!(format!("{:?}", runtime.adapter_info().device_type), "Cpu");
    let mut gpu = base
        .fuse_pointwise()
        .unwrap()
        .compile_graph_training_wgpu(runtime, GraphGradientPolicy::ModuleCompatible)
        .unwrap();
    gpu.upload_batch(x.data(), &[1.; 8]).unwrap();
    for _ in 0..16 {
        gpu.step(0.05).unwrap();
        gpu.loss_snapshot().unwrap().read().unwrap();
    }
    gpu.step(0.).unwrap();
    let saved = gpu.state_snapshot().unwrap().read().unwrap();
    let updated = InferencePlan::from_graph_definition(saved.graph).unwrap();
    let updated = InferencePlan::from_json(&updated.to_json().unwrap()).unwrap();
    drop(gpu);
    assert_eq!(
        base.apply_parameters_to(&mut model, &updated, ModuleOptimizerStatePolicy::Reject)
            .unwrap(),
        6
    );
    let result = model.forward(&x).unwrap();
    assert_ne!(result, original);
    for (a, b) in result.data().iter().zip(&saved.prediction) {
        assert!((a - b).abs() < 2e-5);
    }
    assert!(base
        .apply_parameters_to(&mut model, &updated, ModuleOptimizerStatePolicy::Reject)
        .is_err());
}
