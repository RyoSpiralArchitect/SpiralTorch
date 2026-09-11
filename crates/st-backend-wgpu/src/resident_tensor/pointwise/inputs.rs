//! Owning input handles for clients that cannot pass borrowed Rust slices.
use super::*;

#[derive(Default)]
pub struct PointwiseInputs {
    values: Vec<ResidentTensor>,
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn bindings_validate_context_and_replacements_before_mutation_on_real_gpu() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("pointwise.bindings").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let d = TensorDevice::new(runtime).unwrap();
    let a = d.upload(&[2], &[1., 2.]).unwrap();
    let b = d.upload(&[], &[2.]).unwrap();
    let mut inputs = PointwiseInputs::new();
    inputs.add(&a).unwrap();
    inputs.add(&b).unwrap();
    let chain = PointwiseChain::new(
        2,
        vec![st_kernel_contracts::pointwise::PointwiseStep::named("multiply", Some(1)).unwrap()],
    )
    .unwrap();
    let plan = inputs.compile(chain).unwrap();
    let private = TensorDevice::new(
        pollster::block_on(WgpuRuntime::request_headless("pointwise.alien")).unwrap(),
    )
    .unwrap();
    let alien = private.upload(&[2], &[9.; 2]).unwrap();
    assert!(inputs.set(0, &alien).is_err());
    assert!(inputs.add(&alien).is_err());
    assert!(inputs.set(0, &b).is_err());
    assert!(inputs.set(2, &a).is_err());
    let original = inputs.run(&plan, PointwiseExecution::Fused).unwrap();
    inputs.set(0, &d.upload(&[2], &[3., 4.]).unwrap()).unwrap();
    let next = inputs.run(&plan, PointwiseExecution::Fused).unwrap();
    drop(inputs);
    drop(plan);
    assert_eq!(original.snapshot().unwrap().read().unwrap(), vec![2., 4.]);
    assert_eq!(next.snapshot().unwrap().read().unwrap(), vec![6., 8.]);
    let mut full = PointwiseInputs::new();
    for _ in 0..16 {
        full.add(&a).unwrap();
    }
    assert!(full.add(&a).is_err());
    assert_eq!(full.len(), 16);
}

impl PointwiseInputs {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// No upload, copy, or submission. Inputs must share the same device/queue.
    pub fn add(&mut self, tensor: &ResidentTensor) -> Result<(), TensorError> {
        if self.len() == 16 {
            return Err(PointwiseError::Budget.into());
        }
        if let Some(first) = self.values.first() {
            tensor.require_context(first.device().runtime().context())?;
        }
        self.values.push(tensor.clone());
        Ok(())
    }

    /// Preserve each slot's exact logical layout, including strides and offset.
    /// Failed replacement leaves all previously bound inputs unchanged.
    pub fn set(&mut self, slot: usize, tensor: &ResidentTensor) -> Result<(), TensorError> {
        let current = self.values.get(slot).ok_or(PointwiseError::Operands)?;
        tensor.require_context(current.device().runtime().context())?;
        if tensor.layout() != current.layout() {
            return Err(PointwiseError::LayoutMismatch.into());
        }
        self.values[slot] = tensor.clone();
        Ok(())
    }

    /// Only layouts and the device are retained by the compiled plan, not values.
    pub fn compile(&self, chain: PointwiseChain) -> Result<PointwisePlan, TensorError> {
        let first = self.values.first().ok_or(PointwiseError::Operands)?;
        PointwisePlan::new(
            first.device().clone(),
            chain,
            self.values.iter().map(|t| t.layout().clone()).collect(),
        )
    }

    pub fn run(
        &self,
        plan: &PointwisePlan,
        execution: PointwiseExecution,
    ) -> Result<ResidentTensor, TensorError> {
        plan.run(&self.values.iter().collect::<Vec<_>>(), execution)
    }
}
