#![cfg(not(target_arch = "wasm32"))]

use st_nn::{
    module::{Module, Parameter},
    resident::{InferenceError, InferenceOp},
    Gelu, Linear, PureResult, Relu, Scaler, Sequential, Tensor,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    rc::Rc,
};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
struct CountingAllocator;
fn allocated() {
    let _ = COUNTING.try_with(|active| {
        if active.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
    });
}
// Only this integration-test binary installs the counter. Storage and thread
// semantics are those of System; no production allocator or timing is changed.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated();
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated();
        }
        pointer
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        if !pointer.is_null() {
            allocated();
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            COUNTING.with(|active| active.set(false));
        }
    }
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|active| assert!(!active.replace(true)));
    let reset = Reset;
    let result = black_box(operation());
    drop(reset);
    (result, ALLOCATIONS.with(Cell::get))
}

fn leaves(blocks: usize) -> Vec<Box<dyn Module>> {
    let mut layers: Vec<Box<dyn Module>> = Vec::new();
    for i in 0..blocks {
        layers.push(Box::new(Scaler::new(format!("gain{i}"), 7).unwrap()));
        layers.push(Box::new(Linear::new(format!("linear{i}"), 7, 7).unwrap()));
        layers.push(Box::new(Gelu::new()));
        layers.push(Box::new(Relu::new()));
    }
    layers
}
fn model(blocks: usize) -> Sequential {
    let mut model = Sequential::new();
    for layer in leaves(blocks) {
        model.push_boxed(layer);
    }
    model
}
fn same(a: &[InferenceOp], b: &[InferenceOp]) {
    let tensor = |a: &Tensor, b: &Tensor| {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.layout(), b.layout());
        assert!(a
            .data()
            .iter()
            .zip(b.data())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    };
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        match (a, b) {
            (
                InferenceOp::Linear {
                    weight: a,
                    bias: ab,
                },
                InferenceOp::Linear {
                    weight: b,
                    bias: bb,
                },
            ) => {
                tensor(a, b);
                tensor(ab, bb);
            }
            (InferenceOp::Scale { gain: a }, InferenceOp::Scale { gain: b }) => tensor(a, b),
            (InferenceOp::Gelu, InferenceOp::Gelu) | (InferenceOp::Relu, InferenceOp::Relu) => (),
            _ => panic!("descriptor kinds differ"),
        }
    }
}

#[test]
fn flat_original_model_uses_one_descriptor_allocation_not_one_per_leaf() {
    let model = model(16);
    let legacy = leaves(16);
    // Algorithm-level control: the previous Vec-per-leaf assembly, using the
    // same checked leaf descriptors. This is not a GPU throughput benchmark.
    let (old, old_allocations) = measured(|| {
        let mut operations = Vec::new();
        for layer in &legacy {
            operations.extend(black_box(layer).inference_ops().unwrap());
        }
        operations
    });
    let (current, allocations) = measured(|| black_box(&model).inference_ops().unwrap());
    assert_eq!(current.len(), 64);
    assert_eq!(allocations, 1);
    assert!(old_allocations >= 64);
    same(&old, &current);
    println!("descriptor allocations: previous assembly={old_allocations}, current={allocations}");
}

#[test]
fn nested_append_reuses_caller_capacity_without_leaf_allocations() {
    let mut nested = Sequential::new();
    nested.push(model(8));
    nested.push(model(8));
    let mut operations = Vec::with_capacity(65);
    let expected = nested.inference_ops().unwrap();
    for _ in 0..20 {
        operations.clear();
        operations.push(InferenceOp::Gelu);
        let (result, allocations) =
            measured(|| black_box(&nested).append_inference_ops(&mut operations));
        result.unwrap();
        assert_eq!(allocations, 0);
        assert!(matches!(operations[0], InferenceOp::Gelu));
        same(&operations[1..], &expected);
    }
    let empty = Sequential::new();
    assert_eq!(measured(|| empty.inference_ops().unwrap()).1, 0);
    println!("nested caller-capacity assembly: 0 allocations across each of 20 calls");
}

struct Legacy {
    calls: Rc<Cell<usize>>,
    reject: bool,
}
impl Module for Legacy {
    fn inference_ops(&self) -> Result<Vec<InferenceOp>, InferenceError> {
        self.calls.set(self.calls.get() + 1);
        if self.reject {
            return Err(InferenceError::UnsupportedModule("legacy fixture"));
        }
        Ok(vec![InferenceOp::Relu, InferenceOp::Gelu])
    }
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        Gelu::new().forward(&Relu::new().forward(input)?)
    }
    fn backward(&mut self, input: &Tensor, gradient: &Tensor) -> PureResult<Tensor> {
        let value = Relu::new().forward(input)?;
        Relu::new().backward(input, &Gelu::new().backward(&value, gradient)?)
    }
    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[test]
fn legacy_custom_lowering_is_called_once_and_nested_errors_restore_prefix() {
    let calls = Rc::new(Cell::new(0));
    let mut nested = model(1);
    nested.push(Legacy {
        calls: calls.clone(),
        reject: false,
    });
    assert_eq!(nested.inference_ops().unwrap().len(), 6);
    assert_eq!(calls.get(), 1);
    let mut failed = Sequential::new();
    failed.push(nested);
    failed.push(Legacy {
        calls: calls.clone(),
        reject: true,
    });
    let prefix = Tensor::from_vec(1, 1, vec![-0.]).unwrap();
    let mut operations = vec![InferenceOp::Scale {
        gain: prefix.clone(),
    }];
    assert!(failed.append_inference_ops(&mut operations).is_err());
    same(&operations, &[InferenceOp::Scale { gain: prefix }]);
    assert_eq!(calls.get(), 3);
}

#[test]
fn appending_still_checks_finite_linear_parameters_before_exposing_descriptors() {
    let mut linear = Linear::new("invalid", 3, 7).unwrap();
    linear
        .visit_parameters_mut(&mut |p| {
            if p.name().ends_with("::weight") {
                p.value_mut().data_mut()[20] = f32::NAN;
            }
            Ok(())
        })
        .unwrap();
    assert!(linear.inference_ops().is_err());
    let mut sequence = Sequential::new();
    sequence.push(Relu::new());
    sequence.push(linear);
    let mut operations = vec![InferenceOp::Gelu];
    assert!(sequence.append_inference_ops(&mut operations).is_err());
    assert_eq!(operations.len(), 1);
    assert!(matches!(operations[0], InferenceOp::Gelu));
}
