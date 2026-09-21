//! Real Linear/MLP calls and the layout preparation they consume.
use serde_json::{json, Value};
use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Gelu, Linear, Module, Sequential};
use st_tensor::{
    AttentionBackend, LayerNormBackend, Layout as TensorLayout, MatmulBackend, PackedB,
    SoftmaxBackend, Tensor, TensorUtilBackend, Tile,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

struct Allocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
fn count(size: usize) {
    if COUNT.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

enum Output {
    Tensor(Tensor),
    Pack(PackedB),
}
impl Output {
    fn values(&self) -> &[f32] {
        match self {
            Self::Tensor(tensor) => tensor.data(),
            Self::Pack(pack) => pack.as_slice(),
        }
    }
}

fn measure(mut case: Value, expected: &[f64], mut run: impl FnMut() -> Output) -> Value {
    for _ in 0..3 {
        black_box(run());
    }
    let mut elapsed = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        for _ in 0..2 {
            black_box(run());
        }
        elapsed.push(start.elapsed().as_nanos() as f64 / 2.0);
    }
    CALLS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNT.store(true, Ordering::Relaxed);
    let output = run();
    COUNT.store(false, Ordering::Relaxed);
    let calls = CALLS.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    let values = output.values();
    assert_eq!(values.len(), expected.len());
    let valid = values.iter().zip(expected).all(|(&a, &b)| {
        a.is_finite() && b.is_finite() && (f64::from(a) - b).abs() <= 1e-4 + 1e-4 * b.abs()
    });
    assert!(valid, "independent f64 reference failed: {case}");
    case["valid"] = json!(valid);
    case["max_abs"] = json!(values
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f64::from(a) - b).abs())
        .fold(0.0f64, f64::max));
    case["f32_reference_bits_equal"] = json!(values
        .iter()
        .zip(expected)
        .all(|(&a, &b)| a.to_bits() == (b as f32).to_bits()));
    case["elapsed_ns"] = json!(elapsed);
    case["allocation_calls"] = json!(calls);
    case["allocated_bytes"] = json!(bytes);
    case
}

fn fixture(len: usize, modulus: usize, denominator: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (i % modulus) as f32 - (modulus / 2) as f32)
        .map(|value| value / denominator)
        .collect()
}

fn layer(name: &str, inner: usize, cols: usize) -> Linear {
    let mut layer = Linear::new(name, inner, cols).unwrap();
    layer
        .visit_parameters_mut(&mut |parameter| {
            let (r, c) = parameter.value().shape();
            let values = if parameter.name().ends_with("::weight") {
                fixture(r * c, 17, 256.0)
            } else {
                fixture(c, 7, 128.0)
            };
            parameter.load_value(&Tensor::from_vec(r, c, values)?)
        })
        .unwrap();
    layer
}

fn reference(input: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let weights = fixture(inner * cols, 17, 256.0);
    let bias = fixture(cols, 7, 128.0);
    let mut output = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for k in 0..inner {
                sum += input[r * inner + k] * f64::from(weights[k * cols + c]);
            }
            output[r * cols + c] = sum + f64::from(bias[c]);
        }
    }
    output
}

fn main() {
    let mut cases = Vec::new();
    for (rows, inner, cols) in [
        (1, 64, 64),
        (8, 768, 3072),
        (32, 768, 3072),
        (64, 256, 1024),
        (17, 137, 195),
        (65, 1025, 97),
    ] {
        let weight = Tensor::from_vec(inner, cols, fixture(inner * cols, 17, 256.0)).unwrap();
        let column = weight.to_layout(TensorLayout::ColMajor).unwrap();
        let mut transposed = vec![0.0; inner * cols];
        for r in 0..inner {
            for c in 0..cols {
                transposed[c * inner + r] = f64::from(weight.data()[r * cols + c]);
            }
        }
        let original: Vec<_> = weight.data().iter().map(|&v| f64::from(v)).collect();
        for operation in ["pack", "transpose", "column_transpose_pack"] {
            cases.push(measure(
                json!({"operation": operation, "rows": rows, "inner": inner, "cols": cols}),
                if operation == "column_transpose_pack" {
                    &original
                } else {
                    &transposed
                },
                || match operation {
                    "pack" => Output::Pack(
                        PackedB::from_tensor(black_box(&weight), Tile::col_major()).unwrap(),
                    ),
                    "transpose" => Output::Tensor(
                        weight
                            .transpose_with_backend(TensorUtilBackend::Cpu)
                            .unwrap(),
                    ),
                    _ => Output::Pack(
                        PackedB::from_tensor_transpose(black_box(&column), Tile::col_major())
                            .unwrap(),
                    ),
                },
            ));
        }
        let input_values = fixture(rows * inner, 13, 64.0);
        let input = Tensor::from_vec(rows, inner, input_values.clone()).unwrap();
        let wide: Vec<_> = input_values.iter().map(|&v| f64::from(v)).collect();
        let first = reference(&wide, rows, inner, cols);
        let hidden: Vec<_> = first
            .iter()
            .map(|&v| {
                0.5 * v
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
            })
            .collect();
        let second = reference(&hidden, rows, cols, inner);
        for (backend, label) in [
            (MatmulBackend::Auto, "auto"),
            (MatmulBackend::CpuFaer, "faer"),
            (MatmulBackend::CpuSimd, "cpu_simd"),
        ] {
            let _guard = push_backend_policy(BackendPolicy::explicit(
                DeviceCaps::cpu(),
                backend,
                backend,
                LayerNormBackend::Cpu,
                AttentionBackend::Cpu,
                SoftmaxBackend::Cpu,
            ));
            for mlp in [false, true] {
                let mut model = Sequential::new();
                model.push(layer("first", inner, cols));
                if mlp {
                    model.push(Gelu::new());
                    model.push(layer("second", cols, inner));
                }
                for invalidate in [false, true] {
                    cases.push(measure(json!({"operation": if mlp {"mlp"} else {"linear"},
                        "rows": rows, "inner": inner, "cols": cols, "backend": label, "invalidate": invalidate}),
                        if mlp { &second } else { &first }, || {
                            if invalidate {
                                model.visit_parameters_mut(&mut |parameter| {
                                    // Force the real Parameter finite/pack caches to be refreshed.
                                    let _ = parameter.value_mut();
                                    Ok(())
                                }).unwrap();
                            }
                            Output::Tensor(model.forward(black_box(&input)).unwrap())
                        }));
                }
            }
        }
    }
    println!(
        "{}",
        json!({"schema": "spiraltorch.cpu_nn_layout.v1", "cases": cases,
        "warmups": 3, "intervals": 9, "repetitions": 2, "tolerance": {"atol": 1e-4, "rtol": 1e-4},
        "rayon_threads": std::env::var("RAYON_NUM_THREADS").ok(),
        "boundary": "CPU-only real Sequential Linear/GELU calls, output allocation and free included; invalidation models cache refresh without an optimizer update; no FT or GPU performance claim"})
    );
}
