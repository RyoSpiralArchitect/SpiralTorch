//! Checked host GELU calls; allocation/free are included, setup is not.
use serde_json::json;
use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Gelu, Module};
use st_tensor::Tensor;
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

fn main() {
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let mut cases = Vec::new();
    for (rows, cols) in [
        (1, 1),
        (1, 8),
        (1, 32),
        (1, 64),
        (8, 3072),
        (32, 3072),
        (64, 1024),
        (17, 195),
        (65, 97),
    ] {
        let values: Vec<_> = (0..rows * cols)
            .map(|i| (i % 257) as f32 / 32.0 - 4.0)
            .collect();
        let seeds: Vec<_> = (0..rows * cols)
            .map(|i| (i % 29) as f32 / 16.0 - 0.5)
            .collect();
        let input = Tensor::from_vec(rows, cols, values.clone()).unwrap();
        let seed = Tensor::from_vec(rows, cols, seeds.clone()).unwrap();
        let mut layer = Gelu::new();
        for backward in [false, true] {
            let expected: Vec<_> = values
                .iter()
                .zip(&seeds)
                .map(|(&x, &g)| {
                    let (x, g) = (f64::from(x), f64::from(g));
                    let c = (2.0 / std::f64::consts::PI).sqrt();
                    let t = (c * (x + 0.044715 * x * x * x)).tanh();
                    if backward {
                        (0.5 * (1.0 + t)
                            + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x))
                            * g
                    } else {
                        0.5 * x * (1.0 + t)
                    }
                })
                .collect();
            let mut run = || {
                if backward {
                    layer.backward(black_box(&input), black_box(&seed)).unwrap()
                } else {
                    layer.forward(black_box(&input)).unwrap()
                }
            };
            for _ in 0..3 {
                black_box(run());
            }
            let mut elapsed = Vec::new();
            for _ in 0..15 {
                let start = Instant::now();
                for _ in 0..8 {
                    black_box(run());
                }
                elapsed.push(start.elapsed().as_nanos() as f64 / 8.0);
            }
            CALLS.store(0, Ordering::Relaxed);
            BYTES.store(0, Ordering::Relaxed);
            COUNT.store(true, Ordering::Relaxed);
            let output = run();
            COUNT.store(false, Ordering::Relaxed);
            let calls = CALLS.load(Ordering::Relaxed);
            let bytes = BYTES.load(Ordering::Relaxed);
            assert_eq!(output.len(), expected.len());
            let valid = output.data().iter().zip(&expected).all(|(&a, &b)| {
                a.is_finite() && b.is_finite() && (f64::from(a) - b).abs() <= 2e-6 * (1.0 + b.abs())
            });
            assert!(valid);
            cases.push(json!({"rows": rows, "cols": cols, "backward": backward,
                "valid": valid, "elapsed_ns": elapsed, "allocation_calls": calls,
                "allocated_bytes": bytes, "output_bits": output.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>()}));
        }
    }
    println!(
        "{}",
        json!({"schema": "spiraltorch.cpu_gelu.v1", "cases": cases,
        "warmups": 3, "intervals": 15, "repetitions": 8,
        "boundary": "Row-major CPU Module GELU forward and supplied-seed VJP; no optimizer, GPU or FT claim"})
    );
}
