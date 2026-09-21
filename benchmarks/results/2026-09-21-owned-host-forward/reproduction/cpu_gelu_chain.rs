//! Allocation stress control, not a replacement for the real Linear/MLP grid.
use serde_json::json;
use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Gelu, Module, Sequential};
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
        let input = Tensor::from_vec(rows, cols, values.clone()).unwrap();
        for depth in [1, 4, 16] {
            let mut model = Sequential::new();
            for _ in 0..depth {
                model.push(Gelu::new());
            }
            let expected: Vec<_> = values
                .iter()
                .map(|&x| {
                    let mut x = f64::from(x);
                    for _ in 0..depth {
                        x = 0.5
                            * x
                            * (1.0
                                + ((2.0 / std::f64::consts::PI).sqrt()
                                    * (x + 0.044715 * x * x * x))
                                    .tanh());
                    }
                    x
                })
                .collect();
            let run = || model.forward(black_box(&input)).unwrap();
            for _ in 0..3 {
                black_box(run());
            }
            let mut elapsed_ns = Vec::new();
            for _ in 0..15 {
                let start = Instant::now();
                for _ in 0..8 {
                    black_box(run());
                }
                elapsed_ns.push(start.elapsed().as_nanos() as f64 / 8.0);
            }
            CALLS.store(0, Ordering::Relaxed);
            BYTES.store(0, Ordering::Relaxed);
            COUNT.store(true, Ordering::Relaxed);
            let output = run();
            COUNT.store(false, Ordering::Relaxed);
            let allocation_calls = CALLS.load(Ordering::Relaxed);
            let allocated_bytes = BYTES.load(Ordering::Relaxed);
            let valid = output.data().iter().zip(&expected).all(|(&a, &b)| {
                a.is_finite() && (f64::from(a) - b).abs() <= 2e-6 * (1.0 + b.abs())
            });
            assert!(valid);
            assert_eq!(input.data(), values.as_slice());
            cases.push(json!({"rows": rows, "cols": cols, "depth": depth, "valid": valid,
                "elapsed_ns": elapsed_ns, "allocation_calls": allocation_calls, "allocated_bytes": allocated_bytes,
                "output_bits": output.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>()}));
        }
    }
    println!(
        "{}",
        json!({"schema": "spiraltorch.cpu_gelu_chain.v1", "cases": cases,
        "warmups": 3, "intervals": 15, "repetitions": 8,
        "boundary": "Borrowed-input Sequential GELU chains, output allocation/free included; an allocation stress control, not model/training throughput"})
    );
}
