//! Fixed packed/unpacked CPU cases; destination allocation and prepacking are outside timing.
use serde_json::json;
use st_tensor::backend::cpu_dense;
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
    let mut cases = Vec::new();
    for (rows, inner, cols) in [
        (1, 31, 8),
        (3, 37, 13),
        (4, 5, 16),
        (7, 31, 8),
        (8, 31, 8),
        (8, 7, 12),
        (17, 37, 29),
        (32, 64, 32),
        (96, 128, 96),
        (65, 257, 49),
        (129, 129, 97),
        (129, 1025, 33),
        (1, 4096, 1),
        (16, 0, 12),
        (0, 31, 8),
    ] {
        let lhs: Vec<f32> = (0..rows * inner)
            .map(|i| ((i * 17 % 127) as f32 - 63.0) / 31.0)
            .collect();
        let rhs: Vec<f32> = (0..inner * cols)
            .map(|i| ((i * 29 % 131) as f32 - 65.0) / 37.0)
            .collect();
        let packed = cpu_dense::prepack_rhs(&rhs, inner, cols).unwrap();
        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                for k in 0..inner {
                    expected[r * cols + c] += lhs[r * inner + k] * rhs[k * cols + c];
                }
            }
        }
        for use_packed in [false, true] {
            let mut dst = vec![0.0; rows * cols];
            let mut run = || {
                if use_packed {
                    cpu_dense::matmul_packed_into(
                        black_box(&mut dst),
                        black_box(&lhs),
                        black_box(&packed),
                        rows,
                        inner,
                        cols,
                    )
                    .unwrap();
                } else {
                    cpu_dense::matmul_into(
                        black_box(&mut dst),
                        black_box(&lhs),
                        black_box(&rhs),
                        rows,
                        inner,
                        cols,
                    )
                    .unwrap();
                }
            };
            for _ in 0..3 {
                run();
            }
            let mut elapsed_ns = Vec::new();
            for _ in 0..9 {
                let start = Instant::now();
                for _ in 0..16 {
                    run();
                }
                elapsed_ns.push(start.elapsed().as_nanos() as f64 / 16.0);
            }
            CALLS.store(0, Ordering::Relaxed);
            BYTES.store(0, Ordering::Relaxed);
            COUNT.store(true, Ordering::Relaxed);
            run();
            COUNT.store(false, Ordering::Relaxed);
            let allocation_calls = CALLS.load(Ordering::Relaxed);
            let allocated_bytes = BYTES.load(Ordering::Relaxed);
            let bitwise_equal = dst
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            cases.push(
                json!({"rows": rows, "inner": inner, "cols": cols, "packed": use_packed,
                "bitwise_equal": bitwise_equal, "elapsed_ns": elapsed_ns, "repetitions": 16,
                "allocation_calls": allocation_calls, "allocated_bytes": allocated_bytes}),
            );
        }
    }
    println!(
        "{}",
        json!({"schema": "spiraltorch.cpu_dense_workspace.v1", "cases": cases,
        "rayon_threads": std::env::var("RAYON_NUM_THREADS").ok(),
        "deterministic": std::env::var("SPIRAL_DETERMINISTIC").ok(),
        "boundary": "public CPU matmul into reused output, prepacking excluded; one warmed allocation sample; nine untrimmed timing intervals"})
    );
}
