//! Fixed CPU forward-only cases for older-source optimizations.
//! Run the same harness on both revisions; a false numerical gate is retained.

use serde_json::{json, Value};
use spiral_selfsup::contrastive::{
    info_nce_loss, info_nce_loss_tensor, info_nce_loss_tensor_as_result,
};
use st_frac::mellin_types::ComplexScalar;
use st_frac::{fractal_field::FractalFieldGenerator, mellin::MellinLogGrid};
use st_tensor::{Layout, Tensor};
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

struct CountedAllocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

fn count(size: usize) {
    if COUNT.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: AllocLayout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;

fn measure<T>(mut run: impl FnMut() -> T, repetitions: usize) -> Value {
    for _ in 0..3 {
        drop(black_box(run()));
    }
    let mut elapsed_ns = Vec::with_capacity(9);
    for _ in 0..9 {
        let start = Instant::now();
        for _ in 0..repetitions {
            drop(black_box(run()));
        }
        elapsed_ns.push(start.elapsed().as_nanos() as f64 / repetitions as f64);
    }
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNT.store(true, Ordering::Relaxed);
    let output = black_box(run());
    COUNT.store(false, Ordering::Relaxed);
    drop(output);
    json!({
        "elapsed_ns": elapsed_ns, "repetitions": repetitions,
        "rust_allocation_calls": ALLOCATIONS.load(Ordering::Relaxed),
        "rust_allocated_bytes": BYTES.load(Ordering::Relaxed),
    })
}

fn fixture(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 48) as i32 - 32768) as f32 / 32768.0
                })
                .collect()
        })
        .collect()
}

fn tensor(rows: &[Vec<f32>], layout: Layout) -> Tensor {
    Tensor::from_vec(
        rows.len(),
        rows[0].len(),
        rows.iter().flatten().copied().collect(),
    )
    .unwrap()
    .to_layout(layout)
    .unwrap()
}

fn reference(a: &[Vec<f32>], p: &[Vec<f32>], normalize: bool) -> (Vec<f64>, f64) {
    let norm = |row: &[f32]| {
        row.iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt()
            .max(f64::from(f32::EPSILON))
    };
    let mut logits = Vec::with_capacity(a.len() * p.len());
    for anchor in a {
        for positive in p {
            let dot: f64 = anchor
                .iter()
                .zip(positive)
                .map(|(&a, &p)| f64::from(a) * f64::from(p))
                .sum();
            let denom = if normalize {
                norm(anchor) * norm(positive)
            } else {
                1.0
            };
            logits.push(dot / denom / f64::from(0.3f32));
        }
    }
    let loss = logits
        .chunks(a.len())
        .enumerate()
        .map(|(i, row)| {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max - row[i] + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
        })
        .sum::<f64>()
        / a.len() as f64;
    (logits, loss)
}

fn validation(logits: &[f32], loss: f32, expected: &(Vec<f64>, f64)) -> Value {
    let max_abs = logits
        .iter()
        .zip(&expected.0)
        .map(|(&a, &b)| (f64::from(a) - b).abs())
        .fold(0.0, f64::max);
    let loss_abs = (f64::from(loss) - expected.1).abs();
    let passed = logits.len() == expected.0.len()
        && logits
            .iter()
            .zip(&expected.0)
            .all(|(&a, &b)| a.is_finite() && (f64::from(a) - b).abs() <= 1e-4 + 1e-4 * b.abs())
        && loss.is_finite()
        && loss_abs <= 1e-4 + 1e-4 * expected.1.abs();
    json!({"passed": passed, "max_logit_abs": max_abs, "loss_abs": loss_abs, "loss": loss, "reference_loss": expected.1})
}

fn main() {
    let label = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "unspecified".into());
    let mut cases = Vec::new();
    for (batch, dims) in [(8, 31), (32, 64), (96, 128)] {
        let anchors = fixture(batch, dims, 17);
        let positives = fixture(batch, dims, 29);
        for normalize in [false, true] {
            let expected = reference(&anchors, &positives, normalize);
            let result = info_nce_loss(&anchors, &positives, 0.3, normalize).unwrap();
            cases.push(json!({
                "kind": "info_nce", "api": "vector", "layout": "rows", "batch": batch, "dims": dims, "normalize": normalize,
                "validation": validation(&result.logits, result.loss, &expected),
                "measurement": measure(|| info_nce_loss(black_box(&anchors), black_box(&positives), 0.3, normalize).unwrap(), 32),
            }));
            for (name, layout) in [
                ("row_major", Layout::RowMajor),
                ("col_major", Layout::ColMajor),
            ] {
                let a = tensor(&anchors, layout);
                let p = tensor(&positives, layout);
                let result = info_nce_loss_tensor(&a, &p, 0.3, normalize).unwrap();
                cases.push(json!({
                    "kind": "info_nce", "api": "tensor", "layout": name, "batch": batch, "dims": dims, "normalize": normalize,
                    "validation": validation(result.logits.data(), result.loss, &expected),
                    "measurement": measure(|| info_nce_loss_tensor(black_box(&a), black_box(&p), 0.3, normalize).unwrap(), 32),
                }));
                let result = info_nce_loss_tensor_as_result(&a, &p, 0.3, normalize).unwrap();
                cases.push(json!({
                    "kind": "info_nce", "api": "tensor_as_result", "layout": name, "batch": batch, "dims": dims, "normalize": normalize,
                    "validation": validation(&result.logits, result.loss, &expected),
                    "measurement": measure(|| info_nce_loss_tensor_as_result(black_box(&a), black_box(&p), 0.3, normalize).unwrap(), 32),
                }));
            }
        }
    }
    for len in [16, 4096, 65536] {
        for (octaves, iterations) in [(1, 1), (4, 16)] {
            let base = MellinLogGrid::from_function(-2.0, 4.0 / len as f32, len, |x| {
                ComplexScalar::new(x, -x * 0.5)
            })
            .unwrap();
            let generator = FractalFieldGenerator::new(octaves, 2.0, 0.5, iterations).unwrap();
            let branch = generator
                .branching_field(base.log_start(), base.log_step(), len)
                .unwrap();
            let result = generator.weave_with_grid(&base).unwrap();
            let bitwise_equal = result
                .samples()
                .iter()
                .zip(base.samples())
                .zip(&branch)
                .all(|((&a, &b), &c)| {
                    let expected = b + c;
                    a.re.to_bits() == expected.re.to_bits()
                        && a.im.to_bits() == expected.im.to_bits()
                });
            cases.push(json!({
                "kind": "fractal_weave", "len": len, "octaves": octaves, "iterations": iterations,
                "validation": {"passed": bitwise_equal, "bitwise_equal": bitwise_equal},
                "measurement": measure(|| generator.weave_with_grid(black_box(&base)).unwrap(), 4),
            }));
        }
    }
    println!(
        "{}",
        json!({
            "schema": "spiraltorch.source_crosscut.v1", "label": label, "cases": cases,
            "timing_boundary": "CPU forward API including output allocation and drop; prebuilt inputs; no backward",
            "allocation_boundary": "one warmed call; Rust global allocator requests including realloc sizes, not peak live memory or GPU allocations",
            "fixture": {"anchor_seed": 17, "positive_seed": 29, "temperature": 0.3f32},
            "tolerance": {"atol": 1e-4, "rtol": 1e-4},
        })
    );
}
