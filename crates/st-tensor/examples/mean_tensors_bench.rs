//! Paired CPU benchmark against the pre-extraction Golden reduction.
use serde_json::json;
use st_tensor::{mean_tensors_scaled, Layout, Tensor};
use std::{hint::black_box, time::Instant};

fn legacy(partials: &[Tensor], scale: f32) -> Tensor {
    let (rows, cols) = partials[0].shape();
    let mut accum = vec![0.0f64; rows * cols];
    for tensor in partials {
        let logical = tensor.to_layout(Layout::RowMajor).unwrap();
        for (dst, src) in accum.iter_mut().zip(logical.data()) {
            assert!(src.is_finite());
            *dst += f64::from(*src);
        }
    }
    let mut output = Vec::with_capacity(accum.len());
    for value in accum {
        let value = (value / partials.len() as f64 * f64::from(scale)) as f32;
        assert!(value.is_finite());
        output.push(value);
    }
    Tensor::from_vec(rows, cols, output).unwrap()
}

fn checksum(tensor: &Tensor) -> u32 {
    tensor.data().iter().fold(2166136261u32, |hash, value| {
        (hash ^ value.to_bits()).wrapping_mul(16777619)
    })
}

fn main() {
    let mut cases = Vec::new();
    for seed in [17, 29, 43] {
        for count in [4, 16, 64] {
            for (rows, cols) in [(1, 1025), (32, 2048), (128, 2048)] {
                for col_major in [false, true] {
                    let partials: Vec<_> = (0..count)
                        .map(|p| {
                            let tensor = Tensor::from_fn(rows, cols, |r, c| {
                                (((r * cols + c) * 17 + p * 131 + seed * 73) % 4093) as f32 / 64.0
                                    - 2046.0 / 64.0
                            })
                            .unwrap();
                            if col_major && p % 2 == 1 {
                                tensor.to_layout(Layout::ColMajor).unwrap()
                            } else {
                                tensor
                            }
                        })
                        .collect();
                    let scale = 1.25;
                    let expected = legacy(&partials, scale);
                    let actual = mean_tensors_scaled(&partials, scale).unwrap();
                    assert_eq!(
                        actual
                            .data()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .data()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    let mut legacy_ms = Vec::new();
                    let mut blocked_ms = Vec::new();
                    for sample in 0..19 {
                        for candidate in if sample % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        } {
                            let start = Instant::now();
                            let result = if candidate {
                                mean_tensors_scaled(black_box(&partials), scale).unwrap()
                            } else {
                                legacy(black_box(&partials), scale)
                            };
                            black_box(&result);
                            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                            if sample >= 3 {
                                if candidate {
                                    blocked_ms.push(elapsed);
                                } else {
                                    legacy_ms.push(elapsed);
                                }
                            }
                        }
                    }
                    cases.push(json!({"rows":rows,"cols":cols,"count":count,"seed":seed,"mixed_col_major":col_major,"scale":scale,"checksum":checksum(&actual),"legacy_ms":legacy_ms,"blocked_ms":blocked_ms}));
                }
            }
        }
    }
    println!(
        "{}",
        json!({"contract":"ordered-f64-mean-scaled.v1","execution":"native_cpu","warmup":3,"samples":16,"cases":cases})
    );
}
