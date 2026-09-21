#![cfg(not(target_arch = "wasm32"))]

use st_core::backend::device_caps::DeviceCaps;
use st_nn::execution::{push_backend_policy, BackendPolicy};
use st_nn::{Linear, Module};
use st_tensor::{
    AttentionBackend, LayerNormBackend, Layout, MatmulBackend, SoftmaxBackend, Tensor,
};

fn assert_values(actual: &Tensor, expected: &[f64]) {
    assert_eq!(actual.data().len(), expected.len());
    for (&value, &reference) in actual.data().iter().zip(expected) {
        assert_eq!(f64::from(value), reference);
    }
}

#[test]
fn linear_layouts_refresh_forward_and_transposed_gradient_packs() {
    let (rows, inner, cols) = (5, 33, 65);
    let x: Vec<_> = (0..rows * inner)
        .map(|i| ((i % 13) as f32 - 6.0) / 16.0)
        .collect();
    let g: Vec<_> = (0..rows * cols)
        .map(|i| ((i % 7) as f32 - 3.0) / 16.0)
        .collect();
    let input = Tensor::from_vec(rows, inner, x.clone()).unwrap();
    let grad = Tensor::from_vec(rows, cols, g.clone()).unwrap();
    for backend in [
        MatmulBackend::Auto,
        MatmulBackend::CpuFaer,
        MatmulBackend::CpuSimd,
    ] {
        let _guard = push_backend_policy(BackendPolicy::explicit(
            DeviceCaps::cpu(),
            backend,
            backend,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        ));
        for layout in [Layout::RowMajor, Layout::ColMajor] {
            let mut layer = Linear::new("test", inner, cols).unwrap();
            for revision in [0, 1] {
                let w: Vec<_> = (0..inner * cols)
                    .map(|i| ((i % 17) as f32 - 8.0 + revision as f32) / 32.0)
                    .collect();
                let weights = Tensor::from_vec(inner, cols, w.clone())
                    .unwrap()
                    .to_layout(layout)
                    .unwrap();
                layer
                    .visit_parameters_mut(&mut |parameter| {
                        parameter.zero_gradient();
                        if parameter.name().ends_with("::weight") {
                            parameter.load_value(&weights)?;
                        }
                        Ok(())
                    })
                    .unwrap();
                let mut y = vec![0.0; rows * cols];
                let mut dx = vec![0.0; rows * inner];
                let mut dw = vec![0.0; inner * cols];
                let mut db = vec![0.0; cols];
                for r in 0..rows {
                    for c in 0..cols {
                        db[c] += f64::from(g[r * cols + c]);
                        for k in 0..inner {
                            y[r * cols + c] +=
                                f64::from(x[r * inner + k]) * f64::from(w[k * cols + c]);
                            dx[r * inner + k] +=
                                f64::from(g[r * cols + c]) * f64::from(w[k * cols + c]);
                            dw[k * cols + c] +=
                                f64::from(x[r * inner + k]) * f64::from(g[r * cols + c]);
                        }
                    }
                }
                assert_values(&layer.forward(&input).unwrap(), &y);
                assert_values(&layer.forward(&input).unwrap(), &y);
                assert_values(&layer.backward(&input, &grad).unwrap(), &dx);
                assert_values(layer.weight().gradient().unwrap(), &dw);
                assert_values(layer.bias().gradient().unwrap(), &db);
            }
        }
    }
}
