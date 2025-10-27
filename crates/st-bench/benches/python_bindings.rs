use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::PyErr;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn ensure_python_paths(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let path: &PyList = sys.getattr("path")?.downcast()?;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "could not determine repository root from CARGO_MANIFEST_DIR",
            )
        })?;

    let binding_root = repo_root.join("bindings").join("st-py");
    for entry in [repo_root, binding_root.as_path()] {
        let entry_str = entry
            .canonicalize()
            .unwrap_or_else(|_| entry.to_path_buf())
            .to_string_lossy()
            .into_owned();

        let mut found = false;
        for item in path.iter() {
            let existing: String = item.extract()?;
            if existing == entry_str {
                found = true;
                break;
            }
        }

        if !found {
            path.insert(0, entry_str)?;
        }
    }

    Ok(())
}

fn load_tensor_class() -> (Py<PyAny>, String) {
    Python::with_gil(|py| -> PyResult<_> {
        ensure_python_paths(py)?;
        let module = py.import("spiraltorch")?;
        let version = module
            .getattr("__version__")
            .ok()
            .and_then(|obj| obj.extract().ok())
            .unwrap_or_else(|| "unknown".to_string());
        let flavor = if version.to_lowercase().contains("stub") {
            "stub"
        } else {
            "native"
        };
        let tensor = module.getattr("Tensor")?;
        Ok((tensor.into(), format!("{flavor}_v{version}")))
    })
    .expect("failed to import spiraltorch.Tensor")
}

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| rng.gen_range(-0.5_f32..0.5_f32))
        .collect()
}

fn bench_tensor_matmul(c: &mut Criterion, tensor_cls: &Py<PyAny>, flavor: &str) {
    let mut group = c.benchmark_group("py_tensor_matmul");
    let cases = [
        (64usize, 64usize, 64usize, 11u64),
        (128, 128, 128, 19u64),
        (256, 256, 256, 37u64),
    ];

    for &(m, k, n, seed) in &cases {
        let lhs_data = random_matrix(m, k, seed);
        let rhs_data = random_matrix(k, n, seed + 1);
        let (lhs, rhs) = Python::with_gil(|py| -> PyResult<_> {
            let lhs = tensor_cls.call1(py, (m, k, &lhs_data))?;
            let rhs = tensor_cls.call1(py, (k, n, &rhs_data))?;
            Ok::<_, PyErr>((lhs.into(), rhs.into()))
        })
        .expect("failed to allocate python tensors for matmul");

        let bench_id = BenchmarkId::new(flavor, format!("{m}x{k}_by_{k}x{n}"));
        group.bench_function(bench_id, move |b| {
            b.iter(|| {
                Python::with_gil(|py| -> PyResult<()> {
                    let lhs_ref = lhs.as_ref(py);
                    let rhs_ref = rhs.as_ref(py);
                    let _ = lhs_ref.call_method1("matmul", (rhs_ref,))?;
                    Ok(())
                })
                .expect("python matmul invocation failed");
            });
        });
    }

    group.finish();
}

fn bench_tensor_hadamard(c: &mut Criterion, tensor_cls: &Py<PyAny>, flavor: &str) {
    let mut group = c.benchmark_group("py_tensor_hadamard");
    let cases = [
        (256usize, 256usize, 5u64),
        (512, 512, 17u64),
        (1024, 1024, 29u64),
    ];

    for &(rows, cols, seed) in &cases {
        let lhs_data = random_matrix(rows, cols, seed);
        let rhs_data = random_matrix(rows, cols, seed + 1);
        let (lhs, rhs) = Python::with_gil(|py| -> PyResult<_> {
            let lhs = tensor_cls.call1(py, (rows, cols, &lhs_data))?;
            let rhs = tensor_cls.call1(py, (rows, cols, &rhs_data))?;
            Ok::<_, PyErr>((lhs.into(), rhs.into()))
        })
        .expect("failed to allocate python tensors for hadamard");

        let bench_id = BenchmarkId::new(flavor, format!("{rows}x{cols}"));
        group.bench_function(bench_id, move |b| {
            b.iter(|| {
                Python::with_gil(|py| -> PyResult<()> {
                    let lhs_ref = lhs.as_ref(py);
                    let rhs_ref = rhs.as_ref(py);
                    let _ = lhs_ref.call_method1("hadamard", (rhs_ref,))?;
                    Ok(())
                })
                .expect("python hadamard invocation failed");
            });
        });
    }

    group.finish();
}

fn python_tensor_benchmarks(c: &mut Criterion) {
    let (tensor_cls, flavor) = load_tensor_class();
    bench_tensor_matmul(c, &tensor_cls, &flavor);
    bench_tensor_hadamard(c, &tensor_cls, &flavor);
}

criterion_group!(python_binding, python_tensor_benchmarks);
criterion_main!(python_binding);
