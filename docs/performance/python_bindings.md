# Python binding microbenchmarks

The `st-bench` harness now includes a Criterion suite that exercises the
high-level `spiraltorch.Tensor` API from Python.  The goal is to keep an eye on
cross-language call overhead and provide a quick litmus test for the native
bindings compared to the pure-Python stub fallback.

## What is covered

* **Matrix multiplication** via `Tensor.matmul` for three square matrix sizes.
* **Element-wise products** via `Tensor.hadamard` spanning medium to large
  matrices.

Both families reuse deterministic pseudorandom inputs so that repeated runs can
be compared directly.

## Running the benchmarks

```bash
cargo bench --bench python_bindings -- --sample-size 30
```

When the compiled extension from `bindings/st-py` is not available, the loader
falls back to the pure-Python stub implementation.  The benchmark labels expose
which path was exercised using the module version string—for example
`native_v0.1.0` for the optimized bindings or `stub_v0.0.0+stub` when the stub is
active.

If you are iterating on the Rust extension, rebuild it via
`maturin develop -m bindings/st-py/Cargo.toml --locked --features wgpu,logic,kdsl`
before re-running the suite so the latest code is measured.
