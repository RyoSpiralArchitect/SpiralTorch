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
`maturin develop -m bindings/st-py/Cargo.toml --locked --features logic,kdsl`
before re-running the suite so the latest code is measured.

## Output Buffers And Threads

The native `out=` implementations retain an exclusive PyO3 borrow for the full
operation, including while the GIL is released. This covers ordinary matmul,
SIMD-prepacked matmul, and the bias/residual ReLU and GELU variants. Concurrent
or reentrant Python access to that same output tensor raises a borrow error;
the guard is released on success and error. Input/output aliases which conflict
with an input borrow are rejected before computation.

Only `&mut Tensor` crosses the detached computation boundary. No raw-pointer
ownership bypass, process-wide lock or extra output snapshot is involved.
Unrelated Python work can still run while native math executes. Existing
allocation and copy-on-write behavior is retained; this is not a promise of
zero allocation for the whole operation. GELU output replacement retains its
existing semantics rather than introducing a new in-place kernel.

The guard protects access through the Python Tensor object, not synchronization
of external mutable DLPack aliases or unrelated devices/streams. Use separate
outputs for concurrent computations and observe those interop contracts separately.

`bindings/st-py/tests/test_tensor_out.py` runs against the actual installed wheel
and is included in the existing Python smoke job. Rust `tensor_out_guard` tests
exercise both a real competing Python thread and reentrant borrowing, verify
GIL release, retain the original output storage address, and cover error cleanup.
