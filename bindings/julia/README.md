# SpiralTorch.jl

A lightweight Julia wrapper around the `libspiraltorch_sys` shared library. The
module loads the SpiralTorch tensor runtime using `ccall` and exposes a
`Tensor` type that owns the underlying handle. Basic construction utilities and
conversion helpers are available so Julia integrations can exchange tensors with
Rust without manual pointer juggling.

## Quick start

1. Build the shared library:

   ```bash
   cargo build -p spiraltorch-sys --release
   ```

2. Point Julia to the compiled artifact and experiment with the wrapper:

   ```julia
   using SpiralTorch

   ENV["SPIRALTORCH_SYS_LIBRARY"] = "/path/to/libspiraltorch_sys.so"
   SpiralTorch.version()
   t = SpiralTorch.Tensor(rand(Float32, 2, 2))
   u = SpiralTorch.Tensor(ones(Float32, 2, 2))
   SpiralTorch.to_array(t + u)
   SpiralTorch.to_array(0.5f0 * t)
   SpiralTorch.to_array(t * u)
   SpiralTorch.to_array(t .* u)
   SpiralTorch.to_array(transpose(t))
   SpiralTorch.to_array(reshape(t, 4, 1))
   rt = SpiralTorch.Runtime()
   SpiralTorch.worker_count(rt)
   SpiralTorch.to_array(SpiralTorch.matmul(rt, t, u))
   ```

The wrapper automatically disposes tensors via finalizers, but you can call
`SpiralTorch.clear_error!()` to reset the global error slot if you want to
inspect transient failures.

## Supported operations

- `+` / `-` perform element-wise addition and subtraction, returning new
  `Tensor` instances.
- `*` between two tensors dispatches to matrix multiplication while scalar
  multiplication scales individual elements.
- `.*` performs Hadamard (element-wise) multiplication.
- `transpose(tensor)` surfaces the backend implementation without copying on the
  Julia side.
- `reshape(tensor, rows, cols)` returns a view with the requested shape,
  mirroring the Rust API expectations.
- `to_array(tensor)` materialises the data into a Julia `Matrix{Float32}` for
  native manipulation.
- `Runtime(; worker_threads, thread_name)` mirrors the Rust “golden runtime” so
  Julia callers can reuse the cooperative worker pool. Methods such as
  `SpiralTorch.matmul(runtime, lhs, rhs)` and `SpiralTorch.scale(runtime, tensor,
  value)` dispatch work through the runtime instead of the direct blocking path.
- `SpiralTorch.parallel_matmul`, `SpiralTorch.parallel_add`,
  `SpiralTorch.parallel_hadamard`, and `SpiralTorch.parallel_scale` spread
  batches of operations across Julia threads while respecting the runtime's
  cooperative worker pool, making it easy to saturate heavy pipelines without
  building your own task orchestration.
- `SpiralTorch.roundtable_classify` mirrors the Rust roundtable heuristic so you
  can split gradient magnitudes into Above/Here/Beneath bands and inspect the
  aggregate energy carried by each.

### Parallel helpers

To fan out a batch of multiplications, combine Julia threading with the runtime
helpers:

```julia
using SpiralTorch

rt = SpiralTorch.Runtime()
pairs = [(rand(Float32, 64, 64), rand(Float32, 64, 64)) for _ in 1:8]
results = SpiralTorch.parallel_matmul(rt, first.(pairs), last.(pairs); threads=4)
for tensor in results
    display(size(SpiralTorch.to_array(tensor)))
end
```

Pass `materialize=true` to `parallel_*` functions when you need eager Julia
arrays instead of lazy tensors.

### Roundtable classification

Feed any gradient vector into `roundtable_classify` to inspect how the
roundtable scheduler would split its mass:

```julia
using SpiralTorch

gradient = [0.9f0, 0.1f0, 0.5f0, 0.05f0, 0.2f0, 0.8f0, 0.4f0, 0.02f0]
bands, summary = SpiralTorch.roundtable_classify(gradient; above=2, here=3, beneath=2, tolerance=0.01f0)
println(bands)
println(summary)
```
