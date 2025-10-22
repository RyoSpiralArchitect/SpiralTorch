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
   SpiralTorch.to_array(t)
   ```

The wrapper automatically disposes tensors via finalizers, but you can call
`SpiralTorch.clear_error!()` to reset the global error slot if you want to
inspect transient failures.
