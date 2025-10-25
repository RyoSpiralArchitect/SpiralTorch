# SpiralTorch Go bindings

These bindings wrap the `spiraltorch-sys` C-ABI and expose a small Go-native
API for constructing tensors and copying data without dipping into unsafe
pointers. They are intentionally minimal and designed to grow alongside the
Julia bindings as more runtime functionality is stabilised.

## Building

Compile the shared library first:

```bash
cargo build -p spiraltorch-sys --release
```

Ensure the resulting `libspiraltorch_sys` is discoverable at runtime (e.g. by
updating `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`). You can then build Go
programs that import `github.com/spiraltorch/spiraltorch-go`:

```bash
go run ./examples/tensor_dump
```

The sample demonstrates element-wise arithmetic, Hadamard products, matrix
multiplication, transposition, and reshaping using the high-level helpers
exposed by the `Tensor` type. It also spins up the shared `GoldenRuntime` to
dispatch a `Matmul` on the same worker threads that power the Rust core.

### Custom linker flags

If the library lives in a non-standard location you can set `CGO_LDFLAGS`:

```bash
CGO_LDFLAGS="-L/path/to/target/release" go test ./...
```

### Error handling

Every FFI call updates a thread-local error slot in `spiraltorch-sys`. The Go
wrapper surfaces this as `error` values so callers can rely on idiomatic Go
control flow instead of manual error string inspection.

## Available operations

- `Add`, `Sub`, `Scale`, and `Hadamard` allocate new tensors while preserving the
  originals.
- `Matmul` executes matrix multiplication (`lhs @ rhs`) using the same backend
  selection heuristics as the Rust runtime.
- `NewTensorFromMatrix` converts a rectangular `[][]float32` into a tensor without manual flattening.
- `NewTensorFromColumns` accepts column-major data and converts it into a tensor while handling rectangular validation for you.
- `Tensor.ToMatrix` materialises tensor contents back into an owned `[][]float32` view.
- `Tensor.Columns`, `Tensor.Column`, and `Tensor.Row` expose convenient accessors for copying columnar or row slices.
- `Data` copies tensor contents into Go slices for inspection or interop.
- `Transpose` and `Reshape` provide lightweight access to shape manipulation
  without manually flattening and rebuilding tensors.
- `NewRuntime` constructs the cooperative golden runtime so long-running
  programs can reuse the same worker threads as the Rust API. Methods such as
  `Runtime.Matmul`, `Runtime.Add`, and `Runtime.Hadamard` mirror the tensor
  helpers while scheduling the work on the runtime.
- `Runtime.ParallelMatmul`, `Runtime.ParallelAdd`, and `Runtime.ParallelHadamard`
  dispatch batches of tensor pairs concurrently, automatically matching the
  runtime's worker pool (or a caller-provided concurrency limit) so heavy
  pipelines can be saturated from Go without manual goroutine orchestration.

See `examples/parallel_runtime` for a practical demonstration that executes
multiple matrix multiplications in parallel before staging a batch of element-
wise additions.
