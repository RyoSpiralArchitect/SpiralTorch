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

### Custom linker flags

If the library lives in a non-standard location you can set `CGO_LDFLAGS`:

```bash
CGO_LDFLAGS="-L/path/to/target/release" go test ./...
```

### Error handling

Every FFI call updates a thread-local error slot in `spiraltorch-sys`. The Go
wrapper surfaces this as `error` values so callers can rely on idiomatic Go
control flow instead of manual error string inspection.
