# Julia and Go Integration Strategy

SpiralTorch already exposes Rust and Python entry points; the next expansion pushes the runtime into Julia and Go ecosystems without
compromising ergonomics or performance. This brief outlines how to incubate the integrations and graduate them into supported
paths.

## Guiding Principles
- **Native-first bindings.** Favor thin idiomatic wrappers that call into the existing Rust crates so that execution paths stay
  unified across languages.
- **Shared artifacts.** Reuse the `spiraltorch-sys` C-ABI shims, build metadata, and test fixtures to avoid duplicated
  maintenance.
- **Progressive rollout.** Start with read-only tensor inspection and inference pipelines, then expand toward training once the
  memory ownership model is proven.

## Julia Integration
1. **Package skeleton.** Bootstrap a `SpiralTorch.jl` package that loads the Rust dynamic library via `ccall`, mirroring the
   Python wheel layout.
2. **Array bridging.** Implement conversion utilities between `Array{Float32}`/`CuArray` and the SpiralTorch tensor handles.
   Validate zero-copy sharing for CPU paths using memory-mapped buffers.
3. **Autodiff hooks.** Prototype a custom ChainRules.jl rule set that forwards gradient requests into the hypergrad tape.
4. **Telemetry surface.** Wire Julia logging macros to the tracing channel exported by `st-core` so observability dashboards stay
   consistent.

## Go Integration
1. **Module layout.** Create a Go module (`spiraltorch-go`) that links against the C-ABI via cgo. Provide builders for tensors,
   trainers, and device contexts.
2. **Concurrency model.** Map the Rust async runtime into Go goroutines with a lightweight task scheduler that preserves
   cancellation semantics.
3. **Memory safety checks.** Add integration tests that fuzz ownership transitions across Go's garbage collector and the Rust
   borrow checker by stress-testing tensor lifetimes.
4. **Deployment targets.** Package minimal inference binaries as Go `main` examples to demonstrate cross-compilation for Linux,
   macOS, and ARM SBCs.

## Shared Roadmap
- Track readiness in `docs/backend_matrix.md` by adding Julia and Go columns with feature coverage.
- Extend the CI pipeline to build smoke-test artifacts for both bindings on nightly schedules.
- Document cookbook-style examples (Julia notebooks, Go command-line samples) once the APIs stabilize.

