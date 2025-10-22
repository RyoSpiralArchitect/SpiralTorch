# Julia and Go Integration Strategy

SpiralTorch already exposes Rust and Python entry points; the next expansion is
now grounded by a common C-ABI layer that powers both Julia and Go bindings. The
`spiraltorch-sys` crate ships a stable surface for basic tensor lifecycle
operations so higher-level packages can focus on idiomatic ergonomics without
reimplementing the runtime.

## Guiding Principles
- **Native-first bindings.** Favor thin idiomatic wrappers that call into the
  shared Rust crates so execution paths stay unified across languages.
- **Shared artifacts.** The `spiraltorch-sys` dynamic library centralises the
  ABI. Foreign bindings should link against it rather than vend bespoke shims.
- **Progressive rollout.** Start with read-only tensor inspection and inference
  workflows, then expand toward training once the ownership model is proven.

## C-ABI foundation (`bindings/spiraltorch-sys`)
- Export version queries, tensor constructors (`zeros`, `from_dense`), shape and
  element inspection, a safe data copy primitive, and arithmetic helpers that
  now include addition, subtraction, scaling, Hadamard products, transposition,
  reshaping, and matrix multiplication.
- Maintain a thread-safe error slot so foreign callers can surface rich error
  messages instead of opaque status codes.
- Provide unit tests that cover the ABI round-trip to prevent regressions before
  downstream bindings pull new releases.

## Julia integration (`bindings/julia`)
- The `SpiralTorch.jl` module loads `libspiraltorch_sys` via `ccall`, offers a
  garbage-collected `Tensor` wrapper, overloads `+`, `-`, `.*`, and `*` (matrix
  and scalar) and surfaces helpers for transposition, reshaping, and converting
  between Julia matrices and SpiralTorch tensors.
- Library discovery prefers the `SPIRALTORCH_SYS_LIBRARY` environment variable
  before falling back to bundled paths, making ad-hoc experimentation easy.
- Future steps: wire ChainRules.jl gradient definitions once the autodiff tape
  lands in the ABI, and expose telemetry streams so Julia observability stays in
  lockstep with Rust/Python.

## Go integration (`bindings/go`)
- The Go module links with cgo, wraps the tensor lifecycle behind idiomatic Go
  functions, exposes `Add`/`Sub`/`Scale`/`Hadamard`/`Matmul` as well as
  `Transpose` and `Reshape`, and provides an example program that prints tensor
  contents and composite operations.
- Runtime errors convert into `error` values, embracing Go's standard control
  flow while reusing the shared error slot from `spiraltorch-sys`.
- Future steps: model builders that mirror the Rust planner, goroutine-aware
  scheduling primitives, and deployment helpers for cross-compiling inference
  binaries.

## Shared roadmap
- Track readiness in `docs/backend_matrix.md` by adding Julia and Go columns
  with feature coverage as bindings evolve beyond tensor primitives.
- Extend the CI pipeline to build smoke-test artifacts for both bindings on
  nightly schedules.
- Document cookbook-style examples (Julia notebooks, Go command-line samples)
  once the APIs stabilise and cover more than tensor shuttling.
