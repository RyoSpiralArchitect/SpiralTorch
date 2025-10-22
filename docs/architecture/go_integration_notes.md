# Go Integration Exploration

_Date: 2025-10-21_

## 1. Use-cases and Requirements

### Candidate scenarios in `examples/`
- **Edge inference demo orchestration** – provide a lightweight HTTP control plane written in Go that can dispatch jobs to Rust-based inference kernels, offering easy deployment to systems engineers familiar with Go tooling.
- **Telemetry fan-out** – create a Go sidecar in examples that exposes Prometheus metrics and forwards structured traces while Rust workers focus on compute-heavy logic.
- **Configuration playground** – expose a Go-based CLI for tweaking experiment knobs (YAML/JSON) and communicating with Rust services via HTTP for quick iteration during tutorials.

### Candidate scenarios in `tools/`
- **Pipeline automation** – Go binaries that wrap existing Rust CLIs, orchestrating long-running benchmarking workflows with concurrency primitives.
- **Monitoring adapters** – Go daemons that integrate with common observability stacks (OpenTelemetry Collector, Grafana Agent) and translate metrics emitted by Rust components.
- **Developer tooling** – lightweight Go services that expose REST/gRPC APIs for IDE plugins or local dashboards interacting with Rust libraries.

### Cross-cutting requirements
- **Deterministic builds**: lock Go toolchains via `go env -w GOTOOLCHAIN` or container images; reuse Rust's existing toolchain pinning.
- **Interoperability**: prefer protocol definitions (OpenAPI/gRPC) that can be shared; maintain JSON schema parity.
- **Operational parity**: ensure logging, metrics, and tracing conventions mirror Rust (structured logging, OTLP exporters).
- **CI friendliness**: add `go fmt`/`go test` hooks next to existing `cargo` workflows and keep runtimes below five minutes.

## 2. Proposed Go module structure and protocol choices

```
examples/go_bridge_poc/
├── cmd/            # Thin binaries (server, future CLIs)
├── internal/       # Core HTTP handlers, domain logic
├── pkg/            # (Future) reusable libraries exported to other repos
└── rust_client/    # Cross-language integration examples
```

- **`cmd/`** hosts entrypoints with wiring and environment parsing.
- **`internal/`** holds implementation details not meant for reuse outside the repository, mirroring Rust's `crate::internal` separation.
- **`pkg/`** remains empty for now but is reserved for code that may be consumed by other modules (aligning with Go's ecosystem norms).

### Protocol evaluation

| Protocol | Pros | Cons | Fit vs. Rust |
| --- | --- | --- | --- |
| JSON over HTTP (current PoC) | Human-friendly, zero extra tooling, mirrors existing REST utilities | Manual schema drift risk, no streaming | Rust has mature HTTP clients (`reqwest`, `ureq`); minimal overhead |
| gRPC with protobuf | Strong typing, streaming, language-agnostic | Requires maintaining `.proto` files, heavier CI dependencies | Rust `tonic` matches Go `grpc-go`; useful for high-throughput cases |
| Message queues (NATS/Kafka) | Decoupled, backpressure support | Operational overhead, adds infra dependency | Rust async story is solid (Tokio) but increases deployment complexity |

Near-term recommendation: keep JSON/HTTP for prototyping, graduate to gRPC once schemas stabilize or when streaming/strong contracts are required.

## 3. Proof of Concept summary

- Implemented `examples/go_bridge_poc/cmd/server` – a Go HTTP service exposing `/healthz` and `/predict` (sums numeric arrays and returns aggregate stats).
- Added `examples/go_bridge_poc/rust_client` – a Rust CLI that sends JSON payloads to the Go service and prints the response using `ureq` and `clap`.

### CI build & test memo

```
# Go formatting and tests (extend `.github/workflows/ci.yml`)
GO111MODULE=on GOTOOLCHAIN=local go fmt ./...
GO111MODULE=on GOTOOLCHAIN=local GOPROXY=off go test ./...

# Rust client lint/test (optional for nightly runs)
cargo fmt --manifest-path examples/go_bridge_poc/rust_client/Cargo.toml -- --check
cargo clippy --manifest-path examples/go_bridge_poc/rust_client/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path examples/go_bridge_poc/rust_client/Cargo.toml
```

`GOTOOLCHAIN=local` ensures CI reuses the pinned system Go toolchain instead of attempting to download version-specific toolchains, which can stall in restricted network environments.

These commands are safe to run independently of the main workspace because the client is not part of `Cargo.toml`'s workspace members.

## 4. Outcomes, gaps, and next steps

### Outcomes
- Captured concrete use-cases for adopting Go within `examples/` and `tools/` directories.
- Established an idiomatic Go module layout aligned with Rust's package organization.
- Delivered a runnable Go↔Rust PoC demonstrating request/response semantics, shared data models, and defensive request validation to surface schema drift early.

### Outstanding questions
- Authentication and authorization model for cross-language calls.
- How to share schema definitions (OpenAPI vs. protobuf) without duplicating structs.
- Whether Go code should live in the main workspace or in a dedicated submodule to isolate dependencies.

### Roadmap
1. **Schema governance** – introduce a shared OpenAPI/protobuf definition and code generation for both languages.
2. **Observability parity** – add OpenTelemetry exporters to the Go service and integrate with existing Rust tracing.
3. **CI integration** – update GitHub Actions to execute the commands listed above, including caching for Go modules.
4. **Performance evaluation** – benchmark the Go service vs. direct Rust implementations to measure overhead.
5. **Security hardening** – add TLS/mTLS support and input validation layers before exposing the bridge externally.
