# Go ↔ Rust Bridge Proof of Concept

This example shows how a minimal Go HTTP service can exchange data with a Rust client.  It is designed to be a discussion starter for future tooling in `examples/` and `tools/`.

The service performs basic request validation to prevent schema drift:

- rejects non-JSON payloads and objects with unknown fields,
- ensures the `input` array is present and non-empty,
- caps payloads at 256KiB to guard against resource exhaustion,
- returns structured error messages alongside the appropriate HTTP status.

## Layout

```
examples/go_bridge_poc/
├── cmd/server          # Entrypoint wiring logging + HTTP server
├── internal/api        # Request/response types and handlers
└── rust_client         # Standalone Rust binary that exercises the API
```

## Running the service

```
cd examples/go_bridge_poc
GO111MODULE=on GOTOOLCHAIN=local go run ./cmd/server
```

The server listens on `:8080` by default. Override it with the `GO_BRIDGE_HTTP_ADDR` environment variable:

```
GO111MODULE=on GOTOOLCHAIN=local GO_BRIDGE_HTTP_ADDR=127.0.0.1:8090 go run ./cmd/server
```

Graceful shutdown is enabled – send `Ctrl+C` (SIGINT) or SIGTERM and the server drains in-flight requests before exiting.

## Running the Rust client

In a separate shell:

```
cd examples/go_bridge_poc/rust_client
cargo run --release -- --endpoint http://127.0.0.1:8080/predict --values 1 2 3 4
```

## Running tests

To exercise the Go handlers without needing external toolchain downloads, run:

```
cd examples/go_bridge_poc
GO111MODULE=on GOTOOLCHAIN=local GOPROXY=off go test ./...
```

The client will send a JSON payload and print the computed summary returned by the Go service.
