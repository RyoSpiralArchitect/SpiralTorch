# Go ↔ Rust Bridge Proof of Concept

This example shows how a minimal Go HTTP service can exchange data with a Rust client.  It is designed to be a discussion starter for future tooling in `examples/` and `tools/`.

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
GO111MODULE=on go run ./cmd/server
```

## Running the Rust client

In a separate shell:

```
cd examples/go_bridge_poc/rust_client
cargo run --release -- --endpoint http://127.0.0.1:8080/predict --values 1 2 3 4
```

The client will send a JSON payload and print the computed summary returned by the Go service.
