# Go ↔ Rust Bridge Proof of Concept

This example shows how a minimal Go HTTP service can exchange data with a Rust client.  It is designed to be a discussion starter for future tooling in `examples/` and `tools/`.

## Layout

```
examples/go_bridge_poc/
├── cmd/server          # Entrypoint wiring, logging, graceful shutdown
├── internal/api        # Request/response types, handlers, and tests
└── rust_client         # Standalone Rust binary that exercises the API
```

## Running the service

```
cd examples/go_bridge_poc
GO111MODULE=on go run ./cmd/server
```

The server now supports graceful shutdown on `SIGINT`/`SIGTERM` and validates prediction payloads before responding.

## Running the Rust client

In a separate shell:

```
cd examples/go_bridge_poc/rust_client
cargo run --release -- --endpoint http://127.0.0.1:8080/predict --values 1 2 3 4
```

The client will send a JSON payload and print the computed summary returned by the Go service, including minimum and maximum values.

## Testing

```bash
cd examples/go_bridge_poc
GO111MODULE=on go test ./...

cd rust_client
cargo test
```
