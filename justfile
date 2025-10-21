set shell := ["bash", "-cu"]

fmt:
    cargo fmt --all

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

clean:
    cargo clean

core-build:
    cargo build -p st-core --release

core-test:
    cargo test -p st-core --release -- --nocapture

wgpu:
    cargo build -p st-tensor --features wgpu --release

stack:
    cargo build -p st-nn --release && \
    cargo build -p st-spiral-rl --release && \
    cargo build -p st-rec --release

safety-suite:
    cargo test --manifest-path crates/spiral-safety/Cargo.toml

all: fmt clippy core-build core-test stack
