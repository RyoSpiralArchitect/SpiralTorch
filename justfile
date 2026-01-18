set shell := ["bash", "-cu"]

fmt:
    cargo fmt --all

clippy:
    cargo clippy --workspace --all-targets

clippy-strict list="configs/clippy_strict_packages.txt":
    bash scripts/run_rust_clippy_strict.sh "{{list}}"

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

selfsup-train:
    cargo run -p spiral-selfsup --example selfsup_train --release

selfsup-eval:
    cargo test -p spiral-selfsup -- --nocapture

safety-suite:
    cargo test --manifest-path crates/spiral-safety/Cargo.toml

all: fmt clippy core-build core-test stack

setup-julia:
    ./scripts/setup_julia_env.sh

setup-go:
    ./scripts/setup_go_env.sh

julia-lint:
    ./scripts/run_julia_checks.sh lint

julia-test:
    ./scripts/run_julia_checks.sh test

go-lint:
    ./scripts/run_go_checks.sh lint

go-test:
    ./scripts/run_go_checks.sh test

distributed-selfsup config="configs/selfsup_distributed.toml":
    echo "Launching distributed self-supervised trainer with config: {{config}}" && \
    cargo test --manifest-path crates/spiral-selfsup/Cargo.toml --test distributed_selfsup -- --ignored

docs-check:
    PYTHONNOUSERSITE=1 python3 tools/check_example_gallery.py
    PYTHONNOUSERSITE=1 python3 tools/run_readme_python_blocks.py
    PYTHONNOUSERSITE=1 cargo run -p st-bench --bin backend_matrix_md -- --check --doc docs/backend_matrix.md
