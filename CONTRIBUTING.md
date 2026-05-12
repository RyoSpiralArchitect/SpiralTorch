# Contributing to SpiralTorch

Thanks for helping improve SpiralTorch. This repo is **AGPL-3.0-or-later** licensed; please read the license obligations before redistributing derivative works.

## Quick links

- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Maintainers: `MAINTAINERS.md`

## Local development

### Prerequisites

- Rust (stable): `rustup show`
- Rust nightly with rustfmt for workspace formatting:
  `rustup toolchain install nightly-2026-04-15 --component rustfmt`
- Python (recommended for wheel/docs tooling): 3.12+
- Protobuf compiler (`protoc`) for `--all-targets` checks that build TensorBoard examples
- (Optional) `just` for task shortcuts

### Common tasks (cross-platform)

```bash
# Format
cargo +nightly-2026-04-15 fmt --all

# Lint
cargo clippy --workspace --all-targets

# CI-equivalent lint gate
just ci-lint
# or, without just:
bash scripts/run_ci_lint_local.sh

# Core build + tests
cargo build -p st-core --release
cargo test  -p st-core --release -- --nocapture

# Full workspace inventory
python3 tools/list_workspace_crates.py
```

`just ci-lint` mirrors the CI `ubuntu / lint` job: it updates the local Rust
stable toolchain, installs the pinned nightly rustfmt, runs workspace clippy,
and then runs the strict clippy subset. The script requires `protoc`; if a
local `.buildenv/protoc-bin/protoc` shim exists, it is added to `PATH`.

### Windows helper script

If you prefer PowerShell entry points, `scripts/dev.ps1` mirrors the most common `just` tasks:

```powershell
pwsh -File scripts/dev.ps1 fmt
pwsh -File scripts/dev.ps1 core-test
pwsh -File scripts/dev.ps1 wheel
```

## Python wheels

From the repository root:

```bash
python -m pip install -U "maturin>=1,<2"
maturin build -m bindings/st-py/Cargo.toml --release --locked --features wgpu,logic,kdsl
python -m pip install --force-reinstall --no-cache-dir target/wheels/spiraltorch-*.whl
```

## CI expectations

- Keep changes scoped and well-tested for the crates/modules they touch.
- Prefer adding small regression tests when fixing a bug.
- Avoid introducing new dependencies unless they materially improve safety, performance, or maintenance.

## Documentation expectations

- Update docs when you change APIs or workflows.
- Prefer ASCII for version constraints (e.g. `>= 3.8`) to avoid encoding issues across shells.
