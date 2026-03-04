# Contributing to SpiralTorch

Thanks for helping improve SpiralTorch. This repo is **AGPL-3.0-or-later** licensed; please read the license obligations before redistributing derivative works.

## Quick links

- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Maintainers: `MAINTAINERS.md`

## Local development

### Prerequisites

- Rust (stable): `rustup show`
- Python (recommended for wheel/docs tooling): 3.12+
- (Optional) `just` for task shortcuts

### Common tasks (cross-platform)

```bash
# Format
cargo fmt --all

# Lint
cargo clippy --workspace --all-targets

# Core build + tests
cargo build -p st-core --release
cargo test  -p st-core --release -- --nocapture
```

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

