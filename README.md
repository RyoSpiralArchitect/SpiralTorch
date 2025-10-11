# SpiralTorch

**SpiralTorch v1.3.21** — Rust-based tensor/autograd with optional GPU backends (WGPU / CUDA / MPS) and Python wheels (PyO3 + maturin).

- License: AGPL-3.0-or-later
- CPU builds by default; GPU paths are feature-gated: `--features wgpu`, `--features cuda`, `--features mps`.
- Build banner: ASCII art printed via `build.rs` on successful cargo build.

See `docs/RELEASE_NOTES_v1_3_21.md` for this patch's details.
