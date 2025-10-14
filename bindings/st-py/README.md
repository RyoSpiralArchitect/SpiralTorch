# SpiralTorch Python bindings

This package exposes a thin, dependency-light bridge to SpiralTorch's
unified heuristics so Python callers can inspect or reuse the same
Rust-first planning logic.

The bindings are intentionally minimal—no NumPy or PyTorch shims—so the
wheel can stay lightweight while still collaborating with the Rust
runtime.
