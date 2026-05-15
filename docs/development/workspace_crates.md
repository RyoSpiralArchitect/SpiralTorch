# Workspace Crates

SpiralTorch has 34 Cargo workspace packages. `cargo build --workspace` and
`cargo check --workspace --all-targets` address the full member set, while a
plain `cargo build` follows the smaller `default-members` list in the root
`Cargo.toml`.

Use this page when a task says "all crates" or when you need to tell whether a
package is covered by the default local build surface.

## Quick Commands

```bash
# Inventory with default-member coverage.
python3 tools/list_workspace_crates.py

# Only the packages that are workspace members but not default-members.
python3 tools/list_workspace_crates.py --non-default-only

# Full Rust check surface. Requires protoc because spiral-selfsup examples
# build tboard/prost code under --all-targets.
cargo check --workspace --all-targets
```

## Members

| Package | Path | Default member | Description |
|---|---|---:|---|
| `st-core` | `crates/st-core` | yes | SpiralTorch core (distributed orchestrators, heuristics, runtime) |
| `st-backend-hip` | `crates/st-backend-hip` | yes | HIP (ROCm) backend for SpiralTorch |
| `st-backend-wgpu` | `crates/st-backend-wgpu` | no | WGPU backend kernels and pipeline loaders for SpiralTorch |
| `st-kdsl` | `crates/st-kdsl` | yes | Kernel DSL, autotuning, and self-rewrite helpers for SpiralTorch |
| `st-logic` | `crates/st-logic` | yes | Soft-logic, temporal dynamics, and quantum-reality helpers for SpiralTorch |
| `st-kv` | `crates/st-kv` | yes | Optional Redis-backed key-value helpers for SpiralTorch |
| `st-frac` | `crates/st-frac` | no | SpiralTorch fractional calculus (Grunwald-Letnikov) operators for tensors. |
| `st-tensor` | `crates/st-tensor` | yes | SpiralTorch tensor utilities: fractional calculus (GL 1D) CPU + optional WGPU kernel |
| `st-softlogic` | `crates/st-softlogic` | yes | Shared soft logic primitives for SpiralTorch heuristics |
| `st-refract` | `crates/st-refract` | yes | Lowering utilities for mapping SpiralK soft-logic directives to the kernel DSL |
| `st-sync` | `crates/st-sync` | yes | Phase synchronization helpers for merging SpiralK refract maps |
| `st-zeta` | `crates/st-zeta` | yes | Telemetry bridge for emitting SoftLogic feedback metrics |
| `st-nn` | `crates/st-nn` | yes | High-level nn.Module-style API for SpiralTorch |
| `st-qr-studio` | `crates/st-qr-studio` | yes | Quantum Reality Studio utilities for SpiralTorch |
| `st-text` | `crates/st-text` | yes | Resonance-to-language narrators for SpiralTorch |
| `st-vision` | `crates/st-vision` | yes | Z-space native vision utilities for SpiralTorch |
| `st-robotics` | `crates/st-robotics` | yes | Robotics sensor fusion and runtime utilities for SpiralTorch |
| `st-spiral-rl` | `crates/st-spiral-rl` | yes | Reinforcement learning harness for SpiralTorch |
| `st-rec` | `crates/st-rec` | yes | Recommendation and ranking harness for SpiralTorch |
| `st-bench` | `crates/st-bench` | yes | Benchmark and backend-matrix harnesses for SpiralTorch |
| `st-amg` | `crates/st-amg` | yes | Algebraic multigrid heuristics and roundtable learning hooks for SpiralTorch |
| `st-metrics` | `crates/st-metrics` | no | Lightweight registry for SpiralTorch training metrics |
| `spiral-selfsup` | `crates/spiral-selfsup` | no | Self-supervised learning objectives for SpiralTorch |
| `spiral-opt` | `crates/spiral-opt` | yes | Quantization-aware training and structured pruning utilities for SpiralTorch |
| `spiral-hpo` | `crates/spiral-hpo` | no | Deterministic hyper-parameter search strategies for SpiralTorch |
| `spiral-safety` | `crates/spiral-safety` | no | Safety policy evaluation for SpiralTorch runtime surfaces |
| `spiral-config` | `crates/spiral-config` | no | Shared runtime configuration helpers for SpiralTorch |
| `cobol_bridge` | `bindings/cobol_bridge` | no | C ABI bridge exposing SpiralTorch text resonance helpers to COBOL callers |
| `julia-ffi-poc` | `crates/julia-ffi-poc` | no | Proof-of-concept crate for embedding Julia heuristics into SpiralTorch |
| `spiraltorch-sys` | `bindings/spiraltorch-sys` | no | C-ABI shims that expose a stable surface for foreign-language bindings |
| `spiraltorch-py` | `bindings/st-py` | no | Python bindings for SpiralTorch heuristics |
| `spiraltorch-wasm` | `bindings/st-wasm` | yes | WebAssembly bindings for SpiralTorch fractal canvases and heuristics |
| `st-xai-cli` | `tools/xai-cli` | no | Batch explainability driver for SpiralTorch vision models |
| `visualize-z-volume` | `tools/visualize-z-volume` | no | CLI for rendering synthetic Z-space volumes through SpiralTorch WGPU tools |

## CI Boundary

The current CI build matrix validates `st-core`, selected upper-stack crates,
WASM, Python packaging, and lint surfaces. It does not currently run one
dedicated all-workspace `cargo check --workspace --all-targets` job for every
pull request. See `notes/open-questions/004-all-crate-ci-coverage.md` before
changing that boundary.
