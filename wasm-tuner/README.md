# WASM Tuner (skeleton)

Measure kernels in-browser (WebGPU), export `tuner.json`, then run:
```
cargo run --manifest-path tools/Cargo.toml --bin gen_heuristics_table -- tuner.json   crates/st-core/src/backend/wgpu_heuristics_generated.rs   crates/st-core/src/backend/generated/midk_heuristics_generated.rs
```
