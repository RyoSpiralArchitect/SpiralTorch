# Resident NeRF correctness fixture

The same Rust fixture runs a real `ResidentNerf -> ResidentGraph -> ResidentNerf`
chain on native WGPU and browser WebGPU. It covers 36 conditions: 1/65/256 rays,
1/8/64 samples, constant/varying affine fields, midpoint/seeded stratification.
Each chain makes one terminal observation, with no intermediate readback.
The fixture also requires five rejection/ownership guards. This is not a NeRF
training run or a performance benchmark. The field is an affine NN graph, not
an automatic conversion of `st_vision::NerfField` or its positional encodings.

```sh
cargo run --locked --release -p st-backend-wgpu --example resident_nerf > native.json
cargo build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown --example resident_nerf_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_nerf_browser.wasm --target web --out-dir /absolute/new/wasm-dir --out-name spiraltorch_wasm
node tools/test_resident_browser.cjs /absolute/new/wasm-dir /absolute/path/to/chrome /absolute/new/browser.json '' '' '' '' nerf
python3 -I -B benchmarks/nerf-resident-wgpu/compare.py native.json /absolute/new/browser.json > torch-control.json
```

Use a wasm-bindgen CLI matching the locked crate version. The browser driver
needs Playwright (`NODE_PATH` may point to an existing installation) and creates
an isolated headless profile, never the user's browser profile. Native tests
reject CPU adapters. The browser page requires a separate non-fallback adapter
probe and records it; that probe does not attest the Rust runtime's device.
The vendored WGPU browser adapter metadata can be masked. Native GPU unit
regressions are mandatory when requested:

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --lib nerf -- --test-threads=1
```

The emitted ray/parameter inputs, expected/actual vectors and independent
PyTorch eager CPU control remain raw replay artifacts. Publish compact results
and hashes separately. PyTorch decodes inputs as f32 and uses f64 optical-depth
prefixes/integration; native/browser shaders use f32 plus integer-defined
rounded compensated sums. Tests cover thin opacity, opaque tails and long
prefixes, but do not establish f64 equivalence over extreme/subnormal ranges.

Admission regressions need only the Python standard library:

```sh
python3 -I -B benchmarks/nerf-resident-wgpu/test_compare.py
```
