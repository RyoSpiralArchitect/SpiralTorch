# NeRF ray integration fixture

This fixture calls the real `st-vision::NerfTrainer`, `st-nn` field and tensor
backend. It can run natively and under Node/WASM; it is not a public Python or
browser NeRF API. The Python control independently reconstructs the same field
in PyTorch and differentiates a one-step MSE update with autograd.

Use a **dedicated** Cargo target directory for this standalone workspace, not a
shared workspace/client build target. The lockfile intentionally reuses the
admitted positional-geometry fixture versions; do not refresh dependencies as
part of a numerical/performance comparison.

```sh
export CARGO_TARGET_DIR=/absolute/local/nerf-fixture-target
export CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4
cargo +1.98.0 build --locked --release --manifest-path benchmarks/nerf-ray-integral/Cargo.toml
"$CARGO_TARGET_DIR/release/nerf-ray-integral-fixture" > /absolute/local/native.json
cargo +1.98.0 build --locked --release --lib --target wasm32-unknown-unknown --manifest-path benchmarks/nerf-ray-integral/Cargo.toml
wasm-bindgen "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/nerf_ray_integral_fixture.wasm" --target nodejs --out-dir /absolute/local/nerf-wasm
node benchmarks/nerf-ray-integral/wasm.cjs /absolute/local/nerf-wasm/nerf_ray_integral_fixture.js > /absolute/local/wasm.json
python3 benchmarks/nerf-ray-integral/torch_reference.py /absolute/local/native.json > /absolute/local/torch.json
```

The bindgen CLI must be version `0.2.104`. Run workers serially without competing
builds/GPU/training loads. Preserve raw reports locally: they contain parameter,
input and output arrays. Publish compact all-condition results, validation and
SHA-256 records separately. Repeat in alternating order before drawing timing
conclusions; a single worker report is only a diagnostic.

There are 18 rendering conditions: 1/32/256 rays, 1/8/64 samples per ray,
constant/varying fields. Each condition has 5 warmups and 9 intervals of 4
renders; native/WASM timing includes result destruction, but excludes value
export, parameter setup and JSON serialization. The WASM timing includes its
JS call/free boundary. PyTorch is a vectorized eager CPU control, not its fastest
possible implementation, and omits Rust's input validation. Both use f32 field
arithmetic and f64 ray integration. Do not label timing ratios a generic backend
or quality win. The historical half-interval renderer is numerically invalid
for the full-interval task, so its speed is not a valid correctness-matched
baseline.
