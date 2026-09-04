# Tensor Backend Feature Ownership

`st-tensor` owns the meaning and execution of Tensor operations. Cargo feature
names select compiled implementations; they do not prove that an adapter exists,
that a workload is supported, or that an operation ran on a GPU.

## Build Configurations

| Selection | Compiled implementation |
| --- | --- |
| Defaults | CPU, faer, fractional WGPU, and dense WGPU |
| `--no-default-features --features cpu,faer` | CPU only, no WGPU dependency |
| `--no-default-features --features cpu,wgpu_frac` | Fractional WGPU, no dense provider |
| `--no-default-features --features cpu,wgpu_dense` | Dense and fractional WGPU, without the alias |
| `--no-default-features --features cpu,wgpu` | Compatibility alias for the dense provider |
| `--no-default-features --features cpu,wgpu_f16` | Dense provider with optional f16 shader selection |

`wgpu_dense` owns dense matrix multiplication, prepacked multiplication,
LayerNorm, attention, softmax/hardmax, and dense Tensor utilities. It controls
their Rust GPU selectors, implementation branches, and provider capability
observations. `wgpu` forwards to that feature; implementation code must not
require the alias as a second switch. Default and granular builds expose
`MatmulBackend::GpuWgpu`, `SoftmaxBackend::GpuWgpu`, and `HardmaxBackend::GpuWgpu`.

The fractional feature continues to own fractional kernels and shared GPU
dependencies. Enabling it alone must leave dense capability at `NotBuilt`.
Optional f16 compilation does not imply device support or force f16 execution.

## Runtime Evidence

`observe_tensor_execution_capability()` checks a concrete workload and runs a
provider dispatch sentinel before reporting `Ready`. The corresponding Tensor
API must remain reachable in the same feature composition. Runtime adapter and
shape checks, automatic size heuristics, fallback policy, and receipt vocabulary
are unchanged by feature ownership.

Run the feature regression on a machine with a supported GPU:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
SPIRALTORCH_EXPECT_NO_WGPU_ALIAS=1 \
cargo test --locked -p st-tensor --no-default-features \
  --features cpu,wgpu_dense --release --test wgpu_feature_ownership
```

The regression compares CPU/GPU results across all six capability families,
requires direct-WGPU completed receipts, checks the existing hardmax kernel,
and verifies that the compatibility alias was not enabled indirectly. Hardmax
currently uses its existing backend metadata, not a new execution receipt.
CI also covers defaults, the alias, f16 selection, CPU-only, and fractional-only
builds. Live GPU checks are opt-in locally and mandatory in the macOS CI step.

Higher-level crates retain their own integration features. Python's existing
`wgpu` feature already forwards the alias; this repair changes no Python feature
default or wheel policy. WASM compile checks verify the granular owner, but do
not claim new browser-side synchronous GPU execution.
