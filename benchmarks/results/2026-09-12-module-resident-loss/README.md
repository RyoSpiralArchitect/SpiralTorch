# Ordinary MSE In Resident Learning

Source: `635acccadc437e7aca0cc5c5e18b52f564f58f5d`.
The usual Rust `MeanSquaredError` now supplies owning GPU loss and prediction
cotangent tensors to the existing resident autograd/learner. Python and WASM
call that same implementation. See [usage](../../../docs/module_resident_loss.md).

This is a working high-level Loss connection, not a new optimizer or a claim
that generic `ModuleTrainer` policies have been ported. The shared training MSE
shader is byte-identical to the parent; the new path reuses its typed Params.

## Verified Learning

The source-bound run completed all 59 stages in 570.46 seconds, including
native Rust, Python GPU/CPU-only, browser WebGPU, generated/shipped TypeScript,
and independent eager PyTorch CPU/MPS replay with MPS fallback disabled.

Each native/browser fixture trains an ordinary Scaler/Linear/GELU/ReLU Module
with N-D input `[2,3,4]` for 64 submitted updates. All GPU observations follow
the loop. Tests retain per-step snapshots/receipts, check every update against
ordinary Rust Modules, then return all seven parameter tensors to the original
Module and verify its forward output. These are small deterministic fixtures,
not LLM fine-tuning or a generalization experiment.

| Seed | Update Policy | Initial MSE | Final MSE |
| --- | --- | --- | --- |
| 17 | Exact | 0.06976920 | 0.04017514 |
| 17 | ModuleCompatible | 0.06976920 | 0.04011932 |
| 29 | Exact | 0.05806962 | 0.04194250 |
| 29 | ModuleCompatible | 0.05806962 | 0.04216456 |
| 43 | Exact | 0.07904899 | 0.04237609 |
| 43 | ModuleCompatible | 0.07904899 | 0.04223000 |

The native and browser initial/final values agree. The new Loss section has
52 PyTorch replay cases and 36,920 tensor/scalar comparisons, maximum absolute
error `4.470348358154297e-8`. Including existing graph tests: 172 cases, 54,056
comparisons, maximum error `5.7220458984375e-6`. Tolerance is `atol=2e-5`,
`rtol=2e-4`; comparison counts are not individual element counts.

Python and WASM public clients each run 32 updates for both explicit policies,
reject update 33 with an overflowing loss but finite seed, then recover on
update 34. Retained loss/gradient pairs stay valid after producer drop. Other
checks cover N-D views, explicit broadcasts, empty inputs, partial reductions,
input guard propagation, shape mismatch and committed-plan rejection.
Empty MSE is explicitly zero in SpiralTorch, unlike PyTorch mean(empty); its
independent reference override is labeled in the raw results.

## Evidence And Limits

[summary.json](summary.json) includes per-stage test counts and all trajectories.
[manifest.json](manifest.json) binds 182 compressed raw/source records to their
original and compressed hashes. The archive retains all run logs and control
samples, including preliminary logs labeled as pre-commit history. Thirty-nine
local frozen products were checked, including 22 candidate products; binaries
remain in the recorded local run directories rather than being committed.
[verification.json](verification.json) records an independent decompression,
original-file/Git-blob comparison, and product-hash check of the archive.

Native device: Apple M4/Metal. Browser: WebGPU, physical GPU UNKNOWN. Host
exclusivity UNKNOWN. Existing ordinary forward controls were retained, but this
run is not a sustained-interval performance comparison and makes no speedup
claim. The existing ignored `wgpu_frac` live-adapter test remains ignored;
resident GPU tests actually ran. No CUDA/Furnace, generic trainer migration,
optimizer-state transfer, release, push or merge is included.
