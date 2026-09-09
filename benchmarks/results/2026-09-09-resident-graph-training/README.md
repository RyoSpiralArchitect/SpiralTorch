# Resident Graph Training: Owned Gains

Measured source: `e4ea73aee6a1f4d92b2bb6dcc893fd0dbb28cde2`.
Source tree: `6b7d1ebc0b906e63568d5fe97e022e3613a40b84`.
The native executable and browser WASM both reported that exact clean source.
Publication adds this evidence only; it is not a new measured binary.

## What Ran

The existing Rust `Sequential` lowers input Scaler -> Linear/GELU -> hidden
Scaler/ReLU -> Linear -> output Scaler into one graph-owned GPU training
transaction. Native uses Apple M4 / Metal. Browser uses Chrome 152.0.7977.83 /
BrowserWebGpu; its Rust adapter record does **not** expose hardware identity.
Do not infer a browser GPU identity or a native/browser speed comparison.

Both clients passed six fixtures: seeds 17/29/43, shapes `[4]`, `[3,4]`,
`[2,129,4]`, and both `Exact` / `ModuleCompatible` gradient policies. Each has
a zero-rate derivative probe followed by eight plain-SGD updates. All three gains
actually change. Independent ordinary Rust Module comparisons include loss,
prediction, input VJP, raw/effective parameter gradients and all updated values.
Max native CPU-reference error: `4.47e-8`.

Each client also passed standalone ReLU/GELU/Scaler without a Linear, parameter
snapshots before a step (including zero parameters), immutable delayed snapshots,
v2 weight-only export/recompile, same-device resident batch upload, and resume.

## Independent Torch Replay

PyTorch **2.12.1**, eager CPU and MPS, with `PYTORCH_ENABLE_MPS_FALLBACK=0`:

- **24** source-bound graph replays passed: 2 clients x 2 Torch devices x 6 cases.
- **5,184** tensor/scalar comparisons; max absolute error **`1.1920929e-7`**.
- Thresholds fixed before replay: `atol=2e-5`, `rtol=2e-4`.
- Exact uses mathematical mean-MSE VJPs. ModuleCompatible explicitly divides
  only gain optimizer gradients by leading-row count. It does not redefine the
  input VJP or scale Linear gradients again.

The new graph replay has no excluded case. Existing N-D/VJP regressions were
also rerun: N-D **48** checks passed, max `1.49e-8`; pointwise VJP **26** matched
checks plus **2 retained MPS empty-reference gaps**, max matched `2.26498e-5`.
Those two are the previously documented PyTorch MPS empty-placeholder assertion,
not a SpiralTorch fallback or newly waived mismatch. Full errors are retained.
The old native/browser dense learning fixtures also passed their CPU checks.

## Guards and Checks

Gain-candidate overflow rejects before any weight/bias/gain commit. Gain
unbroadcast overflow also rejects even when element contributions are finite.
All parameter bits stay unchanged, and both returned activation and input-VJP
tensors retain the failed whole-step guard. Upstream resident input failures
remain active across repeated steps; a new valid batch clears the old failure.

Local checks: 18 shared-contract tests, 110 backend tests with runtime tests
enabled, 742 NN library tests, four resident integration tests, and the final
standalone-graph integration pass. Native/WASM backend strict Clippy passes.
NN library/examples Clippy completed with existing warnings, none in new graph
files; it is not a repository-wide warning-free claim. Default and WGPU Python
binding `cargo check` passed. CI includes the real graph runtime test and the
WASM graph example build.

## Scope and Reproduction

This is **correctness and transactional-update evidence, not a throughput
benchmark**, FT quality win, generic autograd, production Python/JS graph-training
binding, resident GNN or CUDA result. Existing dense fast paths remain unchanged.
General forward-only compilation and pooled pointwise VJP scratch are future work.
See [API and commands](../../../docs/resident_graph_training.md).

`raw/*.gz` are byte-preserving deterministic gzip archives of fixture JSON,
Torch replays, compile/test logs and the frozen-run receipt. Native binaries and
WASM assets remain frozen locally; their hashes are in that receipt. The
[manifest](manifest.json) records compressed and decompressed hashes. Verify with:

```sh
shasum -a 256 -c SHA256SUMS
```
