# Class-Last Loss In Resident Graph Learning

Verified source: `c6df288631c4ce35e58ce9da149f6bec867eade5`.
Feature commit: `318a1072d095efb0772760ef25d723b777673b4c`.
The ordinary Rust `CrossEntropyWithLogits` now supplies owning GPU loss and
prediction cotangent tensors to the existing resident VJP/learner. Python and
WASM call that implementation through shared Rust label/configuration contracts.
See the [usage guide](../../../docs/module_resident_classification.md).

This connects high-level classification learning without reading intermediate
tensors back to the CPU. It does not automatically migrate `pure::Tensor`,
generic `ModuleTrainer` policies or optimizer state to the GPU. Unlike the
preceding MSE connection, classification has a new numerical kernel; only the
existing MSE training shader is unchanged from the feature base.

## Verified Learning

All 65 source-bound stages passed in 573.58 seconds: native Rust, Python GPU and
CPU-only builds, browser WebGPU, generated/shipped TypeScript, and independent
eager PyTorch 2.12.1 CPU/MPS replay with MPS fallback disabled. The unchanged
guide code separately ran 32 updates on the frozen GPU library and returned
weights to the original Module with zero maximum output difference.

Each native/browser fixture trains an ordinary Scaler/Linear/GELU/ReLU Module
on N-D input `[2,3,4]` for 64 submitted updates, with no GPU observations inside
the loop. Retained per-step loss, VJP and update snapshots are checked against
ordinary Rust Modules; all seven parameter tensors are explicitly returned to
the original Module and its forward output checked. Native values:

| Seed | Smoothing | Update Policy | Initial CE | Final CE |
| --- | --- | --- | --- | --- |
| 17 | 0.0 | Exact | 1.08621693 | 0.48732197 |
| 17 | 0.0 | ModuleCompatible | 1.08621693 | 0.67033654 |
| 29 | 0.2 | Exact | 1.07864499 | 1.01682317 |
| 29 | 0.2 | ModuleCompatible | 1.07864499 | 1.02857375 |
| 43 | 0.5 | Exact | 1.14764738 | 1.07099199 |
| 43 | 0.5 | ModuleCompatible | 1.14764738 | 1.07223308 |

All six browser cases also reduce the loss; final values differ from native by
at most `5.960464477539063e-8`. Smoothing changes with seed in this fixture:
this is contract coverage, not an ablation of smoothing, policy superiority,
generalization or LLM fine-tuning quality.

The classification section has 188 matched PyTorch cases and 37,240 tensor or
scalar comparisons. Including existing graph/loss tests: 360 cases and 91,296
comparisons. The maximum absolute error is `1.811981201171875e-5`, from the
50,257-class probe against CPU f32; tolerance is `atol=2e-5`, `rtol=2e-4`.
Counts are comparisons of tensors/scalars, not individual elements.

Each native/browser route includes 43 numerical probes. Extreme finite logit
gaps and tiny smoothing use explicitly labeled PyTorch CPU-f64 references;
these are not silent MPS fallbacks. The `[80,0]` tiny-tail probe is checked
relatively against `exp(-80)` and excluded from matched-comparison counts:
SpiralTorch preserves loss `1.804851041598915e-35` and the negative target
gradient, whereas eager PyTorch f32 CPU/MPS returns zero loss and target
gradient. Four route/reference records retain that precision difference.

Python and WASM clients each exercise three reductions and two update policies:
32 updates, rejection of invalid-label update 33 even at zero rate, and recovery
on update 34 without parameter corruption. Other checks cover N-D views,
broadcasts, ignore-ID rounding boundaries, WASM BigInt range/type rejection,
empty/all-ignored reductions, inherited guards, overflow, shape mismatch,
committed-plan rejection and retained output lifetimes. Labels use finite
integral f32 transport, not arbitrary-width integer tensor storage.

## Evidence And Limits

[summary.json](summary.json) contains trajectories, reference gaps, client
results, per-stage test counts and 46 checked local product hashes, including
22 candidate products. [manifest.json](manifest.json) binds 310 compressed
raw/source records, 111,554,971 uncompressed bytes, to their original and packed
hashes. Binaries remain in the recorded local directories, not in Git.
[verification.json](verification.json) records the separate decompression,
original-file/Git-blob comparison and frozen-product integrity check.

The first source-bound run failed at stage 39 because the type-test helper
assumed a no-argument constructor. The follow-up commit only fixes that helper;
the new full run passes. Both runs, the earlier rejected wide-value prototype,
parser/fixture failures and the documentation helper's single-use-receipt error
remain archived and labeled. They are not rewritten as successful runs.

Native device: Apple M4/Metal. Browser: WebGPU, physical GPU UNKNOWN. Host
exclusivity UNKNOWN. Existing ordinary forward controls were retained, but this
is not sustained-interval performance evidence or a speedup claim. The existing
ignored `wgpu_frac` live-adapter test remains ignored; the resident GPU suites
actually ran. No CUDA/Furnace, generic trainer migration, optimizer-state
transfer, release, push or merge is included.
