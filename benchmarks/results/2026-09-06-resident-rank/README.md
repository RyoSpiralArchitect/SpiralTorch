# Resident Rank: Execution And Timing Diagnostics

## Confirmed Execution

- `rank-comparison-review-ec182d4d.json`: 36/36 strict CUDA/WGPU cases passed
  against PyTorch on RTX 5090; clean source/build identity checks passed.
- `resident-rank-comparison-166984f2.json`: first persistent-buffer baseline,
  54/54 cases passed, with clean source/build binding.
- `resident-rank-comparison-mid-f2b1fbfe.json` and its `mid-repeat` companion:
  parallel MidK, 54/54 cases passed in each run with clean source/build binding.
- `browser-midk.json`: real Chrome WebGPU, 15/15 cases and 340 assertions;
  adapter backend is `BrowserWebGpu`, but the browser does not expose its name.
- `browser-rejected-nan-constant.json` and `browser-rejected-uniformity.json`
  preserve the failures that motivated the shared WGSL fixes. These are not
  successful executions or benchmark samples.

The source commit is embedded in each native result. Browser evidence records
WASM/generated-asset/page hashes, not a native source-binding attestation.
Python wheel execution separately passed 58 tests (one optional test skipped).

## Timing Is Provisional

All three resident runs used identical requests:
`0e80062e7078ed0ad2ffd7e36845288b98a543125decd05d8f693da3eb1caa79`.
Each crosses three seeds, three tiles, TopK/MidK/BottomK, and widths 256/2048.

The table shows medians across the nine seed/tile medians for MidK. Units are
milliseconds per operation; intervals enqueue 16 operations then wait for
completion. These include host API overhead and are not GPU-event timings.

| Width | Initial resident | Parallel MidK | Parallel repeat | PyTorch CUDA repeat |
| --- | ---: | ---: | ---: | ---: |
| 256 | 0.2130 | 0.0594 | 0.0503 | 0.0136 |
| 2048 | 1.5780 | 0.1400 | 0.1058 | 0.0150 |

**Do not treat these deltas as certified speedups.** Another CUDA process was
observed on Furnace during this experiment, and PyTorch controls changed by
roughly 3-5x in the first post-change run. No exclusive reservation or process
gate existed in these retained runs. Their numerical and source-integrity
checks remain valid, but timing isolation is unproven. The runner now rejects
foreign compute PIDs at preflight and postflight; no further timing run was
launched after adding that guard. Other jobs were not stopped or modified.

PyTorch remains faster in the repeated resident API measurement. Persistent
buffers alone gave only a small host-to-host improvement in the first run;
amortizing submissions is a different measurement boundary, not a like-for-like
host-call speedup. The useful algorithmic change is replacing MidK's serial
discard-to-center merge with parallel rank lookup for up to 32 sorted tiles.

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| rank-comparison-review-ec182d4d.json | 0260eaa7c5212e8c602effd9be7bab125243d41dc59e4f87fbdeb8f9605b8e5b |
| resident-rank-comparison-166984f2.json | 221fcd8bb6230866a8b48622db48e52c73c85eea0939ca15e0cab452b4c6effc |
| resident-rank-comparison-mid-f2b1fbfe.json | 80d986777f4f534d8b470a9c99599536b63fa12cf2a013195acb02fb4c8fe729 |
| resident-rank-comparison-mid-repeat-f2b1fbfe.json | 8b63fb4d2eab22065e57f17e2d99d86713006bda813eb84744fc17138508e1fb |
| browser-midk.json | ebd3faf418f2849350d4b313e3e83e9c1f376df4f935725838fbe9d63856bd9b |
| browser-rejected-nan-constant.json | 5dddfe0aee3ccf42d9956ddf6d3518fc91ed7c078c647adf4db6962ffd6b8261 |
| browser-rejected-uniformity.json | ad8fd5d6854cd04b6f9de497b89bf2f7a21f78f70895707d0c9bfa8114e28371 |

See [the API and measurement guide](../../../docs/performance/resident_rank.md)
for reproduction commands and ownership/lifetime boundaries.
