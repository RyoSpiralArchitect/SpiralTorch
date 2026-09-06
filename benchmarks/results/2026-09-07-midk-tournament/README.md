# Cached MidK Tournament: Native and Browser Controls

Rust now selects a cached tile-head winner tournament for MidK with 33-256
tiles and `k > 1`. After the existing exact prefix seek, only the winning
tile's head and its ancestor path change: O(k log tiles) updates instead of
reloading/reducing all heads for every retained rank. This is still exactly
two dispatches, with no intermediate host readback or CPU fallback.

MidK with at most 32 tiles keeps the parallel path. More than 256 tiles,
TopK/BottomK, and single-winner large MidK keep their existing paths.
The old shader is split into a shared prelude and legacy body; their assembled
bytes are **identical** to main before this change:
`d2c2430ef6e86077770c574b347d64f1508d0a2b08df2e901e166e5a6c803804`.
The tournament compiles in a separate module, shares ordering/seek primitives,
and fits the existing 8 KiB workgroup budget. Python/WASM use this same Rust
selection; no second frontend policy, default tile change, or global execution
serialization was added. Pipeline creation cost is not measured here.

## Native Results

The measured source is `1ed7d2f394cf19ad13b33c47b14c3bd4fc4ab2af`; baseline is
`5caf31829ecc668493e2a08d832b7e611745fd10`, whose executable code equals main
`9cdc222489a9dd397eb26ea02c5d1f24b56caeec` (only evidence/docs differ).
Later changes add the matched browser fixture, `k=2` regression coverage,
and this evidence; the measured runtime is unchanged.

Furnace's RTX 5090 passed 204 final cases: two 72-case many-row runs, a 24-case
boundary run, and the normal 36-case UCB/Thompson comparison. There are 6,144
correctness-gated adaptive observations. Controls are separate from policy
feedback, use rotating candidate order, and retain 12 batches after two warmup
batches. Each batch is **16 resident sort+merge operations plus completion**.
Allocation, uploads, control sweeps, policy work, and validation readbacks are
outside adaptive timing. Values AND stable source indices are checked.

Below are candidate/control **mean** latency ratios, medians over seeds
17/29/43 in each repetition. These are three seeds repeated twice, not six
independent seeds. Below 1 is faster; all rows are MidK with `k=65`.

| Rows | Columns | Tile | Run 1 | Run 2 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 8193 | 128 | 0.756 | 0.754 |
| 2 | 8193 | 256 | 0.758 | 0.755 |
| 64 | 8193 | 128 | 0.762 | 0.771 |
| 64 | 8193 | 256 | 0.763 | 0.763 |
| 512 | 1055 | 32 | 0.774 | 0.773 |

At rows=2, cols=8193, tile=256, the boundary probe's ratios for k=1/7/65/256
are 1.001/0.963/0.759/0.585. This diagnoses whole dispatches, not an isolated
merge-stage speedup or training quality.

The 36-case standard suite's hindsight-best WGPU fixed controls still take
**1.66-5.40x** their PyTorch 2.13.0+cu132 CUDA references by median latency.
This is not a general PyTorch win. WGPU/Vulkan and CUDA run in separate
process blocks on the same named RTX 5090, never overlapping; this is not
cross-framework interleaving or GPU-event timing.

## Browser Measurement and Validation

Separate browser sessions drifted enough to reverse small-case comparisons.
Those raw reports remain diagnostics, **not admitted performance evidence**.
One early run overlapped a formatting check and is explicitly labeled.

The new `rank-matched` fixture loads frozen baseline/candidate WASM in one
isolated browser, with separate devices and sequential execution. Each case
alternates baseline-first/candidate-first equally, warms four paired batches,
then retains 16 pairs of **64 operations plus a four-byte completion fence**.
Every batch is checked outside timing. Both module trees and the fixture are
hashed. The deterministic input includes ties, signed zero, and NaNs.

Two 44-case matched runs passed. Across the nine new tile-count conditions:

| k | Run 1 Mean-Ratio Range | Run 2 Mean-Ratio Range |
| ---: | ---: | ---: |
| 7 | 0.919-0.944 | 0.925-0.947 |
| 65 | 0.666-0.701 | 0.667-0.696 |
| 256 | 0.524-0.547 | 0.524-0.540 |

A separate 44-case baseline-vs-itself run yielded ratios 0.978-1.024.
Single-winner and legacy boundary controls remain in the reports; they are
not removed to improve the table. These are two runs of one fixed browser
fixture, not independent seeds. The browser adapter is anonymous: neither
physical GPU identity nor browser/native timing equivalence is claimed.

Additional final checks passed:

- Browser rank boundary/lifecycle suite: 54 cases in each of two runs.
- Browser SpiralK/Black Cat adaptation: 18 cases and 432 observations.
- Rust live WGPU backend: 17 tests on macOS and 17 on Furnace/Linux.
- Packaged Python on Apple M4/Metal: 24 passed, one optional PyTorch test skipped.
- Strict backend/test Clippy on both hosts, formatting, and WASM/WebGPU
  all-targets check. Furnace's private toolchain initially lacked Clippy;
  installing that component and rerunning succeeded. No system toolchain changed.

## Negative Results and Reproduction

`5eca8b88` is valid negative evidence: adding the tree to the legacy entry
slowed many-row parallel MidK by roughly 30-35%. Separating the entry at
`e688e3f0` reduced but did not eliminate that effect. `4e03ff78` restored kind
specialization. The final module split additionally preserves legacy shader
bytes rather than relying on cross-entry dead-code elimination.
We do not attribute separate-session browser drift to a compiler defect.

At rows=512, cols=1023, tile=32 the final native ratios are 1.033/1.014 for
k=7 and 1.021/1.026 for k=65 across the two runs. Thus the large regression
is gone, but this is **not a promise of zero slowdown on every shape**;
individual seed ratios remain in the summary (up to 1.077 for that k=65 cell).

[summary.json](summary.json) revalidates all 20 native runs, regenerates their
inputs/request hashes, checks canonical outputs and Rust feedback receipts,
and retains every per-case ratio, identity, and raw-file hash.
[raw-logs.tar.xz](raw-logs.tar.xz) includes unmodified successful and negative
reports, build/test logs, product hashes, diagnostic recipes, and `analyze.py`.
Python/WASM source labels use retained build logs and product hashes, not an
embedded Git attestation; native executables validate their embedded clean
commit/tree against the frozen checkout and copied executable hash.

Archive SHA-256: `15bc11daa4fb4c1afd8c718d0acf23f0c45447aa5a8ee8b22b6084c8d932d2cb`.
Summary SHA-256: `688a0aa5694810adb33aff6d60a02d6d3a93f53c40aaa55a86ef1560b8e8b373`.
Extraction and recomputation reproduced the summary byte-for-byte.

After extracting the archive, recompute with:

```bash
python -I analyze.py --repo /path/to/SpiralTorch --output recomputed.json
```

Native recipes call the existing `tools/bench_resident_rank_adaptation_vs_torch.py`
from an exact clean checkout with its copied release executable. The included
`probe-rank-batches.py` and `probe-midk-boundary.py` accept `--repo`,
`--executable`, and a new `--output`. Build each executable with
`cargo build --locked --release -p st-core --features wgpu-rt,kdsl --example resident_rank_adaptation_bench`.
Do not treat a shared Cargo target as source identity.

For isolated browser replay, build each checkout's WASM with `webgpu`, run
matching `wasm-bindgen --target web`, and use the checked-in runner:

```bash
node tools/test_resident_browser.cjs CANDIDATE_WEB CHROME NEW_OUTPUT \
  '' '' '' '' rank-matched BASELINE_WEB
```

Set `NODE_PATH` to an isolated Playwright installation if needed. Supplying
the same baseline directory for both modules reproduces the A/A control.
Do not run another GPU workload or same-host build during these measurements.
