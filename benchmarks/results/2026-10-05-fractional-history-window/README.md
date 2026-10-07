# Trainable Full-Normalization Windows

Source: `16379238c73f6890a4f754dccd4ad381436e047b`.
API and mathematical contract: [history windows](../../../docs/fractional_history_window.md).

Rust can select a half-open range of strictly-past GL taps AFTER computing
the original full-kernel normalization and its derivative. Python's existing
gain/angle adapters and compiled WASM expose the same operator. This allows
removing long history without silently changing the retained short taps.
The default full-history path and default adapter state remain unchanged.

## Verification

- 146 Rust tests passed, including six new window tests.
- 805 Python regressions passed, no skips, with 18 existing JIT warnings.
  The 49 new window tests cover independent Torch references, all selective
  VJPs, JVP/angle composition, buffer ownership, state separation and four
  tiny randomly initialized GPT-2 learning/resume fixtures with frozen bases.
- Nine compiled-WASM fixtures passed, including both window learning loops;
  TypeScript declarations passed strict checking.
- Native release and explicit CPU/text binding builds/checks passed.
- Strict native/WASM-core Clippy passed with Rust 1.97. The existing WGPU
  lint attributes require a newer Clippy: the WGPU-enabled WASM check passed
  using the already installed Rust 1.99. No warning gate was suppressed.
  Existing vendored-WGPU and CPU-only dead-code warnings remain.
- All 76 primary files, eight client files, 71 runtime files and two original
  derived records of the preceding angular study were checked unchanged.

The first WGPU-enabled Clippy attempt under 1.97 failed on pre-existing
unknown lint attributes. Its log hash and the successful 1.99 check are
both retained. A reversed-range test fixture was also made lint-portable
without weakening the invalid-bound assertion.

## Compiled WASM Learning

Fixed synthetic target, K=8, shape [2,12,3], 100 scalar SGD updates using
Rust angle/order and gain pullbacks. Each scalar gradient was nonzero on
all updates; these are two explicit windows, not a selected winning run.

| Window | Initial MSE | Final MSE |
| --- | ---: | ---: |
| [1,3) | 0.4922596942 | 0.0000031613 |
| [3,8) | 0.0100405262 | 0.0004226178 |

Zero-gated tail-only adapters can be dormant at integer orders where all
retained coefficients vanish, despite a nonzero order differential. This
case is tested and documented; the learning fixtures explicitly use
noninteger initialization rather than hidden jitter.

`validation.json` binds sources, native/WASM artifacts, local log hashes and
the separately preserved runtime manifest. `compiled-wasm-learning.json`
contains the numeric outcomes. Check these public records with
`shasum -a 256 -c SHA256SUMS`.

This is operator/learning-path evidence, not a pretrained-checkpoint
ablation, general LLM quality, unique parameter recovery, performance or
browser/GPU residency claim. No weights, model text, native packages or
private logs are published. The earlier angular experiment is not rerun or
rewritten; a frozen-checkpoint mechanism probe is separate future work.
