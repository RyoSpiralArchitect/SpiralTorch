# LayerNorm expansion cost, 2026-09-23

Measured source: `6dcf2885682b63463fc4f30dec0b17ce8a3afdb0`, clean and unchanged
through 13 accepted validation stages. Ordinary Tensor routing is unchanged.
This refines the [correct but expensive centered candidate](../2026-09-23-layernorm-centered-vjp/README.md),
without reducing precision, changing fixtures, or widening tolerances.

## Implementation and Validation

- Skip exactly zero significand components and empty residual expansion work.
- Magnitude-sort the operands for error-free FastTwoSum, using three rather
  than six integer-rounded additions. A 16,708-pair real-GPU test checks both
  sum and residual bits against independent unordered CPU TwoSum, including
  cancellation, subnormal values, halfway cases and deterministic random pairs.
- Compute the input-VJP scale once per row instead of repeating compensated
  division per element. Shared memory preflight is updated to 8,256 bytes;
  8,255-byte rejection and the exact boundary are tested without an adapter.
- 203 native backend + 31 contract + 11 existing autograd + 4 existing numerical
  tests passed (249 total), alongside native/WASM strict clippy and formatting.
  Three Python evidence tests also passed.
- Real Chrome 153 / Rust BrowserWebGpu passed the unchanged v2 contract: 7
  boundary cases with all 8 VJP masks, 5 scale-nullspace cases, 2 tiny-epsilon
  cases, 20 dynamic-range variants and 4 guards. The unchanged 400-step learning
  control still reaches `7.81713472e-10` from `1.98799634`, with no intermediate
  host readbacks. This is a small affine-learning control, not an LLM claim.

## Descriptive Comparison

All six shapes and five routes retain three warmups, eighteen intervals per
route, input materialization, forward, all VJPs, scale 0.5, and four owning
CPU outputs. All 540 final numerical/timing intervals passed their gates.
Medians below are milliseconds on the shared M4. This is not isolated,
interleaved multi-version performance admission; Python/ATen and Rust direct
VJP also have different host overheads.

| Shape | Rust CPU | Existing hybrid WGPU | Resident WGPU | PyTorch CPU | PyTorch MPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2x3 | 0.001 | 3.763 | 2.040 | 0.076 | 1.587 |
| 8x257 | 0.019 | 3.669 | 2.344 | 0.076 | 1.595 |
| 32x256 | 0.067 | 3.949 | 2.090 | 0.078 | 1.582 |
| 64x768 | 0.384 | 5.522 | 7.380 | 0.161 | 1.797 |
| 128x1025 | 1.025 | 8.355 | 16.204 | 0.207 | 2.101 |
| 256x256 | 0.508 | 6.212 | 7.512 | 0.164 | 2.032 |

The 128x1025 resident case decreased from the earlier correctness candidate's
70.746 ms to 16.204 ms, but **CPU and MPS still win every shape**. Resident beats
the unchanged hybrid on the three smaller shapes and loses on the three larger
ones. Do not report this as a PyTorch win, a default-route recommendation, or a
whole-training throughput result. The same numerical gates must constrain
future common-case kernels and batched terminal readbacks.

`exploratory-variants.json` publishes every condition and all 1,296 intervals
from four intermediate native variants, not just the best cell. They were
source-stable dirty-source trials; their exact patches remain in the private
raw archive and are hash-addressed. Stage commands, source identities and all
failure statuses are retained, including the temporary malformed test call.
The original scale-nullspace, epsilon and division-refinement failures remain
in this payload and the historical candidate, rather than being overwritten.

## Verification

```sh
python3 -I -B benchmarks/layer-norm-resident/test_evidence.py
python3 -I -B benchmarks/layer-norm-resident/evidence.py verify-public \
  benchmarks/results/2026-09-23-layernorm-expansion-cost
```

See the [protocol](../../layer-norm-resident/README.md) and `validation.json` for
exact replay commands. `verify RAW PUBLIC` also verifies the 140 raw hashes.
Raw logs, patches and WASM binaries remain local; compact results, validation,
source/module hashes and the high-precision oracle are public. The public
payload manifest excludes this Git-versioned README. Prior payloads are intact.
