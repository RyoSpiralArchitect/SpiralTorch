# Portable softmax finite-domain repair and barrier comparison

Measured implementation: `a009340809c176491881e03ce7141dca88ec12d6`.
Hardware/runtime: shared Apple M4, native Metal, Chrome 153.0.8010.53 WebGPU,
PyTorch 2.12.1 eager CPU/MPS (4 intra-op / 1 inter-op threads).
Native exercised workgroup and subgroup correctness; browser exercised only
workgroup. Its separate nonfallback probe does not attest the Rust device.

## Outcome

The old `-1e30` maximum initializer returned **0 instead of 1** for the finite
one-element input `[-f32::MAX]` on the real M4 GPU. The native regression
receipt in exploration preserves that failure, as well as fixture-development
compile/lint errors and the separately corrected f64-buffer test bug.
The fix passes native GPU, strict high-level Tensor, and browser domain gates.
The row-subgroup alias now selects one row per workgroup; both 16-byte aliases
are tested independently of the canonical 32-byte ABI.

The synchronization comparison is deliberately between two **corrected**
implementations. The control adds back two entry barriers; the candidate
retains the first barrier inside each reduction. Both use explicit layouts
and reusable inputs/outputs. Embedded construction removes the filesystem
requirement and shares the same shader semantics on native/WASM.

Final clean-source results retain all **3,888 intervals**, 12 shape/mode
conditions, bursts 1/4, three rotating serial rounds, and nine paired blocks.
The maximum absolute error against the independent oracle is
`2.9802322387695312e-8`; maximum tolerance-scaled error is `0.051752` (gate <= 1).
The separate exploratory 3,888 intervals are retained, not pooled.

| Runtime | Geometric mean control/candidate | Range across 24 cells | Cells above 1 |
| --- | ---: | ---: | ---: |
| Native | 1.001099 | 0.929378 to 1.040693 | 13/24 |
| Browser | 1.008157 | 0.833333 to 1.333333 | 7/24 |

**No robust speedup is established.** Screening was native 1.006985 and browser
0.972367; the browser direction changes between phases. Its coarse clock
quantizes many sub-millisecond intervals. Retain the correctness repair,
portable constructor, and redundant-synchronization cleanup, not a general
performance claim. Per-condition Torch CPU/MPS measurements, including CPU
wins on small inputs, remain in `results.json`.

This is an owning-output application-path comparison, not isolated kernel
time, training, model quality, or universal superiority over PyTorch. Repeated
operations reuse the input and only the last output is observed; this does
not demonstrate a resident attention/NN chain or migrate host Tensor/autograd.

## Validation And Replay

Twenty clean-source stages passed: format; six protocol mutation/oracle tests;
two archive tests; native/WASM strict backend Clippy; 185 real-GPU-enabled
backend tests plus shader syntax; strict Tensor softmax regression; native and
WASM builds/freeze/bindgen; all nine runtime rounds. Actual GPU regressions and
compact archive verification are included in CI.

See [the full protocol and replay commands](../../softmax-portable/README.md).
`source.json` and `validation.json` identify the exact source, commands,
thread limits, stage timings and successful exits. Raw arrays and executables
remain local: 474 files / 1,018,270,732 bytes at publication, recorded by hash
and size in `local-raw-manifest.json`. All failed recorded attempts are included
in `exploration.json`; no unfavorable condition is discarded.

```sh
python3 -I -B benchmarks/softmax-portable/softmax_evidence.py verify benchmarks/results/2026-09-22-softmax-portable
```

Add `--raw-root RAW` to verify all local bytes and recompute the summaries,
or `--source-root CHECKOUT` to check the measured source files. Verification
does not rerun numerical/GPU computation. Replay from the implementation
commit above, not from an arbitrary later branch.
