# Single-Submission Terminal Capture: Correct, But Slower In Browser

**Decision: reject this scheduling optimization.** Combining the forward and
terminal snapshot copy into one submission passed correctness, but the browser
terminal workload became slower in every retained case pair. Keep this result;
the next experiment will use the same terminal API with the original ordering:
submit computation first, then create and submit the snapshot.

Frozen original runtime: `6703df35e8389c3ebb9ddf5eb099f440995a930d`.
Prototype source: `7626150d31fef4da440fedefff66c066514df48a`.
This record describes the rejected prototype, not a claim that the selected
runtime should use single-submission terminal capture.

## Change And Boundaries

Rust `Module::forward_resident_snapshot`, Python `model.forward_snapshot(x)` and
WASM `model.forwardSnapshot(input)` return an owning Tensor snapshot rather than
a GPU Tensor. This prototype reused the original NN cache and appended the
snapshot copy after the existing guard, before submitting the command buffer.
Low-level graphs exposed the equivalent `forward_tensor_snapshot` operation.
Gelu/Relu terminal calls and strided Tensor snapshots also combined their commands.
The ordinary GPU-output API remained separate; no readback staging cache was used.

Both APIs were tested with retained outputs, changed parameters, cache clear,
dropped model/device wrappers, non-contiguous inputs and retained non-finite
guards. The returned Tensor snapshot intentionally has tensor validity, not the
stage-indexed error format of the separate Graph snapshot API. Generic autograd
and ModuleTrainer were not rerouted.

## Fixed Browser Comparison

Two separately reported protocols used four matrices, nine shape/seed cases per
matrix, two warmup and nine retained intervals per route. Each interval contained
256 independent forwards, **each followed by its own completed host read**.
Every returned typed array was retained until interval end and numerically checked
outside the timer. Upload, setup and cold compilation were excluded. Context/case
order alternated, routes rotated, and no samples, slow tails or runs were removed.

One protocol compared ordinary forwarding between source versions. The other
compared original `forward` then `snapshot` with the prototype's `forwardSnapshot`.
The latter includes host wrapper differences as well as GPU scheduling. It does
not isolate GPU queue overhead. Explicit graph controls remain unnormalized.

Primary endpoint: median of 12 case-median prototype/original ratios per shape.
Lower is faster; ranges and slower-pair counts use the same 12 pairs.

| Shape | Ordinary API | Terminal API | Terminal range | Terminal slower pairs | Terminal pooled elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[2,3,7]` | 1.0070 | 1.0745 | 1.0356-1.0836 | 12/12 | 1.0929 |
| `[2,8,64]` | 1.0039 | 1.0538 | 1.0331-1.0606 | 12/12 | 1.0587 |
| `[4,8,128]` | 0.9995 | 1.0230 | 1.0086-1.0288 | 12/12 | 1.0227 |

Ordinary-path Module medians were slower in 12/12, 12/12 and 6/12 pairs, with
pooled ratios 0.9995 / 1.0063 / 0.9986. This is not a zero-regression claim.
Explicit graph control medians were 1.0047 / 0.9995 / 1.0005 in that protocol and
1.0047 / 1.0039 / 1.0043 in the terminal protocol. Full control ranges, pooled
totals and every raw interval are retained, not used to explain away the result.

Chrome `152.0.7977.84` reported `BrowserWebGpu` / `Other`, with no physical
adapter name and `crossOriginIsolated=false`. Browser physical GPU and host
exclusivity remain **UNKNOWN**. Minimum measured intervals were 58.3 ms and
58.0 ms, versus an observed smallest positive clock delta of about 0.1 ms.
That is a granularity diagnostic, not a timing uncertainty bound.

Each protocol checked 405,504 reads including warmup and 697,737,216 returned
values. Combined: **811,008 reads and 1,395,474,432 values**. Maximum absolute
reference error was `2.384185791015625e-7`.

## Separate Native Results

Four fixed native Python pairs alternated process order and included eager
PyTorch CPU/MPS controls, with MPS CPU fallback disabled. The terminal candidate
was used only on the final forward of each eight-forward burst. These burst
measurements have one final read and are not the browser completed-read workload.

| Shape | Terminal completed-read median | Range | Terminal burst median | Range |
| --- | ---: | ---: | ---: | ---: |
| `[2,3,7]` | 1.0265 | 0.6137-1.2127 | 1.0080 | 0.5978-1.1215 |
| `[2,8,64]` | 0.9880 | 0.9619-1.0019 | 0.9988 | 0.9804-1.0181 |
| `[4,8,128]` | 0.9953 | 0.9728-1.0053 | 0.9995 | 0.9868-1.0041 |

Native small-case results are noisy; no universal native improvement or
fastest-PyTorch claim follows. See [summary.json](summary.json) for paired
regression counts and the independent scalar/PyTorch controls.

## Verification And Archive

All 56 verification stages passed in 550.7 s and all 16 measurement stages in
444.3 s. Backend: 134 tests. Tensor: 449 CPU and 497 WGPU tests, with one unchanged
ignored test. NN: 707 CPU and 760 WGPU tests including parameter-storage tracking.
Python GPU suites: 61 tests. Browser Module: 224 assertions. CPU-only surfaces,
parameter handoff, generated/shipped types and learner/autograd/fusion clients
also passed. Native/browser training and VJP comparisons passed 120 eager
PyTorch 2.12.1 CPU/MPS cases, 17,136 comparisons, maximum error
`5.7220458984375e-6`.

The N-D fixture preserves 24 held captures, 48 intervening discard/read cycles,
two delayed invalid-value guards, negative-zero bits and dropped owners. Browser
Rust tested four cancelled ordinary maps plus four cancelled terminal maps;
native blocking reads do not claim cancellation coverage. Terminal snapshots
remained valid after cache clear and module destruction.

[manifest.json](manifest.json) binds 221 compressed source/input/output records:
116,710,083 raw bytes and 2,676,952 compressed bytes, round-trip checked. All 39
original/prototype frozen products were rehashed; binaries remain in the local
verification directories. The literal verification folder name `--launch` is
preserved from the original helper invocation, without renaming or rerunning it.
Both streamed browser matrices were reaggregated and checked against final rows;
native fixtures agree in non-timing case contents. This is a negative performance
record with passing correctness, not permission to discard the measurements.

The original ordering can overlap GPU computation with CPU preparation of the
snapshot. Whether that explains this result is only a hypothesis. A same-API
comparison is needed before attributing the slowdown to that mechanism. No CUDA,
Furnace run, release, push or merge is part of this record.
