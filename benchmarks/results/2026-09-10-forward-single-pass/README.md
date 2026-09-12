# Mixed Forward Graph: One Compute Pass

The mixed resident executor now clears private pointwise flags before one
compute pass and captures those flags afterward. Previously, per-stage clears
and copies repeatedly split the computation. Shaders, accumulation choices,
logical stage/error indices, queue submissions and owning readback semantics
are unchanged. Dense-only inference and graph training scheduling are unchanged.

## Measured Result

Primary measurements use AB then BA runs of frozen baseline/candidate native
executables, release Python bindings and production WASM packages. Each route
has three warmups and nine retained rotated sample blocks. These table entries
are medians of six case medians: three seeds times two repeat rounds, in ms per
forward. Baseline and candidate use identical f32 fixture bits and plan contents.

| Shape / Blocks | Native Scalar H2H | Python Scalar H2H | WASM Scalar H2H | Candidate Torch MPS H2H | Candidate Torch CPU H2H |
| --- | --- | --- | --- | --- | --- |
| `[2,3,7]` / 2 | 0.527 -> 0.243 | 0.553 -> 0.255 | 0.500 -> 0.300 | 0.305 | 0.022 |
| `[2,8,64]` / 8 | 1.620 -> 0.424 | 1.681 -> 0.456 | 1.300 -> 0.400 | 0.640 | 0.103 |
| `[4,8,128]` / 16 | 3.286 -> 0.839 | 3.453 -> 0.913 | 3.450 -> 0.900 | 0.915 | 0.380 |

Each block is Scaler/Linear/tanh-GELU/ReLU. H2H includes input upload and owning
host output. A separate burst route keeps the same input resident for eight
independent forwards, including one final host readback. For width 64, Python's
scalar burst cost fell from 1.123 to 0.184 ms per forward. This is not an
autoregressive dependency chain or GPU timestamp-only timing.

These are bounded diagnostic timings, **not a universal PyTorch win**. Torch
uses eager addmm/bias/tanh-GELU, no `torch.compile`, and one CPU thread. The CPU
control remains fastest for these small models. The largest model's Python and
MPS H2H results overlap across seeds/runs; smaller differences should not be
overinterpreted. Register-2x2/compensated controls are retained and are slower
than scalar/sequential on most of this matrix; tiling and accumulation differ,
so the comparison cannot isolate register tiling alone.

Native/Python report Apple M4 / Metal; Torch 2.12.1 uses explicit CPU/MPS with
MPS fallback disabled. Chrome 152.0.7977.83 reports BrowserWebGpu/Other with an
empty adapter name. Browser physical GPU and macOS GPU contention are **UNKNOWN**.
Native, Python and WASM have different host representations. Python/Torch both
include list conversion; WASM returns Float32Array. Native legacy Module keeps
its normal host caches/behavior, so its difference is not transfer cost alone.

## Verification And Provenance

All 19 candidate checks and 14 repeat-driver steps passed. Verification covers
111 serial backend tests, 733 NN unit tests, 30 Python GPU tests with no skips,
536 browser assertions, native integration tests, CPU feature compilation,
backend Clippy with `-D warnings`, rustfmt and six admission tests. A targeted runtime
regression places an overflow at each of eight alternating dense/pointwise
positions: after workspace reuse/destruction, each owning capture still rejects
the invalid result, and graph snapshots retain the original stage index,
including a later ReLU that masks the value. GPU tensor guards reject too, but
do not retain a graph stage index.

The 18 timing reports contain 8,262 retained timing rows (5,508 in the primary
AB/BA repeats). These are correlated measurements, not independent test cases.
Every timed output was checked outside timing; maximum absolute error across
the captures was `4.172325134277344e-7`.

The source is a working-tree patch against
`83e2eeeb6fd3ce02450b0ebbf87af678515c94dc`. `summary.json` binds all 13 changed/new
source, test, workflow and guide files. Raw receipts retain executable/library/
WASM hashes and commands. `manifest.json` binds compressed and original artifact
bytes; binaries are not included. The added stage-index regression was verified
after the main candidate run, before timing repeats; production source hashes
remained unchanged.

An initial Python admission failure is preserved in `raw/python-torch.json.xz`:
shortest f32 JSON decimals were compared as exact f64 values. Admission now uses
f32 bit equality, not relaxed numerical tolerance; a one-ULP mutation is rejected.
The [earlier default-parallel GPU failure](../2026-09-10-resident-graph-forward/README.md)
remains unexplained. These serial checks neither fix it nor prove it pre-existing.
No unconditional merge approval, CUDA result, training result or release is claimed.

An additional strict `st-nn` example/test Clippy run failed on 22 library lints
in files byte-identical to the base commit. An advisory run then completed and
also found one unchanged shared-fixture `chunks_exact` warning. No new lint was
reported on the benchmark or added test code. The logs and base/current file
hash comparisons are retained in the lint follow-up; broad NN Clippy is **not**
claimed green, and no global lint suppression was added.

See the [API and reproduction guide](../../../docs/resident_graph_forward.md).
Local machine-specific command drivers are retained under `raw/` as evidence.
