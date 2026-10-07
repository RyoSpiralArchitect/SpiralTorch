# Captured Topos Pullbacks In Rust NN

The CPU `st_nn::ToposResonator` now uses the same Rust-owned finite-Picard tape
as the Python/WASM kernel clients. Previously, module backward ran the whole
recurrence twice: once for gradients and once for their audit. A captured
`vjp_audited` instead uses immutable saved sensitivity, retains finite-gradient
and amplification checks, and computes identical audit values. External WGPU
results still receive independent formula-comparison audits.

## Matched CPU Measurements

Apple M4, Rust 1.97 release, float32, two Torch threads, deterministic periodic
inputs. Both methods compute the same finite map and both full per-element
gradients, not a broadcast/reduced gate. Native Rust includes its semantic
audits and parameter-gradient accumulation; Torch does not perform those audits.
This is native Rust NN versus Python Torch dispatch, not the Python bridge.

Four process pairs per condition follow AB / BA / BA / AB. Each Rust process
is followed by an independent Torch check/timing process using the exact saved
input bytes. Both rotate forward and forward+backward order, with two warmups
and twelve measured rounds each. All 64 reports and 1,536 timing samples are
retained. File transport, process startup, validation and gradient reset are
outside timing. There is no timing pass/fail threshold or confidence interval.

Medians of four process medians, milliseconds:

| Shape | Iterations | NN forward before | NN forward after | NN forward + VJPs before | NN forward + VJPs after | Torch forward + VJPs before | Torch forward + VJPs after |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 x 64 | 4 | 0.021969 | 0.023011 | 0.067667 | 0.030959 | 0.204604 | 0.217802 |
| 256 x 768 | 4 | 2.116104 | 2.245104 | 6.455188 | 3.024437 | 4.306750 | 4.386417 |
| 32 x 64 | 16 | 0.062604 | 0.069240 | 0.328010 | 0.077104 | 0.781208 | 0.787823 |
| 256 x 768 | 16 | 5.891958 | 6.670094 | 31.208885 | 7.442292 | 18.745979 | 18.828719 |

Coupling is 0.25 / 0.75 for four / sixteen iterations; saturation 1.0 and
porosity 0.2. Forward+both-VJP medians are about 2.1-4.3x faster than the old NN
path in these cases. **Forward alone is about 5-13% slower.** The four-vector
tape and independent returned Tensor also retain more storage than the old
Tensor-sharing cache; no peak-memory improvement is claimed. Stateless core
forward and Python no-grad remain uncaptured alternatives.

All old/new input, gate, upstream, output and both gradient hashes match exactly
within each condition. Forward/backward audit fields also match. Independent
Torch errors are at most 5.97e-8 for output, 3.58e-7 for input gradient and
4.77e-7 for gate gradient, within rtol=5e-4 / atol=3e-5. This bounded CPU result
does not establish general-library speed, accelerator gains, geometric quality
benefits or language-model throughput. No pretrained training or rescoring ran.

## Validation And Preserved Failure

- All 1,024 st-core and 762 default-feature st-nn library tests pass.
- Saturated/unsaturated audited VJPs, repeated cache reuse, failure recovery,
  parameter invalidation and every state of 100 synthetic SGD updates are
  compared against recomputation. This is connectivity, not a quality study.
- Native WGPU Topos tests execute both forward routes and cross-route backward
  on real hardware; external-executor auditing is not removed.
- NN-enabled wasm32 release builds. Scalar WASM core probes are separate from
  the native NN measurement: all 24 cases, 240 updates, 54 guards and two
  ownership cases match the prior runtime. This is not browser NN throughput.
- Scoped st-core wasm32 Clippy passes. A broader strict st-nn/WGPU Clippy run
  fails on 23 diagnostics in fourteen unchanged files; no workspace-wide lint
  success is claimed. Five malformed/no-clobber reference checks reject under
  Python `-O` too.

Independent source review identified a test-order weakness: cross-route replay
combined both backend gradients and overwrote the original WGPU backward audit
before comparison. The follow-up compares each original route first and tests
accumulation separately. It changes tests only, not the measured production code.

The first baseline probe rejected its own gate-gradient reference before any
measurement report was accepted. Initial gradient allocation preserved negative
zero, whereas later zero-reset accumulation added to positive zero. Exactly
74 of 2,048 entries differed, all only in the sign of zero. Both probes were
rebuilt with an explicitly zero-initialized accumulator before reference and
measurement. No tolerance or equality check was relaxed. The rejected plan,
client/binary hashes and diagnostic receipt remain in the archive; raw logs and
binaries remain local. Earlier studies are untouched.

## Identity And Reproduction

Candidate source: `60eb80fbf0e8103ee348a50d65a4c744196c0f03`.
Baseline source: `e8313540f6a955c965d4b08045b3496e6a72b591`.
Use the exact probe/reference clients identified in `results.json` with both
revisions. The client files were added after the candidate implementation commit.
`measurements.json.gz` contains the original plan and raw numeric JSON strings,
including every pair and both Torch controls. Hash records bind the retained
files but are not independent proof of execution history or pre-run chronology.

```bash
cargo build --locked --offline --release -p st-nn --example topos_module_probe
# Preserve each resulting binary before building the other revision.
"$PROBE" 256 768 16 .75 12 "$NEW_REPORT"
OMP_NUM_THREADS=2 "$PYTHON" -P -B tools/benchmark_topos_module_reference.py \
  "$NEW_REPORT" "$NEW_TORCH_REPORT"
cargo test --locked --offline --release -p st-core --lib
cargo test --locked --offline --release -p st-nn --lib
cargo test --locked --offline --release -p st-nn --features wgpu \
  --lib topos -- --nocapture
cargo build --locked --offline --release -p spiraltorch-wasm \
  --target wasm32-unknown-unknown --features nn
```

Follow the archived complete pairing plan, not only the large case above.
Reports and their `.f32le` companions must use fresh paths. No weight, corpus,
old archive or build cache was deleted.
