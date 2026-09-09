# Resident Training Forward: Measure Before Selecting A Backend Policy

Existing `Sequential`/`Linear`/`Gelu`/`Scaler`/`Relu` plans already lower to
resident N-D training. This change batches only the mixed graph's **forward**
dispatches on Metal. Loss, pointwise VJP/unbroadcast, validation copies and
prepare/vote/commit remain separate and unchanged. Specialized dense training,
ordinary `Module::forward`, autograd and `ModuleTrainer` routing are unchanged.

## Trial And Decision

The initial candidate batched both Metal and BrowserWebGpu. Two whole-process
rounds ran standard and wider matrices, with native/browser order reversed in
the second round. Baseline: `367f194a` (harness only). Trial: `6ef40783`.
The browser correctness admission page was corrected at `9e687b56`; binaries
remained bound to the same frozen trial source. No timing preceded that fix.

Each matrix has three shapes/depths and seeds 17/29/43. Every timed interval
starts from identical weights, performs **eight nonzero-rate SGD updates**, and
observes all eight losses. Two warmups precede eight retained rotated blocks
per cadence. Immediate reads each loss; deferred reads owning snapshots after
enqueue. Uploads, setup, reset and initial/final zero-rate probes are excluded.
This is not GPU timestamp-only timing or one long uninterrupted training run.

The ratios below are geometric means of 18 paired per-case median ratios
(three shapes x three seeds x two rounds), baseline time / trial time.
Greater than one favors the trial; these are correlated diagnostic observations.

| Matrix | Client | Immediate | Deferred |
| --- | --- | ---: | ---: |
| Standard | Native Metal | 1.171 | 1.210 |
| Wide | Native Metal | 1.093 | 1.104 |
| Standard | Browser WebGPU | 1.011 | 0.977 |
| Wide | Browser WebGPU | 1.006 | 0.999 |

**Select Metal batching; retain the original browser schedule.** Browser gains
were negligible and deferred reads sometimes regressed. Metal is not a win on
every sample either: standard immediate case ratios ranged from 0.772 to 1.633,
and wide immediate from 0.967 to 1.382. All regressions and outliers are retained.
Other unmeasured backends keep their existing one-dispatch-per-pass behavior.

Standard shapes/depths are `[2,16,32]/2`, `[2,129,32]/4`, `[4,32,64]/8`.
Wide adds `[4,64,64]/8`, `[2,128,128]/8`, `[2,64,256]/4`. Both use the same bounded
Rust fixture, exact mean-MSE gradients, plain SGD, register-2x2/sequential GEMMs
and tanh-GELU. The wide profile is explicit; the original matrix is unchanged.

The independent eager PyTorch 2.12.1/MPS control still wins overall: its geometric
time ratios to the trial are 0.833/0.852 (standard immediate/deferred) and
0.466/0.487 (wide). The wider gap is roughly twofold; this study does not identify
which remaining kernel/dispatch phase causes it. Torch has no matching per-stage
finite checks or atomic graph rollback, and `torch.compile` was not benchmarked.
No Torch CPU timing or universal fastest-backend claim is made.

## Correctness And Evidence

Trial verification passed 112 serial backend tests, 733 NN unit tests, six native
integration tests, 30 Python GPU tests (zero skips), native/wasm32 strict backend
Clippy, CPU feature compilation and ten benchmark-admission tests.
Native and browser also exercise 32 masked-overflow cases each: eight alternating
dense/gain positions, both gradient policies, zero/nonzero learning rates.
All parameter bits must survive rejection, and rejected snapshots/tensors retain
their guards after workspace recovery and graph drop. A later ReLU cannot hide
the earlier overflow. Independent Torch CPU/MPS replay covers 24 trajectories;
maximum absolute replay error is `1.1920928955078125e-7`.

The trial contains 3,600 timed intervals (2,880 retained), or 28,800 real updates
across repeated reset trajectories. Every loss curve and every final prediction,
input VJP, raw/effective gradient and parameter is validated. `summary.json`
reaggregates every case; `manifest.json` binds raw and compressed bytes. Build
receipts preserve clean-source commits, product hashes and commands. Native
reports Apple M4/Metal. Browser physical GPU identity and macOS GPU contention
remain UNKNOWN; a separate navigator adapter probe is not device attestation.

The initial fixture E0502 compile error and browser admission failure are retained
or explicitly summarized. The earlier one-off default-parallel GPU VJP failure
remains unexplained; these serial checks neither fix it nor prove it pre-existing.
The prior broad NN strict-Clippy failure was not resolved or rerun here. This is
not unconditional merge approval, application learning-quality evidence or CUDA
validation. No push, merge or release is implied.

Use the [training API guide](../../../docs/resident_graph_training.md).
The existing paired runners accept `--graph --matrix wide` (native) and final
arguments `graph wide` (browser); omit `wide` for the unchanged standard matrix.
The selected policy is checked separately from this non-adopted browser trial.
