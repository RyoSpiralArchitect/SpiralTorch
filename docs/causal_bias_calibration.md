# One-time causal geometry strength calibration

Different distance rules can create differently sized attention biases even
with identical raw gains. Before widening geometry comparisons, match a
specified forward-strength statistic without adding normalization to every
training step. This is a Rust-owned initialization transform, not a learned
router, optimizer, or second training implementation.

## Statistic and API

For each block and head, collect its geometry-owned scores on **preselected
training windows**, in `[batch,heads,time,time]` order. For each query row q,
subtract the mean over keys `k <= q`, sum squared residuals and divide by the
total number of causal pairs. Take the square root after aggregating all
calibration batches. Diagonal entries and the zero-energy first row count in
the denominator. Future entries are finite-checked but do not contribute.

Row centering removes offsets that cannot change that row's softmax. Pair-count
weighting prevents small batches from receiving the same weight as large ones.
`CausalBiasMoments::merge` requires the same head count and window length; it
does not establish window identity. Callers must preserve window IDs/order,
model identity, fitted gains and the selected error tolerance.

```rust
use st_kernel_contracts::causal_bias::{CausalBiasMoments, CausalBiasScaleMatch};

let target = CausalBiasMoments::from_scores(shape, &reference_scores)?;
let mut candidate = CausalBiasMoments::from_scores(shape, &candidate_scores)?;
// Merge additional matching calibration batches into BOTH collections.
// candidate.merge(&next_candidate)?;
let fit = CausalBiasScaleMatch::new(&target, &candidate, &old_raw_gains, 1e-5)?;
let new_raw_gains = fit.raw_gains();
```

Candidate scores must already include their old positive `softplus(raw_gain)`.
For each head the new positive gain is the old gain times
`reference_rms / candidate_rms`. Stable log-domain inversion avoids exp overflow
and checks the error after rounding back to f32. Zero signal (including
zero/zero), invalid shapes, nonfinite values, inconsistent coverage and
unrepresentable fits return errors rather than silently clipping. Identity
fits preserve raw gain bits, including signed zero. Failed merges are atomic.

The host calculation does **not** guarantee GPU-rounded RMS. Re-forward the
same windows on the fitted plan and require the preselected realized-RMS gate
before accepting an experiment. A failure is not permission to iteratively
retune the tolerance, select easier windows or use held-out data.

## Resident byte decoder

`ResidentByteDecoderForward::geometry_pair_bias(block)` returns a borrowed
resident tensor for the model-owned geometry only. It excludes external pair
bias, z-bias and Q/K content scores. Ordinary models and nonexistent blocks
return `None`. The accessor itself performs no readback.

The current calibration recipe explicitly snapshots those full tensors to the
host **once before training**. Accumulation and fitting use the shared Rust
contract on native and WASM. It is not a GPU-optimized reduction and makes no
speed claim. Normal forward/backward/SGD remains device-resident.

For a frozen host plan, `plan.causal_geometry()` exposes its geometry. Apply one
fitted gain vector per block with
`plan.with_initial_geometry_raw_gains(&per_block_raw_gains)`. This consumes a
host plan, validates all block/head sizes and finite values, and changes only
gains. It is not a mutation of a live model or a gradient update. Other weights,
metric, curvature, layout and parameter count are preserved; ordinary plans
reject this operation.

Compile the fitted plan, then use ordinary byte-loss forward/backward/SGD.
`initial_checkpoint()` captures the actual fitted values at revision zero.
Later model checkpoints retain the trained values and attempted revision; a
restored model does **not** recalibrate. No new checkpoint schema is needed.
Calibration metadata and training-window provenance remain caller-owned.

## Independent native and browser qualification

Generate a new local independent CPU-f32 PyTorch reference before device runs:

```sh
python3 -I -B tools/generate_resident_byte_geometry_torch_fixture.py \
  /local/new-bias-scale-reference.json --calibrate-flat
cargo run --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric -- /local/new-bias-scale-reference.json \
  --calibrate > /local/native-bias-scale.json
python3 -I -S -B tools/verify_byte_bias_scale.py \
  /local/new-bias-scale-reference.json /local/native-bias-scale.json \
  /local/new-native-comparison.json
```

The frozen synthetic recipe fits flat `4 * squared_chord` to Poincare scores,
using two training batches, B=2/T=4/H=2. One model has a single block with zero
Q/K scores; the other has two blocks, ordinary QKV, external biases and Topos.
The actual post-fit relative-RMS gate is `1e-5`. Every original VJP, all 16
SGD updates and exact same-runtime resume at update 7 retain the existing
`3e-6 + 5e-5*abs(reference)` and geometry relative-L2 `0.002` gates. Changing
terminal targets or external biases cannot change the owned score observations.

Build the [shared browser example](resident_byte_decoder.md#reproducible-verification)
and generate bindings into `target/resident-byte-decoder-web`. Copy the frozen
reference there as `bias-scale-reference.json`, serve the repository only on
loopback and open `crates/st-nn/tests/byte_bias_scale_browser.html`. The page
displays the fixture SHA-256 and downloads the original Rust JSON unchanged.
Use the same independent verifier on that download. New raw scores, weights,
checkpoints and logs stay local; publish scalar results and hashes only.

The verifier recomputes causal moments from raw scores, checks the fitted gain
formula, compares score arrays to Torch, and binds complete before/fitted
checkpoints to exact parameter bits. It delegates full learning/resume checks
to the existing flat-metric verifier. The actual learner also captures its
revision-zero device checkpoint, which must match the fitted checkpoint exactly;
the verifier does not accept a different fit just because it is numerically close.
These are consistency and numeric checks,
not cryptographic execution attestation.

The frozen qualification harness and page pin fixture SHA-256
`3eb8e538371d0aa968abfd1f9d3758b7f83464ce6b36e06fb0fe9c91f2e9a00b`
(PyTorch 2.12.1). A different generator result is a new experiment, not an
automatic replacement for these frozen observations or criteria.

## Experimental boundaries

Native, browser and Torch independently fit their own computed scores. Their
f32 gains may differ by a few ULPs; cross-engine comparisons use the frozen
numeric gates, **not identical cross-engine initialization**. The checkpoint
inside each runtime retains that runtime's actual fitted weights.

Equal centered RMS does not imply equal pair structure, attention entropy,
softmax probabilities or learning dynamics. In particular, changing raw gains
changes `sigmoid(raw) / softplus(raw)`, the relative gain sensitivity under SGD.
The existing nonlinear bounded coordinate encoder is unchanged. This is not a
flat-encoder control, curvature-to-zero limit, or a capacity/compute match.

Corpus-study v3 keeps its strict identical-initial-parameter contract unchanged.
This separate synthetic experiment does not establish a language-quality gain
or add calibrated arms to v3. A future calibrated corpus protocol must identify
its initialization treatment explicitly before learned/head-wise geometry
mixtures or broader geometry families are compared.

The [native/browser qualification record](../benchmarks/results/2026-10-10-byte-bias-strength-calibration/README.md)
contains the fixed gates, measured errors, independent-review repair, regression
scope and hashes. It is synthetic correctness evidence, not a quality result.
