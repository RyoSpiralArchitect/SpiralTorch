# Resident Vision Training Boundary

`st_vision::resident_trainer::ResidentVisionTrainer` owns the real ConvNeXt
classifier, image loader, augmentation/shuffle RNGs, epoch, accepted/rejected
update counts and learning-rate schedule. Enable `st-vision` features `nn,wgpu`.
The classifier uses the existing resident mean cross-entropy and VJP/update
path. Python and JavaScript do not implement another training policy here.

## Submission And Settlement

```rust,ignore
use st_nn::optim::WarmupCosineScheduler;
use st_vision::resident_trainer::{ResidentLearningRate, ResidentVisionTrainer};

let rate = ResidentLearningRate::WarmupCosine {
    state: WarmupCosineScheduler::new(0.01, 0.0001, 100, 10_000)?.state(),
};
let mut trainer = ResidentVisionTrainer::new(
    &host_classifier, device, loader, dataset_sha256, rate,
)?;

let submitted = trainer.submit_next()?; // no host readback
let outcome = trainer.settle()?;        // explicitly observes GPU acceptance
if outcome.accepted {
    // submitted.loss is an optional observation handle, not a new trainer.
}
let checkpoint = trainer.checkpoint_snapshot()?.read()?;
let json = checkpoint.to_json()?;
```

The loader must have explicit targets, start at its first batch and contain a
nonempty integral number of fixed-size batches. Tail batches are rejected at
construction, not silently dropped or padded. Configure the resident transform
dispatcher before construction. Dataset contents, labels and ordering must be
immutable for the caller-supplied dataset SHA256; the trainer does not scan or
authenticate this ID.

The standalone loader still consumes input on successful image submission. This
trainer stages that same preparation and only commits its private loader when
the model update has been submitted; this larger transaction is specific to the
trainer, not a silent change to `DataLoader::next_resident_batch`.

- Host preparation/shape failures leave the loader cursor, RNG, epoch, weights
  and schedule unchanged. The existing resident input preparation is staged,
  not reimplemented, and committed only when the model update is submitted.
- A submitted update consumes its input. Numerical rejection retains every
  weight and advances the rejected count, but not the accepted count/schedule.
- Exactly one update may be pending. Another submission, restore or checkpoint
  is rejected until its acceptance flags have been read.
- An unknown readback failure retains the pending update. It is not guessed to
  mean either acceptance or rejection. Settlement may be retried.
- Epoch is zero-based and advances when submitting the first batch of the next
  epoch. Accepted/rejected counts only change at settlement.

`ResidentLearningRate::Constant { rate }` retains the plain-SGD control.
Warmup/cosine uses the existing `st-nn` scheduler, whose portable state is owned
by `st-core::runtime::trainer_optimizer`. The first proposed rate is its first
`step()` value, obtained through its side-effect-free `preview_step()`. Only
acceptance commits that proposed scheduler state; preparation/rejection does
not emit a misleading scheduler-step event. Restore is constant-time and does
not replay scheduler telemetry. This is not Adam,
momentum, hypergrad or migration of all `ModuleTrainer` policy.

## One Checkpoint Boundary

The versioned `VisionTrainingCheckpoint` binds model, input and trainer payloads
using the existing Rust trainer-checkpoint hash implementation. It validates
the nested contracts, batch size, consumed-input versus attempted-update clock,
accepted plus rejected counts, and scheduler clock. Hashes catch corrupted or
mixed components, not a malicious writer that recomputes them.

Create a new trainer with `ResidentVisionTrainer::from_checkpoint(...)`, supplying
the same dataset identity, loader batch and transform configuration. Alternatively,
`trainer.restore_checkpoint(&checkpoint)` prepares and validates a complete
replacement before changing the live owner; incompatible architecture is rejected.
GPU resources are re-created, not serialized. A snapshot taken at a settled
boundary keeps that boundary even if the live trainer continues or drops.

The same Rust implementation compiles for wasm32 with `settle_async()` and
checkpoint `read_async()`. Dropping a settlement future keeps the pending update
so it can be observed again. A canceled checkpoint mapping only drops that frozen
snapshot; the trainer remains settled. This does not yet expose a JavaScript
DataLoader or a new Python trainer binding, and compilation alone is not evidence
of real-browser training restart.

## Validation

The native GPU tests run uninterrupted training in one child process, save a
second process after attempt 37, terminate it, then restore in a third process
and finish attempt 100. Both constant SGD and shared warmup/cosine use shuffled,
flipped, normalized synthetic images and deliberate invalid class IDs. The test
compares every resumed batch identity, transformed image bit, rate and acceptance
decision, and the complete final checkpoint (all weights, input and trainer).
It also checks pending-boundary rejection, frozen snapshot lifetime, atomic failed
restore, preparation failure and an injected unknown readback error followed by
successful settlement.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-vision \
  --features wgpu --lib resident_trainer::tests -- --test-threads=1 --nocapture
cargo test --locked -p st-nn --lib optim::tests::warmup_scheduler
cargo check --locked -p st-vision --features wgpu --target wasm32-unknown-unknown --lib
```

This is a restart/correctness gate, not a throughput or real-image quality claim.
The next work is thin client exposure, actual browser continuation, and connecting
Rust Z-space optimizer control while preserving the fixed-SGD matched baseline.
