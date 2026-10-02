# Long-Horizon Radius Study: Running

This directory currently records a **started, not completed** experiment.
No 512-step learning outcome or final evaluation result is claimed here.
The executable source/protocol was committed before training at
`82d5ed3d4815ddee32e434941f553709e2e7319a`; it is run from a frozen local copy.

See [the protocol and restart contract](../../../docs/wave_gate_long_horizon.md).
The live run uses the already validated native radius binary, without rebuilding
Rust or downloading models/corpora. Mathematical ownership remains Rust; Python
adds experimental orchestration rather than a second geometric implementation.

## Locked Design

- Cached, frozen float32 CPU GPT-2; gate/bias adapter at `transformer.h.0.mlp`.
- Three seeds, three active arms: tangent, fixed radius 4, learned-from-radius 4.
- 512 updates per run, batch two, 128-token blocks, Adam lr 0.001, strength 0.1.
- 130048 causal targets per arm; **4608 planned primary updates** in total.
- Development: the same 16 previously inspected Pride tail blocks every 128 steps.
- Final endpoints: the other 120 Pride tail blocks and 32 fixed Alice blocks,
  evaluated only after all nine runs and their continuation checks complete.
- Shared zero-update baseline; no best-checkpoint selection, early stopping,
  outcome-driven schedule changes or speed comparison.

The learned-radius arm has one additional parameter. Alice appears in older
unrelated repository studies; neither book is claimed absent from GPT-2
pretraining. Evaluation novelty is limited to this WaveGate comparison.

## Evidence Available Now

`plan.json` fixes corpus/token hashes, model/runtime/source identities, actual
data partitions and every batch schedule. `validation.json` records **47 passing
Python tests** covering geometric clients and restart/evaluation boundaries.
Tests include a real tiny HF interruption/resume with exact parameter, Adam and
history equality; changed-protocol/corrupt-checkpoint rejection; single-writer
exclusion; rejecting mutated-base checkpoints; and preventing early endpoint
evaluation. These tests do not substitute for the live experiment's completion.

Each live checkpoint is flushed before its hash enters an atomic journal.
Original logs, model/cache data and optimizer checkpoints stay local. The eventual
result must match the plan and actual terminal process state before this record
can be promoted to completed evidence. A surviving journal is not proof that a
process is still running. No additional process should be started merely because
an observation call times out.
