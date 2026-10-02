# Long-Horizon WaveGate Learning

The earlier 32-update radius pilot established learning and exact continuation,
but tangent remained best. The next experiment increases the update budget
without changing the Rust map, optimizer, frozen base or identity initialization.
It is not a speed comparison.

## Frozen Protocol

`bindings/st-py/examples/hf_wave_gate_pride_long_horizon.json` fixes three seeds
(41/43/47), 512 updates each, batch size two, 128-token chunks, Adam lr 0.001
and residual strength 0.1. The three active arms are tangent, fixed R=4 and
learned-from-R=4. A shared zero-update model is the baseline. Radius 4 was
selected from the earlier pilot, not independently discovered by this study.
The learned arm has 1537 parameters versus 1536 for the other two.

Each active arm consumes 130048 causal targets, with exactly paired sample
orders across arms. The endpoint is always step 512, not the best intermediate
checkpoint. Development loss uses the same previously inspected 16 blocks every
128 updates, only for descriptive monitoring. No loss-based early stopping or
parameter search is implemented.

Only after **all nine** runs finish, pass frozen-base checks and reproduce their
next update from disk do the following evaluations become available:

- All 120 blocks left after excluding the previous 16-block development probe
  from Pride and Prejudice's final text partition.
- 32 evenly spaced blocks from the existing local Alice's Adventures in
  Wonderland text, with its trailing Gutenberg notice removed.

Token-block hashes must be disjoint across training, development and both final
sets. Corpora, model snapshot, split, schedule, source scripts, bridge, native
binary, versions and model parameters/config are bound into the study identity.
Alice was used by older unrelated repository studies; neither book is claimed
unseen by GPT-2 pretraining. This is a fresh evaluation boundary for this WaveGate
comparison, not proof of generalization to pristine data.

## Restart Semantics

The Unix CPU driver is `bindings/st-py/examples/hf_wave_gate_long_horizon.py`.
Python orchestrates the study; the installed Rust kernel owns the geometric map
and VJP. This example reuses the preceding pilot's adapters, data packing and
paired schedules rather than implementing another geometric formula.

Every 64 updates, a unique small checkpoint stores gate/bias/radius, Adam state,
the exact completed cursor, batch history and diagnostics. It is flushed before
an atomic journal publishes its filename and hash. A writer lock prevents two
processes from advancing one study. Unpublished checkpoint files after an
interruption are not adopted. Frozen-base weights and gradients are checked
before each checkpoint becomes resumable, not merely at the end of an arm.

`--resume` requires the existing study identity and verifies each referenced
checkpoint hash before continuing. Completed arms are reused; an interrupted
arm starts at the next unconsumed batch. Endpoint continuation checks consume
two extra updates per run, without replacing its saved step-512 weights.
Final evaluation therefore always reloads the selected fixed endpoint, not the
validation-only step-513 state. If a completed result is sealed in the journal,
resume verifies it and does not repeat evaluation. An interruption during final
evaluation can repeat that deterministic phase, never alter trained endpoints.

Tests exercise an intentional mid-arm interruption against an uninterrupted
tiny HF model, exact optimizer/parameter/history equality, changed-protocol
rejection, checkpoint corruption, single-writer exclusion, frozen-base mutation,
causal target alignment and blocked evaluation before all arms complete.
The existing Python CI job now runs this suite alongside the Topos, elliptic and
WaveGate gradient tests using the newly built wheel, CPU Torch and Transformers
4.57.6. Model access is offline; tiny models are constructed from configurations.
Required native exports are asserted before pytest so a missing NN build cannot
silently turn the entire geometry suite into skips.

## Run

Use a native build including `nn`, float32 Torch and a cached local GPT-2 model.
Do not rebuild or replace that Python/native runtime during the experiment.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
python bindings/st-py/examples/hf_wave_gate_long_horizon.py --config bindings/st-py/examples/hf_wave_gate_pride_long_horizon.json --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

For an interrupted process, first verify that its actual process/session has
ended, then invoke the same command with `--resume`. A journal or lock file's
presence alone does not show whether a process is alive. `journal.json` reports
training progress; only the hash-bound completed `results.json` is final evidence.
There is no CUDA or resident WGPU claim: Furnace contention is not bypassed by
interrupting another workload, and this experiment reuses the local CPU runtime.

## Reproduce the Summary

Once the process has ended successfully and the journal seals a completed result,
the standard-library-only summarizer derives paired seed comparisons without
loading model weights, importing Torch or running evaluation again:

```bash
python tools/summarize_wave_gate_long_horizon.py --plan "$STUDY/plan.json" --results "$STUDY/results.json" --journal "$STUDY/journal.json" --output "$STUDY/summary.json"
```

It checks result identity/hash, every planned run and its endpoint receipt,
training cursor/batch history, evaluation block counts and reported means. It
refuses partial evidence and never overwrites an earlier summary or input.
The output binds all three input hashes and keeps within-book and transfer
cross-entropy separate, including every seed's difference versus tangent.
Sample standard deviation describes these paired differences; it is not a
confidence interval. Seeds vary sample order, not initialization, and reuse the
same evaluation blocks. No block-independent significance claim follows.
This is a receipt check, not a revalidation of original checkpoint contents or
process termination; those remain part of the local completion verification.
