# HF Repetition Objective Control

The opt-in `st-core::runtime::zspace_repetition_objective` policy connects an
explicit normalization and update-slot schedule to the existing HF/PEFT
repetition-unlikelihood loss. Python and WASM call the same Rust function.
HF/PyTorch still owns the differentiable model, candidate probabilities and
optimizer. This is not a WGPU replacement for the HF model graph.

## Objective And Clock

The existing v3 planner and its candidate/mask rules are unchanged. If `A` is
the mean candidate loss over active positions, the training objective is:

```text
causal_lm_loss + effective_strength * A
effective_strength = base_strength * schedule_scale * normalization_scale
```

- `active_positions` uses normalization scale 1 for a nonempty active set.
- `eligible_targets` uses `active_position_count / eligible_target_count`.
- An empty active set has coefficient zero, without dividing by zero.
- Eligibility follows the selected planner's mask/prefix rule. It is not
  necessarily the number of all supervised causal-LM tokens.
- These are microbatch reductions. Trainer averages the combined loss over an
  accumulation group; this is not a token-weighted global average across
  unequal masked microbatches or ranks. Stock Trainer can normalize its base
  loss differently in that case. Keep valid-label counts matched when making
  ordinary-FT comparisons, or explicitly match the base-loss reduction.

`constant` has schedule scale 1. `linear_decay` has scale 1 through
`start_update`, interpolates to `final_scale` at `end_update`, and holds that
value afterward. Endpoints are absolute, not inferred from the current run's
remaining steps. All counts and slots must be portable safe integers; scales
are finite and bounded. Unknown configuration fields are rejected.

The HF clock is `Trainer.state.global_step` **before** the current microbatch:
all accumulated microbatches share the same slot. This counts completed Trainer
update slots, not necessarily successful optimizer updates: AMP overflow may
skip an optimizer step while Trainer advances. Changing this policy does not
change the candidate list or inference decoding. Its coefficient is not a hard
bound on gradient norm, parameter movement, KL divergence or useful learning.

## Use From Python Or CLI

Leave `objective_control` omitted to preserve the existing v4 recipe and v3
receipt. The new opt-in recipe is v5; its receipt is v4 and accumulates the
**actual weighted auxiliary term**, not nominal strength times a mean loss.
Receipts remain local-process compute-loss observations, not resumed lifetime
aggregates. Evaluation retains the stock causal-LM loss.

```python
import spiraltorch as st
from transformers import Trainer

recipe = st.hf_repetition_unlikelihood_recipe_contract(
    strength=0.1,
    ngram_order=3,
    context_window=128,
    max_candidates_per_position=8,
    candidate_source="model_topk_periodic",
    proposal_top_k=8,
    objective_control={
        "normalization": "active_positions",
        "schedule": {
            "kind": "linear_decay",
            "start_update": 0,
            "end_update": 256,
            "final_scale": 0.0,
        },
    },
)
ControlledTrainer = st.hf_repetition_unlikelihood_trainer_class(Trainer)
# Use the existing HfRepetitionUnlikelihoodCollator with matching candidate
# config, then pass zspace_repetition_unlikelihood_recipe=recipe to the trainer.
```

Equivalent extra flags on an otherwise matched `spiral-hf-finetune` command:

```text
--zspace-repetition-unlikelihood-strength 0.1
--zspace-repetition-unlikelihood-candidate-source model-topk-periodic
--zspace-repetition-unlikelihood-normalization active-positions
--zspace-repetition-unlikelihood-decay-start-update 0
--zspace-repetition-unlikelihood-decay-end-update 256
--zspace-repetition-unlikelihood-decay-final-scale 0.0
```

For a normalization-only experiment, choose `eligible-targets` and omit every
decay flag. Do not change normalization and schedule together in the first
matched comparison. The example values are an experimental candidate, not a
recommended optimum.

Rust clients call `zspace_repetition_objective_control` with the typed request.
Python exposes `st.zspace_repetition_objective_control(...)`. WASM exposes
`zspaceRepetitionObjectiveControlJson` and `zspaceRepetitionObjectiveControlObject`;
they return the same policy identity, counts and coefficient, without client
implementations of the schedule.

## Resume And Verification

The [bounded validation record](../benchmarks/results/2026-10-02-llm-objective-control.md)
includes actual Trainer/LoRA checks, cross-runtime replay and a pretrained GPT-2
execution smoke. None is a long-horizon efficacy result.

Controlled Trainer checkpoints include `spiraltorch-repetition-objective.json`.
Before loading model weights, resume checks the canonical objective recipe,
the saved Trainer clock, accumulation count, per-device batch size and world
size. Missing metadata or changed settings fail closed, including automatic
latest-checkpoint selection. `save_only_model` is unsupported for controlled
Trainer checkpoints because it discards optimizer state. The existing HF/run
identity mechanisms still own model, optimizer, RNG, corpus and input lineage;
this small guard is not a crash-atomic or cryptographic model-checkpoint bundle.

Tests exercise real CPU Trainer updates on locally initialized tiny causal
models, not downloaded pretrained weights or a language-quality benchmark:

```bash
python -I -m pytest -q \
  bindings/st-py/tests/test_repetition_objective.py \
  bindings/st-py/tests/test_hf_repetition_unlikelihood.py \
  bindings/st-py/tests/test_hf_repetition_objective_trainer.py
cargo test -p st-core runtime::zspace_repetition --lib
cargo test -p spiraltorch-wasm repetition_unlikelihood --lib
```

Build the Python wheel from this revision before testing. For actual wasm32
parity, build `spiraltorch-wasm` for `wasm32-unknown-unknown`, generate its Node
module with a matching `wasm-bindgen`, then run:

```bash
python -I tools/verify_repetition_objective_clients.py \
  --wasm-module /path/to/generated/spiraltorch_wasm.js
```

This compares 96 native Rust/Python outputs exactly against both WASM entry
points and checks invalid-request rejection. It is not browser GPU evidence.
The next efficacy test retains ordinary FT and the frozen constant treatment,
changes one policy factor, uses fresh seeds and the full fixed horizon, and
measures held-out causal-LM loss plus generated text. Preserve the earlier
[negative long-horizon result](benchmarks/hf_periodic_gpt2_pride_full_corpus_256step_20260823.json);
implementation correctness does not reverse that result.
