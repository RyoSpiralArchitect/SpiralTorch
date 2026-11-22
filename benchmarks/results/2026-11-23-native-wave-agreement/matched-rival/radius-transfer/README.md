# Does the radius effect transfer to learning at matched step length?

The captured-adjoint study selected nominal R=4, not a new production
default. Here both decoders recompute all activations and adjoints at their
own radius. Compare only the two untrained saved WaveGate states (R=1/4,
same gate, bias, strength, kernel, base; no LoRA). Use the **training** CPU
extension from the earlier matched-rival fit, not the diagnostic extension.
The original packed readout and f64 vocabulary-head backward are explicit.

## Preregistered comparison

- Same 24 training cases and permutations 17, 31, 73. Use binary CE on the
  correct next verb vs its explicit rival; update gate and bias only, along
  the native negative gradient, with requested global L2 step 1/256.
  Alternate arm order across permutations. No optimizer state or radius
  learning. Archive actual directions and rounded states. On **every** R=1
  step, require both the gradient hash and the resulting artifact ID to
  equal the previous same-wheel training journal. Check actual displacement,
  sign and relative vector error (1e-4) on both arms.
- New assessment pairs: parent/child, brother/sister, student/friend,
  chef/guest. Each appears in all four subject/attractor number combinations.
  Prefix: `The <subject> that the <attractor> met`; next token ` is` or
  ` are`. Pin word-to-token mapping with the saved vocab and merges.
  No noun or scored prefix reused from the two prior agreement evaluations.
  This makes no claim about GPT-2's pretraining corpus.
- Do not score until **all** updates are sealed and checked. Score all 16
  new prefixes once on each initial state and the six final states:
  144 updates, 128 full-vocabulary last-token rows. No generations, additional
  checkpoints, previous held-out logits, GPU, network, retries or publication.
  Real CPU process limit 600 seconds; stop on the first failure. The testing
  cohort is deterministic artificial data and has no language interpretation.
- For each radius and order compute mean paired change in gold-minus-rival
  logit from *that radius's own initial state*. Advance this candidate only
  if R=4's change is positive **and** exceeds R=1's change by at least
  **0.01 logit on every order**. This threshold is an engineering decision,
  not a significance test. A failed criterion is a result, not a request
  to search more radii or rescore these prompts. Report absolute margins,
  strict wins/ties, NLL changes, and conflict/congruent splits as secondary
  evidence. Three orders share one test set, not three replications.

## Commands

`prepare` reads the captured training plan, selected radius artifact and old
training journals; it does not open model weights, old evaluation outputs or
score anything. Save its printed SHA-256 separately. Use fresh output paths
and the saved training interpreter, not a newly downloaded wheel.

```sh
python3 -ISB experiment.py prepare rehearsal "$OUT/toy" "$CAPTURE" "$SWEEP" "$MATCHED"
"$TRAIN_PY" -IB experiment.py run rehearsal "$OUT/toy" "$TOY_PIN"
python3 -ISB experiment.py verify rehearsal "$OUT/toy" "$TOY_PIN"
python3 -ISB experiment.py prepare gpt2 "$OUT/real" "$CAPTURE" "$SWEEP" "$MATCHED" --model-dir "$GPT2"
# Only run the following real worker for an explicitly authorized, unused attempt.
"$TRAIN_PY" -IB experiment.py run gpt2 "$OUT/real" "$REAL_PIN" --model-dir "$GPT2"
python3 -ISB experiment.py verify gpt2 "$OUT/real" "$REAL_PIN"
```

Verification reads only the frozen plan, state chain, gradient tape, logits
and receipts; it does not import the extension or call a model.
