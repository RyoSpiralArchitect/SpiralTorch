# Does the prompt form change the first answer token?

For the subsequent native training API work, see the
[QKV file-training interface](../../docs/python/native-qkv-run.md) and the
[bounded GPT-2 update/save/resume qualification](GPT2-QKV-RESUME.md).
That qualification establishes exact CPU continuation for its recorded inputs,
not model quality or geometric superiority.

This matrix was fixed before scoring and has since been used in a private
first-token comparison. **The 32 record combinations are now observed, not a
fresh holdout for future studies.** No per-model scores or model weights are
published here. In the [previous GPT-2 experiment](../results/2026-11-03-native-factorial-answer-training/RESULTS.md),
45-46 of 72 final greedy answers started with LF. The two formats used different
records and changed the distractor ordering, so the historical counts cannot
isolate the effect of form.

The [four-cell extension](FACTORIAL.md) separates the fields line from the
request line and can reuse the authenticated diagonal observations without
executing either saved prompt again.

Use all 32 remaining combinations of the existing three-field vocabulary.
The roster and the 192 previous train/evaluation IDs reconstruct the exact
hash of the earlier 288 excluded IDs. No old scores, weights or model calls are
needed to verify the design. Within each new record, ask for each of the three
fields in both forms, holding values, display order, requested field, answer
and final `Answer:` suffix fixed. Form includes punctuation and question wording:
this is not a test of any one punctuation mark. The order of the two calls
alternates, and formats are compared within the same model. **32 records, not
192 independent trials.** The vocabulary is familiar; only the combinations are new.

```sh
python3 -I -S -B benchmarks/paired-prompt-contrast/matrix.py
python3 -I -S -B benchmarks/paired-prompt-contrast/test_matrix.py
```

The matrix SHA-256 is `afab2502944ffbbf8a1031142dc7d54a1d5f72659122ce356b340ad9642f8c73`.
With the original local GPT-2 tokenizer, all 96 pairs passed the joint
tokenization checks without a model: 20-26 IDs per prompt, LF ID 198, and 23
distinct first answer IDs. The paired token input SHA-256 is
`6544ac114a910e9faa5a7fb1e86f661ec7f99796cd0ddb6d7c6e161e18dcbe41`.
To reproduce this input-only check with `tokenizers` installed (no downloads):

```sh
python3 -I -B benchmarks/paired-prompt-contrast/admit_tokenizer.py /path/to/tokenizer.json
```

For an explicitly selected model and tokenizer, `inspect_pair(pair, encode, model)`
accepts an `encode(text) -> token IDs` callable without special tokens. It verifies
that adding the reference continuation or LF does not retokenize either prompt
prefix and that both forms name the same two candidate IDs. Only the two prefixes
reach the Rust-backed `NativeCausalLm.next_token_contrast(prompt_ids, target_id,
alternative_id)`. It returns the two raw last-position logits, their difference
(target minus alternative), and the greedy ID across **all** vocabulary items.
The first reference token includes any leading space; it need not be a whole word.
In this diagnostic EOS and other non-target, non-LF winners are classified as
`other`, not as LF.

The paired score is the record-form margin minus the question-form margin; its
sign alone does not establish that either candidate won. For one model,
`summarize(pairs, observations)` requires one complete field triplet per
record. It reports the three differences, their per-record means and the
equal-record mean, plus pair counts for all nine target/LF/other transitions
(question to record). This measures a first-token preference, not full-answer accuracy
or a benefit of training. No pretrained evaluation or new training is implied
by generating the matrix or passing its tests.

For saved geometry endpoints, [endpoint_replay.py](endpoint_replay.py) compares
full-vocabulary readouts from a caller-provided model. Give it the fixed cases,
an explicit call budget, and the expected raw logit hashes for every control.
It rotates the endpoint order, records a returned vector before checking its
hash, and stops at the first mismatch. It reports the target versus the **best
incorrect token**, full-vocabulary rank, greedy token and cross-entropy. This
is different from the target-versus-LF contrast above. The caller must bind the
base, adapters and inference runtime and persist the start/raw events. The
helper does not train or load a model.

For application code use the [native target readout](../../docs/python/native-causal-lm.md):
`NativeCausalLm.next_token_score` for a prompt, or `score_token_logits_f32le`
for retained bytes. The stdlib metric here is an independent verifier, not a
second model backend. Its NLL uses an explicit maximum and `log1p` so a very
confident correct answer can still have a small positive loss.

[set_mass_control.py](set_mass_control.py) is a model-free counterfactual for
retained rows. Given a fixed subset of classes, it adds the difference between
reference and candidate *set log odds* to every candidate logit inside the
subset. In real arithmetic this matches the reference mass without changing
relative odds within the set or its complement. The helper returns actual f32
bytes, the realized log-odds residual, and the largest drift of a within-set
margin after rounding. Recompute global ranks: the best competitor may be
outside the set. A set forced to include the known answer is an oracle; a set
chosen by reference top scores alone avoids that label but still needs a teacher.

[field_candidates.py](field_candidates.py) supplies a train-only alternative
for this closed extraction task. Its router reads just the final question or
request suffix, not the record values. Fit on training pairs of field name and
supervised first-token ID; then obtain the field support **without** an answer,
record key, or logits. Ranking a source row inside that set is a teacher-free
forced choice. It is not full-vocabulary decoding or a complete-answer score.
Mass matching on the same set still requires the reference row. Unknown
requests fail rather than falling back to a target-informed or padded set.

For whole answers, [field_completions.py](field_completions.py) fits each field's
menu on complete, supervised training lines. Rank by the **sum of full-vocabulary
negative log probabilities for the content and one final newline**, ascending.
This compares the probabilities of exact, terminated lines, not length-normalized
scores or free generation. A one-token and a longer answer sharing its start
compete through distinct stop/continue events. The menu order breaks exact ties
by token IDs. An answer missing from the training menu cannot win.

`score_menu` takes the prompt's retained first-token NLLs and one callback:
score the next ID with `model.next_token_score(prompt + list(prefix), next_id)`
and return its `nll`. Only a candidate prefix is fed back, never the requested
next token. Compare the plan's exact edge count before inference. Choose the
winner before joining correct answers; report first-only, content-only and
length-normalized rankings separately, not as alternative primary outcomes.

[completion_evidence.py](completion_evidence.py) checks a saved comparison
without running the model. Bind the call journal to its source prompt, model
and immutable hashes first. Recompute the first scores from the full f32 row,
rebuild every candidate from its recorded continuation and stop scores, then
join labels. For error analysis `contrast` splits the gold-minus-rival NLL
gap into head, remaining content and stop **against the same strongest wrong
line**. This localizes a decision; it does not identify a cause or measure a
training intervention. Suffix NLLs are retained native observations, not
independently recomputed probabilities unless their full logits were saved.

For a supervised follow-up, [field_head_pairs.py](field_head_pairs.py) derives
head rivals from a single field's **training menu**. If one complete token
sequence is a proper suffix of another with a different head (e.g. `square`
and `large square`), each head is a training rival for the other answer.
Unpaired examples have no added rival. This is a way to prepare matched
training controls, not a decoder rule, a new optimizer, or evidence of a gain.

For a refused fixed-radius geometry step, [geometry_step_precision.py](geometry_step_precision.py)
checks the stored **origin and already rounded SGD proposal**, not a new gradient.
`certify_f32_delta(origin, proposal, radius)` takes ordered dictionaries of named
trainable little-endian f32 buffers in artifact order. It finds the minimum possible
relative vector error over finite f32 endpoints, using the native binary64 component
residual and an exact sum of its squares. The report separates vector error, length
error, and contributions by tensor. Near a last-bit boundary its exact sum is not a
claim about the last bit of the native compensated reduction.

`vector_impossible` rules out meeting the vector bound for **that** direction,
origin and radius. A witness within bounds proves only coordinate representability;
it is not an artifact, a recipe-validation result, or evidence about language loss.
If only the witness's length misses, the checker does not rule out another point.
The checker runs without model weights or external packages and never applies a step.

[Paired projection language study](PROJECTION.md) runs the native decoder on
synthetic short/long answers, then checks the full-line rankings from saved logits.

[Context preference training](context_token_training.py) compares four ways to
train the same synthetic decoder's geometry on 18 training prompts: the existing
guard, positive-only word CE, entity-matched context pairs, or pairs with the
same negative contexts rotated across entities. All use the same nine-objective
guard. Run `freeze` with the previous training readback, the tested extension and
wheel hashes, and the actual wheel file. Install that wheel into an otherwise
clean environment with `pip install --no-index --no-deps --no-compile` and
run Python with `-I -B`. The launcher checks installed files against the archive
before starting the one allowed `run`; use `audit` afterward without the model
(invoke without arguments for the argument order). The [readback](context_token_training_audit.py)
distinguishes a relative gain from absolute suppression, and word choice from
the complete stopped answer. None of these training measurements tests transfer.

[Primary-objective steering](primary_context_training.py) keeps the same training
roster, wheel checks, four controls and endpoint gates. It first constructs
the four-group proposal and nine-group safe anchor on each arm's own current
model. The three treatments use the native `steer_geometry_step` to turn that
anchor toward their primary, rather than adding another constraint to the hull.
The [model-free readback](primary_context_audit.py) reconstructs each rounded
update and checks finite protected losses separately from primary alignment.

[Entity-state swaps](ENTITY-SWAPS.md) requires the queried entity to determine
the answer. It separates fitting, key-blind scoring on unseen combinations of
known entities, and model-free analysis of all assignment/query/order swaps.

[Projection reach](projection_reach.py) is a training-input-only follow-up.
It fits a signed binding-gradient direction on four old training edges and
physically probes both signs on those and the four other training edges. Dense
and diagonal Q/K/V/output coordinates, the actual geometry parameters, and
the shared-key-bias null control are distinct. The [offline readback](projection_reach_audit.py)
reconstructs the fitted directions and checks finite changes, not just gradient
norms. This does not train a new model or reopen the consumed evaluation deck.

[Partner routing](PARTNER-ROUTING.md) trains equal-size edits to output or Q/K on
one versus two partners per identity. Includes the query-scrambled control,
ordinary checkpoint readback, and the negative transfer result.

[Single-site quartet training](SINGLE-SITE-BINDING.md) fits K or Q separately
in Rust on previously unused partners. It separates an improvement in binding
interaction from actual answer choices, with an independent wheel replay.

[Contextual keys](CONTEXTUAL-KEYS.md) compares learning at two depths and serves
the snapshots from shared factual prefixes. A matched key substitution changes
answer scores but does not turn the deeper key into a better training site.

[Full-vocabulary answer learning](ANSWER-QUARTETS.md) trains a native V projection
with gold next tokens, optionally mixed with quartet supervision. It distinguishes
learning the answer vocabulary from choosing the right answer to each question.

[Finite answer protection](ANSWER-GUARD.md) checks corrected weights on two
training batches before committing, with independent replay of rejected trials
as well as accepted ones. Monotone fit loss is not an answer-quality guarantee.

[Causal key-lag capacity](CAUSAL-KEY-LAG.md) trains fused Q/K/V weights with
ordinary versus explicitly lagged keys, keeping geometry and LoRA frozen.
It records the failed full-batch admission, the microbatched Rust repair,
the primary 32/32 training result, and the secondary specificity failure.
Training-set fit is kept separate from held-out or pretrained-model evidence.
