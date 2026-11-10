# Entity-state swaps

This CPU study asks whether a tiny native decoder learns which state belongs to
which entity. It does not measure natural-language generation or pretrained
GPT-2. No PyTorch, NumPy, new native build, or network fetch is needed.

## Which information must the model use?

A token-ID prompt encodes two facts, a query entity, and a common answer prefix.
One entity is on, the other idle. The two answer IDs are on and standby. A cell
crosses the assignment, which entity is queried, and the fact order. Swapping
the two state tokens preserves the entire prompt multiset and query but reverses
the gold answer. Reversing the order of the two facts preserves the answer.

For eight known entities the train graph connects ring neighbors; the held-out
graph connects distance-three neighbors of the same fixed entity permutation.
Each graph has eight different unordered edges, degree two per entity and all
eight cells per edge. Thus there are 64 training and 64 held-out prompts. This
tests novel combinations, not new vocabulary or new identities.

Each 16-example batch includes both partners for two queried entities. The
matched contrast changes only the two state positions. The rotated negative
has the *same query, query slot, cue order and target* as the matched negative;
it substitutes the queried entity's other training partner at the distractor
position. For either target the negative multiset is identical. Swapping
negatives can still change difficulty and the nonlinear pair loss.

## Training and scoring

Use [entity_swap_study.py](entity_swap_study.py) with the pinned, already-tested
primary-steering wheel. Run the stages in order (invoke without arguments for
the argument list):

1. `freeze`: verify the previous native stack and freeze both splits, separate
   answer keys, 64-step schedule, budgets and decision gates.
2. `train`: update 64 geometry coordinates on a frozen two-layer decoder with
   frozen LoRA, from two retained starting geometries. The retention-only arm
   replays an older language task. The other six arms cross ordinary
   distance-matched cohort SGD or protected primary steering with CE, matched
   preference or rotated preference. Direct arms are not required to preserve
   the nine old objectives. Every attempted step computes the same 88 native
   gradients; all eligible updates use a radius of 0.002.
3. `audit-train`: check gradient ownership, signal order, the actual rounded
   weights and the old task's finite losses without reading the new test set.
   Numerical refusals are terminal per path, not grounds to select an earlier
   endpoint. Only complete endpoints and the two origins are eligible.
4. `score`: a separate worker sees ordered prompt IDs but no answer keys. It
   saves full-vocabulary logits for every distinct eligible model on both
   decks. Training and scoring claim their frozen protocol once.
5. `report`: verify the scoring exit, journals, raw rows and calibration
   against native training CEs before unlocking the labels. Once used, this
   held-out cohort must not be described as fresh for a tuned follow-up.

For one edge and fact order let L be the standby minus on logit and let the
indices be (idle carrier, queried entity). The primary continuous score is
`(L_AA - L_AB - L_BA + L_BB)/4`; additive query, assignment or vocabulary
biases cancel. Average over both orders and then the eight edge blocks.
Improving this interaction is not the same as predicting the right word.
Separately count strict two-choice accuracy, joint assignment/query swaps,
relative rankings and edges with **all eight** decisions correct. Counts from
views of the same edge are not independent replicates. The predeclared
practical thresholds and missing-endpoint rules are part of the frozen plan.

## Does a rank-one first-block output change survive ordinary inference?

On the small synthetic decoder, use the two origins and their saved
direct-matched endpoints. This follow-up uses **only the same training edges**.
It tests local sensitivity, not generalization, training speed or task mastery.
Start with a completed `projection_reach.py` protocol and measurement, plus the
installed CPU wheel identified in that protocol. No new wheel build is needed.

1. Run `rank_one_roster.py DIRECTIONS SHA256 NEW_OUTPUT` on the saved
   `measured/directions.json`. The resulting `rank-one-candidates.json` is
   fitted without check edges or further model calls. It keeps all 32 candidates.
2. Run `rank_one_projection.py freeze PARENT PARENT_PLAN PARENT_MEASUREMENT ROSTER OUT`,
   supplying SHA-256 pins in the three pin slots. This locks the dense axis,
   its zero-bias leading outer product and a matched, perpendicular rank-one
   control on the first output projection. All three have unit joint weight/bias norm.
3. With the pinned wheel's Python, run `rank_one_projection.py run PROTOCOL PIN MEASURED`.
   Both signs at each of two radii go through a virtual projection and, separately,
   an edited copy of the F32 safetensors reimported through the normal inference
   path. On that copy, the final LoRA and geometry get the new base identity;
   every other field is verified unchanged. The source weights are never overwritten.
4. Run `rank_one_projection.py audit PARENT PROTOCOL PIN MEASURED PIN OUT`.
   The audit recomputes the fit-only axes, exact physical checkpoint bytes,
   signed finite effects, and the predictions from archived native adjoints.
   It checks the original dense observations byte for byte and reports both
   positive and negative gate outcomes, including prediction flips.

These are rank-one **weight changes**, not benchmarks of a factorized runtime
or an optimizer that trains two factors. The reconstructed F32 weight change
can differ slightly from the requested outer product because of rounding.

## One-coefficient native learning

`scalar_binding_study.py` follows a qualified rank-one physical trial.
Its scalar adapts the first output projection using the native
`step_pairs` method. It does not select a coefficient by scoring endpoints.
For both original geometries, compare the fit-derived leading axis, a
matched orthogonal axis, and a frozen model. Every trained arm gets the
same six passes over eight old training edges and exactly four matched
pairs per update. The optimization budgets match; the learned coefficients
need not. Each arm restores its compact checkpoint before step 25.

Run `freeze`, `train`, `score`, and `audit` in that order (run the script
without arguments for the signatures). Freeze uses the prior plan,
measurement and audit pins, the fit-only factor roster, and a tested native
wheel. The training worker cannot open the blind questions or answer keys;
the scoring worker cannot open the training labels or fresh key. Only the
final label join uses the fresh key. The score stage compares all 64 logits at
every endpoint to a separately merged and reimported ordinary decoder.
The offset-two evaluation edges reuse the eight entity identities but
not their old training or prior test pairings. Once scored, these prompt
combinations are not a fresh test for a tuned follow-up.

## Does the scalar optimizer actually lower its training objective?

`scalar_full_batch.py` compares full-batch clipped SGD and same-batch
backtracking with hold and the previous minibatch endpoints. Only the 32
offset-one training pairs and model inputs are staged. Both rules take
their gradient on the same 32 pairs. The guarded rule additionally scores
the current ordinary decoder and up to eight feasible candidates; it stops
at the first refusal, without advancing. This is an optimizer experiment,
not another evaluation on the consumed offset-two set.

Run `freeze PRIOR PLAN TRAINING CYCLE WHEEL WHEEL_PIN NATIVE_PIN PROTOCOL`,
then `run PROTOCOL PIN MEASURED` and `audit PROTOCOL PIN MEASURED PIN OUT`
with the script and the pinned CPU Python environment. The cycle pin refers
to the training-only cycle readback, not the previous fresh-set report. Six
full steps match the old 48 minibatches in pair-gradient count, but not in
forward work or runtime. The later fixed limit is 24. Both milestone and
terminal losses are reported, never the best retrospectively chosen step.

## Common ascent for both query slots

`slot_common_output.py` fits output-projection directions on four even
training edges and physically tests them on four different odd training
edges. Neither fold is fresh evaluation data. Build the Rust example with
`cargo build --locked --offline -p st-nn --example output_margin_fit`.
The example sees only the fit packet. It uses `NativeCausalLm::output_margin_bank`
to obtain the gradient of the positive-minus-negative logit contrast once
per pair and space. Reassigning groups or changing the fitting rule does not
rerun the decoder. The compact space fixes the input factor and learns the
output factor directly, without computing a square parameter gradient.

Call `bank.direction(groups, rule)` for an equal-group mean or worst-group
cosine direction. The latter reuses the geometry hull solver; it can refuse
opposed/zero groups rather than silently substituting a mean. Raw group
slopes and normalized certificates are different quantities. Resolved
compact fits can open the existing scalar learner with `dir.scalar_adapter(&model)`;
it requires the original base, LoRA, geometry and readout arithmetic.

To reproduce the experiment, run the script without arguments for its
`freeze`, `run` and `audit` signatures. Freeze uses the previous scalar
full-batch training plan, archived projection plan/measurement, frozen
last-block LoRA JSON, the compiled Rust example and the already validated
CPU wheel. The wheel is an independent consumer of the resulting weight
changes; it need not contain the new Rust fitting API. The physical worker
seals all fit directions before reading the check deck. It runs both signs
at two fixed Frobenius radii through the projection probe and ordinary
reimported weights. The readback checks fit gradients, normalization,
per-slot and per-edge finite changes separately from answer correctness.

## Swap the query, not the facts

`query_swap_binding.py` compares two counterfactuals on the **same**
32-prompt fit roster. Assignment swap (A) changes just the two state cues
and keeps the query fixed. Query swap (Q) changes only the queried-token
ID, keeping both facts byte-identical. Both match the same positive rows
to the same multiset of negative rows. The ungrouped mean derivative is
therefore the same, not an independent success on two tasks.

The Rust-backed Python contrast bank fits the common direction by the
positive prompt's queried entity (eight groups), using A, Q or both; the
equal mean is a control. The joint fit averages both swaps within each
entity; it does not certify 16 independent constraints. All resolved
compact directions train one scalar on identical joint pairs and identical
budgets before check prompts are opened. Dense directions are finite
controls only. An independent, earlier wheel reimports the changed weights
and restores the scalar checkpoints without installing a foreign ML library.

Run the script's `freeze`, `train`, `score`, `audit` commands (invoke without
arguments for signatures). Use the indexed-bank wheel for training, the
previous measurement wheel for scoring; audit does not import a model.
Check each of four different **old training** edges once: both identities
and both swap types must benefit. Full-vocabulary top-1, strict two-choice
accuracy, and logistic loss are separate results. No new evaluation cohort
or pretrained model is used.
