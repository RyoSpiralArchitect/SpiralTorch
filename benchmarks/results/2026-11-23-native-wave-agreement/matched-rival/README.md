# Relative-clause agreement at matched gradient step length

This offline pilot asks whether explicitly contrasting the correct verb with its
number rival teaches GPT-2 to separate them better than full-vocabulary cross
entropy (CE), and whether that effect differs between WaveGate and channel affine.
The earlier simple-agreement pilot is separate; none of its evaluation sentences
are scored here.

## Frozen comparison

- Prefix: `The <subject> that the <attractor> met`. The next token is either
  ` is` or ` are`. All six prefix words and both verbs are single GPT-2 tokens.
  Each noun pair appears in all four subject/attractor number combinations.
- Six training pairs (24 examples), four disjoint evaluation pairs (16 examples);
  nouns are also disjoint from the previous pilot. Past-tense `met` is fixed.
  Orders 17, 31 and 73 are hash permutations of the *same* training cases.
- A frozen pretrained base, no LoRA, one adapter on the last attention output.
  WaveGate (fixed radius) and channel affine share initial gate/bias values,
  strength 0.25, and the same number of trainable scalars. Initial outputs can
  differ, so compare changes from each geometry's own initial state.
- Cross each geometry with full-vocabulary CE or gold-vs-explicit-rival CE.
  Only the final verb is supervised. The latter uses the native declared-rival
  gradient with one masked target, one rival and mix 1. Every update follows the
  negative gradient with requested L2 length 1/256, within relative tolerance
  0.0001 for both realized length and vector error. A zero gradient stops the run.
  Matched *step* length does not enforce matched endpoint distance.
- Four arms x three orders x 24 examples = 288 updates. Verify all stored steps
  before scoring either initial state or the 12 endpoints. Each of those 14 states
  scores all 16 prefixes exactly once (224 next-token predictions); no generation.

## What counts as an effect

The primary measurement is the mean of gold minus rival logit over all held-out
items, not the probability assigned to either verb against the rest of the
vocabulary. For each geometry, the contrasting objective passes only if its mean
margin increases from its own start **and** its margin change exceeds full CE's
on every order. A WaveGate-specific result additionally requires that this
rival-minus-full advantage exceed affine's on every order. Otherwise that
particular claim is not supported, with no rerun or threshold adjustment.
Report strict binary wins/ties, congruent vs conflicting attractors, gold NLL,
log probability of the verb pair and conditional gold log probability as
secondary measurements. Three orders do not make three independent test sets.

## Running and verifying

The driver uses a local, pinned GPT-2 and SpiralTorch extension. There is no
download and no dependency on PyTorch, Transformers or NumPy. The Rust backend
computes logits, differentiates and updates; Python independently checks objective
values on a tiny fixture, actual f32 gradients and weight displacements, and
statistics from archived full-vocabulary logits.

1. Run `python3 -ISB test_matched.py -v` and, with the CPU extension installed,
   `python -IB test_matched.py --native-oracle`.
2. For a disjoint tiny rehearsal use `python3 -ISB run.py prepare --kind rehearsal
   --directory DIR`, then `python -IB run.py run --kind rehearsal --directory DIR
   --pin HASH`. Use the SHA-256 printed by prepare.
3. For an explicitly authorized real run, use a new private directory,
   `--kind gpt2`, and add `--model-dir LOCAL_GPT2` to prepare and run.
   The real worker is local CPU only, single-threaded, bounded to 600 seconds;
   the first failure is terminal. It never overwrites an attempt or a weight file.
4. `python3 -ISB run.py verify --kind gpt2 --directory DIR --pin HASH`
   validates the receipt and replays from archived bytes; no model is loaded.
   For the tiny fixture substitute `rehearsal`.

Keep the private plan, state blobs, gradient tapes, journals, f32 logits, result
and completion or first-failure record together. Replaying saved bytes does not
spend another prediction. Neither success nor failure on this one template and
four held-out noun pairs establishes general language quality.
