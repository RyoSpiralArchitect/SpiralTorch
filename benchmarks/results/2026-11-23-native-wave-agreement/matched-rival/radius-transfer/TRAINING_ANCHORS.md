# What do two objectives ask of the same initial geometry?

Use the three first prefixes (one per order) in the *existing* matched
training journals, and both initial factual hosts (affine and R=1 WaveGate).
No new decoder call, held-out input, generation, update, radius selection or
threshold. Later rows generally have different weights and are **not**
same-state comparators. The number of independent English prefixes is three,
not six or 24. Descriptive analysis, specified after the original experiments.

For each host and prefix:

1. Join the one-position full and explicit-single-rival training receipts at
   the same starting geometry. Check label, position, order, owner, numeric
   policy, individual gradient tapes, frozen layout and both realized 1/256
   parameter updates. Match the rival gradient bytes and factual verb margin
   to the independent initial margin capture. Require both native reports
   to give the same binary32 full-vocabulary CE.
2. The binary anchored loss is negative log probability of the gold verb
   conditional on the pair. Full CE minus this binary loss is negative log
   pair probability. Report the estimate and the interval induced by the
   stored full CE's adjacent binary32 rounding midpoints. This interval is
   not a bound on upstream numerical error. Do not assume that subtracting
   nearly equal rounded losses can resolve an arbitrarily small cost.
3. For each of the two stored displacements d, report dot(g_full, d),
   dot(g_pair, d) and dot(g_full - g_pair, d) with independently accumulated
   wide dot products. The last is an estimated coverage slope from two
   separately rounded native pullbacks. Check their sum and each objective's
   diagonal against its original Rust receipt. Independently reconstruct
   the binary gradient as minus the f32 rival probability times the separate
   unit-margin VJP; retain its relative discrepancy and both coverage slopes.
   Negative predicts a local
   loss decrease, not a measured post-update change. Do not mix gradients
   from different examples, hosts or states.

The archive reader accepts only hashes of the known capture, training
journals/tapes, state blobs and control metadata. It never opens the matched
training study's assessment plan, result payload or vocabulary-logit files;
those files' *manifest entries* are sufficient for lineage. It does not
open the pending radius experiment. Both CLI arguments below are archive
roots, never model directories:

```sh
python3 -ISB test_training_anchors.py -v
python3 -ISB training_anchors.py "$CAPTURE" "$MATCHED" rehearsal "$NEW_TOY_JSON"
python3 -ISB training_anchors.py "$CAPTURE" "$MATCHED" gpt2 "$NEW_TRAIN_JSON"
python3 -ISB verify_training_anchors.py "$MATCHED" "$NEW_TRAIN_JSON" "$TRAIN_JSON_SHA256"
```

## Results on archived GPT-2 training steps

The R=1 and affine hosts each had the same three eligible prefixes. All three
subjects were plural; only `actor.10` had a singular attractor. Other
training rows have no same-initial-state full/choice join in these journals.
No held-out or post-update probabilities were measured.

The conditional probability and pair mass below are *at the initial state*.
The last three columns are local changes in millionths of a unit of negative
log likelihood for the actual stored **choice-trained** step. Negative
means locally better; the last column is the sum of the preceding two
before display rounding.

| Host | Prefix | Gold given pair | Mass of pair in vocabulary | Choice slope | Coverage slope | Full slope |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Affine | driver.11 | 97.856% | 1.5483% | -4.937 | +20.671 | +15.734 |
| Affine | teacher.11 | 97.819% | 2.0186% | -4.825 | -6.776 | -11.601 |
| Affine | actor.10 | 88.016% | 1.5835% | -25.766 | -22.142 | -47.908 |
| WaveGate R=1 | driver.11 | 97.867% | 1.5459% | -0.718 | -7.394 | -8.113 |
| WaveGate R=1 | teacher.11 | 97.833% | 2.0093% | -0.778 | -8.772 | -9.550 |
| WaveGate R=1 | actor.10 | 88.059% | 1.5613% | -4.673 | -5.555 | -10.228 |

For `driver.11`, improving the binary loss on affine costs more in lost
pair mass than it gains inside the pair; WaveGate's local direction instead
improves both. The other two sentences do not show this conflict on either
host. This is about derivatives at each host's own snapshot, *not* a
demonstration that taking the finite update changes next-token probability
by these amounts. Nor does assigning mass to other continuations imply
ungrammatical output. No optimizer, mix, radius or advance gate was changed.

The largest width of a pair-probability rounding interval was 7.56e-9 or
less. The verifier separately reconstructed the probability from the full
objective's stored loss and the rival logit gap at 70 decimal digits.
Independently lifting all weights and gradients to integer units of
2^-149 and subtracting two exact linear forms reproduced all 36 slope
values at binary64 precision. Using the margin capture's separately
backpropagated unit-margin VJP gave the same signs and at most 1.34e-11
absolute difference in the coverage slopes. These checks do not remove
model or linearization error.

Source of the join was committed as `60ccd97` before reading the real
intersection. Six new unit controls and nine existing readout controls
passed, plus a rehearsal on two artificial anchors. No model calls.
Real report SHA-256:
`18e3c788fa7275ebf57d10e4b11c81d1be4342280c693599a5890c7d1d198905`.
