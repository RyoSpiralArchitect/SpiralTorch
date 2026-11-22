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
```
