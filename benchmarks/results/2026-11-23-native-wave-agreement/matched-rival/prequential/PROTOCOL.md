# Do initial margin gradients explain later training-time predictions?

Retrospective, training-only diagnostic. The source experiments are already
complete. Freeze the analysis before opening their later-step measurements.
No model loading, inference, backward passes, updates or held-out reads.

* Use the archived factual affine and WaveGate (R=1) initial margin gradients,
  the matched training journals and their stored binary32 geometry. Use all
  24 training prefixes, once each in each of orders 17, 31 and 73. Rehearse on
  the four-prefix artificial archive, order 11. Do not substitute R=4 results.
* At position j in a rival-trained order, the journal recorded gold-minus-rival
  margin **before** updating on this prefix. Subtract that host and prefix's
  measured initial margin. Predict the difference with the initial factual
  margin gradient dotted with the **incoming** parameter displacement (gate,
  then bias). Use actual stored endpoints, not the nominal learning rate or
  the update on this prefix. Check integer binary32 arithmetic against a
  separate floating-point sum. Verify every source hash and normalized update.
* Compare against zero change and the average tangent from **other lexical
  families** with the same subject number and attractor agreement, on the
  same displacement. Fit no scale or sign. With one family the lexical
  control is unavailable, not zero. The j=0 row in each order is an exact
  initial-margin and same-full-NLL check, not a successful forecast.
* Primary descriptive checks, separately by host: smaller RMSE than zero
  in every order; separately, smaller RMSE than the lexical control in every
  order. Ties do not pass. Retain all individual errors, opposite signs and
  ties. Show j=1 separately (one preceding update); the rest test a sequence
  of updates, not a single-step effect. Show per-order and equal-prefix
  summaries; repeated orders are not independent prompts.
* Secondary observation: compare the full vocabulary NLL logged before each
  prefix under rival-trained vs ordinary full-CE-trained histories, at
  matched host, order and position. Compare at the precision of the source
  full-CE loss (binary32); negative means lower NLL for the rival history.
  The full-CE journal does not contain its binary margin. Do not infer one.
* Only for prefixes also observed at j=0 in one order is an initial full NLL
  available. On later visits to those prefixes, decompose the observed
  rival-history full NLL change into within-pair NLL and pair-coverage NLL
  changes. No claims about the other prefixes' initial full NLL.

The denominator is 138 noninitial **visits**, six initial checks, two hosts
and 24 distinct training prefixes, not 138 independent tests. No flip in
two-token preference implies no change in binary accuracy, even if margins
move. This is not a held-out generalization, checkpoint-endpoint or quality
test. Decoder nonlinearity and finite precision are not separately identified.
