# Endpoint intervention, specified after the main study

The primary endpoint readback is sealed at
`74437a17603c8f19ee3fae3e1802adfaa4f459126c68a8edf3a731b378ea7515`.
The gate received <10% of squared parameter motion on every recorded step.
That fact alone does not measure the parameter block's effect on the decoder.

Use the same pinned CPU extension, frozen base and LoRA. Take **all** 54
endpoints, all 9 origins and the same 16 already-scored synthetic test
prompts. No training data gradients, retraining, new examples, candidate
selection or hyperparameter changes. For each endpoint compare its exact saved
geometry (F), gate only with bias at its original zero (G), and bias only
with gate at its original zero (B). Replay each shared origin O. Verify all
origin and full logits bitwise or stop. Save the actual component weights,
all 11 logits and identities; one bounded CPU launch, <=120s, no retry:
(9+3*54)*16=2736 forwards, no backward/updates.

For each prompt's two-token margin average the contribution of changing a
block first and changing it last. Gate = ((G-O)+(F-B))/2; bias =
((B-O)+(F-G))/2. Also report the full interaction F-G-B+O. Verify additive
closure of the allocated contributions. Compare per-prompt RMS gate vs bias
(no cancellation across prompts); keep signed mean changes under each endpoint's
teacher, all component winner switches vs origin, and the underlying term
sizes. These are interventions at *trained endpoints*. They cannot establish
what would happen if bias were frozen during training. No new transfer claim.
