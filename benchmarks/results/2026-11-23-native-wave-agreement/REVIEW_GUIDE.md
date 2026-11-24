# Native language learning: review guide

This series connects native GPT-2 adaptation, cached scoring, and controlled
tests of what an optional geometry gate changes. Three checkpoints in the
evidence are useful for different questions:

| Question | Start here | Boundary |
| --- | --- | --- |
| What changes on a pretrained decoder? | [Frozen GPT-2 small pilot](README.md), [four local gradient paths](matched-rival/parameter-vjp/RESULTS.md) | Small, fixed language tasks. Training-order repeats are not independent datasets. |
| Does the gate actually read a different signal? | [Attention vs. a uniform or fixed-position value read](matched-rival/placement-transfer/prefix-transport/paired-gap/weakest-lane/attention-value-read/RESULTS.md) | Synthetic decoders; ordinary backbone attention stays intact. |
| Is the effect just a threshold shift? | [Swap the fitted weights and serving read](matched-rival/placement-transfer/prefix-transport/paired-gap/weakest-lane/attention-value-read/crossed-serving/RESULTS.md), [calibrated scalar controls](matched-rival/placement-transfer/prefix-transport/paired-gap/weakest-lane/attention-value-read/crossed-serving/offset-control/RESULTS.md) | Already-scored synthetic cases, not a fresh test of general language ability. |

## Code entry points

- [Native paired objectives](../../../crates/st-nn/src/causal_lm/paired_token_gap.rs) and the [value-read path](../../../crates/st-nn/src/causal_lm/value_read.rs) own the model semantics.
- [Declared-set geometry from Python](../../../bindings/st-py/examples/declared_set_geometry.md) and the [WASM finite-update and cache fixture](../../../bindings/st-wasm/tests/native_causal_lm.md) show which routes are available to each client.
- [Rust offset example](../../../crates/st-nn/examples/paired_token_offset.rs): fit with pairs, then apply one fixed offset to the two-token contrast of a **single** prompt. It does not rewrite the full vocabulary distribution.

Each linked study distinguishes its protocol, result, and limits. Verification
using private original journals requires the named input hashes. The public
tests instead use deterministic tiny fixtures; pretrained weights, private
run archives, wheels and machine-specific build products are not PR inputs.

The [source commit map](source-commit-map.tsv) translates historical IDs in
the frozen records to public commits. Each pair has the same Git tree, commit
message and timestamp. Only machine-local email addresses and the resulting
parent commit IDs changed in the public history.
