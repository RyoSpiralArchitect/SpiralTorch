# Directional Elliptic Comparison: Running

The real experiment has started but has **not completed**. No final learning
advantage is claimed. Source and protocol were committed before launch at
`4227af8c36476eb4c85272dec0d5f379af1b0ec2`; execution uses frozen local copies.

See [the protocol](../../../docs/elliptic_nonlinear_study.md) for the exact arms,
reproduction command and limitations. All three arms share the learned `768 -> 2`
projection and zero-start `9 -> 768` readout: **8450 trainable parameters each**.
The controls are the Rust-derived tangent feature map, componentwise tanh of its
displacement, and the full Rust elliptic/Lie feature map at `(1,u,v)`.

## Fixed Before Outcomes

- Cached frozen float32 CPU GPT-2, explicit `transformer.h.0.mlp` placement.
- Seeds 41/43/47, paired projection initialization and paired minibatch order.
- 512 Adam updates per run, batch two, block size 128, lr 0.001, strength 0.1.
- 130048 causal targets per run; nine runs and 4608 primary updates planned.
- Final scoring only after all runs and exact next-update continuation checks.
- No checkpoint selection, early stopping or endpoint-driven configuration change.

The Pride/Alice evaluation partitions were inspected by the preceding WaveGate
study. This is a reused exploratory benchmark, not a pristine held-out test.
Within this comparison, parameter counts and initialization match; feature rank,
expressivity, typical feature scale, gradient conditioning and compute need not.
The tanh arm is not an unconstrained learned MLP. The 8450-parameter arms must not
be presented as a parameter-matched victory over WaveGate's 1536/1537 parameters.
No speed, text-generation-quality or generalization claim follows from wiring.

## Verification Available

`plan.json` records exact hashes, partitions, batch schedules and both adapter
source identities. `validation.json` records **68 passing local tests** with
zero skips. Tests include actual tiny HF loss gradients through both projections,
paired initial parameters, RNG preservation, local differential agreement,
wrong-control checkpoint rejection and interruption/resume in the second arm.
The shared runner reuses the earlier checkpoint and evaluation boundary instead
of duplicating those mechanisms. Previous published WaveGate summaries remain
byte-for-byte reproducible with the extended summary tool.

The existing native Rust binary and local model/corpora are reused unchanged.
No native build or model download was performed. Raw terminal logs and optimizer
checkpoints stay local. This record must not be promoted to completed evidence
until the actual process terminates successfully and its sealed result verifies.
