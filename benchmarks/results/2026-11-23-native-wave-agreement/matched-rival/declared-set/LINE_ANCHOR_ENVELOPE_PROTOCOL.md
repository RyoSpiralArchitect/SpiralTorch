# Frozen protocol: a bound on the most changed prompt

Source parent: `0ab5119a1c8e1cae0d791ee41b5f0799eae624ed`. Prior native raw SHA-256:
`9a60f7c3592e1844dd15d5a1f71f2e2119794c635c5ab13c7561543fabd07aa9`.
Small generated ordinary/wave decoder and nonzero LoRA, the same three training
examples and anchor order, vocabulary 11, same two epochs. No validation text,
pretrained model calls, GPU, network, release, cache deletion or original modification.

For each owner (LoRA only / trainable geometry only / both), in separate trajectories:
1. Replay the original mean-KL policy against retained native losses, movements,
   trials and final checkpoint pins. T = 1/65536 nat; C = 2T.
2. Primary: mean forward KL in [0.98T,T] **and** every individual forward KL <= C.
3. Separate positive control: max forward KL in [0.98C,C]; no floor on the mean.
   This is not a substitute match under the primary condition.

Full-vocabulary last-position KL(before || proposed), actual f32 stored candidates.
All three conditions use fresh gradients on each current arm, a maximum parameter
distance 1/8, that ceiling first and at most 19 midpoints. On an upper violation
search below; if all upper bounds hold but the required floor is missed, search
above. A too-small first ceiling or exhausted search refuses without publishing or
moving either weights or cursor. No retry, target tuning, altered direction, skipped
example or partial run counted as complete. These searches are not a proof of global
infeasibility along a nonlinear ray.

Record candidate distances, means, and (for the two new policies) worst KL and
first maximizing anchor index. Independently score each actual accepted update and
all tested candidates of a refused search using f64 log-softmax from real f32 logits.
Independent candidate reconstruction on a refusal is verification, not additional
committed training. Record successful update counts, partial endpoints, failure
example, per-prompt movement and cumulative parameter distance. Only compare final
losses of **all three complete arms under the same policy**. All three objectives
are on training examples; do not pool them as accuracy.

Verify strict failure and opt-in structured refusal are atomic, including checkpoint
resume and invalid-plan rejection. Check the same outcomes in installed native Python,
WASM/Node and rendered Chrome; tolerate absolute loss differences up to 3e-6 and
absolute KL differences up to 1e-8 across engines, never tolerate admission outside
the specified bounds. Unexpected process or numeric errors stop the affected run and
are preserved separately from scientific refusals.
