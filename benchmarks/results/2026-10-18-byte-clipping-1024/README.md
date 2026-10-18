# Long Language Clipping And Process Recovery

This follows the [predeclared plan](PLAN.md): one width-16, seed-11 study,
three ordinary/learned-Poincare/frozen-Poincare arms, 1,024 updates, and two
explicit Rust-owned clipping policies. It tests whether geometric model state,
AdamW state and learning histories survive a real process interruption, and
compares the full updates with independent CPU PyTorch. It is not a speed test,
a default-recipe search, or a general language-quality claim.

The source-candidate wheel remains version 0.4.31; this record does not publish
a new PyPI package. The complete inputs, weights, checkpoints and logs are
retained privately. Only scalar measurements, validation records, hashes and
re-execution instructions are published.

## Observed Outcomes

All six full learning schedules complete. Both policies also complete a separate
512-update worker, actual SIGKILL after its durable seal, and fresh-process
512-to-1,024 continuation. Final report/checkpoint bytes, restored inference and
four-byte greedy generation agree exactly with uninterrupted execution; source
runs remain unchanged and parent identities match.

| Arm | Disabled final BPB | Active 0.01 final BPB | Active minus disabled |
| --- | ---: | ---: | ---: |
| Ordinary | 4.499275955 | 4.533757272 | +0.034481318 |
| Learned Poincare | 4.471922866 | 4.522192654 | +0.050269787 |
| Frozen Poincare | 4.496335471 | 4.536193707 | +0.039858237 |

Lower BPB is better. Clipping worsens this measurement in every arm, despite
improving numerical agreement. **The threshold is not promoted to a default.**

- Disabled reproduces both pinned earlier native and Torch case payloads.
  Its strict gate remains failed: ten final Key-bias scalars exceed tolerance
  (3 ordinary, 4 learned, 3 frozen), maximum error `4.8952114e-6`.
- Active passes the unchanged strict gate, including geometry deltas, frozen
  bits and AdamW state. Maximum final-parameter error is `2.9802323e-7` and
  maximum training/validation batch-CE error is `9.5367432e-7`.
- All five learned geometric tensor families have nonzero updates; maximum
  geometry-delta relative L2 discrepancy is `1.0477825e-6`. Frozen geometry bits
  remain unchanged. This is actual learning-state agreement, not just metadata.

[Complete results](results/results.json) retain every condition, all scalar
parameter/moment diagnostics and validation curves. [Inputs](results/inputs.json)
bind the source and requests; [artifact identities](results/artifacts.json)
cover retained private files. Recovery and numerical comparison are separate
outcomes. The active recipe does not repair the original unclipped failure or
requalify the earlier tiny-SGD minimum-update failure.

## Re-Execute With Retained Inputs

Use the authorized original inputs whose hashes are in [PLAN.md](PLAN.md), and
the isolated interpreter containing the exact candidate wheel. The public
record alone does not reconstruct the private corpus or model weights.

```sh
"$PY" -I -B benchmarks/results/2026-10-18-byte-clipping-1024/study.py run \
  --request "$PINNED_V9_REQUEST" --torch "$PRIOR_TORCH_REPORT" \
  --native "$PRIOR_NATIVE_REPORT" --wheel "$CANDIDATE_WHEEL" \
  --output "$NEW_PRIVATE_DIRECTORY"
"$PY" -I -B benchmarks/results/2026-10-18-byte-clipping-1024/study.py publish \
  "$NEW_PRIVATE_DIRECTORY" "$NEW_METRICS_DIRECTORY"
```

Every output directory must be new. GPU work is serialized. Each worker is
bounded to 1,800 seconds; the three-worker recovery command to 6,000 seconds.
No automatic retries or threshold changes follow a failed condition. Compare
the actual saved cursor 512 and final cursor 1,024; do not infer an unsaved best
model from a scalar loss history.

## Evidence Boundaries

The study was launched against base `51d19d373c5d8cee095bbe596ef5b205866d6b7d`
with an uncommitted, hash-pinned harness. The exact executed harness was then
retained in commit `5cbcfbe900a453526e7408204548dcb5ec86db4a`, without changing
files during execution. All recorded source hashes match that commit's blobs.
The original source manifest preserves its launch chronology rather than
substituting the later commit as the original HEAD.

An independent review during the run found cancellation/descendant-cleanup and
failure-retention gaps in that harness. Its executed source was archived before
repair. Neither normal numerical completion nor successful planned interruption
is evidence that those unexercised supervisor paths were safe. The separate
supervisor regression checks must not be attributed to the original long run.

The repaired publisher revalidated the old run using the explicit executed
revision, checked all 719 original source hashes against Git objects and kept
the numerical checkers/model sources unchanged. Its three published outputs
are byte-identical to the pre-repair export. The extra
[final postprocessing receipt](results/postprocessing-final.json) records this
distinction, including the complete pinned original manifest identity.
Historical revalidation uses `publish ... --executed-revision
5cbcfbe900a453526e7408204548dcb5ec86db4a`; it does not rerun training or rewrite
the launch manifest.

The separate [final supervisor recheck](results/supervisor-final.json) runs the fixed
nested supervisors on the unchanged two-update AdamW fixture with a real
step-1 SIGKILL. Recovery passes and final report/checkpoint bytes equal the
earlier fixture's bytes. Twenty-five offline tests cover bounded limits, owned
descendant cleanup, cancellation, isolated CLI startup, malformed partial
reports, failed recovery receipts and historical-source verification. This
does not establish cleanup after killing the supervisor itself with SIGKILL,
arbitrary nested daemons, or processes escaping their owned group. Earlier
postprocessing/recheck receipts remain alongside the final ones; see the
[review corrections](REVIEW.md) for the distinct source revisions.

Exact recovery demonstrates this host's process-interruption path, not power-loss
durability. Restored inference checks use fixed windows and four generated bytes;
they test identity, not free-running text quality. These same validation windows
are development measurements if used for recipe/model selection. A new separate
evaluation is necessary before deployment or a geometric-benefit claim.
