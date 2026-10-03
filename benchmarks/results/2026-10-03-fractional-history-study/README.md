# Independent Fractional History Study

**Status: completed and locally verified.** All twelve runs, their exact
adapter/Adam continuations, and delayed endpoint evaluations finished with
process exit code 0. The [fixed protocol](../../../docs/fractional_history_study.md)
compares pointwise, ordinary learned EMA, fixed GL history and learned GL
history after frozen GPT-2's first MLP. Fractional math and derivatives are
Rust-owned; the EMA control uses ordinary Torch operations.

Three seeds, four active arms, 512 updates per run: 6144 primary updates and
24 additional continuation-only updates. CPU float32, two threads, B2/T128,
768 features, K32, strength 0.1, Adam learning rate 0.001, initial alpha/decay
0.5. Both local and history gates start at zero. All runs and exact next-update
checks finish before the 120 Pride / 32 Alice endpoint blocks are scored.

Total/trainable parameter counts are pointwise 768/768, EMA 1537/1537,
fixed GL 1537/1536 and learned GL 1537/1537. No dummy parameters. History
versus pointwise changes capacity; EMA versus learned GL matches count, not
filter initialization, conditioning, mathematical prior or compute. This is
not a speed comparison. These books are reused exploratory data, not pristine
confirmation. Seeds vary paired minibatch order/coverage, not initialization.

## Outcomes

Mean endpoint cross-entropy in nats per next token; lower is better. Active
means cover all three seeds. The frozen baseline is shared and receives no
updates. All numeric per-seed/block losses remain in the compressed result.

| Arm | Pride (120 Blocks) | Alice (32 Blocks) |
| --- | ---: | ---: |
| Frozen baseline | 4.073713269 | 4.011807486 |
| Pointwise | 4.063162282 | 4.005609319 |
| Learned EMA | 4.057392899 | 4.002461868 |
| Fixed GL history | 4.058227340 | 4.002909807 |
| Learned GL history | 4.056442695 | 4.001746317 |

| Paired Contrast | Pride Mean CE Difference | Alice Mean CE Difference | Improving Seeds (Pride / Alice) |
| --- | ---: | ---: | ---: |
| Fixed GL minus pointwise | -0.004934942 | -0.002699512 | 3/3 / 3/3 |
| Learned GL minus fixed GL | -0.001784645 | -0.001163490 | 3/3 / 3/3 |
| Learned GL minus pointwise | -0.006719587 | -0.003863002 | 3/3 / 3/3 |
| Learned GL minus EMA | -0.000950204 | -0.000715551 | 3/3 / 3/3 |
| EMA minus pointwise | -0.005769383 | -0.003147451 | 3/3 / 3/3 |

All history arms improve over pointwise here. Learned GL slightly improves
over EMA in each seed, but EMA beats fixed-order GL in each seed on both sets.
The much larger history-versus-pointwise difference includes added capacity;
it cannot be attributed uniquely to fractional geometry. The learned/fixed
GL contrast also adds one trainable order scalar.

| Seed | Learned Final Alpha | Learned Final EMA Decay |
| --- | ---: | ---: |
| 41 | 1.034797907 | 0.341286510 |
| 43 | 0.988713503 | 0.352485746 |
| 47 | 0.995676756 | 0.356852889 |

Both gates receive nonzero gradients at all 512 updates in every history arm.
Each learned scalar receives zero gradient behind the initial zero gate and
511 nonzero gradients thereafter. Fixed alpha remains exactly 0.5 with no
order gradient or Adam state. Gate changes and scalar values match the saved
adapter/optimizer contents, not merely a telemetry counter.

**Interpretation limit:** alpha approaches 1, where the strict GL map at step 1
is exactly `-x[t-1]`, with no longer-lag forward taps. EMA decay also decreases.
This is consistent with a short-memory explanation, not proof of one. The
next comparison should include an ordinary single-lag control before claiming
that long fractional memory is responsible. The small EMA gap, three paired
seeds, reused books and one frozen GPT-2 model do not establish significance,
unique geometric advantage, general LLM improvement, or a new default.

The [older full-GL experiment](../2026-10-03-fractional-memory-study/README.md)
remains a separate historical comparison, not extra arms in this study. The
three rerun pointwise controls reproduce its adapter weights, Adam states,
all update/development records and per-block endpoint scores exactly; the
shared baseline also matches. See `ordinary-control-verification.json`.

## Verification And Reproduction

Training clients were frozen at
`e8eff55fd3ae211b51fe3f91b516a1a58344fd2e`; the native package was verified from
`237143ce368fed5d6df2d2be22f42f10c338beb4`. `frozen-runtime-sha256.json` lists
all six client/config/summary files and 67 package files. The later empty-history
fix is not hot-swapped into this run: K32/T128 does not exercise that branch.
Current source includes that fix and is tested separately.

`plan.json.gz` and `results.json.gz` preserve the original JSON bytes under
gzip. `journal.json` seals the result/checkpoint hashes. The read-only saved
content check validates all twelve recipes, cursors, schedules, frozen/trainable
modes, gate/scalar states, Adam state and endpoint receipts. The original
driver executed the frozen-base and exact next-update checks. This evidence
does not expose private weights for independent third-party inspection.

A completed `--resume` exited without rescoring; plan, journal and result bytes
remained unchanged. All 73 frozen files were rechecked. The summary is stable
under different Python hash seeds; CI includes a byte-for-byte rebuild test. Public
checksums bind numeric results and verification records; models, corpora,
checkpoints, native packages and raw logs remain local. The launch preflight's
historical `running` status is unchanged; `validation.json` records completion.

Rebuild the summary without loading models or scoring text:

```sh
RESULT=benchmarks/results/2026-10-03-fractional-history-study
python tools/summarize_wave_gate_long_horizon.py \
  --plan "$RESULT/plan.json.gz" --results "$RESULT/results.json.gz" \
  --journal "$RESULT/journal.json" --output "$NEW_SUMMARY_JSON"
cmp "$NEW_SUMMARY_JSON" "$RESULT/summary.json"
```

For a new training run, use the linked protocol and hash-matching offline
assets. Resume requires the original frozen runtime/configuration, not a later
helper revision. History resets per unpadded block; no packed-document reset,
KV-cache decode, mixed-precision or resident-GPU claim is made.
