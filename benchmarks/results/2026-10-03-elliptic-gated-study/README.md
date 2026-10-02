# Learned Elliptic Context: Frozen Study Launch

**This record is preflight evidence, not a completed learning result.** The
local twelve-run study has been launched from a frozen source/recipe binding.
No pretrained endpoint scores or geometry advantage are claimed here.

See [the experimental design](../../../docs/elliptic_gated_study.md). Four arms
(tangent, elliptic, gated tangent, gated elliptic) use seeds 41,43,47 and 512
primary updates each, for 6144 planned primary updates. Each completed run also
requires two continuation-only validation updates. The gated pair has 8451
parameters each; the pointwise pair has 8450, without dummy parameters.

Preflight: 121 Python tests passed with zero skips, including same-formula
Torch/native comparison at B=2,T=128, exact interrupted/resumed HF training,
paired projection hashes and signed gate trajectories. The extended offline
summarizer reproduces the previous causal study's public summary byte-for-byte.
These tests do not substitute for the running pretrained-model experiment.

`plan.json.gz` is the exact sealed launch plan, including paired schedules and
runtime/source/data hashes. `preflight.json` identifies the tested sources and
launch gates; `SHA256SUMS` covers the public record. Actual binaries, corpus text,
model weights, checkpoints and raw process logs remain local. The original
negative causal-mixing results are unchanged.

After all runs finish, publish every condition's numeric scores, signed gate
trajectories, checkpoint/continuation verification, process completion and a
completed-resume check. Do not mark this study complete from a live process or
an intermediate checkpoint. Reused Pride/Alice endpoints remain exploratory.

## Reproduce

Use the [offline launch procedure](../../../docs/elliptic_gated_study.md#run-offline)
with the public recipe and the native/client revision recorded in the plan.
Preserve the original source/package identity for `--resume`. The shared
summarizer accepts both plain and gzip JSON and must reject incomplete studies.
