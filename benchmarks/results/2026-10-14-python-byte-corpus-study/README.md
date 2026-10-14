# Python Byte-Corpus Client Qualification

This is an **unreleased-source client qualification**, not a new language-quality
or speed study. The model-level 0.4.30 release is already published; the candidate
wheels used here retain that version metadata but are private source builds, not
replacement PyPI artifacts. Candidate hashes distinguish them.
Python GPU execution used an Apple M4 integrated GPU through WGPU/Metal.

## What Ran

- An installed WGPU wheel ran the unchanged
  [v1 pilot](../2026-10-09-byte-corpus-matched-learning/README.md): three seeds,
  ordinary/geometric arms, six cases, 128 accepted updates each. Every case
  passed the original independent CPU-f32 PyTorch numerical gates. The reference
  is the frozen historical result, not a fresh PyTorch run.
- A second execution paused at update 37 and resumed through a serialized
  checkpoint to 128. Complete histories and final weights equal uninterrupted
  execution, including signed-zero/type-sensitive checks.
- The [v8 study](../2026-10-13-byte-corpus-query-controls-v8/README.md) supplied
  independently retained native and browser checkpoints at update 128:
  25 arms x three seeds, 75 cases per client. Python imported each and performed
  a **zero-update** continuation. Each retained its own checkpoint and case
  reports exactly and passed the unchanged historical PyTorch comparison.
  This is completed-state transport, not 75 new learning runs or a general
  cross-device reproducibility claim.
- Tiny live-GPU tests separately train and resume v4/v5 calibrated, v6 headwise,
  v7 shared-mixture and v8 contextual recipes; concurrent calls share an
  immutable recipe, not mutable live models.

The common owner remains Rust. Python orchestrates handles and transports exact
JSON. Ordinary and geometric arms have different parameter counts and compute;
these results do not establish matched-cost superiority, HF FT, KV-cache
inference, or a PyTorch speedup.

## Evidence

`v1-torch-comparison.json` and both `v8-*-comparison.json` preserve every
condition, the numerical criteria and geometry deltas. `v1-resume-verification.json`
and `strict-case-equality.json` preserve continuation/identity checks.
`summary.json` records validation and source/candidate identities;
`raw-manifest.json` hashes the privately retained requests, raw outputs,
checkpoints and logs. `SHA256SUMS` covers this public record.
No model weights or request documents are added here.

The v1 mean geometry-minus-ordinary held-out difference remains approximately
-0.001682 bits/byte, with mixed per-seed signs. It reproduces the old bounded
pilot; it is not a new general efficacy claim.

## Replay

Use an installed candidate wheel built from this source, not published 0.4.30.
Keep the unchanged historical archive directories referenced by `replay.py`
under one local archive root. Their preparation/reconstruction instructions and
input hashes remain in the original v1/v8 records above.

```bash
mkdir "$ARCHIVE/new-python-byte-replay"
python -I -B benchmarks/results/2026-10-14-python-byte-corpus-study/replay.py \
  "$PWD" "$ARCHIVE/new-python-byte-replay"
python -I -B bindings/st-py/tests/test_nn_byte_study.py -v
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_byte_study.py -v
```

Replay refuses to replace output files. It reuses the historical requests and
reference outputs without refitting, changing tolerances, or choosing winners.
Its elapsed times are diagnostic bookkeeping, not performance comparisons.
See the [Python API guide](../../../docs/python/byte-corpus-studies.md).
