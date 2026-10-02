# Chart-Step Training Study

**Status: running at publication.** This is a frozen plan and preflight record,
not a completed comparison. No endpoint score, quality win or throughput claim
is reported here. The training process continues locally; results will only be
published after all twelve runs and their verification complete.

The [Rust operator and protocol](../../../docs/elliptic_chart_step.md) compares
ordinary anchored tangent and native anchored elliptic, each with ordinary Adam
and chart-corrected Adam proposals. All four arms have 8451 parameters and paired
initialization/minibatches. Fixed recipe: three seeds, 512 updates per run,
learning rate 0.001, relative damping 0.1, CPU f32, frozen local GPT-2. Total
planned primary updates: 6144, plus two continuation-only updates per run.

Rust owns the mean chart metric and the norm-preserving correction. Adam consumes
unchanged raw gradients. The correction changes only the orientation projection
proposal, not the gate/readout updates. Input covariance and downstream readout/
loss curvature are omitted; this is not a full natural-gradient optimizer.

`plan.json.gz` is the original unmodified launch plan, including data identity,
batch schedules, code/native hashes and protocol notes. `preflight.json` is a
historical **preflight-only** record. It verifies 16 Rust and 204 Python tests,
strict TypeScript, native builds and actual Node-hosted WASM updates. Tiny HF
interruption/resume checks and rollback tests passed before the full run.
None of those checks establishes pretrained model improvement.

Training source: `76ccee3649bcadd407539de6e14b832cc1e480e0`.
Summary source: `48199ca5163cc1a3ced8662f2f82c6a429e1b328`.
The Python/native package and transitive client files are frozen separately
locally; changing the checkout does not change the running experiment.
Models, corpus text, checkpoints and raw logs remain private/local. Only numeric
outcomes and hash-bound verification will be added here. The two earlier deleted
large benchmark archives are not restored or included in this change.

## Reproduction

Use the pinned source/runtime described in `preflight.json`, build/install its
native Python binding, and provide the same local model/corpora:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_elliptic_chart_step_study.py \
  --config bindings/st-py/examples/hf_elliptic_pride_chart_step.json \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Resume only with the same frozen code/native package and `--resume`. Do not
overwrite or regenerate the launch plan. The study locks endpoint evaluation
until all runs finish and their next updates reproduce exactly. Development
scores do not select settings or checkpoints. Pride/Alice were inspected in
prior experiments, so this is exploratory rather than untouched confirmation.

Once completed, the read-only summarizer reports all five factorial contrasts,
including chart elliptic minus Adam elliptic, every losing seed, gate trajectories
and per-run proposal norms/direction cosines. Historical Adam replays, if exact,
will be labeled reproducibility evidence rather than independent replication.

```sh
python tools/summarize_wave_gate_long_horizon.py \
  --plan "$NEW_STUDY_DIRECTORY/plan.json" \
  --results "$NEW_STUDY_DIRECTORY/results.json" \
  --journal "$NEW_STUDY_DIRECTORY/journal.json" \
  --output "$NEW_SUMMARY_JSON"
```

Do not use the summarizer on a partial run. Checksums cover this publication's
files and do not certify training completion.
