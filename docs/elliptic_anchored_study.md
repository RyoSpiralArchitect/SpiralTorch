# Fixed Anchor Versus Learned Context

The [frozen-checkpoint intervention](elliptic_context_ablation.md) found that a
fixed anchor could improve an already trained elliptic adapter on reused test
sets. It did not establish that training an anchored model would help. This study
tests that distinct hypothesis by training all four arms from identity, using
the [Rust-owned anchored operator](elliptic_anchored_learning.md).

## Fixed Design

`bindings/st-py/examples/hf_elliptic_pride_anchored.json` fixes local float32 GPT-2,
the first MLP output insertion, seeds 41/43/47 and 512 updates per arm. Batch 2,
context 128, strength 0.1, Adam learning rate 0.001, corpus split, evaluation
blocks and CPU thread count are unchanged from the earlier gated study.

| Arm | Feature map | Learned correction target | Parameters at F=768 |
| --- | --- | --- | ---: |
| anchored_tangent | ordinary linearization at the Rust anchor | fixed anchor | 8451 |
| anchored_elliptic | Rust nonlinear elliptic/Lie map | fixed anchor | 8451 |
| gated_tangent | same ordinary linearization | causal feature mixture | 8451 |
| gated_elliptic | same Rust nonlinear map | Rust causal feature mixture | 8451 |

Each correction is `(1-g)*features + g*target`, where `g=tanh(raw_mix)` starts at
zero. The fixed target is `phi(1,0,0)` from the same warp. No dummy parameter,
trainable anchor, quadratic anchor attention, or sequence cache is introduced.
All projection/readout/gate tensors and minibatch schedules are identical within
each seed. Full parameter and projection-only hashes must establish that pairing.
Zero readout makes each adapter initially identical to the frozen model.

All twelve runs are trained anew with one frozen package, not initialized from
earlier endpoints. Exact replays of old gated arms, if observed, demonstrate
reproducibility, not independent replication. The fixed-anchor map is pointwise;
GPT-2's hidden states are still contextual. Gate scaling is partly absorbable by
the readout, so optimization effects are not necessarily added expressivity.

Rust owns nonlinear features, blending and VJPs. The ordinary experimental control
uses its anchor/Jacobian and explicit Torch arithmetic. Equal parameter count and
updates do not imply equal computation, feature rank or mathematical function.
This is a quality/learning comparison, **not a speed comparison**. Speed benchmarks
belong to separate, mathematically equivalent Torch versus native paths.

## Evidence Boundaries

Primary contrast: anchored elliptic minus anchored tangent. Also report geometry
within gated context, anchor minus context within each map, and the difference of
geometry contrasts. Negative CE differences favor the named arm. Preserve all
seeds, all block losses, the frozen baseline and signed gate trajectories. Three
seeds and shared evaluation blocks do not establish statistical significance.

No hyperparameters are selected from development losses. Endpoint evaluation stays
locked until all runs, frozen-base checks and exact next-step adapter/Adam resume
checks pass. The existing exclusive writer lock, immutable checkpoints, hash-sealed
plan and atomic journal are reused. A completed resume must not rerun endpoints.
Only numeric results, source/data hashes and verification receipts are public;
models, text, checkpoints, native packages and raw logs remain local.

Pride and Alice motivated this follow-up after repeated inspection. They are
exploratory reused sets, not untouched confirmation or evidence of general LLM
advantage. A gate moving, a passing derivative test, or a lower development loss
is not by itself a positive outcome.

## Run Offline

Use the anchored native package with matching Python clients, Torch and
Transformers 4.57.6. No downloads, API calls or hosted jobs are needed.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python bindings/st-py/examples/hf_elliptic_anchored_study.py \
  --config bindings/st-py/examples/hf_elliptic_pride_anchored.json \
  --model-dir /path/to/gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --corpus /path/to/pride_and_prejudice.txt \
  --transfer-corpus /path/to/alices_adventure_in_wonderland.txt \
  --output-dir /path/to/new-study
```

`--resume` requires unchanged source/package/data/recipe and the original output
directory. `tools/summarize_wave_gate_long_horizon.py` reads completed sealed
JSON without importing Torch or evaluating a model. This protocol explicitly
sets `reference_arm=anchored_tangent`; older studies retain their original
reference and byte-for-byte summary output.

Preflight tests include `test_elliptic_anchored_study.py`, the shared learning
tests and `test_wave_gate_long_horizon_summary.py`. They verify numerical VJP
parity at the training shape, no accelerator RNG changes, equal initialization,
schema isolation, actual tiny-HF training/interrupted resume and rejection of
unpaired, mislabeled or incomplete outcomes. Preflight is not pretrained-model
quality evidence.

## Completed Outcome

The [completed twelve-run study](../benchmarks/results/2026-10-03-elliptic-anchored-study/README.md)
finds that anchored elliptic improves over its gated-context counterpart in all
seeds on both sets (-0.028651 mean Pride CE, -0.011172 Alice). The ordinary
anchored tangent still wins in every seed: the primary geometry difference is
+0.116164 on Pride and +0.064956 on Alice. Ordinary anchor/context effects are
mixed on transfer, with an explicit seed-43 regression. All losing conditions
are retained, not hidden by the favorable mean interaction.

All 6144 primary updates, 24 continuation checks and endpoint evaluations
completed; checkpoint contents are finite, frozen bases unchanged and completed
resume is a byte-identical no-op. The six gated-context controls exactly replay
their earlier records and scores; they are not independent new replications.
