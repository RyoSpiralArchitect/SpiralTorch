# Same-Pipeline Ordinary Control For HF Geometry

Source candidate after 0.4.31, based on `bc88586a806e766a1654be5f447771c6da661ccb`.
This adds an executable ordinary control, not another trained model or package
release. Rust owns the affine branch and VJP; Python and WASM expose it.

## Results

Offline cached GPT-2 small, frozen float32 base, MPS HF execution and Rust CPU
adapter kernels. Both arms have 6,144 trainable values at MLP outputs 8..11:
`ChannelAffineAdapter` versus radius-free `WaveGateAdapter(curvature=-1)`.
Both start with zero gate/bias, strength 0.1, identical output and first-update
derivatives. Every run uses the public `hf_geometry_finetune.py` example and
`hf_geometry_trainer_class()`, not separate update loops.

| Seed | Ordinary CE | WaveGate CE | WaveGate minus ordinary |
| --- | ---: | ---: | ---: |
| 41 | 5.1397404671 | 5.1407318115 | +0.0009913445 |
| 43 | 5.1396937370 | 5.1407084465 | +0.0010147095 |
| 47 | 5.1399149895 | 5.1408700943 | +0.0009551048 |

The bare-base and identity-start CE are 5.1423892975 in all six runs. Every
intended adapter changes; base weights remain unchanged; reconstructed inference
logits exactly match the live stack. **WaveGate is worse than the ordinary
control in all three seeds of this bounded probe.** No geometric default or
quality claim is promoted.

The fixture is deliberately small and already used for workflow development:
six repository sample documents yield 13 train windows; a related long-form
document yields 30 evaluation windows. This is not an untouched novel or broad
evaluation corpus. Context 32 gives 403 train targets per complete pass and
930 evaluation targets; 60 train and 31 evaluation tail tokens are dropped.
Both arms use 32 optimizer steps, batch size 2, AdamW, linear schedule, LR 0.001,
max gradient norm 0.1, no accumulation, eager attention and no KV cache.
The files are disjoint, but related domain/content remains a limitation.

`results.json` contains every condition, identities and compact window-plan
statistics. The full private summary and predeclared six-case manifest are
retained by hash. The repaired driver was run again from scratch: all six
artifact IDs, checkpoint IDs, evaluation metrics, data and base identities
exactly match the original successful run.

## Verification

- Installed WGPU-default source wheel: 182 Python tests pass. Coverage includes
  numeric/list/buffer equivalence, empty inputs, invalid types/budgets,
  non-contiguous Torch transport, mutation guards, old artifact variants,
  optional-Torch imports and real HF GPT-2/Llama loops.
- Rust CPU: 4 affine and 6 artifact tests pass. Minimal Python
  `--no-default-features --features extension-module,cpu` also checks successfully.
- Actual wasm32 `nn` build and Node execution pass affine, legacy 13-placement
  artifact and document-window checks. Native and WASM issue exactly the same
  new artifact/layout identities. Browser HF execution is not claimed.
- Four stdlib checks pass: success, failure/cancellation, post-run qualification
  failure and CPU/MPS companion selection. Failed and not-started conditions remain in
  `run-manifest.json`; exceptions are rethrown, with no retry or cherry-picking.
- Ordinary control on pretrained GPT-2, seed 41: uninterrupted 32 steps equals
  stop at 16 then fresh-process continuation to 32. Checkpoint geometry bytes,
  optimizer/scheduler, CPU/Python/NumPy RNG and MPS RNG match at both the stop
  boundary and final boundary. Exported artifact and evaluation loss also match.
  This is planned interruption, not a SIGKILL/durable-storage qualification.

All hashes, test counts and continuation checks are in `validation.json`.
Private logs retain the original and revised-driver executions, checkpoints,
learned artifacts and generated text. None of those contents are republished.
The source manifest records publication-time files; five packaged Python
members are byte-checked against the qualified wheel; those installed members
and the native extension also match that wheel. The late Rust changes
after wheel construction were confined to `cfg(test)` tests and were compiled
by the separate Rust test run.

## Replay

Use a source wheel containing this commit, Python 3.12, Torch 2.12.1,
Transformers 4.57.6 and Accelerate 1.14.0. No model or corpus download is performed.
Set `MODEL` to an existing local GPT-2 small directory. The qualification used
revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; actual runtime/data identities
are retained in `results.json`.

Recover only these tracked sample texts into a new private directory if needed,
without restoring the intentionally pruned checkout:

```bash
mkdir -p "$DATA"
git archive a2c73f38e7296ae98650239dcfa9cbe8f2d63921 \
  models/samples/spiral_corpus_en | tar -x -C "$DATA"
CORPUS="$DATA/models/samples/spiral_corpus_en"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
python -I -B benchmarks/results/2026-10-18-hf-affine-control/replay.py \
  --model-dir "$MODEL" --device mps --output "$NEW_PRIVATE_OUTPUT" \
  --train-text "$CORPUS/00_spiral_torch_intro.txt" \
  --train-text "$CORPUS/01_char_language_basics.txt" \
  --train-text "$CORPUS/02_z_space_notes.txt" \
  --train-text "$CORPUS/03_wgpu_and_traces.txt" \
  --train-text "$CORPUS/04_geometry_maxwell.txt" \
  --train-text "$CORPUS/05_roadmap.txt" \
  --eval-text "$CORPUS/06_spiral_longform.txt"
```

For the MPS continuation probe, run the same public FT example with the produced
`ordinary-initial.json`, its separately retained ID, the same files, seed 41
and recipe above, adding `--stop-after 16` into a new output directory. Then run
again into a third new directory with `--resume-from-checkpoint` pointing to
that run's `checkpoint-16`, `--geometry-checkpoint-id` from the retained
`checkpoint_artifact_id`, and `--expected-dataset-id` from its result.
Keep `--max-steps 32` unchanged. Compare only trusted local checkpoint files:

```bash
python -I -B benchmarks/results/2026-10-18-hf-affine-control/verify_continuation.py \
  --full "$NEW_PRIVATE_OUTPUT/ordinary-seed-41" \
  --partial "$PRIVATE_PARTIAL" --resumed "$PRIVATE_RESUMED"
```

HF companion pickle files are not safe inputs from an untrusted source.
Hash equality is not authenticity. The driver manifest is atomic replacement
within a live local process, not fsync-backed crash recovery or a distributed
job service. Language quality, exact continuation, GPU speed and promotion
remain separate decisions.
