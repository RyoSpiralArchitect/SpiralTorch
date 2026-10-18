# Document-Safe Geometric HF Learning

[API and limitations](../../../docs/python/hf-geometry-data.md) |
[Trainer](../../../docs/python/hf-geometry-training.md)

Source-candidate qualification after 0.4.31; not a PyPI release or a quality
claim. Rust owns the single-document window plan. Python returns ordinary HF
rows and supports spawned dataloader workers; WASM exposes the same plan and
identity, not browser HF execution.

## What Was Exercised

- A rebuilt installed macOS arm64 wheel, without a source overlay: **117 tests**
  passed, including 38 Trainer tests and 20 document-dataset tests.
- Three Rust window tests and real wasm32/Node checks, including the shared
  fixture identity, rejected coercions and geometry-artifact regression.
- A locally cached, pretrained GPT-2 base, float32/eager attention on Torch MPS.
  Rust geometry runs on CPU with Torch transfers; this is not a WGPU performance
  comparison. No model/data downloads or external inference API calls.
- Four geometric families in one stack: WaveGate with learned radius, Topos,
  Elliptic anchored residual and Fractional angle/gain history, at MLP sites
  8/9/10/11. All four changed trainable tensors; **12,294 geometric scalars**
  were trainable. Base parameter hashes remained unchanged.
- Six repository sample documents for training, a separate seventh for
  evaluation: 13 train contexts and 30 evaluation contexts of 32 tokens. The
  training input has 476 tokens, drops 60 tail tokens, and exposes 403 shifted
  targets per complete dataset pass. Evaluation exposes 930 targets.
- A 16-update run and a separate 8-update stop plus fresh-process continuation
  using the same 16-update schedule, batch size 2, seed/data seed 41, AdamW at
  0.001 and maximum gradient norm 0.1.
- Final geometry bytes, optimizer, scheduler, CPU/Python/NumPy RNG and MPS RNG
  matched exactly. Exported geometry reconstructed identical inference logits
  in each run.

| Held-out reference | Loss |
| --- | ---: |
| Frozen bare base | 5.142389297485352 |
| Initial geometry | 5.142389297485352 |
| Trained geometry, uninterrupted | 5.140507221221924 |
| Trained geometry, resumed | 5.140507221221924 |

This small loss movement is **not** evidence of an advantage over ordinary FT,
LoRA or another geometry, nor of general language quality. There is one seed,
one small domain sample and no trained matched baseline. The families were
combined, not individually ablated. Native gradients and recovery were the
target of this qualification, not throughput.

## Negative Evidence Retained

The first real-model resume diverged despite identical update-8 geometry and
CPU/MPS RNG. Its final export was
`sha256:ecbd8d7aa07a6e7a7c3c6b6146281562d8a593f1cb9171dc3a0382789b5b4b80`.
A separate CPU regression reproduced the problem in a later epoch.
Accelerate's skip wrapper reset its sampler to epoch zero. The fix lives in
the parent Trainer commit, `77a89e8a84121187bff67b4479fc313d1ed83c16`.

The fixed uninterrupted and resumed exports are both
`sha256:4c907c65189b1543eeca56e61f6cf8bff257adeb9385ccd8cdc06150a61fe131`,
also equal to the original uninterrupted export. Recovery was repaired without
changing the ordinary training trajectory. The old failed run is not deleted
or counted as a success. Resume `train_loss` is an HF partial-run aggregate and
is not compared to the full-run average.

## Reproduce

Use a new private directory and the qualified Python/Torch/Transformers/
Accelerate versions recorded in `validation.json`. Build and install the source
wheel; install its HF geometry dependencies. Tests:

```sh
cargo +1.99.0 test --locked -p st-nn --no-default-features --lib language_windows -- --test-threads=1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python -I -B -m pytest --import-mode=importlib --confcutdir=bindings/st-py/tests \
  --rootdir=bindings/st-py/tests -q bindings/st-py/tests/test_hf_geometry_dataset.py \
  bindings/st-py/tests/test_hf_geometry_trainer.py bindings/st-py/tests/test_geometry_artifact.py \
  bindings/st-py/tests/test_geometry_adapters.py bindings/st-py/tests/test_geometry_optional_dependency.py
cargo +1.99.0 build --locked -p spiraltorch-wasm --target wasm32-unknown-unknown --release --no-default-features --features nn
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm --target nodejs --out-dir "$PRIVATE/wasm"
node bindings/st-wasm/tests/causal_document_windows.cjs "$PRIVATE/wasm/spiraltorch_wasm.js"
```

Use wasm-bindgen 0.2.129. MPS checks require real Apple hardware; they are skipped
elsewhere, and exact numerical identity is qualified only in this environment.

For the offline model probe, set `MODEL` to a local GPT-2 snapshot (qualified
revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`) and `PRIVATE` to a new private
output root. No command below downloads a model. Extract only the sample data:

```sh
export MODEL PRIVATE
mkdir -p "$PRIVATE/data"
git archive a2c73f38e7296ae98650239dcfa9cbe8f2d63921 models/samples/spiral_corpus_en |
  tar -x -C "$PRIVATE/data"
```

Create the explicit initial recipe through public APIs, retaining the file's
exact-byte ID independently. The helper only loads local config/tokenizer files.

```python
import hashlib, os, runpy
from pathlib import Path
import torch
import spiraltorch as st

runtime = runpy.run_path("bindings/st-py/examples/hf_geometry_finetune.py")["local_runtime"]
_, _, report = runtime(Path(os.environ["MODEL"]))
torch.manual_seed(41)
geometry = st.GeometryAdapterStack({
    "transformer.h.8.mlp": st.WaveGateAdapter(768, strength=.1, log_radius=0., learnable_radius=True),
    "transformer.h.9.mlp": st.ToposResonatorAdapter(768, strength=.1),
    "transformer.h.10.mlp": st.EllipticAnchoredResidualAdapter(768, strength=.1),
    "transformer.h.11.mlp": st.FractionalAngleGainHistoryAdapter(
        768, kernel_len=16, initial_angle=.2, strength=.1),
})
artifact = geometry.export_artifact(base_model_identity=report["observed_identity_id"])
path = Path(os.environ["PRIVATE"]) / "initial-geometry.json"
path.write_text(artifact.to_json() + "\n", encoding="utf-8")
print("sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
```

Run `bindings/st-py/examples/hf_geometry_finetune.py` with:
`--model-dir "$MODEL" --geometry-artifact "$PRIVATE/initial-geometry.json"`,
the retained `--geometry-artifact-id`, one `--train-text` for each sample
`00` through `05`, and `--eval-text` for `06_spiral_longform.txt`.
Keep `--context-length 32 --max-steps 16 --save-steps 8 --batch-size 2
--learning-rate .001 --max-grad-norm .1 --seed 41 --device mps` unchanged.

Use three new output directories: a full run, a partial run with
`--stop-after 8`, then a fresh process with the partial run's
`--resume-from-checkpoint .../checkpoint-8`, retained
`--geometry-checkpoint-id` and retained `--expected-dataset-id`.
Invoke Python with `-I -B` and set `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=2`.

Compare the final export and checkpoint geometry bytes; compare optimizer,
scheduler and RNG payloads recursively as in the public Trainer tests. Only
load HF pickle companions from your own trusted runs. The geometry ID alone
does not authenticate companions, the base, tokenizer or full training recipe.

## Publication Boundary

`validation.json` contains source/binary/log hashes, sanitized run metrics and
verification outcomes. Original result JSON (including generated text), input
copies, learned geometry and complete checkpoints remain local. The original
repository samples were not restored into the deletion-heavy worktree.
Independent-review scope and corrections are recorded in `REVIEW.md`.
