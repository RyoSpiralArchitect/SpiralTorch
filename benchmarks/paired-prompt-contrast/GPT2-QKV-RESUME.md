# Pretrained GPT-2 QKV Update And Resume

Status: one bounded CPU qualification passed on its first model execution.
This validates the recorded update/save/resume route, not language quality,
geometric superiority, GPU execution or release readiness.

## Scope

The [file-training interface](../../docs/python/native-qkv-run.md) loaded a
locally saved GPT-2 small (12 layers, width 768, vocabulary 50,257), with an
existing frozen LoRA and WaveGate source. Only the layer-11 QKV weight matrix
was trained. The model, data and native runtime were pinned before execution.
There was no Torch, NumPy, Transformers, Python-safetensors or Accelerate
import, installation, model download or fallback.

Two previously used training documents, lengths 26 and 27, supplied two
selected answer targets each. Their order and unshifted masks were unchanged.
They are not fresh holdout. The fixed configuration was:

```json
{
  "layer": 11,
  "optimizer": {
    "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
    "max_gradient_norm": 1.0
  },
  "training": {
    "context_length": 64, "learning_rate": 0.0001,
    "weight_decay": 0.01, "label_smoothing": 0.0,
    "epochs": 1, "batch_windows": 1, "max_rows": 1
  }
}
```

All paths used the same saved native extension, CPU execution, one Rayon
thread, f32 readout and `save_every=1`. Execution order was continuous two
updates, fresh one update followed by save, then restore and one remaining
update: **four actual updates total**. No additional inference, generation,
quality scoring, retry or GPU run was performed.

## Results

| Check | Observed result |
| --- | --- |
| Admitted corpus plan | Two active windows and two planned updates |
| Initial and first-update checkpoints | Byte-identical across paths |
| Final corpus checkpoint | Continuous and resumed files byte-identical |
| Exported QKV weights and Adam state | Continuous and resumed files byte-identical |
| Training reports | Continuous sequence equals partial plus resumed sequences |
| Weight-only payload | Finite and changed at the first update and endpoint |
| Q/K/V derivative norms | Nonzero for each of the two training examples |
| Original inputs, source and runtime | Before/after hash inventories unchanged |
| Foreign ML imports | Zero attempted imports and zero loaded modules |

The outer coordinator completed in 10.228 seconds against a 600-second
deadline. This includes setup, hashing and artifact checks; it is not a
throughput benchmark. An independent disk-only `cmp` and SHA-256 calculation
confirmed both final checkpoint/export pairs after the run.

The two logged objective values, 3.9925477505 and 2.6678206921, belong to
**different training examples**. They are not a before/after improvement
measurement. The second clipped norm was 1.00000002034, a small f32 rounding
overshoot, not evidence of a strict real-arithmetic norm projection.

## Evidence Identity

Source commit: `37e7ee77f4b5e51e899262412b500247e7936f6c`.
The exact runtime, recipe, data selection and failure gates were frozen before
execution. A capacity refusal before launch was preserved; no model ran in
that refusal. Storage was subsequently reclaimed only from separately approved
unused build caches, without changing the experiment's admission limits.

| Artifact | SHA-256 |
| --- | --- |
| Frozen protocol | `9558b9775ef3acf2588388a16cbd77c0c4a5d25bd9794427414f503df1b30fb8` |
| Complete run receipt | `9309bc172b15506bdfc89d1357af485a23f0a62e9a8686f4b721684b327bf0f5` |
| Final corpus checkpoint, both paths | `199a1d36516444d0cd759cc8a120fb32b9cb4bb8048d25d253fc1f9a62c329de` |
| Final optimizer export, both paths | `c8e07d12ee4b94b8a7dc5b67682394860fb674c315f27fbb4bb9ca057c989760` |
| Initial weight-only payload | `33b4b4fe45b4bbb28633b8036490781ab1418f7b7cbb2150d95dbcfd8b620bce` |
| Final weight-only payload | `17af5c784a47cdb7c7e0b93f64bfac806bac2b93c970b420c88f38d73fc91786` |
| Saved native extension | `f80f9abc5449b80483578ef1d2786f2f8b8cc9d2dd152d727b8631e9685f4ccf` |

The original checkpoints, input corpus, runtime binary, logs and full local
provenance remain privately retained. Hashes identify retained artifacts but
do not authenticate them or make the experiment independently reproducible
from this repository alone. The published interface and synthetic tests are
reproducible without that private study; this exact GPT-2 run is not.

## Interpretation

This is real-model evidence that the Rust-owned QKV learning path can cross
the Python file interface, save, and resume exactly under the specified CPU
conditions with frozen LoRA and geometry. It complements the synthetic
[corpus-session](QKV-CORPUS-SESSION.md) and [file-run](QKV-FILE-RUN.md) checks.
It does not establish long-run stability, other architectures, geometry's
causal benefit or general capability. Existing negative competence findings
in the earlier synthetic experiments remain unchanged.
