# Self-Supervised Recipes

These notebooks outline lightweight training flows for the new contrastive and masked modeling
objectives. They can be executed with any Jupyter runtime that has access to the `spiraltorch`
Python extension module and the pure-Python utilities under `bindings/python/spiral`.

## Contents

- `contrastive_recipe.ipynb` – demonstrates preparing batches, augmentations, and invoking the
  InfoNCE objective from Python for transfer pretraining.
- `masked_modeling_recipe.ipynb` – shows how to mask tokens, compute reconstruction loss, and reuse
  the same augmentation helpers for modality-agnostic setups.

Each notebook follows a deterministic seed schedule to ensure reproducible results when using the
Rust regression tests introduced alongside the objectives.
