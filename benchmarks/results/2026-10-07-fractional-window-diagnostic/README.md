# Saved-Model Full-Normalization Window Diagnostic

New post-training intervention on the three final `history_angle_full`
checkpoints from the [completed angular study](../2026-10-05-fractional-angle-study/README.md).
No training, optimizer steps, checkpoint selection or original artifact edits.

## Numeric Outcomes

All **456 original full-history block scores and six means reproduced exactly**
under the candidate native runtime before any partial-window score was taken.
Every mode retains identical learned parameter bits and full K=32 normalization.
All 120 Pride and 32 Alice blocks are used for each seed (41, 43, 47).
Lower CE is better. Deltas are relative to each seed's unchanged full operator.

| Mode | Pride mean CE | Delta | Alice mean CE | Delta |
| --- | ---: | ---: | ---: | ---: |
| full | 3.99291008 | 0 | 3.96419896 | 0 |
| retained_short [1,3) | 4.02233458 | +0.02942450 | 3.98150282 | +0.01730386 |
| retained_tail [3,32) | 4.02785380 | +0.03494372 | 3.98535837 | +0.02115941 |
| local_only [1,1) | 4.06361376 | +0.07070368 | 4.00574806 | +0.04154910 |

Every intervention increases mean CE for every seed on both books. In this
trained model, removing the long tail hurts even when retained short taps are
not renormalized. Removing short taps also hurts; neither part alone retains
the full benefit. Local-only keeps the learned local gate and is not an
unadapted-base baseline. Nonlinear losses need not add across filter parts.

This is **dependence of the saved trained models**, not evidence that training
only short taps at the same normalization would lose. Endpoints are reused,
exploratory, and potentially present in pretraining. Three old seeds are not
three new independent experiments. No significance, general LLM superiority,
longer-context safety, speed, GPU or browser claim is made. Prior failed
ordinary/GL-short final-parameter tolerance checks remain failed and unchanged.

## Identity And Reproduction

- Diagnostic source: `1ff85d7fd181dcbf9e23936556e53a74c870aaa7`.
- Candidate native build: `16379238c73f6890a4f754dccd4ad381436e047b`, SHA-256 `18f4c1c2bb4b76fa0e8beeca59f7fbfa0dffbc85efd72b96c45c119a63befa58`.
- Original native SHA-256: `7146628d3977d9862151707bdd235029f13e3dab20846142edb30790e4b088b1`.
- Original study: `4cb29fe3a588f7d46df6df775f781141c847b8dac6a8351450b03452c7ae917a`.

`plan.json` freezes all four modes, every seed/block, both native identities and
the zero-tolerance all-seed replay gate. `results.json` retains every numeric
block loss/delta and parameter/recipe receipt. `summary.json` contains direct
seed averages (not a fitted estimate or confidence interval).
`validation.json` records successful process termination, regression results,
private log hashes and before/after verification of 76 study, eight client,
71 original-runtime and 71 candidate-runtime files.

The exact executed diagnostic and verifier helper are archived here. Follow
[the execution recipe](../../../docs/fractional_window_diagnostic.md) with this
archived code, the original frozen client, both runtime manifests and a new
output directory. The [original publication](../2026-10-05-fractional-angle-study/)
and [candidate runtime manifest](../2026-10-05-fractional-history-window/frozen-runtime-sha256.json)
bind prerequisites. Original runtime files are verified separately, not
silently loaded as the candidate. CPU float32, Python 3.12.6, Torch 2.12.1,
Transformers 4.57.6, two threads, batch two, complete 128-token contexts,
first-MLP insertion and disabled KV cache match the original scoring recipe.

Numeric/hash consistency can be checked without model imports:

```sh
python -B -m pytest -q -p no:cacheprovider \
  bindings/st-py/tests/test_fractional_window_results.py
```

Those public checks do not independently rerun private checkpoints. Weights,
corpus text, native binaries and raw private logs remain local. Nothing from
the sealed original study was replaced. The next training comparison should
hold K=32 normalization fixed and compare full versus retained-short learning;
it should not train a dormant integer-order tail with zero feature gates.
