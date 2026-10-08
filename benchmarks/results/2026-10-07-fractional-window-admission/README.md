# Fixed-Normalization Window Learning Admission

Training-only connection check for the [new paired study](../../../docs/fractional_window_study.md).
This receipt does not establish primary-training completion or endpoint quality.

Both Rust-backed arms use K=32 normalization. The retained-short arm then
keeps [1,3); the full arm retains all strictly-past taps. Parameters, angular
chart, amplitude, initialization, Adam and batch schedule are matched. The
retained-short arm is not normalized as a K=3 filter.

On the actual frozen local GPT-2 first-MLP tensor [2,128,768], both adapters
start as identity with the same 1538 parameter bits and matching initial
filters. First CE is identical. After two seed-41 training-only updates,
angle and log-gain gradients are nonzero in both arms. A separate next update
from each restored checkpoint reproduces the update and named Adam state
exactly. Base parameters stay frozen; no heldout loss was computed.

| Arm | Step-two angle gradient | Step-two log-gain gradient |
| --- | ---: | ---: |
| retained short | 0.00004293878010 | -0.00004641809937 |
| full | 0.00003183410445 | -0.00004641809937 |

The different angle derivative is expected: zero tail values at order two
still have a nonzero derivative. It is not a short/full parity failure or
evidence of better final quality.

- Four auxiliary training updates and four continuation-only updates; process exit 0.
- 841 Python regressions pass, no skips, 18 existing Torch JIT warnings; 22 new study tests.
- Source revision: `cbf68cea3cf82678fc63171765394f6b3472820c`.
- Reused native build: `16379238c73f6890a4f754dccd4ad381436e047b`.
- Native SHA-256: `18f4c1c2bb4b76fa0e8beeca59f7fbfa0dffbc85efd72b96c45c119a63befa58`.
- Original 76 study files, eight client files, two sets of 71 runtime files and two saved-model diagnostic files verified unchanged before admission.

`admission.json` retains all scalar training receipts, source/asset hashes
and invariants. `client-sha256.json` binds the nine-file frozen new client;
`validation.json` binds the preflight, regression outcomes and private log
hashes. Follow the study documentation with `--coordinate window`, the fixed
config and original completed angular study. Python 3.12.6, Torch 2.12.1,
Transformers 4.57.6, CPU float32 and two threads match the earlier setup.

No weights, corpus text, native package or raw private logs are published.
The original failed cross-arm final-parameter parity checks remain unchanged.
The planned six 512-update runs are a separate experiment, not completed
evidence in this admission record.
