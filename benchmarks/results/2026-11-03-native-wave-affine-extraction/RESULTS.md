# Native GPT-2 small: fresh extraction split

**The ordinary trainable affine control had lower answer-weighted NLL in all
three orders. None of the differences cleared the preregistered margin, and
WaveGate produced no additional correct answers.** This is not an equivalence
result. The 48 test records are shared across orders.

| Training order | Fixed Wave minus affine NLL | Answers lower / tied / higher for fixed Wave | Learned minus fixed NLL | Learned log radius |
| --- | ---: | ---: | ---: | ---: |
| 109 | +0.0007360140 | 7 / 0 / 41 | -0.0000008480 | +0.0001036022 |
| 113 | +0.0007442452 | 6 / 0 / 42 | +0.0000015779 | +0.0001047711 |
| 127 | +0.0007251597 | 6 / 0 / 42 | -0.0000011451 | +0.0001019030 |

For an X-minus-Y contrast, a negative difference favors X. All primary
differences were inside `+/-log(1.001)` nats. The fixed model's token-weighted
differences also favored affine (+0.000613, +0.000621, +0.000605 nats).
Learning the radius had a much smaller difference and the sign changed with order.

The initial mean answer NLL was 3.886909 for all arms. All nine fits lowered it:
affine ended at 3.879244 to 3.879470 and fixed Wave at 3.879988 to 3.880195.
All endpoints had **30/74** supplied-prefix top-1 tokens and **4/48**
all-tokens-correct answers. No individual answer's hit count changed from the
start in any fit. This does not measure free-running generation. In a descriptive
read-back, 41 of 48 records favored affine in every order, 5 favored fixed Wave
in every order and 2 changed direction. Do not count these as 144 independent answers.

The initial LoRA update ID and loss matched exactly across each cohort.
The largest reported first-update gate difference was below 9e-12, and the bias
differences were zero. The saved fixed radii stayed at 1. Descriptively, affine's
final gate L2 norm was 0.03223 to 0.03224 versus 0.01758 to 0.01790 for fixed
Wave. A parameter norm is not an activation norm or evidence of saturation.
This motivates measuring the actual residual and pullback on training inputs
in a separate study, rather than retuning this held-out split.

## Evidence

- [Frozen design](README.md), source commit `e647fea464db280a6a083739b754d323e274d203`.
- [Input manifest](receipts/manifest.json), SHA-256 `b23955a9d88a0d09254708e874ded78c25b9a25994f5fa6e42dbc3a1be4d2777`. 96 train / 48 held out, 148 / 74 scored tokens. All 144 records from the earlier pilot were excluded. All 207 installed package files matched the pinned wheel.
- [Result and receipt pins](receipts/result.json), SHA-256 `5e84b71d96f6c478f22db01e6f581a6b7c9106e39bbee42b4aa810dca6833dd3`. One attempt, 864 joint SGD arm-updates, about 120.3 process CPU seconds including preparation and 126.8 seconds elapsed. No intermediary held-out probes, retries, GPU, external model runtime, or model publication.
- [File-only verifier](verify.py): `python3 -I -B -S verify.py receipts --describe` checks the archived order, all 864 update links, nine checkpoint boundaries, four scoring endpoints and six final contrasts. With the private original directory and `--originals` it also checks the actual nine checkpoint files, both inner adapter hashes, training masks and token IDs. No weights or encoded corpus are included in the receipt export.
