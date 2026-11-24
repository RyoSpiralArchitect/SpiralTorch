# Finite agreement-margin changes before a prefix's own update

On archived GPT-2 small training trajectories, an initial gate/bias margin
gradient predicted the direction of later movement on **69/69** noninitial
affine visits and **59/69** WaveGate R=1 visits (eight opposite, two observed
ties). Both did better than forecasting no change in each of three orders.
Only affine beat the matched lexical control in every order. No binary verb
preference flipped; the gold verb already led on all 24 initial prefixes.

The [rule](PROTOCOL.md) was committed at `0bb56392a87c` before this archive
analysis. The earlier independent [four-map diagnostic](../parameter-vjp/README.md)
ran 48 real forward/backwards and 192 local pullbacks: for both factual hosts
all 24 prefixes met its local projection rule, zero met its saturation rule.
That is a statement about the magnitude and proximity of *gradients*. The
new analysis uses the recorded **finite forward margins** from actual prior
updates on *other* sentences, without running the model again.

## Margin forecasts

RMSE in millionths of a logit margin. Each cell covers 23 noninitial visits;
the six initial checks were excluded. Lexical = mean forecast from the other
five noun families with identical subject and attractor number. No predictor
was fitted to outcomes. Each host uses its own starting margin and gradients.

| Host | Order | No change | Own gradient | Lexical control |
| --- | ---: | ---: | ---: | ---: |
| Affine | 17 | 741.7 | **65.8** | 125.2 |
| Affine | 31 | 733.9 | **73.6** | 126.3 |
| Affine | 73 | 705.6 | **82.0** | 104.2 |
| Wave R=1 | 17 | 76.0 | **29.8** | 35.2 |
| Wave R=1 | 31 | 60.4 | **30.0** | 32.4 |
| Wave R=1 | 73 | 76.3 | 33.0 | **31.9** |

With equal weight per prefix rather than per visit, mean measured changes
were +549.3 millionths for affine and +19.0 for Wave. After exactly one prior
update, the forecasts got all three signs right for each host; RMSE was 34.0
and 10.8 millionths, respectively. These are three paths through the same
24 prefixes, not 138 independent prompts. Smaller absolute error for Wave
does not mean a better forecast: its observed movements were much smaller too.

## Where the probability went

On the same incoming prefixes, rival-only training gave a **higher** full
vocabulary NLL than ordinary CE training on 69/69 affine and 68/69 Wave
visits. Equal-prefix mean penalties were +0.001740 and +0.000235 nats.
This compares full NLL between **two trained histories** at the same prefix;
it is not a change from initialization.

Three prefixes did have measured initial full NLL because each occurred first
in one order. We can compare their other two visits with initialization:

| Host (6 later visits) | Full NLL change, rival training | Within verb pair | Pair coverage cost | Full NLL change, ordinary training |
| --- | ---: | ---: | ---: | ---: |
| Affine | +696.7 | -33.9 | +730.6 | -2254.2 |
| Wave R=1 | -35.6 | -2.2 | -33.4 | -247.0 |

Values are means in millionths of a nat. Coverage cost is
`-log(P(gold)+P(rival))`; full NLL equals within-pair NLL plus this cost,
up to the stored full loss's binary32 rounding. On the previously identified
[`driver.11` prefix](../radius-transfer/objective-metric/RESULTS.md) after
15 or 11 **other** updates, respectively, affine rival training raised full
NLL by +870 or +514 millionths; Wave rival training lowered it by -42 or -50.
Ordinary CE lowered full NLL more in both cases and hosts. This is not the
effect of updating on `driver.11` itself, nor evidence that a particular
local component caused the finite difference.

The practical lead is to track **within-pair and pair-coverage costs together**
when testing a less attenuating geometry, rather than optimize only the
binary margin. No new training, held-out scoring, R=4 trajectory, GPU use or
quality improvement is claimed.

## Verification

Seven unit tests, the four-prefix rehearsal, then one real-archive read.
Both reports replayed byte-for-byte from a 356-member training-input-only
bundle. Both original manifests are included as hashes and names, not as
permission to read excluded files. Neither decoder weights, held-out logits,
the original mixed training/evaluation plan nor the mixed final results are
needed. The archive check also recomputes observed margins, incoming state
links, loss differences and per-order RMSE directly from the journals.

| Private artifact | SHA-256 |
| --- | --- |
| GPT-2 analysis JSON | `ac75d8f1f7eeb9bbdc7789865116b2580d17f2dab1f2b94f13b1b6420ef17e20` |
| Artificial analysis JSON | `247e79ffc971ad0eb9db4ec500e728d2a00204557141ca72ed7b511e95bc6df5` |
| Verified training-input ZIP | `9e5a9be22ee63f9b082de3382259153106a009a445f72deb17a16f6c8e5e90b1` |

To reproduce from private training-only inputs, set `CAPTURE` and `MATCHED`
to the recovery tree's `capture/` and `matched/` directories, and `OUT`
to an empty directory. From here, no ML runtime is imported:

```sh
python3 -ISB test_prequential.py
python3 -ISB study.py rehearsal "$CAPTURE" "$MATCHED" "$OUT/toy.json"
python3 -ISB study.py gpt2 "$CAPTURE" "$MATCHED" "$OUT/real.json"
python3 -ISB archive_check.py "$CAPTURE" "$MATCHED" "$OUT/toy.json" "$OUT/real.json" "$OUT/inputs.zip"
```

The report hashes are deterministic at these source versions. ZIP container
timestamps can change a newly created archive's hash; verification is by the
complete member set, every member digest, and exact replayed report bytes.
