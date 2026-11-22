# The pretrained decoder already wins the restricted agreement task on the training prefixes

These are the original, **pre-update** logits in the 48 margin-capture
receipts, not new model evaluations. Each host saw the same 24 sentences
from six lexical quartets. For each quartet we verified a subject-token-only
switch at each of the two fixed attractor numbers, giving 12 matched pairs.

| Initial host | Subject response S | Constant plural bias B | Attractor response A | S x A interaction | Correctly switched pairs | Two-choice wins | Vocabulary greedy: either verb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Affine | 2.99028 | -0.02703 | 0.78723 | -0.23048 | 12/12 | 24/24 | 0/24 |
| WaveGate R=1 | 2.98989 | -0.02209 | 0.78758 | -0.23184 | 12/12 | 24/24 | 0/24 |

Units are logits; positive B favors the plural verb. Correctly switched
means both strict decisions are correct, not only that the plural logit gap
increased. No ties occurred in the binary comparisons. Holding the attractor
fixed and changing only the subject increased the plural-minus-singular gap
by 4.665 to 7.537 logits for affine, and 4.660 to 7.534 for WaveGate.
The greedy IDs were read from the capture, not inferred from these margins.
Choosing a third token is not itself a grammatical error: these prefixes
admit other continuations. The two stored logits do not identify either
verb's full-vocabulary probability.

The frozen primary metric already cancels a constant verb bias because of
the balanced design. A positive training effect on it would measure
stronger subject selectivity, **not automatically more correct choices or
more probability for the verbs**. The novelty and scores of the separate
16-prompt cohort are not inferred from these training rows.

## Probability split on the completed artificial rehearsal

On the four previously stored artificial stages, the paired readout
reproduced the original margin contrasts, NLL changes and no-advance
decision. In each arm, conditional verb choice improved but the pair
lost probability to the rest of the vocabulary. Mean changes from each
arm's own initial state (NLL units; negative is better):

| Toy arm | Within-pair choice | Pair coverage | Full-vocabulary gold NLL |
| --- | ---: | ---: | ---: |
| R=1 | -0.000005638 | +0.000009052 | +0.000003414 |
| R=4 | -0.000005659 | +0.000009193 | +0.000003534 |

These tiny changes illustrate a metric distinction on synthetic weights;
they do not predict an English-language effect. No real or artificial model
was loaded, trained or scored for this readout. It does not choose a radius
or change the training protocol, primary threshold or existing wheel.

See [the definitions and commands](AGREEMENT_READOUT.md). Nine readout
controls and the original three design controls passed. The captured
training-row digest is
`386def6c7b639879be6aa53476c88e286880bbe4475814c0a00560825c4ec6f6`;
the sealed miniature result is
`53c7bc15999a9eddea11b5329f957c9b9f24d54b11f7491d50f181f36a562c58`.
The verified train-only readout SHA-256 is
`6f24508e436dd279e764c15708915b7d200f6899d557dba1b6172e1e1c57af6b`;
the miniature readout is
`a04b89f9e1e4d2536507ab6827aeb9a2e8a2182354240d7ee377fcd7ae522612`.
