# WaveGate: distance to the full gradient is not direction

On 24 previously used subject/verb agreement training prefixes from GPT-2
small, **projection alone** was closer to the full gate+bias derivative in
Euclidean distance, while **saturation alone** was closer in angle. This held
on 24/24 prefixes on each of the two factual hosts. These are the same 24
prefixes twice, not 48 independent examples.

## Original, bounded measurement

The [preregistered rule](README.md) jointly required attenuation versus affine
and smaller distance to the full derivative. Both factual hosts passed the
projection rule on 24/24 rows, above the required 18/24 per host. Neither
passed the symmetric saturation rule on any row. Each training prefix had one
forward/backward on each fixed host, then four **local** pullbacks used the
same captured site input and arriving gradient within that call. The two hosts
had bit-identical site inputs; their arriving gradients need not be identical.
The seed was the gold verb's logit minus the alternative verb's logit at the
last attention output, not CE. No LoRA was attached.

| Factual host | Projection only | Saturation only | Full WaveGate |
| --- | ---: | ---: | ---: |
| Affine | 0.01556 | 0.91214 | 0.16280 |
| WaveGate | 0.01556 | 0.92061 | 0.16382 |

Values are median gate+bias gradient lengths divided by the local affine
gradient length on the same row. The full map is not just the projection:
its typical gradient is substantially larger than projection alone. The
maximum relative L2 errors against the available original scaled binary-CE
anchor gradients were 1.48e-6 and 1.08e-6 (rounded up), below the 1e-3 limit.

The original synthetic checks passed (four Rust tests; three Python binding
checks including finite differences). The real CPU worker completed 48/48
forward/backwards, 192 local maps, in 7.10 s of a 300 s cap, on its first
attempt. No weight update, held-out evaluation, generation or GPU.

## Descriptive read-back: separate size from direction

Using only the saved f32 derivative vectors, we computed cosine to the full
map, separately from Euclidean distance, with no fitted thresholds. This
analysis was **not** the preregistered criterion. Undefined cosines are not
counted as wins; there were none here.

| Factual host | Block | Median cosine (projection, full) | Median cosine (saturation, full) |
| --- | --- | ---: | ---: |
| Affine | gate | 0.74278 | 0.99659 |
| Affine | bias | 0.99953 | 0.99961 |
| Affine | gate+bias | 0.87081 | 0.99864 |
| WaveGate | gate | 0.75387 | 0.99660 |
| WaveGate | bias | 0.99954 | 0.99961 |
| WaveGate | gate+bias | 0.87806 | 0.99865 |

Within **each block and each host** the projection-only vector was nearer
in Euclidean distance on 24/24 rows, yet the saturation-only vector had a
higher cosine on 24/24. The angular gap is much clearer for gate than bias.
Changing a single learning-rate multiplier could rescale a gradient but
cannot fix a direction difference. This observation motivates measuring
direction and length separately; it does not establish the causal effect of
either component on learning, held-out accuracy or generation.

To reproduce without weights, a Python binding or an ML framework, pass
the original training-only GPT-2 archive subdirectory to:

```sh
python3 -ISB test_block_readback.py
python3 -ISB block_readback.py "$GPT2_CAPTURE"
```

The reader verifies four SHA-256 pins, reproduces the original result bytes
from the journal, then computes the block table. Original raw journal SHA-256:
`386def6c7b639879be6aa53476c88e286880bbe4475814c0a00560825c4ec6f6`.
Original result: `aaf38a614d3189511926fdb961a9dd637afb2074f12ab2ab73458591c1762238`.
New descriptive report: `57d0766a4ba38e8b290ac632757fc2441d65eab6efe16a78b728d6c22b623989`.
No new real-model call was made for this read-back.

## Can the two corrections simply be added?

No on this capture. For each fixed input and arriving gradient, the additive
prediction is `g_saturation + g_projection - g_affine`: subtract the shared
baseline once. Its error vector relative to `g_full` is the factorial
interaction, not a new gradient through a different decoder. We report its
length divided by the full gradient length, and separately the cosine of
the additive prediction to the full gradient. These are descriptive
measurements from the original 48 calls, not a new success criterion.

| Factual host | Block | Median relative error of addition | Median cosine (addition, full) |
| --- | --- | ---: | ---: |
| Affine | gate | 3.958 | 0.0223 |
| WaveGate | gate | 3.811 | 0.0236 |
| Affine | bias | 0.919 | 0.8329 |
| WaveGate | bias | 0.919 | 0.8419 |
| Affine | gate+bias | 2.728 | 0.0346 |
| WaveGate | gate+bias | 2.584 | 0.0367 |

For the combined block, the additive error exceeded the full vector's own
length on **24/24 rows on each host** (minimum ratios 1.333 and 1.331).
Gate and bias should not be conflated: on gate the additive prediction
was nearly orthogonal to the actual pullback, whereas bias retained much
more directional agreement. Size-only attenuation is not a substitute for
the composed VJP. This is not evidence that composition improves training,
accuracy or generalization. The two hosts reuse the same 24 training
prefixes, and this contrast does not uniquely attribute the interaction to
one operator.

```sh
python3 -ISB test_interaction.py
python3 -ISB interaction.py "$GPT2_CAPTURE" "$OUT/interaction.json"
```

The reader checks all four original pins and reproduces the primary result
before computing the interaction. It does not import a model or read the
held-out data. The new JSON retains all per-row values as well as the ranges,
medians and undefined counts. SHA-256:
`74f2a56f11cb873cb12bca264f770a74d9c21c9aae32397650ffbbfbb1eb3abc`.
