# What changed at the trained GPT-2 attention output?

**At the nine frozen endpoints, fixed-radius WaveGate's forward residual was
16.3-16.7% of the matched affine arm's, but its correction to the arriving
loss gradient was 78.8-79.7% of affine's.** Both quantities were smaller for
fixed Wave on all 96 paired training documents in each order. These are the
same 96 documents across orders, not 288 independent samples. No weights were
updated and no held-out documents were opened for this diagnostic.

| Training order | Residual/input: affine | Residual/input: fixed Wave | Fixed/affine residual | Fixed/affine adjoint correction | Mean / min Wave radial gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 109 | 0.00109849 | 0.00017964 | 16.35% | 79.30% | 0.8124 / 0.4086 |
| 113 | 0.00109814 | 0.00017901 | 16.30% | 78.83% | 0.8137 / 0.4089 |
| 127 | 0.00109856 | 0.00018317 | 16.67% | 79.73% | 0.8049 / 0.4078 |

The adjoint correction is `||dx - dy|| / ||dy||`, **not** the ratio of full
gradient norms. It was 0.0002393-0.0002395 for affine and 0.0001887-0.0001909
for fixed Wave. Including the identity path, `||dx|| / ||dy||` was 1.000018
to 1.000024 in all nine snapshots. The forward inputs had very similar RMS
(about 21.75345), but are not asserted to be the same hidden vectors.

The learned-radius snapshots had relative residuals 0.00017903-0.00018319;
they did not recover the affine amplitude. The within-arm document RMS of the
gate gradient L2 was 0.06542 for affine versus 0.04781-0.04819 for fixed Wave.
This is not the norm of a gradient taken over the whole corpus.

## Where Did Contraction Occur?

For fixed Wave, no gate parameter crossed the saturation threshold. Exactly
**one pre-saturation affine coordinate per document** crossed it in each order:
96 out of 1,689,600 coordinates, no nonfinite values. The mean absolute slope
at that stage was 0.99994. The subsequent radial projection had mean
tangential gain 0.9295-0.9328 and mean radial gain 0.8049-0.8137. The latter
fell to about 0.408 on the most contracted row. These are local Jacobian gains,
not full-network gradient loss.

This rules out *ubiquitous coordinate clipping at these endpoints*, not a
high-leverage rare coordinate. The present receipts do not contain its
position, channel or energy. They also cannot assign the forward gap to
projection rather than the different learned gates/biases. The next discriminating
control is the **same captured input, same gate/bias**, comparing the plain
affine branch, saturation only, projection only and both. No parameter retuning
or access to the old held-out split is needed for that question.

## Evidence

The [frozen method](README.md) ran once for 182.46 process CPU seconds, 193.54 s
elapsed. The installed instrument wheel matched all 207 package files. Nine
old endpoint files and the one training file were checked against their pins.
The held-out scores from the previous quality comparison are not an input.

The [result](receipts/result.json) and the compressed scalar journal in
[the evidence archive](receipts/archive.json) preserve all 864 measurements.
To recheck every owner and window, recompute both gains, and verify the summary
without a model: `python3 -I -B -S verify.py receipts --archive`. To check the
original train file and nine checkpoint hashes too, add `--originals "$PREVIOUS_PRIVATE_RUN"`.
No checkpoint, raw activation, gradient array, text, token ID or model weight
is included in the export.

Native code: `03bb3eda3`. Frozen measurement: `89778f768`. The ordinary path
and its opt-in observer gave identical native SGD checkpoint bytes on synthetic
tests. Across the old and instrumented wheels, three synthetic arms x three
updates had the same hash over loss, full gradients and new adapter IDs:
`9f55410dea64eabf29e4c1bcf81bc3f6e4f1cb2f9adf4234c8cf2db6620c4ac1`.
These checks establish compatibility on the fixtures, not a new quality result.
