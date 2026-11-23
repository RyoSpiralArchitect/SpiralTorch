# Reading real attention did not beat a uniform value read on subject agreement

The added gate really used the decoder's Q/K-weighted value. On **all 192**
distinct formal base/training-prefix combinations it differed from the uniform
read of the same V. That fact did not predict a transfer benefit. Of the three
carriers, **only the uniform read** produced jointly correct singular and plural
verbs on fresh lexical combinations; all of those successes were on base 563.

| Transfer teacher | Gate source | Both / 144 pairs | Individual / 288 | Mean weaker signed margin |
| --- | --- | ---: | ---: | ---: |
| Head | Zero gate (shared) | 0 | 144 | -0.22371 |
| Head | Actual attention | 0 | 142 | -0.15151 |
| Head | Uniform V | 24 | 167 | -0.13383 |
| Head | Fixed slot 2 | 0 | 132 | -0.21949 |
| Nearby | Zero gate (shared) | 0 | 144 | -0.21810 |
| Nearby | Actual attention | 0 | 144 | -0.15899 |
| Nearby | Uniform V | 14 | 158 | -0.14392 |
| Nearby | Fixed slot 2 | 0 | 136 | -0.19218 |

Each denominator repeats **16** transfer pairs across three bases and three
orders. The 24 head successes were **eight distinct pairs on one base**, four
per template, each reproduced in all three orders. The 14 nearby successes
covered six pairs on that base. No zero-margin ties. None of the three bases
passed the preregistered head mechanism, continuous-margin or task gate for
actual attention against the origin and both controls. The nearby mechanism
gate passed only on base 563; its task gate passed on none. Do not promote a
default or claim a pretrained language-model effect from this.

**How different was the read?** Across the six base/template groups the mean
attention mass on the head marker ranged from 0.1173 to 0.1314 (uniform 0.125).
Mean total variation from uniform ranged from 0.0214 to 0.0441. Nevertheless
the median relative L2 difference between the actual value vector and its
uniform counterpart ranged from 0.0680 to 0.5672 across these groups. The
two trained endpoints chose different verbs on 43/288 head predictions and
30/288 nearby predictions. Attention mass is not an attribution of a decision.

A **post hoc**, training-only readback of the stored subgradient choices helps
locate the divergence. For the head teacher, actual and uniform selected the
same weaker lane throughout all 64 updates on bases 557 and 569 in all orders.
On base 563 they first diverged at updates 48, 14 and 38 for orders 17, 31
and 73, and differed on 8, 9 and 8 updates respectively. The sources and
continuous gradients can already differ before a discrete lane changes.
This observation does not separate the effect of learning a different gate
from that of reading a different V at serving time.

## Native contract

`spiraltorch.attention_value_gate.v1` is an opt-in two-input residual on a
decoder MLP: `M + strength * (gate * source + bias)`. The original attention
and MLP are still present. The recipe owns `features`, `strength`, `max_values`
and a route, one of `{"kind":"attention"}`, `{"kind":"uniform"}` or
`{"kind":"fixed","position":2}`. The fixed route reads the current slot until
the declared absolute position is visible. The attention route reuses the
already computed head contexts. Its derivative flows into Q, K **and** V;
controls scatter only into V. Full, independent-batch and chunked KV inference
share this definition. The adapter adds no persistent KV rows. A one-input
geometry call cannot silently substitute a value source.

Python calls the same Rust model as ordinary forward and paired training:
`model.last_mlp_value_read_signal(ids, 7, 8)` exposes factual Q/K/V and the
unit-margin gate derivative at a single active final MLP. An explicit standalone
two-input calculation uses `runtime.forward_value_read_sequence(site, mlp, source, shape)`;
`batch.vjp_input(dy)` and `batch.vjp_value_source(dy)` refer to distinct inputs.
This experiment used a numerical frozen LoRA and updated only three gate weights.

## Evidence

[Registration](PROTOCOL.md) and [all 108 transfer cells](orders.tsv).
Formal: 3,456 updates, 576 origin captures and 576 calibration prefixes,
108 first/last physical checks, 4,032 training and 2,016 transfer forwards.
Maximum relative delivered-RMS error 2.91e-6. Rehearsal: 24 updates, separate
from formal. No repeated scientific worker or test-set tuning. The initial
Python *binding preflight* stopped for disk space before launching Python;
its receipt remains in the archive separately from the successful execution.
No pretrained GPT-2 calls, GPU, network, or public release.

With the extracted archive, replay the sealed updates, readouts and attribution
without importing an ML framework:

```sh
python3 -ISB audit.py "$CAPTURE/evidence/formal" "$CAPTURE/parents/base/formal"
python3 -ISB readback.py "$CAPTURE/evidence/formal" "$FORMAL_RESULT_SHA256" "$EMPTY/decisions.json"
python3 -ISB lane_switches.py "$CAPTURE/evidence/formal" "$FORMAL_RESULT_SHA256" "$EMPTY/lanes.json"
```

| Artifact | SHA-256 |
| --- | --- |
| Formal result | `161b6a3b980d9f0a916f3f99968d397e0b004eb58dc8e12b47208b0f9001c87a` |
| Rehearsal result | `60927d5428f4b7626c486b5007270e5e5814e2d9fe5a427e9f05ae7c9c178a0a` |
| All transfer cells | `47b6fd86cd2382159f826c031ec689159a7637250033dfdb2bd8e50871b10a4b` |
| Matched source/decision readback | `f785bad0bf023fdbf5bf0b75d6217c6677642c8633b583a3d020178c2d948e4e` |
| Post hoc training lane readback | `77e4ec6064de353b6e2058ab74d9b27e3432a0773fcb81e9a0d83d51321a92ea` |
| CPU development wheel | `c702b7158c085c6907214c87e0dc052349f1b54ebc1c40f9d9c754ad6829c857` |
| Verified private archive (167 files) | `8a0d747f394342a5af7a525aca1076a5b29f6cf377fb535f636ec1e5076a9e19` |
