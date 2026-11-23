# Adjacent changes did not improve paired verb agreement

**Keep the level carrier as the default.** The opt-in native change carrier
works in the forward pass, input/gate/bias adjoints, residual calibration,
saved adapters and bounded cached inference. It does **not** make the two
supervised numbers independently steerable on this synthetic transfer task.

The [frozen protocol](PROTOCOL.md) compared reading the MLP activations with
reading their successive differences. Nothing is subtracted across documents;
the first change is zero. Both routes use the same rotation, frozen host/LoRA,
three learned gates, weakest-lane loss, paired schedule and delivered residual
RMS. Each zero-gate origin produced bit-identical full-vocabulary scores
across carriers on both partitions. The change route retains one extra f32 per
channel in the inference cache, never the full history. Old v1/v2 recipes and
their attribution schema are unchanged; the opt-in recipe is
`spiraltorch.prefix_transport_adapter.v3`.

| Transfer task | Carrier | Both verbs correct / 144 pairs | Individual / 288 | Mean weaker signed margin |
| --- | --- | ---: | ---: | ---: |
| Head number | Initial, shared | 0 | 144 | -0.23064 |
| Head number | Level | 0 | 138 | -0.17475 |
| Head number | Change | 0 | 138 | -0.21345 |
| Nearby number | Initial, shared | 0 | 144 | -0.22873 |
| Nearby number | Level | 9 | 153 | -0.18442 |
| Nearby number | Change | 0 | 143 | -0.20759 |

The denominator is 16 transfer pairs per base x three bases x three orders,
not 144 independent linguistic examples. The nine nearby successes were three
distinct pairs from base 563, each reproduced in all three orders. No endpoint
in the table tied at a zero weaker margin. Neither the head mechanism gate nor
the task gate passed on any of the three bases; the nearby control passed
neither gate either. Mean head margin was worse for changes than levels on
each base, despite improving in aggregate relative to the common origin.

At initialization the signed head gradients opposed one another in the
calibrated metric for **83/96** distinct base/pair combinations with levels,
**90/96** with changes. For the nearby teacher this was 91/96 versus 96/96.
All capacity ratios were defined and no ideal one-step direction exceeded the
physical cap. This is local directional evidence, not a causal explanation for
every endpoint. Removing a time-constant input is not the same as isolating
the subject. These results do not establish a GPT-2, English, or GPU effect.

Formal: 36 endpoints, 2,304 native two-lane updates, 384 origin captures, 384
calibration prefixes, 72 first/last physical checks, 2,688 train and 1,344
transfer full-logit forwards. Maximum relative delivered-RMS error 2.84e-6.
The 16-update rehearsal is separate. Both workers completed on the first
attempt. All [72 transfer cells](orders.tsv) are retained, not just the best
order. Replay without importing a model (CAPTURE is the extracted archive):

```sh
python3 -ISB audit.py "$CAPTURE/evidence/formal" "$CAPTURE/parents/base/formal"
python3 -ISB readback.py "$CAPTURE/evidence/formal" \
  9cb213e1135bf5a8b8ff7c834ec466099392124e640012481eaba934e4a5b117
```

| Object | SHA-256 |
| --- | --- |
| Formal result | `9cb213e1135bf5a8b8ff7c834ec466099392124e640012481eaba934e4a5b117` |
| Rehearsal result | `89587b9d7516edd7a83d7d78fd7e99d55a3ad472f0c12bf7930f5e8406ea248f` |
| Complete table | `17f524666caa90df77eb6203d4b8eb4fa777e85ed46b7185809cd991593a8d0d` |
| Development CPU wheel | `c5ea0bc509e31db66798f2285352584b13dfbb05f8301413e5f751fac0eb5448` |
| Private archive (133 verified files) | `c043026593a6125abeb1bf9dc93a6167ee0404713e0ab599bc6d44ba36807669` |
