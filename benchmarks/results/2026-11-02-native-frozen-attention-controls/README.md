# Does training the inserted attention geometry help?

This is a matched follow-up to the [native GPT-2 attention study](../2026-11-02-native-joint-attention-transfer/README.md).
Plain LoRA starts at the pretrained model, but attaching the calibrated geometry
changes that function before training. The missing control is **LoRA trained with
the same geometry present but frozen**. Subtracting the initial loss does not
substitute for that trajectory.

The [protocol](plan.json) freezes six additional fits (two geometries, three order
seeds) and the analysis before starting any of them. The checkpoint, source
wheel, initial LoRA and geometry, training windows/order, SGD, 512-update budget
and fresh-process cut at 256 are the same as for each retained joint arm.
Only the native optimizer's group selection changes, from `joint` to `lora`.
No optimizer or gradient routing is reimplemented in Python. The selected
gradients still propagate through the installed, fixed geometry.

The primary comparison is joint minus fixed WaveGate on *Mansfield Park*;
negative favors learning the geometry. The affine comparison, static-branch
difference and difference in differences are reported separately on each book.
All 18 sections were **already inspected** in the original study. This tests a
missing matched control, not new independent generalization, a radius-only
effect, or generation quality. No new text is decoded.

## Original outcome: stopped before scoring

The first attempt hit `ENOSPC` while writing the second half of the
seed-139 affine step log. [File-only inspection](interrupted_run.json) verifies
four terminal checkpoints (affine and WaveGate at seeds 131 and 137) and
2,352 complete step records in total. For seed 139 the last sealed checkpoint
is update **256**, not 304: the next 48 reports have no corresponding persisted
weights. The 139 WaveGate fit never started. The supervisor could not persist
its own first-error JSON; the original worker stderr and source-file hashes
are retained. No checkpoint was regenerated and no worker was retried.

All six terminals were required before evaluation. **There is no held-out
score or joint-versus-fixed conclusion for this attempt.** To authenticate the
retained failure rather than launch another study:

```sh
python3.12 -I -S -B audit_interrupted.py --out "$PRIMARY" --old "$OLD" --data "$DATA" --model "$MODEL" --wheel "$WHEEL" | cmp - interrupted_run.json
```

## Reproduce

Use the parent study's retained training/evaluation inputs and exact scientific
wheel (not the public release). Set `PRIMARY` to the completed joint run;
`OLD`, `DATA`, `MODEL`, `PY` and `WHEEL` refer to its original saved inputs and
isolated installed Python. The controller admits one attempt per physical primary
run directory and plan hash; a failed worker is retained, not retried.

```sh
python3.12 -I -S -B -m unittest discover -s . -p 'test_*.py'
python3.12 -I -S -B control_study.py check --out "$PRIMARY" --old "$OLD" --data "$DATA" --model "$MODEL" --wheel "$WHEEL"
python3.12 -I -S -B control_study.py run --out "$PRIMARY" --old "$OLD" --data "$DATA" --model "$MODEL" --wheel "$WHEEL" --python "$PY"
python3.12 -I -S -B control_study.py audit --out "$PRIMARY" --old "$OLD" --data "$DATA" --model "$MODEL" --wheel "$WHEEL"
```

The private audit checks every update's two geometry owners, initial and terminal
parameter hashes, the original first loss and first LoRA update, both halves'
ordered data identity, the checkpoint bytes, and all 12 terminal/anchor owners
in every native section report. Each model scores 4,536 shifted targets per
book. All six new terminal states are sealed before held-out scoring.
