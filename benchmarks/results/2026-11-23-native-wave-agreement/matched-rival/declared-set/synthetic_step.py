"""Finite native geometry steps on in-memory artificial heads; no model files or corpora."""
import argparse
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import runpy
import sys


def need(ok, message):
    if not ok:
        raise ValueError(message)


class NativeOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "safetensors", "accelerate"}:
            raise ImportError("external ML runtime: "+fullname)


def run():
    root = Path(__file__).resolve().parents[5]
    sys.meta_path.insert(0, NativeOnly())
    fixture = runpy.run_path(str(root/"bindings/st-py/tests/test_native_causal_lm.py"))["NativeCausalLmTests"]()
    ids, shape, kw = [2,0], [1,2], dict(contrasts=[(1,0,[1])])
    def read(m):
        scores = m.declared_set_nll(ids, shape, **kw)
        gold, rival, *outside = map(math.exp, m.forward(ids[:-1], [1,1], last_only=True).values())
        pair, total = gold+rival, gold+rival+math.fsum(outside)
        truth = (math.log(pair/gold), math.log(total/pair), math.log(total/gold))
        for key, v in zip(("within_nll","coverage_nll","full_nll"), truth):
            need(abs(scores[key]-v) < 1e-12, "loss differs from actual logits: "+key)
        return dict(scores, set_probability=pair/total, gold_probability=gold/total)
    rows = []
    cases = [
        ("full_guard_misses_leak", [(0.,2.)], 1./65536, "repair"),
        ("exact_opposition", [(0.,0.)], 1./65536, "infeasible"),
        ("two_outsiders_small_step", [(-4.,0.),(6.,0.)], 1./65536, "accept"),
        ("two_outsiders_large_step", [(-4.,0.),(6.,0.)], 1./128, "finite_reject"),
    ]
    for name, head, length, expected in cases:
        m = fixture._uniform_set_geometry(*head)
        within, coverage, _ = m.geometry_declared_set_gradients(ids, shape, **kw)
        initial = read(m)
        raw, _ = m.step_geometry_gradient_at_distance(within, length)
        record = dict(case=name, status=expected, length=length, before=initial, unguarded=read(raw))
        try:
            result, step = m.step_declared_set_geometry_at_distance(ids, shape, target_l2=length, **kw)
        except ValueError as error:
            need(expected=="infeasible" and "feasible interval" in str(error), str(error))
            record.update(error=str(error), returned=read(m))
        else:
            need(expected!="infeasible" and step["accepted"]==(expected in ("repair","accept")), "unexpected decision")
            proposal, guard = m.guard_geometry_step(raw, [[within],[coverage]])
            proposed = read(proposal)
            need(all(step["proposed"][key] == proposed[key] for key in step["proposed"]), "wrong measured candidate")
            need((guard["mix"]>0.)==(expected=="repair") and all(x<0. for x in guard["applied_loss_dots"]),
                 "unexpected first-order direction")
            need(result.geometry_artifact_id==(proposal if step["accepted"] else m).geometry_artifact_id,
                 "wrong selected snapshot")
            record.update(proposed=proposed, returned=read(result), receipt=step)
        need(initial==read(m), "original model changed")
        rows.append(record)
    return dict(schema="spiraltorch.declared_set_artificial_steps.v1", rows=rows, pretrained_calls=0,
                heldout_scores=0, fixture="deterministic, zero hidden input, final LayerNorm and trainable bias")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = (json.dumps(run(), sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()
    with args.output.open("xb") as f:
        f.write(raw)
    print(hashlib.sha256(raw).hexdigest(), args.output.name)
