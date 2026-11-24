"""Frozen pairs and source identities for the two-lane native learning study."""
import argparse
from itertools import product
import json
from pathlib import Path
import random
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
old = SimpleNamespace(**runpy.run_path(str(HERE.parent / "channel-mix/worker.py")))
c, v, k, d, s, need = old.c, old.v, old.k, old.d, old.s, old.need
ARMS = ("binary", "rank", "anchored_rank")
TEACHERS = ("head", "attractor")
BANK = {"formal": "7e98f8143acddefe18bc6f3111d534631592cb97aa8761f33b8f65c9d61e8a00",
        "rehearsal": "421fb21966f609fb557cebe5b3bb5d0732215bd25f0a4e844fc2f68e65587abe"}
SELECT = {"formal": "c177b60f7f173faf0ef4b517199ebabd06afa5de983d33ef956eea50ebba7c2a",
          "rehearsal": "75797dd56a035f3e6ffc743e86facf938a9f2ee3749b967ac0bb364755dfd965"}
read = lambda p, sha: json.loads(c.pinned(p, sha))


def rows(kind, transfer=False):
    need(kind in BANK, "unknown split")
    out = []
    for h in (range(3, 7) if kind == "formal" else (3,)):
        cyc = lambda shift: 3 + (h - 3 + shift) % 4
        nearby = [cyc(2)] if transfer else [cyc(1), cyc(-1)]
        nearby = nearby if kind == "formal" else [3]
        context = cyc(1 if transfer else 2) if kind == "formal" else (6 if transfer else 5)
        for a, i, j in product(nearby, (1, 2), (1, 2)):
            for t, p in (("head_first_modifier", [0, context, h, i, 9, a, j, 10]),
                         ("nearby_first_modifier", [0, h, i, 9, context, a, j, 10])):
                frame = "%d%d%d" % (h, context, a)
                out.append(dict(key="%s.%s.%d%d" % (t, frame, i, j), template=t,
                                frame=frame, head=i-1, attractor=j-1, prompt=p))
    return out


def pairs(cases, teacher, seed=None):
    need(teacher in TEACHERS, "unknown teacher")
    nuisance = TEACHERS[1-TEACHERS.index(teacher)]
    groups = {}
    for r in cases:
        group = groups.setdefault((r["template"], r["frame"], r[nuisance]), {})
        need(r[teacher] in (0, 1) and r[teacher] not in group, "duplicate supervised number")
        group[r[teacher]] = r
    ps = []
    for key in sorted(groups):
        g = groups[key]
        need(set(g) == {0, 1}, "incomplete number pair")
        a, b = g[0]["prompt"], g[1]["prompt"]
        changes = [(x,y) for x,y in zip(a,b) if x != y]
        need(len(a) == len(b) == 8 and changes == [(1,2)], "not a matched number intervention")
        ps.append((g[0],g[1]))
    if seed is not None:
        random.Random(seed).shuffle(ps)
    return ps


def sources():
    files = [HERE / n for n in ("PROTOCOL.md","design.py","checks.py","run.py","audit.py")]
    files += [HERE.parent / n for n in ("channel-mix/worker.py","channel-mix/design.py",
              "residual-metric/study.py","prefix-signal/study.py","residual-metric/verify_exact.py")]
    files += [REPO / n for n in ("crates/st-nn/src/causal_lm.rs",
              "crates/st-nn/src/causal_lm/context_preference.rs",
              "crates/st-nn/src/causal_lm/paired_token_gap.rs",
              "crates/st-nn/src/causal_lm/objective.rs",
              "crates/st-nn/src/causal_lm/learning.rs",
              "crates/st-nn/src/causal_lm/adaptation.rs",
              "bindings/st-py/src/causal_lm.rs",
              "bindings/st-py/src/causal_lm/paired_gap.rs",
              "bindings/st-py/src/causal_lm/session.rs",
              "crates/st-nn/src/causal_lm/geometry_cohort/prefix_metric.rs",
              "crates/st-nn/src/geometry_runtime/prefix_transport.rs",
              "bindings/st-py/tests/test_native_paired_token_gap.py")]
    return {p.relative_to(REPO).as_posix():d.digest(p.read_bytes()) for p in files}


def prepare(root, bank, selector, residual):
    historical = set()
    inherited = {}
    for kind in BANK:
        info, sp = c.data(bank/kind, BANK[kind])
        last = read(selector/kind/"inputs.json", SELECT[kind])
        need(last["bank"] == BANK[kind], "wrong selector parent")
        six = read(residual/kind/"spec.json", c.PREVIOUS[kind][0])
        need(six["train"] == sp["train"], "six-token training provenance")
        for case in six["train"] + six["test"] + read(bank/kind/"evaluation.json", info["evaluation"]) + read(selector/kind/"transfer.json", last["transfer"]):
            historical.add(tuple(case["prompt"]))
        inherited[kind] = {key:sp[key] for key in ("weights","parent_manifest","bases","orders","epochs")}
    fresh = set()
    for kind in BANK:
        for transfer, count in ((False, 64),(True, 32)):
            cases = rows(kind, transfer)
            ps = {tuple(r["prompt"]) for r in cases}
            need(len(ps) == len(cases) == (count if kind=="formal" else 8)
                 and not ps.intersection(fresh | historical), "prompt collision")
            for t in TEACHERS:
                pairs(cases,t)
            fresh.update(ps)
    pins = {}
    for kind in BANK:
        out = root/kind
        out.mkdir(mode=0o700)
        sp = inherited[kind]
        need((sp["bases"], sp["orders"], sp["epochs"]) ==
             (([557,563,569],[17,31,73],2) if kind=="formal" else ([571],[5],1)), "unfrozen design")
        a = d.save(out/"training.json",dict(kind=kind, train=rows(kind), **sp))
        b = d.save(out/"transfer.json",rows(kind,True))
        pins[kind] = d.save(out/"inputs.json",dict(training=a,transfer=b,kind=kind,bank=BANK[kind],
                                                  selector=SELECT[kind],sources=sources()))
    return pins


def data(root, pin):
    inp = read(root/"inputs.json", pin)
    sp = read(root/"training.json", inp["training"])
    need(inp["sources"] == sources() and inp["bank"] == BANK[sp["kind"]] and inp["selector"] == SELECT[sp["kind"]]
         and sp["kind"] == inp["kind"] and sp["train"] == rows(sp["kind"]), "training manifest drift")
    return inp, sp


def cells(sp,b):
    return product((b,), ARMS, TEACHERS, sp["orders"])


def name(b,arm=None,teacher=None,order=None):
    return str(b)+"-origin" if arm is None else "%d-%s-%s-%d" % (b,arm,teacher,order)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("root","bank","selector","residual"):
        ap.add_argument(n,type=Path)
    print(json.dumps(prepare(**vars(ap.parse_args())),sort_keys=True))
