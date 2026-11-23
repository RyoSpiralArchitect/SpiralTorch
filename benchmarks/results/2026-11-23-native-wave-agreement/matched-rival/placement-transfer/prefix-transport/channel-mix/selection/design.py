"""Freeze two-modifier prompts, training identities and the finite route bank. No model import."""
import argparse
from itertools import product
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[7]
old = SimpleNamespace(**runpy.run_path(str(HERE.parent / "worker.py")))
c, s, d, need = old.c, old.s, old.d, old.need
MAPS = c.MAPS
TEACHERS = ("head", "attractor")
PINS = {
    "formal": ("7e98f8143acddefe18bc6f3111d534631592cb97aa8761f33b8f65c9d61e8a00",
               "a353ec8146c663e8bc3d62b1770d0daff1196d012e6c08a197b3c07a13d18f36"),
    "rehearsal": ("421fb21966f609fb557cebe5b3bb5d0732215bd25f0a4e844fc2f68e65587abe",
                  "352d8adc4dbe7b0e62d652e15349c6b7f915c88af4a008368cabde460e5e68c3"),
}
EXT = "14db09cd803b37b73437cd86e9822d5c4c88929e9a8481d598d5d2dd441fde05"

def read(path, pin):
    return json.loads(c.pinned(path, pin))

def sha(path):
    return d.digest(path.read_bytes())

def fresh(kind):
    need(kind in PINS, "unknown split")
    rows = []
    for h in (range(3, 7) if kind == "formal" else [3]):
        a, left, right = ((3 + (h-3+2)%4, 3 + (h-3+1)%4, 3 + (h-3+3)%4)
                          if kind == "formal" else (3, 5, 6))
        for i,j in product((1,2), repeat=2):
            for form, ids in (("hh", [h,left,right,i,9,a,j,10]), ("aa",[h,i,9,a,left,right,j,10])):
                frame = "%d%d%d%d" % (h,left,right,a)
                rows.append(dict(key="%s.%s.%d%d" % (form,frame,i,j), template=form, frame=frame,
                                 head=i-1, attractor=j-1, prompt=ids))
    return rows

def sources():
    rel = ["crates/st-nn/src/causal_lm.rs", "crates/st-nn/src/causal_lm/sum.rs",
           "crates/st-nn/src/causal_lm/route_calibration.rs", "tools/native_matched_route_selector.rs"]
    paths = [REPO/p for p in rel] + [HERE/n for n in ("PROTOCOL.md","design.py","run.py","audit.py")]
    return {str(p.relative_to(REPO)):sha(p) for p in paths}

def prepare(dest, bank, residual, selector, kind):
    seen = set()
    for k, (plan, _) in PINS.items():
        obj, sp = c.data(bank/k, plan)
        known = read(residual/k/"spec.json", c.PREVIOUS[k][0])
        for r in sp["train"] + known["test"] + read(bank/k/"evaluation.json", obj["evaluation"]):
            seen.add(tuple(r["prompt"]))
    a,b = ({tuple(r["prompt"]) for r in fresh(k)} for k in ("formal","rehearsal"))
    need(len(a)==32 and len(b)==8 and not a&b and not (a|b)&seen, "prompt collision")
    plan, trained = PINS[kind]
    _, sp = c.data(bank/kind, plan)
    m = read(bank/kind/"trained.json", trained)
    need(m["extension"]==EXT and m["inputs"]==plan and tuple(sp["maps"])==MAPS and tuple(d.TEACHERS)==TEACHERS,
         "bank input provenance")
    root = dest/kind
    root.mkdir(mode=0o700)
    evaluation = d.save(root/"transfer.json", fresh(kind))
    return d.save(root/"inputs.json", dict(kind=kind, bank=plan, trained=trained, transfer=evaluation,
                                          extension=EXT, selector=sha(selector), sources=sources()))

def load_inputs(root, bank, pin):
    cfg = read(root/"inputs.json",pin)
    need((cfg["bank"],cfg["trained"])==PINS[cfg["kind"]] and cfg["extension"]==EXT and
         cfg["sources"]==sources(), "study or source identity drift")
    _,sp = c.data(bank,cfg["bank"])
    return cfg, sp, read(bank/"trained.json",cfg["trained"])["files"]

def cells(sp):
    for b in sp["bases"]:
        for m in MAPS:
            yield b,m,s.key(b,m)
            for t,o in product(TEACHERS,sp["orders"]):
                yield b,m,s.key(b,m,t,o)

def quartets(rows):
    groups = {}
    keys = set()
    for r in rows:
        key,frame,h,a = (r[x] for x in ("key","frame","head","attractor"))
        need(h in (0,1) and a in (0,1) and key not in keys,"invalid label or duplicate key")
        keys.add(key)
        g = groups.setdefault(frame,{})
        need((h,a) not in g,"duplicate assignment in frame")
        g[h,a] = key
    order = list(product((0,1),repeat=2))
    need(groups and all(set(g)==set(order) for g in groups.values()),"incomplete training frame")
    return [tuple(groups[f][k] for k in order) for f in sorted(groups)]

def parent_artifact(bank,manifest,name):
    p = "weights/"+name+".json"
    return c.pinned(bank/"training"/p,manifest[p])

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for n in ("dest","bank","residual","selector"):
        cli.add_argument(n,type=Path)
    cli.add_argument("kind",choices=PINS)
    print(prepare(**vars(cli.parse_args())))
