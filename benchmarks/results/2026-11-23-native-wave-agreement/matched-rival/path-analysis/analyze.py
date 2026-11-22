"""Explain an archived update path without loading a model or reading its logits."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE.parent / "run.py"))
D = R["D"]
need, digest, encoded = D["need"], D["digest"], D["encoded"]
BINS = ("same_same", "same_flip", "flip_same", "flip_flip")
FLIPS = ("subject_at_sg", "subject_at_pl", "attractor_at_sg", "attractor_at_pl")


def block_dot(x, y, width):
    return tuple(math.fsum(x[i]*y[i] for i in range(j, j+width)) for j in (0, width))


def parts(gate, bias):
    return {"gate": gate, "bias": bias, "total": gate+bias}


def scaled(terms, factor):
    return parts(*(factor*math.fsum(t[i] for t in terms) for i in (0, 1)))


def check_equal(x, y, message):
    need(math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-12), message)


def energy_components(labels, steps, width, diag, energy):
    # Each signed column has squared norm n; balanced cells make them orthogonal.
    n = len(steps)
    folded = list(zip(*steps))
    def component(signs):
        vec = [math.fsum(v*s for v, s in zip(col, signs)) for col in folded]
        return scaled([block_dot(vec, vec, width)], 1/(n*energy))
    head = [1 if h else -1 for _, h, _ in labels]
    near = [1 if a else -1 for _, _, a in labels]
    out = {"common": component([1]*n), "subject": component(head), "attractor": component(near),
           "interaction": component([h*a for h, a in zip(head, near)])}
    cells = defaultdict(list)
    for (_, h, a), vec in zip(labels, steps):
        cells[h, a].append(vec)
    residual = []
    for group in cells.values():
        mean = [math.fsum(col)/len(group) for col in zip(*group)]
        for vec in group:
            diff = D["delta"](vec, mean)
            residual.append(block_dot(diff, diff, width))
    out["within_cell"] = scaled(residual, 1/energy)
    budget = scaled(diag, 1/energy)
    for k in budget:
        check_equal(math.fsum(v[k] for v in out.values()), budget[k], "factor energy does not close")
    return out


def describe(labels, steps, endpoint):
    n, width = len(steps), len(endpoint)//2
    need(n and width and len(labels)==n and len(endpoint)==2*width and
         all(len(u)==2*width and all(map(math.isfinite, u)) for u in steps), "invalid step layout")
    need(all(isinstance(h, bool) and isinstance(a, bool) for _, h, a in labels), "invalid number label")
    pairs = {p for p, _, _ in labels}
    need(len(set(labels))==n==4*len(pairs), "expected one of every number combination per lexical pair")
    for end, coords in zip(endpoint, zip(*steps)):
        check_equal(end, math.fsum(coords), "trajectory does not telescope to the stored endpoint")
    diag = [block_dot(u, u, width) for u in steps]
    norms = [math.sqrt(g+b) for g, b in diag]
    need(all(v>0 and math.isfinite(v) for v in norms), "undefined angle for a zero/nonfinite step")
    path = math.fsum(norms)
    energy = math.fsum(g+b for g, b in diag)
    bins = {k: [] for k in BINS}
    flips = {k: [] for k in FLIPS}
    for j, ((p, h, a), v) in enumerate(zip(labels, steps)):
        for i in range(j):
            pi, hi, ai = labels[i]
            g, b = block_dot(steps[i], v, width)
            cosine = (g+b)/(norms[i]*norms[j])
            key = ("flip" if h!=hi else "same")+"_"+("flip" if a!=ai else "same")
            bins[key].append((g, b, cosine))
            if p==pi and (h!=hi) != (a!=ai):
                key = ("subject_at_"+("pl" if a else "sg") if h!=hi else
                       "attractor_at_"+("pl" if h else "sg"))
                flips[key].append(cosine)
    need(all(len(x)==len(pairs) for x in flips.values()), "incomplete matched lexical flips")
    terms = {"diagonal": scaled(diag, 1/path**2)}
    means = {}
    for name, sample in bins.items():
        terms[name] = scaled(sample, 2/path**2)
        means[name] = dict(n=len(sample), negative=sum(c<0 for _, _, c in sample),
                           mean_cosine=math.fsum(c for _, _, c in sample)/len(sample) if sample else None)
    need(sum(x["n"] for x in means.values())==n*(n-1)//2, "lost a step pair")
    end_ratio = scaled([block_dot(endpoint, endpoint, width)], 1/path**2)
    for k in end_ratio:
        check_equal(math.fsum(t[k] for t in terms.values()), end_ratio[k], "Gram partition does not close")
    factors = energy_components(labels, steps, width, diag, energy)
    check_equal(factors["common"]["total"]*n*energy/path**2, end_ratio["total"], "common/endpoint mismatch")
    return dict(n=n, path=path, endpoint=D["length"](endpoint), retention_squared=end_ratio,
                gram=terms, pairs=means, energy_fraction=factors,
                matched={k:dict(n=len(v), mean_cosine=math.fsum(v)/len(v)) for k,v in flips.items()})


def difference(wave, affine):
    def sub(x, y):
        return {k: x[k]-y[k] for k in ("gate", "bias", "total")}
    gap = sub(affine["retention_squared"], wave["retention_squared"])
    gram = {k: sub(affine["gram"][k], wave["gram"][k]) for k in wave["gram"]}
    for k in gap:
        check_equal(sum(v[k] for v in gram.values()), gap[k], "affine-wave gap does not close")
    opp = {k: gram["flip_same"][k]+gram["flip_flip"][k] for k in gap}
    return dict(gap=gap, gram=gram, opposite_subject=opp,
                opposite_majority=gap["total"]>0 and opp["total"]>gap["total"]/2,
                stronger_subject_reversal=all(wave["matched"][k]["mean_cosine"]<0 and
                    wave["matched"][k]["mean_cosine"]<affine["matched"][k]["mean_cosine"]
                    for k in ("subject_at_sg", "subject_at_pl")))


def input_role(relative, kind):
    names = relative.parts
    if names==("SHA256SUMS",):
        return "manifest"
    if names==(kind, "plan.json"):
        return "plan"
    if names[:2]==(kind, "attempt"):
        if len(names)==3:
            if names[2] in ("identity.json", "receipt.json", "sealed.json"):
                return "metadata"
            if relative.suffix in (".g32", ".jsonl"):
                return "gradient_tape" if relative.suffix==".g32" else "journal"
        elif len(names)==4 and names[2]=="states" and relative.suffix==".json":
            return "state"
    raise ValueError("not a training input: "+str(relative))


def observe_inputs(archive, kind):
    seen = {}
    def observe(event, args):
        if event=="import" and args[0].split(".")[0] in {"spiraltorch","numpy","torch","transformers","safetensors"}:
            raise ValueError("no model runtime in a path analysis")
        if event!="open" or not isinstance(args[0], (str,bytes)):
            return
        location = Path(os.path.abspath(os.fsdecode(args[0])))
        if archive in location.parents:
            need(not args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND),
                 "analysis must not change its input")
            rel = location.relative_to(archive)
            seen[str(rel)] = input_role(rel, kind)
    sys.addaudithook(observe)
    return seen


def analyze(archive, manifest_pin, kind):
    need(kind in ("gpt2", "rehearsal"), "unknown archive kind")
    sums = (archive/"SHA256SUMS").read_bytes()
    need(digest(sums)==manifest_pin, "untrusted source manifest")
    entries = [s.split("  ", 1) for s in sums.decode("ascii").splitlines()]
    files = {name: pin for pin, name in entries}
    need(len(files)==len(entries), "duplicate manifest entry")
    directory = archive/kind
    root = directory/"attempt"
    def read(path):
        pin = files[str(path.relative_to(archive))]
        raw = path.read_bytes()
        need(digest(raw)==pin, "input changed: "+path.name)
        return json.loads(raw)
    spec = read(directory/"plan.json")
    rec = read(root/"receipt.json")
    identity, sealed = read(root/"identity.json"), read(root/"sealed.json")
    need(spec["kind"]==kind and spec["sources"]==D["sources"]() and
         not (root/"failure.json").exists() and rec["plan"]==files[kind+"/plan.json"] and
         rec["result_sha256"]==files[kind+"/attempt/worker-result.json"] and
         0<rec["seconds"]<=spec["seconds"] and rec["min_free"]>=R["FLOOR"] and
         identity["backend"]=="rust_f32_cpu_prepacked" and spec["extension_pin"] in (None, identity["extension"]),
         "archive is not the pinned successful native run")
    audits = R["audit_training"](spec, root, sealed, identity["base"])
    @lru_cache(maxsize=16)
    def state(pin, kind):
        path = R["blob"](root, pin)
        need(files[str(path.relative_to(archive))]==pin[7:], "state manifest mismatch")
        gate, bias = R["OLD"]["check_geometry"](read(path), spec, identity["base"], kind)
        return D["f32"](gate+bias)
    case_by_key = {r["key"]:r for r in spec["train"]}
    routes, comparisons = {}, {}
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            geom = arm.split("_")[0]
            stage = "%s-%s"%(seed, arm)
            audit = audits[stage]
            name = "%s/attempt/%s" % (kind, stage)
            need(audit["journal_sha256"]==files[name+".jsonl"] and audit["tape_sha256"]==files[name+".g32"],
                 "untrusted journal or tape")
            journal = (root/(stage+".jsonl")).read_bytes()
            need(digest(journal)==audit["journal_sha256"], "journal changed after validation")
            initial = state(sealed["initial-"+geom], geom)
            previous = initial
            labels, steps = [], []
            for line in journal.splitlines():
                r = json.loads(line)
                case = case_by_key[r["key"]]
                h = case["plural"]
                a = h ^ case["conflict"]
                need(case["key"]=="%s.%d%d"%(case["pair"], h, a), "case/number mismatch")
                after = state(r["step"]["result_id"], geom)
                steps.append(D["delta"](after, previous))
                labels.append((case["pair"], h, a))
                previous = after
            last = state(sealed[stage], geom)
            v = describe(labels, steps, D["delta"](last, initial))
            check_equal(v["path"], audit["path_l2"], "wrong archived path")
            check_equal(v["endpoint"], audit["endpoint_l2"], "wrong archived endpoint")
            routes[stage] = v
        comparisons[str(seed)] = {o:difference(routes["%s-wave_%s"%(seed,o)], routes["%s-affine_%s"%(seed,o)])
                                  for o in ("full","rival")}
    gates = {k:all(c["rival"][k] for c in comparisons.values())
             for k in ("opposite_majority","stronger_subject_reversal")}
    return dict(schema="spiraltorch.matched_update_cancellation.v1", kind=kind, manifest=manifest_pin,
                source_sha256=digest(Path(__file__).read_bytes()), plan=rec["plan"], receipt=files[kind+"/attempt/receipt.json"],
                archived_updates=sum(r["n"] for r in routes.values()), routes=routes, comparisons=comparisons,
                rival_gates=gates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--manifest", required=True)
    p.add_argument("--kind", choices=("gpt2","rehearsal"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root, out = a.archive.resolve(), a.output.resolve()
    need(out!=root and root not in out.parents and not out.exists(), "output must be new and outside archive")
    seen = observe_inputs(root, a.kind)
    result = analyze(root, a.manifest, a.kind)
    result["archive_files_read"] = dict(Counter(seen.values()))
    R["write"](out, encoded(result))
    print(json.dumps({k:result[k] for k in ("kind","archived_updates","rival_gates")}))


if __name__=="__main__":
    main()
