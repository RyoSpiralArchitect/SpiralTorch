"""Read only sealed training-gradient tapes. No decoder, tokenizer, forward or backward."""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import struct
import sys


HERE = Path(__file__).resolve().parent
S = runpy.run_path(str(HERE.parent / "sweep.py"))
need, sha, enc, write = (S[k] for k in ("need", "sha", "encode", "write"))
SWEEP = "8edc3b8339e61dc1f0e8a1e30558af2c1b257f5d1f8e70fc3e3a1d3c76296e74"
DISTANCE = 1 / 256
TOLERANCE = 2.5e-7
METRICS = ("t11", "t14", "t41", "t44", "delta", "donor", "receiver", "c11", "c44", "cosine_change")


def checked_cases(cases):
    need(len({c["key"] for c in cases}) == len(cases) and cases, "duplicate or absent training rows")
    groups = sorted({c["pair"] for c in cases})
    verbs = {}
    for group in groups:
        subset = [c for c in cases if c["pair"] == group]
        need(len(subset) == 4 and {(c["plural"], c["conflict"]) for c in subset}
             == {(p, f) for p in (False, True) for f in (False, True)}, "incomplete lexical quartet")
        for c in subset:
            need(type(c["plural"]) is bool and type(c["conflict"]) is bool
                 and c["gold"] != c["rival"]
                 and verbs.setdefault(c["plural"], (c["gold"], c["rival"])) ==
                 (c["gold"], c["rival"]), "inconsistent supervision")
    need(verbs[False] == verbs[True][::-1], "verbs are not mutual rivals")
    return groups


def archive(capture, sweep, tag):
    kind = {"real": "gpt2", "toy": "rehearsal"}[tag]
    p, parent, pins = S["inputs"](capture, kind)
    need("evaluation" not in p and checked_cases(p["train"]), "unexpected data scope")
    manifest = (sweep / "SHA256SUMS").read_bytes()
    need(sha(manifest) == SWEEP, "unrecognized local gradient archive")
    checks = {name: h for h, name in (l.split("  ", 1) for l in manifest.decode().splitlines())}
    def read(name):
        raw = (sweep / name).read_bytes()
        need(sha(raw) == checks[name], "changed archive member: " + name)
        return raw
    raw = read(tag + "/local.jsonl")
    summary = json.loads(read(tag + "/summary.json"))
    spec = json.loads(read(tag + ".json"))
    need(spec["kind"] == kind and spec["inputs"] == pins and spec["source_manifest"] == S["ARCHIVE"]
         and spec["parent_adapter"] == p["geometry"]["both"]["pin"] and spec["sources"] ==
         {k:sha((HERE.parent / k).read_bytes()) for k in ("sweep.py", "README.md", "../bounded.py")}
         and summary["protocol"] == checks[tag + ".json"] and summary["local_sha256"] == sha(raw)
         and summary["selected"] == (4 if tag == "real" else None), "wrong capture lineage")
    anchors = {}
    for radius in (1, 4):
        state = read(tag + "/radius-%d.json" % radius)
        S["check_variant"](json.loads(S["raw_record"](p, "both")), json.loads(state), radius)
        anchors[radius] = "sha256:" + sha(state)
    need(anchors[1] == spec["parent_adapter"] and
         (tag == "toy" or anchors[4] == summary["selected_artifact"]), "wrong radius artifacts")
    lines = raw.splitlines()
    need(len(lines) == len(parent) * len(S["RADII"]), "incomplete local tape")
    indices = [c["key"] for c in p["train"]]
    vectors = {}
    for i, line in enumerate(lines):
        r = S["RADII"][i // len(parent)]
        if r not in anchors:
            continue
        obj, original = json.loads(line), parent[i % len(parent)]
        need((obj["key"],obj["host"],obj["radius"],obj["artifact"]) ==
             (original["key"],original["host"],r,anchors[r]), "wrong row, host or radius")
        if r == 1:
            S["assert_anchor"](obj, original)
        vectors[r, obj["host"], obj["key"]] = [
            v for k in S["BLOCKS"] for v in S["vector"](S["unpack"](obj[k]), p["shape"][1])]
    return p["train"], {h: [vectors[r,h,k] for r in (1,4) for k in indices] for h in S["HOSTS"]}


def reference(rows):
    # Products of two binary32 inputs are exact in binary64.
    return [[math.fsum(x*y for x,y in zip(a,b)) for b in rows] for a in rows]


def norms(matrix, size):
    need(len(matrix) == size and all(len(row) == size for row in matrix), "wrong Gram shape")
    diag = [matrix[i][i] for i in range(size)]
    need(all(math.isfinite(v) and v > 0 for v in diag), "undefined zero or nonfinite direction")
    return [math.sqrt(v) for v in diag]


def check_gram(actual, exact):
    n = len(exact)
    norms(actual,n)
    scale = norms(exact,n)
    largest = 0.
    for i in range(n):
        for j in range(n):
            v = actual[i][j]
            need(math.isfinite(v) and v == actual[j][i], "nonfinite or asymmetric Gram")
            err = abs(v - exact[i][j]) / (scale[i]*scale[j])
            need(err <= TOLERANCE, "wide dot disagrees with independent sum")
            largest = max(largest, err)
    return largest


def pair(j, i, size, g, n):
    t, c = {}, {}
    for r, a in ((1,j), (4,size+j)):
        for s, b in ((1,i), (4,size+i)):
            k = str(r) + str(s)
            t[k] = DISTANCE * g[a][b] / n[b]
            c[k] = g[a][b] / (n[a]*n[b])
    d = t["44"] - t["11"]
    donor = ((t["14"] - t["11"]) + (t["44"] - t["41"])) / 2
    receiver = ((t["41"] - t["11"]) + (t["44"] - t["14"])) / 2
    need(abs(d - donor - receiver) <= 1e-13 * max(abs(v) for v in t.values()), "bad decomposition")
    return dict(t11=t["11"],t14=t["14"],t41=t["41"],t44=t["44"],delta=d,donor=donor,receiver=receiver,
                c11=c["11"],c44=c["44"],cosine_change=c["44"]-c["11"])


def average(rows):
    need(rows, "cannot average zero eligible pairs")
    return dict(n=len(rows), **{k:math.fsum(v[k] for v in rows)/len(rows) for k in METRICS})


def analyze(cases, matrix):
    groups = checked_cases(cases)
    size = len(cases)
    n = norms(matrix, 2*size)
    if len(groups) == 1:
        return [], dict(n=0, consistent=None, reason="no different lexical quartet")
    rows = []
    for j, target in enumerate(cases):
        one = []
        for i, source in enumerate(cases):
            if target["pair"] == source["pair"]:
                continue
            one.append(dict(target=target["key"],source=source["key"],fold=target["pair"],
                            same=target["gold"] == source["gold"], **pair(j,i,size,matrix,n)))
        need(len(one) == size-4 and sum(row["same"] for row in one)*2 == len(one), "unbalanced donors")
        rows.extend(one)
    folds = {group: average([r for r in rows if r["fold"] == group]) for group in groups}
    passed = {group: v["t44"] > 0 and v["delta"] > 0 for group,v in folds.items()}
    return rows, dict(n=len(rows), folds=folds, fold_pass=passed, consistent=all(passed.values()),
                      all=average(rows), same=average([r for r in rows if r["same"]]),
                      opposite=average([r for r in rows if not r["same"]]))


def f32_bytes(matrix):
    return b"".join(struct.pack("<%df" % len(row), *row) for row in matrix)


def run(mode, capture, sweep, tag, ext, out):
    cases, vectors = archive(capture, sweep, tag)
    need(len(ext) == 64 and all(ch in "0123456789abcdef" for ch in ext), "invalid extension pin")
    if mode == "run":
        out.mkdir(mode=0o700, exist_ok=False)
        os.environ.update(SPIRALTORCH_BACKEND="cpu", SPIRALTORCH_AUTOTUNE="0",
                          SPIRALTORCH_AUTOTUNE_STORE=str(out/"tuning.json"))
        for pkg in ("numpy","torch","transformers","safetensors"):
            need(pkg not in sys.modules, "unexpected ML runtime")
            sys.modules[pkg] = None
        import spiraltorch as st
        need(sha(Path(st._rs.__file__).read_bytes()) == ext, "wrong native library")
    summary = dict(schema="spiraltorch.local_cross_prefix_transfer.v1",kind=tag,
                   sweep_manifest=SWEEP,capture_manifest=S["ARCHIVE"],extension=ext,
                   protocol=sha((HERE/"README.md").read_bytes()),analysis=sha(Path(__file__).read_bytes()),
                   native_grams=2,decoder_calls=0,geometry_calls=0,updates=0,examples_scored=0,hosts={})
    for host, ys in vectors.items():
        size = len(ys)
        path = out/(host+".f32le")
        if mode == "run":
            t = st.Tensor(size, len(ys[0]), [x for row in ys for x in row])
            matrix = t.row_gram_wide().tolist()
            write(path, f32_bytes(matrix))
        else:
            raw = path.read_bytes()
            need(len(raw) == 4*size*size, "truncated native Gram")
            matrix = list(struct.iter_unpack("<%df" % size, raw))
        exact = reference(ys)
        error = check_gram(matrix, exact)
        rows, calc = analyze(cases,matrix)
        _, oracle = analyze(cases,exact)
        need(calc.get("fold_pass") == oracle.get("fold_pass") and calc["consistent"] == oracle["consistent"],
             "fold classification depends on output precision")
        pairs = b"".join(map(enc,rows))
        dest = out/(host+".jsonl")
        if mode == "run":
            write(dest,pairs)
        else:
            need(pairs == dest.read_bytes(),"pair table does not replay")
        summary["hosts"][host] = dict(native=calc,reference=oracle,max_scaled_error=error,
                                      gram=sha(f32_bytes(matrix)),pairs=sha(pairs))
    status = [h["native"]["consistent"] for h in summary["hosts"].values()]
    summary["consistent"] = None if None in status else all(status)
    raw = enc(summary)
    if mode == "run":
        write(out/"result.json", raw)
    else:
        need((out/"result.json").read_bytes() == raw, "result does not replay")
    print(json.dumps(dict(kind=tag, per_host=len(rows), consistent=summary["consistent"], digest=sha(raw))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("run","verify"))
    ap.add_argument("capture",type=Path)
    ap.add_argument("sweep",type=Path)
    ap.add_argument("tag",choices=("toy","real"))
    ap.add_argument("extension")
    ap.add_argument("output",type=Path)
    a = ap.parse_args()
    run(a.mode,a.capture,a.sweep,a.tag,a.extension,a.output)
