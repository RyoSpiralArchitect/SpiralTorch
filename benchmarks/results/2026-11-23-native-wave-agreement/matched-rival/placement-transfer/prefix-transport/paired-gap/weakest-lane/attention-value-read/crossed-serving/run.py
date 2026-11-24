"""Native inference only. Stop on a changed score, owner, physical witness or frozen plan."""
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_design as c


def physical(st, model, traces, cell):
    rt = st.NativeGeometryRuntime(model.geometry_artifact())
    got = []
    n = 0
    for i in range(0, len(traces), 16):
        chunk = traces[i:i + 16]
        inputs = [v for tr in chunk for v in tr["site_input"]]
        sources = [v for tr in chunk for v in tr["source"]]
        raw = rt.forward_value_read_sequence(c.x.s.SITE, inputs, sources, [len(chunk), 1]).output
        c.need(len(raw) == len(inputs), "bad native witness shape")
        got.extend([raw[j:j+3] for j in range(0, len(raw), 3)])
        n += 1
    # Packing also distinguishes +0 and -0 at every channel.
    expected = c.delivery(cell["params"], traces)
    pack = struct.Struct("<3f").pack
    c.need(len(got) == len(expected) and all(pack(*y) == pack(*z) for y, z in zip(got, expected))
           and c.d.digest(c.d.encoded(got)) == cell["site_rows_sha256"], "native site delivery")
    return n, got


def run(parent, out, kind, freeze):
    frozen = c.x.read(out / "frozen.json", freeze)
    c.need(kind in c.PINS and frozen["code"] == c.code() and frozen["parents"] == c.PINS, "plan lineage")
    if kind == "formal":
        admit = c.s.k.read(out / "admitted.json")
        result = (out / "rehearsal/result.json").read_bytes()
        receipt = (out / "rehearsal-cpu/receipt.json").read_bytes()
        c.need(admit == dict(frozen=freeze, result=c.d.digest(result), receipt=c.d.digest(receipt))
               and c.s.k.read(out / "rehearsal-cpu/receipt.json")["success"], "rehearsal not audited")
    ctx = c.saved(parent, kind)
    plan = c.check_plan(parent, out, kind, frozen)
    traces = c.banks(ctx)
    hosts = c.s.old.hosts(out / "base" / kind, ctx.sp)
    dest = out / kind
    dest.mkdir(mode=0o700)
    c.d.save(dest / "started.json", dict(frozen=freeze, plan=frozen["plans"][kind], retry=False))
    sys.path.insert(0, str(parent / "site"))
    st = c.s.v.sig.native(c.EXTENSION)
    rows, scores = c.transfer(ctx)
    bases = {b: c.s.old.decoder(st, out / "base" / kind, ctx.sp, b, hosts[str(b)]) for b in ctx.sp["bases"]}
    calls = batches = diagonal = 0
    with (dest / "witnesses.jsonl").open("xb") as w, (dest / "scores.jsonl").open("xb") as z:
        for cell in plan["cells"]:
            b, route, donor = cell["base"], cell["route"], cell["donor"]
            art = st.GeometryAdapterArtifact.create_json(json.dumps(cell["artifact"]))
            model = bases[b].with_geometry(art)
            oid, vec, raw = c.s.owner(model, hosts[str(b)], route)
            c.need(vec == cell["params"], "serving coordinates changed")
            factual = cell["scale"] == "raw" and route == donor
            if factual:
                c.need(oid == cell["donor_id"] and c.d.digest(raw) == cell["donor_file"], "factual artifact bytes")
            eligible = scores[c.s.arm(b, donor, cell["teacher"], cell["order"])]
            batches_used, site = physical(st, model, traces[b, route], cell)
            batches += batches_used
            c.s.old.archive_step(w, dict(name=cell["name"], state=raw.decode(), owner=oid, rows=site))
            for r in rows:
                values = model.forward(r["prompt"], [1, len(r["prompt"])], last_only=True).values()
                logits = c.s.v.u.bits(values)
                if factual:
                    c.need(logits == c.s.v.u.bits(eligible[r["key"]]), "diagonal logits differ")
                    diagonal += 1
                calls += 1
                c.s.old.archive_step(z, dict(name=cell["name"], owner=oid, key=r["key"], logits=values))
    c.need((len(plan["cells"]), calls, batches) == c.COUNTS[kind] and diagonal * 4 == calls, "budget")
    receipt = dict(frozen=freeze, plans=frozen["plans"][kind], forwards=calls, diagonal=diagonal,
                   site_batches=batches, updates=0, scores=c.d.digest((dest/"scores.jsonl").read_bytes()),
                   witnesses=c.d.digest((dest/"witnesses.jsonl").read_bytes()))
    print(c.d.save(dest / "scored.json", receipt), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:])
