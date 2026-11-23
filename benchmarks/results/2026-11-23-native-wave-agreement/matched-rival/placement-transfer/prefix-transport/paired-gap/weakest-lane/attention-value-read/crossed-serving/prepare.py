"""Freeze both train-only treatment plans. Extract only pinned tiny synthetic hosts."""
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_design as c


def prepare(parent, out):
    c.need(not any(out.iterdir()), "output is not empty")
    archive = parent / "journals-portable.tar.gz"
    c.need(c.d.digest(archive.read_bytes()) == c.ARCHIVE, "wrong parent archive")
    installed = json.loads((parent / "installed.json").read_bytes())
    c.need((installed["extension"], installed["wheel_sha256"]) == (c.EXTENSION, c.WHEEL), "wrong CPU extension")
    host_root = out / "base"
    host_root.mkdir(mode=0o700)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        c.need(len(members) == len({m.name for m in members}), "duplicate archive member")
        pins = json.load(tar.extractfile("SHA256.json"))
        restored = {}
        for kind, bases in (("formal", (557, 563, 569)), ("rehearsal", (571,))):
            for n in ["attempt/hosts.json", "complete.json"] + [f"base-{b}.safetensors" for b in bases]:
                key = "parents/base/" + kind + "/" + n
                item = tar.getmember(key)
                c.need(item.isfile(), key)
                buf = tar.extractfile(item).read()
                c.need(c.d.digest(buf) == pins[key], key)
                path = host_root / kind / n
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with path.open("xb") as f:
                    f.write(buf)
                restored[key] = pins[key]
    plans = {kind: c.d.save(out / (kind + "-plan.json"), c.plan(parent, host_root, kind)) for kind in c.PINS}
    return c.d.save(out / "frozen.json", dict(schema="spiraltorch.attention_value_route_cross.v1",
                    code=c.code(), parents=c.PINS, archive=c.ARCHIVE, hosts=restored, plans=plans,
                    wheel=c.WHEEL, extension=c.EXTENSION, counts=c.COUNTS, updates=0, retries=0))


if __name__ == "__main__":
    print(prepare(*map(Path, sys.argv[1:])))
