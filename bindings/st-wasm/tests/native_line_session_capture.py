"""Extract the demo's local download from Chrome --dump-dom ...?capture, offline."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys


class Capture(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current=None
        self.chunks={}
        self.passed=False

    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=="pre" and a.get("id") in ("line-transfer","owner-transfer","ray-transfer","result"):
            self.current=a["id"]
            if self.current in self.chunks:
                raise ValueError("duplicate demo field")
            self.chunks[self.current]=[]
            if self.current=="result":
                self.passed=a.get("data-status")=="passed"

    def handle_endtag(self, tag):
        if tag=="pre":
            self.current=None

    def handle_data(self, value):
        if self.current:
            self.chunks[self.current].append(value)


def extract(dom, out, owner_out=None, ray_out=None):
    parser=Capture()
    parser.feed(dom.read_text())
    parser.close()
    assert parser.passed
    result=json.loads("".join(parser.chunks["result"]))["line_session"]
    raw="".join(parser.chunks["line-transfer"]).encode()
    wrapper=json.loads(raw)
    plan=(Path(__file__).with_name("native_line_session_plan.json")).read_bytes()
    pin=lambda data: "sha256:"+hashlib.sha256(data).hexdigest()
    assert wrapper["plan_id"]==pin(plan) and len(wrapper["cases"])==4
    for r in wrapper["cases"].values():
        assert pin(r["payload"].encode())==r["id"] and json.loads(r["payload"])["updates"]==4
    with out.open("xb") as stream:
        stream.write(raw)
    summary=json.loads("".join(parser.chunks["result"]))
    if owner_out is not None:
        result={"line_session":result}
    for field, key, target, version in [("owner-transfer","line_owners",owner_out,"v2"),("ray-transfer","line_ray",ray_out,"v3")]:
        if target is None:
            continue
        content="".join(parser.chunks[field]).encode()
        owners=json.loads(content)
        assert set(owners)=={"lora","geometry","joint"}
        for mode,row in owners.items():
            snap=json.loads(row["payload"])
            assert pin(row["payload"].encode())==row["id"] and snap["mode"]==mode and snap["updates"]==4
            assert snap["schema"]=="spiraltorch.native_causal_lm_line_checkpoint."+version
        with target.open("xb") as stream:
            stream.write(content)
        result[key]=summary[key]
    print(json.dumps(result))


if __name__=="__main__":
    extract(*map(Path,sys.argv[1:]))
