"""Admit the formal worker only after a successful bounded rehearsal and independent raw replay."""
import json
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    root, parent = map(Path,sys.argv[1:3])
    inputs = sys.argv[3]
    a=runpy.run_path(str(Path(__file__).with_name("audit.py")))
    d,need=a["d"],a["need"]
    work=root/"rehearsal"
    receipt=(root/"rehearsal-cpu/receipt.json").read_bytes()
    rr=json.loads(receipt)
    need(rr["success"] and rr["exit_code"]==0 and not rr["retry"] and 0 < rr["elapsed"] <= rr["seconds_limit"]==120,
         "not a clean first rehearsal")
    pin=lambda n:d.digest((work/n).read_bytes())
    result=a["result"](work,parent/"rehearsal",inputs,pin("trained.json"),pin("training-checked.json"),pin("scored.json"))
    data=d.encoded(result)
    need(data==(work/"result.json").read_bytes() and result["kind"]=="rehearsal", "rehearsal did not reproduce")
    print(d.save(root/"rehearsal-succeeded.json",dict(result=d.digest(data),receipt=d.digest(receipt))))
