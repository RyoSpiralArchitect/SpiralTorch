"""Pack both one-shot workers, the wheel, exact data and parent owners for independent readback."""
import argparse
import json
from pathlib import Path
import runpy

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","previous","parent","bank","selector","residual","output"):
        p.add_argument(n,type=Path)
    a=vars(p.parse_args());prev=a.pop("previous")
    pack=runpy.run_path(str(Path(__file__).resolve().parents[2]/"archive.py"))["pack"]
    result=pack(**a,additional_parents=(("weakest",prev,("inputs.json","training.json","transfer.json")),),
                evidence_files=("installed.json","installed.py","admitted.json","orders.tsv"),
                evidence_dirs=("formal-cpu","rehearsal-cpu","binding-cpu","rust-cpu","causal-lm-cpu","wheel-cpu","wheel"))
    print(json.dumps(result,sort_keys=True))
