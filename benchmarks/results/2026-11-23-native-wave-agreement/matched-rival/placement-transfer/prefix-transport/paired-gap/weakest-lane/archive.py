"""Archive the exact experimental inputs, worker attempts, and subsequent diagnostic (including its first version)."""
import argparse
import json
from pathlib import Path
import runpy

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","previous","parent","bank","selector","residual","output"):
        p.add_argument(n,type=Path)
    a=vars(p.parse_args());prev=a.pop("previous")
    pack=runpy.run_path(str(Path(__file__).resolve().parents[1]/"archive.py"))["pack"]
    result=pack(**a,additional_parents=(("paired",prev,("inputs.json","training.json","transfer.json")),),
                evidence_files=("installed.json","installed.py","admitted.json","decisions.json","orders38.tsv",
                                "capacity-exact.json","capacity.json","capacity-first-source.tar.gz"),
                evidence_dirs=("formal-cpu","rehearsal-cpu","binding-cpu","rust-cpu","wheel-cpu","wheel"))
    print(json.dumps(result,sort_keys=True))
