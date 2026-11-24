"""Package the actual one-shot workers, CPU wheel and minimal synthetic parent evidence."""
import argparse
from pathlib import Path
import json
import runpy

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","previous","parent","bank","selector","residual","output"):p.add_argument(n,type=Path)
    a=vars(p.parse_args());previous=a.pop("previous")
    pack=runpy.run_path(str(Path(__file__).resolve().parents[2]/"archive.py"))["pack"]
    print(json.dumps(pack(**a,additional_parents=(("innovation",previous,("inputs.json","training.json","transfer.json")),),
        evidence_files=("installed.json","installed.py","admitted.json","orders.tsv","readback.json"),
        evidence_dirs=("formal-cpu","rehearsal-cpu","binding-cpu","binding-executed-cpu","rust-cpu","causal-lm-cpu","wheel-cpu","wheel")),sort_keys=True))
