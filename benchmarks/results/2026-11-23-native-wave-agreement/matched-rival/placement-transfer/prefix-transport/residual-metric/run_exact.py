"""New split only; pair the frozen training worker with the divide-last exact archive reader."""
import argparse
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
audit=SimpleNamespace(**runpy.run_path(str(HERE/"verify_exact.py")))
v=audit.v1.v


def run(kind,root,parent,signal,extension):
    split=root/kind
    split.mkdir(mode=0o700,exist_ok=False)
    parent,signal=parent/kind,signal/kind
    pin=v.d.save(split/"spec.json",v.plan(kind,parent,signal,extension))
    v.d.save(split/"reader.json",dict(spec=pin,reader=v.d.digest((HERE/"verify_exact.py").read_bytes()),
                                      launcher=v.d.digest(Path(__file__).read_bytes()),frozen_reader=audit.V1))
    v.run(split,parent,signal,pin)
    print(audit.audit(split,parent,signal,root/(kind+"-exact.json"),root/(kind+"-exact.tsv")))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("kind",choices=("rehearsal","formal"))
    for name in ("root","parent","signal"): cli.add_argument(name,type=Path)
    cli.add_argument("extension")
    run(**vars(cli.parse_args()))
