"""CPU worker for one split. Run only through the bounded launcher; no retries."""
import argparse
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
v=SimpleNamespace(**runpy.run_path(str(HERE/"verify.py")))


def campaign(kind,root,parent,signal,extension):
    here=root/kind
    here.mkdir(mode=0o700)
    study=v.v
    sp=study.plan(kind,parent/kind,signal/kind,extension)
    pin=study.d.save(here/"spec.json",sp)
    print(kind,"spec",pin,flush=True)
    study.run(here,parent/kind,signal/kind,pin)
    v.main(here,parent/kind,signal/kind,root/(kind+".json"),root/(kind+".tsv"))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("kind",choices=("rehearsal","formal"))
    for key in ("root","parent","signal"): cli.add_argument(key,type=Path)
    cli.add_argument("extension")
    campaign(**vars(cli.parse_args()))
