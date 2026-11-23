"""Run rehearsal and then formal once under one caller-owned wall-time and disk guard."""
import subprocess
import sys
from pathlib import Path

root,previous,masked,interpreter,launcher=map(Path,sys.argv[1:6])
extension,rehearsal_pin=sys.argv[6:]
here=Path(__file__).resolve().parent
for kind in ("rehearsal","formal"):
    args=[kind,root/kind,previous/kind,masked/kind]
    if kind=="formal":
        pin=subprocess.check_output([sys.executable,"-ISB",here/"study.py","prepare",*args,extension],text=True).strip()
    else:
        pin=rehearsal_pin
    subprocess.run([interpreter,"-IB",launcher,here/"study.py","run",*args,pin],check=True)
    subprocess.run([sys.executable,"-ISB",here/"verify.py",*args[1:],root/(kind+".json"),
                    root/(kind+"-cells.tsv"),root/(kind+"-pairs.tsv")],check=True)
