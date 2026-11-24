"""Run one partition once, including a separate no-ML audit in the same deadline."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cross_design as c


def worker(parent, out, kind, pin):
    argv = [sys.executable, "-ISB"]
    args = [str(parent), str(out), kind, pin]
    subprocess.run(argv + [str(HERE / "run.py")] + args, check=True)
    subprocess.run(argv + [str(HERE / "audit.py")] + args, check=True)


def once(parent, out, kind, pin):
    c.need(kind in c.COUNTS and not (out/kind).exists() and not (out/(kind+"-cpu")).exists(),
           "refusing to retry or overwrite")
    c.need(c.x.read(out / "frozen.json", pin)["code"] == c.code(), "sources changed after freeze")
    os.environ.update(SPIRALTORCH_BACKEND="cpu", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                      RAYON_NUM_THREADS="1")
    scoped = HERE.parents[5] / "parameter-vjp/bounded.py"
    runner = runpy.run_path(str(scoped))["run"]
    dest = out / (kind + "-cpu")
    runner(dest, 60 if kind == "rehearsal" else 120,
           [sys.executable, "-ISB", str(HERE / "workflow.py"), "worker", str(parent), str(out), kind, pin])
    receipt = (dest / "receipt.json").read_bytes()
    result = (out / kind / "result.json").read_bytes()
    c.need(json.loads(receipt)["success"] and json.loads(result)["frozen"] == pin, "no successful audited result")
    completed = dict(frozen=pin, result=c.d.digest(result), receipt=c.d.digest(receipt))
    print(c.d.save(out / ("admitted.json" if kind == "rehearsal" else "completed.json"), completed))


if __name__ == "__main__":
    action, parent, out, kind, pin = sys.argv[1:]
    {"once": once, "worker": worker}[action](Path(parent), Path(out), kind, pin)
