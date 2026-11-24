"""Run one local CPU process group once; preserve its first failure and outputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def run(root, seconds, command):
    root.mkdir(mode=0o700, exist_ok=False)
    start = time.monotonic()
    free = lambda: shutil.disk_usage(root).free
    low = free()
    receipt = dict(seconds_limit=seconds, retry=False, success=False)
    try:
        if low<(4<<30)+(128<<20):
            raise ValueError("insufficient free space before launch")
        with (root/"process.log").open("xb") as log:
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    low=min(low,free())
                    if low<4<<30 or time.monotonic()-start>=seconds:
                        raise ValueError("time or storage limit")
                    time.sleep(.10)
                if p.returncode:
                    raise ValueError("worker exit %s" % p.returncode)
            finally:
                if p.poll() is None:
                    try:
                        os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
                receipt["exit_code"]=p.returncode
        low=min(low,free())
        if low<4<<30 or time.monotonic()-start>seconds:
            raise ValueError("end limit")
        receipt["success"]=True
    except BaseException as exc:
        receipt["error"]=str(exc)
        raise
    finally:
        receipt.update(elapsed=time.monotonic()-start,min_free=low)
        with (root/"receipt.json").open("x") as f:
            json.dump(receipt,f,sort_keys=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("seconds",type=int)
    p.add_argument("command",nargs=argparse.REMAINDER)
    args=p.parse_args()
    run(args.root,args.seconds,args.command)
