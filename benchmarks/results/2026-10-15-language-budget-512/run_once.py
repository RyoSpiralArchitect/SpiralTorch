"""Claim this bounded pilot once before starting its original frozen runner."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def claim(output, request_hash, runner_hash, adopted_pid=None):
    # Leave the claim on failure: restarting requires an explicit recovery plan,
    # never an implicit second consumption of the predeclared training budget.
    path = output / "native-run-claim.json"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(dict(schema="spiraltorch.byte_corpus.pilot_claim.v1",
             request_sha256=request_hash, runner_sha256=runner_hash,
             recorded_at=datetime.now(timezone.utc).isoformat(),
             adopted_existing_pid=adopted_pid, before_gpu_launch=adopted_pid is None),out,indent=2)
        out.write("\n")
    if adopted_pid is None and any(output.glob("native-*.json")):
        # The newly written claim itself is excluded from the collision test.
        existing = [p for p in output.glob("native-*.json") if p != path]
        if existing:
            raise ValueError("native outputs already exist; no training started")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("runner","repo","output","historical"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--adopt-pid",type=int)
    args=parser.parse_args()
    preflight=json.loads((args.output/"prepared.json").read_bytes())
    request_hash=sha((args.output/"request.json").read_bytes())
    runner_hash=sha(args.runner.read_bytes())
    if (request_hash!=preflight["request_sha256"] or runner_hash!=preflight["runner_sha256"]
            or sha((args.output/"PLAN.md").read_bytes())!=preflight["plan_sha256"]):
        raise ValueError("frozen pilot identity changed")
    if args.adopt_pid is not None:
        if args.adopt_pid<=0:
            raise ValueError("invalid existing process")
        command=subprocess.check_output(["ps","-p",str(args.adopt_pid),"-o","command="],text=True)
        if str(args.runner.resolve()) not in command or str(args.output.resolve()) not in command or " run --repo " not in command:
            raise ValueError("existing process is not the pinned pilot")
        claim(args.output,request_hash,runner_hash,args.adopt_pid)
        print(json.dumps(dict(existing_run_adopted=True,pid=args.adopt_pid,new_training_started=False)),flush=True)
        return
    claim(args.output,request_hash,runner_hash)
    result=subprocess.run([sys.executable,"-I","-B",str(args.runner),"run","--repo",str(args.repo),
                           "--historical",str(args.historical),"--output",str(args.output)],check=False)
    sys.exit(result.returncode)


if __name__=="__main__":
    main()
