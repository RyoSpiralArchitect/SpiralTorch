#!/usr/bin/env python3
"""Single-owner hosted handoff. Dispatch is explicit; there is no retry path."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_handoff as handoff
from byte_release_handoff import contract as c

ROOT = Path(__file__).resolve().parents[1]
REPO = "RyoSpiralArchitect/SpiralTorch"
WORKFLOW = "byte_release_qualification.yml"
PACKET = ".github/byte-release-input.zip.age"
SCHEMA = "spiraltorch.byte_release.hosted.v1"


def branch(candidate):
    if type(candidate) is not int or candidate not in (1, 2):
        raise ValueError("only the two admitted candidate branches exist")
    return f"spiralreality/byte-release-candidate-{candidate}-20261025"


def api(path, payload=None):
    command = ["gh", "api", f"repos/{REPO}/{path}"]
    if payload is not None:
        command += ["--method", "POST", "--input", "-"]
    result = subprocess.run(command, input=None if payload is None else c.encoded(payload),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
        raise RuntimeError("GitHub response failed or exceeded its bound; do not resubmit")
    return json.loads(result.stdout) if result.stdout.strip() else None


def runs(candidate):
    return api(f"actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&branch={branch(candidate)}&per_page=2")


def controller_identity(root, source):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()
    head = git("rev-parse", "HEAD")
    if (git("status", "--porcelain", "--untracked-files=all") or git("rev-parse", "HEAD^") != source
            or git("diff", "--name-only", "--no-renames", source, head) != PACKET):
        raise ValueError("controller must add only one ciphertext file to the frozen source")
    return head, handoff.regular(Path(root) / PACKET, handoff.MAX_INPUT_ARCHIVE + 1024 * 1024)


def dispatch(ledger, controller, packet_receipt, recipient, execute=False):
    ledger = Path(ledger)
    candidate = 2 if (ledger / "candidate-2").exists() else 1
    directory = ledger / f"candidate-{candidate}"
    admission = handoff.verify_environment(directory, "local-metal")
    if admission["candidate"] != candidate:
        raise ValueError("candidate slot differs")
    if any((directory / ("hosted-metal." + suffix + ".json")).exists() for suffix in ("dispatch", "start", "result")):
        raise ValueError("hosted ownership was already reserved; observe the original run")
    packet = c.read(packet_receipt)
    if (packet.get("schema") != handoff.SCHEMA or packet.get("kind") != "input"
            or packet.get("admission_sha256") != c.digest(admission) or packet.get("release_ready") is not False
            or not c.is_hash(packet.get("archive_sha256"))
            or not re.fullmatch(r"age1[0-9a-z]{58}", recipient)):
        raise ValueError("input packet or public result recipient differs")
    head, ciphertext = controller_identity(controller, admission["source"]["commit"])
    remote = api("git/ref/heads/" + branch(candidate))
    if remote.get("object", {}).get("sha") != head:
        raise ValueError("remote controller is not the inspected commit")
    prior = runs(candidate)
    if prior.get("total_count") != 0 or prior.get("workflow_runs") != []:
        raise ValueError("this candidate branch already has a dispatch; never retry it")
    secret_name = f"ST_BYTE_RELEASE_CANDIDATE_{candidate}_INPUT_IDENTITY"
    if api("actions/secrets/" + secret_name).get("name") != secret_name:
        raise ValueError("the declared temporary input secret is unavailable")
    inputs = dict(candidate=str(candidate), source_commit=admission["source"]["commit"],
                  admission_sha256=c.digest(admission), input_plaintext_sha256=packet["archive_sha256"],
                  input_ciphertext_sha256=ciphertext["sha256"], result_recipient=recipient)
    claim = dict(schema=SCHEMA, candidate=candidate, admission_sha256=c.digest(admission),
                 controller_commit=head, branch=branch(candidate), inputs=inputs,
                 release_ready=False)
    if not execute:
        return dict(claim, preflight="passed", dispatched=False)
    c.save(directory / "hosted-metal.dispatch.json", dict(claim, claimed_unix=time.time()))
    try:
        api(f"actions/workflows/{WORKFLOW}/dispatches", {"ref": branch(candidate), "inputs": inputs})
    except BaseException as error:
        c.save(directory / "hosted-metal.dispatch-response.json", {
            "outcome": "ambiguous_do_not_retry", "error_type": type(error).__name__, "release_ready": False,
        })
        raise
    c.save(directory / "hosted-metal.dispatch-response.json", {"outcome": "submitted", "release_ready": False})
    return dict(claim, dispatched=True)


def unique_run(candidate, controller_commit, expected_id=None):
    result = runs(candidate)
    rows = result.get("workflow_runs", [])
    if result.get("total_count") != 1 or len(rows) != 1:
        raise ValueError("a unique first dispatch is not observable; do not dispatch again")
    row = rows[0]
    if (type(row.get("id")) is not int or row.get("run_attempt") != 1
            or row.get("event") != "workflow_dispatch" or row.get("head_sha") != controller_commit
            or row.get("head_branch") != branch(candidate)
            or (expected_id is not None and row["id"] != expected_id)):
        raise ValueError("hosted run identity or attempt differs")
    return row


def bind_run(directory):
    directory = Path(directory)
    admission = c.validate_admission(c.read(directory / "admission.json"))
    claim = c.read(directory / "hosted-metal.dispatch.json")
    if claim.get("admission_sha256") != c.digest(admission) or claim.get("candidate") != admission["candidate"]:
        raise ValueError("dispatch belongs to another candidate")
    run = unique_run(admission["candidate"], claim["controller_commit"])
    binding = dict(schema=SCHEMA, run_id=run["id"], run_attempt=1, controller_commit=run["head_sha"],
                   admission_sha256=c.digest(admission), release_ready=False)
    path = directory / "hosted-metal.run.json"
    if path.exists():
        if c.read(path) != binding:
            raise ValueError("an earlier different run binding already exists")
    else:
        c.save(path, binding)
    return binding


def admit_run(output, env=None):
    env = os.environ if env is None else env
    record = dict(schema=SCHEMA, admitted=False, release_ready=False)
    try:
        candidate = int(env["CANDIDATE"])
        source, head = env["SOURCE_COMMIT"], env["GITHUB_SHA"]
        if (env["GITHUB_REPOSITORY"] != REPO or env["GITHUB_RUN_ATTEMPT"] != "1"
                or env["GITHUB_REF"] != "refs/heads/" + branch(candidate)
                or not c.is_hash(source, 40) or not c.is_hash(head, 40)
                or not env["GITHUB_RUN_ID"].isdigit()
                or any(not c.is_hash(env[key]) for key in ("ADMISSION_SHA256", "INPUT_PLAINTEXT_SHA256", "INPUT_CIPHERTEXT_SHA256"))):
            raise ValueError("hosted invocation is outside the fixed first-attempt policy")
        observed, ciphertext = controller_identity(ROOT, source)
        if observed != head or ciphertext["sha256"] != env["INPUT_CIPHERTEXT_SHA256"]:
            raise ValueError("hosted source or ciphertext differs")
        run = unique_run(candidate, head, int(env["GITHUB_RUN_ID"]))
        record.update(admitted=True, candidate=candidate, run_id=run["id"], run_attempt=1,
                      controller_commit=head, source_commit=source, admission_sha256=env["ADMISSION_SHA256"],
                      image_os=env.get("ImageOS"), image_version=env.get("ImageVersion"))
        return record
    except BaseException as error:
        record["error_type"] = type(error).__name__
        raise
    finally:
        c.save(output, record)


def verify_input(root, candidate, expected):
    root = Path(root)
    directory = root / "ledger" / f"candidate-{candidate}"
    admission = handoff.verify_environment(directory, "local-metal")
    if admission["candidate"] != candidate or c.digest(admission) != expected or c.source_identity(ROOT) != admission["source"]:
        raise ValueError("input or frozen execution source differs")
    wheel = root / "wheel" / admission["wheels"]["macos"]["filename"]
    if handoff.artifacts.wheel_inventory(wheel) != admission["wheels"]["macos"]:
        raise ValueError("transported wheel differs")
    c.admit(root / "ledger", admission)
    return dict(admission_sha256=expected, wheel=str(wheel), release_ready=False)


def pack_result(evidence, bundle, candidate, output, expected):
    paths = handoff.inventory(evidence)
    base = Path(bundle) / "ledger" / f"candidate-{candidate}"
    for name in ("admission.json", "hosted-metal.start.json", "hosted-metal.result.json"):
        if (base / name).is_file():
            paths[f"ledger/candidate-{candidate}/{name}"] = base / name
    if (base / "hosted-metal").is_dir():
        paths.update({f"ledger/candidate-{candidate}/hosted-metal/{name}": path
                      for name, path in handoff.inventory(base / "hosted-metal").items()})
    return handoff.pack(paths, output, "result", expected)


def recover_result(directory, archive, expected_sha256):
    """Import the first terminal return; incomplete/failed jobs never turn green."""
    directory = Path(directory)
    admission = c.validate_admission(c.read(directory / "admission.json"))
    claim = c.read(directory / "hosted-metal.dispatch.json")
    binding = c.read(directory / "hosted-metal.run.json")
    if (claim.get("admission_sha256") != c.digest(admission) or binding.get("admission_sha256") != c.digest(admission)
            or binding.get("controller_commit") != claim.get("controller_commit")):
        raise ValueError("return belongs to another dispatch")
    run = unique_run(admission["candidate"], claim["controller_commit"], binding["run_id"])
    if run.get("status") != "completed":
        raise ValueError("the original hosted run is still live; only observe it")
    if any((directory / name).exists() for name in ("hosted-metal", "hosted-metal.start.json", "hosted-metal.result.json")):
        raise ValueError("hosted outcome already exists; never replace its first result")
    raw = directory / f"hosted-return-{run['id']}"
    handoff.unpack(archive, raw, expected_sha256, "result", c.digest(admission))
    remote = raw / "ledger" / f"candidate-{admission['candidate']}"
    outcome, reason = None, "missing_or_invalid_remote_acceptance"
    try:
        context = c.read(raw / "hosted-context.json")
        if (context.get("schema") != SCHEMA or context.get("admitted") is not True
                or context.get("run_id") != run["id"] or context.get("run_attempt") != 1
                or context.get("controller_commit") != claim["controller_commit"]
                or context.get("source_commit") != admission["source"]["commit"]
                or context.get("admission_sha256") != c.digest(admission)
                or c.read(remote / "admission.json") != admission):
            raise ValueError("return context differs")
        result = c.read(remote / "hosted-metal/result.json")
        receipt = c.read(remote / "hosted-metal.result.json")
        start = c.read(remote / "hosted-metal.start.json")
        if (result.get("schema") != "spiraltorch.byte_release.environment_schedule.v1"
                or result.get("environment") != "hosted-metal" or result.get("admission_sha256") != c.digest(admission)
                or result.get("release_ready") is not False
                or result.get("status") not in ("verified_rounds", "failed", "timeout", "cancelled")
                or receipt.get("status") != result["status"] or receipt.get("evidence_sha256") != c.digest(result)
                or receipt.get("admission_sha256") != c.digest(admission) or receipt.get("release_ready") is not False
                or start.get("environment") != "hosted-metal" or start.get("admission_sha256") != c.digest(admission)):
            raise ValueError("return acceptance receipt differs")
        if result["status"] == "verified_rounds":
            handoff.verify_environment(remote, "hosted-metal")
            if run.get("conclusion") != "success":
                reason = "numerical_rounds_returned_but_hosted_job_did_not_succeed"
                raise ValueError(reason)
        outcome = result
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        # The decrypted raw return stays untouched, even if its records are
        # malformed. A derived failure is explicitly not a native worker result.
        outcome = None
    destination = directory / "hosted-metal"
    destination.mkdir(mode=0o700)
    if outcome is not None:
        for name, source in handoff.inventory(remote / "hosted-metal").items():
            target = destination / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with source.open("rb") as reader, target.open("xb") as writer:
                os.fchmod(writer.fileno(), 0o600)
                shutil.copyfileobj(reader, writer, handoff.CHUNK)
        c.save(directory / "hosted-metal.start.json", start)
        if outcome["status"] == "verified_rounds":
            handoff.verify_environment(directory, "hosted-metal", receipt=receipt)
    else:
        c.save(directory / "hosted-metal.start.json", dict(environment="hosted-metal",
               admission_sha256=c.digest(admission), origin="coordinator_observed_terminal_run", run_id=run["id"], release_ready=False))
        outcome = dict(schema="spiraltorch.byte_release.environment_schedule.v1", environment="hosted-metal",
                       admission_sha256=c.digest(admission), status="failed", release_ready=False,
                       origin="coordinator_observed_terminal_run", reason=reason, run_id=run["id"],
                       github_conclusion=run.get("conclusion"), gpu_execution="not_inferred",
                       raw_evidence_directory=raw.name, archive_sha256=expected_sha256)
        c.save(destination / "result.json", outcome)
    c.finish_environment(directory, "hosted-metal", outcome)
    if outcome["status"] == "verified_rounds":
        handoff.verify_environment(directory, "hosted-metal")
    result = dict(schema=SCHEMA, run_id=run["id"], admission_sha256=c.digest(admission),
                  archive_sha256=expected_sha256, status=outcome["status"], release_ready=False)
    c.save(directory / "hosted-metal.import.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    submit = commands.add_parser("dispatch")
    for name in ("ledger", "controller", "packet-receipt"):
        submit.add_argument("--" + name, type=Path, required=True)
    submit.add_argument("--recipient", required=True)
    submit.add_argument("--execute", action="store_true")
    bind = commands.add_parser("bind-run")
    bind.add_argument("--candidate-directory", type=Path, required=True)
    admit = commands.add_parser("admit-run")
    admit.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-input")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    verify.add_argument("--admission-sha256", required=True)
    result = commands.add_parser("pack-result")
    for name in ("evidence", "input", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    result.add_argument("--admission-sha256", required=True)
    recover = commands.add_parser("recover-result")
    recover.add_argument("--candidate-directory", type=Path, required=True)
    recover.add_argument("--archive", type=Path, required=True)
    recover.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if args.operation == "dispatch":
        value = dispatch(args.ledger, args.controller, args.packet_receipt, args.recipient, args.execute)
    elif args.operation == "bind-run":
        value = bind_run(args.candidate_directory)
    elif args.operation == "admit-run":
        value = admit_run(args.output)
    elif args.operation == "verify-input":
        value = verify_input(args.input, args.candidate, args.admission_sha256)
    elif args.operation == "recover-result":
        value = recover_result(args.candidate_directory, args.archive, args.sha256)
    else:
        value = pack_result(args.evidence, args.input, args.candidate, args.output, args.admission_sha256)
    print(c.encoded(value).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
