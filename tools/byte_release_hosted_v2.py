#!/usr/bin/env python3
"""Transport-only controller. Live operations require a separately pinned amendment."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_hosted as old
import byte_release_transport_v2 as t
import owned_process

c, handoff = t.c, t.v1
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "spiraltorch.byte_release.hosted.v2"
FETCH_SECONDS = 300
api = old.api


def workflow_id():
    row = api(f"actions/workflows/{t.WORKFLOW}")
    if (type(row) is not dict or type(row.get("id")) is not int or row["id"] <= 0
            or row.get("path") != ".github/workflows/" + t.WORKFLOW):
        raise ValueError("remote workflow namespace differs")
    return row["id"]


def runs(candidate):
    identifier = workflow_id()
    result = api(f"actions/workflows/{identifier}/runs?event=workflow_dispatch&branch={t.branch(candidate)}&per_page=2")
    if (type(result) is not dict or type(result.get("total_count")) is not int
            or result["total_count"] < 0 or type(result.get("workflow_runs")) is not list):
        raise ValueError("remote run listing is malformed")
    return identifier, result


def unique_run(candidate, controller, expected_id=None):
    identifier, result = runs(candidate)
    rows = result.get("workflow_runs", [])
    if result.get("total_count") != 1 or len(rows) != 1:
        raise ValueError("a unique first dispatch is not observable; never retry")
    row = rows[0]
    if (type(row) is not dict or type(row.get("id")) is not int or row["id"] <= 0
            or type(row.get("workflow_id")) is not int or row["workflow_id"] != identifier
            or type(row.get("run_attempt")) is not int or row["run_attempt"] != 1
            or row.get("event") != "workflow_dispatch" or row.get("head_sha") != controller
            or row.get("head_branch") != t.branch(candidate)
            or (expected_id is not None and row["id"] != expected_id)):
        raise ValueError("remote run identity differs")
    return row


def no_hosted_claim(directory):
    paths = [Path(directory) / "hosted-metal"] + [
        Path(directory) / ("hosted-metal." + suffix + ".json")
        for suffix in ("dispatch", "dispatch-response", "start", "run", "result", "import")]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("hosted ownership is already reserved or consumed")


def dispatch(ledger, controller, source, amendment, expected, recipient, execute=False):
    descriptor = t.read_record(Path(controller) / t.DESCRIPTOR)
    head = t.validate_controller(controller, source, amendment, descriptor, expected)
    candidate = descriptor["candidate"]
    directory = Path(ledger) / f"candidate-{candidate}"
    admission = handoff.verify_environment(directory, "local-metal")
    if (admission["candidate"] != candidate or c.digest(admission) != amendment["admission_sha256"]
            or admission["source"] != c.source_identity(source)
            or c.read(directory / "local-metal.result.json")["evidence_sha256"]
            != amendment["local_evidence_sha256"]
            or (candidate == 1 and (Path(ledger) / "candidate-2").exists())):
        raise ValueError("local candidate or execution source differs")
    no_hosted_claim(directory)
    if recipient != amendment["result_recipient"]:
        raise ValueError("public result recipient differs from the amendment")
    if api("git/ref/heads/" + t.branch(candidate)).get("object", {}).get("sha") != head:
        raise ValueError("remote controller differs")
    identifier, prior = runs(candidate)
    if prior.get("total_count") != 0 or prior.get("workflow_runs") != []:
        raise ValueError("candidate workflow already has a dispatch")
    suffixes = ("INPUT_IDENTITY",) if descriptor["schema"] == t.GITHUB_DESCRIPTOR_SCHEMA else ("INPUT_IDENTITY", "INPUT_URL")
    for suffix in suffixes:
        name = f"ST_BYTE_RELEASE_CANDIDATE_{candidate}_{suffix}"
        if api("actions/secrets/" + name).get("name") != name:
            raise ValueError("temporary transport secret is unavailable")
    inputs = dict(candidate=str(candidate), source_commit=amendment["source_commit"],
                  admission_sha256=amendment["admission_sha256"],
                  input_plaintext_sha256=descriptor["plaintext"]["sha256"],
                  input_ciphertext_sha256=descriptor["ciphertext"]["sha256"],
                  result_recipient=recipient, transport_amendment=c.encoded(amendment).decode(),
                  transport_amendment_sha256=expected)
    claim = dict(schema=SCHEMA, candidate=candidate, admission_sha256=c.digest(admission),
                 controller_commit=head, workflow_id=identifier, branch=t.branch(candidate),
                 amendment=amendment, amendment_sha256=expected, descriptor=descriptor,
                 inputs=inputs, release_ready=False)
    if not execute:
        return dict(claim, preflight="passed", dispatched=False)
    c.save(directory / "hosted-metal.dispatch.json", dict(claim, claimed_unix=time.time()))
    try:
        api(f"actions/workflows/{identifier}/dispatches", {"ref": t.branch(candidate), "inputs": inputs})
    except BaseException as error:
        c.save(directory / "hosted-metal.dispatch-response.json",
               dict(outcome="ambiguous_do_not_retry", error_type=type(error).__name__, release_ready=False))
        raise
    c.save(directory / "hosted-metal.dispatch-response.json", dict(outcome="submitted", release_ready=False))
    return dict(claim, dispatched=True)


def load_claim(directory):
    directory = Path(directory)
    admission = c.validate_admission(c.read(directory / "admission.json"))
    handoff.verify_environment(directory, "local-metal")
    claim = c.read(directory / "hosted-metal.dispatch.json")
    amendment, descriptor = claim.get("amendment"), claim.get("descriptor")
    t.validate_amendment(amendment, claim.get("amendment_sha256"), descriptor)
    if (claim.get("schema") != SCHEMA or claim.get("release_ready") is not False
            or claim.get("candidate") != admission["candidate"]
            or claim.get("admission_sha256") != c.digest(admission)
            or amendment["admission_sha256"] != c.digest(admission)
            or amendment["source_commit"] != admission["source"]["commit"]
            or amendment["source_tree"] != admission["source"]["tree"]
            or claim.get("controller_commit") != amendment["controller_commit"]
            or claim.get("branch") != t.branch(admission["candidate"])
            or type(claim.get("workflow_id")) is not int or claim["workflow_id"] <= 0
            or c.read(directory / "local-metal.result.json")["evidence_sha256"]
            != amendment["local_evidence_sha256"]):
        raise ValueError("dispatch is not bound to this admission")
    return admission, claim


def bind_run(directory):
    admission, claim = load_claim(directory)
    run = unique_run(admission["candidate"], claim["controller_commit"])
    if run["workflow_id"] != claim["workflow_id"]:
        raise ValueError("resolved workflow identity changed")
    binding = dict(schema=SCHEMA, run_id=run["id"], run_attempt=1,
                   workflow_id=run["workflow_id"], controller_commit=run["head_sha"],
                   admission_sha256=c.digest(admission), amendment_sha256=claim["amendment_sha256"],
                   release_ready=False)
    path = Path(directory) / "hosted-metal.run.json"
    if path.exists():
        if c.read(path) != binding:
            raise ValueError("a different first run is already bound")
    else:
        c.save(path, binding)
    return binding


def environment_records(env):
    descriptor = t.read_record(ROOT / t.DESCRIPTOR)
    amendment = t.parse_record(env["TRANSPORT_AMENDMENT"])
    expected = env["TRANSPORT_AMENDMENT_SHA256"]
    t.validate_amendment(amendment, expected, descriptor)
    selectors = dict(CANDIDATE=str(descriptor["candidate"]),
                     SOURCE_COMMIT=amendment["source_commit"],
                     ADMISSION_SHA256=amendment["admission_sha256"],
                     RESULT_RECIPIENT=amendment["result_recipient"],
                     INPUT_PLAINTEXT_SHA256=descriptor["plaintext"]["sha256"],
                     INPUT_CIPHERTEXT_SHA256=descriptor["ciphertext"]["sha256"])
    if any(env.get(key) != value for key, value in selectors.items()):
        raise ValueError("invocation selectors differ from the pinned input")
    return amendment, descriptor, expected


def admit_run(source, output, env=None):
    env = os.environ if env is None else env
    record = dict(schema=SCHEMA, admitted=False, release_ready=False)
    try:
        amendment, descriptor, expected = environment_records(env)
        candidate = descriptor["candidate"]
        if (env.get("GITHUB_REPOSITORY") != t.REPO or env.get("GITHUB_RUN_ATTEMPT") != "1"
                or env.get("GITHUB_REF") != "refs/heads/" + t.branch(candidate)
                or env.get("GITHUB_SHA") != amendment["controller_commit"]
                or not re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", ""))):
            raise ValueError("invocation is outside the fixed first-attempt policy")
        head = t.validate_controller(ROOT, source, amendment, descriptor, expected)
        run = unique_run(candidate, head, int(env["GITHUB_RUN_ID"]))
        record.update(admitted=True, candidate=candidate, run_id=run["id"], run_attempt=1,
                      workflow_id=run["workflow_id"], controller_commit=head,
                      source_commit=amendment["source_commit"],
                      admission_sha256=amendment["admission_sha256"],
                      amendment_sha256=expected, descriptor_sha256=c.digest(descriptor),
                      image_os=env.get("ImageOS"), image_version=env.get("ImageVersion"))
        c.save(Path(output).parent / "transport-amendment.json", amendment)
        c.save(Path(output).parent / "input-descriptor.json", descriptor)
        return record
    except BaseException as error:
        record["error_type"] = type(error).__name__
        raise
    finally:
        c.save(output, record)


def admitted_context(env):
    amendment, descriptor, expected = environment_records(env)
    context = c.read(Path(env["EVIDENCE_ROOT"]) / "hosted-context.json")
    if not re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", "")):
        raise ValueError("input preparation run identity is invalid")
    required = dict(schema=SCHEMA, admitted=True, release_ready=False, candidate=descriptor["candidate"],
                    run_id=int(env["GITHUB_RUN_ID"]), run_attempt=1,
                    controller_commit=amendment["controller_commit"], source_commit=amendment["source_commit"],
                    admission_sha256=amendment["admission_sha256"], amendment_sha256=expected,
                    descriptor_sha256=c.digest(descriptor))
    if (any(context.get(key) != value or type(context.get(key)) is not type(value)
            for key, value in required.items())
            or type(context.get("workflow_id")) is not int or context["workflow_id"] <= 0
            or env.get("GITHUB_RUN_ATTEMPT") != "1" or env.get("GITHUB_REPOSITORY") != t.REPO
            or env.get("GITHUB_SHA") != amendment["controller_commit"]
            or env.get("GITHUB_REF") != "refs/heads/" + t.branch(descriptor["candidate"])):
        raise ValueError("input preparation lacks an admitted first-run context")
    return amendment, descriptor, expected


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("object redirects are forbidden")


def download_ciphertext(url, descriptor, output, opener=None):
    t.validate_url(url, descriptor)
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(url, timeout=30) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("object response is not the pinned direct download")
            length = response.headers.get("Content-Length")
            if length is not None and length != str(descriptor["ciphertext"]["bytes"]):
                raise ValueError("object response length differs")
            return t.copy_verified(response, output, descriptor["ciphertext"])
    except Exception:
        # urllib errors can contain a credential-bearing URL.
        raise ValueError("object transfer failed; keep the first partial file; never retry") from None


def fetch_input(output, env=None):
    env = os.environ if env is None else env
    _amendment, descriptor, _expected = admitted_context(env)
    if descriptor["schema"] == t.GITHUB_DESCRIPTOR_SCHEMA:
        if env.get("INPUT_OBJECT_URL") or not env.get("GH_TOKEN"):
            raise ValueError("GitHub input requires its read-only token and no direct URL")
    else:
        t.validate_url(env["INPUT_OBJECT_URL"], descriptor)
    log = Path(output).with_name(Path(output).name + ".fetch.log")
    with log.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        outcome = owned_process.run(
            [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
             "fetch-worker", "--output", str(output)], timeout=FETCH_SECONDS, stdout=stream)
    if outcome.returncode != 0:
        raise ValueError("bounded object transfer failed; no retry")
    if handoff.regular(output, t.MAX_CIPHER) != descriptor["ciphertext"]:
        raise ValueError("fetched ciphertext differs")
    return dict(status="verified_ciphertext", release_ready=False)


def prepare_input(source, archive, output, env=None):
    env = os.environ if env is None else env
    if any(env.get(key) for key in ("INPUT_OBJECT_URL", "INPUT_IDENTITY", "GH_TOKEN", "GITHUB_TOKEN")):
        raise ValueError("transport credentials must be removed before frozen verification")
    amendment, descriptor, expected = admitted_context(env)
    t.validate_controller(ROOT, source, amendment, descriptor, expected)
    t.unpack_input(archive, output, descriptor)
    admission = t.verify_input_tree(output, amendment, descriptor)
    if c.source_identity(source) != admission["source"]:
        raise ValueError("frozen execution source differs from admission")
    evidence = Path(env["EVIDENCE_ROOT"])
    stdout, stderr = evidence / "frozen-input-verifier.stdout", evidence / "frozen-input-verifier.stderr"
    record = dict(status="started", admission_sha256=c.digest(admission),
                  source_commit=amendment["source_commit"], release_ready=False)
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            os.fchmod(out.fileno(), 0o600)
            os.fchmod(err.fileno(), 0o600)
            result = owned_process.run(
                [sys.executable, "-I", "-S", "-B", "tools/byte_release_hosted.py", "verify-input",
                 "--input", str(Path(output).resolve()), "--candidate", str(admission["candidate"]),
                 "--admission-sha256", c.digest(admission)],
                cwd=source, stdout=out, stderr=err, timeout=300)
        record.update(status="exited", returncode=result.returncode)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        c.save(evidence / "frozen-input-verifier-process.json", record)
    if result.returncode != 0 or stderr.stat().st_size:
        raise ValueError("original frozen input verifier failed")
    receipt = t.read_record(stdout)
    expected_receipt = dict(admission_sha256=c.digest(admission), release_ready=False,
                            wheel=str(Path(output).resolve() / "wheel" / admission["wheels"]["macos"]["filename"]))
    if receipt != expected_receipt or receipt.get("release_ready") is not False:
        raise ValueError("frozen input verification receipt differs")
    return receipt


def recover_result(directory, archive, expected_sha256):
    directory = Path(directory)
    admission, claim = load_claim(directory)
    binding = c.read(directory / "hosted-metal.run.json")
    if (binding.get("schema") != SCHEMA or binding.get("release_ready") is not False
            or binding.get("admission_sha256") != c.digest(admission)
            or binding.get("amendment_sha256") != claim["amendment_sha256"]
            or binding.get("controller_commit") != claim["controller_commit"]
            or type(binding.get("workflow_id")) is not int or binding["workflow_id"] != claim["workflow_id"]
            or type(binding.get("run_attempt")) is not int or binding["run_attempt"] != 1
            or type(binding.get("run_id")) is not int or binding["run_id"] <= 0):
        raise ValueError("return binding differs")
    run = unique_run(admission["candidate"], claim["controller_commit"], binding["run_id"])
    if run["workflow_id"] != claim["workflow_id"] or run.get("status") != "completed":
        raise ValueError("bound workflow is changed or still live; only observe")
    if any((directory / name).exists() for name in ("hosted-metal", "hosted-metal.start.json", "hosted-metal.result.json")):
        raise ValueError("first hosted outcome already exists")
    raw = directory / f"hosted-return-{run['id']}"
    handoff.unpack(archive, raw, expected_sha256, "result", c.digest(admission))
    remote = raw / "ledger" / f"candidate-{admission['candidate']}"
    outcome, reason = None, "missing_or_invalid_remote_acceptance"
    try:
        context = c.read(raw / "hosted-context.json")
        returned_amendment = t.read_record(raw / "transport-amendment.json")
        returned_descriptor = t.read_record(raw / "input-descriptor.json")
        t.validate_amendment(returned_amendment, claim["amendment_sha256"], returned_descriptor)
        returned_admission = c.validate_admission(c.read(remote / "admission.json"))
        required = dict(schema=SCHEMA, admitted=True, release_ready=False, candidate=admission["candidate"],
                        run_id=run["id"], run_attempt=1, workflow_id=claim["workflow_id"],
                        controller_commit=claim["controller_commit"],
                        source_commit=admission["source"]["commit"], admission_sha256=c.digest(admission),
                        amendment_sha256=claim["amendment_sha256"], descriptor_sha256=c.digest(claim["descriptor"]))
        if (any(context.get(key) != value or type(context.get(key)) is not type(value)
                for key, value in required.items())
                or c.digest(returned_descriptor) != c.digest(claim["descriptor"])
                or c.digest(returned_admission) != c.digest(admission)):
            raise ValueError("return context is not bound to the approved transport")
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
               admission_sha256=c.digest(admission), origin="coordinator_observed_terminal_run",
               run_id=run["id"], release_ready=False))
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
                  amendment_sha256=claim["amendment_sha256"], archive_sha256=expected_sha256,
                  status=outcome["status"], release_ready=False)
    c.save(directory / "hosted-metal.import.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    submit = commands.add_parser("dispatch")
    for key in ("ledger", "controller", "source", "amendment"):
        submit.add_argument("--" + key, type=Path, required=True)
    submit.add_argument("--amendment-sha256", required=True)
    submit.add_argument("--recipient", required=True)
    submit.add_argument("--execute", action="store_true")
    bind = commands.add_parser("bind-run")
    bind.add_argument("--candidate-directory", type=Path, required=True)
    admit = commands.add_parser("admit-run")
    admit.add_argument("--source", type=Path, required=True)
    admit.add_argument("--output", type=Path, required=True)
    for operation in ("fetch-input", "fetch-worker"):
        commands.add_parser(operation).add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-input")
    for key in ("source", "input", "output"):
        prepare.add_argument("--" + key, type=Path, required=True)
    pack = commands.add_parser("pack-result")
    for key in ("evidence", "input", "output"):
        pack.add_argument("--" + key, type=Path, required=True)
    pack.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    pack.add_argument("--admission-sha256", required=True)
    recover = commands.add_parser("recover-result")
    for key in ("candidate-directory", "archive"):
        recover.add_argument("--" + key, type=Path, required=True)
    recover.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if args.operation == "dispatch":
        value = dispatch(args.ledger, args.controller, args.source, t.read_record(args.amendment),
                         args.amendment_sha256, args.recipient, args.execute)
    elif args.operation == "bind-run":
        value = bind_run(args.candidate_directory)
    elif args.operation == "admit-run":
        value = admit_run(args.source, args.output)
    elif args.operation == "fetch-input":
        value = fetch_input(args.output)
    elif args.operation == "fetch-worker":
        _amendment, descriptor, _expected = admitted_context(os.environ)
        if descriptor["schema"] == t.GITHUB_DESCRIPTOR_SCHEMA:
            from byte_release_github_asset import download
            if os.environ.get("INPUT_OBJECT_URL"):
                raise ValueError("GitHub input cannot fall back to a direct URL")
            value = download(descriptor, os.environ.get("GH_TOKEN"), args.output)
        else:
            value = download_ciphertext(os.environ["INPUT_OBJECT_URL"], descriptor, args.output)
    elif args.operation == "prepare-input":
        value = prepare_input(args.source, args.input, args.output)
    elif args.operation == "pack-result":
        value = old.pack_result(args.evidence, args.input, args.candidate, args.output, args.admission_sha256)
    else:
        value = recover_result(args.candidate_directory, args.archive, args.sha256)
    print(c.encoded(value).decode())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Never include exception text that may echo a signed URL or credential.
        print(c.encoded(dict(error_type=type(error).__name__, release_ready=False)).decode(), file=sys.stderr)
        raise SystemExit(1)
