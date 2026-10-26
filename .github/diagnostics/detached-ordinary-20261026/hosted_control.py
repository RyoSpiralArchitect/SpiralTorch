"""One campaign, one explicit dispatch. No build, native import or GPU call."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hosted_transport as transport  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO = "RyoSpiralArchitect/SpiralTorch"
BRANCH = "spiralreality/detached-ordinary-hosted-20261026"
BASE = "c1f63f8864e558c8ca82b0ce766e574ad2d59423"
WORKFLOW = ".github/workflows/byte_study_replay.yml"
DIRECTORY = ".github/diagnostics/detached-ordinary-20261026"
PACKET = DIRECTORY + "/input.zip.age"
SCHEMA = "spiraltorch.detached_ordinary.control.v1"
INPUT_SHA = "3640ac2af74240653e2ed219f5770b1c70e865d07bd64fe927bb7fb49bdffb09"
REQUEST_SHA = "b18f2e92fa1b87c5254ad865e07102b26a9df7df870ef342eb73d0b3a5bfdd8f"
SECRET_NAME = "ST_DETACHED_ORDINARY_20261026_INPUT_IDENTITY"
TEMPLATE = "hosted_workflow.template.yml"
FILES = {
    WORKFLOW,
    PACKET,
    *(
        DIRECTORY + "/" + name
        for name in ("hosted_control.py", "hosted_transport.py", TEMPLATE)
    ),
}
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
RECIPIENT = re.compile(r"age1[0-9a-z]{58}\Z")


def read_json(path):
    return transport.decode(transport.read(path))


def save(path, value):
    path = Path(path)
    transport.no_links(path)
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(transport.encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def api(endpoint, payload=None):
    command = ["gh", "api", f"repos/{REPO}/{endpoint}"]
    if payload is not None:
        command += ["--method", "POST", "--input", "-"]
    response = subprocess.run(
        command,
        input=None if payload is None else transport.encoded(payload),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if response.returncode or len(response.stdout) > 4 * 1024 * 1024:
        raise RuntimeError(
            "GitHub request failed or exceeded the bound; never resubmit"
        )
    return transport.decode(response.stdout) if response.stdout.strip() else None


def plan(value):
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema",
            "campaign",
            "input_plaintext_sha256",
            "input_ciphertext_sha256",
            "result_recipient",
        }
        or value["schema"] != SCHEMA
        or value["campaign"] != transport.CAMPAIGN
        or value["input_plaintext_sha256"] != INPUT_SHA
        or type(value["input_ciphertext_sha256"]) is not str
        or not transport.SHA.fullmatch(value["input_ciphertext_sha256"])
        or type(value["result_recipient"]) is not str
        or not RECIPIENT.fullmatch(value["result_recipient"])
    ):
        raise ValueError("not the fixed public launch plan")
    return value


def render(public_plan):
    public_plan = plan(public_plan)
    text = transport.read(ROOT / TEMPLATE).decode()
    for marker, key in (
        ("__INPUT_CIPHERTEXT_SHA256__", "input_ciphertext_sha256"),
        ("__INPUT_PLAINTEXT_SHA256__", "input_plaintext_sha256"),
        ("__RESULT_RECIPIENT__", "result_recipient"),
    ):
        if text.count(marker) != 1:
            raise ValueError("template placeholder count differs")
        text = text.replace(marker, public_plan[key])
    if "__INPUT_" in text or "__RESULT_" in text:
        raise ValueError("unrendered placeholder")
    return text.encode()


def git(root, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    return result.stdout.decode().strip()


def controller_identity(root, public_plan):
    root = Path(root)
    public_plan = plan(public_plan)
    head = git(root, "rev-parse", "HEAD")
    if (
        not SHA40.fullmatch(head)
        or git(root, "rev-parse", "HEAD^") != BASE
        or git(root, "status", "--porcelain", "--untracked-files=all")
        or set(
            git(root, "diff", "--name-only", "--no-renames", BASE, head).splitlines()
        )
        != FILES
    ):
        raise ValueError("controller is not the narrow clean diagnostic branch")
    for name in ("hosted_control.py", "hosted_transport.py", TEMPLATE):
        if transport.read(root / DIRECTORY / name) != transport.read(ROOT / name):
            raise ValueError("controller differs from the inspected local tools")
    if transport.read(root / WORKFLOW) != render(public_plan):
        raise ValueError("workflow is not the exact rendering")
    if (
        transport.sha(transport.read(root / PACKET))
        != public_plan["input_ciphertext_sha256"]
    ):
        raise ValueError("controller ciphertext differs")
    return head


def campaign_runs(api_fn=api):
    return api_fn(
        "actions/workflows/byte_study_replay.yml/runs?event=workflow_dispatch&branch="
        + quote(BRANCH, safe="")
        + "&per_page=2"
    )


def unique_run(head, expected_id=None, api_fn=api):
    listing = campaign_runs(api_fn)
    rows = listing.get("workflow_runs", [])
    if (
        type(listing.get("total_count")) is not int
        or listing["total_count"] != 1
        or len(rows) != 1
    ):
        raise ValueError("no unique first run; observe only, never resubmit")
    row = rows[0]
    if (
        type(row.get("id")) is not int
        or row["id"] <= 0
        or type(row.get("run_attempt")) is not int
        or row["run_attempt"] != 1
        or row.get("event") != "workflow_dispatch"
        or row.get("head_sha") != head
        or row.get("head_branch") != BRANCH
        or (expected_id is not None and row["id"] != expected_id)
    ):
        raise ValueError("run identity or attempt differs")
    return row


def validate_claim(value):
    if (
        value.get("schema") != SCHEMA
        or value.get("campaign") != transport.CAMPAIGN
        or value.get("branch") != BRANCH
        or value.get("workflow") != WORKFLOW
        or value.get("release_ready") is not False
        or type(value.get("controller_commit")) is not str
        or not SHA40.fullmatch(value["controller_commit"])
    ):
        raise ValueError("foreign dispatch claim")
    plan(value["plan"])
    return value


def dispatch(state, controller, public_plan, execute=False, api_fn=api):
    state = Path(state)
    transport.no_links(state)
    if any(
        (state / name).exists()
        for name in (
            "dispatch-claim.json",
            "dispatch-response.json",
            "run-binding.json",
        )
    ):
        raise ValueError("dispatch already claimed; observe the first outcome")
    public_plan = plan(public_plan)
    head = controller_identity(controller, public_plan)
    if api_fn("git/ref/heads/" + BRANCH).get("object", {}).get("sha") != head:
        raise ValueError("remote branch differs from inspected controller")
    previous = campaign_runs(api_fn)
    if (
        type(previous.get("total_count")) is not int
        or previous["total_count"] != 0
        or previous.get("workflow_runs") != []
    ):
        raise ValueError("this campaign already has a manual run")
    if api_fn("actions/secrets/" + SECRET_NAME).get("name") != SECRET_NAME:
        raise ValueError("fresh campaign input secret is unavailable")
    claim = dict(
        schema=SCHEMA,
        campaign=transport.CAMPAIGN,
        branch=BRANCH,
        workflow=WORKFLOW,
        controller_commit=head,
        plan=public_plan,
        release_ready=False,
    )
    if not execute:
        return dict(claim, dispatched=False)
    state.mkdir(mode=0o700, exist_ok=True)
    save(state / "dispatch-claim.json", dict(claim, claimed_unix=time.time()))
    try:
        api_fn(
            "actions/workflows/byte_study_replay.yml/dispatches",
            {"ref": BRANCH, "inputs": {"harness_commit": head}},
        )
    except BaseException as error:
        save(
            state / "dispatch-response.json",
            dict(
                outcome="ambiguous_do_not_retry",
                error_type=type(error).__name__,
                release_ready=False,
            ),
        )
        raise
    save(
        state / "dispatch-response.json", dict(outcome="submitted", release_ready=False)
    )
    return dict(claim, dispatched=True)


def bind_run(state, api_fn=api):
    state = Path(state)
    claim = validate_claim(read_json(state / "dispatch-claim.json"))
    row = unique_run(claim["controller_commit"], api_fn=api_fn)
    binding = dict(
        schema=SCHEMA,
        campaign=transport.CAMPAIGN,
        run_id=row["id"],
        run_attempt=1,
        controller_commit=row["head_sha"],
        release_ready=False,
    )
    path = state / "run-binding.json"
    if path.exists():
        if transport.encoded(read_json(path)) != transport.encoded(binding):
            raise ValueError("an earlier different binding is already saved")
    else:
        save(path, binding)
    return dict(binding, status=row.get("status"), conclusion=row.get("conclusion"))


def admit(controller, output, env=None, api_fn=api):
    env = os.environ if env is None else env
    output = Path(output)
    with output.with_suffix(".claim").open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
    record = dict(
        schema=SCHEMA, campaign=transport.CAMPAIGN, admitted=False, release_ready=False
    )
    try:
        head, run_id = env["GITHUB_SHA"], env["GITHUB_RUN_ID"]
        public_plan = plan(
            dict(
                schema=SCHEMA,
                campaign=transport.CAMPAIGN,
                input_plaintext_sha256=env["INPUT_PLAINTEXT_SHA256"],
                input_ciphertext_sha256=env["INPUT_CIPHERTEXT_SHA256"],
                result_recipient=env["RESULT_RECIPIENT"],
            )
        )
        if (
            env["GITHUB_REPOSITORY"] != REPO
            or env["GITHUB_EVENT_NAME"] != "workflow_dispatch"
            or env["GITHUB_REF"] != "refs/heads/" + BRANCH
            or env["GITHUB_RUN_ATTEMPT"] != "1"
            or env["EXPECTED_HARNESS"] != head
            or not SHA40.fullmatch(head)
            or re.fullmatch(r"[1-9][0-9]*", run_id) is None
            or env["RUNNER_OS"] != "macOS"
            or env["RUNNER_ARCH"] != "ARM64"
            or not env.get("ImageOS")
            or not env.get("ImageVersion")
        ):
            raise ValueError("hosted invocation is outside the one-shot scope")
        if controller_identity(controller, public_plan) != head:
            raise ValueError("checked-out controller differs")
        unique_run(head, int(run_id), api_fn)
        record.update(
            admitted=True,
            run_id=int(run_id),
            run_attempt=1,
            controller_commit=head,
            branch=BRANCH,
            plan=public_plan,
            image_os=env["ImageOS"],
            image_version=env["ImageVersion"],
            runner_os=env["RUNNER_OS"],
            runner_arch=env["RUNNER_ARCH"],
            diagnostic_timeout_seconds=600,
            job_timeout_minutes=30,
        )
        return record
    except BaseException as error:
        record["error_type"] = type(error).__name__
        raise
    finally:
        save(output, record)


def collect(bundle, evidence):
    """Preserve existing first-attempt files, even when the diagnostic failed."""
    evidence = Path(evidence)
    transport.no_links(evidence)
    source = Path(bundle) / "detached-ordinary-localization-v1-attempt"
    record = dict(
        schema=SCHEMA,
        campaign=transport.CAMPAIGN,
        diagnostic_present=source.exists(),
        release_ready=False,
    )
    if source.exists():
        paths = transport.inventory(source)
        destination = evidence / "diagnostic"
        destination.mkdir(mode=0o700)
        hashes, total = {}, 0
        for name, path in paths.items():
            raw = transport.read(path)
            total += len(raw)
            if total > transport.MAX_BYTES:
                raise ValueError("diagnostic evidence exceeds bound")
            target = destination / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(raw)
            hashes[name] = transport.sha(raw)
        if transport.inventory(source) != paths or any(
            transport.sha(transport.read(path)) != hashes[name]
            for name, path in paths.items()
        ):
            raise ValueError("diagnostic changed while collecting")
        record["files"] = hashes
    save(evidence / "collection.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("preflight", "dispatch-once"):
        child = sub.add_parser(name)
        child.add_argument("--state", type=Path, required=True)
        child.add_argument("--controller", type=Path, required=True)
        child.add_argument("--plan", type=Path, required=True)
    child = sub.add_parser("bind-run")
    child.add_argument("--state", type=Path, required=True)
    child = sub.add_parser("admit")
    child.add_argument("--controller", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    child = sub.add_parser("collect")
    child.add_argument("--input", type=Path, required=True)
    child.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.operation in ("preflight", "dispatch-once"):
        result = dispatch(
            args.state,
            args.controller,
            read_json(args.plan),
            args.operation == "dispatch-once",
        )
    elif args.operation == "bind-run":
        result = bind_run(args.state)
    elif args.operation == "admit":
        result = admit(args.controller, args.output)
    else:
        result = collect(args.input, args.evidence)
    print(transport.encoded(result).decode(), end="")


if __name__ == "__main__":
    main()
