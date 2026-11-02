"""First-attempt and environment admission. No package import or device access."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess

BRANCH = "spiralreality/embedding-trio-hosted-20261103"
EXPECTED = {
    "repository": "RyoSpiralArchitect/SpiralTorch", "event": "workflow_dispatch",
    "ref": "refs/heads/" + BRANCH, "run_attempt": "1",
    "runner": "github-hosted", "runner_os": "macOS", "runner_arch": "ARM64",
    "image_os": "macos26", "image_version": "20261026.0455.2",
    "machine": "arm64", "macos": "26.6.2", "os_build": "25G83", "python": "3.12.10",
}
ADAPTER = ('AdapterInfo { name: "Apple Paravirtual device", vendor: 0, device: 0, '
           'device_type: IntegratedGpu, driver: "", driver_info: "", backend: Metal }')


def inspect():
    def output(*args):
        return subprocess.check_output(args, text=True, timeout=5).strip()
    fields = {k: os.environ.get(v, "") for k, v in {
        "repository": "GITHUB_REPOSITORY", "event": "GITHUB_EVENT_NAME", "ref": "GITHUB_REF",
        "run_id": "GITHUB_RUN_ID", "run_attempt": "GITHUB_RUN_ATTEMPT", "runner": "TRIO_RUNNER_KIND",
        "runner_os": "RUNNER_OS", "runner_arch": "RUNNER_ARCH", "image_os": "ImageOS", "image_version": "ImageVersion",
        "dispatch_sha": "TRIO_EXPECTED_SHA", "github_sha": "GITHUB_SHA"}.items()}
    return dict(fields, machine=platform.machine(), macos=output("/usr/bin/sw_vers", "-productVersion"),
                os_build=output("/usr/bin/sw_vers", "-buildVersion"), python=platform.python_version(),
                checkout_sha=output("git", "rev-parse", "HEAD"))


def mismatches(fields):
    bad = [k for k, v in EXPECTED.items() if fields.get(k) != v]
    if not re.fullmatch(r"[1-9][0-9]*", fields.get("run_id", "")):
        bad.append("run_id")
    sha = fields.get("github_sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or sha != fields.get("dispatch_sha") or sha != fields.get("checkout_sha"):
        bad.append("reviewed_commit")
    return bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fields = inspect()
    bad = mismatches(fields)
    with args.output.open("x") as stream:
        json.dump({"observed": fields, "mismatches": bad, "gpu_calls": 0}, stream, indent=2)
    if bad:
        raise SystemExit("preflight stopped before GPU: " + ", ".join(bad))
