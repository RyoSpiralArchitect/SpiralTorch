"""One small read-only Actions consumer preflight; never a GPU qualification."""
import argparse
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_actions_download as download
import owned_process

a, g = download.a, download.g
t, c = a.t, a.t.c
ROOT = Path(__file__).resolve().parents[1]
BRANCH = "spiralreality/byte-release-actions-probe-media-20261030"
PIN = ".github/byte-release-actions-pin.json"
SCHEMA = "spiraltorch.byte_release.actions_probe.v1"
FETCH_SECONDS = 300


def checked_context(env=None):
    env = os.environ if env is None else env
    pin = a.validate_pin(t.read_record(ROOT / PIN))
    head = env.get("EXPECTED_READER_HEAD")
    if (pin["artifact"]["bytes"] > 65536 or pin["ciphertext"]["bytes"] > 65536
            or not c.is_hash(head, 40) or env.get("GITHUB_SHA") != head
            or env.get("GITHUB_REPOSITORY") != t.REPO or env.get("GITHUB_REF") != "refs/heads/" + BRANCH
            or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or env.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch(r"[1-9][0-9]{0,18}", env.get("GITHUB_RUN_ID", ""))
            or env.get("EXPECTED_PIN_SHA256") != c.digest(pin) or not env.get("GH_TOKEN")
            or any(env.get(key) for key in ("GITHUB_TOKEN", "INPUT_IDENTITY", "INPUT_OBJECT_URL"))):
        raise ValueError("unbound small read-only preflight")
    source = c.source_identity(ROOT)
    if source["commit"] != head:
        raise ValueError("reader checkout differs")
    context = dict(schema=SCHEMA, head=head, tree=source["tree"], pin_sha256=c.digest(pin),
        run_id=int(env["GITHUB_RUN_ID"]), run_attempt=1, branch=BRANCH,
        workflow_id=pin["bridge"]["workflow_id"], fetch_seconds=FETCH_SECONDS, release_ready=False)
    return context, pin


def unique_invocation(context, pin, token):
    opener = g.urllib.request.build_opener(g.urllib.request.ProxyHandler({}), g._NoRedirect(), g._RawResponses())
    path = (f"{g.API_ROOT}/repos/{t.REPO}/actions/workflows/{context['workflow_id']}/runs"
            f"?event=workflow_dispatch&branch={BRANCH}&per_page=2")
    listing = g._metadata(opener, path, token)
    if (type(listing.get("total_count")) is not int or listing["total_count"] != 1
            or type(listing.get("workflow_runs")) is not list or len(listing["workflow_runs"]) != 1):
        raise ValueError("consumer must be the unique first submission")
    row = listing["workflow_runs"][0]
    a._expect(row, dict(id=context["run_id"], run_attempt=1, head_sha=context["head"], head_branch=BRANCH,
        workflow_id=context["workflow_id"], path=".github/workflows/" + t.WORKFLOW,
        event="workflow_dispatch", status="in_progress", conclusion=None))
    for key in ("repository", "head_repository"):
        a._expect(row.get(key), dict(id=pin["repository_id"], full_name=t.REPO))


def expected_receipt(pin):
    return dict(pin_sha256=c.digest(pin), bridge_run_id=pin["bridge"]["run_id"], bridge_run_attempt=1,
        artifact_id=pin["artifact"]["id"], release_ready=False, status="verified_pinned_ciphertext",
        archive_sha256=pin["artifact"]["sha256"],
        ciphertext={key: pin["ciphertext"][key] for key in ("bytes", "sha256")})


def fetch_worker(output):
    context, pin = checked_context()
    if c.digest(t.read_record(output / "claim.json")) != c.digest(context):
        raise ValueError("reader worker does not own this first claim")
    unique_invocation(context, pin, os.environ["GH_TOKEN"])
    receipt = download.download(pin, os.environ["GH_TOKEN"], output / "input.zip", output / "input.age")
    c.save(output / "download.json", dict(context=context, transfer=receipt))


def run(output):
    context, pin = checked_context()
    output = Path(output).absolute()
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("reader output must not follow links")
    output.mkdir(mode=0o700, exist_ok=False)
    c.save(output / "claim.json", context)
    result = dict(context=context, status="failed", error_type=None, worker_exit_code=None, release_ready=False)
    try:
        with (output / "worker.log").open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            child = owned_process.run([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                "fetch-worker", "--output", str(output)], timeout=FETCH_SECONDS, stdout=stream)
        result["worker_exit_code"] = child.returncode
        if child.returncode != 0:
            raise ValueError("read-only first fetch failed; never retry")
        for name, value in (("input.zip", pin["artifact"]), ("input.age", pin["ciphertext"])):
            if t.v1.regular(output / name, 65536) != {key: value[key] for key in ("bytes", "sha256")}:
                raise ValueError("reader bytes differ from the fixed pin")
        receipt = t.read_record(output / "download.json")
        expected = dict(context=context, transfer=expected_receipt(pin))
        if c.digest(receipt) != c.digest(expected):
            raise ValueError("reader receipt differs")
        result.update(status="verified_read_only_ciphertext", transfer=receipt["transfer"])
    except BaseException as error:
        result["error_type"] = type(error).__name__
        raise
    finally:
        c.save(output / "result.json", result)
    return result


def render():
    return '''name: Read-only Actions Input Preflight (CPU only)
on:
  workflow_dispatch:
    inputs:
      expected_head:
        description: "Exact reviewed read-only consumer commit"
        type: string
        required: true
      pin_sha256:
        description: "Canonical digest of the fixed producer artifact pin"
        type: string
        required: true
permissions:
  contents: read
  actions: read
concurrency:
  group: byte-release-actions-probe
  cancel-in-progress: false
defaults:
  run:
    shell: bash --noprofile --norc -e -o pipefail {0}
jobs:
  read-only-consumer:
    runs-on: macos-26
    timeout-minutes: 15
    steps:
      - name: Guard the first read-only invocation
        env:
          EXPECTED_HEAD: ${{ inputs.expected_head }}
        run: |
          test "$GITHUB_RUN_ATTEMPT" = 1
          test "$GITHUB_EVENT_NAME" = workflow_dispatch
          test "$GITHUB_REF" = refs/heads/spiralreality/byte-release-actions-probe-media-20261030
          test "$GITHUB_SHA" = "$EXPECTED_HEAD"
      - uses: actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09 # v5
        with:
          ref: ${{ github.sha }}
          persist-credentials: false
      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6
        with:
          python-version: '3.12.10'
      - name: Verify the pinned Actions input with the read-only token
        env:
          GH_TOKEN: ${{ github.token }}
          EXPECTED_READER_HEAD: ${{ inputs.expected_head }}
          EXPECTED_PIN_SHA256: ${{ inputs.pin_sha256 }}
        run: |
          umask 077
          python -I -S -B tools/byte_release_actions_probe.py run --output "$RUNNER_TEMP/actions-input-probe"
      - name: Retain the first read-only outcome
        if: ${{ always() }}
        uses: actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9 # v7
        with:
          name: actions-input-probe-${{ github.run_id }}-${{ github.run_attempt }}
          path: |
            ${{ runner.temp }}/actions-input-probe/*.json
            ${{ runner.temp }}/actions-input-probe/worker.log
            ${{ runner.temp }}/actions-input-probe/input.zip
            ${{ runner.temp }}/actions-input-probe/input.age
          if-no-files-found: error
          retention-days: 7
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "fetch-worker", "render"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.operation == "render":
            print(render(), end="")
        elif args.operation == "fetch-worker":
            fetch_worker(args.output)
        else:
            run(args.output)
    except Exception as error:
        print(c.encoded(dict(error_type=type(error).__name__, release_ready=False)).decode(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
