"""First-attempt CPU ciphertext staging; never decrypt or execute its contents.

Only the dedicated bridge job may hold contents:write. The reader still performs
GETs only; artifact upload belongs to the separately reviewed pinned action.
"""
import argparse
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_actions_input as a
import byte_release_github_asset as g
import owned_process

t, c = a.t, a.t.c
ROOT = Path(__file__).resolve().parents[1]
SPEC = ".github/byte-release-bridge.json"
SCHEMA = "spiraltorch.byte_release.input_bridge.v1"
FETCH_SECONDS = 300


def validate_spec(spec):
    t.exact_keys(spec, ("schema", "branch", "workflow_id", "artifact_name", "input", "release_ready"))
    if (spec["schema"] != SCHEMA or spec["release_ready"] is not False
            or type(spec["branch"]) is not str or not spec["branch"].startswith(a.BRIDGE_PREFIX)):
        raise ValueError("invalid CPU bridge specification")
    a._name(spec["branch"][len(a.BRIDGE_PREFIX):])
    a._name(spec["artifact_name"])
    a._positive(spec["workflow_id"])
    t.validate_descriptor(spec["input"])
    if spec["input"]["schema"] != t.GITHUB_DESCRIPTOR_SCHEMA:
        raise ValueError("bridge accepts only a pinned unpublished asset")
    return spec


def checked_context(env=None):
    env = os.environ if env is None else env
    spec = validate_spec(t.read_record(ROOT / SPEC))
    head = env.get("EXPECTED_BRIDGE_HEAD")
    if (not c.is_hash(head, 40) or env.get("GITHUB_SHA") != head
            or env.get("GITHUB_REPOSITORY") != t.REPO
            or env.get("GITHUB_REF") != "refs/heads/" + spec["branch"]
            or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or env.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch(r"[1-9][0-9]{0,18}", env.get("GITHUB_RUN_ID", ""))
            or env.get("EXPECTED_BRIDGE_SHA256") != c.digest(spec)
            or not env.get("GH_TOKEN")
            or any(env.get(key) for key in ("GITHUB_TOKEN", "INPUT_IDENTITY", "INPUT_OBJECT_URL"))):
        raise ValueError("unbound bridge or unexpected credential")
    source = c.source_identity(ROOT)
    if source["commit"] != head:
        raise ValueError("bridge checkout differs")
    context = dict(schema=SCHEMA, head=head, tree=source["tree"], spec_sha256=c.digest(spec),
                   run_id=int(env["GITHUB_RUN_ID"]), run_attempt=1, branch=spec["branch"],
                   workflow_id=spec["workflow_id"], fetch_seconds=FETCH_SECONDS, release_ready=False)
    return context, spec


def unique_invocation(context, spec, token, opener=None):
    opener = opener or g.urllib.request.build_opener(g.urllib.request.ProxyHandler({}), g._NoRedirect(), g._RawResponses())
    path = (f"{g.API_ROOT}/repos/{t.REPO}/actions/workflows/{spec['workflow_id']}/runs"
            f"?event=workflow_dispatch&branch={spec['branch']}&per_page=2")
    listing = g._metadata(opener, path, token)
    if (type(listing.get("total_count")) is not int or listing["total_count"] != 1
            or type(listing.get("workflow_runs")) is not list or len(listing["workflow_runs"]) != 1):
        raise ValueError("bridge must be the unique first submission")
    row = listing["workflow_runs"][0]
    a._expect(row, dict(id=context["run_id"], run_attempt=1, head_sha=context["head"],
                       head_branch=spec["branch"], workflow_id=spec["workflow_id"],
                       path=".github/workflows/" + t.WORKFLOW, event="workflow_dispatch",
                       status="in_progress", conclusion=None))
    for key in ("repository", "head_repository"):
        a._expect(row.get(key), dict(id=spec["input"]["object"]["repository_id"], full_name=t.REPO))


def verify_transfer(transfer, descriptor):
    t.exact_keys(transfer, ("status", "repository_id", "release_id", "asset_id", "bytes", "sha256", "path", "release_ready"))
    expected = dict(status="verified_ciphertext", **descriptor["ciphertext"],
                    **{key: descriptor["object"][key] for key in ("repository_id", "release_id", "asset_id")},
                    path=transfer["path"], release_ready=False)
    if transfer["path"] not in ("direct_api", "single_cdn_redirect") or c.digest(transfer) != c.digest(expected):
        raise ValueError("bridge transfer receipt differs")


def fetch_worker(output):
    context, spec = checked_context()
    if c.digest(t.read_record(output / "claim.json")) != c.digest(context):
        raise ValueError("worker does not own the first bridge claim")
    unique_invocation(context, spec, os.environ["GH_TOKEN"])
    descriptor = spec["input"]
    transfer = g.download(descriptor, os.environ["GH_TOKEN"], output / "ciphertext" / descriptor["object"]["asset_name"])
    c.save(output / "verified.json", dict(context=context, transfer=transfer))


def run(output):
    context, spec = checked_context()
    output = Path(output).absolute()
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("bridge output must not follow links")
    output.mkdir(mode=0o700, exist_ok=False)
    (output / "ciphertext").mkdir(mode=0o700)
    c.save(output / "claim.json", context)
    result = dict(context=context, status="failed", error_type=None, worker_exit_code=None, release_ready=False)
    try:
        with (output / "worker.log").open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            child = owned_process.run(
                [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                 "fetch-worker", "--output", str(output)], timeout=FETCH_SECONDS, stdout=stream)
        result["worker_exit_code"] = child.returncode
        if child.returncode != 0:
            raise ValueError("bridge fetch failed; do not retry")
        descriptor = spec["input"]
        if t.v1.regular(output / "ciphertext" / descriptor["object"]["asset_name"], t.MAX_CIPHER) != descriptor["ciphertext"]:
            raise ValueError("bridge ciphertext differs")
        receipt = t.read_record(output / "verified.json")
        t.exact_keys(receipt, ("context", "transfer"))
        if c.digest(receipt["context"]) != c.digest(context):
            raise ValueError("bridge receipt is unbound")
        verify_transfer(receipt["transfer"], descriptor)
        result.update(status="verified_ciphertext_for_staging", transfer=receipt["transfer"])
    except BaseException as error:
        result["error_type"] = type(error).__name__
        raise
    finally:
        c.save(output / "result.json", result)
    return result


def render(spec):
    validate_spec(spec)
    # This workflow is only for a separate, unmergeable CPU bridge branch.
    text = '''name: Byte Release Ciphertext Bridge (CPU only)
on:
  workflow_dispatch:
    inputs:
      expected_head:
        description: "Exact reviewed CPU bridge commit"
        type: string
        required: true
      bridge_sha256:
        description: "Canonical digest of the committed bridge specification"
        type: string
        required: true
permissions:
  contents: read
  actions: read
concurrency:
  group: byte-release-input-bridge
  cancel-in-progress: false
defaults:
  run:
    shell: bash --noprofile --norc -e -o pipefail {0}
jobs:
  ciphertext-bridge:
    runs-on: ubuntu-24.04
    timeout-minutes: 15
    permissions:
      contents: write
      actions: read
    steps:
      - name: Guard the first bridge invocation
        env:
          EXPECTED_HEAD: ${{ inputs.expected_head }}
        run: |
          test "$GITHUB_RUN_ATTEMPT" = 1
          test "$GITHUB_EVENT_NAME" = workflow_dispatch
          test "$GITHUB_REF" = refs/heads/__BRANCH__
          test "$GITHUB_SHA" = "$EXPECTED_HEAD"
      - uses: actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09 # v5
        with:
          ref: ${{ github.sha }}
          persist-credentials: false
      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6
        with:
          python-version: '3.12.10'
      - name: Fetch and verify only the pinned ciphertext
        env:
          GH_TOKEN: ${{ github.token }}
          EXPECTED_BRIDGE_HEAD: ${{ inputs.expected_head }}
          EXPECTED_BRIDGE_SHA256: ${{ inputs.bridge_sha256 }}
        run: |
          umask 077
          python -I -S -B tools/byte_release_input_bridge.py run --output "$RUNNER_TEMP/input-bridge"
      - name: Stage only the successfully verified ciphertext
        uses: actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9 # v7
        with:
          name: __ARTIFACT__
          path: ${{ runner.temp }}/input-bridge/ciphertext/__MEMBER__
          if-no-files-found: error
          compression-level: 0
          retention-days: 7
      - name: Retain the first bridge outcome
        if: ${{ always() }}
        uses: actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9 # v7
        with:
          name: bridge-receipts-${{ github.run_id }}-${{ github.run_attempt }}
          path: |
            ${{ runner.temp }}/input-bridge/*.json
            ${{ runner.temp }}/input-bridge/worker.log
          if-no-files-found: error
          retention-days: 7
'''
    for before, after in (("__BRANCH__", spec["branch"]), ("__ARTIFACT__", spec["artifact_name"]),
                          ("__MEMBER__", spec["input"]["object"]["asset_name"])):
        text = text.replace(before, after)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "fetch-worker", "render"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.operation == "render":
            print(render(t.read_record(args.spec)), end="")
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
