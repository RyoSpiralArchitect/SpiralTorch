"""One CPU-only small-object probe, never a candidate qualification runner."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_github_asset as github
import owned_process

t, c = github.t, github.t.c
ROOT = Path(__file__).resolve().parents[1]
BRANCH = "spiralreality/byte-release-github-preflight-20261030"
SOURCE = "e9294fc6648758da60d16ebea70feaf79a1cf994"
TEXT = b"SpiralTorch GitHub read-only synthetic transport preflight. Not a candidate.\n"
SYNTHETIC = hashlib.sha256(b"spiraltorch.github-reader.synthetic-not-admitted.v1").hexdigest()
SCHEMA = "spiraltorch.github_reader.preflight.v1"
FETCH_SECONDS = 300


def checked_context(env=None):
    env = os.environ if env is None else env
    descriptor = t.read_record(ROOT / t.DESCRIPTOR)
    t.validate_descriptor(descriptor)
    if (descriptor["schema"] != t.GITHUB_DESCRIPTOR_SCHEMA
            or descriptor["candidate"] != 1
            or descriptor["admission_sha256"] != SYNTHETIC
            or descriptor["source_commit"] != SOURCE
            or descriptor["plaintext"] != {"bytes": len(TEXT), "sha256": hashlib.sha256(TEXT).hexdigest()}
            or descriptor["ciphertext"]["bytes"] > 64 * 1024):
        raise ValueError("only the small synthetic descriptor is permitted")
    expected = env.get("EXPECTED_PREFLIGHT_HEAD")
    if (not c.is_hash(expected, 40) or env.get("GITHUB_SHA") != expected
            or env.get("GITHUB_REPOSITORY") != t.REPO
            or env.get("GITHUB_REF") != "refs/heads/" + BRANCH
            or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or env.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch(r"[1-9][0-9]{0,18}", env.get("GITHUB_RUN_ID", ""))
            or env.get("EXPECTED_DESCRIPTOR_SHA256") != c.digest(descriptor)
            or not env.get("GH_TOKEN")
            or any(env.get(key) for key in ("GITHUB_TOKEN", "INPUT_IDENTITY", "INPUT_OBJECT_URL"))):
        raise ValueError("unbound preflight or unexpected credential")
    source = c.source_identity(ROOT)
    if source["commit"] != expected:
        raise ValueError("checkout differs from the pinned preflight head")
    context = dict(schema=SCHEMA, preflight_only=True, release_ready=False,
                   head=expected, tree=source["tree"], descriptor_sha256=c.digest(descriptor),
                   run_id=int(env["GITHUB_RUN_ID"]), run_attempt=1,
                   branch=BRANCH, fetch_seconds=FETCH_SECONDS)
    return context, descriptor


def fetch_worker(output):
    context, descriptor = checked_context()
    if c.digest(t.read_record(output / "claim.json")) != c.digest(context):
        raise ValueError("worker does not own the first preflight claim")
    transfer = github.download(descriptor, os.environ["GH_TOKEN"], output / "input.age")
    c.save(output / "verified.json", dict(context=context, transfer=transfer))


def run(output):
    context, descriptor = checked_context()
    output = Path(output).absolute()
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("preflight output must not follow links")
    output.mkdir(mode=0o700, exist_ok=False)
    c.save(output / "claim.json", context)
    result = dict(context=context, status="failed", error_type=None, worker_exit_code=None)
    try:
        with (output / "worker.log").open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            child = owned_process.run(
                [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                 "fetch-worker", "--output", str(output)],
                timeout=FETCH_SECONDS, stdout=stream)
        result["worker_exit_code"] = child.returncode
        if child.returncode != 0:
            raise ValueError("first fetch failed; do not retry")
        if t.v1.regular(output / "input.age", 64 * 1024) != descriptor["ciphertext"]:
            raise ValueError("downloaded synthetic bytes differ")
        verified = t.read_record(output / "verified.json")
        t.exact_keys(verified, ("context", "transfer"))
        if c.digest(verified["context"]) != c.digest(context):
            raise ValueError("worker receipt is unbound")
        transfer = verified["transfer"]
        t.exact_keys(transfer, ("status", "repository_id", "release_id", "asset_id", "bytes", "sha256", "path", "release_ready"))
        expected = dict(status="verified_ciphertext", **descriptor["ciphertext"],
                        **{key: descriptor["object"][key] for key in ("repository_id", "release_id", "asset_id")},
                        path=transfer["path"], release_ready=False)
        if (transfer["path"] not in ("direct_api", "single_cdn_redirect")
                or c.digest(transfer) != c.digest(expected)):
            raise ValueError("worker transfer identity differs")
        result.update(status="verified_small_ciphertext", transfer=transfer)
    except BaseException as error:
        result["error_type"] = type(error).__name__
        c.save(output / "result.json", result)
        raise
    c.save(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "fetch-worker"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.operation == "fetch-worker":
            fetch_worker(args.output)
        else:
            run(args.output)
    except Exception as error:
        print(c.encoded(dict(preflight_only=True, release_ready=False,
                             error_type=type(error).__name__)).decode(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
