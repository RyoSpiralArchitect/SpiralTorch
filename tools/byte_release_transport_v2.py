"""Versioned transport checks, separate from the frozen numerical admission.

An amendment is data, not human authorization. Its expected digest must come
from the separately approved coordinator record. No routine in this module
imports a native package, launches learning, or changes an admission.
"""
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from urllib.parse import urlsplit
import zipfile

import byte_release_handoff as v1
from byte_release_handoff import contract as c

DESCRIPTOR_SCHEMA = "spiraltorch.byte_release.transport_input.v2"
GITHUB_DESCRIPTOR_SCHEMA = "spiraltorch.byte_release.transport_github_input.v1"
ACTIONS_DESCRIPTOR_SCHEMA = "spiraltorch.byte_release.transport_actions_input.v1"
AMENDMENT_SCHEMA = "spiraltorch.byte_release.transport_amendment.v1"
REPO = "RyoSpiralArchitect/SpiralTorch"
WORKFLOW = "byte_release_qualification.yml"
CONCURRENCY = "byte-release-qualification-20261025"
DESCRIPTOR = ".github/byte-release-input.json"
MAX_INPUT = 128 * 1024 * 1024
MAX_CIPHER = 129 * 1024 * 1024
MAX_RECORD = 64 * 1024
LIMITS = {"input_bytes": MAX_INPUT, "ciphertext_bytes": MAX_CIPHER,
          "expanded_bytes": v1.MAX_TOTAL, "member_bytes": v1.MAX_FILE,
          "member_count": v1.MAX_FILES, "manifest_bytes": 8 * 1024 * 1024}
RUNTIME_FILES = (
    "tools/byte_release_transport_v2.py", "tools/byte_release_hosted_v2.py",
    "tools/byte_release_transport_workflow_v2.py",
)
GITHUB_RUNTIME_FILE = "tools/byte_release_github_asset.py"
ACTIONS_RUNTIME_FILES = (GITHUB_RUNTIME_FILE, "tools/byte_release_actions_input.py",
                         "tools/byte_release_actions_download.py")
CHANGE_ALLOWLIST = frozenset((
    DESCRIPTOR, ".github/workflows/" + WORKFLOW,
    *RUNTIME_FILES,
    GITHUB_RUNTIME_FILE, "tools/test_byte_release_github_asset.py",
    *ACTIONS_RUNTIME_FILES, "tools/test_byte_release_actions_input.py",
    "tools/test_byte_release_actions_download.py", "tools/test_byte_release_actions_hosted.py",
    "docs/ops/byte-release-actions-download.md",
    "tools/test_byte_release_transport_v2.py", "tools/test_byte_release_hosted_v2.py",
    "docs/ops/byte-release-transport-v2.md",
))


def branch(candidate):
    if type(candidate) is not int or candidate not in (1, 2):
        raise ValueError("unknown candidate")
    return f"spiralreality/byte-release-candidate-{candidate}-20261025"


def exact_keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("unexpected record fields")


def identity(value, limit):
    exact_keys(value, ("bytes", "sha256"))
    if (type(value["bytes"]) is not int or not 0 < value["bytes"] <= limit
            or not c.is_hash(value["sha256"])):
        raise ValueError("invalid bounded identity")
    return value


def parse_record(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, bytes) or len(data) > MAX_RECORD:
        raise ValueError("oversized transport record")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate transport key")
            result[key] = value
        return result

    def reject(_value):
        raise ValueError("non-finite transport value")

    return json.loads(data, object_pairs_hook=unique, parse_constant=reject)


def read_record(path):
    v1.regular(path, MAX_RECORD)
    return parse_record(Path(path).read_bytes())


def validate_descriptor(value):
    exact_keys(value, ("schema", "candidate", "admission_sha256", "source_commit",
                       "plaintext", "ciphertext", "object", "release_ready"))
    branch(value["candidate"])
    if (value["schema"] not in (DESCRIPTOR_SCHEMA, GITHUB_DESCRIPTOR_SCHEMA, ACTIONS_DESCRIPTOR_SCHEMA) or value["release_ready"] is not False
            or not c.is_hash(value["admission_sha256"])
            or not c.is_hash(value["source_commit"], 40)):
        raise ValueError("invalid input descriptor")
    identity(value["plaintext"], MAX_INPUT)
    identity(value["ciphertext"], MAX_CIPHER)
    if value["schema"] == ACTIONS_DESCRIPTOR_SCHEMA:
        from byte_release_actions_input import validate_pin
        pin = validate_pin(value["object"])
        if {key: pin["ciphertext"][key] for key in ("bytes", "sha256")} != value["ciphertext"]:
            raise ValueError("Actions pin differs from the candidate ciphertext")
        return value
    if value["schema"] == GITHUB_DESCRIPTOR_SCHEMA:
        from byte_release_github_asset import validate_locator
        validate_locator(value["object"])
        return value
    location = value["object"]
    exact_keys(location, ("host", "path"))
    host, path = location["host"], location["path"]
    if (type(host) is not str or len(host) > 253 or host != host.lower()
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host)
            or "." not in host or any(not part or len(part) > 63 or
                part.startswith("-") or part.endswith("-") for part in host.split("."))):
        raise ValueError("invalid object host")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("literal object addresses are not supported")
    # Resolvers also accept short, octal and hexadecimal IPv4 representations.
    numeric = r"(?:0x[0-9a-f]+|[0-9]+)"
    if re.fullmatch(rf"{numeric}(?:\.{numeric}){{0,3}}", host):
        raise ValueError("numeric object addresses are not supported")
    if (type(path) is not str or not path.startswith("/")
            or not v1.name_ok(path[1:]) or not path.endswith(".age")):
        raise ValueError("invalid opaque object path")
    return value


def validate_amendment(value, expected, descriptor):
    validate_descriptor(descriptor)
    exact_keys(value, ("schema", "decision", "candidate", "admission_sha256",
                       "local_evidence_sha256", "source_commit", "source_tree",
                       "controller_commit", "controller_tree", "descriptor_sha256",
                       "review_record_sha256", "result_recipient", "repository", "workflow", "branch",
                       "concurrency", "limits", "release_ready"))
    if not c.is_hash(expected) or c.digest(value) != expected:
        raise ValueError("amendment differs from the external pin")
    if (value["schema"] != AMENDMENT_SCHEMA or value["decision"] != c.DECISION
            or value["candidate"] != descriptor["candidate"]
            or type(value["candidate"]) is not int
            or value["admission_sha256"] != descriptor["admission_sha256"]
            or value["source_commit"] != descriptor["source_commit"]
            or value["descriptor_sha256"] != c.digest(descriptor)
            or value["repository"] != REPO or value["workflow"] != WORKFLOW
            or value["branch"] != branch(value["candidate"])
            or value["concurrency"] != CONCURRENCY
            or value["limits"] != LIMITS or type(value["limits"]) is not dict
            or any(type(item) is not int for item in value["limits"].values())
            or value["release_ready"] is not False):
        raise ValueError("amendment changes the fixed transport policy")
    for key in ("source_commit", "source_tree", "controller_commit", "controller_tree"):
        if not c.is_hash(value[key], 40):
            raise ValueError("invalid source or controller identity")
    for key in ("local_evidence_sha256", "review_record_sha256"):
        if not c.is_hash(value[key]):
            raise ValueError("invalid evidence or review identity")
    if (type(value["result_recipient"]) is not str
            or not re.fullmatch(r"age1[0-9a-z]{58}", value["result_recipient"])):
        raise ValueError("invalid pinned public result recipient")
    return value


def validate_url(url, descriptor):
    validate_descriptor(descriptor)
    if descriptor["schema"] != DESCRIPTOR_SCHEMA:
        raise ValueError("direct URL requires the direct-object descriptor")
    if (type(url) is not str or len(url) > 8192
            or any(ord(char) < 33 or ord(char) > 126 for char in url)):
        raise ValueError("invalid private object URL")
    parsed = urlsplit(url)
    location = descriptor["object"]
    if (parsed.scheme != "https" or parsed.netloc != location["host"]
            or parsed.path != location["path"] or parsed.fragment
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("object URL is outside the pinned endpoint")
    return url


def copy_verified(stream, output, expected):
    """Keep partial first transfers, but return a receipt only after exact EOF."""
    identity(expected, MAX_CIPHER)
    output = Path(output)
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError("object output must not follow a link")
    digest, total = hashlib.sha256(), 0
    with output.open("xb") as destination:
        os.fchmod(destination.fileno(), 0o600)
        while True:
            block = stream.read(min(v1.CHUNK, expected["bytes"] - total + 1))
            if not block:
                break
            total += len(block)
            if total > expected["bytes"]:
                raise ValueError("object stream exceeded its pinned size")
            destination.write(block)
            digest.update(block)
        destination.flush()
        os.fsync(destination.fileno())
    if total != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("object stream identity differs")
    return dict(expected)


def unpack_input(archive_path, output, descriptor):
    """New outer limit; preserve v1's inner format and complete evidence bytes."""
    validate_descriptor(descriptor)
    expected = descriptor["plaintext"]
    if v1.regular(archive_path, MAX_INPUT) != expected:
        raise ValueError("input ZIP identity differs")
    output = Path(output)
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError("input output must not follow a link")
    output.mkdir(mode=0o700, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if (not 2 <= len(names) <= v1.MAX_FILES + 1 or names[0] != v1.MANIFEST
                or len(set(names)) != len(names)):
            raise ValueError("invalid input roster")
        for member in members:
            mode = member.external_attr >> 16
            if (not v1.name_ok(member.filename) or member.is_dir() or member.flag_bits & 1
                    or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or member.file_size > v1.MAX_FILE
                    or Path(member.filename).name.endswith("-key.txt")):
                raise ValueError("unsafe input member")
        if (members[0].file_size > LIMITS["manifest_bytes"]
                or sum(member.file_size for member in members) > v1.MAX_TOTAL):
            raise ValueError("input expansion exceeds its bounds")
        manifest_path = output / v1.MANIFEST
        with manifest_path.open("xb") as destination:
            os.fchmod(destination.fileno(), 0o600)
            destination.write(archive.read(v1.MANIFEST))
        manifest = c.read(manifest_path, LIMITS["manifest_bytes"])
        exact_keys(manifest, ("schema", "kind", "admission_sha256", "release_ready", "files"))
        files = manifest["files"]
        if (manifest["schema"] != v1.SCHEMA or manifest["kind"] != "input"
                or manifest["admission_sha256"] != descriptor["admission_sha256"]
                or manifest["release_ready"] is not False or type(files) is not dict
                or set(files) != set(names[1:])):
            raise ValueError("input manifest differs")
        for member in members[1:]:
            row = files[member.filename]
            exact_keys(row, ("bytes", "sha256"))
            if (type(row["bytes"]) is not int or row["bytes"] != member.file_size
                    or not c.is_hash(row["sha256"])):
                raise ValueError("input member descriptor differs")
            target = output / member.filename
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            total, digest = 0, hashlib.sha256()
            with archive.open(member) as reader, target.open("xb") as writer:
                os.fchmod(writer.fileno(), 0o600)
                while block := reader.read(v1.CHUNK):
                    total += len(block)
                    if total > row["bytes"]:
                        raise ValueError("input member exceeded its size")
                    writer.write(block)
                    digest.update(block)
            if total != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("input member bytes differ")
    if v1.regular(archive_path, MAX_INPUT) != expected:
        raise ValueError("input ZIP changed during extraction")
    return manifest


def verify_input_tree(root, amendment, descriptor):
    root = Path(root)
    directory = root / "ledger" / f"candidate-{descriptor['candidate']}"
    admission = v1.verify_environment(directory, "local-metal")
    local = c.read(directory / "local-metal.result.json")
    if (c.digest(admission) != descriptor["admission_sha256"]
            or admission["candidate"] != descriptor["candidate"]
            or admission["source"]["commit"] != amendment["source_commit"]
            or admission["source"]["tree"] != amendment["source_tree"]
            or local["evidence_sha256"] != amendment["local_evidence_sha256"]):
        raise ValueError("input does not contain the pinned complete local proof")
    wheel = root / "wheel" / admission["wheels"]["macos"]["filename"]
    if v1.artifacts.wheel_inventory(wheel) != admission["wheels"]["macos"]:
        raise ValueError("input wheel differs")
    return admission


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def clean_tree(root):
    if git(root, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("source or controller checkout is dirty")
    return git(root, "rev-parse", "HEAD", "HEAD^{tree}").decode().splitlines()


def tree_entries(root):
    result = {}
    for row in git(root, "ls-tree", "-rz", "HEAD").split(b"\0"):
        if row:
            metadata, name = row.split(b"\t", 1)
            result[name.decode()] = metadata.decode()
    return result


def validate_controller(controller, source, amendment, descriptor, expected):
    from byte_release_transport_workflow_v2 import render
    validate_amendment(amendment, expected, descriptor)
    if (clean_tree(controller) != [amendment["controller_commit"], amendment["controller_tree"]]
            or clean_tree(source) != [amendment["source_commit"], amendment["source_tree"]]):
        raise ValueError("source or controller revision differs")
    if read_record(Path(controller) / DESCRIPTOR) != descriptor:
        raise ValueError("committed descriptor differs")
    original, changed = tree_entries(source), tree_entries(controller)
    extra = ACTIONS_RUNTIME_FILES if descriptor["schema"] == ACTIONS_DESCRIPTOR_SCHEMA else (
        (GITHUB_RUNTIME_FILE,) if descriptor["schema"] == GITHUB_DESCRIPTOR_SCHEMA else ())
    runtime = RUNTIME_FILES + extra
    if any(not changed.get(name, "").startswith("100644 blob ") for name in runtime):
        raise ValueError("controller runtime modules must be tracked ordinary files")
    difference = {name for name in set(original) | set(changed) if original.get(name) != changed.get(name)}
    if not difference <= CHANGE_ALLOWLIST or DESCRIPTOR not in difference:
        raise ValueError("controller modifies files outside the transport boundary")
    for name in difference:
        if name not in changed or not changed[name].startswith("100644 blob "):
            raise ValueError("controller additions must be ordinary non-executable files")
    workflow = ".github/workflows/" + WORKFLOW
    baseline = (Path(source) / workflow).read_text()
    if (Path(controller) / workflow).read_text() != render(
            baseline, github_asset=descriptor["schema"] == GITHUB_DESCRIPTOR_SCHEMA,
            actions_artifact=descriptor["schema"] == ACTIONS_DESCRIPTOR_SCHEMA):
        raise ValueError("controller execution envelope differs")
    return amendment["controller_commit"]
