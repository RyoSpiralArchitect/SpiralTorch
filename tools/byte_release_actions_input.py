"""Offline verification of one pinned Actions artifact containing ciphertext.

Metadata is supplied evidence, not authentication or authorization. This module
does not fetch, acquire tokens, dispatch, decrypt, admit a candidate or run ML.
"""
from pathlib import Path
import re
import stat
import struct
import zipfile

import byte_release_transport_v2 as t

SCHEMA = "spiraltorch.byte_release.actions_input_pin.v1"
MAX_ARCHIVE = t.MAX_CIPHER
MAX_CENTRAL_DIRECTORY = 64 * 1024
BRIDGE_PREFIX = "spiralreality/byte-release-input-bridge-"


def _positive(value):
    if type(value) is not int or not 0 < value <= 2**63 - 1:
        raise ValueError("invalid Actions object identity")


def _name(value):
    if (type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,179}", value)
            or ".." in value or value.endswith((".", ".lock"))):
        raise ValueError("invalid opaque Actions object name")


def validate_pin(pin):
    t.exact_keys(pin, ("schema", "repository", "repository_id", "bridge", "artifact", "ciphertext", "release_ready"))
    if pin["schema"] != SCHEMA or pin["repository"] != t.REPO or pin["release_ready"] is not False:
        raise ValueError("invalid Actions input pin")
    _positive(pin["repository_id"])
    bridge = pin["bridge"]
    t.exact_keys(bridge, ("run_id", "run_attempt", "head_sha", "branch", "workflow_id", "workflow_path"))
    for key in ("run_id", "workflow_id"):
        _positive(bridge[key])
    if (type(bridge["run_attempt"]) is not int or bridge["run_attempt"] != 1
            or not t.c.is_hash(bridge["head_sha"], 40)
            or bridge["workflow_path"] != ".github/workflows/" + t.WORKFLOW
            or type(bridge["branch"]) is not str or not bridge["branch"].startswith(BRIDGE_PREFIX)):
        raise ValueError("input is not bound to a first bridge run")
    _name(bridge["branch"][len(BRIDGE_PREFIX):])
    artifact, ciphertext = pin["artifact"], pin["ciphertext"]
    t.exact_keys(artifact, ("id", "name", "bytes", "sha256"))
    t.exact_keys(ciphertext, ("name", "bytes", "sha256"))
    _positive(artifact["id"])
    for value in (artifact, ciphertext):
        _name(value["name"])
        t.identity({key: value[key] for key in ("bytes", "sha256")}, MAX_ARCHIVE)
    if not ciphertext["name"].endswith(".age"):
        raise ValueError("input member must be the pinned age ciphertext")
    return pin


def _expect(observed, expected):
    if type(observed) is not dict or any(
            type(observed.get(key)) is not type(value) or observed[key] != value
            for key, value in expected.items()):
        raise ValueError("Actions input provenance differs")


def verify_metadata(pin, run, artifact):
    validate_pin(pin)
    bridge, outer = pin["bridge"], pin["artifact"]
    _expect(run, dict(id=bridge["run_id"], run_attempt=1, head_sha=bridge["head_sha"],
                     head_branch=bridge["branch"], workflow_id=bridge["workflow_id"],
                     path=bridge["workflow_path"], event="workflow_dispatch",
                     status="completed", conclusion="success"))
    for key in ("repository", "head_repository"):
        _expect(run.get(key), dict(id=pin["repository_id"], full_name=t.REPO))
    _expect(artifact, dict(id=outer["id"], name=outer["name"], expired=False,
                          size_in_bytes=outer["bytes"], digest="sha256:" + outer["sha256"]))
    _expect(artifact.get("workflow_run"), dict(id=bridge["run_id"], repository_id=pin["repository_id"],
            head_repository_id=pin["repository_id"], head_branch=bridge["branch"], head_sha=bridge["head_sha"]))
    return dict(pin_sha256=t.c.digest(pin), bridge_run_id=bridge["run_id"],
                bridge_run_attempt=1, artifact_id=outer["id"], release_ready=False)


def _bounded_directory(path):
    # Bound directory parsing before ZipFile can allocate arbitrary member lists.
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell() - 22
        if end < 0:
            raise ValueError("missing classic ZIP end record")
        start = max(0, end - 20)
        stream.seek(start)
        tail = stream.read(end + 22 - start)
    if len(tail) != end + 22 - start:
        raise ValueError("truncated ZIP end record")
    # ZipFile follows this locator even if the classic directory counts fit.
    if end >= 20 and tail[:4] == b"PK\x06\x07":
        raise ValueError("ZIP64 directory overrides are not supported")
    fields = struct.unpack("<4s4H2IH", tail[-22:])
    signature, disk, directory_disk, entries, total, size, offset, comment = fields
    if (signature != b"PK\x05\x06" or disk != 0 or directory_disk != 0
            or entries != 1 or total != 1 or comment != 0
            or not 46 <= size <= MAX_CENTRAL_DIRECTORY or offset + size != end):
        raise ValueError("input requires a bounded classic single-member ZIP")


def unpack_input(pin, run, artifact, archive_path, output):
    receipt = verify_metadata(pin, run, artifact)
    archive_path, output = Path(archive_path), Path(output)
    outer = {key: pin["artifact"][key] for key in ("bytes", "sha256")}
    ciphertext = {key: pin["ciphertext"][key] for key in ("bytes", "sha256")}
    if output.exists() or output.is_symlink():
        raise FileExistsError("first ciphertext output already exists")
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError("ciphertext output must not follow links")
    if t.v1.regular(archive_path, MAX_ARCHIVE) != outer:
        raise ValueError("Actions archive bytes differ from the external pin")
    _bounded_directory(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) != 1:
            raise ValueError("Actions input must contain exactly one ciphertext")
        member = members[0]
        mode = member.external_attr >> 16
        if (member.filename != pin["ciphertext"]["name"] or member.is_dir()
                or member.flag_bits & 1 or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or member.file_size != ciphertext["bytes"]):
            raise ValueError("Actions input member differs from the external pin")
        with archive.open(member) as stream:
            t.copy_verified(stream, output, ciphertext)
    if t.v1.regular(archive_path, MAX_ARCHIVE) != outer:
        raise ValueError("Actions archive changed during extraction")
    return dict(receipt, status="verified_pinned_ciphertext",
                archive_sha256=outer["sha256"], ciphertext=ciphertext)
