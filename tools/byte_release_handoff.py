#!/usr/bin/env python3
"""Bounded, plaintext handoff archives; encrypt outside this tool before upload.

This verifies the hash chain of already accepted local evidence, not GPU
execution again. It never imports the native package, dispatches a job, or
resets a candidate. Archive authentication comes from the separately pinned
digest and age envelope, not from a self-authored manifest alone.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_artifacts as artifacts
import byte_release_contract as contract

SCHEMA = "spiraltorch.byte_release.handoff.v1"
MANIFEST = "transport-manifest.json"
MAX_FILES = 16384
MAX_FILE = 256 * 1024 * 1024
MAX_TOTAL = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE = MAX_TOTAL + 16 * 1024 * 1024
MAX_INPUT_ARCHIVE = 96 * 1024 * 1024
CHUNK = 1024 * 1024


def name_ok(name):
    return (contract.relative_file(name) and len(name) <= 255
            and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name) is not None)


def regular(path, limit=MAX_FILE):
    path = Path(path)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("handoff must not follow symbolic links")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limit:
        raise ValueError("handoff file is not bounded and regular")
    return {"bytes": info.st_size, "sha256": artifacts.sha256(path)}


def inventory(root):
    root = Path(root)
    if not root.is_dir() or any(item.is_symlink() for item in (root, *root.parents)):
        raise ValueError("handoff root must be a plain directory")
    files, pending, count = {}, [root], 0
    while pending:
        with os.scandir(pending.pop()) as children:
            for child in children:
                count += 1
                if count > 2 * MAX_FILES:
                    raise ValueError("too many handoff entries")
                path = Path(child.path)
                name = path.relative_to(root).as_posix()
                if not name_ok(name) or path.name.endswith("-key.txt"):
                    raise ValueError("unsafe or private-key handoff path")
                mode = child.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    pending.append(path)
                elif stat.S_ISREG(mode):
                    files[name] = path
                else:
                    raise ValueError("links and special files are not handoff evidence")
    if not 1 <= len(files) <= MAX_FILES or MANIFEST in files:
        raise ValueError("invalid handoff file count or reserved path")
    return dict(sorted(files.items()))


def verify_environment(directory, environment, *, receipt=None):
    """Read only: receipt -> schedule -> round seal -> exact retained bytes."""
    directory = Path(directory)
    admission = contract.validate_admission(contract.read(directory / "admission.json"))
    if environment not in contract.schedule.ENVIRONMENTS:
        raise ValueError("unknown environment")
    if receipt is None:
        receipt = contract.read(directory / (environment + ".result.json"))
    root = directory / environment
    result = contract.read(root / "result.json")
    if (receipt.get("release_ready") is not False or receipt.get("status") != "verified_rounds"
            or receipt.get("admission_sha256") != contract.digest(admission)
            or receipt.get("evidence_sha256") != contract.digest(result)
            or result.get("schema") != "spiraltorch.byte_release.environment_schedule.v1"
            or result.get("admission_sha256") != contract.digest(admission)
            or result.get("environment") != environment or result.get("status") != "verified_rounds"
            or result.get("release_ready") is not False
            or result.get("budget_seconds") != admission["environment_budget_seconds"][environment]
            or result.get("planned_rounds") != 3 or len(result.get("rounds", [])) != 3):
        raise ValueError("environment lacks a matching complete acceptance receipt")
    claimed = contract.read(directory / (environment + ".start.json"))
    if claimed.get("admission_sha256") != contract.digest(admission) or claimed.get("environment") != environment:
        raise ValueError("environment claim differs")
    roster = {"start.json", "result.json"}
    for index, row in enumerate(result["rounds"], 1):
        if (row.get("round") != index or row.get("status") != "verified" or row.get("returncode") != 0
                or not contract.is_hash(row.get("evidence_sha256"))):
            raise ValueError("round acceptance is incomplete")
        location = root / f"round-{index}"
        seal = contract.read(location / "verified-files.json")
        if (type(seal) is not dict or not seal or contract.digest(seal) != row["evidence_sha256"]
                or contract.read(location / "result.json") != row):
            raise ValueError("round seal differs from the accepted schedule")
        for name, expected in seal.items():
            if not name_ok(name) or not contract.is_hash(expected) or regular(location / name)["sha256"] != expected:
                raise ValueError("accepted round bytes changed")
        if set(inventory(location)) != set(seal) | {"verified-files.json", "result.json"}:
            raise ValueError("accepted round has missing or extra files")
        roster.update(f"round-{index}/{name}" for name in (*seal, "verified-files.json", "result.json"))
    if set(inventory(root)) != roster:
        raise ValueError("environment evidence roster differs")
    return admission


def pack(paths, output, kind, admission_sha256, pins=None):
    if kind not in ("input", "result") or not contract.is_hash(admission_sha256):
        raise ValueError("handoff requires a kind and frozen admission digest")
    if (not 1 <= len(paths) <= MAX_FILES or MANIFEST in paths
            or any(not name_ok(name) or Path(name).name.endswith("-key.txt") for name in paths)):
        raise ValueError("invalid handoff roster")
    expected = {name: regular(path) for name, path in sorted(paths.items())}
    if any(expected.get(name) != value for name, value in (pins or {}).items()):
        raise ValueError("handoff payload differs from its frozen identity")
    if sum(row["bytes"] for row in expected.values()) > MAX_TOTAL:
        raise ValueError("handoff exceeds total size bound")
    manifest = contract.encoded(dict(schema=SCHEMA, kind=kind, admission_sha256=admission_sha256,
                                    release_ready=False, files=expected))
    if len(manifest) > 8 * 1024 * 1024 or len(manifest) + sum(row["bytes"] for row in expected.values()) > MAX_TOTAL:
        raise ValueError("handoff manifest exceeds size bound")
    output = Path(output)
    if any(item.is_symlink() for item in output.parents):
        raise ValueError("archive parent must not follow a link")
    with output.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(MANIFEST, manifest)
            for name, path in sorted(paths.items()):
                archive.write(path, name)
                if regular(path) != expected[name]:
                    raise ValueError("input changed while packing")
        stream.flush()
        os.fsync(stream.fileno())
    identity = regular(output, MAX_INPUT_ARCHIVE if kind == "input" else MAX_ARCHIVE)
    return dict(schema=SCHEMA, kind=kind, admission_sha256=admission_sha256,
                archive_sha256=identity["sha256"], bytes=identity["bytes"], release_ready=False)


def unpack(archive_path, output, expected_sha256, kind, admission_sha256):
    limit = MAX_INPUT_ARCHIVE if kind == "input" else MAX_ARCHIVE
    if (kind not in ("input", "result") or not contract.is_hash(expected_sha256)
            or not contract.is_hash(admission_sha256) or regular(archive_path, limit)["sha256"] != expected_sha256):
        raise ValueError("handoff archive identity differs")
    output = Path(output)
    if any(item.is_symlink() for item in output.parents):
        raise ValueError("output parent must not follow a link")
    output.mkdir(mode=0o700, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if not 2 <= len(names) <= MAX_FILES + 1 or names[0] != MANIFEST or len(set(names)) != len(names):
            raise ValueError("invalid archive roster")
        for member in members:
            mode = member.external_attr >> 16
            if (not name_ok(member.filename) or member.is_dir() or member.flag_bits & 1
                    or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)) or member.file_size > MAX_FILE):
                raise ValueError("invalid archive member")
        if members[0].file_size > 8 * 1024 * 1024 or sum(item.file_size for item in members) > MAX_TOTAL:
            raise ValueError("archive decompression exceeds bound")
        manifest_path = output / MANIFEST
        manifest_path.write_bytes(archive.read(MANIFEST))
        manifest = contract.read(manifest_path)
        expected = manifest.get("files")
        if (manifest.get("schema") != SCHEMA or manifest.get("kind") != kind
                or manifest.get("admission_sha256") != admission_sha256 or manifest.get("release_ready") is not False
                or type(expected) is not dict or set(expected) != set(names[1:])):
            raise ValueError("handoff manifest differs")
        for member in members[1:]:
            name, row = member.filename, expected[member.filename]
            if (type(row) is not dict or type(row.get("bytes")) is not int or row["bytes"] != member.file_size
                    or not contract.is_hash(row.get("sha256")) or Path(name).name.endswith("-key.txt")):
                raise ValueError("handoff descriptor differs")
            target = output / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            digest, total = hashlib.sha256(), 0
            with archive.open(member) as source, target.open("xb") as destination:
                os.fchmod(destination.fileno(), 0o600)
                while block := source.read(CHUNK):
                    total += len(block)
                    if total > row["bytes"]:
                        raise ValueError("member expanded beyond declared bound")
                    destination.write(block)
                    digest.update(block)
            if total != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("handoff bytes differ")
    if regular(archive_path, limit)["sha256"] != expected_sha256:
        raise ValueError("archive changed during unpacking")
    return manifest


def pack_input(ledger, wheel, output):
    ledger = Path(ledger)
    if contract.read(ledger / "decision.json") != {"decision": contract.DECISION, "candidate_limit": 2}:
        raise ValueError("unexpected canonical decision")
    candidate = 2 if (ledger / "candidate-2").exists() else 1
    directory = ledger / f"candidate-{candidate}"
    admission = verify_environment(directory, "local-metal")
    if admission["candidate"] != candidate or artifacts.wheel_inventory(wheel) != admission["wheels"]["macos"]:
        raise ValueError("handoff candidate or macOS wheel differs")
    if any((directory / ("hosted-metal." + suffix + ".json")).exists() for suffix in ("start", "result", "dispatch")):
        raise ValueError("hosted ownership is already reserved or consumed")
    paths = {"ledger/decision.json": ledger / "decision.json"}
    for index in range(1, candidate + 1):
        prefix = f"candidate-{index}"
        base = ledger / prefix
        paths[f"ledger/{prefix}/admission.json"] = base / "admission.json"
        for environment in contract.schedule.ENVIRONMENTS:
            for suffix in ("start", "result"):
                name = f"{environment}.{suffix}.json"
                if (base / name).is_file():
                    paths[f"ledger/{prefix}/{name}"] = base / name
    paths.update({f"ledger/candidate-{candidate}/local-metal/{name}": path
                  for name, path in inventory(directory / "local-metal").items()})
    macos = admission["wheels"]["macos"]
    wheel_name = "wheel/" + macos["filename"]
    paths[wheel_name] = Path(wheel)
    result = pack(paths, output, "input", contract.digest(admission),
                  {wheel_name: {key: macos[key] for key in ("bytes", "sha256")}})
    if verify_environment(directory, "local-metal") != admission:
        raise ValueError("admission changed while packing")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack-input", "pack-result", "unpack-input", "unpack-result"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--admission-sha256")
    args = parser.parse_args(argv)
    if args.operation == "pack-input":
        if args.wheel is None:
            parser.error("input packing requires the frozen macOS wheel")
        result = pack_input(args.input, args.wheel, args.output)
    elif args.operation == "pack-result":
        if args.output.resolve().is_relative_to(args.input.resolve()):
            parser.error("archive output must be outside evidence")
        result = pack(inventory(args.input), args.output, "result", args.admission_sha256)
    else:
        result = unpack(args.input, args.output, args.sha256, args.operation.removeprefix("unpack-"), args.admission_sha256)
        result = {key: value for key, value in result.items() if key != "files"}
    print(contract.encoded(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
