#!/usr/bin/env python3
"""Bounded private archives for the fixed-wheel pair; no execution or network."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_study_prelude as prelude  # noqa: E402

B = prelude.retained.boundary
SCHEMA = "spiraltorch.prelude_pair.transport.v1"
MANIFEST = "transport-manifest.json"
WHEEL = "wheel/spiraltorch-0.4.32-cp38-abi3-macosx_11_0_arm64.whl"
ORIGINAL = {
    "first-checkpoint.json": "aef5dda4a9372604c8e1c6db82f22cd826a8be8aab6172dc43d60e908264445b",
    "first-report.json": "8bb0f68edae547dc3ed81022a8f33e0575397d16deb94388526fbfee370e9b6a",
    "full-report.json": "1ce904a8e58d94d60e6acb0a907b68465f847738427664a294a9362fe083470f",
    "request.json": "6711703a49361d1d2d577912dbe86d47d4d42122273fee797fb27fd6abf8c52c",
    "resumed-checkpoint.json": "706e696a0ec64ac6ef71938b59c164e6ede8fb0ef2866bbf91c9f856cd226c88",
    "resumed-report.json": "b4a5e768df836720ccf12b6b64bdf68e114ce06a8f1e019a16ed6e5411a139d8",
    "runtime.json": "ec66f2e56e0b76e38ccddd63ee9d2d7424c79a944cfe30d081d957c9701dbe17",
    "summary.json": "b76fac61decd1747287320d4386da55028b7bddf7d53ee7770eda2eeca35a6e1",
}
MAX_INPUT = 64 * 1024 * 1024
MAX_RESULT = 2 * 1024 * 1024 * 1024
MAX_FILE = 256 * 1024 * 1024
MAX_FILES = 4096
MAX_MANIFEST = 2 * 1024 * 1024
CHUNK = 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def canonical(name):
    if (not isinstance(name, str) or len(name) > 255
            or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name)
            or any(part in (".", "..") for part in name.split("/"))):
        raise ValueError("noncanonical archive member")
    return name


def raw_file(path, limit=MAX_FILE):
    path = Path(path)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("symbolic links are not transport inputs")
    return B.read(path, limit=limit)


def identity(raw):
    return {"bytes": len(raw), "sha256": B.digest(raw)}


def native_wheel(raw):
    if B.digest(raw) != prelude.WHEEL_SHA256:
        raise ValueError("requires the exact recovered wheel, not a rebuild")
    with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
        natives = [name for name in wheel.namelist()
                   if name.endswith(".so") and PurePosixPath(name).name.startswith("spiraltorch.")]
        if (len(natives) != 1 or wheel.getinfo(natives[0]).file_size > MAX_FILE
                or B.digest(wheel.read(natives[0])) != prelude.NATIVE_SHA256):
            raise ValueError("native payload differs")


def regular_paths(root):
    root = Path(root)
    if any(path.is_symlink() for path in (root, *root.parents)) or not root.is_dir():
        raise ValueError("evidence root must be a plain directory")
    files, pending, entries = {}, [root], 0
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > MAX_FILES * 2:
                    raise ValueError("too many evidence entries")
                name = canonical(Path(child.path).relative_to(root).as_posix())
                mode = child.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    pending.append(Path(child.path))
                elif stat.S_ISREG(mode):
                    files[name] = Path(child.path)
                    if len(files) > MAX_FILES:
                        raise ValueError("too many evidence files")
                else:
                    raise ValueError("links and special files are not evidence")
    if not files or MANIFEST in files:
        raise ValueError("empty evidence or reserved manifest name")
    return dict(sorted(files.items()))


def digest_archive(path, limit):
    path = Path(path)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("archive must not follow a link")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("expected bounded regular archive")
        total, digest = 0, hashlib.sha256()
        while block := stream.read(CHUNK):
            total += len(block)
            if total > limit:
                raise ValueError("archive grew beyond its limit")
            digest.update(block)
    return digest.hexdigest()


def write_archive(paths, output, expected, limit, manifest=None):
    output = Path(output)
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError("output parent must not follow a link")
    if sum(row["bytes"] for row in expected.values()) > limit:
        raise ValueError("payload exceeds archive limit")
    with output.open("xb") as raw:
        os.fchmod(raw.fileno(), 0o600)
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                names = ([MANIFEST] if manifest is not None else []) + sorted(paths)
                for name in names:
                    canonical(name)
                    payload = manifest if name == MANIFEST else raw_file(paths[name])
                    if name != MANIFEST and identity(payload) != expected[name]:
                        raise ValueError("file changed before packing")
                    member = tarfile.TarInfo(name)
                    member.size, member.mode = len(payload), 0o600
                    archive.addfile(member, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    for name, path in paths.items():
        if identity(raw_file(path)) != expected[name]:
            raise ValueError("file changed during packing")
    return digest_archive(output, limit + MAX_MANIFEST + 8 * 1024 * 1024)


def pack_input(failures, wheel, output):
    if Path(output).resolve().is_relative_to(Path(failures).resolve()):
        raise ValueError("input archive must be outside the preserved failure directory")
    paths = {"original/version-9/" + name: Path(failures) / "version-9" / name for name in ORIGINAL}
    paths[WHEEL] = Path(wheel)
    expected = {}
    for name, path in paths.items():
        raw = raw_file(path)
        if name == WHEEL:
            native_wheel(raw)
        elif B.digest(raw) != ORIGINAL[PurePosixPath(name).name]:
            raise ValueError("original input identity differs")
        expected[name] = identity(raw)
    sha = write_archive(paths, output, expected, MAX_INPUT)
    return {"schema": SCHEMA, "kind": "fixed_input", "release_ready": False,
            "archive_sha256": sha, "files": expected}


def pack_result(root, output):
    root, output = Path(root), Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("archive output must be outside evidence")
    paths = regular_paths(root)
    expected, total = {}, 0
    for name, path in paths.items():
        expected[name] = identity(raw_file(path))
        total += expected[name]["bytes"]
        if total > MAX_RESULT:
            raise ValueError("result payload exceeds limit")
    manifest = {"schema": SCHEMA, "kind": "evidence_only", "release_ready": False, "files": expected}
    raw = encoded(manifest)
    if len(raw) > MAX_MANIFEST:
        raise ValueError("manifest too large")
    sha = write_archive(paths, output, expected, MAX_RESULT, raw)
    if regular_paths(root) != paths:
        raise ValueError("evidence roster changed during packing")
    return {"schema": SCHEMA, "kind": "result_archive", "release_ready": False,
            "archive_sha256": sha, "files": expected}


def members(stream, total_limit, file_limit, count_limit):
    """Parse only plain USTAR, before any extensible/PAX metadata is processed."""
    total, count, names = 0, 0, set()
    while True:
        header = stream.read(512)
        if len(header) != 512:
            raise ValueError("truncated archive header")
        if header == bytes(512):
            if stream.read(512) != bytes(512):
                raise ValueError("archive requires two terminal blocks")
            tail = stream.read(10241)
            if len(tail) > 10240 or any(tail) or len(tail) % 512:
                raise ValueError("trailing or oversized archive data")
            return
        if header[257:265] != b"ustar\x0000":
            raise ValueError("only plain USTAR headers are accepted")
        member = tarfile.TarInfo.frombuf(header, "ascii", "strict")
        name = canonical(member.name)
        count += 1
        total += member.size
        if (name in names or member.type != tarfile.REGTYPE or member.linkname
                or member.size < 0 or member.size > file_limit or total > total_limit
                or (name == MANIFEST and member.size > MAX_MANIFEST)
                or count > count_limit):
            raise ValueError("invalid member type, count, size or duplicate")
        names.add(name)
        data = stream.read(member.size)
        if len(data) != member.size:
            raise ValueError("truncated member")
        padding = (-member.size) % 512
        if stream.read(padding) != bytes(padding):
            raise ValueError("invalid member padding")
        yield name, data


def expected_result(raw):
    if len(raw) > MAX_MANIFEST:
        raise ValueError("manifest too large")
    value = B.decode(raw)
    if (not isinstance(value, dict) or set(value) != {"schema", "kind", "release_ready", "files"}
            or value["schema"] != SCHEMA or value["kind"] != "evidence_only"
            or value["release_ready"] is not False):
        raise ValueError("invalid result manifest")
    files = value["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("invalid manifest file count")
    total = 0
    for name, row in files.items():
        canonical(name)
        if (name == MANIFEST or not isinstance(row, dict) or set(row) != {"bytes", "sha256"}
                or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= MAX_FILE
                or not isinstance(row["sha256"], str) or not SHA.fullmatch(row["sha256"])):
            raise ValueError("invalid manifest entry")
        total += row["bytes"]
    if total > MAX_RESULT:
        raise ValueError("manifest exceeds result bound")
    return files


def save_member(root, name, raw):
    target = root / name
    for parent in reversed(target.parents):
        if parent != root and parent.is_relative_to(root):
            parent.mkdir(mode=0o700, exist_ok=True)
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("member parent is not a directory")
    with target.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)


def unpack(archive, output, expected_sha, kind):
    if kind not in ("input", "result") or not SHA.fullmatch(expected_sha):
        raise ValueError("requires archive kind and a pinned SHA-256")
    limit = MAX_INPUT if kind == "input" else MAX_RESULT
    compressed_limit = limit + MAX_MANIFEST + 8 * 1024 * 1024
    if digest_archive(archive, compressed_limit) != expected_sha:
        raise ValueError("archive identity differs")
    output = Path(output)
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError("output parent must not follow a link")
    output.mkdir(mode=0o700, exist_ok=False)
    expected = ({"original/version-9/" + name: sha for name, sha in ORIGINAL.items()}
                | {WHEEL: prelude.WHEEL_SHA256}) if kind == "input" else None
    seen = set()
    # Fresh private outputs are deliberately left incomplete on any failure.
    # Never resume them or infer success merely from the presence of files.
    with gzip.open(archive, "rb") as stream:
        for name, raw in members(stream, limit + MAX_MANIFEST, MAX_FILE, MAX_FILES + 1):
            if kind == "result" and expected is None:
                if name != MANIFEST:
                    raise ValueError("result manifest must be first")
                expected = expected_result(raw)
                save_member(output, name, raw)
                continue
            if expected is None or name not in expected:
                raise ValueError("unexpected archive member")
            if (identity(raw) if kind == "result" else B.digest(raw)) != expected[name]:
                raise ValueError("member identity differs")
            if kind == "input" and name == WHEEL:
                native_wheel(raw)
            save_member(output, name, raw)
            seen.add(name)
    if expected is None or seen != set(expected):
        raise ValueError("missing archive member")
    if digest_archive(archive, compressed_limit) != expected_sha:
        raise ValueError("archive changed while unpacking")
    return {"schema": SCHEMA, "kind": kind, "archive_sha256": expected_sha,
            "file_count": len(seen), "transport_verified": True, "release_ready": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack-input", "unpack-input", "pack-result", "unpack-result"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args(argv)
    if args.operation == "pack-input":
        if args.wheel is None or args.sha256 is not None:
            parser.error("pack-input requires --wheel and no --sha256")
        result = pack_input(args.input, args.wheel, args.output)
    elif args.operation == "pack-result":
        if args.wheel is not None or args.sha256 is not None:
            parser.error("pack-result does not accept wheel or hash overrides")
        result = pack_result(args.input, args.output)
    else:
        if args.sha256 is None or args.wheel is not None:
            parser.error("unpack requires --sha256 and no --wheel")
        result = unpack(args.input, args.output, args.sha256, args.operation.removeprefix("unpack-"))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
