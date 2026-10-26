"""Fixed diagnostic archives only; no native imports, network or execution."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import zipfile

SCHEMA = "spiraltorch.detached_forward.transport.v1"
CAMPAIGN = "detached-forward-20261026"
MANIFEST = "transport-manifest.json"
WHEEL = "artifact/wheels/spiraltorch-0.4.32-cp38-abi3-macosx_14_0_x86_64.macosx_14_0_arm64.macosx_14_0_universal2.whl"
NATIVE_SHA = "509a7a7b6fb33f622bc0f74ae1a976684a89c19d3ba473f39a8664e86f3de369"
PINS = {
    "detached_forward_probe.py": "dabbb5ce50eaefbf1f28bc9479d13a643162512043ddf1fd712c5fd56bdc5d20",
    "test_detached_forward_probe.py": "c463e4ac542b97df7aadeb01423111b6bd0a1fcbd2b5114d398b4213e384ebe1",
    "harness/tools/byte_study_diagnostics.py": "410be6cf04025c5e8cfbb469df43f048928db40fefba018826f90c365b4eab63",
    "harness/tools/run_byte_study_replay.py": "65716498e3e0c2a3071de5bdef853f84bdbe2255a77cbd5def12e03296cfb97b",
    "harness/tools/owned_process.py": "a5dc7ccf742576b6969ee73f76a6d858246f9726a66c7517001c3c65f8f77b60",
    "release-v0-4-33-first-detached-failure.zip": "0be8054e75a3d9ebf4740ace78f83e560d7ac5aee0ed1921546b23a68a3f1bde",
    "artifact/artifact.json": "fb8fcaa269c72ca03bbb0b1c89d641b1476f6985b27419d20fc4fd26a4abdd57",
    WHEEL: "86a928cc71f83231af4e920bbcdcec218668633f8d6428ff5db36bad37a8dd98",
}
MAX_BYTES = 96 * 1024 * 1024
MAX_FILES = 256
MAX_MANIFEST = 64 * 1024
SHA = re.compile(r"[a-f0-9]{64}\Z")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("nonfinite JSON")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def canonical(name):
    if (type(name) is not str or len(name) > 255
            or re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name) is None
            or any(part in (".", "..") for part in name.split("/"))
            or name.endswith(("-key.txt", ".key", ".pem"))):
        raise ValueError("unsafe or private-key path")
    return name


def no_links(path):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("links are outside the transport contract")


def read(path):
    path = Path(path)
    no_links(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_BYTES:
            raise ValueError("expected bounded regular file")
        raw = stream.read(MAX_BYTES + 1)
        if len(raw) != info.st_size or len(raw) > MAX_BYTES:
            raise ValueError("file grew or changed while reading")
        return raw


def inventory(root):
    root = Path(root)
    no_links(root)
    if not root.is_dir():
        raise ValueError("missing evidence directory")
    pending, found, count = [root], {}, 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for item in entries:
                count += 1
                if count > MAX_FILES * 2:
                    raise ValueError("too many entries")
                path = Path(item.path)
                name = canonical(path.relative_to(root).as_posix())
                mode = item.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    pending.append(path)
                elif stat.S_ISREG(mode):
                    found[name] = path
                else:
                    raise ValueError("special evidence file")
    if not 1 <= len(found) <= MAX_FILES or MANIFEST in found:
        raise ValueError("invalid evidence roster")
    return dict(sorted(found.items()))


def check_wheel(raw):
    if sha(raw) != PINS[WHEEL]:
        raise ValueError("wheel is not the fixed saved artifact")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist()
                 if Path(name).name.startswith("spiraltorch.") and name.endswith(".so")]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > MAX_BYTES:
            raise ValueError("native payload is missing or oversized")
        if sha(archive.read(names[0])) != NATIVE_SHA:
            raise ValueError("native payload differs")


def pack(paths, output, kind):
    if kind not in ("input", "result") or not 1 <= len(paths) <= MAX_FILES:
        raise ValueError("invalid packet kind or file count")
    if kind == "input" and set(paths) != set(PINS):
        raise ValueError("fixed input roster differs")
    output = Path(output)
    no_links(output)
    if any(output.resolve() == Path(path).resolve() for path in paths.values()):
        raise ValueError("output aliases input")
    files, total = {}, 0
    for name, path in sorted(paths.items()):
        canonical(name)
        if name == MANIFEST:
            raise ValueError("reserved manifest path")
        raw = read(path)
        files[name] = dict(bytes=len(raw), sha256=sha(raw))
        total += len(raw)
        if total > MAX_BYTES:
            raise ValueError("packet exceeds size bound")
        if kind == "input" and files[name]["sha256"] != PINS[name]:
            raise ValueError("fixed input bytes differ")
        if kind == "input" and name == WHEEL:
            check_wheel(raw)
    manifest = encoded(dict(schema=SCHEMA, campaign=CAMPAIGN, kind=kind,
                            release_ready=False, files=files))
    if len(manifest) > MAX_MANIFEST or total + len(manifest) > MAX_BYTES:
        raise ValueError("manifest exceeds size bound")
    with output.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(MANIFEST, manifest)
            for name, path in sorted(paths.items()):
                raw = read(path)
                if dict(bytes=len(raw), sha256=sha(raw)) != files[name]:
                    raise ValueError("input changed before packing")
                archive.writestr(name, raw)
        stream.flush()
        os.fsync(stream.fileno())
    for name, path in paths.items():
        raw = read(path)
        if dict(bytes=len(raw), sha256=sha(raw)) != files[name]:
            raise ValueError("input changed during packing")
    raw = read(output)
    return dict(schema=SCHEMA, campaign=CAMPAIGN, kind=kind, release_ready=False,
                archive_sha256=sha(raw), bytes=len(raw), files=files)


def unpack(archive_path, output, expected_sha, kind):
    if kind not in ("input", "result") or not SHA.fullmatch(expected_sha):
        raise ValueError("unpack needs an external hash and declared kind")
    raw = read(archive_path)
    if sha(raw) != expected_sha:
        raise ValueError("archive identity differs")
    output = Path(output)
    no_links(output)
    if output.exists():
        raise ValueError("output already exists; never overwrite evidence")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if (not 2 <= len(names) <= MAX_FILES + 1 or names[0] != MANIFEST
                or len(set(names)) != len(names)):
            raise ValueError("invalid archive roster")
        total = 0
        for entry in members:
            canonical(entry.filename)
            mode = entry.external_attr >> 16
            total += entry.file_size
            if (entry.is_dir() or entry.flag_bits & 1 or entry.file_size > MAX_BYTES
                    or total > MAX_BYTES or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError("invalid archive type or size")
        if members[0].file_size > MAX_MANIFEST:
            raise ValueError("oversized manifest")
        manifest = decode(archive.read(MANIFEST))
        if (type(manifest) is not dict
                or set(manifest) != {"schema", "campaign", "kind", "release_ready", "files"}
                or manifest["schema"] != SCHEMA or manifest["campaign"] != CAMPAIGN
                or manifest["kind"] != kind or manifest["release_ready"] is not False
                or type(manifest["files"]) is not dict
                or set(manifest["files"]) != set(names[1:])):
            raise ValueError("manifest does not bind this packet")
        expected = manifest["files"]
        if kind == "input" and set(expected) != set(PINS):
            raise ValueError("fixed input roster differs")
        for name, entry in zip(names[1:], members[1:]):
            row = expected[name]
            if (type(row) is not dict or set(row) != {"bytes", "sha256"}
                    or type(row["bytes"]) is not int or row["bytes"] != entry.file_size
                    or type(row["sha256"]) is not str or not SHA.fullmatch(row["sha256"])):
                raise ValueError("invalid manifest entry")
            payload = archive.read(name)
            if sha(payload) != row["sha256"] or (kind == "input" and row["sha256"] != PINS[name]):
                raise ValueError("member identity differs")
            if kind == "input" and name == WHEEL:
                check_wheel(payload)
        output.mkdir(mode=0o700)
        for name in names:
            target = output / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(archive.read(name))
    return dict(schema=SCHEMA, campaign=CAMPAIGN, kind=kind, archive_sha256=expected_sha,
                files_verified=len(names)-1, release_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack-input", "unpack-input", "pack-result", "unpack-result"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.operation == "pack-input":
        if args.artifact is None or args.sha256 is not None:
            parser.error("pack-input requires the saved artifact directory")
        paths = inventory(args.input)
        if set(paths) != {name for name in PINS if not name.startswith("artifact/")}:
            parser.error("payload has extra or missing files")
        paths.update({name: args.artifact / name.removeprefix("artifact/")
                      for name in PINS if name.startswith("artifact/")})
        result = pack(paths, args.output, "input")
    elif args.operation == "pack-result":
        if args.artifact is not None or args.sha256 is not None:
            parser.error("pack-result accepts only evidence paths")
        if args.output.resolve().is_relative_to(args.input.resolve()):
            parser.error("result archive must be outside evidence")
        result = pack(inventory(args.input), args.output, "result")
    else:
        if args.sha256 is None or args.artifact is not None:
            parser.error("unpack requires the expected archive hash")
        result = unpack(args.input, args.output, args.sha256, args.operation.removeprefix("unpack-"))
    print(encoded(result).decode(), end="")


if __name__ == "__main__":
    main()
