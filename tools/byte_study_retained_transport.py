#!/usr/bin/env python3
"""Private archive transport only; no native import or GPU execution."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

PACKET_SHA256 = "be7dc0c48c6b23c93964ec9319ce8ae490ea473ddefa1a715c792a2ff61e4605"
PACKET_NAMES = {
    "version-9/" + name for name in (
        "request.json", "full-report.json", "resumed-report.json",
        "first-report.json", "first-checkpoint.json", "resumed-checkpoint.json",
        "runtime.json", "summary.json",
    )
}
MAX_PACKET = 8 * 1024 * 1024
MAX_MEMBER = 32 * 1024 * 1024
MAX_UNPACKED = 128 * 1024 * 1024
MAX_RESULT = 2 * 1024 * 1024 * 1024
MANIFEST_NAME = "transport-manifest.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, output):
    archive, output = Path(archive), Path(output)
    if archive.stat().st_size > MAX_PACKET or digest(archive) != PACKET_SHA256:
        raise ValueError("original packet identity differs")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if (len(names) != len(PACKET_NAMES) or set(names) != PACKET_NAMES
                or any(not member.isfile() or member.size < 0
                       or member.size > MAX_MEMBER for member in members)
                or sum(member.size for member in members) > MAX_UNPACKED):
            raise ValueError("packet must contain exactly the eight original regular files")
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        (output / "version-9").mkdir(mode=0o700)
        for member in members:
            target = output / member.name
            with bundle.extractfile(member) as source, target.open("xb") as destination:
                target.chmod(0o600)
                payload = source.read(MAX_MEMBER + 1)
                if len(payload) != member.size:
                    raise ValueError("packet member size differs")
                destination.write(payload)


def pack(root, output):
    root, output = Path(root), Path(output)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evidence root must be a regular directory")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("output must be outside the evidence directory")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in paths):
        raise ValueError("evidence must not contain links or special files")
    files = [path for path in paths if path.is_file()]
    if not files or any(str(path.relative_to(root)) == MANIFEST_NAME for path in files):
        raise ValueError("empty evidence or reserved manifest name")
    if sum(path.stat().st_size for path in files) > MAX_RESULT:
        raise ValueError("evidence archive exceeds its bounded size")
    manifest = {
        "schema": "spiraltorch.retained_transport.v1", "release_ready": False,
        "files": {str(path.relative_to(root)): {
            "sha256": digest(path), "bytes": path.stat().st_size,
        } for path in files},
    }
    raw = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    with output.open("xb") as stream:
        output.chmod(0o600)
        with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size, info.mode = len(raw), 0o600
            bundle.addfile(info, io.BytesIO(raw))
            for path in files:
                name = str(path.relative_to(root))
                info = tarfile.TarInfo(name)
                info.size, info.mode = manifest["files"][name]["bytes"], 0o600
                with path.open("rb") as source:
                    bundle.addfile(info, source)
                if digest(path) != manifest["files"][name]["sha256"]:
                    raise ValueError("evidence changed during packing")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("unpack", "pack"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "unpack":
        unpack(args.input, args.output)
    else:
        pack(args.input, args.output)


if __name__ == "__main__":
    main()
