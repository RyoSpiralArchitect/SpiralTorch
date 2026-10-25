"""Read-only identity checks for exact installed release wheels; no native import."""
from email.parser import BytesParser
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

MAX_MEMBER = 2 * 1024 ** 3
MAX_TOTAL = 4 * 1024 ** 3
PACKAGE_ROOTS = ("spiraltorch", "spiral", "examples")


def sha256(path):
    with Path(path).open("rb") as stream:
        return stream_digest(stream)


def stream_digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def wheel_inventory(path):
    """Freeze every original wheel file, including the facade and metadata."""
    path = Path(path)
    original_digest = sha256(path)
    files = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > 8192:
            raise ValueError("wheel entry count is outside the fixed bound")
        total = 0
        seen = set()
        for entry in entries:
            name = entry.filename
            parts = PurePosixPath(name).parts
            mode = entry.external_attr >> 16
            if (name in seen or not parts or name.startswith("/") or "\\" in name
                    or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
                    or stat.S_ISLNK(mode) or entry.flag_bits & 1):
                raise ValueError("wheel contains an unsafe or duplicate member")
            seen.add(name)
            if entry.is_dir():
                continue
            if not 0 <= entry.file_size <= MAX_MEMBER:
                raise ValueError("wheel member exceeds the fixed size bound")
            total += entry.file_size
            if total > MAX_TOTAL:
                raise ValueError("wheel exceeds the fixed total size bound")
            with archive.open(entry) as stream:
                files[name] = {"sha256": stream_digest(stream), "bytes": entry.file_size}
        metadata = [name for name in files if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("expected one wheel distribution")
        info = metadata[0].rsplit("/", 1)[0]
        if files[metadata[0]]["bytes"] > 1024 * 1024:
            raise ValueError("wheel metadata exceeds the fixed size bound")
        message = BytesParser().parsebytes(archive.read(metadata[0]))
        version = message.get("Version", "")
        if (message.get("Name", "").lower() != "spiraltorch"
                or not re.fullmatch(r"\d+\.\d+\.\d+", version)
                or not path.name.startswith(f"spiraltorch-{version}-") or path.suffix != ".whl"
                or info != f"spiraltorch-{version}.dist-info"):
            raise ValueError("wheel distribution, filename or release version differs")
        if ("spiraltorch/__init__.py" not in files or info + "/WHEEL" not in files
                or info + "/RECORD" not in files
                or any(not name.startswith(tuple(root + "/" for root in PACKAGE_ROOTS) + (info + "/",))
                       for name in files)
                or any(name.endswith((".pyc", ".pyo")) for name in files)):
            raise ValueError("unexpected wheel layout or precompiled Python code")
        if files[info + "/WHEEL"]["bytes"] > 16 * 1024:
            raise ValueError("wheel tags exceed the fixed size bound")
        tags = BytesParser().parsebytes(archive.read(info + "/WHEEL")).get_all("Tag", [])
    native = [name for name in files if PurePosixPath(name).name.startswith("spiraltorch.")
              and name.endswith((".so", ".pyd"))]
    if len(native) != 1 or not tags:
        raise ValueError("expected one native extension and explicit wheel tags")
    platforms = set()
    for tag in tags:
        fields = tag.split("-")
        if len(fields) != 3 or fields[1] != "abi3" or fields[0] != "cp38":
            raise ValueError("qualification requires the release cp38-abi3 wheels")
        target = fields[2]
        if target.startswith("macosx_") and target.endswith(("_x86_64", "_arm64", "_universal2")):
            platforms.add("macos")
        elif target.startswith(("manylinux_", "manylinux2014_")) and target.endswith("_x86_64"):
            platforms.add("linux")
        elif target == "win_amd64":
            platforms.add("windows")
        else:
            raise ValueError("unexpected release wheel platform")
    if len(platforms) != 1:
        raise ValueError("wheel mixes platform families")
    if native[0].endswith(".pyd") != (platforms == {"windows"}):
        raise ValueError("native extension does not match the wheel platform")
    if platforms == {"macos"} and not any(tag.endswith("_universal2") for tag in tags):
        raise ValueError("macOS release wheel must include universal2")
    filename_tags = path.name[:-4].split("-")[2:]
    if len(filename_tags) != 3 or set(tags) != {
            f"{python}-{abi}-{target}" for python in filename_tags[0].split(".")
            for abi in filename_tags[1].split(".") for target in filename_tags[2].split(".")}:
        raise ValueError("wheel filename tags and metadata differ")
    if sha256(path) != original_digest:
        raise ValueError("wheel changed during inventory")
    return {"filename": path.name, "sha256": original_digest, "bytes": path.stat().st_size,
            "version": version, "platform": platforms.pop(), "tags": sorted(tags),
            "native": dict(path=native[0], **files[native[0]]), "files": files}


def verify_installed_files(inventory, root, package_origin, metadata_version, loaded_origins=()):
    """Check source bytes before import; install with pip --no-compile.

    RECORD is rewritten by installers. The other original metadata is checked;
    installer-added metadata is not executable package code and is not asserted
    to have been part of the wheel.
    """
    root = Path(root).resolve(strict=True)
    package = root / "spiraltorch"
    files = inventory["files"]
    if metadata_version != inventory["version"]:
        raise ValueError("installed distribution version differs")
    if Path(package_origin).resolve(strict=True) != package / "__init__.py":
        raise ValueError("Python package origin is not the admitted installation")
    expected = {name for name in files if name.startswith(tuple(p + "/" for p in PACKAGE_ROOTS))}
    actual = set()
    for package_name in PACKAGE_ROOTS:
        directory = root / package_name
        if directory.is_symlink():
            raise ValueError("installed package contains a symbolic link")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("installed package contains a symbolic link")
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError("installed package has missing, extra or cached Python files")
    for name, entry in files.items():
        if name.endswith(".dist-info/RECORD"):
            continue
        path = root / name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]):
            raise ValueError("installed file differs from the wheel: " + name)
    for origin in loaded_origins:
        path = Path(origin).resolve(strict=True)
        if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in expected:
            raise ValueError("loaded SpiralTorch module is outside the admitted wheel")
    return {"version": inventory["version"], "wheel_sha256": inventory["sha256"],
            "package_files": len(expected), "native_sha256": inventory["native"]["sha256"],
            "all_original_files_match": True, "python_bytecode_absent": True}


def installed_identity(inventory):
    distribution = importlib.metadata.distribution("spiraltorch")
    spec = importlib.util.find_spec("spiraltorch")
    if spec is None or not spec.origin:
        raise ValueError("SpiralTorch package origin is unavailable")
    origins = [getattr(module, "__file__", None) for name, module in tuple(sys.modules.items())
               if name.split(".", 1)[0] in PACKAGE_ROOTS]
    return verify_installed_files(inventory, distribution.locate_file(""), spec.origin,
                                  distribution.version, [value for value in origins if value])
