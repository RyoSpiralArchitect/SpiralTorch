"""Pinned, single-attempt wheel builds and non-release original-byte receipts.

The build command compiles only when explicitly invoked. Pins, archive checks
and installation checks never import SpiralTorch or execute a GPU operation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import site
import subprocess
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_artifacts as artifacts
import owned_process

RUST = "1.99.1"
MATURIN = "1.15.0"
SCHEMA = "spiraltorch.wheel_build_evidence.v1"
ROLES = ("ci-gpu", "ci-portable", "manual", "release")
MAX_RETAINED_OUTPUTS = 8
POSIX_BUILD_SECONDS = 6600
WINDOWS_STEP_SECONDS = 7200
SOURCE_FILES = ("Cargo.lock", "Cargo.toml", "rust-toolchain.toml",
                "bindings/st-py/Cargo.toml", "bindings/st-py/pyproject.toml")


def write_json(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(raw)


def build_plan(role, system, wheel_dir):
    if role not in ROLES or system not in ("Darwin", "Linux", "Windows"):
        raise ValueError("unknown build role or platform")
    if role == "ci-gpu" and system != "Darwin":
        raise ValueError("ordinary GPU CI must retain its native macOS route")
    if role == "ci-portable" and system != "Linux":
        raise ValueError("portable CI must retain its Linux route")
    public_layout = role in ("manual", "release")
    features = [] if role == "ci-gpu" else ["logic", "kdsl"]
    target = "universal2-apple-darwin" if public_layout and system == "Darwin" else None
    compatibility = "manylinux2014" if public_layout and system == "Linux" else None
    command = ["maturin", "build", "--release", "--locked", "-m", "bindings/st-py/Cargo.toml",
               "--out", str(wheel_dir)]
    if target:
        command += ["--target", target]
    if compatibility:
        command += ["--compatibility", compatibility, "--zig"]
    if features:
        command += ["--features", ",".join(features)]
    return {"role": role, "system": system, "profile": "release", "locked": True,
            "default_features": True, "explicit_features": features, "target": target,
            "compatibility": compatibility, "macos_deployment_target": "14.0" if target else None,
            "command": command}


def probe(command, root):
    result = subprocess.run(command, cwd=root, check=True, capture_output=True,
                            text=True, timeout=30)
    if len(result.stdout) > 64 * 1024:
        raise ValueError("tool identity exceeds the receipt bound")
    return result.stdout.strip()


def source_identity(root):
    if probe(["git", "status", "--porcelain", "--untracked-files=all"], root):
        raise ValueError("build provenance requires a clean committed source tree")
    commit, tree = probe(["git", "rev-parse", "HEAD", "HEAD^{tree}"], root).splitlines()
    version = tomllib.loads((root / "bindings/st-py/pyproject.toml").read_text())["project"]["version"]
    native_version = tomllib.loads((root / "bindings/st-py/Cargo.toml").read_text())["package"]["version"]
    if version != native_version:
        raise ValueError("Python/native source versions differ")
    return {"commit": commit, "tree": tree, "package_version": version,
            "files": {name: artifacts.sha256(root / name) for name in SOURCE_FILES}}


def build_inputs(root, plan):
    # Overrides must not silently change the pinned recipe; never log their values.
    forbidden = {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC", "RUSTC_WRAPPER",
                 "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS",
                 "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"}
    if any(value and (key in forbidden or key.startswith("CARGO_PROFILE_") or
                     key.startswith("CARGO_TARGET_") and key.endswith("_RUSTFLAGS"))
           for key, value in os.environ.items()):
        raise ValueError("undeclared compiler/profile override")
    if os.environ.get("MACOSX_DEPLOYMENT_TARGET") != plan["macos_deployment_target"]:
        raise ValueError("deployment target differs from the declared build role")
    if os.environ.get("RUSTUP_TOOLCHAIN") != RUST:
        raise ValueError("the fixed Rust toolchain must be explicit for child commands")
    rust = probe(["rustc", "-vV"], root)
    cargo = probe(["cargo", "--version"], root)
    maturin = probe(["maturin", "--version"], root)
    toolchain_cargo = probe(["rustup", "run", RUST, "cargo", "--version"], root)
    if (f"release: {RUST}" not in rust.splitlines() or
            cargo != toolchain_cargo or not cargo.startswith("cargo ") or maturin != f"maturin {MATURIN}"):
        raise ValueError("actual build tools differ from the fixed pins")
    configs = {}
    for depth, parent in enumerate((root, *root.parents)):
        for name in ("config", "config.toml"):
            path = parent / ".cargo" / name
            if path.is_file():
                configs[f"ancestor-{depth}/{name}"] = artifacts.sha256(path)
    cargo_home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    for name in ("config", "config.toml"):
        path = cargo_home / name
        if path.is_file():
            configs[f"cargo-home/{name}"] = artifacts.sha256(path)
    runner = {}
    for name in ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH",
                 "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "GITHUB_SHA"):
        value = os.environ.get(name)
        if value is not None:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value):
                raise ValueError("invalid runner identity field")
            runner[name] = value
    return {"source": source_identity(root), "requested_pins": {"rust": RUST, "maturin": MATURIN},
            "observed_tools": {"rustc": rust, "cargo": cargo, "maturin": maturin,
                               "pinned_toolchain_cargo": toolchain_cargo},
            "python": sys.version, "machine": platform.machine(), "os_release": platform.release(),
            "sdk_version": probe(["xcrun", "--sdk", "macosx", "--show-sdk-version"], root)
            if plan["system"] == "Darwin" else None,
            "cargo_config_sha256": configs, "runner": runner,
            "build_environment_sha256": {
                name: hashlib.sha256(os.environ[name].encode()).hexdigest()
                for name in ("CC", "CXX", "AR", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
                             "SDKROOT", "DEVELOPER_DIR", "CARGO_INCREMENTAL", "RUSTUP_TOOLCHAIN",
                             "PYO3_PYTHON", "PYO3_CONFIG_FILE", "PYO3_CROSS", "PYO3_CROSS_LIB_DIR")
                if name in os.environ},
            "python_build_packages": installed_build_packages()}


def build_package_locations():
    # -S on Python <=3.13 hides the venv prefix. Keep the executable's lexical
    # path: resolving a venv symlink would point at the base interpreter.
    executable = Path(sys.executable).absolute()
    config = next((base / "pyvenv.cfg" for base in (executable.parent, executable.parent.parent)
                   if (base / "pyvenv.cfg").is_file()), None)
    if config is None:
        return site.getsitepackages([sys.prefix, sys.exec_prefix])
    if config.stat().st_size > 16 * 1024:
        raise ValueError("venv configuration exceeds the metadata bound")
    values = dict(line.strip().lower().split("=", 1) for line in config.read_text().splitlines() if "=" in line)
    values = {key.strip(): value.strip() for key, value in values.items()}
    prefixes = [str(config.parent)]
    if values.get("include-system-site-packages", "false") == "true":
        prefixes += [sys.base_prefix, sys.base_exec_prefix]
    # getsitepackages only computes paths; it does not run site.main, .pth
    # files or sitecustomize. It also preserves CPython's lib64/lib ordering.
    return site.getsitepackages(prefixes)


def installed_build_packages():
    versions = dict.fromkeys(("pip", "wheel", "maturin", "ziglang"))
    # Build preflight runs with -S: inspect distribution metadata explicitly,
    # without executing site hooks or importing these packages.
    for location in build_package_locations():
        seen = set()
        for distribution in importlib.metadata.distributions(path=[location]):
            name = distribution.metadata.get("Name", "").lower().replace("_", "-")
            if name in versions:
                if name in seen:
                    raise ValueError("ambiguous Python build-tool installation")
                seen.add(name)
                if versions[name] is None:
                    versions[name] = distribution.version
    return versions


def validate_wheel_role(inventory, plan):
    expected = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}[plan["system"]]
    if inventory["platform"] != expected:
        raise ValueError("wheel platform differs from the build role")
    tags = inventory["tags"]
    universal = any(tag.endswith("_universal2") for tag in tags)
    if expected == "macos":
        if universal != (plan["target"] == "universal2-apple-darwin"):
            raise ValueError("native and universal2 roles cannot substitute for one another")
        if universal and not all(tag.startswith("cp38-abi3-macosx_14_0_") for tag in tags):
            raise ValueError("release-layout macOS floor must stay 14.0")
    if plan["compatibility"] and not all(
            tag.endswith(("-manylinux2014_x86_64", "-manylinux_2_17_x86_64")) for tag in tags):
        raise ValueError("release-layout Linux compatibility must stay manylinux2014")


def run_build_command(command, root, timeout_seconds):
    if os.name == "posix":
        return owned_process.run(command, cwd=root, timeout=timeout_seconds, stdout=None, stderr=None)
    # Hosted Windows cancellation belongs to the Actions step's process owner.
    # Do not pretend subprocess.run(timeout=...) can clean up a Cargo child tree.
    if os.name != "nt" or os.environ.get("GITHUB_ACTIONS") != "true" or timeout_seconds != WINDOWS_STEP_SECONDS:
        raise ValueError("Windows builds require the declared 120-minute Actions step")
    return subprocess.run(command, cwd=root, check=False)


def write_outcome(path, value, *, preserve_error=None):
    if preserve_error is None:
        preserve_error = sys.exc_info()[0] is not None
    try:
        write_json(path, value)
    except Exception:
        if not preserve_error:
            raise
        print("Could not persist outcome; original failure is preserved.", file=sys.stderr)


def retain_build_outputs(wheel_dir, evidence):
    """Preserve bounded raw outputs, including failed/ambiguous build results."""
    parent_error = sys.exc_info()[1]
    paths = sorted(wheel_dir.glob("*.whl"))
    record = {"schema": SCHEMA, "release_ready": False, "complete": False,
              "output_count": len(paths), "files": []}
    total = 0
    first_error = None
    try:
        if len(paths) > MAX_RETAINED_OUTPUTS:
            raise ValueError("wheel output count exceeds the retention bound")
        for path in paths:
            entry = {"filename": path.name, "status": "not_retained"}
            record["files"].append(entry)
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("wheel output must be a regular non-symlink file")
                size = path.stat().st_size
                if size > artifacts.MAX_TOTAL - total:
                    raise ValueError("wheel outputs exceed the total retention size bound")
                total += size
                original_digest = artifacts.sha256(path)
                destination = evidence / path.name
                with path.open("rb") as source, destination.open("xb") as target:
                    remaining = size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("original wheel shrank during retention")
                        target.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise ValueError("original wheel grew during retention")
                if (path.is_symlink() or path.stat().st_size != size or
                        artifacts.sha256(destination) != original_digest or
                        artifacts.sha256(path) != original_digest):
                    raise ValueError("original wheel changed during retention")
                entry.update(status="retained", sha256=original_digest, bytes=size)
            except Exception as error:
                entry["error_type"] = type(error).__name__
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        record["complete"] = True
        if len(record["files"]) == 1:
            original = {key: record["files"][0][key] for key in ("filename", "sha256", "bytes")}
            write_json(evidence / "original.json", {**original, "release_ready": False})
        return record["files"]
    finally:
        # Report retention-only errors to the outer build boundary, but never
        # replace an earlier local copy failure with a receipt-write failure.
        write_outcome(evidence / "outputs.json", record,
                      preserve_error=sys.exc_info()[1] is not parent_error)


def collect_after_step(wheel_dir, evidence):
    """Separate post-cancellation snapshot; never overwrite first-build records."""
    wheel_dir, evidence = (Path(value).resolve() for value in (wheel_dir, evidence))
    if wheel_dir == evidence or evidence.is_relative_to(wheel_dir) or wheel_dir.is_relative_to(evidence):
        raise ValueError("wheel output and evidence paths must be disjoint")
    recovery = evidence / "after-step"
    recovery.mkdir(parents=True, exist_ok=False)
    outcome = {"schema": SCHEMA, "release_ready": False, "status": "collection_failed"}
    try:
        outputs = retain_build_outputs(wheel_dir, recovery)
        outcome["status"] = "raw_outputs_retained_not_qualified" if outputs else "no_outputs_found"
        return outputs
    except BaseException as error:
        outcome["error_type"] = type(error).__name__
        raise
    finally:
        write_outcome(recovery / "outcome.json", outcome)


def run_build(root, role, wheel_dir, evidence, *, timeout_seconds=None):
    root, wheel_dir, evidence = (Path(value).resolve() for value in (root, wheel_dir, evidence))
    if timeout_seconds is None:
        timeout_seconds = WINDOWS_STEP_SECONDS if os.name == "nt" else POSIX_BUILD_SECONDS
    if not 1 <= timeout_seconds <= 21600:
        raise ValueError("build timeout must be within the outer six-hour bound")
    evidence.mkdir(parents=True, exist_ok=False)
    outcome = {"schema": SCHEMA, "release_ready": False, "role": role, "status": "preflight_failed"}
    try:
        plan = build_plan(role, platform.system(), wheel_dir)
        if wheel_dir == evidence or evidence.is_relative_to(wheel_dir) or wheel_dir.is_relative_to(evidence):
            raise ValueError("wheel output and evidence paths must be disjoint")
        wheel_dir.mkdir(parents=True, exist_ok=True)
        if list(wheel_dir.glob("*.whl")):
            raise ValueError("wheel output already contains an artifact")
        inputs = build_inputs(root, plan)
        write_json(evidence / "inputs.json", {"schema": SCHEMA, "plan": plan, **inputs,
                                              "timeout_seconds": timeout_seconds,
                                              "deadline_owner": "process_group" if os.name == "posix"
                                              else "github_actions_step", "release_ready": False})
        outcome["status"] = "build_failed"
        try:
            result = run_build_command(plan["command"], root, timeout_seconds)
            outcome["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError("original build command failed")
        finally:
            pending_error = sys.exc_info()[0] is not None
            if not pending_error:
                outcome["status"] = "retention_failed"
            try:
                outputs = retain_build_outputs(wheel_dir, evidence)
            except BaseException as error:
                if not pending_error:
                    raise
                outcome["retention_error_type"] = type(error).__name__
        outcome["status"] = "retention_failed"
        if len(outputs) != 1:
            raise ValueError("expected exactly one original wheel")
        wheel = wheel_dir / outputs[0]["filename"]
        inventory = artifacts.wheel_archive_inventory(evidence / wheel.name)
        validate_wheel_role(inventory, plan)
        if inventory["version"] != inputs["source"]["package_version"]:
            raise ValueError("wheel version differs from the original source")
        if (artifacts.wheel_archive_inventory(evidence / wheel.name) != inventory or
                artifacts.wheel_archive_inventory(wheel) != inventory):
            raise ValueError("original wheel changed during retention")
        if source_identity(root) != inputs["source"]:
            raise ValueError("source inputs changed during the build")
        receipt = {"schema": SCHEMA, "role": role, "release_ready": False,
                   "source_commit": inputs["source"]["commit"], "wheel_sha256": inventory["sha256"],
                   "inputs_sha256": artifacts.sha256(evidence / "inputs.json"),
                   "toolchain": inputs["observed_tools"]["rustc"],
                   "features": ["default", *plan["explicit_features"]], "wheel": inventory}
        write_json(evidence / "build.json", receipt)
        outcome["status"] = "built_and_retained_not_qualified"
        return receipt
    except subprocess.TimeoutExpired:
        outcome["status"] = "preflight_timeout" if outcome["status"] == "preflight_failed" else "build_timeout"
        raise
    except (KeyboardInterrupt, SystemExit) as error:
        outcome["status"] = "cancelled"
        outcome["error_type"] = type(error).__name__
        raise
    except BaseException as error:
        outcome["error_type"] = type(error).__name__
        raise
    finally:
        write_outcome(evidence / "outcome.json", outcome)


def _verify_install(evidence, wheel_dir):
    evidence = Path(evidence)
    receipt = json.loads((evidence / "build.json").read_text(encoding="utf-8"))
    inputs = json.loads((evidence / "inputs.json").read_text(encoding="utf-8"))
    if (receipt.get("schema") != SCHEMA or receipt.get("release_ready") is not False or
            receipt.get("role") not in ROLES or
            receipt.get("inputs_sha256") != artifacts.sha256(evidence / "inputs.json")):
        raise ValueError("invalid non-release build receipt")
    inventory = receipt["wheel"]
    plan = inputs["plan"]
    if (receipt["role"] != plan["role"] or receipt["source_commit"] != inputs["source"]["commit"] or
            receipt["wheel_sha256"] != inventory["sha256"] or
            plan != build_plan(plan["role"], plan["system"], plan["command"][7])):
        raise ValueError("inconsistent build receipt identity")
    filename = inventory["filename"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("invalid retained wheel filename")
    for base in (evidence, Path(wheel_dir)):
        if (base / filename).is_symlink() or artifacts.wheel_archive_inventory(base / filename) != inventory:
            raise ValueError("wheel differs from the original build receipt")
    validate_wheel_role(inventory, inputs["plan"])
    identity = artifacts.installed_identity(inventory)
    write_json(evidence / "installed.json", {"schema": SCHEMA, "release_ready": False,
                                            "build_receipt_sha256": artifacts.sha256(evidence / "build.json"),
                                            **identity})
    return identity


def verify_install(evidence, wheel_dir):
    outcome = {"schema": SCHEMA, "release_ready": False, "status": "installation_rejected"}
    try:
        identity = _verify_install(evidence, wheel_dir)
        outcome["status"] = "installation_verified_not_qualified"
        return identity
    except BaseException as error:
        outcome["error_type"] = type(error).__name__
        raise
    finally:
        write_outcome(Path(evidence) / "installation-outcome.json", outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pins")
    build = commands.add_parser("build")
    build.add_argument("--role", choices=ROLES, required=True)
    build.add_argument("--wheel-dir", type=Path, required=True)
    build.add_argument("--evidence", type=Path, required=True)
    build.add_argument("--timeout-seconds", type=int)
    collect = commands.add_parser("collect-after-step")
    collect.add_argument("--wheel-dir", type=Path, required=True)
    collect.add_argument("--evidence", type=Path, required=True)
    verify = commands.add_parser("verify-install")
    verify.add_argument("--wheel-dir", type=Path, required=True)
    verify.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pins":
        print(f"rust={RUST}\nmaturin={MATURIN}")
    elif args.command == "build":
        run_build(Path(__file__).resolve().parents[1], args.role, args.wheel_dir, args.evidence,
                  timeout_seconds=args.timeout_seconds)
    elif args.command == "collect-after-step":
        collect_after_step(args.wheel_dir, args.evidence)
    else:
        verify_install(args.evidence, args.wheel_dir)


if __name__ == "__main__":
    main()
