from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib

import pytest

from spiraltorch import hf_cli


PY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PY_ROOT / "examples"
CLI_PATH = PY_ROOT / "spiraltorch" / "hf_cli.py"
MANIFEST_PATH = PY_ROOT / "spiraltorch" / "hf_cli_payloads.json"


def _local_example_modules() -> dict[str, Path]:
    return {path.stem: path for path in EXAMPLES_ROOT.glob("*.py")}


def _entrypoint_payloads() -> set[str]:
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    return {
        str(node.args[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_example"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def _payload_closure(roots: set[str]) -> set[str]:
    modules = _local_example_modules()
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        filename = pending.pop()
        if filename in seen:
            continue
        seen.add(filename)
        path = EXAMPLES_ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dependencies: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.update(
                    alias.name.split(".")[0]
                    for alias in node.names
                    if alias.name.split(".")[0] in modules
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".")[0]
                if module in modules:
                    dependencies.add(module)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "with_name"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                value = node.args[0].value
                if (
                    isinstance(value, str)
                    and value.endswith(".py")
                    and (EXAMPLES_ROOT / value).is_file()
                ):
                    dependencies.add(Path(value).stem)
        pending.extend(
            f"{module}.py" for module in dependencies if f"{module}.py" not in seen
        )
    return seen


def test_payload_manifest_matches_every_delegated_cli_and_dependency() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entrypoints = _entrypoint_payloads()
    required = _payload_closure(entrypoints)

    assert manifest["schema"] == "spiraltorch.hf_cli_payloads.v1"
    assert set(manifest["entrypoint_payloads"]) == entrypoints
    assert set(manifest["required_payloads"]) == required
    assert set(hf_cli.HF_CLI_EXAMPLE_PAYLOADS) == entrypoints
    assert set(hf_cli.HF_CLI_REQUIRED_PAYLOADS) == required
    assert all((EXAMPLES_ROOT / filename).is_file() for filename in required)


def test_maturin_includes_cli_payloads_in_wheel_and_sdist() -> None:
    pyproject = tomllib.loads((PY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = pyproject["tool"]["maturin"]["include"]

    assert {
        "path": "examples/*.py",
        "format": ["sdist", "wheel"],
    } in includes


def test_example_path_rejects_payloads_outside_the_manifest() -> None:
    with pytest.raises(ValueError, match="unsupported SpiralTorch HF CLI payload"):
        hf_cli._example_path("../spiraltorch/__init__.py")
