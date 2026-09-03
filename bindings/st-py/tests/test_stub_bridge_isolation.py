from __future__ import annotations

import builtins
import contextlib
import importlib.machinery
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys
import types
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[3]
BRIDGE = "_spiraltorch_stub_py_bridge"


def _graph():
    roots = ("spiraltorch", "spiraltorch_native", "spiral_rl", "rl", BRIDGE, "torch")
    return {
        name: value
        for name, value in sys.modules.items()
        if name in roots or any(name.startswith(root + ".") for root in roots[:-1])
    }


@contextlib.contextmanager
def _source_helpers():
    name = "_spiral_boundary_test_helpers"
    previous = {
        key: value
        for key, value in sys.modules.items()
        if key == name or key.startswith(name + ".")
    }
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "bindings" / "st-py" / "spiral" / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key, None)
        sys.modules.update(previous)


@pytest.mark.parametrize("torch_state", ["absent", "blocked", "module"])
def test_source_read_failure_restores_modules(
    monkeypatch, stub_spiraltorch_context, torch_state
):
    if torch_state == "absent":
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    else:
        value = None if torch_state == "blocked" else types.ModuleType("torch")
        monkeypatch.setitem(sys.modules, "torch", value)
    monkeypatch.setitem(sys.modules, BRIDGE + ".previous", types.ModuleType("previous"))
    before = _graph()
    original_read = Path.read_text

    def fail_source_read(path, *args, **kwargs):
        if path == ROOT / "spiraltorch" / "__init__.py":
            raise OSError("stub-source-read-failed")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_source_read)
    with pytest.raises(OSError, match="stub-source-read-failed"):
        with stub_spiraltorch_context(allow_numpy=False):
            pytest.fail("fixture must not yield after setup fails")
    assert _graph() == before


@pytest.mark.parametrize("allow_numpy", [False, True])
def test_install_failure_restores_import_hooks(
    monkeypatch, stub_spiraltorch_context, allow_numpy
):
    before = _graph()
    original_import = builtins.__import__
    original_find_spec = importlib.util.find_spec
    original_warn = warnings.warn

    def fail_install(message, *args, **kwargs):
        if str(message).startswith("Using SpiralTorch Python stub"):
            raise RuntimeError("stub-install-failed")
        return original_warn(message, *args, **kwargs)

    monkeypatch.setattr(warnings, "warn", fail_install)
    with pytest.raises(RuntimeError, match="stub-install-failed"):
        with stub_spiraltorch_context(allow_numpy=allow_numpy):
            pytest.fail("fixture must not yield after installation fails")
    assert builtins.__import__ is original_import
    assert importlib.util.find_spec is original_find_spec
    assert _graph() == before


def test_bridge_loader_failure_rolls_back_private_graph(
    monkeypatch, stub_spiraltorch_context
):
    pytest.importorskip("numpy")
    original_exec = importlib.machinery.SourceFileLoader.exec_module

    def fail_bridge(loader, module):
        if module.__name__ == BRIDGE:
            sys.modules[BRIDGE + ".partial"] = types.ModuleType("partial")
            raise RuntimeError("bridge-loader-failed")
        return original_exec(loader, module)

    before = _graph()
    with stub_spiraltorch_context(allow_numpy=False) as module:
        previous = types.ModuleType(BRIDGE)
        sys.modules[BRIDGE] = previous
        monkeypatch.setattr(importlib.machinery.SourceFileLoader, "exec_module", fail_bridge)
        with pytest.raises(RuntimeError, match="bridge-loader-failed"):
            module._install_stub_bindings(
                module, ModuleNotFoundError("spiraltorch", name="spiraltorch")
            )
        assert sys.modules[BRIDGE] is previous
        assert not any(name.startswith(BRIDGE + ".") for name in sys.modules)
    assert _graph() == before


def test_repeated_numpy_bridge_uses_shipped_helpers_and_current_tensor(
    stub_spiraltorch_context,
):
    np = pytest.importorskip("numpy")
    before = _graph()
    tensors = []
    for _ in range(2):
        with stub_spiraltorch_context(allow_numpy=True) as module:
            assert "numpy" in module.available_stub_backends()
            tensors.append(module.Tensor)
            helper = sys.modules["spiraltorch.hypergrad"]
            assert helper.st is module
            assert Path(helper.__file__).resolve() == (
                ROOT / "bindings" / "st-py" / "spiral" / "hypergrad.py"
            )
            assert callable(module.hypergrad)
            with pytest.raises(RuntimeError, match="native extension"):
                with module.hypergrad_session(1, 2):
                    pytest.fail("stub must not pretend to create a native tape")
            hints = module.suggest_hypergrad_operator({"summary": {"mean_abs": 1.0}})
            assert all(math.isfinite(value) for value in hints.values())
            assert "dormancy" in hints
            np.testing.assert_array_equal(
                module.normalize_batch([[1.0, 2.0]]), [[-1.0, 1.0]]
            )
        assert _graph() == before
    assert tensors[0] is not tensors[1]


def test_body_failure_restores_module_graph(stub_spiraltorch_context):
    before = _graph()
    with pytest.raises(RuntimeError, match="body-failed"):
        with stub_spiraltorch_context(allow_numpy=False):
            raise RuntimeError("body-failed")
    assert _graph() == before


def test_source_helpers_do_not_require_native_export_on_import(stub_spiraltorch_context):
    with stub_spiraltorch_context(allow_numpy=False):
        with _source_helpers() as helpers:
            assert helpers.__name__ + ".export" not in sys.modules
            assert callable(helpers.hypergrad_summary_dict)
            assert "ExportConfig" in dir(helpers)
            with pytest.raises(ImportError, match="export"):
                _ = helpers.ExportConfig
            assert helpers.__name__ + ".export" not in sys.modules


def test_source_export_helpers_still_resolve_native_implementation():
    with _source_helpers() as helpers:
        assert helpers.__name__ + ".export" not in sys.modules
        config_type = helpers.ExportConfig
        exported = sys.modules[helpers.__name__ + ".export"]
        assert config_type is exported.ExportConfig
        assert helpers.ExportConfig is config_type
        pipeline = helpers.ExportPipeline([0.1, -0.3, 0.7], config_type())
        report = pipeline.run(apply_pruning=False)
        assert "quantization" in report
        assert len(pipeline.weights) == 3


@pytest.fixture
def source_without_native(tmp_path):
    shutil.copytree(
        ROOT / "bindings" / "st-py" / "spiraltorch",
        tmp_path / "spiraltorch",
        ignore=shutil.ignore_patterns("*.so", "*.pyd", "*.dylib", "__pycache__"),
    )
    return tmp_path


def test_packaged_source_helpers_import_without_native_or_optional_dependencies(
    source_without_native,
):
    code = """
import spiraltorch
from spiraltorch.ecosystem import (
    bound_external_state_tensors, external_tensor_last_token, tensor_from_external,
)
from spiraltorch.nn import (
    CategoricalCrossEntropy, SoftmaxCrossEntropy, compare_sparse_finetune_summaries,
    LoraLinear, ZSpaceProjector, sparse_classification_delta,
)

assert spiraltorch._rs is None
assert callable(compare_sparse_finetune_summaries)
assert callable(LoraLinear)
assert CategoricalCrossEntropy is SoftmaxCrossEntropy
assert callable(ZSpaceProjector)
assert callable(sparse_classification_delta)
for loss_type in (CategoricalCrossEntropy, SoftmaxCrossEntropy):
    try:
        loss_type()
    except RuntimeError as exc:
        assert "compiled nn extension" in str(exc), str(exc)
    else:
        raise AssertionError("native loss must not be simulated")
assert bound_external_state_tensors({"w": [[1, 2], [3, 4]]}, {"w": (1, 1)}) == {"w": [[1]]}
assert external_tensor_last_token([[[1, 2], [3, 4]]]) == [3, 4]
try:
    tensor_from_external([[1, 2]])
except RuntimeError as exc:
    assert "native extension" in str(exc), str(exc)
else:
    raise AssertionError("native Tensor conversion must not be simulated")
"""
    result = subprocess.run(
        [
            sys.executable, "-I", "-S", "-c",
            f"import sys; sys.path.insert(0, {str(source_without_native)!r});\n" + code,
        ],
        cwd=source_without_native,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "example", ["byte_lm_finetune.py", "checkpoint_preflight.py", "byte_lm_mlp_lora_sweep.py"]
)
def test_example_help_without_native_or_optional_dependencies(source_without_native, example):
    examples = ROOT / "bindings" / "st-py" / "examples"
    code = (
        f"import sys, runpy; sys.path[:0] = {[str(source_without_native), str(examples)]!r};\n"
        "import spiraltorch; assert spiraltorch._rs is None;\n"
        f"sys.argv = [{example!r}, '--help'];\n"
        f"runpy.run_path({str(examples / example)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=source_without_native,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
