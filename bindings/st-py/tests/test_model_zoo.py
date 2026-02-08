from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


_BINDINGS_DIR = Path(__file__).resolve().parents[1] / "spiraltorch"


def _load_model_zoo(stub_spiraltorch, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "spiraltorch.model_zoo", raising=False)
    package_paths = list(getattr(stub_spiraltorch, "__path__", []))
    bindings_path = str(_BINDINGS_DIR)
    if bindings_path not in package_paths:
        package_paths.append(bindings_path)
        stub_spiraltorch.__path__ = package_paths
    return importlib.import_module("spiraltorch.model_zoo")


def _write_script(path: Path) -> None:
    path.write_text("print('ok')\n", encoding="utf-8")


def test_list_models_respects_internal_filter(stub_spiraltorch, monkeypatch, tmp_path: Path):
    scripts = tmp_path / "models" / "python"
    scripts.mkdir(parents=True)
    _write_script(scripts / "mlp_regression.py")
    _write_script(scripts / "custom_recipe.py")
    _write_script(scripts / "_softlogic_cli.py")

    monkeypatch.setenv("SPIRALTORCH_MODEL_ZOO_ROOT", str(tmp_path))
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)

    public = model_zoo.list_models()
    all_entries = model_zoo.list_models(include_internal=True)

    public_keys = {entry.key for entry in public}
    all_keys = {entry.key for entry in all_entries}
    assert "mlp_regression" in public_keys
    assert "custom_recipe" in public_keys
    assert "_softlogic_cli" not in public_keys
    assert "_softlogic_cli" in all_keys


def test_find_and_resolve_model_script(stub_spiraltorch, monkeypatch, tmp_path: Path):
    scripts = tmp_path / "models" / "python"
    scripts.mkdir(parents=True)
    _write_script(scripts / "llm_char_finetune.py")

    monkeypatch.setenv("SPIRALTORCH_MODEL_ZOO_ROOT", str(tmp_path))
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)

    entry = model_zoo.find_model("llm_char")
    assert entry.key == "llm_char_finetune"
    resolved = model_zoo.resolve_model_script("llm_char_finetune")
    assert resolved == (scripts / "llm_char_finetune.py").resolve()


def test_find_model_supports_contains_match(stub_spiraltorch, monkeypatch, tmp_path: Path):
    scripts = tmp_path / "models" / "python"
    scripts.mkdir(parents=True)
    _write_script(scripts / "my_custom_recipe.py")

    monkeypatch.setenv("SPIRALTORCH_MODEL_ZOO_ROOT", str(tmp_path))
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)

    entry = model_zoo.find_model("custom_recipe")
    assert entry.key == "my_custom_recipe"


def test_build_and_run_dry_command(stub_spiraltorch, monkeypatch, tmp_path: Path):
    scripts = tmp_path / "models" / "python"
    scripts.mkdir(parents=True)
    _write_script(scripts / "zconv_classification.py")

    monkeypatch.setenv("SPIRALTORCH_MODEL_ZOO_ROOT", str(tmp_path))
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)

    command = model_zoo.build_model_command(
        "zconv_classification",
        "--epochs",
        "1",
        python_executable="python-test",
    )
    assert command[0] == "python-test"
    assert command[1].endswith("zconv_classification.py")
    assert command[-2:] == ["--epochs", "1"]

    dry = model_zoo.run_model(
        "zconv_classification",
        "--help",
        python_executable="python-test",
        dry_run=True,
    )
    assert isinstance(dry, list)
    assert dry[0] == "python-test"
    assert dry[-1] == "--help"


def test_summary_counts_models(stub_spiraltorch, monkeypatch, tmp_path: Path):
    scripts = tmp_path / "models" / "python"
    scripts.mkdir(parents=True)
    _write_script(scripts / "mlp_regression.py")
    _write_script(scripts / "vision_conv_pool_classification.py")

    monkeypatch.setenv("SPIRALTORCH_MODEL_ZOO_ROOT", str(tmp_path))
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)

    summary = model_zoo.model_zoo_summary()
    assert summary["count"] == 2
    assert summary["tasks"]["classification"] == 1
    assert summary["tasks"]["regression"] == 1


def test_available_only_filters_metadata_only_entries(stub_spiraltorch, monkeypatch):
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)
    monkeypatch.setattr(model_zoo, "_iter_script_paths", lambda *args, **kwargs: [])

    with_missing = model_zoo.list_models()
    available_only = model_zoo.list_models(available_only=True)

    assert with_missing
    assert all(entry.script_path is None for entry in with_missing)
    assert available_only == []


def test_suggest_models_ranks_preferred_tags(stub_spiraltorch, monkeypatch, tmp_path: Path):
    scripts = tmp_path / "models" / "python"
    scripts.mkdir(parents=True)
    _write_script(scripts / "llm_char_finetune.py")
    _write_script(scripts / "llm_char_coherence_scan.py")

    monkeypatch.setenv("SPIRALTORCH_MODEL_ZOO_ROOT", str(tmp_path))
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)

    entries = model_zoo.suggest_models(
        "llm_char",
        available_only=True,
        prefer_tags=["coherence"],
    )
    assert entries
    assert entries[0].key == "llm_char_coherence_scan"


def test_recommend_model_respects_required_tags(stub_spiraltorch, monkeypatch, tmp_path: Path):
    scripts = tmp_path / "models" / "python"
    scripts.mkdir(parents=True)
    _write_script(scripts / "zconv_classification.py")
    _write_script(scripts / "vision_conv_pool_classification.py")

    monkeypatch.setenv("SPIRALTORCH_MODEL_ZOO_ROOT", str(tmp_path))
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)

    entry = model_zoo.recommend_model(
        task="classification",
        required_tags=["zspace"],
        available_only=True,
    )
    assert entry.key == "zconv_classification"


def test_cli_suggest_json_output(stub_spiraltorch, monkeypatch, tmp_path: Path, capsys):
    scripts = tmp_path / "models" / "python"
    scripts.mkdir(parents=True)
    _write_script(scripts / "llm_char_finetune.py")
    _write_script(scripts / "llm_char_coherence_scan.py")

    monkeypatch.setenv("SPIRALTORCH_MODEL_ZOO_ROOT", str(tmp_path))
    model_zoo = _load_model_zoo(stub_spiraltorch, monkeypatch)

    code = model_zoo.main(
        [
            "suggest",
            "llm_char",
            "--root",
            str(tmp_path),
            "--available-only",
            "--prefer-tag",
            "coherence",
            "--limit",
            "1",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["key"] == "llm_char_coherence_scan"
