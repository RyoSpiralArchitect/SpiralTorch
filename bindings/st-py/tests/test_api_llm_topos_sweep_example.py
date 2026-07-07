from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

st = pytest.importorskip("spiraltorch")


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example():
    path = EXAMPLES_DIR / "api_llm_topos_sweep.py"
    spec = importlib.util.spec_from_file_location("api_llm_topos_sweep_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_llm_topos_sweep_example_offline_writes_report(tmp_path, capsys) -> None:
    example = _load_example()
    capsys.readouterr()

    example.main(
        [
            "--out-dir",
            str(tmp_path),
            "--prompt-limit",
            "2",
            "--context-prompt",
            "--include-penalties",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    report_path = tmp_path / "report.json"
    assert printed["report"] == str(report_path)
    assert printed["prompt_count"] == 2
    assert printed["labels"] == ["open", "contextual", "guarded"]
    assert printed["mode_counts"]["exploratory"] == 1
    assert printed["mode_counts"]["contextual"] == 1
    assert printed["mode_counts"]["guarded"] == 1
    assert printed["report_comparison"]["winners"]["widest_temperature_range"] == "current"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["kind"] == "spiraltorch.api_llm_topos_sweep_report"
    assert report["prompt_count"] == 2
    assert report["route_count"] == 3
    assert report["adapter_winners"]["highest_request_temperature"] == "open"
    assert (tmp_path / "traces" / "00-open.jsonl").exists()
    assert (tmp_path / "traces" / "01-contextual.jsonl").exists()
    assert (tmp_path / "traces" / "02-guarded.jsonl").exists()


def test_api_llm_topos_sweep_example_dry_run_writes_plan(tmp_path, capsys) -> None:
    example = _load_example()
    capsys.readouterr()

    example.main(["--out-dir", str(tmp_path), "--dry-run", "--prompt-limit", "1"])

    printed = json.loads(capsys.readouterr().out)
    assert printed["kind"] == "spiraltorch.api_llm_topos_sweep_plan"
    assert printed["prompt_count"] == 1
    assert printed["labels"] == ["open", "contextual", "guarded"]
    assert printed["live_provider"] == "offline"
    assert json.loads((tmp_path / "plan.json").read_text(encoding="utf-8")) == printed
