"""Installed CLI entrypoints for Hugging Face/Z-Space workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

from .hf_ft import (
    hf_finetune_model_profile_cli_args,
    hf_finetune_model_profile_lines,
    resolve_hf_finetune_model_profile,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_ROOT = _PACKAGE_ROOT / "examples"


def _example_path(name: str) -> Path:
    path = _EXAMPLES_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(
            "SpiralTorch HF CLI example payload is missing: "
            f"{path}. Reinstall the wheel or run the repository example directly."
        )
    return path


def _module_name(path: Path) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in path.stem)
    return f"_spiraltorch_installed_{token}"


def _load_example(name: str) -> ModuleType:
    path = _example_path(name)
    module_name = _module_name(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load SpiralTorch HF CLI example: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_EXAMPLES_ROOT))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop(module_name, None)
    return module


def _run_example(name: str, argv: Sequence[str] | None = None) -> int:
    module = _load_example(name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise AttributeError(f"SpiralTorch HF CLI example has no main(): {name}")
    return int(main(None if argv is None else list(argv)))


def profile_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a SpiralTorch Hugging Face fine-tune model profile.",
    )
    parser.add_argument("--model-configs", type=Path, default=None)
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--cli-args",
        action="store_true",
        help="Print bridge CLI args instead of compact profile lines.",
    )
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--no-training", action="store_true")
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args(argv)
    profile = resolve_hf_finetune_model_profile(
        args.model_configs,
        profile=args.model_profile,
    )
    if args.json:
        print(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cli_args:
        print(
            " ".join(
                hf_finetune_model_profile_cli_args(
                    profile,
                    include_model=not args.no_model,
                    include_training=not args.no_training,
                    include_generation=not args.no_generation,
                )
            )
        )
        return 0
    for line in hf_finetune_model_profile_lines(profile):
        print(line)
    return 0


def finetune_bridge_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_bridge.py", argv)


def finetune_sweep_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_sweep.py", argv)


def checkpoint_generation_control_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_checkpoint_generation_control.py", argv)


def zspace_generation_control_sweep_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_zspace_generation_control_sweep.py", argv)


def zspace_generation_control_compare_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_zspace_generation_control_compare.py", argv)


def finetune_generation_curve_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_generation_curve.py", argv)
