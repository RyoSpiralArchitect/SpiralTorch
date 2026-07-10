#!/usr/bin/env python3
"""Run every installed SpiralTorch HF/Z-Space console entrypoint through --help."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys


PREFIXES = (
    "spiral-hf-",
    "spiral-zspace-inference-distortion-",
)
MODULE_ENTRYPOINTS = (
    "spiraltorch.hf_artifact_probe_worker",
    "spiraltorch.hf_finetune_entrypoint",
)


def main() -> int:
    distribution = importlib.metadata.distribution("spiraltorch")
    names = sorted(
        entry.name
        for entry in distribution.entry_points
        if entry.group == "console_scripts" and entry.name.startswith(PREFIXES)
    )
    if not names:
        raise RuntimeError(
            "installed spiraltorch distribution has no HF CLI entrypoints"
        )
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    bin_dir = str(Path(sys.executable).parent)
    env["PATH"] = os.pathsep.join((bin_dir, env.get("PATH", "")))
    failures: list[str] = []
    for name in names:
        executable = shutil.which(name, path=env["PATH"])
        if executable is None:
            failures.append(f"{name}: executable missing")
            continue
        completed = subprocess.run(
            [executable, "--help"],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=60.0,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-500:].strip()
            failures.append(f"{name}: exit={completed.returncode} {detail}")
    for module in MODULE_ENTRYPOINTS:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=60.0,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-500:].strip()
            failures.append(f"{module}: exit={completed.returncode} {detail}")
    if failures:
        raise RuntimeError("HF CLI smoke failed:\n" + "\n".join(failures))
    print(f"hf_console_scripts=ok count={len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
