#!/usr/bin/env python3
"""Compatibility entrypoint for the strict, matched-input backend benchmark."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "tools/bench_backend_vs_torch.py"), run_name="__main__")
