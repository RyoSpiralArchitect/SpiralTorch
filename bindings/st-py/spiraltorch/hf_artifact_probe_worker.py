"""Private fresh-process worker for Hugging Face artifact qualification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .hf_generation import hf_causal_lm_artifact_probe_report


def _error_report(request: Mapping[str, object], exc: Exception) -> dict[str, object]:
    return {
        "row_type": "hf_causal_lm_artifact_probe",
        "status": "error",
        "artifact": {
            "artifact_kind": request.get("artifact_kind"),
            "artifact_source": request.get("model_name_or_path"),
            "adapter_loaded": False,
        },
        "prompt": request.get("prompt"),
        "device": request.get("device"),
        "new_token_count": None,
        "generated_text_changed": None,
        "generation": {
            "max_new_tokens": request.get("max_new_tokens"),
            "do_sample": request.get("do_sample"),
        },
        "local_files_only": request.get("local_files_only"),
        "worker_pid": os.getpid(),
        "error": f"{exc.__class__.__name__}: {exc}",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one SpiralTorch HF artifact probe from a JSON request.",
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    request: dict[str, object] = {}
    try:
        loaded = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("artifact probe request must be a JSON object")
        request = dict(loaded)
        report = hf_causal_lm_artifact_probe_report(**request)
        report["worker_pid"] = os.getpid()
    except Exception as exc:
        report = _error_report(request, exc)
        print(f"hf_artifact_probe_worker_error {report['error']}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
