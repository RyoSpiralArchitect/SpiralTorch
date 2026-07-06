# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Compare two API-model LLM Z-space trace artifacts without network access."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import spiraltorch as st


def response(model: str, text: str, *, prompt_tokens: int, completion_tokens: int) -> dict[str, object]:
    return {
        "model": model,
        "output_text": text,
        "status": "completed",
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def write_trace(path: Path, *, model: str, latency_ms: float, text: str) -> str:
    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider="local-demo",
        model=model,
        create_session=False,
    )
    runtime.record_response(
        response(model, text, prompt_tokens=8, completion_tokens=len(text.split())),
        prompt=f"route with {model}",
        latency_ms=latency_ms,
    )
    return runtime.write_jsonl(path)


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        fast = write_trace(
            root / "fast.jsonl",
            model="fast-api-demo",
            latency_ms=75.0,
            text="Fast Z-space route.",
        )
        deep = write_trace(
            root / "deep.jsonl",
            model="deep-api-demo",
            latency_ms=900.0,
            text="A more expansive API model route through Z-space.",
        )
        comparison = st.compare_api_llm_trace_runs({"fast": fast, "deep": deep})

    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
