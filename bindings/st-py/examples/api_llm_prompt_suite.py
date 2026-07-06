# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Run a keyless API-model prompt suite through SpiralTorch Z-space."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import spiraltorch as st


def bipolar_geometry_api(prompt: str, *, max_tokens: int = 32) -> dict[str, object]:
    words = prompt.split()
    text = (
        "Bipolar geometry routes the hosted answer as a pair of opposed "
        f"Z-space poles: {words[0] if words else 'prompt'} and inference."
    )
    return {
        "model": "local-bipolar-geometry-demo",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": max(1, len(words)),
            "output_tokens": min(max_tokens, max(1, len(text.split()))),
        },
    }


def main() -> None:
    prompts = [
        "Describe SpiralTorch inference as bipolar geometry.",
        "Name one trace signal for a hosted LLM Z-space route.",
        "Give one safe next step for API-model fine-tuning.",
    ]
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "api-llm-prompt-suite.jsonl"
        suite = st.run_api_llm_prompt_suite(
            prompts,
            bipolar_geometry_api,
            z_state=[0.12, -0.04, 0.33, -0.11],
            provider="local-demo",
            model="local-bipolar-geometry-demo",
            create_session=False,
            jsonl_out=path,
            max_tokens=32,
        )
        comparison = st.compare_api_llm_trace_runs({"bipolar-suite": suite["jsonl"]})

    print(
        json.dumps(
            {
                "suite": {
                    "kind": suite["kind"],
                    "count": suite["count"],
                    "summary": suite["summary"],
                },
                "comparison": comparison,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
