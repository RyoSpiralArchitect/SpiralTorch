# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Compare provider-shaped API-model prompt suites without network access."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import spiraltorch as st


def openai_shaped(prompt: str) -> dict[str, object]:
    text = f"OpenAI-shaped route compresses '{prompt}' into a fast Z-space pole."
    return {
        "model": "openai-shaped-bipolar-demo",
        "output_text": text,
        "status": "completed",
        "usage": {
            "input_tokens": max(1, len(prompt.split())),
            "output_tokens": max(1, len(text.split())),
        },
    }


def anthropic_shaped(prompt: str) -> dict[str, object]:
    text = (
        "Anthropic-shaped route expands the opposite pole before folding "
        f"'{prompt}' back into bipolar Z-space."
    )
    return {
        "model": "anthropic-shaped-bipolar-demo",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": max(1, len(prompt.split())),
            "output_tokens": max(1, len(text.split())),
        },
    }


def main() -> None:
    prompts = [
        "Name the convergent pole of bipolar inference.",
        "Name the divergent pole of bipolar inference.",
        "Choose one trace signal for stable Z-space routing.",
    ]
    with TemporaryDirectory() as tmp:
        matrix = st.run_api_llm_prompt_suite_matrix(
            prompts,
            {
                "openai-shaped": openai_shaped,
                "anthropic-shaped": anthropic_shaped,
            },
            z_state=[0.12, -0.04, 0.33, -0.11],
            providers={
                "openai-shaped": "openai",
                "anthropic-shaped": "anthropic",
            },
            models={
                "openai-shaped": "openai-shaped-bipolar-demo",
                "anthropic-shaped": "anthropic-shaped-bipolar-demo",
            },
            create_session=False,
            jsonl_dir=Path(tmp),
        )

    print(
        json.dumps(
            {
                "kind": matrix["kind"],
                "labels": matrix["labels"],
                "trace_paths": matrix["trace_paths"],
                "comparison": matrix["comparison"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
