# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""API-model LLM response -> Z-space runtime quickstart.

This example uses a local fake API callable so it runs without network access or
an API key.  Replace ``fake_api`` with a hosted-model SDK call, or pass an
already materialised response mapping into ``runtime.record_response(...)``.
"""

from __future__ import annotations

import json

import spiraltorch as st


def fake_api(prompt: str) -> dict[str, object]:
    """Return an OpenAI-compatible chat-completion-shaped response."""

    return {
        "id": "chatcmpl-local-demo",
        "object": "chat.completion",
        "model": "api-model-demo",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"Z-space heard: {prompt}",
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "Z",
                            "logprob": -0.25,
                            "top_logprobs": [{"token": "Z", "logprob": -0.25}],
                        }
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 8,
            "total_tokens": 14,
        },
    }


def main() -> None:
    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        backend="auto",
        provider="local-demo",
    )
    trace = runtime.call(fake_api, "spiral route")
    payload = trace.as_dict()

    print("text:", trace.text)
    print("metrics:", json.dumps(payload["metrics"], indent=2, sort_keys=True))
    print("confidence:", payload["inference"]["confidence"])
    print("device:", payload["device_preflight"])


if __name__ == "__main__":
    main()
