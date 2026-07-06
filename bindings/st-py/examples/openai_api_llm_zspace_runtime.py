# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""OpenAI Responses API -> SpiralTorch Z-space runtime smoke.

Requires ``OPENAI_API_KEY`` and the optional ``openai`` Python package:

    python bindings/st-py/examples/openai_api_llm_zspace_runtime.py \
        --prompt "Describe SpiralTorch entering Z-space runtime."

The API key is read by the OpenAI SDK from the environment and is never printed.
"""

from __future__ import annotations

import argparse
import json
import os

import spiraltorch as st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="In seven words, describe SpiralTorch entering Z-space runtime.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--max-output-tokens", type=int, default=48)
    parser.add_argument("--jsonl-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        backend=args.backend,
        provider="openai",
        model=args.model,
    )
    trace = runtime.call_openai_responses(
        args.prompt,
        max_output_tokens=args.max_output_tokens,
    )
    payload = trace.as_dict()

    print("model:", trace.model)
    print("text:", trace.text)
    print("usage:", json.dumps(payload["usage"], sort_keys=True))
    print("metrics:", json.dumps(payload["metrics"], indent=2, sort_keys=True))
    print("confidence:", payload["inference"]["confidence"])
    print("device:", payload["device_preflight"])
    if args.jsonl_out:
        path = runtime.write_jsonl(args.jsonl_out)
        summary = st.summarize_api_llm_trace_events(path)
        print("jsonl:", path)
        print(
            "summary:",
            json.dumps(
                {
                    "count": summary["count"],
                    "models": summary["models"],
                    "total_tokens": summary["total_tokens"],
                    "runtime_statuses": summary["runtime_statuses"],
                    "last_text_preview": summary["last_text_preview"],
                },
                indent=2,
                sort_keys=True,
            ),
        )


if __name__ == "__main__":
    main()
