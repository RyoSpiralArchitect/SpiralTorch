# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of SpiralTorch -- Licensed under AGPL-3.0-or-later.
"""Run API-model inference through several open-topos runtime postures.

The default mode is fully offline and keyless: a provider-shaped local callable
receives the same prompts with different SpiralTorch topological adapters, then
writes JSONL traces plus a compact ``report.json``.  Pass ``--live-provider`` to
route the same sweep through OpenAI or Anthropic SDK callables.

Examples:

    PYTHONPATH=bindings/st-py python3 \
        bindings/st-py/examples/api_llm_topos_sweep.py \
        --prompt-limit 3 --context-prompt --out-dir /tmp/spiraltorch-topos-sweep

    OPENAI_API_KEY=... PYTHONPATH=bindings/st-py python3 \
        bindings/st-py/examples/api_llm_topos_sweep.py \
        --live-provider openai-responses --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import spiraltorch as st


SYSTEM_PROMPT = (
    "You are evaluating SpiralTorch open-topos runtime routing beside hosted "
    "LLM inference. Answer concretely, keep Z-space telemetry explicit, and do "
    "not invent measurements that are not in the prompt."
)

PROMPTS = [
    "Explain how an open-topos route should steer hosted LLM sampling.",
    "Compare an exploratory route with a guarded route for one API-model answer.",
    "Name one metric that should make a topological runtime route more conservative.",
    "Give a compact audit note for a Z-space trace carrying topos telemetry.",
]

DEFAULT_TOPOS_PROFILES: dict[str, dict[str, Any]] = {
    "open": {
        "porosity": 0.7,
        "max_depth": 10,
        "max_volume": 100,
        "observed_depth": 1,
        "visited_volume": 8,
    },
    "contextual": {
        "porosity": 0.25,
        "max_depth": 10,
        "max_volume": 100,
        "observed_depth": 4,
        "visited_volume": 25,
    },
    "guarded": {
        "porosity": 0.02,
        "max_depth": 10,
        "max_volume": 100,
        "observed_depth": 9,
        "visited_volume": 95,
    },
}


def _z_state(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("z-state must contain at least one float")
    return parsed


def _prompts(*, prompt_limit: int, repeat: int) -> list[str]:
    selected = PROMPTS if prompt_limit <= 0 else PROMPTS[:prompt_limit]
    repeats = max(1, repeat)
    if repeats == 1:
        return list(selected)
    result: list[str] = []
    for run_index in range(repeats):
        for prompt in selected:
            result.append(f"{prompt}\nRepeat index: {run_index + 1}.")
    return result


def _load_profiles(path: str | os.PathLike[str] | None) -> Mapping[str, Any]:
    if path is None:
        return DEFAULT_TOPOS_PROFILES
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("profile JSON must contain a label-to-profile object")
    return payload


def offline_topos_provider(prompt: str, **request: Any) -> dict[str, Any]:
    """Provider-shaped local callable used for keyless smoke runs."""

    temperature = float(request.get("temperature") or 0.0)
    top_p = float(request.get("top_p") or 0.0)
    if "topos:sweep:guarded" in prompt:
        route = "guarded"
    elif "topos:sweep:contextual" in prompt:
        route = "contextual"
    elif "topos:sweep:open" in prompt:
        route = "open"
    elif temperature < 0.7:
        route = "guarded"
    elif temperature > 1.05:
        route = "open"
    else:
        route = "contextual"
    text = (
        f"{route} topos route sets temperature={temperature:.3f}, "
        f"top_p={top_p:.3f}, and records Z-space runtime telemetry."
    )
    return {
        "model": "local-topos-sweep-demo",
        "output_text": text,
        "status": "completed",
        "usage": {
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": max(1, len(text.split())),
        },
    }


def _live_invoke(
    provider: str,
    *,
    model: str | None,
    max_tokens: int,
) -> tuple[Callable[..., Any], str, str | None, dict[str, Any]]:
    if provider == "offline":
        return offline_topos_provider, "local-demo", model or "local-topos-sweep-demo", {}
    if provider in {"openai-responses", "openai-chat"} and not os.environ.get(
        "OPENAI_API_KEY"
    ):
        raise SystemExit("OPENAI_API_KEY is required for --live-provider openai-*")
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required for --live-provider anthropic")

    if provider == "openai-responses":
        model_value = model or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
        return (
            st.make_openai_responses_invoke(model=model_value),
            "openai",
            model_value,
            {"instructions": SYSTEM_PROMPT, "max_output_tokens": max(1, max_tokens)},
        )
    if provider == "openai-chat":
        model_value = model or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
        return (
            st.make_openai_chat_invoke(model=model_value, system=SYSTEM_PROMPT),
            "openai",
            model_value,
            {"max_tokens": max(1, max_tokens)},
        )
    if provider == "anthropic":
        model_value = model or os.environ.get("ANTHROPIC_MODEL") or "claude-fable-5"
        return (
            st.make_anthropic_messages_invoke(model=model_value, system=SYSTEM_PROMPT),
            "anthropic",
            model_value,
            {"max_tokens": max(1, max_tokens)},
        )
    raise ValueError(f"unknown live provider: {provider}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="api-llm-topos-sweep")
    parser.add_argument("--prompt-limit", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--create-session", action="store_true")
    parser.add_argument("--context-prompt", action="store_true")
    parser.add_argument("--max-telemetry", type=int, default=64)
    parser.add_argument("--z-state", type=_z_state, default=[0.2, -0.1, 0.4, 0.05])
    parser.add_argument("--profile-json", help="label-to-profile JSON object")
    parser.add_argument(
        "--live-provider",
        choices=("offline", "openai-responses", "openai-chat", "anthropic"),
        default="offline",
    )
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-temperature", type=float, default=0.9)
    parser.add_argument("--base-top-p", type=float, default=0.95)
    parser.add_argument("--include-penalties", action="store_true")
    parser.add_argument("--near-best-tolerance", type=float, default=0.05)
    parser.add_argument("--report-samples-per-route", type=int, default=2)
    parser.add_argument("--report-text-chars", type=int, default=360)
    parser.add_argument("--report-pair-rows", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _prompts(prompt_limit=args.prompt_limit, repeat=args.repeat)
    profiles = _load_profiles(args.profile_json)
    request_options = {
        "base_temperature": args.base_temperature,
        "base_top_p": args.base_top_p,
        "include_penalties": args.include_penalties,
    }
    if args.dry_run:
        plan = {
            "kind": "spiraltorch.api_llm_topos_sweep_plan",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "out_dir": str(out_dir),
            "prompt_count": len(prompts),
            "labels": list(profiles.keys()),
            "live_provider": args.live_provider,
            "request_options": request_options,
            "report_options": {
                "max_samples_per_route": args.report_samples_per_route,
                "max_text_chars": args.report_text_chars,
                "max_pair_rows": args.report_pair_rows,
            },
        }
        (out_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    invoke, provider, model, provider_request = _live_invoke(
        args.live_provider,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    result = st.run_api_llm_topos_sweep(
        prompts,
        invoke,
        z_state=args.z_state,
        topos_profiles=profiles,
        backend=args.backend,
        provider=provider,
        model=model,
        create_session=args.create_session,
        jsonl_dir=out_dir / "traces",
        context_prompt=args.context_prompt,
        context_prompt_options={"max_telemetry": args.max_telemetry},
        request_options=request_options,
        near_best_tolerance=args.near_best_tolerance,
        report_out=out_dir / "report.json",
        report_options={
            "max_samples_per_route": args.report_samples_per_route,
            "max_text_chars": args.report_text_chars,
            "max_pair_rows": args.report_pair_rows,
        },
        **provider_request,
    )
    report_comparison = st.compare_api_llm_topos_sweep_reports(
        {"current": result["report_path"]}
    )
    print(
        json.dumps(
            {
                "report": result["report_path"],
                "prompt_count": result["prompt_count"],
                "labels": result["labels"],
                "mode_counts": result["report"]["mode_counts"],
                "trace_winners": (result["comparison"] or {}).get("winners"),
                "adapter_winners": result["report"]["adapter_winners"],
                "response_winners": result["report"]["response_winners"],
                "selection_profiles": result["report"]["selection_profiles"],
                "response_route_rows": result["report"]["response_route_rows"],
                "response_pair_rows": result["report"]["response_pair_rows"],
                "response_samples": result["report"]["response_samples"],
                "report_comparison": report_comparison,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
