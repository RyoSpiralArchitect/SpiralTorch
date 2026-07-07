#!/usr/bin/env python3
"""Train an stAgent-style route policy from an API LLM open-topos sweep report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import spiraltorch as st


PROMPTS = [
    "Explain how SpiralTorch should pick an open-topos inference route.",
    "Choose a cautious route when Z-space telemetry looks unstable.",
    "Name the runtime signal that should promote a route for hosted inference.",
]

TOPOS_PROFILES: dict[str, dict[str, Any]] = {
    "open": {
        "porosity": 0.72,
        "max_depth": 10,
        "max_volume": 100,
        "observed_depth": 1,
        "visited_volume": 8,
    },
    "contextual": {
        "porosity": 0.28,
        "max_depth": 10,
        "max_volume": 100,
        "observed_depth": 4,
        "visited_volume": 32,
    },
    "guarded": {
        "porosity": 0.03,
        "max_depth": 10,
        "max_volume": 100,
        "observed_depth": 9,
        "visited_volume": 96,
    },
}


class DemoRouteAgent:
    """Small Python fallback with the stAgent policy/update shape."""

    def __init__(self, *, action_dim: int, learning_rate: float) -> None:
        self.q_values = [0.0 for _ in range(action_dim)]
        self.learning_rate = float(learning_rate)
        self.epsilon = 0.0

    def policy_report(self, state: int) -> dict[str, Any]:
        return {
            "state": state,
            "q_values": list(self.q_values),
            "epsilon": self.epsilon,
            "agent": "demo-python-fallback",
        }

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(epsilon)

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        del state, next_state
        self.q_values[action] += self.learning_rate * (reward - self.q_values[action])

    def select_action_trace(self, state: int) -> dict[str, Any]:
        action = max(range(len(self.q_values)), key=lambda index: self.q_values[index])
        return {
            "state": state,
            "action": action,
            "q_values": list(self.q_values),
            "epsilon_before": self.epsilon,
            "epsilon_after": self.epsilon,
            "explored": False,
            "greedy_action": action,
            "agent": "demo-python-fallback",
        }


def offline_topos_provider(prompt: str, **request: Any) -> dict[str, Any]:
    """Provider-shaped local callable for keyless route-policy demos."""

    temperature = float(request.get("temperature") or 0.0)
    if "topos:sweep:guarded" in prompt:
        route = "guarded"
    elif "topos:sweep:contextual" in prompt:
        route = "contextual"
    else:
        route = "open"
    text = (
        f"{route} route uses temperature={temperature:.3f} and carries "
        "Z-space/topos telemetry into the route-policy learner."
    )
    return {
        "model": "local-topos-stagent-demo",
        "output_text": text,
        "status": "completed",
        "usage": {
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": max(1, len(text.split())),
        },
    }


def _load_report(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("report JSON must contain an object")
    return payload


def _build_offline_report(
    *,
    out_dir: Path,
    prompt_limit: int,
    context_prompt: bool,
) -> tuple[Mapping[str, Any], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS[: max(1, prompt_limit)]
    result = st.run_api_llm_topos_sweep(
        prompts,
        offline_topos_provider,
        z_state=[0.2, -0.1, 0.4, 0.05],
        topos_profiles=TOPOS_PROFILES,
        provider="local-demo",
        model="local-topos-stagent-demo",
        create_session=False,
        jsonl_dir=out_dir / "traces",
        context_prompt=context_prompt,
        request_options={
            "base_temperature": 0.9,
            "base_top_p": 0.95,
            "include_penalties": True,
        },
        report_out=out_dir / "report.json",
    )
    return result["report"], str(result["report_path"])


def _supports_route_policy_trace(agent: Any) -> bool:
    if callable(getattr(agent, "select_action_trace", None)):
        return True
    policy_report = getattr(agent, "policy_report", None)
    if not callable(policy_report):
        return False
    try:
        report = policy_report(0)
    except Exception:
        return False
    q_values = report.get("q_values") if isinstance(report, Mapping) else None
    return isinstance(q_values, Sequence) and not isinstance(
        q_values,
        (str, bytes, bytearray),
    )


def _agent(
    *,
    action_dim: int,
    learning_rate: float,
    strict_native: bool,
) -> tuple[Any, str]:
    agent_cls = getattr(getattr(st, "rl", None), "stAgent", None)
    if agent_cls is None:
        agent_cls = getattr(st, "stAgent", None)
    if agent_cls is not None:
        try:
            native_agent = agent_cls(
                state_dim=1,
                action_dim=action_dim,
                discount=0.0,
                learning_rate=learning_rate,
            )
            if _supports_route_policy_trace(native_agent):
                return native_agent, "spiraltorch.rl.stAgent"
            if strict_native:
                raise SystemExit(
                    "native stAgent lacks select_action_trace() or policy_report().q_values"
                )
        except RuntimeError as exc:
            if strict_native:
                raise SystemExit(f"native stAgent is not available: {exc}") from exc
    elif strict_native:
        raise SystemExit("spiraltorch.rl.stAgent is not available in this build")
    fallback = DemoRouteAgent(action_dim=action_dim, learning_rate=learning_rate)
    return fallback, "demo-python-fallback"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Existing topos sweep report.json")
    parser.add_argument("--out-dir", type=Path, default=Path("topos-stagent-route-policy"))
    parser.add_argument("--prompt-limit", type=int, default=3)
    parser.add_argument("--context-prompt", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("balanced", "quality", "grounded", "efficiency", "latency"),
        default="balanced",
    )
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--selection-epsilon", type=float, default=0.0)
    parser.add_argument("--policy-out", type=Path)
    parser.add_argument("--strict-native-stagent", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.report is not None:
        report = _load_report(args.report)
        report_path = str(args.report)
        selection_profiles = None
    else:
        report, report_path = _build_offline_report(
            out_dir=args.out_dir,
            prompt_limit=args.prompt_limit,
            context_prompt=args.context_prompt,
        )
        selection_profiles = TOPOS_PROFILES

    route_rewards = st.api_llm_topos_sweep_route_rewards(report, profile=args.profile)
    agent, agent_kind = _agent(
        action_dim=len(route_rewards),
        learning_rate=args.learning_rate,
        strict_native=args.strict_native_stagent,
    )
    policy = st.train_stagent_topos_route_policy(
        report,
        agent,
        profile=args.profile,
        episodes=args.episodes,
        selection_epsilon=args.selection_epsilon,
    )
    selection = st.api_llm_topos_route_policy_selection(
        policy,
        report=report,
        topos_profiles=selection_profiles,
        request_options={
            "base_temperature": 0.9,
            "base_top_p": 0.95,
            "include_penalties": True,
        },
    )
    summary = {
        "kind": "spiraltorch.api_llm_topos_stagent_route_policy_demo",
        "agent": agent_kind,
        "report": report_path,
        "profile": args.profile,
        "labels": policy["labels"],
        "selected_label": policy["selected_label"],
        "selected_reward": policy["selected_reward"],
        "selection_trace": policy["selection_trace"],
        "policy_selection": selection,
        "policy_after": policy["policy_after"],
        "selected_request": selection["request"],
        "selected_runtime_route": selection["runtime_route"],
        "route_rewards": route_rewards,
        "update_count": policy["update_count"],
    }
    if args.policy_out is not None:
        args.policy_out.parent.mkdir(parents=True, exist_ok=True)
        args.policy_out.write_text(
            json.dumps(policy, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["policy_out"] = str(args.policy_out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
