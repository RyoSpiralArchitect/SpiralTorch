#!/usr/bin/env python3
"""Run a tiny stAgent bandit loop and persist auditable policy traces."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import spiraltorch as st


def _reward(action: int, rng: random.Random) -> float:
    probability = 0.66 if action == 0 else 0.38
    return 1.0 if rng.random() < probability else 0.0


def _agent_class() -> Any:
    agent = getattr(getattr(st, "rl", None), "stAgent", None)
    if agent is None:
        raise SystemExit("spiraltorch.rl.stAgent is not available in this build")
    return agent


def run_trace(*, steps: int, seed: int, jsonl_out: Path | None) -> dict[str, Any]:
    rng = random.Random(seed)
    try:
        agent = _agent_class()(
            state_dim=1,
            action_dim=2,
            discount=0.0,
            learning_rate=5e-2,
        )
    except RuntimeError as exc:
        raise SystemExit(f"spiraltorch.rl.stAgent requires the native extension: {exc}") from exc
    events: list[dict[str, Any]] = []
    wins = 0.0
    pulls = [0, 0]

    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    handle = jsonl_out.open("w", encoding="utf-8") if jsonl_out is not None else None
    try:
        for step in range(steps):
            epsilon = max(0.02, 0.35 * (1.0 - step / max(1, steps)))
            agent.set_epsilon(float(epsilon))
            trace = dict(agent.select_action_trace(0))
            action = int(trace["action"])
            reward = _reward(action, rng)
            agent.update(0, action, reward, 0)

            wins += reward
            pulls[action] += 1
            event = {
                "kind": "spiraltorch.rl.stagent_policy_trace_event",
                "step": step,
                "reward": reward,
                "pulls": list(pulls),
                **trace,
            }
            events.append(event)
            if handle is not None:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
    finally:
        if handle is not None:
            handle.close()

    final_policy = dict(agent.policy_report(0))
    return {
        "kind": "spiraltorch.rl.stagent_policy_trace_summary",
        "steps": steps,
        "seed": seed,
        "win_rate": wins / max(1, steps),
        "pulls": pulls,
        "final_policy": final_policy,
        "event_count": len(events),
        "events": events[-5:],
        "jsonl_out": str(jsonl_out) if jsonl_out is not None else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jsonl-out", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_trace(steps=max(1, args.steps), seed=args.seed, jsonl_out=args.jsonl_out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
