"""Training CLI integrating SpiralTorch search strategies."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    import spiraltorch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("spiraltorch must be installed to use the training CLI") from exc

# Ensure repository root is importable so that tracking connectors are discoverable.
_REPO_ROOT = Path(__file__).resolve()
for _ in range(4):
    _REPO_ROOT = _REPO_ROOT.parent
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from tools.tracking import base as tracking_base  # type: ignore

LOGGER = logging.getLogger("spiral.cli")


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise RuntimeError(
                f"{path} is not valid JSON and PyYAML is unavailable for YAML parsing"
            )
        return yaml.safe_load(text)


def parse_tracker_specs(specs: Iterable[str]) -> List[Tuple[str, Dict[str, Any]]]:
    parsed: List[Tuple[str, Dict[str, Any]]] = []
    for spec in specs:
        if not spec:
            continue
        name, _, param_str = spec.partition(":")
        params: Dict[str, Any] = {}
        if param_str:
            for pair in param_str.split(","):
                if not pair:
                    continue
                key, _, value = pair.partition("=")
                params[key.strip()] = value.strip()
        parsed.append((name.strip(), params))
    return parsed


def resolve_callable(spec: Any) -> Tuple[Callable[..., Any], Dict[str, Any]]:
    if callable(spec):
        return spec, {}
    if isinstance(spec, dict):
        callable_spec = spec.get("callable")
        kwargs = spec.get("kwargs", {})
    else:
        callable_spec = spec
        kwargs = {}
    if not isinstance(callable_spec, str):
        raise ValueError("Objective specification must provide a callable path")
    module_name, _, func_name = callable_spec.partition(":")
    if not module_name or not func_name:
        raise ValueError(f"Invalid callable specification '{callable_spec}'")
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"Resolved object '{callable_spec}' is not callable")
    return func, kwargs


class TrackerAdapter:
    """Adapter translating SearchLoop dictionaries to TrackingCallback events."""

    def __init__(self, callback: tracking_base.TrackingCallback) -> None:
        self._callback = callback

    @staticmethod
    def _to_event(payload: Dict[str, Any], metric: Optional[float] = None) -> tracking_base.TrialEvent:
        trial_id = int(payload["id"])
        params = dict(payload.get("params", {}))
        metric_value = payload.get("metric")
        if metric is not None:
            metric_value = metric
        return tracking_base.TrialEvent(id=trial_id, params=params, metric=metric_value)

    def on_trial_start(self, payload: Dict[str, Any]) -> None:
        event = self._to_event(payload)
        self._callback.on_trial_start(event)

    def on_trial_end(self, payload: Dict[str, Any], metric: Optional[float]) -> None:
        event = self._to_event(payload, metric)
        self._callback.on_trial_end(event)

    def on_checkpoint(self, checkpoint_json: str) -> None:
        self._callback.on_checkpoint(checkpoint_json)


def build_tracker(specs: Iterable[str]):
    parsed = parse_tracker_specs(specs)
    callbacks = []
    for name, params in parsed:
        tracker = tracking_base.build_tracker(name, **params)
        if tracker is None:
            LOGGER.warning("Tracker '%s' could not be initialised; skipping", name)
            continue
        callbacks.append(tracker)
    if callbacks:
        return TrackerAdapter(tracking_base.CompositeTracker(callbacks))
    return None


def ensure_directory(path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def write_checkpoint(loop: "spiraltorch.hpo.SearchLoop", checkpoint_path: Path) -> None:
    ensure_directory(checkpoint_path)
    checkpoint_path.write_text(loop.checkpoint())


def run_search(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    tracker = build_tracker(args.tracker or [])

    space = config.get("space")
    strategy = config.get("strategy")
    if space is None or strategy is None:
        raise RuntimeError("Configuration must define 'space' and 'strategy'")

    resource = config.get("resource")
    objective_config = config.get("objective")
    if objective_config is None:
        raise RuntimeError("Configuration must define an 'objective'")

    objective_fn, objective_kwargs = resolve_callable(objective_config)
    maximize = False
    if isinstance(objective_config, dict):
        maximize = bool(objective_config.get("maximize", False))

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_text = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            checkpoint_text = resume_path.read_text()
        elif checkpoint_path and checkpoint_path.exists():
            checkpoint_text = checkpoint_path.read_text()

    loop = (
        spiraltorch.hpo.SearchLoop.from_checkpoint(space, checkpoint_text, tracker)
        if checkpoint_text
        else spiraltorch.hpo.SearchLoop.create(space, strategy, resource, tracker)
    )

    completed_records = [dict(record) for record in loop.completed()]
    best_compare: Optional[float] = None
    best_record: Optional[Dict[str, Any]] = None
    for record in completed_records:
        if "metric" not in record or record["metric"] is None:
            continue
        compare = float(record["metric"])
        if best_compare is None or compare < best_compare:
            best_compare = compare
            best_record = record

    max_trials = args.max_trials or config.get("max_trials")
    if max_trials is None:
        raise RuntimeError("Either --max-trials or configuration key 'max_trials' must be set")
    target_trials = int(max_trials)

    LOGGER.info(
        "Checkpoint contains %s completed trials; running towards %s total",
        len(completed_records),
        target_trials,
    )
    remaining = target_trials - len(completed_records)
    if remaining <= 0:
        LOGGER.info("Target already satisfied; no additional trials required")
    else:
        for _ in range(remaining):
            suggestion = loop.suggest()
            trial_id = suggestion["id"]
            params = dict(suggestion["params"])
            metric = objective_fn(params, **objective_kwargs)
            if not isinstance(metric, (int, float)):
                raise TypeError("Objective function must return a numeric metric")
            metric_value = float(metric)
            observed_value = -metric_value if maximize else metric_value
            loop.observe(trial_id, observed_value)
            if checkpoint_path:
                write_checkpoint(loop, checkpoint_path)
            if best_compare is None or observed_value < best_compare:
                best_compare = observed_value
                best_record = {"id": trial_id, "params": params, "metric": observed_value}
            LOGGER.info("Trial %s metric=%s", trial_id, metric_value)

    if best_record:
        reported_metric = -best_compare if maximize else best_compare
        best_output = {
            "id": best_record["id"],
            "metric": reported_metric,
            "params": best_record.get("params", {}),
        }
        LOGGER.info("Best trial %s metric=%s", best_output["id"], best_output["metric"])
        if args.output:
            ensure_directory(Path(args.output))
            Path(args.output).write_text(json.dumps(best_output, indent=2))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpiralTorch training CLI")
    sub = parser.add_subparsers(dest="command")

    search = sub.add_parser("search", help="Run a hyper-parameter search loop")
    search.add_argument("--config", required=True, help="Path to JSON/YAML config file")
    search.add_argument("--max-trials", type=int, help="Override maximum trials")
    search.add_argument("--checkpoint", help="Path to write checkpoints")
    search.add_argument("--resume", help="Resume from a checkpoint")
    search.add_argument("--tracker", action="append", help="Enable tracker(s), e.g. mlflow or wandb")
    search.add_argument("--output", help="Write the best trial JSON to this path")
    search.set_defaults(func=run_search)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = create_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
