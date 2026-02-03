from __future__ import annotations

import os
from typing import Any

import spiraltorch as st

_SOFTLOGIC_FIELDS: tuple[str, ...] = (
    "inertia",
    "inertia_min",
    "inertia_drift_k",
    "inertia_z_k",
    "drift_gain",
    "psi_gain",
    "loss_gain",
    "floor",
    "scale_gain",
    "region_gain",
    "region_factor_gain",
    "energy_equalize_gain",
    "mean_normalize_gain",
)

_SOFTLOGIC_FLAG_TO_FIELD: dict[str, str] = {
    "--softlogic-inertia": "inertia",
    "--softlogic-inertia-min": "inertia_min",
    "--softlogic-inertia-drift-k": "inertia_drift_k",
    "--softlogic-inertia-z-k": "inertia_z_k",
    "--softlogic-drift-gain": "drift_gain",
    "--softlogic-psi-gain": "psi_gain",
    "--softlogic-loss-gain": "loss_gain",
    "--softlogic-floor": "floor",
    "--softlogic-scale-gain": "scale_gain",
    "--softlogic-region-gain": "region_gain",
    "--softlogic-region-factor-gain": "region_factor_gain",
    "--softlogic-energy-equalize-gain": "energy_equalize_gain",
    "--softlogic-mean-normalize-gain": "mean_normalize_gain",
}

_SOFTLOGIC_ENV_VARS: tuple[str, ...] = (
    "SPIRAL_SOFTLOGIC_INERTIA",
    "SPIRAL_SOFTLOGIC_INERTIA_MIN",
    "SPIRAL_SOFTLOGIC_INERTIA_DRIFT_K",
    "SPIRAL_SOFTLOGIC_INERTIA_Z_K",
    "SPIRAL_SOFTLOGIC_DRIFT_GAIN",
    "SPIRAL_SOFTLOGIC_PSI_GAIN",
    "SPIRAL_SOFTLOGIC_LOSS_GAIN",
    "SPIRAL_SOFTLOGIC_FLOOR",
    "SPIRAL_SOFTLOGIC_SCALE_GAIN",
    "SPIRAL_SOFTLOGIC_REGION_GAIN",
    "SPIRAL_SOFTLOGIC_REGION_FACTOR_GAIN",
    "SPIRAL_SOFTLOGIC_ENERGY_EQUALIZE_GAIN",
    "SPIRAL_SOFTLOGIC_MEAN_NORMALIZE_GAIN",
)


def usage_flags() -> str:
    return (
        "[--softlogic-inertia F] [--softlogic-inertia-min F] "
        "[--softlogic-inertia-drift-k F] [--softlogic-inertia-z-k F] "
        "[--softlogic-drift-gain F] [--softlogic-psi-gain F] [--softlogic-loss-gain F] "
        "[--softlogic-floor F] [--softlogic-scale-gain F] [--softlogic-region-gain F] "
        "[--softlogic-region-factor-gain F] [--softlogic-reset]"
        " [--softlogic-energy-equalize-gain F] [--softlogic-mean-normalize-gain F]"
    )


def pop_softlogic_flags(args: list[str]) -> dict[str, Any]:
    reset = False
    overrides: dict[str, float] = {}

    idx = 0
    while idx < len(args):
        flag = str(args[idx])
        if flag == "--softlogic-reset":
            reset = True
            del args[idx]
            continue

        field = _SOFTLOGIC_FLAG_TO_FIELD.get(flag)
        if field is None:
            idx += 1
            continue

        if idx + 1 >= len(args):
            raise ValueError(f"{flag} requires a value")
        try:
            parsed = float(args[idx + 1])
        except ValueError as exc:
            raise ValueError(f"{flag} expects a float") from exc

        overrides[field] = parsed
        del args[idx : idx + 2]

    return {"reset": reset, "overrides": overrides}


def _env_overrides() -> dict[str, str]:
    out: dict[str, str] = {}
    for key in _SOFTLOGIC_ENV_VARS:
        if key in os.environ:
            out[key] = str(os.environ.get(key, ""))
    return out


def _config_to_dict(config: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in _SOFTLOGIC_FIELDS:
        value = getattr(config, key, None)
        if value is None:
            continue
        out[key] = float(value)
    return out


def apply_softlogic_cli(trainer: st.nn.ModuleTrainer, cli: dict[str, Any]) -> dict[str, Any]:
    reset = bool(cli.get("reset", False))
    overrides: dict[str, float] = dict(cli.get("overrides") or {})

    if reset:
        trainer.reset_softlogic()

    base = trainer.softlogic_config()
    if overrides:
        values = _config_to_dict(base)
        values.update(overrides)
        trainer.set_softlogic_config(st.nn.SoftLogicConfig(**values))

    active = trainer.softlogic_config()
    env_overrides = _env_overrides()
    source = "cli" if overrides else ("env" if env_overrides else "default")

    return {
        "source": source,
        "config": _config_to_dict(active),
        "overrides": overrides or None,
        "env": env_overrides or None,
    }
