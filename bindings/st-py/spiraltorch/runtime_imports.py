from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = [
    "TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS",
    "csv_values",
    "unique_csv_values",
    "runtime_import_preset_modules",
]

TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS: dict[str, list[str]] = {
    "transformers": ["transformers"],
    "torch-transformers": ["transformers", "torch"],
    "hf-runtime": ["transformers", "torch", "tokenizers"],
}


def csv_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split(",") if part and part != "none"]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item) and str(item) != "none"]
    return [str(value)] if str(value) else []


def unique_csv_values(value: object) -> list[str]:
    return list(dict.fromkeys(csv_values(value)))


def runtime_import_preset_modules(
    presets: object,
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    module_map = preset_modules or TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS
    rows = []
    for preset in unique_csv_values(presets):
        modules = [str(module) for module in module_map.get(preset, [])]
        rows.append(f"{preset}={'|'.join(modules) or 'none'}")
    return rows
