from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = [
    "TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS",
    "csv_label",
    "csv_values",
    "unique_csv_values",
    "runtime_import_preset_modules",
    "runtime_import_preset_modules_label",
    "runtime_import_preset_missing_modules_label",
    "runtime_import_preset_status_rows",
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


def csv_label(values: object) -> str:
    items = csv_values(values)
    return ",".join(items) if items else "none"


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


def runtime_import_preset_status_rows(
    presets: object,
    probes: Iterable[Mapping[str, object]],
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
) -> list[dict[str, object]]:
    module_map = preset_modules or TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS
    probes_by_module = {
        str(probe["module"]): probe
        for probe in probes
        if isinstance(probe, Mapping) and probe.get("module")
    }
    rows = []
    for preset in unique_csv_values(presets):
        modules = [str(module) for module in module_map.get(preset, [])]
        imported = [
            module
            for module in modules
            if probes_by_module.get(module, {}).get("imported") is True
        ]
        missing = [module for module in modules if module not in imported]
        rows.append(
            {
                "preset": preset,
                "modules": modules,
                "imported": imported,
                "missing": missing,
                "passed": not missing,
            }
        )
    return rows


def runtime_import_preset_modules_label(rows: Iterable[Mapping[str, object]]) -> str:
    return csv_label(
        [
            f"{row['preset']}={'|'.join(row['modules']) or 'none'}"
            for row in rows
        ]
    )


def runtime_import_preset_missing_modules_label(
    rows: Iterable[Mapping[str, object]],
) -> str:
    return csv_label(
        [
            f"{row['preset']}={'|'.join(row['missing']) or 'none'}"
            for row in rows
            if row["missing"]
        ]
    )
