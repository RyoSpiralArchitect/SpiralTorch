from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = [
    "TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS",
    "csv_label",
    "csv_values",
    "unique_csv_values",
    "runtime_import_preset_modules",
    "runtime_import_preset_module_map",
    "runtime_import_preset_module_rows",
    "runtime_import_preset_modules_label",
    "runtime_import_preset_missing_modules_label",
    "runtime_import_preset_status_rows",
    "required_runtime_import_presets_from_source",
    "required_runtime_import_presets_from_args",
    "required_runtime_imports_from_source",
    "required_runtime_imports_from_args",
    "runtime_import_names_from_source",
    "runtime_import_names_from_args",
    "runtime_import_presets_from_source",
    "runtime_import_presets_from_args",
    "runtime_imports_from_args",
    "runtime_imports_from_source",
    "runtime_import_required_gate_fields",
    "runtime_import_requirement_failures",
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


def unique_stripped_values(value: object) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in csv_values(value)
            if item.strip()
        )
    )


def _runtime_import_source_values(
    source: object,
    key: str | None,
) -> list[str]:
    if key is None:
        return []
    if isinstance(source, Mapping):
        return csv_values(source.get(key))
    return csv_values(getattr(source, key, None))


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


def runtime_import_preset_module_map(value: object) -> dict[str, str]:
    modules = {}
    for item in csv_values(value):
        preset, sep, _module_list = item.partition("=")
        if preset and sep:
            modules[preset] = item
    return modules


def runtime_import_preset_module_rows(value: object, presets: object) -> list[str]:
    module_map = runtime_import_preset_module_map(value)
    return [
        module_map[preset]
        for preset in csv_values(presets)
        if preset in module_map
    ]


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


def runtime_import_presets_from_source(
    source: object,
    *,
    runtime_import_presets_key: str = "runtime_import_presets",
    required_runtime_import_presets_key: str = "required_runtime_import_presets",
) -> list[str]:
    return unique_stripped_values(
        [
            *_runtime_import_source_values(source, runtime_import_presets_key),
            *_runtime_import_source_values(source, required_runtime_import_presets_key),
        ]
    )


def runtime_import_presets_from_args(args: object) -> list[str]:
    return runtime_import_presets_from_source(args)


def runtime_imports_from_source(
    source: object,
    *,
    runtime_imports_key: str = "runtime_imports",
) -> list[str]:
    return unique_stripped_values(
        _runtime_import_source_values(source, runtime_imports_key)
    )


def runtime_imports_from_args(args: object) -> list[str]:
    return runtime_imports_from_source(args)


def required_runtime_imports_from_source(
    source: object,
    *,
    required_runtime_imports_key: str = "required_runtime_imports",
) -> list[str]:
    return unique_stripped_values(
        _runtime_import_source_values(source, required_runtime_imports_key)
    )


def required_runtime_imports_from_args(args: object) -> list[str]:
    return required_runtime_imports_from_source(args)


def required_runtime_import_presets_from_source(
    source: object,
    *,
    required_runtime_import_presets_key: str = "required_runtime_import_presets",
) -> list[str]:
    return unique_stripped_values(
        _runtime_import_source_values(source, required_runtime_import_presets_key)
    )


def required_runtime_import_presets_from_args(args: object) -> list[str]:
    return required_runtime_import_presets_from_source(args)


def runtime_import_names_from_source(
    source: object,
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
    runtime_imports_key: str = "runtime_imports",
    runtime_import_presets_key: str = "runtime_import_presets",
    required_runtime_imports_key: str = "required_runtime_imports",
    required_runtime_import_presets_key: str = "required_runtime_import_presets",
) -> list[str]:
    module_map = preset_modules or TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS
    names = []
    for preset in runtime_import_presets_from_source(
        source,
        runtime_import_presets_key=runtime_import_presets_key,
        required_runtime_import_presets_key=required_runtime_import_presets_key,
    ):
        names.extend(str(module) for module in module_map.get(preset, []))
    names.extend(_runtime_import_source_values(source, runtime_imports_key))
    names.extend(_runtime_import_source_values(source, required_runtime_imports_key))
    return unique_stripped_values(names)


def runtime_import_names_from_args(
    args: object,
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    return runtime_import_names_from_source(args, preset_modules=preset_modules)


def runtime_import_required_gate_fields(
    required_imports: object,
    required_presets: object,
    *,
    imported_modules: object | None = None,
    observed_presets: object | None = None,
    satisfied_presets: object | None = None,
    failed_presets: object | None = None,
    probes: Iterable[Mapping[str, object]] | None = None,
    preset_status: Iterable[Mapping[str, object]] | None = None,
    field_prefix: str = "",
    include_failed_presets: bool = False,
) -> dict[str, object]:
    required = unique_csv_values(required_imports)
    required_preset_rows = unique_csv_values(required_presets)
    if probes is not None:
        imported = [
            str(probe["module"])
            for probe in probes
            if isinstance(probe, Mapping)
            and probe.get("module")
            and probe.get("imported") is True
        ]
    else:
        imported = unique_csv_values(imported_modules)

    if preset_status is not None:
        status_rows = [row for row in preset_status if isinstance(row, Mapping)]
        observed = [
            str(row["preset"])
            for row in status_rows
            if row.get("preset")
        ]
        satisfied = [
            str(row["preset"])
            for row in status_rows
            if row.get("preset") and row.get("passed") is True
        ]
        failed = [
            str(row["preset"])
            for row in status_rows
            if row.get("preset") and row.get("passed") is not True
        ]
    else:
        observed = unique_csv_values(observed_presets)
        satisfied = unique_csv_values(satisfied_presets)
        failed = unique_csv_values(failed_presets)

    missing = [module for module in required if module not in imported]
    missing_presets = [
        preset for preset in required_preset_rows if preset not in observed
    ]
    unsatisfied_presets = [
        preset
        for preset in required_preset_rows
        if preset in observed and preset not in satisfied
    ]
    gate_requested = bool(required)
    preset_gate_requested = bool(required_preset_rows)
    fields: dict[str, object] = {
        f"{field_prefix}required_runtime_imports": csv_label(required),
        f"{field_prefix}required_runtime_imports_imported": (
            csv_label(imported) if gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_imports_missing": (
            csv_label(missing) if gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_imports_passed": (
            None if not gate_requested else not missing
        ),
        f"{field_prefix}required_runtime_import_presets": (
            csv_label(required_preset_rows)
        ),
        f"{field_prefix}required_runtime_import_presets_observed": (
            csv_label(observed) if preset_gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_import_presets_satisfied": (
            csv_label(satisfied) if preset_gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_import_presets_missing": (
            csv_label(missing_presets) if preset_gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_import_presets_unsatisfied": (
            csv_label(unsatisfied_presets) if preset_gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_import_presets_passed": (
            None
            if not preset_gate_requested
            else not missing_presets and not unsatisfied_presets
        ),
    }
    if include_failed_presets:
        fields[f"{field_prefix}required_runtime_import_presets_failed"] = (
            csv_label(failed) if preset_gate_requested else "none"
        )
    return fields


def runtime_import_requirement_failures(
    row: Mapping[str, object],
    *,
    field_prefix: str = "",
    failure_prefix: str = "runtime_import",
) -> list[str]:
    failures = []
    if row.get(f"{field_prefix}required_runtime_imports_passed") is False:
        for module_name in sorted(
            csv_values(row.get(f"{field_prefix}required_runtime_imports_missing"))
        ):
            failures.append(f"{failure_prefix}_missing:{module_name}")
    if row.get(f"{field_prefix}required_runtime_import_presets_passed") is False:
        for preset in sorted(
            csv_values(
                row.get(f"{field_prefix}required_runtime_import_presets_missing")
            )
        ):
            failures.append(f"{failure_prefix}_preset_missing:{preset}")
        for preset in sorted(
            csv_values(
                row.get(
                    f"{field_prefix}required_runtime_import_presets_unsatisfied"
                )
            )
        ):
            failures.append(f"{failure_prefix}_preset_unsatisfied:{preset}")
    return failures
